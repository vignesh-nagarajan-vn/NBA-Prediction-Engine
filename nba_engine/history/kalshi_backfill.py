"""Backfill settled Kalshi NBA game markets and their 1-minute price candles.

Markets settled before Kalshi's historical cutoff live under /historical/...; newer ones under
the regular endpoints. Candle payloads differ slightly (plain keys vs *_dollars / *_fp), so
both are normalized here.

Tables written to {data_dir}/history/:
  kalshi_markets   one row per market (team) with result and open/close times
  kalshi_candles   one row per market per minute with trading activity: yes bid/ask/last close
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from nba_engine.config.settings import NBA_GAME_SERIES
from nba_engine.kalshi.rest import KalshiAPIError, KalshiRest

log = logging.getLogger(__name__)

MAX_CANDLES = 5000  # Kalshi rejects ranges with more 1-minute candles than this


def _ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _f(d: dict | None, *keys: str) -> float | None:
    if not d:
        return None
    for k in keys:
        if d.get(k) not in (None, ""):
            return float(d[k])
    return None


def normalize_candle(ticker: str, c: dict) -> dict:
    price, bid, ask = c.get("price") or {}, c.get("yes_bid") or {}, c.get("yes_ask") or {}
    return {
        "ticker": ticker,
        "end_ts": int(c["end_period_ts"]),
        "yes_bid": _f(bid, "close_dollars", "close"),
        "yes_ask": _f(ask, "close_dollars", "close"),
        "last": _f(price, "close_dollars", "close"),
        "volume": _f(c, "volume_fp", "volume") or 0.0,
        "open_interest": _f(c, "open_interest_fp", "open_interest"),
    }


def market_row(m: dict, archived: bool) -> dict:
    return {
        "ticker": m["ticker"],
        "event_ticker": m["event_ticker"],
        "team": m["ticker"].rsplit("-", 1)[1],
        "result": m.get("result") or "",
        "open_time": m.get("open_time"),
        "close_time": m.get("close_time"),
        "archived": archived,
        "volume": float(m.get("volume_fp") or m.get("volume") or 0),
    }


class KalshiBackfill:
    def __init__(self, rest: KalshiRest, data_dir: Path | str):
        self.rest = rest
        self.data_dir = Path(data_dir)
        self.cache = self.data_dir / "cache" / "kalshi" / "candles"
        self.out = self.data_dir / "history"

    async def list_markets(self) -> list[dict]:
        rows: dict[str, dict] = {}
        async for m in self.rest.paginate("/historical/markets", "markets",
                                          {"series_ticker": NBA_GAME_SERIES, "limit": 1000}):
            rows[m["ticker"]] = market_row(m, archived=True)
        async for m in self.rest.paginate("/markets", "markets",
                                          {"series_ticker": NBA_GAME_SERIES,
                                           "status": "settled", "limit": 1000}):
            rows.setdefault(m["ticker"], market_row(m, archived=False))
        return [r for r in rows.values() if r["result"] in ("yes", "no")]

    async def candles(self, mkt: dict) -> list[dict]:
        path = self.cache / f"{mkt['ticker']}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        start, end = _ts(mkt["open_time"]), _ts(mkt["close_time"])
        base = (f"/historical/markets/{mkt['ticker']}/candlesticks" if mkt["archived"] else
                f"/series/{NBA_GAME_SERIES}/markets/{mkt['ticker']}/candlesticks")
        out: list[dict] = []
        chunk = (MAX_CANDLES - 10) * 60
        lo = start
        while lo < end:
            hi = min(end, lo + chunk)
            try:
                data = await self.rest.request("GET", base, params={
                    "start_ts": lo, "end_ts": hi, "period_interval": 1})
            except KalshiAPIError as e:
                log.warning("candles failed for %s: %s", mkt["ticker"], e)
                return []
            out += [normalize_candle(mkt["ticker"], c) for c in data.get("candlesticks", [])]
            lo = hi
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(out, f)
        return out

    async def run(self) -> None:
        markets = await self.list_markets()
        log.info("kalshi: %d settled NBA game markets", len(markets))
        self.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(markets).to_parquet(self.out / "kalshi_markets.parquet", index=False)
        all_candles: list[dict] = []
        for i, m in enumerate(markets, 1):
            all_candles += await self.candles(m)
            if i % 200 == 0:
                log.info("kalshi candles %d/%d", i, len(markets))
        df = pd.DataFrame(all_candles)
        df.to_parquet(self.out / "kalshi_candles.parquet", index=False)
        log.info("wrote kalshi_candles: %d rows", len(df))
