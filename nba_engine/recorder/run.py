"""Always-on recorder: Kalshi NBA game books/trades + ESPN live play-by-play + free odds.

Two Kalshi modes:
- WebSocket (API key configured): real-time orderbook_delta / ticker / trade for every open
  KXNBAGAME market, plus a REST cross-check of the rebuilt books every 60s (Phase 0 gate).
- REST polling (no key): batch order-book snapshots every 2s while games are live (30s
  otherwise) and per-market trade polling. Coarser, but needs no account setup.

Streams written under {data_dir}/raw/:
  games, kalshi_ws, kalshi_books, kalshi_trades, book_checks, espn_plays, espn_state,
  espn_odds; plus full ESPN game summaries as JSON under espn_summaries/ once games finish.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import logging
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from nba_engine.config.settings import NBA_GAME_SERIES, Settings
from nba_engine.data.espn import EspnClient
from nba_engine.data.live_feed import PlayTracker, parse_plays
from nba_engine.data.schedule import ET, GameMarkets, match_games
from nba_engine.kalshi.auth import KalshiSigner
from nba_engine.kalshi.orderbook import OrderBook
from nba_engine.kalshi.rest import KalshiRest
from nba_engine.kalshi.ws import KalshiWS
from nba_engine.storage.sink import ParquetSink, to_json

log = logging.getLogger(__name__)


def now_ms() -> int:
    return int(time.time() * 1000)


class Recorder:
    def __init__(self, settings: Settings, force_rest: bool = False):
        self.s = settings
        self.signer = (KalshiSigner.from_file(settings.kalshi_key_id,
                                              settings.kalshi_private_key_path)
                       if settings.has_kalshi_auth else None)
        self.use_ws = self.signer is not None and not force_rest
        self.rest = KalshiRest(settings.rest_base, signer=self.signer,
                               max_rps=settings.kalshi_max_rps)
        self.espn = EspnClient()
        self.raw_dir = Path(settings.data_dir) / "raw"
        self.sink = ParquetSink(self.raw_dir)
        self.ws = (KalshiWS(settings.ws_url, self.signer, on_message=self._on_ws)
                   if self.use_ws else None)
        self.games: list[GameMarkets] = []
        self.plays = PlayTracker()
        self._trade_cursor: dict[str, int] = {}  # ticker -> last trade ts (s)
        self._seen_trades: set[str] = set()
        self._last_trade_poll: dict[str, float] = {}
        self._finalized: set[str] = set()
        self._last_book_poll: dict[str, float] = {}
        self._last_book_write: dict[str, tuple[str, float]] = {}  # ticker -> (levels, time)
        self._last_odds: dict[str, float] = {}
        self.stats = {"book_checks": 0, "book_mismatches": 0}

    # ---- helpers -------------------------------------------------------------------------
    @property
    def all_tickers(self) -> list[str]:
        return sorted({t for g in self.games for t in g.tickers})

    def live_games(self) -> list[GameMarkets]:
        return [g for g in self.games if g.game and g.game.is_live]

    def _near_tip(self, g: GameMarkets, hours: float) -> bool:
        if not g.game:
            return False
        dt = (g.game.tip_utc - datetime.now(UTC)).total_seconds() / 3600
        return -4 <= dt <= hours

    # ---- loops ---------------------------------------------------------------------------
    async def schedule_loop(self) -> None:
        while True:
            try:
                await self.refresh_schedule()
            except Exception:
                log.exception("schedule refresh failed")
            interval = 60.0 if self.live_games() else self.s.schedule_refresh_s
            await asyncio.sleep(interval)

    async def refresh_schedule(self) -> None:
        today_et = datetime.now(ET).date()
        espn_games = []
        for day in (today_et - timedelta(days=1), today_et, today_et + timedelta(days=1)):
            espn_games += await self.espn.scoreboard(day)
        events = await self.rest.get_events(NBA_GAME_SERIES, status="open")
        # Keep games still being settled (closed but not yet final on Kalshi) by also
        # matching ESPN games that are live/recent even if their event left "open".
        self.games = [g for g in match_games(espn_games, events)
                      if g.markets or (g.game and (g.game.is_live or self._near_tip(g, 1)))]
        recv = now_ms()
        for g in self.games:
            gm = g.game
            self.sink.write("games", {
                "recv_ms": recv, "event_ticker": g.event_ticker, "tickers": ",".join(g.tickers),
                "espn_id": gm.espn_id if gm else None,
                "tip_utc": gm.tip_utc.isoformat() if gm else None,
                "state": gm.state if gm else None, "period": gm.period if gm else None,
                "clock": gm.clock if gm else None,
                "home": gm.home.code if gm else None, "away": gm.away.code if gm else None,
                "home_score": gm.home_score if gm else None,
                "away_score": gm.away_score if gm else None,
            })
        if self.ws is not None:
            await self.ws.set_markets(self.all_tickers)
        log.info("schedule: %d events, %d markets, %d live", len(self.games),
                 len(self.all_tickers), len(self.live_games()))
        await self._snapshot_odds()

    async def _snapshot_odds(self) -> None:
        """Free pregame odds from ESPN's pickcenter, every 15 min for games within 12h."""
        for g in self.games:
            if not g.game or g.game.state != "pre" or not self._near_tip(g, 12):
                continue
            if time.time() - self._last_odds.get(g.event_ticker, 0) < 900:
                continue
            try:
                summ = await self.espn.summary(g.game.espn_id)
            except Exception as e:  # noqa: BLE001
                log.warning("espn odds fetch failed for %s: %s", g.event_ticker, e)
                continue
            self._last_odds[g.event_ticker] = time.time()
            self.sink.write("espn_odds", {
                "event_ticker": g.event_ticker, "espn_id": g.game.espn_id,
                "pickcenter_json": to_json(summ.get("pickcenter")),
                "odds_json": to_json(summ.get("odds")),
                "injuries_json": to_json(summ.get("injuries")),
            })

    def hot_tickers(self) -> set[str]:
        """Markets for games that are live or tip within the hour."""
        return {t for g in self.games if g.game and (g.game.is_live or self._near_tip(g, 1))
                for t in g.tickers}

    async def book_poll_loop(self) -> None:
        """REST mode: batch order-book snapshots. Hot markets every book_poll_live_s, others
        every book_poll_idle_s. Unchanged books are skipped (a heartbeat row every 60s still
        proves the recorder was watching)."""
        while True:
            start = time.monotonic()
            hot = self.hot_tickers()
            due = [t for t in self.all_tickers
                   if start - self._last_book_poll.get(t, 0)
                   >= (self.s.book_poll_live_s if t in hot else self.s.book_poll_idle_s) - 0.05]
            if due:
                try:
                    books = await self.rest.get_orderbooks(due)
                    recv = now_ms()
                    for t in due:
                        self._last_book_poll[t] = start
                    for t, ob in books.items():
                        self._write_book(t, ob, recv, source="rest_poll")
                except Exception:
                    log.exception("book poll failed")
            await asyncio.sleep(max(0.0, self.s.book_poll_live_s - (time.monotonic() - start)))

    def _write_book(self, ticker: str, ob_fp: dict, recv: int, source: str) -> OrderBook:
        book = OrderBook.from_rest(ticker, ob_fp)
        levels = to_json(book.to_levels())
        prev = self._last_book_write.get(ticker)
        if prev and prev[0] == levels and time.time() - prev[1] < 60:
            return book
        self._last_book_write[ticker] = (levels, time.time())
        self.sink.write("kalshi_books", {
            "recv_ms": recv, "source": source, "ticker": ticker,
            "changed": prev is None or prev[0] != levels,
            "yes_bid": float(book.best_yes_bid()) if book.yes else None,
            "yes_ask": float(book.best_yes_ask()) if book.no else None,
            "levels_json": levels,
        })
        return book

    async def trade_poll_loop(self) -> None:
        """REST mode: trades for live/near-tip markets every few seconds, others every 2 min."""
        while True:
            hot = self.hot_tickers()
            for t in self.all_tickers:
                every = self.s.trades_poll_s if t in hot else 120.0
                if t not in self._last_trade_poll and t not in hot:
                    # Stagger cold markets across the window instead of polling all at once.
                    self._last_trade_poll[t] = time.monotonic() - random.uniform(0, every)
                    continue
                if time.monotonic() - self._last_trade_poll.get(t, 0) < every:
                    continue
                self._last_trade_poll[t] = time.monotonic()
                try:
                    await self._poll_trades(t)
                except Exception as e:  # noqa: BLE001
                    log.warning("trade poll failed for %s: %s", t, e)
            await asyncio.sleep(1.0)

    async def _poll_trades(self, ticker: str) -> None:
        min_ts = self._trade_cursor.get(ticker)
        if min_ts is None:
            # First poll: only a short lookback. Full trade history stays available from
            # GET /markets/trades and is backfilled separately (Phase 1), not by the recorder.
            min_ts = int(time.time()) - 600
        trades = await self.rest.get_trades(ticker, min_ts=min_ts - 1, max_pages=5)
        recv = now_ms()
        newest = min_ts
        for tr in trades:
            tid = tr.get("trade_id")
            if tid in self._seen_trades:
                continue
            self._seen_trades.add(tid)
            created = tr.get("created_time", "")
            with contextlib.suppress(ValueError):
                newest = max(newest, int(datetime.fromisoformat(
                    created.replace("Z", "+00:00")).timestamp()))
            self.sink.write("kalshi_trades", {
                "recv_ms": recv, "source": "rest_poll", "trade_id": tid, "ticker": ticker,
                "created_time": created, "yes_price": float(tr.get("yes_price_dollars") or 0),
                "count": float(tr.get("count_fp") or 0),
                "taker_outcome_side": tr.get("taker_outcome_side"),
                "is_block_trade": bool(tr.get("is_block_trade")),
            })
        self._trade_cursor[ticker] = newest

    async def _on_ws(self, msg: dict, recv_ms: int) -> None:
        body = msg.get("msg") or {}
        self.sink.write("kalshi_ws", {
            "recv_ms": recv_ms, "type": msg.get("type"), "sid": msg.get("sid"),
            "seq": msg.get("seq"), "ticker": body.get("market_ticker"),
            "gap": bool(msg.get("_gap")), "raw_json": to_json(msg),
        })

    async def book_check_loop(self) -> None:
        """WS mode: compare rebuilt books to REST snapshots (Phase 0 exit gate)."""
        while True:
            await asyncio.sleep(60)
            if self.ws is None or not self.ws.books:
                continue
            try:
                tickers = sorted(self.ws.books)
                rest_books = await self.rest.get_orderbooks(tickers)
                recv = now_ms()
                for t, ob in rest_books.items():
                    local = self.ws.books.get(t)
                    if local is None:
                        continue
                    remote = OrderBook.from_rest(t, ob)
                    exact = local.yes == remote.yes and local.no == remote.no
                    top = (local.best_yes_bid() == remote.best_yes_bid()
                           and local.best_yes_ask() == remote.best_yes_ask())
                    self.stats["book_checks"] += 1
                    if not exact:
                        self.stats["book_mismatches"] += 1
                    self.sink.write("book_checks", {
                        "recv_ms": recv, "ticker": t, "exact_match": exact, "top_match": top,
                        "ws_age_ms": recv - local.last_update_ms if local.last_update_ms else None,
                        "ws_gaps_total": self.ws.gaps, "ws_reconnects": self.ws.reconnects,
                    })
            except Exception:
                log.exception("book check failed")

    async def espn_live_loop(self) -> None:
        while True:
            for g in list(self.games):
                gm = g.game
                if not gm:
                    continue
                recently_final = gm.is_final and gm.espn_id not in self._finalized
                if not (gm.is_live or recently_final):
                    continue
                try:
                    summ = await self.espn.summary(gm.espn_id)
                except Exception as e:  # noqa: BLE001
                    log.warning("espn summary failed for %s: %s", gm.espn_id, e)
                    continue
                recv = now_ms()
                for p in self.plays.new_plays(parse_plays(gm.espn_id, summ, recv)):
                    self.sink.write("espn_plays", {"event_ticker": g.event_ticker,
                                                   **p.to_record()})
                status = (summ.get("header", {}).get("competitions") or [{}])[0].get("status", {})
                self.sink.write("espn_state", {
                    "recv_ms": recv, "event_ticker": g.event_ticker, "espn_id": gm.espn_id,
                    "status_json": to_json(status),
                })
                if recently_final:
                    self._save_summary(gm.espn_id, summ)
                    self._finalized.add(gm.espn_id)
            await asyncio.sleep(self.s.espn_live_poll_s)

    def _save_summary(self, espn_id: str, summ: dict) -> None:
        out = self.raw_dir / "espn_summaries"
        out.mkdir(parents=True, exist_ok=True)
        with gzip.open(out / f"{espn_id}.json.gz", "wt", encoding="utf-8") as f:
            f.write(to_json(summ))

    async def flush_loop(self) -> None:
        while True:
            await asyncio.sleep(10)
            self.sink.flush()

    async def status_loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            extra = (f" ws_gaps={self.ws.gaps} reconnects={self.ws.reconnects}"
                     if self.ws else "")
            log.info("rows written: %s%s checks=%s", dict(self.sink.rows_written), extra,
                     self.stats)

    # ---- entry ---------------------------------------------------------------------------
    async def run(self, duration_s: float | None = None) -> None:
        log.info("recorder starting (mode=%s, env=%s, data=%s)",
                 "websocket" if self.use_ws else "rest-poll", self.s.kalshi_env, self.raw_dir)
        await self.refresh_schedule()
        loops = [self.schedule_loop(), self.espn_live_loop(), self.flush_loop(),
                 self.status_loop()]
        if self.ws is not None:
            loops += [self.ws.run(), self.book_check_loop()]
        else:
            loops += [self.book_poll_loop(), self.trade_poll_loop()]
        tasks = [asyncio.create_task(c) for c in loops]
        try:
            if duration_s:
                await asyncio.sleep(duration_s)
            else:
                await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.ws is not None:
                await self.ws.stop()
            n = self.sink.flush()
            await self.rest.aclose()
            await self.espn.aclose()
            log.info("recorder stopped; final flush %d rows; totals %s", n,
                     dict(self.sink.rows_written))
