"""Kalshi WebSocket client with local order books, gap detection and auto-reconnect.

- The connection itself requires a signed handshake (GET /trade-api/ws/v2), even for public
  channels.
- orderbook_delta sends an orderbook_snapshot per market, then deltas. `seq` is contiguous
  per subscription (sid); on a gap we request fresh snapshots via update_subscription
  (action get_snapshot) and keep applying.
- Every raw message is passed to `on_message` (for recording) with the local receive time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from urllib.parse import urlsplit

import websockets

from nba_engine.kalshi.auth import KalshiSigner
from nba_engine.kalshi.orderbook import OrderBook

log = logging.getLogger(__name__)

MessageHandler = Callable[[dict, int], Awaitable[None] | None]


class SequenceTracker:
    """Tracks contiguous `seq` per subscription id. Returns True when a gap is detected."""

    def __init__(self) -> None:
        self._last: dict[int, int] = {}

    def check(self, sid: int, seq: int | None) -> bool:
        if seq is None:
            return False
        last = self._last.get(sid)
        self._last[sid] = seq
        return last is not None and seq != last + 1

    def reset(self) -> None:
        self._last.clear()


class KalshiWS:
    def __init__(
        self,
        url: str,
        signer: KalshiSigner,
        on_message: MessageHandler | None = None,
        channels: Iterable[str] = ("orderbook_delta", "ticker", "trade"),
    ):
        self.url = url
        self.signer = signer
        self.on_message = on_message
        self.channels = list(channels)
        self.books: dict[str, OrderBook] = {}
        self.tickers: set[str] = set()
        self.gaps = 0
        self.reconnects = 0
        self._seq = SequenceTracker()
        self._ws = None
        self._cmd_id = 0
        self._sids: dict[str, int] = {}  # channel -> sid
        self._stop = asyncio.Event()

    # ---- public API ----------------------------------------------------------------------
    async def set_markets(self, tickers: Iterable[str]) -> None:
        """Change the subscribed market set (adds/removes on the live connection)."""
        new = set(tickers)
        added, removed = new - self.tickers, self.tickers - new
        self.tickers = new
        for t in removed:
            self.books.pop(t, None)
        if self._ws is None:
            return
        if not self._sids:
            if added:
                await self._subscribe(sorted(added))
            return
        for sid in self._sids.values():
            if added:
                await self._send({"cmd": "update_subscription",
                                  "params": {"sid": sid, "action": "add_markets",
                                             "market_tickers": sorted(added)}})
            if removed:
                await self._send({"cmd": "update_subscription",
                                  "params": {"sid": sid, "action": "delete_markets",
                                             "market_tickers": sorted(removed)}})

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._run_once()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - reconnect on anything
                log.warning("kalshi ws disconnected: %r; reconnecting in %.1fs", e, backoff)
            finally:
                self._ws = None
                self._sids.clear()
                self._seq.reset()
            if self._stop.is_set():
                break
            self.reconnects += 1
            await asyncio.sleep(backoff * (0.5 + random.random()))
            backoff = min(backoff * 2, 60.0)

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            await self._ws.close()

    # ---- internals -----------------------------------------------------------------------
    async def _run_once(self) -> None:
        headers = self.signer.headers("GET", urlsplit(self.url).path)
        async with websockets.connect(self.url, additional_headers=headers, ping_interval=20,
                                      ping_timeout=20, max_size=2**24) as ws:
            self._ws = ws
            log.info("kalshi ws connected (%d markets)", len(self.tickers))
            if self.tickers:
                await self._subscribe(sorted(self.tickers))
            async for raw in ws:
                recv_ms = int(time.time() * 1000)
                msg = json.loads(raw)
                await self._handle(msg, recv_ms)

    async def _subscribe(self, tickers: list[str]) -> None:
        await self._send({"cmd": "subscribe",
                          "params": {"channels": self.channels, "market_tickers": tickers}})

    async def _send(self, payload: dict) -> int:
        self._cmd_id += 1
        payload = {"id": self._cmd_id, **payload}
        if self._ws is not None:
            await self._ws.send(json.dumps(payload))
        return self._cmd_id

    async def _handle(self, msg: dict, recv_ms: int) -> None:
        mtype = msg.get("type")
        body = msg.get("msg", {}) or {}
        if mtype == "subscribed" and body.get("channel"):
            self._sids[body["channel"]] = body.get("sid")
        elif mtype in ("orderbook_snapshot", "orderbook_delta"):
            await self._handle_book(mtype, msg, body, recv_ms)
        elif mtype == "error":
            log.error("kalshi ws error: %s", body)
        elif mtype in ("trade",) and self._seq.check(msg.get("sid", -1), msg.get("seq")):
            # Trades can't be replayed over WS; flag the gap so REST backfill can cover it.
            self.gaps += 1
            log.warning("kalshi ws trade seq gap on sid %s", msg.get("sid"))
            msg["_gap"] = True
        if self.on_message is not None:
            res = self.on_message(msg, recv_ms)
            if asyncio.iscoroutine(res):
                await res

    async def _handle_book(self, mtype: str, msg: dict, body: dict, recv_ms: int) -> None:
        ticker = body.get("market_ticker")
        if not ticker:
            return
        sid = msg.get("sid", -1)
        gap = self._seq.check(sid, msg.get("seq"))
        book = self.books.setdefault(ticker, OrderBook(ticker))
        if mtype == "orderbook_snapshot":
            book.apply_snapshot(body.get("yes_dollars_fp"), body.get("no_dollars_fp"),
                                clock=body.get("clock", 0), ts_ms=recv_ms)
        else:
            book.apply_delta(body["side"], body["price_dollars"], body["delta_fp"],
                             clock=body.get("clock", 0), ts_ms=body.get("ts_ms") or recv_ms)
        if gap:
            self.gaps += 1
            msg["_gap"] = True
            log.warning("kalshi ws book seq gap on sid %s; requesting snapshots", sid)
            await self._send({"cmd": "update_subscription",
                              "params": {"sid": sid, "action": "get_snapshot",
                                         "market_tickers": sorted(self.tickers)}})
