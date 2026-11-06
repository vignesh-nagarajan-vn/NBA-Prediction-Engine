"""Async Kalshi REST client (trade-api v2).

Public market-data calls work without a key. Portfolio and order calls need a KalshiSigner.
Order placement uses the V2 endpoint (`POST /portfolio/events/orders`, single-book bid/ask,
fixed-point dollar prices); the legacy `/portfolio/orders` POST is being deprecated.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from decimal import Decimal
from typing import Any, Literal

import httpx

from nba_engine.kalshi.auth import KalshiSigner

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class KalshiAPIError(RuntimeError):
    def __init__(self, status: int, body: str, method: str, path: str):
        super().__init__(f"{method} {path} -> HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class RateLimiter:
    """Simple token bucket: `rate` requests/second, burst of `burst`."""

    def __init__(self, rate: float, burst: float | None = None):
        self.rate = rate
        self.capacity = burst if burst is not None else max(1.0, rate)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()
        self._cooldown_until = 0.0

    def cooldown(self, seconds: float) -> None:
        """Pause ALL requests (called on HTTP 429 so concurrent loops back off together)."""
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)
        self._tokens = 0.0

    async def acquire(self, cost: float = 1.0) -> None:
        async with self._lock:
            wait = self._cooldown_until - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                await asyncio.sleep((cost - self._tokens) / self.rate)


class KalshiRest:
    def __init__(
        self,
        base_url: str,
        signer: KalshiSigner | None = None,
        max_rps: float = 8.0,
        timeout: float = 10.0,
        max_retries: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self.limiter = RateLimiter(max_rps)
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"Accept": "application/json", "User-Agent": "nba-engine/0.1"}
        )

    async def __aenter__(self) -> KalshiRest:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- core ----------------------------------------------------------------------------
    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | list[tuple[str, Any]] | None = None,
        json: dict | None = None,
        auth: bool = False,
    ) -> Any:
        url = self.base_url + path
        for attempt in range(self.max_retries + 1):
            await self.limiter.acquire()
            headers = {}
            if auth or self.signer is not None:
                if self.signer is None:
                    raise RuntimeError(f"{method} {path} requires a Kalshi API key")
                headers = self.signer.headers(method, url)
            try:
                resp = await self._client.request(method, url, params=params, json=json,
                                                  headers=headers)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if attempt >= self.max_retries:
                    raise
                delay = min(30.0, 0.5 * 2**attempt) * (0.5 + random.random())
                log.warning("kalshi %s %s transport error %s; retry in %.1fs", method, path, e,
                            delay)
                await asyncio.sleep(delay)
                continue
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                delay = (float(retry_after) if retry_after and retry_after.isdigit()
                         else min(30.0, 1.0 * 2**attempt) * (0.75 + 0.5 * random.random()))
                if resp.status_code == 429:
                    self.limiter.cooldown(delay)
                log.warning("kalshi %s %s HTTP %s; retry in %.1fs", method, path,
                            resp.status_code, delay)
                await asyncio.sleep(delay)
                continue
            if resp.status_code >= 400:
                raise KalshiAPIError(resp.status_code, resp.text, method, path)
            if resp.status_code == 204 or not resp.content:
                return None
            return resp.json()
        raise AssertionError("unreachable")

    async def paginate(self, path: str, key: str, params: dict[str, Any] | None = None,
                       auth: bool = False, max_pages: int = 1000) -> AsyncIterator[dict]:
        params = dict(params or {})
        for _ in range(max_pages):
            data = await self.request("GET", path, params=params, auth=auth)
            for item in data.get(key, []) or []:
                yield item
            cursor = data.get("cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    # ---- exchange / market data (public) -------------------------------------------------
    async def exchange_status(self) -> dict:
        return await self.request("GET", "/exchange/status")

    async def get_series(self, series_ticker: str) -> dict:
        return (await self.request("GET", f"/series/{series_ticker}"))["series"]

    async def get_events(self, series_ticker: str, status: str | None = "open",
                         with_nested_markets: bool = True) -> list[dict]:
        params: dict[str, Any] = {"series_ticker": series_ticker, "limit": 200,
                                  "with_nested_markets": str(with_nested_markets).lower()}
        if status:
            params["status"] = status
        return [e async for e in self.paginate("/events", "events", params)]

    async def get_event(self, event_ticker: str) -> dict:
        return await self.request("GET", f"/events/{event_ticker}",
                                  params={"with_nested_markets": "true"})

    async def get_markets(self, **params) -> list[dict]:
        params.setdefault("limit", 1000)
        return [m async for m in self.paginate("/markets", "markets", params)]

    async def get_market(self, ticker: str) -> dict:
        return (await self.request("GET", f"/markets/{ticker}"))["market"]

    async def get_orderbook(self, ticker: str, depth: int | None = None) -> dict:
        params = {"depth": depth} if depth else None
        return (await self.request("GET", f"/markets/{ticker}/orderbook", params=params))[
            "orderbook_fp"]

    async def get_orderbooks(self, tickers: Iterable[str]) -> dict[str, dict]:
        """Batch order books, 100 tickers per request. Returns {ticker: orderbook_fp}."""
        tickers = list(tickers)
        out: dict[str, dict] = {}
        for i in range(0, len(tickers), 100):
            chunk = tickers[i:i + 100]
            data = await self.request("GET", "/markets/orderbooks",
                                      params=[("tickers", t) for t in chunk])
            for ob in data.get("orderbooks", []):
                out[ob["ticker"]] = ob.get("orderbook_fp") or {}
        return out

    async def get_trades(self, ticker: str | None = None, min_ts: int | None = None,
                         max_ts: int | None = None, limit: int = 1000,
                         max_pages: int = 50) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        return [t async for t in self.paginate("/markets/trades", "trades", params,
                                               max_pages=max_pages)]

    async def get_candlesticks(self, series_ticker: str, ticker: str, start_ts: int,
                               end_ts: int, period_interval: int = 1) -> list[dict]:
        data = await self.request(
            "GET", f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval})
        return data.get("candlesticks", [])

    # ---- portfolio (auth) ----------------------------------------------------------------
    async def get_balance(self) -> dict:
        return await self.request("GET", "/portfolio/balance", auth=True)

    async def get_positions(self, **params) -> dict:
        return await self.request("GET", "/portfolio/positions", params=params or None,
                                  auth=True)

    async def get_orders(self, **params) -> list[dict]:
        return [o async for o in self.paginate("/portfolio/orders", "orders", params,
                                               auth=True)]

    async def get_fills(self, **params) -> list[dict]:
        return [f async for f in self.paginate("/portfolio/fills", "fills", params, auth=True)]

    # ---- orders (auth) -------------------------------------------------------------------
    async def create_order(
        self,
        ticker: str,
        side: Literal["bid", "ask"],
        count: Decimal | int | str,
        price: Decimal | str,
        *,
        time_in_force: Literal["good_till_canceled", "immediate_or_cancel",
                               "fill_or_kill"] = "good_till_canceled",
        post_only: bool = False,
        reduce_only: bool = False,
        expiration_time: int | None = None,
        client_order_id: str | None = None,
        self_trade_prevention_type: Literal["taker_at_cross", "maker"] = "taker_at_cross",
        cancel_order_on_pause: bool = True,
    ) -> dict:
        """Place an order on the single YES book. side='bid' buys YES, side='ask' sells YES
        (equivalently buys NO at 1 - price). Price is in dollars, e.g. "0.5600"."""
        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "count": f"{Decimal(str(count)):.2f}",
            "price": f"{Decimal(str(price)):.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention_type,
            "post_only": post_only,
            "reduce_only": reduce_only,
            "cancel_order_on_pause": cancel_order_on_pause,
            "client_order_id": client_order_id or str(uuid.uuid4()),
        }
        if expiration_time is not None:
            body["expiration_time"] = expiration_time
        return await self.request("POST", "/portfolio/events/orders", json=body, auth=True)

    async def cancel_order(self, order_id: str) -> dict | None:
        return await self.request("DELETE", f"/portfolio/events/orders/{order_id}", auth=True)

    async def cancel_all_orders(self) -> None:
        await self.request("DELETE", "/portfolio/events/orders", auth=True)
