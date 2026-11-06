import json
from decimal import Decimal as D

from cryptography.hazmat.primitives.asymmetric import ed25519

from nba_engine.kalshi.auth import KalshiSigner
from nba_engine.kalshi.ws import KalshiWS, SequenceTracker


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))


def _ws():
    recorded = []
    ws = KalshiWS("wss://example/trade-api/ws/v2",
                  KalshiSigner("k", ed25519.Ed25519PrivateKey.generate()),
                  on_message=lambda m, t: recorded.append(m))
    ws._ws = FakeSocket()
    ws.tickers = {"T"}
    return ws, recorded


def test_sequence_tracker():
    st = SequenceTracker()
    assert not st.check(1, 1)
    assert not st.check(1, 2)
    assert st.check(1, 4)  # gap
    assert not st.check(2, 10)  # independent sid
    assert not st.check(1, None)


async def test_snapshot_then_deltas_build_book():
    ws, recorded = _ws()
    await ws._handle({"type": "subscribed", "msg": {"channel": "orderbook_delta", "sid": 1}}, 0)
    await ws._handle({"type": "orderbook_snapshot", "sid": 1, "seq": 1, "msg": {
        "market_ticker": "T", "clock": 10,
        "yes_dollars_fp": [["0.4500", "100.00"]], "no_dollars_fp": [["0.5300", "50.00"]]}}, 1)
    await ws._handle({"type": "orderbook_delta", "sid": 1, "seq": 2, "msg": {
        "market_ticker": "T", "clock": 11, "price_dollars": "0.4600", "delta_fp": "20.00",
        "side": "yes", "ts_ms": 5}}, 2)
    book = ws.books["T"]
    assert book.best_yes_bid() == D("0.46")
    assert book.best_yes_ask() == D("0.47")
    assert ws.gaps == 0
    assert len(recorded) == 3
    assert ws._sids == {"orderbook_delta": 1}


async def test_gap_requests_snapshot():
    ws, _ = _ws()
    snap = {"market_ticker": "T", "clock": 1, "yes_dollars_fp": [], "no_dollars_fp": []}
    await ws._handle({"type": "orderbook_snapshot", "sid": 3, "seq": 1, "msg": snap}, 0)
    delta = {"market_ticker": "T", "clock": 5, "price_dollars": "0.40", "delta_fp": "1",
             "side": "yes"}
    msg = {"type": "orderbook_delta", "sid": 3, "seq": 5, "msg": delta}
    await ws._handle(msg, 1)
    assert ws.gaps == 1 and msg["_gap"] is True
    cmd = ws._ws.sent[-1]
    assert cmd["cmd"] == "update_subscription"
    assert cmd["params"] == {"sid": 3, "action": "get_snapshot", "market_tickers": ["T"]}


async def test_set_markets_updates_every_channel():
    ws, _ = _ws()
    ws._sids = {"orderbook_delta": 1, "ticker": 2, "trade": 3}
    await ws.set_markets({"T", "U"})
    adds = [c for c in ws._ws.sent if c["params"]["action"] == "add_markets"]
    assert {c["params"]["sid"] for c in adds} == {1, 2, 3}
    assert all(c["params"]["market_tickers"] == ["U"] for c in adds)
