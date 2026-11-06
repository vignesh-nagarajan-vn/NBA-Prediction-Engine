"""Smoke tests against live public endpoints (no API key). Run: pytest -m network"""

import pytest

from nba_engine.config.settings import KALSHI_HOSTS, NBA_GAME_SERIES
from nba_engine.data.espn import EspnClient
from nba_engine.kalshi.orderbook import OrderBook
from nba_engine.kalshi.rest import KalshiRest

pytestmark = pytest.mark.network


async def test_kalshi_public_market_data():
    async with KalshiRest(KALSHI_HOSTS["prod"]["rest"]) as rest:
        series = await rest.get_series(NBA_GAME_SERIES)
        assert series["fee_type"] in {"quadratic", "quadratic_with_maker_fees"}
        events = await rest.get_events(NBA_GAME_SERIES, status="open")
        tickers = [m["ticker"] for e in events for m in e["markets"]]
        if not tickers:
            pytest.skip("no open NBA markets right now")
        books = await rest.get_orderbooks(tickers[:4])
        assert set(books) <= set(tickers)
        for t, ob in books.items():
            book = OrderBook.from_rest(t, ob)
            assert not book.is_crossed()


async def test_espn_scoreboard():
    espn = EspnClient()
    try:
        games = await espn.scoreboard()
    finally:
        await espn.aclose()
    assert isinstance(games, list)
