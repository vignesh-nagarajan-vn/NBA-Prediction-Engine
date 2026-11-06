from datetime import UTC, datetime

import pytest

from nba_engine.data.espn import EspnGame, parse_scoreboard
from nba_engine.data.schedule import kalshi_event_ticker, match_games, parse_event_ticker
from nba_engine.data.teams import BY_CODE, TEAMS, from_espn


def _game(away, home, tip):
    return EspnGame("1", tip, BY_CODE[home], BY_CODE[away], "STATUS_SCHEDULED", "pre", 0,
                    "0.0", 0, 0)


def test_thirty_unique_teams():
    assert len(TEAMS) == 30
    assert len({t.code for t in TEAMS}) == 30
    assert len({t.espn for t in TEAMS}) == 30
    assert from_espn("GS").code == "GSW"
    assert from_espn("utah").code == "UTA"


def test_event_ticker_uses_eastern_date():
    # ORL @ PHX tips 02:00Z Nov 6 = 9pm ET Nov 5 -> Kalshi dates it Nov 5
    tip = datetime(2026, 11, 6, 2, 0, tzinfo=UTC)
    assert kalshi_event_ticker("ORL", "PHX", tip) == "KXNBAGAME-26NOV05ORLPHX"


def test_parse_event_ticker():
    assert parse_event_ticker("KXNBAGAME-26NOV05MIAGSW") == ("26NOV05", "MIA", "GSW")
    with pytest.raises(ValueError):
        parse_event_ticker("KXNBAGAME-26NOV05MIAGS")


def test_match_games_joins_and_keeps_unmatched_events():
    tip = datetime(2026, 11, 6, 3, 0, tzinfo=UTC)
    games = [_game("MIA", "GSW", tip)]
    events = [
        {"event_ticker": "KXNBAGAME-26NOV05MIAGSW", "markets": [
            {"ticker": "KXNBAGAME-26NOV05MIAGSW-MIA"}, {"ticker": "KXNBAGAME-26NOV05MIAGSW-GSW"}]},
        {"event_ticker": "KXNBAGAME-26NOV09LALATL", "markets": [
            {"ticker": "KXNBAGAME-26NOV09LALATL-LAL"}]},
    ]
    out = match_games(games, events)
    assert out[0].game is games[0]
    assert set(out[0].markets) == {"MIA", "GSW"}
    assert out[0].label == "MIA @ GSW"
    assert out[1].game is None and out[1].tickers == ["KXNBAGAME-26NOV09LALATL-LAL"]


def test_parse_scoreboard_minimal():
    data = {"events": [{
        "id": "401909937", "date": "2026-11-06T03:00Z",
        "status": {"period": 2, "displayClock": "5:12",
                   "type": {"name": "STATUS_IN_PROGRESS", "state": "in"}},
        "competitions": [{"competitors": [
            {"homeAway": "home", "score": "55", "team": {"abbreviation": "GS"}},
            {"homeAway": "away", "score": "50", "team": {"abbreviation": "MIA"}},
        ]}],
    }]}
    (g,) = parse_scoreboard(data)
    assert (g.home.code, g.away.code, g.home_score, g.away_score) == ("GSW", "MIA", 55, 50)
    assert g.is_live and g.period == 2
