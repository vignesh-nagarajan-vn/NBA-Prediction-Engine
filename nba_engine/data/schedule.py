"""Map ESPN games to Kalshi KXNBAGAME events and markets.

Kalshi event ticker: KXNBAGAME-{YY}{MON}{DD}{AWAY}{HOME}, with the date being the game's
US/Eastern calendar date, e.g. KXNBAGAME-26NOV05ORLPHX (Orlando at Phoenix, Nov 5 2026).
Each event has one market per team: {event}-{TEAM}, YES = that team wins.
Kalshi's `occurrence_datetime` is NOT tipoff (it is roughly tip + 3h), so tip times
come from ESPN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from nba_engine.config.settings import NBA_GAME_SERIES
from nba_engine.data.espn import EspnGame

ET = ZoneInfo("America/New_York")


def kalshi_event_ticker(away_code: str, home_code: str, tip_utc: datetime,
                        series: str = NBA_GAME_SERIES) -> str:
    d = tip_utc.astimezone(ET)
    return f"{series}-{d:%y}{d.strftime('%b').upper()}{d:%d}{away_code}{home_code}"


def parse_event_ticker(event_ticker: str) -> tuple[str, str, str]:
    """Returns (date_part 'YYMONDD', away_code, home_code). Team codes are 3 letters."""
    tail = event_ticker.split("-", 1)[1]
    date_part, teams = tail[:7], tail[7:]
    if len(teams) != 6:
        raise ValueError(f"unexpected event ticker format: {event_ticker}")
    return date_part, teams[:3], teams[3:]


@dataclass
class GameMarkets:
    game: EspnGame | None
    event_ticker: str
    markets: dict[str, dict] = field(default_factory=dict)  # team code -> Kalshi market dict

    @property
    def tickers(self) -> list[str]:
        return sorted(m["ticker"] for m in self.markets.values())

    @property
    def label(self) -> str:
        _, away, home = parse_event_ticker(self.event_ticker)
        return f"{away} @ {home}"


def match_games(espn_games: list[EspnGame], kalshi_events: list[dict]) -> list[GameMarkets]:
    """Join ESPN games to Kalshi events. Kalshi events without an ESPN match are kept with
    game=None (e.g. games further out than the ESPN scoreboard we fetched)."""
    by_ticker = {e["event_ticker"]: e for e in kalshi_events}
    out: list[GameMarkets] = []
    seen: set[str] = set()
    for g in espn_games:
        et = kalshi_event_ticker(g.away.code, g.home.code, g.tip_utc)
        ev = by_ticker.get(et)
        markets = {}
        if ev:
            for m in ev.get("markets", []):
                markets[m["ticker"].rsplit("-", 1)[1]] = m
            seen.add(et)
        out.append(GameMarkets(game=g, event_ticker=et, markets=markets))
    for et, ev in by_ticker.items():
        if et in seen:
            continue
        markets = {m["ticker"].rsplit("-", 1)[1]: m for m in ev.get("markets", [])}
        out.append(GameMarkets(game=None, event_ticker=et, markets=markets))
    return out
