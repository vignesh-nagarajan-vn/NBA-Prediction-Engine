"""Canonical NBA team table and abbreviation mapping across sources.

Canonical code = Kalshi's abbreviation (used in KXNBAGAME tickers).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    code: str  # canonical (Kalshi)
    espn: str
    name: str


TEAMS: tuple[Team, ...] = (
    Team("ATL", "ATL", "Atlanta Hawks"),
    Team("BOS", "BOS", "Boston Celtics"),
    Team("BKN", "BKN", "Brooklyn Nets"),
    Team("CHA", "CHA", "Charlotte Hornets"),
    Team("CHI", "CHI", "Chicago Bulls"),
    Team("CLE", "CLE", "Cleveland Cavaliers"),
    Team("DAL", "DAL", "Dallas Mavericks"),
    Team("DEN", "DEN", "Denver Nuggets"),
    Team("DET", "DET", "Detroit Pistons"),
    Team("GSW", "GS", "Golden State Warriors"),
    Team("HOU", "HOU", "Houston Rockets"),
    Team("IND", "IND", "Indiana Pacers"),
    Team("LAC", "LAC", "LA Clippers"),
    Team("LAL", "LAL", "Los Angeles Lakers"),
    Team("MEM", "MEM", "Memphis Grizzlies"),
    Team("MIA", "MIA", "Miami Heat"),
    Team("MIL", "MIL", "Milwaukee Bucks"),
    Team("MIN", "MIN", "Minnesota Timberwolves"),
    Team("NOP", "NO", "New Orleans Pelicans"),
    Team("NYK", "NY", "New York Knicks"),
    Team("OKC", "OKC", "Oklahoma City Thunder"),
    Team("ORL", "ORL", "Orlando Magic"),
    Team("PHI", "PHI", "Philadelphia 76ers"),
    Team("PHX", "PHX", "Phoenix Suns"),
    Team("POR", "POR", "Portland Trail Blazers"),
    Team("SAC", "SAC", "Sacramento Kings"),
    Team("SAS", "SA", "San Antonio Spurs"),
    Team("TOR", "TOR", "Toronto Raptors"),
    Team("UTA", "UTAH", "Utah Jazz"),
    Team("WAS", "WSH", "Washington Wizards"),
)

BY_CODE = {t.code: t for t in TEAMS}
BY_ESPN = {t.espn: t for t in TEAMS}


def from_espn(abbr: str) -> Team:
    try:
        return BY_ESPN[abbr.upper()]
    except KeyError as e:
        raise KeyError(f"Unknown ESPN team abbreviation {abbr!r}") from e
