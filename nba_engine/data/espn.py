"""ESPN public JSON (scoreboard + game summary with play-by-play).

These endpoints are free and unauthenticated but unofficial: they can change without notice
and lag the arena by several seconds. Every play carries ESPN's `wallclock`, so we can
measure that lag against our own receive time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

import httpx

from nba_engine.data.teams import Team, from_espn

log = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


@dataclass
class EspnGame:
    espn_id: str
    tip_utc: datetime
    home: Team
    away: Team
    status: str  # STATUS_SCHEDULED / STATUS_IN_PROGRESS / STATUS_HALFTIME / STATUS_FINAL ...
    state: str  # pre / in / post
    period: int
    clock: str
    home_score: int
    away_score: int

    @property
    def is_live(self) -> bool:
        return self.state == "in"

    @property
    def is_final(self) -> bool:
        return self.state == "post"


def parse_scoreboard(data: dict) -> list[EspnGame]:
    games = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        try:
            home = from_espn(teams["home"]["team"]["abbreviation"])
            away = from_espn(teams["away"]["team"]["abbreviation"])
        except KeyError as e:  # All-Star / exhibition teams
            log.debug("skipping ESPN event %s: %s", ev.get("id"), e)
            continue
        st = ev.get("status", {})
        games.append(EspnGame(
            espn_id=str(ev["id"]),
            tip_utc=datetime.fromisoformat(ev["date"].replace("Z", "+00:00")),
            home=home,
            away=away,
            status=st.get("type", {}).get("name", ""),
            state=st.get("type", {}).get("state", ""),
            period=int(st.get("period") or 0),
            clock=str(st.get("displayClock") or ""),
            home_score=int(teams["home"].get("score") or 0),
            away_score=int(teams["away"].get("score") or 0),
        ))
    return games


class EspnClient:
    def __init__(self, timeout: float = 10.0):
        self._client = httpx.AsyncClient(timeout=timeout, headers={"Accept": "application/json"})

    async def aclose(self) -> None:
        await self._client.aclose()

    async def scoreboard_raw(self, day: date | None = None) -> dict:
        params = {"dates": day.strftime("%Y%m%d")} if day else None
        r = await self._client.get(f"{BASE}/scoreboard", params=params)
        r.raise_for_status()
        return r.json()

    async def scoreboard(self, day: date | None = None) -> list[EspnGame]:
        return parse_scoreboard(await self.scoreboard_raw(day))

    async def summary(self, espn_id: str) -> dict:
        r = await self._client.get(f"{BASE}/summary", params={"event": espn_id})
        r.raise_for_status()
        return r.json()
