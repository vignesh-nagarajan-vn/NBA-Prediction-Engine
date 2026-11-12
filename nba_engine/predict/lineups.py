"""Projected lineups for games that haven't happened yet.

Roster: players who logged minutes for a team in its last `recent_games` games, whose most
recent appearance was for that team (so traded players move with the trade).
Availability: players listed as Out (or Doubtful) on ESPN's injury report miss games within
`injury_horizon_days`; everyone else is assumed available. Each projected player becomes a
box-score row with zero minutes, which the player-rating code reads as available
("COACH'S DECISION") or unavailable ("OUT (INJURY)"), exactly like historical DNP rows.
"""

from __future__ import annotations

import re

import httpx
import pandas as pd

from nba_engine.data.espn import BASE
from nba_engine.data.teams import BY_ESPN, TEAMS

_ID_RE = re.compile(r"/id/(\d+)/")
_NAME_TO_CODE = {t.name: t.code for t in TEAMS} | {"LA Clippers": "LAC",
                                                  "Los Angeles Clippers": "LAC"}
OUT_STATUSES = {"Out", "Doubtful"}


def fetch_injuries(client: httpx.Client | None = None) -> pd.DataFrame:
    c = client or httpx.Client(timeout=20)
    data = c.get(f"{BASE}/injuries").json()
    rows = []
    for team in data.get("injuries", []):
        code = _NAME_TO_CODE.get(team.get("displayName", ""))
        for inj in team.get("injuries", []):
            ath = inj.get("athlete") or {}
            pid = None
            for link in ath.get("links", []):
                m = _ID_RE.search(link.get("href", ""))
                if m:
                    pid = m.group(1)
                    break
            team_abbr = ((ath.get("team") or {}).get("abbreviation") or "")
            rows.append({
                "player_id": pid,
                "player": ath.get("displayName"),
                "team": code or (BY_ESPN[team_abbr].code if team_abbr in BY_ESPN else None),
                "status": inj.get("status"),
                "date": inj.get("date"),
                "comment": inj.get("shortComment"),
            })
    return pd.DataFrame(rows)


def current_rosters(games: pd.DataFrame, box: pd.DataFrame, as_of: pd.Timestamp,
                    recent_games: int = 8) -> dict[str, list[str]]:
    done = games[games["completed"] & (games["tip_utc"] < as_of)].sort_values("tip_utc")
    b = box.merge(done[["game_id", "tip_utc"]], on="game_id")
    b = b[b["minutes"] > 0].sort_values("tip_utc")
    last_team = b.groupby("player_id")["team"].last()
    rosters: dict[str, list[str]] = {}
    for t in TEAMS:
        tg = done[(done["home"] == t.code) | (done["away"] == t.code)].tail(recent_games)
        played = b[b["game_id"].isin(tg["game_id"]) & (b["team"] == t.code)]
        rosters[t.code] = [p for p in played["player_id"].unique() if last_team.get(p) == t.code]
    return rosters


def projected_box(upcoming: pd.DataFrame, rosters: dict[str, list[str]],
                  injuries: pd.DataFrame | None, as_of: pd.Timestamp,
                  injury_horizon_days: float = 10.0) -> pd.DataFrame:
    out_ids: set[str] = set()
    if injuries is not None and len(injuries):
        out_ids = set(injuries.loc[injuries["status"].isin(OUT_STATUSES), "player_id"].dropna())
    rows = []
    for g in upcoming.itertuples(index=False):
        near = (g.tip_utc - as_of).total_seconds() / 86400 <= injury_horizon_days
        for team in (g.home, g.away):
            for pid in rosters.get(team, []):
                out = near and pid in out_ids
                rows.append({"game_id": g.game_id, "team": team, "player_id": pid,
                             "player": None, "did_not_play": True, "minutes": 0.0,
                             "reason": "OUT (INJURY)" if out else "COACH'S DECISION"})
    return pd.DataFrame(rows)
