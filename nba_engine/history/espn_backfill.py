"""Backfill NBA history from ESPN: games, player box scores and closing lines.

Raw responses are cached gzipped under {data_dir}/cache/espn/ so tables can be rebuilt (and
play-by-play reused for the in-game model in Phase 3) without refetching. Parsed tables go to
{data_dir}/history/*.parquet:

  games        one row per game (regular season, play-in, playoffs)
  player_box   one row per player per game, including DNP / inactive rows
  odds         one row per game per odds provider in ESPN's pickcenter (closing lines)
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from nba_engine.data.espn import BASE
from nba_engine.data.teams import BY_ESPN

log = logging.getLogger(__name__)

FIRST_DAY = date(2014, 10, 1)  # 2014-15 season (ESPN season year 2015)
SEASON_TYPES = {2: "regular", 3: "playoffs", 5: "play_in"}


def _read_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _write_gz(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    tmp.replace(path)


class EspnBackfill:
    def __init__(self, data_dir: Path | str, rps: float = 4.0, concurrency: int = 4):
        self.data_dir = Path(data_dir)
        self.cache = self.data_dir / "cache" / "espn"
        self.out = self.data_dir / "history"
        self.rps = rps
        self.sem = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None
        self._next_slot = 0.0
        self._slot_lock = asyncio.Lock()

    # ---- fetching ------------------------------------------------------------------------
    async def _pace(self) -> None:
        async with self._slot_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + 1.0 / self.rps
        if wait > 0:
            await asyncio.sleep(wait)

    async def _get(self, url: str, params: dict) -> dict | None:
        assert self._client is not None
        for attempt in range(5):
            async with self.sem:
                await self._pace()
                try:
                    r = await self._client.get(url, params=params)
                except httpx.HTTPError as e:
                    log.debug("espn %s %s error %s", url, params, e)
                    r = None
            if r is not None and r.status_code == 200:
                return r.json()
            if r is not None and r.status_code == 404:
                return None
            await asyncio.sleep(min(60, 2**attempt) * (0.5 + random.random()))
        log.warning("espn giving up on %s %s", url, params)
        return None

    async def scoreboard(self, day: date, refresh: bool = False) -> dict | None:
        path = self.cache / "scoreboard" / f"{day:%Y%m%d}.json.gz"
        if path.exists() and not refresh:
            return _read_gz(path)
        data = await self._get(f"{BASE}/scoreboard", {"dates": f"{day:%Y%m%d}"})
        if data is not None:
            _write_gz(path, data)
        return data

    async def summary(self, game_id: str) -> dict | None:
        path = self.cache / "summary" / f"{game_id}.json.gz"
        if path.exists():
            return _read_gz(path)
        data = await self._get(f"{BASE}/summary", {"event": game_id})
        if data is not None:
            _write_gz(path, data)
        return data

    async def run(self, start: date = FIRST_DAY, end: date | None = None) -> None:
        end = end or datetime.now(UTC).date()
        days = [start + timedelta(d) for d in range((end - start).days + 1)]
        recent = end - timedelta(days=3)
        async with httpx.AsyncClient(timeout=30, headers={"Accept": "application/json"}) as c:
            self._client = c
            log.info("fetching %d scoreboards", len(days))
            boards = await asyncio.gather(*(self.scoreboard(d, refresh=d >= recent)
                                            for d in days))
            game_ids = []
            for b in boards:
                for g in parse_scoreboard_games(b or {}):
                    if g["completed"]:
                        game_ids.append(g["game_id"])
            todo = [g for g in game_ids
                    if not (self.cache / "summary" / f"{g}.json.gz").exists()]
            log.info("%d completed games, %d summaries to fetch", len(game_ids), len(todo))
            done = 0
            for i in range(0, len(todo), 200):
                await asyncio.gather(*(self.summary(g) for g in todo[i:i + 200]))
                done += len(todo[i:i + 200])
                log.info("summaries %d/%d", done, len(todo))
        self._client = None

    # ---- table building ------------------------------------------------------------------
    def build_tables(self) -> dict[str, pd.DataFrame]:
        games: dict[str, dict] = {}
        for path in sorted((self.cache / "scoreboard").glob("*.json.gz")):
            for g in parse_scoreboard_games(_read_gz(path)):
                games[g["game_id"]] = g
        players, odds = [], []
        for gid, g in games.items():
            path = self.cache / "summary" / f"{gid}.json.gz"
            if not g["completed"] or not path.exists():
                continue
            s = _read_gz(path)
            players += parse_player_box(gid, s)
            odds += parse_odds(gid, s)
        tables = {
            "games": pd.DataFrame(list(games.values())).sort_values("tip_utc"),
            "player_box": pd.DataFrame(players),
            "odds": pd.DataFrame(odds),
        }
        self.out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_parquet(self.out / f"{name}.parquet", index=False)
            log.info("wrote %s: %d rows", name, len(df))
        return tables


# ---- parsers (pure functions, unit-tested) -------------------------------------------------
def parse_scoreboard_games(data: dict) -> list[dict]:
    out = []
    for ev in data.get("events", []):
        stype = (ev.get("season") or {}).get("type")
        if stype not in SEASON_TYPES:
            continue
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if set(teams) != {"home", "away"}:
            continue
        h = BY_ESPN.get(teams["home"]["team"].get("abbreviation", ""))
        a = BY_ESPN.get(teams["away"]["team"].get("abbreviation", ""))
        if not h or not a:  # All-Star, international exhibition teams
            continue
        status = (ev.get("status") or {}).get("type") or {}
        addr = (comp.get("venue") or {}).get("address") or {}
        out.append({
            "game_id": str(ev["id"]),
            "season": int((ev.get("season") or {}).get("year") or 0),
            "season_type": SEASON_TYPES[stype],
            "tip_utc": pd.Timestamp(ev["date"]).tz_convert("UTC")
            if pd.Timestamp(ev["date"]).tzinfo else pd.Timestamp(ev["date"], tz="UTC"),
            "home": h.code,
            "away": a.code,
            "home_score": int(teams["home"].get("score") or 0),
            "away_score": int(teams["away"].get("score") or 0),
            "completed": bool(status.get("completed")),
            "neutral": bool(comp.get("neutralSite")),
            "venue_city": addr.get("city"),
        })
    return out


def _minutes(v) -> float:
    if v in (None, "", "--"):
        return 0.0
    try:
        if ":" in str(v):
            m, s = str(v).split(":")
            return int(m) + int(s) / 60
        return float(v)
    except ValueError:
        return 0.0


def _int(v) -> int:
    try:
        return int(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return 0


def parse_player_box(game_id: str, summary: dict) -> list[dict]:
    rows = []
    for team_block in (summary.get("boxscore") or {}).get("players", []) or []:
        team = BY_ESPN.get((team_block.get("team") or {}).get("abbreviation", ""))
        if not team:
            continue
        for stat in team_block.get("statistics", []) or []:
            keys = stat.get("keys") or []
            for a in stat.get("athletes", []) or []:
                vals = dict(zip(keys, a.get("stats") or [], strict=False))
                ath = a.get("athlete") or {}
                fga = str(vals.get("fieldGoalsMade-fieldGoalsAttempted", "0-0")).split("-")
                fta = str(vals.get("freeThrowsMade-freeThrowsAttempted", "0-0")).split("-")
                rows.append({
                    "game_id": game_id,
                    "team": team.code,
                    "player_id": str(ath.get("id")),
                    "player": ath.get("displayName"),
                    "starter": bool(a.get("starter")),
                    "did_not_play": bool(a.get("didNotPlay")),
                    "reason": a.get("reason") or "",
                    "active": bool(a.get("active", True)),
                    "minutes": _minutes(vals.get("minutes")),
                    "points": _int(vals.get("points")),
                    "rebounds": _int(vals.get("rebounds")),
                    "assists": _int(vals.get("assists")),
                    "turnovers": _int(vals.get("turnovers")),
                    "steals": _int(vals.get("steals")),
                    "blocks": _int(vals.get("blocks")),
                    "fga": _int(fga[1]) if len(fga) == 2 else 0,
                    "fta": _int(fta[1]) if len(fta) == 2 else 0,
                    "plus_minus": _int(vals.get("plusMinus")),
                })
    return rows


def parse_odds(game_id: str, summary: dict) -> list[dict]:
    """ESPN pickcenter. `spread` is the HOME team's line (negative = home favored)."""
    rows = []
    for p in summary.get("pickcenter", []) or []:
        home, away = p.get("homeTeamOdds") or {}, p.get("awayTeamOdds") or {}
        rows.append({
            "game_id": game_id,
            "provider": (p.get("provider") or {}).get("name"),
            "priority": (p.get("provider") or {}).get("priority"),
            "spread": p.get("spread"),
            "over_under": p.get("overUnder"),
            "home_ml": home.get("moneyLine"),
            "away_ml": away.get("moneyLine"),
            "home_win_pct": home.get("winPercentage"),
            "away_win_pct": away.get("winPercentage"),
        })
    return rows
