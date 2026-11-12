"""Play-by-play extraction and lineup stints from cached ESPN game summaries.

plays   one row per ESPN play: period, game clock, score, type, team, participants
stints  one row per stretch of game with the same ten players on the floor: duration and
        points scored by each side. Used for stint-level player ratings (who was actually on
        the court together), which separate teammates far better than game-level minutes.

Lineup reconstruction: period 1 starts with the box-score starters. ESPN does not log
substitutions made between periods, so each later period's starting five is inferred the
standard way: a player is on the floor at the period start if his first event of the period
is anything other than being subbed in; players still short are carried over from the
previous period's closing lineup if they were never subbed in or out. Stints whose lineups
cannot be resolved to exactly five per side are flagged invalid and dropped from ratings.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nba_engine.data.teams import BY_ESPN

log = logging.getLogger(__name__)

SUB = "Substitution"


def _clock_s(v: str | None) -> float:
    if not v:
        return 0.0
    try:
        if ":" in v:
            m, s = v.split(":")
            return int(m) * 60 + float(s)
        return float(v)
    except ValueError:
        return 0.0


def period_len(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


def elapsed_s(period: int, clock: float) -> float:
    before = sum(period_len(p) for p in range(1, period))
    return before + period_len(period) - clock


def extract_plays(game_id: str, summary: dict) -> list[dict]:
    comps = ((summary.get("header") or {}).get("competitions") or [{}])[0].get("competitors", [])
    team_of = {}
    for c in comps:
        t = BY_ESPN.get((c.get("team") or {}).get("abbreviation", ""))
        if t:
            team_of[str(c.get("id"))] = t.code
    rows = []
    for p in summary.get("plays", []) or []:
        period = int((p.get("period") or {}).get("number") or 0)
        clock = _clock_s((p.get("clock") or {}).get("displayValue"))
        parts = [str((a.get("athlete") or {}).get("id")) for a in p.get("participants", []) or []]
        rows.append({
            "game_id": game_id,
            "seq": int(p.get("sequenceNumber") or 0),
            "period": period,
            "clock_s": clock,
            "elapsed_s": elapsed_s(period, clock) if period else 0.0,
            "home_score": int(p.get("homeScore") or 0),
            "away_score": int(p.get("awayScore") or 0),
            "type": str((p.get("type") or {}).get("text") or "").replace("\n", " "),
            "scoring": bool(p.get("scoringPlay")),
            "score_value": int(p.get("scoreValue") or 0),
            "team": team_of.get(str((p.get("team") or {}).get("id"))),
            "p1": parts[0] if len(parts) > 0 else None,
            "p2": parts[1] if len(parts) > 1 else None,
            "p3": parts[2] if len(parts) > 2 else None,
            "wallclock": p.get("wallclock"),
        })
    return rows


def build_plays_table(cache_dir: Path, game_ids: list[str], out_path: Path) -> pd.DataFrame:
    frames, batch = [], []
    for i, gid in enumerate(game_ids, 1):
        path = cache_dir / f"{gid}.json.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            batch += extract_plays(gid, json.load(f))
        if len(batch) > 500_000:
            frames.append(pd.DataFrame(batch))
            batch = []
        if i % 2000 == 0:
            log.info("plays: %d/%d games", i, len(game_ids))
    if batch:
        frames.append(pd.DataFrame(batch))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["game_id", "period", "elapsed_s", "seq"], kind="stable")
    df.to_parquet(out_path, index=False)
    log.info("wrote %s: %d plays", out_path, len(df))
    return df


# ---- stints ------------------------------------------------------------------------------
def _period_starters(pp: pd.DataFrame, team: str, roster: set[str], prev_end: list[str]
                     ) -> list[str] | None:
    first_event: dict[str, str] = {}
    subbed: set[str] = set()
    for r in pp.itertuples(index=False):
        if r.type == SUB and r.team == team:
            ins, outs = r.p1, r.p2
            if ins in roster:
                first_event.setdefault(ins, "in")
                subbed.add(ins)
            if outs in roster:
                first_event.setdefault(outs, "out")
                subbed.add(outs)
            continue
        for pid in (r.p1, r.p2, r.p3):
            if pid in roster:
                first_event.setdefault(pid, "play")
    on = [p for p, e in first_event.items() if e in ("out", "play")]
    if len(on) < 5:
        for p in prev_end:
            if p not in on and p not in subbed and len(on) < 5:
                on.append(p)
    return on if len(on) == 5 else None


def game_stints(game_id: str, plays: pd.DataFrame, box: pd.DataFrame, home: str, away: str
                ) -> list[dict]:
    rosters = {t: set(box.loc[box["team"] == t, "player_id"]) for t in (home, away)}
    starters = {t: list(box.loc[(box["team"] == t) & box["starter"], "player_id"])
                for t in (home, away)}
    if any(len(starters[t]) != 5 for t in (home, away)):
        return []
    out = []
    prev_end = starters
    for period, pp in plays.groupby("period", sort=True):
        if period < 1:
            continue
        if period == 1:
            on = {t: list(starters[t]) for t in (home, away)}
        else:
            on = {}
            for t in (home, away):
                s = _period_starters(pp, t, rosters[t], prev_end[t])
                on[t] = s if s is not None else []
        valid = all(len(on[t]) == 5 for t in (home, away))
        start_t = elapsed_s(period, period_len(period))
        first = pp.iloc[0]
        start_h, start_a = first.home_score, first.away_score
        # Score just before the first play of the period: subtract that play's points.
        if first.scoring:
            if first.team == home:
                start_h -= first.score_value
            elif first.team == away:
                start_a -= first.score_value
        cur_h, cur_a = start_h, start_a
        for r in pp.itertuples(index=False):
            if r.type == SUB and r.team in (home, away):
                ins, outs = r.p1, r.p2
                if r.elapsed_s > start_t or r.home_score != start_h or r.away_score != start_a:
                    out.append(_stint(game_id, period, start_t, r.elapsed_s, start_h, start_a,
                                      r.home_score, r.away_score, on, home, away, valid))
                    start_t, start_h, start_a = r.elapsed_s, r.home_score, r.away_score
                lineup = on[r.team]
                if outs in lineup and ins not in lineup:
                    lineup[lineup.index(outs)] = ins
                else:
                    valid = False  # inconsistent sub: lineup unknown until next period
            cur_h, cur_a = r.home_score, r.away_score
        end_t = elapsed_s(period, 0.0)
        out.append(_stint(game_id, period, start_t, end_t, start_h, start_a, cur_h, cur_a,
                          on, home, away, valid))
        prev_end = {t: list(on[t]) for t in (home, away)}
    return out


def _stint(gid, period, t0, t1, h0, a0, h1, a1, on, home, away, valid) -> dict:
    ok = valid and len(on[home]) == 5 and len(on[away]) == 5
    row = {"game_id": gid, "period": period, "start_s": t0, "end_s": t1,
           "dur_s": max(0.0, t1 - t0), "home_pts": h1 - h0, "away_pts": a1 - a0,
           "start_margin": h0 - a0, "valid": ok}
    for side, t in (("h", home), ("a", away)):
        ids = on[t] if ok else [None] * 5
        for k in range(5):
            row[f"{side}{k}"] = ids[k]
    return row


def build_stints(plays: pd.DataFrame, box: pd.DataFrame, games: pd.DataFrame,
                 out_path: Path) -> pd.DataFrame:
    meta = games.set_index("game_id")[["home", "away"]]
    box_by = {g: df for g, df in box.groupby("game_id", sort=False)}
    rows = []
    for i, (gid, gp) in enumerate(plays.groupby("game_id", sort=False), 1):
        if gid not in meta.index or gid not in box_by:
            continue
        rows += game_stints(gid, gp, box_by[gid], meta.at[gid, "home"], meta.at[gid, "away"])
        if i % 2000 == 0:
            log.info("stints: %d games", i)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    v = df["valid"]
    log.info("wrote %s: %d stints, %.1f%% valid, %.1f%% of game time valid", out_path,
             len(df), 100 * v.mean(), 100 * df.loc[v, "dur_s"].sum() / df["dur_s"].sum())
    return df


def stint_quality(stints: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per game: share of game time with valid lineups and whether stint points add up."""
    g = stints.groupby("game_id").agg(valid_time=("dur_s", lambda s: np.nan),
                                      total_time=("dur_s", "sum"),
                                      home_pts=("home_pts", "sum"), away_pts=("away_pts", "sum"))
    vt = stints[stints["valid"]].groupby("game_id")["dur_s"].sum()
    g["valid_time"] = vt.reindex(g.index).fillna(0) / g["total_time"]
    m = games.set_index("game_id")
    g["points_match"] = ((g["home_pts"] == m["home_score"].reindex(g.index))
                         & (g["away_pts"] == m["away_score"].reindex(g.index)))
    return g.reset_index()
