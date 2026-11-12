"""Forecasts: upcoming games, rest-of-season projections, and projection backtests.

All three share one recipe so they can be validated together:
  1. take history up to a cutoff,
  2. project lineups for later games (rosters at the cutoff, minus injured players),
  3. compute features with ratings frozen at the cutoff,
  4. predict margins with a model trained only on games before the cutoff.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from nba_engine.history.espn_backfill import EspnBackfill, parse_scoreboard_games
from nba_engine.models.pregame import EnsembleModel, build_dataset
from nba_engine.predict.lineups import current_rosters, projected_box
from nba_engine.predict.season import SimConfig, simulate_season

log = logging.getLogger(__name__)

KALSHI_PUBLIC = "https://external-api.kalshi.com/trade-api/v2"


def fetch_schedule(data_dir: Path | str, start: date, end: date) -> pd.DataFrame:
    """ESPN games between two dates (inclusive). Dates from today on are always refetched;
    earlier dates come from the cache."""
    eb = EspnBackfill(data_dir)
    today = date.today()

    async def go():
        async with httpx.AsyncClient(timeout=30, headers={"Accept": "application/json"}) as c:
            eb._client = c
            days = [start + timedelta(d) for d in range((end - start).days + 1)]
            return await asyncio.gather(*(eb.scoreboard(d, refresh=d >= today - timedelta(1))
                                          for d in days))

    boards = asyncio.run(go())
    rows = [g for b in boards for g in parse_scoreboard_games(b or {})]
    df = pd.DataFrame(rows).drop_duplicates("game_id")
    df["tip_utc"] = pd.to_datetime(df["tip_utc"], utc=True)
    return df.sort_values("tip_utc").reset_index(drop=True)


def truncate_history(hist: dict, cutoff: pd.Timestamp) -> dict:
    g = hist["games"].copy()
    g["tip_utc"] = pd.to_datetime(g["tip_utc"], utc=True)
    g = g[g["completed"] & (g["tip_utc"] < cutoff)]
    out = {**hist, "games": g,
           "player_box": hist["player_box"][hist["player_box"]["game_id"].isin(g["game_id"])]}
    st = hist.get("stints")
    if st is not None and len(st):
        out["stints"] = st[st["game_id"].isin(g["game_id"])]
    return out


def forecast_games(hist: dict, future: pd.DataFrame, cutoff: pd.Timestamp, params: dict,
                   injuries: pd.DataFrame | None = None, train_from: int = 2016
                   ) -> tuple[pd.DataFrame, EnsembleModel]:
    """Margins and win probabilities for `future` games using only data before `cutoff`."""
    h = truncate_history(hist, cutoff)
    fut = future.copy()
    fut["completed"] = False
    fut["home_score"] = 0
    fut["away_score"] = 0
    rosters = current_rosters(h["games"], h["player_box"], cutoff)
    box = projected_box(fut, rosters, injuries, cutoff)
    df = build_dataset(h, player_kwargs=params["player"], team_kwargs=params["team"],
                       eff_kwargs=params["eff"], upcoming=fut, upcoming_box=box)
    done = df[df["completed"] & (df["season"] >= train_from)]
    model = EnsembleModel(gbm_weight=params["gbm_weight"]).fit(done)
    up = df[~df["completed"]].copy()
    up["mu"] = model.predict_margin(up)
    up["p_home"] = model.predict_proba(up)
    return up, model


def kalshi_open_prices() -> pd.DataFrame:
    """Current Kalshi mid prices for open NBA game markets (public, no key)."""
    rows = []
    with httpx.Client(timeout=20) as c:
        cursor = None
        while True:
            params = {"series_ticker": "KXNBAGAME", "status": "open", "limit": 200,
                      "with_nested_markets": "true"}
            if cursor:
                params["cursor"] = cursor
            d = c.get(f"{KALSHI_PUBLIC}/events", params=params).json()
            for ev in d.get("events", []):
                for m in ev.get("markets", []):
                    try:
                        bid, ask = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if ask > bid > 0:
                        rows.append({"event_ticker": ev["event_ticker"],
                                     "team": m["ticker"].rsplit("-", 1)[1],
                                     "mid": (bid + ask) / 2})
            cursor = d.get("cursor")
            if not cursor:
                break
    return pd.DataFrame(rows)


def season_records(games: pd.DataFrame, season: int, cutoff: pd.Timestamp
                   ) -> tuple[dict, dict]:
    g = games[(games["season"] == season) & (games["season_type"] == "regular")
              & games["completed"] & (pd.to_datetime(games["tip_utc"], utc=True) < cutoff)]
    w: dict[str, int] = {}
    lo: dict[str, int] = {}
    for r in g.itertuples(index=False):
        win, lose = (r.home, r.away) if r.home_score > r.away_score else (r.away, r.home)
        w[win] = w.get(win, 0) + 1
        lo[lose] = lo.get(lose, 0) + 1
    return w, lo


def project_season(hist: dict, schedule: pd.DataFrame, season: int, cutoff: pd.Timestamp,
                   params: dict, injuries: pd.DataFrame | None = None,
                   cfg: SimConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    remaining = schedule[(schedule["season"] == season)
                         & (schedule["season_type"] == "regular")
                         & (schedule["tip_utc"] >= cutoff)].copy()
    games_pred, model = forecast_games(hist, remaining, cutoff, params, injuries)
    cfg = cfg or SimConfig(sigma=model.sigma_)
    w, lo = season_records(hist["games"], season, cutoff)
    table = simulate_season(games_pred, w, lo, cfg)
    return table, games_pred


def backtest_projection(hist: dict, season: int, cutoff: pd.Timestamp, params: dict,
                        cfg: SimConfig | None = None) -> pd.DataFrame:
    """Project a past season from `cutoff` and compare with what actually happened."""
    g = hist["games"].copy()
    g["tip_utc"] = pd.to_datetime(g["tip_utc"], utc=True)
    sched = g[(g["season"] == season) & (g["season_type"] == "regular")]
    table, _ = project_season(hist, sched, season, cutoff, params, cfg=cfg)
    final_w, final_l = season_records(g, season, pd.Timestamp.max.tz_localize("UTC"))
    now_w, now_l = season_records(g, season, cutoff)
    prev_w, prev_l = season_records(g, season - 1, pd.Timestamp.max.tz_localize("UTC"))
    po = g[(g["season"] == season) & (g["season_type"] == "playoffs")]
    playoff_teams = set(po["home"]) | set(po["away"])
    t = table.copy()
    t["season"] = season
    t["actual_wins"] = t["team"].map(final_w).fillna(0)
    t["actual_games"] = t["team"].map(lambda c: final_w.get(c, 0) + final_l.get(c, 0))
    t["made_playoffs"] = t["team"].isin(playoff_teams).astype(int)
    played = t["team"].map(lambda c: now_w.get(c, 0) + now_l.get(c, 0)).replace(0, np.nan)
    t["naive_pace_wins"] = (t["team"].map(now_w).fillna(0) / played * t["actual_games"]
                            ).fillna(t["actual_games"] / 2)
    prev_games = t["team"].map(lambda c: prev_w.get(c, 0) + prev_l.get(c, 0)).replace(0, np.nan)
    t["naive_last_season_wins"] = (t["team"].map(prev_w).fillna(0) / prev_games
                                   * t["actual_games"]).fillna(t["actual_games"] / 2)
    return t
