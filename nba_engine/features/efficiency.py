"""Pace-adjusted team offense/defense ratings, computed online.

Possessions per team-game are estimated from the box score:
    poss = FGA - OREB + TOV + 0.44 * FTA     (averaged over both teams)
Two decay-weighted ridge regressions are updated day by day:
    offense:  ORtg(off vs def) = mu + o[off] - d[def] + hca * side        (side +-0.5)
    pace:     poss(game)       = mu_p + p[home] + p[away]
Before each game they give expected possessions and each side's points per 100
possessions, hence an efficiency-based margin and total.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_engine.data.teams import TEAMS

TEAM_INDEX = {t.code: i for i, t in enumerate(TEAMS)}
N_TEAMS = len(TEAMS)


def team_game_totals(box: pd.DataFrame) -> pd.DataFrame:
    cols = ["fga", "fta", "oreb", "turnovers", "points", "minutes"]
    t = box.groupby(["game_id", "team"], sort=False)[cols].sum().reset_index()
    t["poss"] = t["fga"] - t["oreb"] + t["turnovers"] + 0.44 * t["fta"]
    return t


class DecayedRidge:
    """Recursive exponentially-weighted ridge regression with unpenalized columns."""

    def __init__(self, n: int, ridge: float, free: list[int], half_life_days: float):
        self.A = np.zeros((n, n))
        self.b = np.zeros(n)
        self.reg = np.full(n, ridge)
        self.reg[free] = 1e-6
        self.hl = half_life_days
        self.theta = np.zeros(n)

    def decay(self, days: float) -> None:
        f = 0.5 ** (days / self.hl)
        self.A *= f
        self.b *= f

    def solve(self) -> np.ndarray:
        if self.b.any():
            self.theta = np.linalg.solve(self.A + np.diag(self.reg), self.b)
        return self.theta

    def add(self, x: np.ndarray, y: float, w: float = 1.0) -> None:
        self.A += w * np.outer(x, x)
        self.b += w * x * y


def efficiency_ratings(games: pd.DataFrame, box: pd.DataFrame, half_life_days: float = 75.0,
                       ridge: float = 3.0) -> pd.DataFrame:
    g = games.sort_values("tip_utc").reset_index(drop=True)
    tot = team_game_totals(box).set_index(["game_id", "team"])
    n_off = 2 * N_TEAMS + 2  # o, d, mu, hca
    off = DecayedRidge(n_off, ridge, free=[2 * N_TEAMS, 2 * N_TEAMS + 1],
                       half_life_days=half_life_days)
    pace = DecayedRidge(N_TEAMS + 1, ridge, free=[N_TEAMS], half_life_days=half_life_days)
    # Sensible starting point before any data: league-average offense and pace.
    off.theta[2 * N_TEAMS] = 110.0
    pace.theta[N_TEAMS] = 98.0
    day_of = g["tip_utc"].dt.tz_convert("America/New_York").dt.normalize()
    prev_day = None
    rows = []

    def off_row(o_team: str, d_team: str, side: float) -> np.ndarray:
        x = np.zeros(n_off)
        x[TEAM_INDEX[o_team]] = 1.0
        x[N_TEAMS + TEAM_INDEX[d_team]] = -1.0
        x[2 * N_TEAMS] = 1.0
        x[2 * N_TEAMS + 1] = side
        return x

    for i, r in enumerate(g.itertuples(index=False)):
        day = day_of.iloc[i]
        if day != prev_day:
            if prev_day is not None:
                d = (day - prev_day).days
                off.decay(d)
                pace.decay(d)
                off.solve()
                pace.solve()
            prev_day = day
        side = 0.0 if r.neutral else 0.5
        xh, xa = off_row(r.home, r.away, side), off_row(r.away, r.home, -side)
        xp = np.zeros(N_TEAMS + 1)
        xp[TEAM_INDEX[r.home]] = 1.0
        xp[TEAM_INDEX[r.away]] = 1.0
        xp[N_TEAMS] = 1.0
        ortg_h, ortg_a = float(xh @ off.theta), float(xa @ off.theta)
        poss = float(xp @ pace.theta)
        rows.append({"game_id": r.game_id, "ortg_home": ortg_h, "ortg_away": ortg_a,
                     "pace_pred": poss,
                     "eff_margin_pred": poss * (ortg_h - ortg_a) / 100,
                     "total_pred": poss * (ortg_h + ortg_a) / 100,
                     "off_home": off.theta[TEAM_INDEX[r.home]],
                     "def_home": off.theta[N_TEAMS + TEAM_INDEX[r.home]],
                     "off_away": off.theta[TEAM_INDEX[r.away]],
                     "def_away": off.theta[N_TEAMS + TEAM_INDEX[r.away]]})
        if not getattr(r, "completed", True):
            continue
        try:
            th, ta = tot.loc[(r.game_id, r.home)], tot.loc[(r.game_id, r.away)]
        except KeyError:
            continue
        p = (th["poss"] + ta["poss"]) / 2
        if p < 60:  # incomplete box score
            continue
        off.add(xh, 100 * r.home_score / p)
        off.add(xa, 100 * r.away_score / p)
        # Pace per 48 minutes so overtime games don't inflate it.
        mins = th["minutes"] if th["minutes"] > 200 else 240.0
        pace.add(xp, p * 240.0 / mins)
    return pd.DataFrame(rows)
