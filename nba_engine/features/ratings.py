"""Team strength ratings computed online (each game sees only earlier games).

- Elo (FiveThirtyEight-style NBA Elo with margin-of-victory multiplier, season reversion).
- Decayed ridge margin ratings: margin_home = hca * (1 - neutral) + r_home - r_away, fit by
  exponentially time-weighted ridge regression, updated recursively day by day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_engine.data.teams import TEAMS

TEAM_INDEX = {t.code: i for i, t in enumerate(TEAMS)}


def elo_ratings(games: pd.DataFrame, k: float = 20.0, hca: float = 100.0,
                revert: float = 0.25, mean: float = 1505.0) -> pd.DataFrame:
    """Pre-game Elo for both teams. `games` must be completed games."""
    g = games.sort_values("tip_utc")
    elo = {t.code: mean for t in TEAMS}
    season = None
    rows = []
    for r in g.itertuples(index=False):
        if r.season != season:
            if season is not None:
                elo = {t: (1 - revert) * e + revert * mean for t, e in elo.items()}
            season = r.season
        eh, ea = elo[r.home], elo[r.away]
        rows.append({"game_id": r.game_id, "elo_home": eh, "elo_away": ea})
        adv = 0.0 if r.neutral else hca
        exp_home = 1 / (1 + 10 ** (-(eh + adv - ea) / 400))
        margin = r.home_score - r.away_score
        if margin == 0:
            continue
        win = 1.0 if margin > 0 else 0.0
        winner_diff = (eh + adv - ea) if margin > 0 else (ea - eh - adv)
        mult = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)
        delta = k * mult * (win - exp_home)
        elo[r.home] = eh + delta
        elo[r.away] = ea - delta
    out = pd.DataFrame(rows)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    return out


def margin_ratings(games: pd.DataFrame, half_life_days: float = 60.0,
                   ridge: float = 4.0) -> pd.DataFrame:
    """Pre-game predicted margin from decayed ridge team ratings.

    Recursive weighted least squares: before each new day the accumulated normal equations
    are multiplied by 0.5 ** (days / half_life), so old games fade smoothly. Parameters are
    30 team ratings plus home-court advantage; ridge shrinks team ratings toward 0.
    """
    g = games.sort_values("tip_utc").reset_index(drop=True)
    n = len(TEAMS) + 1  # + hca
    A = np.zeros((n, n))
    b = np.zeros(n)
    reg = np.full(n, ridge)
    reg[-1] = 1e-3
    day_of = g["tip_utc"].dt.tz_convert("America/New_York").dt.normalize()
    prev_day = None
    theta = np.zeros(n)
    rows = []
    for i, r in enumerate(g.itertuples(index=False)):
        day = day_of.iloc[i]
        if day != prev_day:
            if prev_day is not None:
                decay = 0.5 ** ((day - prev_day).days / half_life_days)
                A *= decay
                b *= decay
                theta = np.linalg.solve(A + np.diag(reg), b)
            prev_day = day
        x = np.zeros(n)
        x[TEAM_INDEX[r.home]] = 1.0
        x[TEAM_INDEX[r.away]] = -1.0
        x[-1] = 0.0 if r.neutral else 1.0
        rows.append({"game_id": r.game_id,
                     "rating_home": theta[TEAM_INDEX[r.home]],
                     "rating_away": theta[TEAM_INDEX[r.away]],
                     "rating_hca": theta[-1],
                     "rating_margin_pred": float(x @ theta)})
        # Only completed games carry information; the update applies from the next day.
        if getattr(r, "completed", True):
            A += np.outer(x, x)
            b += x * (r.home_score - r.away_score)
    return pd.DataFrame(rows)
