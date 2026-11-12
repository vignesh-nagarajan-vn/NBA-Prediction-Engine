"""In-game win probability.

Starting point is the classic random-walk model (Stern, 1994): with d = home lead, T seconds
left and pregame expected margin mu, the final margin is roughly
    d + mu * tau + noise * sqrt(tau),   tau = T / 2880,
so P(home wins) = Phi((d + mu * tau) / (sigma * sqrt(tau))).

We fit a logistic model on those same standardized quantities, plus who has the ball and
game-phase interactions, so the shape is learned from 10+ seasons of play-by-play rather
than assumed. Training uses only seasons before the evaluation seasons, and mu comes from
the walk-forward pregame model, so nothing peeks at the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_engine.models.metrics import fit_logistic, sigmoid

SIGMA = 12.0
REG_SECONDS = 2880.0


def possession_proxy(plays: pd.DataFrame) -> np.ndarray:
    """+1 home ball, -1 away ball, 0 unknown, inferred from the previous event:
    defensive/offensive rebound -> rebounder's team; turnover -> other team; made field goal
    or the final free throw of a trip -> other team."""
    t = plays["type"].str.lower()
    team_home = (plays["team"] == plays["home"]).astype(int) * 2 - 1
    team_home = team_home.where(plays["team"].notna(), 0)
    poss = pd.Series(0, index=plays.index, dtype=float)
    reb = t.str.contains("rebound")
    tov = t.str.contains("turnover")
    ft = t.str.contains("free throw")
    last_ft = ft & (t.str.contains("1 of 1") | t.str.contains("2 of 2")
                    | t.str.contains("3 of 3"))
    made_fg = plays["scoring"] & ~ft
    poss[reb] = team_home[reb]
    poss[tov] = -team_home[tov]
    poss[made_fg | (last_ft & plays["scoring"])] = -team_home[made_fg | (last_ft
                                                                         & plays["scoring"])]
    # Carry the last known possession forward within each game period.
    poss = poss.replace(0, np.nan)
    poss = poss.groupby([plays["game_id"], plays["period"]]).ffill().fillna(0)
    return poss.to_numpy()


def game_states(plays: pd.DataFrame, games: pd.DataFrame, pregame_mu: pd.Series,
                every_s: float = 0.0) -> pd.DataFrame:
    """One row per play (optionally thinned to one per `every_s` seconds of game time)."""
    meta = games.set_index("game_id")
    p = plays[plays["game_id"].isin(meta.index)].copy()
    p["home"] = p["game_id"].map(meta["home"])
    p["away"] = p["game_id"].map(meta["away"])
    p["poss"] = possession_proxy(p)
    reg_left = np.clip(REG_SECONDS - p["elapsed_s"], 0, None)
    ot_left = np.where(p["period"] > 4, p["clock_s"], 0.0)
    p["t_left"] = np.where(p["period"] > 4, ot_left, reg_left)
    p["d"] = p["home_score"] - p["away_score"]
    p["mu"] = p["game_id"].map(pregame_mu)
    final = meta["home_score"] > meta["away_score"]
    p["home_win"] = p["game_id"].map(final).astype(float)
    p["season"] = p["game_id"].map(meta["season"])
    p = p.dropna(subset=["mu", "home_win"])
    p = p[p["period"] >= 1]
    if every_s > 0:
        p["bucket"] = (p["elapsed_s"] // every_s).astype(int)
        p = p.groupby(["game_id", "bucket"], sort=False).tail(1)
    return p.reset_index(drop=True)


def stern_prob(d, t_left, mu, sigma: float = SIGMA) -> np.ndarray:
    tau = np.asarray(t_left, float) / REG_SECONDS
    s = sigma * np.sqrt(tau) + 1e-3
    z = (np.asarray(d, float) + np.asarray(mu, float) * tau) / s
    p = norm.cdf(z)
    # At the horn the leader has won; a tie goes to overtime (about a coin flip plus mu).
    done = tau <= 0
    p = np.where(done & (np.asarray(d) > 0), 1.0, p)
    p = np.where(done & (np.asarray(d) < 0), 0.0, p)
    return np.clip(p, 1e-4, 1 - 1e-4)


PHASES = [(0, 720), (720, 1440), (1440, 2160), (2160, 2580), (2580, 2880.1)]


def _design(df: pd.DataFrame) -> np.ndarray:
    tau = df["t_left"].to_numpy(float) / REG_SECONDS
    s = SIGMA * np.sqrt(tau) + 1.0
    zd = df["d"].to_numpy(float) / s
    zm = df["mu"].to_numpy(float) * tau / s
    zp = df["poss"].to_numpy(float) / s
    elapsed = REG_SECONDS - df["t_left"].to_numpy(float)
    cols = [np.ones(len(df)), zm, zp]
    for lo, hi in PHASES:  # let the weight on the current lead vary by game phase
        cols.append(zd * ((elapsed >= lo) & (elapsed < hi)))
    return np.column_stack(cols)


@dataclass
class InGameModel:
    w: np.ndarray | None = None

    def fit(self, states: pd.DataFrame) -> InGameModel:
        live = states[states["t_left"] > 0]
        self.w = fit_logistic(_design(live), live["home_win"].to_numpy(float), l2=1e-3)
        return self

    def predict(self, states: pd.DataFrame) -> np.ndarray:
        p = sigmoid(_design(states) @ self.w)
        d = states["d"].to_numpy()
        done = states["t_left"].to_numpy() <= 0
        p = np.where(done & (d > 0), 1.0, p)
        p = np.where(done & (d < 0), 0.0, p)
        return np.clip(p, 1e-4, 1 - 1e-4)
