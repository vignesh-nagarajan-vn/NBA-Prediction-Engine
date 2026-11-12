"""Rest-of-season Monte Carlo: standings, play-in, playoffs and title odds.

Each remaining game has a model margin mu (from the pregame model with ratings frozen at
the cutoff). Every simulated season also draws a persistent strength shock per team,
delta_t ~ N(0, tau^2), because a team's true quality is uncertain and errors are correlated
across its games: game margin = mu + delta_home - delta_away + noise, noise ~ N(0, sigma^2).

Playoff matchups use team power ratings fit to the remaining-game mu values
(mu ~ hca * court + s_home - s_away), plus the same per-simulation shocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_engine.data.teams import CONFERENCE, TEAMS

CODES = [t.code for t in TEAMS]
IDX = {c: i for i, c in enumerate(CODES)}


@dataclass
class SimConfig:
    n_sims: int = 20000
    tau: float = 2.5  # sd of season-long team strength uncertainty (points)
    sigma: float = 12.5  # single-game margin noise
    playoff_hca: float = 2.5
    seed: int = 7


def team_power(remaining: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Least-squares team powers from predicted margins: mu = hca*court + s_h - s_a."""
    n = len(CODES)
    X = np.zeros((len(remaining), n + 1))
    for k, r in enumerate(remaining.itertuples(index=False)):
        X[k, IDX[r.home]] = 1
        X[k, IDX[r.away]] = -1
        X[k, n] = 0.0 if r.neutral else 1.0
    reg = np.full(n + 1, 1e-3)
    sol = np.linalg.solve(X.T @ X + np.diag(reg), X.T @ remaining["mu"].to_numpy(float))
    s = sol[:n] - sol[:n].mean()
    return s, float(sol[n])


def _series(strength_hi: np.ndarray, strength_lo: np.ndarray, rng, cfg: SimConfig,
            best_of: int = 7) -> np.ndarray:
    """Vectorized best-of-N series; higher seed has home court in games 1,2,5,7.
    Returns a boolean array: True where the higher seed wins."""
    home_pattern = [1, 1, 0, 0, 1, 0, 1][:best_of]
    need = best_of // 2 + 1
    wins_hi = np.zeros(len(strength_hi), dtype=int)
    wins_lo = np.zeros(len(strength_hi), dtype=int)
    for home_hi in home_pattern:
        hca = cfg.playoff_hca if home_hi else -cfg.playoff_hca
        p = norm.cdf((strength_hi - strength_lo + hca) / cfg.sigma)
        live = (wins_hi < need) & (wins_lo < need)
        won = rng.random(len(p)) < p
        wins_hi += live & won
        wins_lo += live & ~won
    return wins_hi >= need


def _single(a: np.ndarray, b: np.ndarray, rng, cfg: SimConfig) -> np.ndarray:
    """One game, `a` at home. True where `a` wins."""
    p = norm.cdf((a - b + cfg.playoff_hca) / cfg.sigma)
    return rng.random(len(p)) < p


def simulate_season(remaining: pd.DataFrame, current_wins: dict[str, int],
                    current_losses: dict[str, int], cfg: SimConfig | None = None
                    ) -> pd.DataFrame:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    N, T = cfg.n_sims, len(CODES)
    delta = rng.normal(0, cfg.tau, size=(N, T))
    h = remaining["home"].map(IDX).to_numpy()
    a = remaining["away"].map(IDX).to_numpy()
    mu = remaining["mu"].to_numpy(float)

    wins = np.tile(np.array([current_wins.get(c, 0) for c in CODES], float), (N, 1))
    # Simulate in chunks of games to bound memory.
    for lo in range(0, len(mu), 200):
        sl = slice(lo, lo + 200)
        m = mu[sl][None, :] + delta[:, h[sl]] - delta[:, a[sl]]
        home_win = rng.random(m.shape) < norm.cdf(m / cfg.sigma)
        np.add.at(wins.T, h[sl], home_win.T)
        np.add.at(wins.T, a[sl], (~home_win).T)
    total_games = np.array([current_wins.get(c, 0) + current_losses.get(c, 0) for c in CODES])
    remaining_games = np.bincount(h, minlength=T) + np.bincount(a, minlength=T)

    power, _ = team_power(remaining)
    strength = power[None, :] + delta  # (N, T)

    keys = ["seed1", "top6", "playin", "playoffs", "conf_semis", "conf_finals", "finals",
            "title"]
    counts = {k: np.zeros(T, dtype=np.int64) for k in keys}

    def tally(key: str, arr: np.ndarray) -> None:
        counts[key] += np.bincount(arr.ravel(), minlength=T)
    tiebreak = rng.random((N, T)) * 1e-3
    finalists = {}
    for conf in ("East", "West"):
        members = np.array([IDX[c] for c in CODES if CONFERENCE[c] == conf])
        order = np.argsort(-(wins[:, members] + tiebreak[:, members]), axis=1)
        seeds = members[order]  # (N, 15) team index by seed
        rows = np.arange(N)

        def st(idx, rows=rows):
            return strength[rows, idx]

        # Play-in: 7v8 (winner = 7 seed), 9v10, loser(7v8) v winner(9v10) -> 8 seed.
        s7, s8, s9, s10 = seeds[:, 6], seeds[:, 7], seeds[:, 8], seeds[:, 9]
        w78 = _single(st(s7), st(s8), rng, cfg)
        seed7 = np.where(w78, s7, s8)
        loser78 = np.where(w78, s8, s7)
        w910 = _single(st(s9), st(s10), rng, cfg)
        win910 = np.where(w910, s9, s10)
        w_last = _single(st(loser78), st(win910), rng, cfg)
        seed8 = np.where(w_last, loser78, win910)
        bracket = np.column_stack([seeds[:, 0], seed8, seeds[:, 3], seeds[:, 4],
                                   seeds[:, 1], seed7, seeds[:, 2], seeds[:, 5]])
        tally("top6", seeds[:, :6])
        tally("playin", seeds[:, 6:10])
        tally("seed1", seeds[:, 0])
        tally("playoffs", bracket)
        # Rounds: pairs (0,1),(2,3),(4,5),(6,7) -> winners; seed order decides home court.
        seed_rank = np.empty((N, T), dtype=int)
        seed_rank[rows[:, None], seeds] = np.arange(15)[None, :]
        seed_rank[rows, seed7] = 6
        seed_rank[rows, seed8] = 7
        alive = bracket
        for rnd in ("conf_semis", "conf_finals", "finals"):
            nxt = []
            for j in range(0, alive.shape[1], 2):
                x, y = alive[:, j], alive[:, j + 1]
                x_hi = seed_rank[rows, x] < seed_rank[rows, y]
                hi = np.where(x_hi, x, y)
                lo_ = np.where(x_hi, y, x)
                hi_wins = _series(st(hi), st(lo_), rng, cfg)
                nxt.append(np.where(hi_wins, hi, lo_))
            alive = np.column_stack(nxt)
            tally(rnd, alive)
        finalists[conf] = alive[:, 0]

    # NBA Finals: better regular-season record has home court.
    e, w = finalists["East"], finalists["West"]
    rows = np.arange(N)
    e_hi = wins[rows, e] + tiebreak[rows, e] > wins[rows, w] + tiebreak[rows, w]
    hi = np.where(e_hi, e, w)
    lo_ = np.where(e_hi, w, e)
    champ = np.where(_series(strength[rows, hi], strength[rows, lo_], rng, cfg), hi, lo_)
    tally("title", champ)

    out = []
    for c in CODES:
        i = IDX[c]
        wv = wins[:, i]
        out.append({
            "team": c, "conference": CONFERENCE[c],
            "wins_now": current_wins.get(c, 0), "losses_now": current_losses.get(c, 0),
            "games_left": int(remaining_games[i]),
            "proj_wins": float(wv.mean()),
            "proj_losses": float(total_games[i] + remaining_games[i] - wv.mean()),
            "wins_p10": float(np.percentile(wv, 10)), "wins_p90": float(np.percentile(wv, 90)),
            "power": float(power[i]),
            **{f"p_{k}": counts[k][i] / N for k in keys},
        })
    return pd.DataFrame(out).sort_values(["conference", "proj_wins"], ascending=[True, False])
