"""Player-impact ratings and lineup-adjusted team strength.

Ratings: game-level adjusted plus-minus. Each game is one regression row:
    margin_home = hca * (1 - neutral) + sum_home(min_p / 48 * r_p) - sum_away(min_p / 48 * r_p)
solved by exponentially time-weighted ridge regression. Ratings shrink toward a BOX-SCORE
PRIOR rather than toward 0: each player's decayed per-48 box line (points, rebounds, assists,
steals, blocks, turnovers, shooting volume, fouls) mapped to a rating by coefficients fit once,
on the warm-up seasons only, by regressing the plus-minus ratings on those box lines. The
prior is itself shrunk toward replacement level for low-minute players. This mostly helps
rookies, new signings and bench players, whom pure plus-minus can barely separate.
Ratings are re-solved every `solve_every_days` using only games before that day.

Pregame strength for a team = sum over AVAILABLE players of expected_minutes / 48 * r_p, with
expected minutes from each player's recent minutes, rescaled to 240 team minutes. Injured and
inactive players drop out, so the strength moves when stars sit.

Availability comes from the box score (played, or DNP for coach's decision = available;
DNP for injury/illness/rest or inactive = unavailable). Inactive lists are public before tip,
so this is close to what is known pregame; late scratches are the main exception.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_NEW_PLAYER_MIN = 12.0
BOX_STATS = ["points", "oreb", "dreb", "assists", "steals", "blocks", "turnovers", "fga",
             "fta", "fg3m", "fouls"]
REPLACEMENT = -2.0  # per-48 rating of a freely available player
PRIOR_MINUTES = 400.0  # minutes of evidence at which the box line gets half weight


def is_available(row) -> bool:
    if row.minutes > 0:
        return True
    return bool(row.did_not_play) and "COACH" in str(row.reason).upper()


class PlayerRatings:
    def __init__(self, half_life_days: float = 365.0, ridge: float = 8.0,
                 solve_every_days: int = 7, ewma_alpha: float = 0.25,
                 box_prior: bool = True, prior_fit_season: int = 2017):
        self.hl = half_life_days
        self.ridge = ridge
        self.solve_every = solve_every_days
        self.alpha = ewma_alpha
        self.box_prior = box_prior
        self.prior_fit_season = prior_fit_season
        self.box_coef_: np.ndarray | None = None

    def run(self, games: pd.DataFrame, box: pd.DataFrame,
            stints: pd.DataFrame | None = None) -> pd.DataFrame:
        """`stints` (optional): lineup stints from play-by-play. When a game has valid
        lineups for at least 90% of its time, each stint is its own regression row
        (players on court, duration, point differential) instead of one row per game."""
        g = games.sort_values("tip_utc").reset_index(drop=True)
        stint_rows: dict[str, np.ndarray] = {}
        if stints is not None and len(stints):
            st = stints[stints["game_id"].isin(set(g["game_id"]))]
            cover = st[st["valid"]].groupby("game_id")["dur_s"].sum() / st.groupby(
                "game_id")["dur_s"].sum()
            good = set(cover[cover >= 0.9].index)
            st = st[st["valid"] & st["game_id"].isin(good) & (st["dur_s"] > 0)]
            cols = ["h0", "h1", "h2", "h3", "h4", "a0", "a1", "a2", "a3", "a4"]
            for gid, df in st.groupby("game_id", sort=False):
                stint_rows[gid] = df[[*cols, "dur_s", "home_pts", "away_pts"]].to_numpy()
        box = box[box["game_id"].isin(set(g["game_id"]))]
        pids = sorted(box["player_id"].unique())
        pidx = {p: i for i, p in enumerate(pids)}
        n = len(pids) + 1  # + hca (last)
        hca = n - 1
        A = np.zeros((n, n))
        b = np.zeros(n)
        last_seen_scale = np.zeros(n)  # time weight of most recent appearance
        theta = np.zeros(n)
        box_S = np.zeros((n, len(BOX_STATS)))  # time-weighted box totals per player
        box_M = np.zeros(n)  # time-weighted minutes per player
        stat_cols = [c for c in BOX_STATS if c in box.columns]
        if len(stat_cols) < len(BOX_STATS):
            self.box_prior = False  # older tables without the full box line
        by_game = {gid: df for gid, df in box.groupby("game_id", sort=False)}
        ewma: dict[str, float] = {}
        t0 = g["tip_utc"].iloc[0]
        last_solve_day = None
        out = []
        day_index = (g["tip_utc"] - t0).dt.days.to_numpy()
        for i, r in enumerate(g.itertuples(index=False)):
            day = int(day_index[i])
            if last_solve_day is None or day - last_solve_day >= self.solve_every:
                if (self.box_prior and self.box_coef_ is None
                        and r.season >= self.prior_fit_season):
                    self._fit_box_coef(theta, box_S, box_M, day)
                prior = self._prior(box_S, box_M, day) if self.box_coef_ is not None else None
                theta = self._solve(A, b, last_seen_scale, day, hca, prior)
                last_solve_day = day
            rows = by_game.get(r.game_id)
            feats = {"game_id": r.game_id}
            team_lists = {}
            for side, team in (("home", r.home), ("away", r.away)):
                tb = rows[rows["team"] == team] if rows is not None else None
                avail, exp = [], []
                n_out_regulars = 0
                if tb is not None:
                    for p in tb.itertuples(index=False):
                        if is_available(p):
                            avail.append(p.player_id)
                            exp.append(ewma.get(p.player_id, DEFAULT_NEW_PLAYER_MIN))
                        elif ewma.get(p.player_id, 0) >= 20:
                            n_out_regulars += 1
                exp_arr = np.array(exp, dtype=float)
                if exp_arr.sum() > 0:
                    exp_arr *= 240.0 / exp_arr.sum()
                rat = np.array([theta[pidx[p]] for p in avail]) if avail else np.zeros(0)
                feats[f"{side}_player_strength"] = float((exp_arr / 48.0 * rat).sum())
                feats[f"{side}_out_regulars"] = n_out_regulars
                feats[f"{side}_n_box"] = 0 if tb is None else len(tb)
                team_lists[side] = tb
            feats["player_strength_diff"] = (feats["home_player_strength"]
                                             - feats["away_player_strength"])
            feats["player_hca"] = float(theta[hca])
            out.append(feats)

            # ---- post-game update with ACTUAL minutes (information for future games only)
            if rows is None or not getattr(r, "completed", True):
                continue
            idx, val = [hca], [0.0 if r.neutral else 1.0]
            for side, sign in (("home", 1.0), ("away", -1.0)):
                tb = team_lists[side]
                for p in tb.itertuples(index=False):
                    if p.minutes > 0:
                        k = pidx[p.player_id]
                        idx.append(k)
                        val.append(sign * p.minutes / 48.0)
                        if self.box_prior:
                            w_now = 2.0 ** (day / self.hl)
                            box_S[k] += w_now * np.array([getattr(p, c) for c in BOX_STATS],
                                                         dtype=float)
                            box_M[k] += w_now * p.minutes
                        prev = ewma.get(p.player_id)
                        ewma[p.player_id] = (p.minutes if prev is None
                                             else (1 - self.alpha) * prev + self.alpha * p.minutes)
            w = 2.0 ** (day / self.hl)  # growing weights == decaying old games
            srows = stint_rows.get(r.game_id)
            if srows is not None:
                hca_on = 0.0 if r.neutral else 1.0
                for row in srows:
                    try:
                        ix = np.array([pidx[q] for q in row[:10]] + [hca])
                    except KeyError:
                        continue
                    f = float(row[10]) / 2880.0
                    v = np.array([f] * 5 + [-f] * 5 + [f * hca_on])
                    A[np.ix_(ix, ix)] += w * np.outer(v, v)
                    b[ix] += w * v * (float(row[11]) - float(row[12]))
                    last_seen_scale[ix] = w
                continue
            ix = np.array(idx)
            v = np.array(val)
            A[np.ix_(ix, ix)] += w * np.outer(v, v)
            b[ix] += w * v * (r.home_score - r.away_score)
            last_seen_scale[ix] = w
        df = pd.DataFrame(out)
        self.theta_, self.player_ids_ = theta, pids
        return df

    def _per48(self, box_S, box_M) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            x = 48.0 * box_S / box_M[:, None]
        return np.nan_to_num(x)

    def _fit_box_coef(self, theta, box_S, box_M, day) -> None:
        """Map per-48 box lines to plus-minus ratings using players with real minutes."""
        scale = 2.0 ** (day / self.hl)
        mins = box_M / scale
        keep = np.where(mins[:-1] >= 500)[0]
        if len(keep) < 100:
            return
        X = np.column_stack([np.ones(len(keep)), self._per48(box_S, box_M)[keep]])
        y = theta[keep]
        w = np.sqrt(mins[keep])
        reg = np.full(X.shape[1], 1.0)
        reg[0] = 0.0
        Xw = X * w[:, None]
        self.box_coef_ = np.linalg.solve(Xw.T @ Xw + np.diag(reg), Xw.T @ (y * w))
        log.info("box prior fit on %d players", len(keep))

    def _prior(self, box_S, box_M, day) -> np.ndarray:
        scale = 2.0 ** (day / self.hl)
        mins = box_M / scale
        X = np.column_stack([np.ones(len(box_M)), self._per48(box_S, box_M)])
        raw = X @ self.box_coef_
        wgt = mins / (mins + PRIOR_MINUTES)
        prior = wgt * raw + (1 - wgt) * REPLACEMENT
        prior[-1] = 0.0  # hca slot
        return prior

    def _solve(self, A, b, last_seen, day, hca, prior=None) -> np.ndarray:
        scale = 2.0 ** (day / self.hl)
        base = np.zeros_like(b) if prior is None else prior.copy()
        base[hca] = 0.0
        if not b.any():
            return base
        # Only players seen in roughly the last 4 half-lives matter; others sit at the prior.
        active = np.where(last_seen >= scale * 2.0 ** -4)[0]
        active = np.union1d(active, [hca])
        sub = A[np.ix_(active, active)].copy()
        reg = np.full(len(active), self.ridge * scale)
        reg[active == hca] = 1e-3 * scale
        sub[np.diag_indices_from(sub)] += reg
        theta = base
        # Ridge toward the prior: (A + L) t = b + L * prior
        theta[active] = np.linalg.solve(sub, b[active] + reg * base[active])
        return theta

    def top_players(self, box: pd.DataFrame, k: int = 15) -> pd.DataFrame:
        names = box.drop_duplicates("player_id").set_index("player_id")["player"]
        s = pd.Series(self.theta_[:-1], index=self.player_ids_).sort_values(ascending=False)
        return pd.DataFrame({"player": names.reindex(s.index[:k]).values,
                             "rating": s.iloc[:k].round(2).values})

