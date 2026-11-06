"""Player-impact ratings and lineup-adjusted team strength.

Ratings: game-level adjusted plus-minus. Each game is one regression row:
    margin_home = hca * (1 - neutral) + sum_home(min_p / 48 * r_p) - sum_away(min_p / 48 * r_p)
solved by exponentially time-weighted ridge regression (ratings shrink toward 0 = average).
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


def is_available(row) -> bool:
    if row.minutes > 0:
        return True
    return bool(row.did_not_play) and "COACH" in str(row.reason).upper()


class PlayerRatings:
    def __init__(self, half_life_days: float = 365.0, ridge: float = 8.0,
                 solve_every_days: int = 7, ewma_alpha: float = 0.25):
        self.hl = half_life_days
        self.ridge = ridge
        self.solve_every = solve_every_days
        self.alpha = ewma_alpha

    def run(self, games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
        g = games.sort_values("tip_utc").reset_index(drop=True)
        box = box[box["game_id"].isin(set(g["game_id"]))]
        pids = sorted(box["player_id"].unique())
        pidx = {p: i for i, p in enumerate(pids)}
        n = len(pids) + 1  # + hca (last)
        hca = n - 1
        A = np.zeros((n, n))
        b = np.zeros(n)
        last_seen_scale = np.zeros(n)  # time weight of most recent appearance
        theta = np.zeros(n)
        by_game = {gid: df for gid, df in box.groupby("game_id", sort=False)}
        ewma: dict[str, float] = {}
        t0 = g["tip_utc"].iloc[0]
        last_solve_day = None
        out = []
        day_index = (g["tip_utc"] - t0).dt.days.to_numpy()
        for i, r in enumerate(g.itertuples(index=False)):
            day = int(day_index[i])
            if last_solve_day is None or day - last_solve_day >= self.solve_every:
                theta = self._solve(A, b, last_seen_scale, day, hca)
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
                        idx.append(pidx[p.player_id])
                        val.append(sign * p.minutes / 48.0)
                        prev = ewma.get(p.player_id)
                        ewma[p.player_id] = (p.minutes if prev is None
                                             else (1 - self.alpha) * prev + self.alpha * p.minutes)
            w = 2.0 ** (day / self.hl)  # growing weights == decaying old games
            ix = np.array(idx)
            v = np.array(val)
            A[np.ix_(ix, ix)] += w * np.outer(v, v)
            b[ix] += w * v * (r.home_score - r.away_score)
            last_seen_scale[ix] = w
        df = pd.DataFrame(out)
        self.theta_, self.player_ids_ = theta, pids
        return df

    def _solve(self, A, b, last_seen, day, hca) -> np.ndarray:
        scale = 2.0 ** (day / self.hl)
        if not b.any():
            return np.zeros_like(b)
        # Only players seen in roughly the last 4 half-lives matter; others stay at 0.
        active = np.where(last_seen >= scale * 2.0 ** -4)[0]
        active = np.union1d(active, [hca])
        sub = A[np.ix_(active, active)].copy()
        reg = np.full(len(active), self.ridge * scale)
        reg[active == hca] = 1e-3 * scale
        sub[np.diag_indices_from(sub)] += reg
        theta = np.zeros_like(b)
        theta[active] = np.linalg.solve(sub, b[active])
        return theta

    def top_players(self, box: pd.DataFrame, k: int = 15) -> pd.DataFrame:
        names = box.drop_duplicates("player_id").set_index("player_id")["player"]
        s = pd.Series(self.theta_[:-1], index=self.player_ids_).sort_values(ascending=False)
        return pd.DataFrame({"player": names.reindex(s.index[:k]).values,
                             "rating": s.iloc[:k].round(2).values})

