"""Kalshi price snapshots and the sentiment / bias-correction ("debias") layer.

Snapshots: for every game and decision horizon (e.g. 6h before tip), the executable Kalshi
quotes for both team markets, the implied P(home), recent price move, recent volume, plus
the quotes at tip (for closing-line value). Built from 1-minute candles, which carry the
close of the best bid/ask each minute but not depth.

Debias model: logistic regression of the outcome on logit(Kalshi price) and crowd-sentiment
features, fit only on earlier games:
    logit P(home wins) = a + b * logit(p_kalshi) + c * |logit p| * logit p   (longshot shape)
                         + d * popular-team tilt + e * recent-streak tilt
                         + f * price momentum + g * volume surge
If the crowd is unbiased, b = 1 and everything else is 0, and the layer does nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_engine.data.schedule import ET, parse_event_ticker
from nba_engine.models.metrics import logit, sigmoid

# Large national fan bases / heavy national TV exposure: candidates for retail overpricing.
POPULAR = {"LAL", "GSW", "BOS", "NYK", "CHI", "MIA", "PHI", "DAL"}

HORIZONS_MIN = [24 * 60, 12 * 60, 6 * 60, 3 * 60, 60, 15]


def _quotes_asof(candles: pd.DataFrame, targets: pd.DataFrame, tolerance_min: int
                 ) -> pd.DataFrame:
    """For each (ticker, ts) in targets, the last candle at or before ts within tolerance."""
    c = candles.sort_values("end_ts")
    t = targets.sort_values("ts")
    out = pd.merge_asof(t, c, left_on="ts", right_on="end_ts", by="ticker",
                        direction="backward", tolerance=tolerance_min * 60)
    return out


def team_form(games: pd.DataFrame) -> pd.DataFrame:
    """Signed win streak and last-game margin per team going INTO each game."""
    g = games.sort_values("tip_utc")
    streak: dict[str, int] = {}
    last_margin: dict[str, int] = {}
    rows = []
    for r in g.itertuples(index=False):
        rows.append({"game_id": r.game_id,
                     "home_streak": streak.get(r.home, 0), "away_streak": streak.get(r.away, 0),
                     "home_last_margin": last_margin.get(r.home, 0),
                     "away_last_margin": last_margin.get(r.away, 0)})
        if not getattr(r, "completed", True):
            continue
        m = r.home_score - r.away_score
        for team, won, mm in ((r.home, m > 0, m), (r.away, m < 0, -m)):
            s = streak.get(team, 0)
            streak[team] = (s + 1 if s >= 0 else 1) if won else (s - 1 if s <= 0 else -1)
            last_margin[team] = mm
    return pd.DataFrame(rows)


def build_snapshots(markets: pd.DataFrame, candles: pd.DataFrame, games: pd.DataFrame,
                    horizons_min: list[int] = HORIZONS_MIN, staleness_min: int = 60
                    ) -> pd.DataFrame:
    g = games[games["completed"]].copy()
    g["et_day"] = g["tip_utc"].dt.tz_convert(ET).dt.strftime("%y%b%d").str.upper()
    key = {(r.et_day, r.away, r.home): r for r in g.itertuples(index=False)}
    pairs = []
    for ev, mk in markets.groupby("event_ticker"):
        try:
            day, away, home = parse_event_ticker(ev)
        except ValueError:
            continue
        gm = key.get((day, away, home))
        teams = dict(zip(mk["team"], mk["ticker"], strict=False))
        if gm is None or home not in teams or away not in teams:
            continue
        pairs.append((gm, teams[home], teams[away]))
    if not pairs:
        return pd.DataFrame()

    targets = []
    for gm, th, ta in pairs:
        tip = int(gm.tip_utc.timestamp())
        for h in [*horizons_min, 0]:
            for side, tk in (("home", th), ("away", ta)):
                targets.append({"game_id": gm.game_id, "horizon_min": h, "side": side,
                                "ticker": tk, "ts": tip - h * 60})
                if h:
                    targets.append({"game_id": gm.game_id, "horizon_min": -h, "side": side,
                                    "ticker": tk, "ts": tip - h * 60 - 360 * 60})  # 6h earlier
    t = pd.DataFrame(targets)
    cols = ["ticker", "end_ts", "yes_bid", "yes_ask", "last", "volume"]
    q = _quotes_asof(candles[cols], t, staleness_min)

    # Volume in the hour before each snapshot vs the market's average hourly volume so far.
    c = candles.sort_values(["ticker", "end_ts"])[["ticker", "end_ts", "volume"]].copy()
    c["cumvol"] = c.groupby("ticker")["volume"].cumsum()
    first_ts = c.groupby("ticker")["end_ts"].min()
    v_now = pd.merge_asof(t.sort_values("ts"), c.sort_values("end_ts"), left_on="ts",
                          right_on="end_ts", by="ticker", direction="backward")
    t2 = t.assign(ts=t["ts"] - 3600)
    v_prev = pd.merge_asof(t2.sort_values("ts"), c.sort_values("end_ts"), left_on="ts",
                           right_on="end_ts", by="ticker", direction="backward")
    v_now = v_now.set_index(["game_id", "horizon_min", "side"])["cumvol"]
    v_prev = v_prev.set_index(["game_id", "horizon_min", "side"])["cumvol"]

    q = q.set_index(["game_id", "horizon_min", "side"])
    q["vol_1h"] = (v_now - v_prev.reindex(v_now.index).fillna(0)).reindex(q.index)
    q["age_h"] = (q["ts"] - q["ticker"].map(first_ts)) / 3600
    q = q.reset_index()

    wide = q.pivot_table(index=["game_id", "horizon_min"], columns="side",
                         values=["yes_bid", "yes_ask", "last", "vol_1h", "age_h"],
                         aggfunc="first")
    wide.columns = [f"{s}_{v}" for v, s in wide.columns]
    wide = wide.reset_index()

    def implied(df: pd.DataFrame) -> pd.Series:
        hm = (df["home_yes_bid"] + df["home_yes_ask"]) / 2
        am = (df["away_yes_bid"] + df["away_yes_ask"]) / 2
        ok_h = (df["home_yes_ask"] > df["home_yes_bid"]) & (df["home_yes_ask"]
                                                            - df["home_yes_bid"] <= 0.10)
        ok_a = (df["away_yes_ask"] > df["away_yes_bid"]) & (df["away_yes_ask"]
                                                            - df["away_yes_bid"] <= 0.10)
        ph = hm.where(ok_h)
        pa = (1 - am).where(ok_a)
        return pd.concat([ph, pa], axis=1).mean(axis=1)

    wide["p_kalshi"] = implied(wide)
    snaps = wide[wide["horizon_min"] > 0].copy()
    tipq = wide[wide["horizon_min"] == 0].set_index("game_id")
    earlier = wide[wide["horizon_min"] < 0].assign(horizon_min=lambda d: -d["horizon_min"])
    earlier = earlier.set_index(["game_id", "horizon_min"])["p_kalshi"]
    snaps["p_kalshi_6h_ago"] = earlier.reindex(
        pd.MultiIndex.from_frame(snaps[["game_id", "horizon_min"]])).to_numpy()
    for col in ("home_yes_bid", "home_yes_ask", "away_yes_bid", "away_yes_ask", "p_kalshi"):
        snaps[f"tip_{col}"] = snaps["game_id"].map(tipq[col])
    vol = snaps[["home_vol_1h", "away_vol_1h"]].sum(axis=1, min_count=1)
    snaps["vol_1h"] = vol

    meta = g.set_index("game_id")
    snaps["home"] = snaps["game_id"].map(meta["home"])
    snaps["away"] = snaps["game_id"].map(meta["away"])
    snaps["tip_utc"] = snaps["game_id"].map(meta["tip_utc"])
    snaps["season"] = snaps["game_id"].map(meta["season"])
    snaps["home_win"] = snaps["game_id"].map(
        (meta["home_score"] > meta["away_score"]).astype(int))
    snaps = snaps.merge(team_form(games), on="game_id", how="left")
    return snaps.dropna(subset=["p_kalshi"]).reset_index(drop=True)


DEBIAS_FEATURES = ["lp", "lp_abs_lp", "popular_tilt", "streak_tilt", "momentum", "vol_surge",
                   "lp_x_late"]


def debias_features(s: pd.DataFrame) -> pd.DataFrame:
    lp = logit(s["p_kalshi"])
    f = pd.DataFrame(index=s.index)
    f["lp"] = lp
    f["lp_abs_lp"] = lp * np.abs(lp)
    f["popular_tilt"] = (s["home"].isin(POPULAR).astype(int)
                         - s["away"].isin(POPULAR).astype(int))
    f["streak_tilt"] = np.clip(s["home_streak"].fillna(0), -5, 5) - np.clip(
        s["away_streak"].fillna(0), -5, 5)
    prev = s["p_kalshi_6h_ago"].fillna(s["p_kalshi"])
    f["momentum"] = lp - logit(prev)
    f["vol_surge"] = np.log1p(s["vol_1h"].fillna(0).clip(lower=0)) / 10
    f["lp_x_late"] = lp * (s["horizon_min"] <= 60).astype(int)
    return f


class Debiaser:
    """Walk-forward logistic bias correction for Kalshi prices."""

    def __init__(self, l2: float = 20.0):
        self.l2 = l2
        self.w: np.ndarray | None = None

    def fit(self, s: pd.DataFrame) -> Debiaser:
        X = debias_features(s).to_numpy(float)
        X = np.column_stack([np.ones(len(X)), X])
        # Shrink toward the "no bias" model (intercept 0, slope on lp = 1, rest 0) by
        # fitting the residual on top of an offset of lp.
        self.w = _fit_offset_logistic(X, s["home_win"].to_numpy(float), offset_col=1,
                                      l2=self.l2)
        return self

    def predict(self, s: pd.DataFrame) -> np.ndarray:
        X = debias_features(s).to_numpy(float)
        X = np.column_stack([np.ones(len(X)), X])
        return sigmoid(X[:, 1] + X @ self.w)

    def coef(self) -> pd.Series:
        names = ["intercept", *DEBIAS_FEATURES]
        w = self.w.copy()
        w[1] += 1.0  # report the total slope on lp
        return pd.Series(w, index=names)


def _fit_offset_logistic(X, y, offset_col: int, l2: float, iters: int = 50) -> np.ndarray:
    """Logistic regression with logit = X[:, offset_col] + X @ w and an L2 penalty on w,
    i.e. ridge-shrunk toward 'the market price is right'."""
    off = X[:, offset_col]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(off + X @ w)
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def walk_forward_debias(snaps: pd.DataFrame, min_train_games: int = 300,
                        refit_days: int = 14) -> pd.Series:
    """Out-of-sample debiased probabilities: each snapshot is scored by a model fit only on
    games that tipped at least a day before it, refit every `refit_days`."""
    s = snaps.sort_values("tip_utc")
    out = pd.Series(np.nan, index=s.index)
    days = s["tip_utc"].dt.floor("D")
    start = days.min()
    cur = start
    end = days.max()
    while cur <= end:
        nxt = cur + pd.Timedelta(days=refit_days)
        train = s[s["tip_utc"] < cur - pd.Timedelta(days=1)]
        test_idx = s.index[(days >= cur) & (days < nxt)]
        if train["game_id"].nunique() >= min_train_games and len(test_idx):
            model = Debiaser().fit(train)
            out.loc[test_idx] = model.predict(s.loc[test_idx])
        cur = nxt
    return out.reindex(snaps.index)
