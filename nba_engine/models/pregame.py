"""Pregame model: predicted home margin mu from rating and context features, then
P(home win) = Phi(mu / sigma).

Two learners predict the margin and are averaged:
- a ridge linear model (stable, interpretable), and
- a gradient-boosted tree model (captures interactions such as rest x travel, or rating
  reliability early in the season).

Everything is walk-forward: each season is predicted by models fit only on earlier seasons,
and every feature is computed from games before the one being predicted. Scheduled (future)
games flow through the same feature code using projected lineups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_engine.features.efficiency import efficiency_ratings
from nba_engine.features.market import closing_lines, kalshi_pretip
from nba_engine.features.players import PlayerRatings
from nba_engine.features.ratings import elo_ratings, margin_ratings
from nba_engine.features.schedule_context import schedule_features
from nba_engine.models.gbm import GBMRegressor

log = logging.getLogger(__name__)

FEATURES = [
    "elo_diff", "rating_margin_pred", "eff_margin_pred", "player_strength_diff",
    "home_court", "rest_diff", "home_b2b", "away_b2b", "home_3in4", "away_3in4",
    "home_travel_k", "away_travel_k", "home_tz_abs", "away_tz_abs",
    "altitude_home", "home_out_regulars", "away_out_regulars", "playoffs",
]
# Extra inputs only the tree model sees (they matter through interactions).
GBM_EXTRA = ["home_games_played", "away_games_played", "total_pred", "pace_pred",
             "home_player_strength", "away_player_strength"]


def load_history(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    h = Path(data_dir) / "history"
    out = {}
    for name in ("games", "player_box", "odds", "kalshi_markets", "kalshi_candles", "stints"):
        path = h / f"{name}.parquet"
        out[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return out


def _games_played(games: pd.DataFrame) -> pd.DataFrame:
    count: dict[tuple[int, str], int] = {}
    rows = []
    for r in games.itertuples(index=False):
        h, a = (r.season, r.home), (r.season, r.away)
        rows.append({"game_id": r.game_id, "home_games_played": count.get(h, 0),
                     "away_games_played": count.get(a, 0)})
        count[h] = count.get(h, 0) + 1
        count[a] = count.get(a, 0) + 1
    return pd.DataFrame(rows)


def build_dataset(hist: dict[str, pd.DataFrame], player_kwargs: dict | None = None,
                  team_kwargs: dict | None = None, eff_kwargs: dict | None = None,
                  upcoming: pd.DataFrame | None = None,
                  upcoming_box: pd.DataFrame | None = None,
                  player_model_out: list | None = None) -> pd.DataFrame:
    """Features for every completed game, plus `upcoming` scheduled games (with projected
    lineups in `upcoming_box`) appended at the end with completed=False. Pass a list as
    `player_model_out` to receive the fitted PlayerRatings object."""
    games = hist["games"]
    games = games[games["completed"]].copy()
    box = hist["player_box"]
    if upcoming is not None and len(upcoming):
        games = pd.concat([games, upcoming.assign(completed=False)], ignore_index=True)
        if upcoming_box is not None and len(upcoming_box):
            box = pd.concat([box, upcoming_box], ignore_index=True)
    games["tip_utc"] = pd.to_datetime(games["tip_utc"], utc=True)
    games = games.sort_values("tip_utc").reset_index(drop=True)
    log.info("features: %d games (%d scheduled)", len(games), int((~games["completed"]).sum()))
    df = games.copy()
    df = df.merge(elo_ratings(games), on="game_id")
    df = df.merge(margin_ratings(games, **(team_kwargs or {})), on="game_id")
    df = df.merge(efficiency_ratings(games, box, **(eff_kwargs or {})), on="game_id")
    df = df.merge(schedule_features(games), on="game_id")
    df = df.merge(_games_played(games), on="game_id")
    pk = dict(player_kwargs or {})
    use_stints = pk.pop("use_stints", True)
    pr = PlayerRatings(**pk)
    stints = hist.get("stints") if use_stints else None
    df = df.merge(pr.run(games, box, stints=stints), on="game_id")
    if not hist["odds"].empty:
        df = df.merge(closing_lines(hist["odds"]), on="game_id", how="left")
    if not hist["kalshi_markets"].empty:
        df = df.merge(kalshi_pretip(hist["kalshi_markets"], hist["kalshi_candles"],
                                    games[games["completed"]]), on="game_id", how="left")
    df["margin"] = df["home_score"] - df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(int)
    df["home_court"] = (~df["neutral"]).astype(int)
    df["playoffs"] = (df["season_type"] != "regular").astype(int)
    df["home_travel_k"] = df["home_travel_km"] / 1000
    df["away_travel_k"] = df["away_travel_km"] / 1000
    df["home_tz_abs"] = df["home_tz_shift"].abs()
    df["away_tz_abs"] = df["away_tz_shift"].abs()
    if player_model_out is not None:
        player_model_out.append(pr)
    return df


@dataclass
class MarginModel:
    """Ridge linear margin model."""
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    ridge: float = 1.0
    coef_: np.ndarray | None = None
    sigma_: float = 12.0

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(df)), X])

    def fit(self, df: pd.DataFrame) -> MarginModel:
        X, y = self._X(df), df["margin"].to_numpy(float)
        reg = np.full(X.shape[1], self.ridge)
        reg[0] = 0.0
        self.coef_ = np.linalg.solve(X.T @ X + np.diag(reg), X.T @ y)
        self.sigma_ = float(np.std(y - X @ self.coef_))
        return self

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        return self._X(df) @ self.coef_

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return norm.cdf(self.predict_margin(df) / self.sigma_)

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": ["intercept", *self.features], "coef": self.coef_})


@dataclass
class EnsembleModel:
    """Weighted average of the linear and gradient-boosted margin predictions."""
    gbm_weight: float = 0.5
    gbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 300, "learning_rate": 0.04, "max_depth": 3,
        "min_samples_leaf": 80, "l2": 1.0, "random_state": 0})
    linear: MarginModel = field(default_factory=MarginModel)
    gbm: GBMRegressor | None = None
    sigma_: float = 12.0

    @property
    def gbm_features(self) -> list[str]:
        return [*FEATURES, *GBM_EXTRA]

    def fit(self, df: pd.DataFrame) -> EnsembleModel:
        self.linear.fit(df)
        if self.gbm_weight > 0:
            self.gbm = GBMRegressor(**self.gbm_params)
            self.gbm.fit(df[self.gbm_features].to_numpy(float), df["margin"].to_numpy(float))
        # sigma from out-of-fold-ish residuals: the in-sample GBM fit is optimistic, so take
        # the linear residual spread (close to out-of-sample for a 19-parameter model).
        self.sigma_ = self.linear.sigma_
        return self

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        mu = self.linear.predict_margin(df)
        if self.gbm is not None and self.gbm_weight > 0:
            g = self.gbm.predict(df[self.gbm_features].to_numpy(float))
            mu = (1 - self.gbm_weight) * mu + self.gbm_weight * g
        return mu

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return norm.cdf(self.predict_margin(df) / self.sigma_)

    def coef_table(self) -> pd.DataFrame:
        return self.linear.coef_table()


def spread_sigma(train: pd.DataFrame) -> float:
    m = train.dropna(subset=["home_spread"])
    return float(np.std(m["margin"] + m["home_spread"])) if len(m) > 100 else 12.0


def walk_forward(df: pd.DataFrame, first_season: int, warmup_seasons: int = 1,
                 gbm_weight: float = 0.5) -> tuple[pd.DataFrame, dict[int, EnsembleModel]]:
    """Predict each season >= first_season with a model fit on completed games of seasons
    before it (excluding the first `warmup_seasons`, where ratings are still burning in)."""
    done = df[df["completed"]] if "completed" in df else df
    seasons = sorted(done["season"].unique())
    min_train = seasons[0] + warmup_seasons
    preds, models = [], {}
    for s in [x for x in seasons if x >= first_season]:
        train = done[(done["season"] >= min_train) & (done["season"] < s)]
        test = done[done["season"] == s].copy()
        if len(train) < 500 or test.empty:
            continue
        model = EnsembleModel(gbm_weight=gbm_weight).fit(train)
        models[s] = model
        test["mu"] = model.predict_margin(test)
        test["p_model"] = model.predict_proba(test)
        test["p_linear"] = norm.cdf(model.linear.predict_margin(test) / model.sigma_)
        sig = spread_sigma(train)
        test["p_spread"] = norm.cdf(-test["home_spread"] / sig)
        preds.append(test)
    return pd.concat(preds, ignore_index=True), models
