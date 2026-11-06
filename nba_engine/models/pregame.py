"""Pregame model: predicted home margin mu from rating and context features, then
P(home win) = Phi(mu / sigma). Trained and evaluated walk-forward by season: each season is
predicted by a model fit only on earlier seasons, and every feature is computed from games
before the one being predicted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_engine.features.market import closing_lines, kalshi_pretip
from nba_engine.features.players import PlayerRatings
from nba_engine.features.ratings import elo_ratings, margin_ratings
from nba_engine.features.schedule_context import schedule_features

log = logging.getLogger(__name__)

FEATURES = [
    "elo_diff", "rating_margin_pred", "player_strength_diff", "home_court",
    "rest_diff", "home_b2b", "away_b2b", "home_3in4", "away_3in4",
    "home_travel_k", "away_travel_k", "home_tz_abs", "away_tz_abs",
    "altitude_home", "home_out_regulars", "away_out_regulars", "playoffs",
]


def load_history(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    h = Path(data_dir) / "history"
    out = {}
    for name in ("games", "player_box", "odds", "kalshi_markets", "kalshi_candles"):
        path = h / f"{name}.parquet"
        out[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return out


def build_dataset(hist: dict[str, pd.DataFrame], player_kwargs: dict | None = None
                  ) -> pd.DataFrame:
    games = hist["games"]
    games = games[games["completed"]].sort_values("tip_utc").reset_index(drop=True)
    games["tip_utc"] = pd.to_datetime(games["tip_utc"], utc=True)
    log.info("features: %d completed games", len(games))
    df = games.copy()
    df = df.merge(elo_ratings(games), on="game_id")
    df = df.merge(margin_ratings(games), on="game_id")
    df = df.merge(schedule_features(games), on="game_id")
    pr = PlayerRatings(**(player_kwargs or {}))
    df = df.merge(pr.run(games, hist["player_box"]), on="game_id")
    if not hist["odds"].empty:
        df = df.merge(closing_lines(hist["odds"]), on="game_id", how="left")
    if not hist["kalshi_markets"].empty:
        df = df.merge(kalshi_pretip(hist["kalshi_markets"], hist["kalshi_candles"], games),
                      on="game_id", how="left")
    df["margin"] = df["home_score"] - df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(int)
    df["home_court"] = (~df["neutral"]).astype(int)
    df["playoffs"] = (df["season_type"] != "regular").astype(int)
    df["home_travel_k"] = df["home_travel_km"] / 1000
    df["away_travel_k"] = df["away_travel_km"] / 1000
    df["home_tz_abs"] = df["home_tz_shift"].abs()
    df["away_tz_abs"] = df["away_tz_shift"].abs()
    df.attrs["player_model"] = pr
    return df


@dataclass
class MarginModel:
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


def spread_sigma(train: pd.DataFrame) -> float:
    m = train.dropna(subset=["home_spread"])
    return float(np.std(m["margin"] + m["home_spread"])) if len(m) > 100 else 12.0


def walk_forward(df: pd.DataFrame, first_season: int, warmup_seasons: int = 1
                 ) -> tuple[pd.DataFrame, dict[int, MarginModel]]:
    """Predict each season >= first_season with a model fit on seasons before it
    (excluding the first `warmup_seasons`, where ratings are still burning in)."""
    seasons = sorted(df["season"].unique())
    min_train = seasons[0] + warmup_seasons
    preds, models = [], {}
    for s in [x for x in seasons if x >= first_season]:
        train = df[(df["season"] >= min_train) & (df["season"] < s)]
        test = df[df["season"] == s].copy()
        if len(train) < 500 or test.empty:
            continue
        model = MarginModel().fit(train)
        models[s] = model
        test["mu"] = model.predict_margin(test)
        test["p_model"] = model.predict_proba(test)
        sig = spread_sigma(train)
        test["p_spread"] = norm.cdf(-test["home_spread"] / sig)
        preds.append(test)
    return pd.concat(preds, ignore_index=True), models
