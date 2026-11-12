"""Hyperparameter tuning with an honest split.

Settings are chosen on VALIDATION seasons only (2016-17 to 2021-22). The HOLDOUT seasons
(2022-23 onward, including the current season) are scored once with the chosen settings and
never used for selection, so holdout numbers are a fair estimate of live accuracy.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import pandas as pd

from nba_engine.models.metrics import log_loss
from nba_engine.models.pregame import build_dataset, walk_forward

log = logging.getLogger(__name__)

VALIDATION = range(2017, 2023)
HOLDOUT_FROM = 2023
PARAMS_PATH = Path(__file__).resolve().parents[1] / "config" / "model_params.json"

DEFAULT_PARAMS = {
    "player": {"ridge": 3.0, "half_life_days": 365.0, "box_prior": True, "use_stints": True},
    "team": {"half_life_days": 60.0, "ridge": 4.0},
    "eff": {"half_life_days": 75.0, "ridge": 3.0},
    "gbm_weight": 0.5,
}


def load_params() -> dict:
    if PARAMS_PATH.exists():
        return json.loads(PARAMS_PATH.read_text())
    return json.loads(json.dumps(DEFAULT_PARAMS))


def score(pred: pd.DataFrame, seasons) -> dict:
    d = pred[pred["season"].isin(list(seasons))].dropna(subset=["p_spread"])
    return {"games": len(d), "model": log_loss(d["home_win"], d["p_model"]),
            "market": log_loss(d["home_win"], d["p_spread"]),
            "gap": log_loss(d["home_win"], d["p_model"]) - log_loss(d["home_win"],
                                                                     d["p_spread"])}


def tune(hist: dict, out_csv: Path, players: list[dict] | None = None,
         gbm_weights: list[float] | None = None) -> dict:
    players = players or [
        {"ridge": 3.0, "half_life_days": 365.0, "box_prior": True, "use_stints": True},
        {"ridge": 8.0, "half_life_days": 365.0, "box_prior": True, "use_stints": True},
        {"ridge": 3.0, "half_life_days": 730.0, "box_prior": True, "use_stints": True},
        {"ridge": 8.0, "half_life_days": 365.0, "box_prior": True, "use_stints": False},
    ]
    grid_data = list(itertools.product(players, [{"half_life_days": 60.0, "ridge": 4.0}],
                                       [{"half_life_days": 75.0, "ridge": 3.0}]))
    gbm_weights = gbm_weights or [0.0, 0.3, 0.6]
    rows, best = [], None
    for player, team, eff in grid_data:
        df = build_dataset(hist, player_kwargs=player, team_kwargs=team, eff_kwargs=eff)
        for w in gbm_weights:
            pred, _ = walk_forward(df, first_season=2017, gbm_weight=w)
            v = score(pred, VALIDATION)
            row = {"player": json.dumps(player), "team": json.dumps(team),
                   "eff": json.dumps(eff), "gbm_weight": w, "val_logloss": v["model"],
                   "val_gap": v["gap"]}
            rows.append(row)
            log.info("tune %s", row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            if best is None or v["model"] < best[0]:
                best = (v["model"], {"player": player, "team": team, "eff": eff,
                                     "gbm_weight": w})
    params = best[1]
    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    log.info("best validation log loss %.4f with %s", best[0], params)
    return params
