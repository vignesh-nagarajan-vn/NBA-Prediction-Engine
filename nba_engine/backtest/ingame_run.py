"""Phase 3 runner: train/evaluate the in-game model and study Kalshi in-game prices."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nba_engine.backtest.ingame import (
    FadeConfig,
    attach_states,
    kalshi_game_minutes,
    simulate_fade,
)
from nba_engine.backtest.pregame import bootstrap_ci
from nba_engine.models.ingame import InGameModel, game_states, stern_prob
from nba_engine.models.metrics import log_loss, reliability
from nba_engine.models.report import _fmt_table

log = logging.getLogger(__name__)

TRAIN = range(2017, 2023)


def _phase(t_left: pd.Series) -> pd.Series:
    el = 2880 - t_left
    return pd.cut(el, [-1, 720, 1440, 2160, 2880.5, 1e9],
                  labels=["Q1", "Q2", "Q3", "Q4", "OT"]).astype(str)


def run(data_dir: Path | str, report_path: Path) -> dict:
    h = Path(data_dir) / "history"
    games = pd.read_parquet(h / "games.parquet")
    games = games[games["completed"]]
    preds = pd.read_parquet(h / "pregame_predictions.parquet")
    mu = preds.set_index("game_id")["mu"]
    plays = pd.read_parquet(h / "plays.parquet")
    log.info("in-game: %d plays", len(plays))

    states = game_states(plays, games, mu, every_s=20)
    states.loc[states["period"] > 4, "t_left"] = states.loc[states["period"] > 4, "clock_s"]
    model = InGameModel().fit(states[states["season"].isin(list(TRAIN))])
    test = states[states["season"] >= 2023].copy()
    test["p_model"] = model.predict(test)
    test["p_stern"] = stern_prob(test["d"], test["t_left"], test["mu"])
    test["phase"] = _phase(test["t_left"])
    live = test[test["t_left"] > 0]

    by_phase = []
    for ph, d in live.groupby("phase"):
        by_phase.append({"phase": ph, "states": len(d),
                         "model_logloss": log_loss(d["home_win"], d["p_model"]),
                         "stern_logloss": log_loss(d["home_win"], d["p_stern"])})
    by_phase = pd.DataFrame(by_phase)
    rel = pd.DataFrame(reliability(live["home_win"], live["p_model"], 10),
                       columns=["predicted", "observed", "states"])

    # ---- Kalshi comparison (contemporaneous) --------------------------------------------
    markets = pd.read_parquet(h / "kalshi_markets.parquet")
    candles = pd.read_parquet(h / "kalshi_candles.parquet")
    minutes = kalshi_game_minutes(markets, candles, games)
    kal_games = set(minutes["game_id"])
    full = game_states(plays[plays["game_id"].isin(kal_games)], games, mu, every_s=0)
    full.loc[full["period"] > 4, "t_left"] = full.loc[full["period"] > 4, "clock_s"]
    full["p_model"] = model.predict(full)
    now = attach_states(minutes, full, delay_s=0)
    now = now[now["t_left"] > 0]
    now["phase"] = _phase(now["t_left"])
    kal_rows = []
    for ph, d in now.groupby("phase"):
        kal_rows.append({"phase": ph, "minutes": len(d),
                         "kalshi_logloss": log_loss(d["home_win"], d["p_kalshi"]),
                         "model_logloss": log_loss(d["home_win"], d["p_model"]),
                         "mean_abs_gap": float((d["p_model"] - d["p_kalshi"]).abs().mean())})
    kal = pd.DataFrame(kal_rows)
    kal_all = {"minutes": len(now), "games": now["game_id"].nunique(),
               "kalshi": log_loss(now["home_win"], now["p_kalshi"]),
               "model": log_loss(now["home_win"], now["p_model"])}

    # ---- Fade strategy with a 30s feed delay, dead balls only -----------------------------
    delayed = attach_states(minutes, full, delay_s=30)
    strat = []
    trades_keep = None
    for th in (0.05, 0.08, 0.12, 0.16):
        for hold in (3, 10):
            t = simulate_fade(delayed, FadeConfig(threshold=th, hold_min=hold))
            if t.empty:
                strat.append({"threshold": th, "hold_min": hold, "trades": 0})
                continue
            lo, hi = bootstrap_ci(t["pnl"].to_numpy())
            strat.append({"threshold": th, "hold_min": hold, "trades": len(t),
                          "roi": t["pnl"].sum() / t["cost"].sum(),
                          "pnl_per_trade_$": t["pnl"].mean(),
                          "ci90_$": f"{lo:.2f} to {hi:.2f}",
                          "kalshi_moved_toward_model_c": 100 * t["kalshi_move"].mean()})
            if th == 0.08 and hold == 10:
                trades_keep = t
    strat = pd.DataFrame(strat)

    # Save artifacts for figures.
    now.to_parquet(h / "ingame_kalshi_minutes.parquet", index=False)
    test.sample(min(len(test), 400_000), random_state=0).to_parquet(
        h / "ingame_test_states.parquet", index=False)
    if trades_keep is not None:
        trades_keep.to_parquet(h / "ingame_fade_trades.parquet", index=False)

    lines = [
        "# Phase 3 report: in-game win probability and Kalshi live prices",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Model trained on 2016-17 to "
        "2021-22 play-by-play; everything below is scored on 2022-23 onward._",
        "",
        "## Model accuracy by game phase (holdout seasons)",
        "",
        "`stern` is the classic random-walk formula; `model` is our fitted version with "
        "possession and phase-specific weights. Both use the pregame expected margin.",
        "",
        _fmt_table(by_phase),
        "",
        f"Overall: model {log_loss(live['home_win'], live['p_model']):.4f}, "
        f"stern {log_loss(live['home_win'], live['p_stern']):.4f} log loss over "
        f"{len(live):,} game states.",
        "",
        "Calibration (predicted vs observed home win rate):",
        "",
        _fmt_table(rel),
        "",
        "## Against Kalshi's live prices",
        "",
        f"{kal_all['minutes']:,} game-minutes across {kal_all['games']} games where Kalshi "
        "traded. Kalshi's price vs our model given the same moment's game state:",
        "",
        _fmt_table(kal),
        "",
        f"Overall: Kalshi {kal_all['kalshi']:.4f} vs model {kal_all['model']:.4f}.",
        "",
        "## Fading swings (strategy S3)",
        "",
        "Trade only at dead balls, using game state from 30 seconds earlier (free-feed lag). "
        "Buy the side the model prefers when the gap exceeds the threshold, pay the ask, "
        "sell at the bid `hold_min` minutes later (or settle if the game ends), taker fees "
        "both ways, 100 contracts. `kalshi_moved_toward_model_c` is how far Kalshi's price "
        "moved in our direction while we held, in cents.",
        "",
        _fmt_table(strat),
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"by_phase": by_phase, "kalshi": kal, "kalshi_all": kal_all, "strategy": strat,
            "model": model}
