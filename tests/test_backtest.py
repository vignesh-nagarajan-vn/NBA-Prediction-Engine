import numpy as np
import pandas as pd

from nba_engine.backtest.pregame import TradeConfig, bootstrap_ci, simulate
from nba_engine.models.debias import team_form


def _snap(**kw):
    base = {"game_id": "g", "season": 2026, "tip_utc": pd.Timestamp("2026-01-01T00:00Z"),
            "horizon_min": 60, "home": "BOS", "away": "NYK", "home_win": 1,
            "home_yes_bid": 0.49, "home_yes_ask": 0.50, "away_yes_bid": 0.49,
            "away_yes_ask": 0.51, "tip_home_yes_bid": 0.55, "tip_away_yes_bid": 0.43,
            "tip_p_kalshi": 0.56, "fair": 0.60}
    base.update(kw)
    return base


def test_simulate_buys_home_and_accounts_fees():
    s = pd.DataFrame([_snap()])
    t = simulate(s, "fair", TradeConfig(threshold=0.03, contracts=100))
    assert len(t) == 1
    r = t.iloc[0]
    assert r.side == "home" and r.ask == 0.50
    fee = 1.75  # 0.07 * 100 * 0.5 * 0.5
    assert abs(r.cost - (50 + fee)) < 1e-9
    assert abs(r.pnl_hold - (100 - 50 - fee)) < 1e-9
    exit_fee = float(np.ceil(0.07 * 100 * 0.55 * 0.45 * 100) / 100)
    assert abs(r.pnl_clv_exit - (55 - exit_fee - 50 - fee)) < 1e-9
    assert abs(r.clv_cents - 6.0) < 1e-9


def test_simulate_takes_away_side_and_one_trade_per_game():
    s = pd.DataFrame([_snap(fair=0.40, horizon_min=360), _snap(fair=0.30, horizon_min=60)])
    t = simulate(s, "fair", TradeConfig(threshold=0.03))
    assert len(t) == 1
    assert t.iloc[0].side == "away" and t.iloc[0].horizon_min == 360
    assert t.iloc[0].won == 0


def test_no_trade_when_edge_below_threshold_after_fees():
    s = pd.DataFrame([_snap(fair=0.53)])  # 3c raw edge, ~1.75c fee
    assert simulate(s, "fair", TradeConfig(threshold=0.03)).empty


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).normal(1.0, 1.0, 400)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_team_form_streaks_are_pregame():
    t = pd.Timestamp
    games = pd.DataFrame([
        {"game_id": "1", "tip_utc": t("2026-01-01T00:00Z"), "home": "BOS", "away": "NYK",
         "home_score": 100, "away_score": 90, "completed": True},
        {"game_id": "2", "tip_utc": t("2026-01-02T00:00Z"), "home": "NYK", "away": "BOS",
         "home_score": 80, "away_score": 90, "completed": True},
        {"game_id": "3", "tip_utc": t("2026-01-03T00:00Z"), "home": "BOS", "away": "NYK",
         "home_score": 0, "away_score": 0, "completed": False},
    ])
    f = team_form(games).set_index("game_id")
    assert f.loc["1", "home_streak"] == 0
    assert f.loc["3", "home_streak"] == 2 and f.loc["3", "away_streak"] == -2
    assert f.loc["3", "home_last_margin"] == 10
