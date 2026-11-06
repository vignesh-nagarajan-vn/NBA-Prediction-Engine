"""Pregame strategy backtest on historical Kalshi quotes.

S1 value:  at each decision horizon, buy the side (home or away team's YES market) whose
           fair probability beats the ASK by more than `threshold` after the taker fee; hold
           to settlement.
S2 CLV:    same entries, but sell at the BID at tip instead of holding (pure closing-line
           value trade, no game risk).

Conservative fills: entries pay the ask, exits receive the bid (no mid fills), every order
pays Kalshi's taker fee with per-order rounding, and each game gets at most one entry per
strategy (the earliest horizon that triggers). Candles do not carry depth, so size is a flat
`contracts` per trade; real fills at size may be worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_engine.kalshi.fees import order_fee


@dataclass
class TradeConfig:
    threshold: float = 0.03
    contracts: int = 100
    min_horizon_min: int = 15
    max_horizon_min: int = 24 * 60
    fee_type: str = "quadratic_with_maker_fees"


def _fee(price: float, n: int, fee_type: str) -> float:
    return float(order_fee(round(price, 4), n, maker=False, fee_type=fee_type))


def simulate(snaps: pd.DataFrame, fair_col: str, cfg: TradeConfig) -> pd.DataFrame:
    """Returns one row per trade with S1 (hold) and S2 (exit at tip) P&L in dollars."""
    s = snaps[(snaps["horizon_min"] >= cfg.min_horizon_min)
              & (snaps["horizon_min"] <= cfg.max_horizon_min)].dropna(subset=[fair_col])
    s = s.sort_values(["game_id", "horizon_min"], ascending=[True, False])
    s = s.assign(fair_value_=s[fair_col])  # column names like "kalshi+model" aren't attributes
    trades = []
    for gid, gdf in s.groupby("game_id", sort=False):
        for r in gdf.itertuples(index=False):
            p_home = r.fair_value_
            best = None
            for side, p_win, ask, tip_bid in (
                ("home", p_home, r.home_yes_ask, r.tip_home_yes_bid),
                ("away", 1 - p_home, r.away_yes_ask, r.tip_away_yes_bid),
            ):
                if not (0 < ask < 1) or not np.isfinite(ask):
                    continue
                fee_pc = _fee(ask, cfg.contracts, cfg.fee_type) / cfg.contracts
                edge = p_win - ask - fee_pc
                if edge > cfg.threshold and (best is None or edge > best[1]):
                    best = (side, edge, ask, tip_bid, p_win, fee_pc)
            if best is None:
                continue
            side, edge, ask, tip_bid, p_win, fee_pc = best
            n = cfg.contracts
            won = r.home_win == 1 if side == "home" else r.home_win == 0
            entry_cost = ask * n + fee_pc * n
            s1 = (n if won else 0.0) - entry_cost
            s2 = np.nan
            if np.isfinite(tip_bid) and tip_bid > 0:
                s2 = tip_bid * n - _fee(tip_bid, n, cfg.fee_type) - entry_cost
            tip_mid_side = (r.tip_p_kalshi if side == "home" else 1 - r.tip_p_kalshi)
            trades.append({
                "game_id": gid, "season": r.season, "tip_utc": r.tip_utc,
                "horizon_min": r.horizon_min, "side": side,
                "team": r.home if side == "home" else r.away,
                "fair": p_win, "ask": ask, "edge": edge, "won": int(won),
                "cost": entry_cost, "pnl_hold": s1, "pnl_clv_exit": s2,
                "clv_cents": (tip_mid_side - ask) * 100 if np.isfinite(tip_mid_side) else np.nan,
            })
            break  # one entry per game
    return pd.DataFrame(trades)


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.10, seed: int = 0
                 ) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize_trades(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    lo, hi = bootstrap_ci(t["pnl_hold"].to_numpy())
    lo2, hi2 = bootstrap_ci(t["pnl_clv_exit"].to_numpy())
    clo, chi = bootstrap_ci(t["clv_cents"].to_numpy())
    return {
        "trades": len(t),
        "win_rate": t["won"].mean(),
        "avg_edge_c": t["edge"].mean() * 100,
        "hold_pnl": t["pnl_hold"].sum(),
        "hold_roi": t["pnl_hold"].sum() / t["cost"].sum(),
        "hold_pnl_ci90": (lo, hi),
        "clv_exit_pnl": t["pnl_clv_exit"].sum(),
        "clv_exit_roi": t["pnl_clv_exit"].sum() / t.loc[t["pnl_clv_exit"].notna(), "cost"].sum()
        if t["pnl_clv_exit"].notna().any() else np.nan,
        "clv_exit_ci90": (lo2, hi2),
        "mean_clv_c": t["clv_cents"].mean(),
        "clv_ci90": (clo, chi),
    }
