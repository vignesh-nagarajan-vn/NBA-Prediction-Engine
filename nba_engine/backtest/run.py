"""Phase 2 pregame evaluation: fair-value variants vs Kalshi, scored out of sample and traded.

Fair-value variants (all strictly walk-forward):
  kalshi_raw        Kalshi's own implied probability (the bar to beat)
  kalshi_debiased   Kalshi price through the sentiment / bias-correction layer
  model             our pregame model alone
  kalshi+model      logistic blend of Kalshi price and model
  kalshi+book       blend of Kalshi price and the sportsbook closing line (only near tip,
                    since we only have CLOSING lines, which are not known hours earlier)
  kalshi+book+model all three, near tip only
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nba_engine.backtest.pregame import TradeConfig, simulate, summarize_trades
from nba_engine.models.debias import build_snapshots, walk_forward_debias
from nba_engine.models.metrics import fit_logistic, log_loss, logit, sigmoid
from nba_engine.models.report import _fmt_table

log = logging.getLogger(__name__)

NEAR_TIP_MIN = 60  # closing lines are only usable this close to tip


def walk_forward_blend(s: pd.DataFrame, cols: list[str], min_train_games: int = 300,
                       refit_days: int = 14, l2: float = 1.0) -> pd.Series:
    s = s.sort_values("tip_utc")
    out = pd.Series(np.nan, index=s.index)
    ok = s[cols].notna().all(axis=1)
    days = s["tip_utc"].dt.floor("D")
    cur, end = days.min(), days.max()
    while cur <= end:
        nxt = cur + pd.Timedelta(days=refit_days)
        tr = s[ok & (s["tip_utc"] < cur - pd.Timedelta(days=1))]
        te = s.index[ok & (days >= cur) & (days < nxt)]
        if tr["game_id"].nunique() >= min_train_games and len(te):
            X = np.column_stack([np.ones(len(tr))] + [logit(tr[c]) for c in cols])
            w = fit_logistic(X, tr["home_win"], l2=l2)
            Xt = np.column_stack([np.ones(len(te))] + [logit(s.loc[te, c]) for c in cols])
            out.loc[te] = sigmoid(Xt @ w)
        cur = nxt
    return out.reindex(s.index)


def add_fair_values(snaps: pd.DataFrame, preds: pd.DataFrame | None) -> pd.DataFrame:
    s = snaps.copy()
    s["kalshi_raw"] = s["p_kalshi"]
    s["kalshi_debiased"] = walk_forward_debias(s)
    if preds is not None and not preds.empty:
        p = preds[["game_id", "p_model", "p_spread", "ml_p_home"]].drop_duplicates("game_id")
        s = s.merge(p, on="game_id", how="left")
        s["model"] = s["p_model"]
        s["kalshi+model"] = walk_forward_blend(s, ["p_kalshi", "p_model"])
        near = s["horizon_min"] <= NEAR_TIP_MIN
        s.loc[~near, "p_spread_near"] = np.nan
        s.loc[near, "p_spread_near"] = s.loc[near, "p_spread"]
        s["kalshi+book"] = walk_forward_blend(s, ["p_kalshi", "p_spread_near"])
        s["kalshi+book+model"] = walk_forward_blend(s, ["p_kalshi", "p_spread_near",
                                                        "p_model"])
    return s


VARIANTS = ["kalshi_raw", "kalshi_debiased", "model", "kalshi+model", "kalshi+book",
            "kalshi+book+model"]


def scoring_table(s: pd.DataFrame) -> pd.DataFrame:
    """Log loss per variant per horizon, on rows where every available variant is scored."""
    rows = []
    variants = [v for v in VARIANTS if v in s.columns]
    for h, d in s.groupby("horizon_min"):
        present = [v for v in variants if d[v].notna().any()]
        dd = d.dropna(subset=present)
        if len(dd) < 50:
            continue
        row = {"horizon": f"{h // 60}h" if h >= 60 else f"{h}m", "games": len(dd)}
        base = log_loss(dd["home_win"], dd["kalshi_raw"])
        for v in variants:
            row[v] = (log_loss(dd["home_win"], dd[v]) - base) if v in present else np.nan
        row["kalshi_raw"] = base
        rows.append(row)
    return pd.DataFrame(rows)


def trading_table(s: pd.DataFrame, thresholds=(0.02, 0.03, 0.05)) -> pd.DataFrame:
    rows = []
    for v in [x for x in VARIANTS if x in s.columns and x != "kalshi_raw"]:
        for th in thresholds:
            t = simulate(s, v, TradeConfig(threshold=th))
            r = summarize_trades(t)
            if not r.get("trades"):
                rows.append({"fair_value": v, "threshold": th, "trades": 0})
                continue
            rows.append({
                "fair_value": v, "threshold": th, "trades": r["trades"],
                "win_rate": r["win_rate"], "hold_roi": r["hold_roi"],
                "hold_pnl_$": r["hold_pnl"],
                "hold_ci90_$/trade": f"{r['hold_pnl_ci90'][0]:.2f} to {r['hold_pnl_ci90'][1]:.2f}",
                "clv_exit_roi": r["clv_exit_roi"],
                "mean_clv_c": r["mean_clv_c"],
                "clv_ci90_c": f"{r['clv_ci90'][0]:.2f} to {r['clv_ci90'][1]:.2f}",
            })
    return pd.DataFrame(rows)


def run(data_dir: Path | str, report_path: Path) -> dict:
    h = Path(data_dir) / "history"
    games = pd.read_parquet(h / "games.parquet")
    games["tip_utc"] = pd.to_datetime(games["tip_utc"], utc=True)
    markets = pd.read_parquet(h / "kalshi_markets.parquet")
    candles = pd.read_parquet(h / "kalshi_candles.parquet")
    pred_path = h / "pregame_predictions.parquet"
    preds = pd.read_parquet(pred_path) if pred_path.exists() else None
    snaps = build_snapshots(markets, candles, games)
    s = add_fair_values(snaps, preds)
    s.to_parquet(h / "kalshi_snapshots.parquet", index=False)
    score = scoring_table(s)
    trade = trading_table(s)
    lines = [
        "# Phase 2 report: pregame fair value vs. Kalshi",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. {s['game_id'].nunique()} games with "
        f"Kalshi quotes, {s['tip_utc'].min():%Y-%m-%d} to {s['tip_utc'].max():%Y-%m-%d}. "
        "Every probability is out of sample (fit only on games at least a day earlier)._",
        "",
        "## Which fair value predicts outcomes best?",
        "",
        "`kalshi_raw` is Kalshi's own log loss at that horizon. Other columns are the "
        "**difference** from it: negative means more accurate than Kalshi. Horizons are time "
        "before tip. Sportsbook blends exist only within 1h of tip (we only have closing "
        "lines).",
        "",
        _fmt_table(score),
        "",
        "## Trading it",
        "",
        "Buy the YES side whose fair value beats the ask by more than `threshold` after the "
        "taker fee, earliest horizon from 24h to 15m, one trade per game, 100 contracts. "
        "`hold` holds to settlement; `clv_exit` sells at the bid at tip. Fills are "
        "conservative (pay the ask, sell the bid) but ignore depth.",
        "",
        _fmt_table(trade),
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"score": score, "trade": trade}
