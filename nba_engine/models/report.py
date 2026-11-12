"""Phase 1 report: pregame model vs sportsbook closing lines and Kalshi pre-tip prices."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nba_engine.models.metrics import fit_logistic, log_loss, logit, reliability, sigmoid, summarize


def _fmt_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.astype(object).iterrows():  # object dtype keeps ints as ints
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float | np.floating):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "-")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def blend_walk_forward(pred: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Out-of-sample logistic blend of logit(prob) columns, fit on earlier seasons only."""
    out = pd.Series(np.nan, index=pred.index)
    d = pred.dropna(subset=cols)
    for s in sorted(d["season"].unique()):
        tr, te = d[d["season"] < s], d[d["season"] == s]
        if len(tr) < 1000:
            continue
        Xtr = np.column_stack([np.ones(len(tr))] + [logit(tr[c]) for c in cols])
        w = fit_logistic(Xtr, tr["home_win"])
        Xte = np.column_stack([np.ones(len(te))] + [logit(te[c]) for c in cols])
        out.loc[te.index] = sigmoid(Xte @ w)
    return out


def build_report(pred: pd.DataFrame, models: dict, df_all: pd.DataFrame,
                 top_players: pd.DataFrame | None, path: Path) -> dict:
    lines = ["# Phase 1 report: pregame model vs. the market",
             f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from walk-forward predictions. "
             "Each season is predicted by a model trained only on earlier seasons._", ""]

    # ---- coverage
    cov = df_all.groupby("season").agg(
        games=("game_id", "size"),
        with_spread=("home_spread", lambda s: int(s.notna().sum())),
        with_moneyline=("ml_p_home", lambda s: int(s.notna().sum())),
        with_kalshi=("kalshi_p_home", lambda s: int(s.notna().sum())),
    ).reset_index()
    lines += ["## Data coverage", "", _fmt_table(cov), ""]

    # ---- per-season model vs market (common subset)
    rows = []
    for s, d in pred.groupby("season"):
        d = d.dropna(subset=["p_spread"])
        if d.empty:
            continue
        m, k = summarize(d["home_win"], d["p_model"]), summarize(d["home_win"], d["p_spread"])
        rows.append({"season": int(s), "games": m["n"], "model_logloss": m["log_loss"],
                     "market_logloss": k["log_loss"],
                     "gap": m["log_loss"] - k["log_loss"], "model_acc": m["accuracy"],
                     "market_acc": k["accuracy"], "model_calib_slope": m["calib_slope"]})
    per = pd.DataFrame(rows)
    pooled = pred.dropna(subset=["p_spread"])
    pm = summarize(pooled["home_win"], pooled["p_model"])
    pk = summarize(pooled["home_win"], pooled["p_spread"])
    gap = pm["log_loss"] - pk["log_loss"]
    gate_ll = gap <= 0.005
    gate_cal = 0.9 <= pm["calib_slope"] <= 1.1
    lines += [
        "## Model vs. sportsbook closing spread",
        "",
        "The market probability converts the median closing spread across ESPN's odds "
        "providers with a normal margin distribution (sigma fit on training seasons). "
        "Lower log loss is better; `gap` is model minus market.",
        "",
        _fmt_table(per),
        "",
        f"**Pooled ({pm['n']} games):** model log loss {pm['log_loss']:.4f} vs market "
        f"{pk['log_loss']:.4f} (gap {gap:+.4f}); model accuracy {pm['accuracy']:.3f} vs "
        f"{pk['accuracy']:.3f}; model calibration slope {pm['calib_slope']:.3f}, "
        f"intercept {pm['calib_intercept']:+.3f}.",
        "",
        "### Phase 1 gate",
        "",
        f"- Log loss within 0.005 of the closing line: **{'PASS' if gate_ll else 'FAIL'}** "
        f"(gap {gap:+.4f})",
        f"- Calibration slope in [0.9, 1.1]: **{'PASS' if gate_cal else 'FAIL'}** "
        f"({pm['calib_slope']:.3f})",
        "",
        "Note: ESPN's closing lines come from mainstream books, not Pinnacle, so this is a "
        "slightly softer benchmark than the plan's sharp close.",
        "",
    ]

    # ---- honest split and the current season
    split_rows = []
    for name, seasons in (("validation (2016-17 to 2021-22, used for tuning)", range(2017, 2023)),
                          ("holdout (2022-23 onward, never used for tuning)", range(2023, 2100))):
        d = pred[pred["season"].isin(list(seasons))].dropna(subset=["p_spread"])
        if d.empty:
            continue
        m, k = summarize(d["home_win"], d["p_model"]), summarize(d["home_win"], d["p_spread"])
        split_rows.append({"split": name, "games": m["n"], "model_logloss": m["log_loss"],
                           "market_logloss": k["log_loss"], "gap": m["log_loss"] - k["log_loss"],
                           "model_acc": m["accuracy"], "market_acc": k["accuracy"]})
    lines += ["### Validation vs. holdout", "",
              "Settings were chosen on the validation seasons only, so the holdout row is the "
              "fair estimate of live accuracy.", "", _fmt_table(pd.DataFrame(split_rows)), ""]

    cur = int(pred["season"].max())
    d = pred[pred["season"] == cur]
    if len(d):
        rows = [{"source": "Our model", **summarize(d["home_win"], d["p_model"])}]
        kk = d.dropna(subset=["kalshi_p_home"])
        if len(kk) > 20:
            rows.append({"source": "Our model (games with Kalshi)",
                         **summarize(kk["home_win"], kk["p_model"])})
            rows.append({"source": "Kalshi pre-tip (same games)",
                         **summarize(kk["home_win"], kk["kalshi_p_home"])})
        ss = d.dropna(subset=["p_spread"])
        if len(ss) > 20:
            rows.append({"source": "Our model (games with a line)",
                         **summarize(ss["home_win"], ss["p_model"])})
            rows.append({"source": "Sportsbook close (same games)",
                         **summarize(ss["home_win"], ss["p_spread"])})
        cs = pd.DataFrame(rows)[["source", "n", "log_loss", "brier", "accuracy"]]
        lines += [f"## This season so far ({cur - 1}-{cur - 2000})", "",
                  "The model is frozen at the start of the season: trained only on earlier "
                  "seasons, with ratings updated game by game as results come in.", "",
                  _fmt_table(cs), ""]

    # ---- does the model add information beyond the market?
    pred = pred.copy()
    pred["p_blend"] = blend_walk_forward(pred, ["p_spread", "p_model"])
    b = pred.dropna(subset=["p_blend"])
    blend_gain = log_loss(b["home_win"], b["p_spread"]) - log_loss(b["home_win"], b["p_blend"])
    lines += [
        "## Does the model add anything the market misses?",
        "",
        "Out-of-sample logistic blend of logit(market) and logit(model), refit each season on "
        "earlier seasons.",
        "",
        f"- Games: {len(b)}; market log loss {log_loss(b['home_win'], b['p_spread']):.4f}, "
        f"blend {log_loss(b['home_win'], b['p_blend']):.4f} "
        f"(improvement {blend_gain:+.4f}).",
        "- A positive improvement means the model carries information the closing line does "
        "not. Near zero means the market already prices everything the model knows.",
        "",
    ]

    # ---- Kalshi
    kal = pred.dropna(subset=["kalshi_p_home", "p_spread"])
    kal_summary = None
    if len(kal) >= 50:
        ks = summarize(kal["home_win"], kal["kalshi_p_home"])
        ms = summarize(kal["home_win"], kal["p_spread"])
        mo = summarize(kal["home_win"], kal["p_model"])
        kal_summary = ks
        diff = (kal["kalshi_p_home"] - kal["p_spread"])
        rel = pd.DataFrame(reliability(kal["home_win"], kal["kalshi_p_home"], bins=10),
                           columns=["kalshi_price", "home_win_rate", "games"])
        lines += [
            "## Kalshi pre-tip prices",
            "",
            "Kalshi P(home win) from the last 1-minute quote before tip (bid/ask mid, or last "
            "trade when the quote is wide), averaged across the two team markets.",
            "",
            _fmt_table(pd.DataFrame([
                {"source": "Kalshi pre-tip", **{k: ks[k] for k in
                                                ("n", "log_loss", "brier", "accuracy",
                                                 "calib_slope", "calib_intercept")}},
                {"source": "Sportsbook close", **{k: ms[k] for k in
                                                  ("n", "log_loss", "brier", "accuracy",
                                                   "calib_slope", "calib_intercept")}},
                {"source": "Our model", **{k: mo[k] for k in
                                           ("n", "log_loss", "brier", "accuracy",
                                            "calib_slope", "calib_intercept")}},
            ])),
            "",
            f"Kalshi minus sportsbook probability: mean {diff.mean():+.4f}, mean absolute "
            f"{diff.abs().mean():.4f}, 90th percentile absolute {diff.abs().quantile(.9):.4f}. "
            f"Median quoted spread at tip: {kal['kalshi_spread'].median():.3f}.",
            "",
            "Reliability of Kalshi prices (a favorite-longshot bias would show longshot bins "
            "winning less often than priced, favorites more often):",
            "",
            _fmt_table(rel),
            "",
        ]

    # ---- coefficients & players
    last = max(models)
    lines += [f"## Model coefficients (season {last} model)", "",
              _fmt_table(models[last].coef_table()), "",
              f"Residual sigma: {models[last].sigma_:.2f} points.", ""]
    if top_players is not None:
        lines += ["## Highest-rated players (latest ratings, points per 48 vs average)", "",
                  _fmt_table(top_players, "{:.2f}"), ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"pooled_model": pm, "pooled_market": pk, "gap": gap, "gate_logloss": gate_ll,
            "gate_calibration": gate_cal, "blend_gain": blend_gain, "kalshi": kal_summary}
