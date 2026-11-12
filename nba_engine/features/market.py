"""Market prices as benchmarks: sportsbook closing lines (ESPN pickcenter) and Kalshi pre-tip
prices for the same games."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_engine.data.schedule import ET, parse_event_ticker


def american_to_prob(ml: pd.Series) -> pd.Series:
    ml = pd.to_numeric(ml, errors="coerce")
    return pd.Series(np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100)), index=ml.index)


def closing_lines(odds: pd.DataFrame) -> pd.DataFrame:
    """Per game: median home spread across providers, and median de-vigged moneyline
    probability (multiplicative de-vig) where moneylines exist."""
    o = odds.copy()
    o["spread"] = pd.to_numeric(o["spread"], errors="coerce")
    ph, pa = american_to_prob(o["home_ml"]), american_to_prob(o["away_ml"])
    o["ml_p_home"] = ph / (ph + pa)
    agg = o.groupby("game_id").agg(home_spread=("spread", "median"),
                                   ml_p_home=("ml_p_home", "median"),
                                   n_providers=("provider", "nunique"))
    return agg.reset_index()


def kalshi_pretip(markets: pd.DataFrame, candles: pd.DataFrame, games: pd.DataFrame,
                  max_staleness_min: int = 360, lead_min: int = 0) -> pd.DataFrame:
    """Kalshi implied P(home win) from the last 1-minute candle ending at or before
    tip - lead_min. Uses the bid/ask mid when the quote is sane, else the last trade, and
    averages the home market with 1 - away market when both exist."""
    if markets.empty or candles.empty:
        return pd.DataFrame(columns=["game_id", "kalshi_p_home", "kalshi_spread"])
    g = games.copy()
    g["et_day"] = g["tip_utc"].dt.tz_convert(ET).dt.strftime("%y%b%d").str.upper()
    key = {(r.et_day, r.away, r.home): (r.game_id, int(r.tip_utc.timestamp()))
           for r in g.itertuples(index=False)}
    c = candles.copy()
    mid = (c["yes_bid"] + c["yes_ask"]) / 2
    sane = (c["yes_ask"] > c["yes_bid"]) & (c["yes_ask"] - c["yes_bid"] <= 0.10)
    c["px"] = np.where(sane, mid, c["last"])
    c["sprd"] = np.where(sane, c["yes_ask"] - c["yes_bid"], np.nan)
    c = c.sort_values(["ticker", "end_ts"])
    by_ticker = {t: df for t, df in c.groupby("ticker", sort=False)}
    rows = {}
    for m in markets.itertuples(index=False):
        try:
            day, away, home = parse_event_ticker(m.event_ticker)
        except ValueError:
            continue
        hit = key.get((day, away, home))
        cdf = by_ticker.get(m.ticker)
        if hit is None or cdf is None:
            continue
        gid, tip = hit
        cutoff = tip - lead_min * 60
        before = cdf[(cdf["end_ts"] <= cutoff) & (cdf["end_ts"] >= cutoff - max_staleness_min * 60)]
        before = before.dropna(subset=["px"])
        if before.empty:
            continue
        last = before.iloc[-1]
        p_home = last["px"] if m.team == home else 1 - last["px"]
        rows.setdefault(gid, []).append((p_home, last["sprd"]))
    out = [{"game_id": gid, "kalshi_p_home": float(np.mean([v[0] for v in vals])),
            "kalshi_spread": float(np.nanmean([v[1] for v in vals]))
            if any(pd.notna(v[1]) for v in vals) else np.nan}
           for gid, vals in rows.items()]
    return pd.DataFrame(out, columns=["game_id", "kalshi_p_home", "kalshi_spread"])
