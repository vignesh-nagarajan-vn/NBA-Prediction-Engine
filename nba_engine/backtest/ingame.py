"""In-game study: our win-probability model vs Kalshi's live prices, and the overreaction
("fade the swing") strategy S3.

Kalshi minute candles are matched to the ESPN game state by wall-clock time. For accuracy we
use the state as of the candle (what a fast observer knew). For trading we use the state as
of `feed_delay_s` earlier (our free feed's lag) and only trade when that delayed state is a
dead ball, so the game was frozen and the stale feed was still accurate.

Strategy: when |model - Kalshi| exceeds a threshold at a dead ball, buy the side the model
favors at the ASK, then sell at the BID `hold_min` minutes later (or hold to settlement if the
game ends first). Taker fees on both legs; one open position per game at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_engine.data.live_feed import is_dead_ball
from nba_engine.data.schedule import ET, parse_event_ticker
from nba_engine.kalshi.fees import order_fee


def kalshi_game_minutes(markets: pd.DataFrame, candles: pd.DataFrame, games: pd.DataFrame
                        ) -> pd.DataFrame:
    """Per game minute after tip: home/away market bid/ask closes and implied P(home)."""
    g = games[games["completed"]].copy()
    g["tip_utc"] = pd.to_datetime(g["tip_utc"], utc=True)
    g["et_day"] = g["tip_utc"].dt.tz_convert(ET).dt.strftime("%y%b%d").str.upper()
    key = {(r.et_day, r.away, r.home): r for r in g.itertuples(index=False)}
    mk = markets.copy()
    side = {}
    for r in mk.itertuples(index=False):
        try:
            day, away, home = parse_event_ticker(r.event_ticker)
        except ValueError:
            continue
        gm = key.get((day, away, home))
        if gm is None:
            continue
        side[r.ticker] = (gm.game_id, "home" if r.team == home else "away",
                          int(gm.tip_utc.timestamp()))
    c = candles[candles["ticker"].isin(side)].copy()
    c["game_id"] = c["ticker"].map(lambda t: side[t][0])
    c["side"] = c["ticker"].map(lambda t: side[t][1])
    c["tip_ts"] = c["ticker"].map(lambda t: side[t][2])
    c = c[(c["end_ts"] >= c["tip_ts"]) & (c["end_ts"] <= c["tip_ts"] + 4 * 3600)]
    w = c.pivot_table(index=["game_id", "end_ts"], columns="side",
                      values=["yes_bid", "yes_ask"], aggfunc="first")
    w.columns = [f"{s}_{v}" for v, s in w.columns]
    w = w.reset_index()
    for s in ("home", "away"):
        for v in ("yes_bid", "yes_ask"):
            col = f"{s}_{v}"
            if col not in w:
                w[col] = np.nan
    ok_h = (w["home_yes_ask"] > w["home_yes_bid"]) & (w["home_yes_ask"] - w["home_yes_bid"] <= .06)
    ok_a = (w["away_yes_ask"] > w["away_yes_bid"]) & (w["away_yes_ask"] - w["away_yes_bid"] <= .06)
    ph = ((w["home_yes_bid"] + w["home_yes_ask"]) / 2).where(ok_h)
    pa = (1 - (w["away_yes_bid"] + w["away_yes_ask"]) / 2).where(ok_a)
    w["p_kalshi"] = pd.concat([ph, pa], axis=1).mean(axis=1)
    return w.dropna(subset=["p_kalshi"]).sort_values(["game_id", "end_ts"])


def attach_states(minutes: pd.DataFrame, states: pd.DataFrame, delay_s: int = 0
                  ) -> pd.DataFrame:
    """For each Kalshi minute, the latest play whose wall clock is <= end_ts - delay_s."""
    s = states.dropna(subset=["wallclock"]).copy()
    wc = pd.to_datetime(s["wallclock"], utc=True, errors="coerce")
    # Resolution-independent Unix seconds (pandas 3 may store datetimes as s/ms/us).
    s["wc_ts"] = (wc - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    s = s.dropna(subset=["wc_ts"])
    s["wc_ts"] = s["wc_ts"].astype("int64")
    s = s.sort_values("wc_ts")
    m = minutes.copy()
    m["lookup_ts"] = m["end_ts"] - delay_s
    m = m.sort_values("lookup_ts")
    cols = ["game_id", "wc_ts", "d", "t_left", "mu", "poss", "type", "home_win", "period",
            "p_model", "season"]
    out = pd.merge_asof(m, s[cols], left_on="lookup_ts", right_on="wc_ts", by="game_id",
                        direction="backward", tolerance=600)
    return out.dropna(subset=["p_model"]).sort_values(["game_id", "end_ts"])


@dataclass
class FadeConfig:
    threshold: float = 0.08
    hold_min: int = 5
    contracts: int = 100
    delay_s: int = 30
    min_t_left: float = 120.0  # never enter in the final 2 minutes


def _fee(p: float, n: int) -> float:
    return float(order_fee(round(min(max(p, 0.0), 1.0), 4), n, maker=False))


def simulate_fade(m: pd.DataFrame, cfg: FadeConfig) -> pd.DataFrame:
    """`m` = minutes with delayed states attached (attach_states with delay)."""
    trades = []
    for gid, g in m.groupby("game_id", sort=False):
        g = g.reset_index(drop=True)
        i = 0
        while i < len(g):
            r = g.iloc[i]
            gap = r["p_model"] - r["p_kalshi"]
            if (abs(gap) < cfg.threshold or r["t_left"] < cfg.min_t_left
                    or not is_dead_ball(str(r["type"]))):
                i += 1
                continue
            side = "home" if gap > 0 else "away"
            ask = r[f"{side}_yes_ask"]
            if not np.isfinite(ask) or ask <= 0 or ask >= 1:
                i += 1
                continue
            j = min(i + cfg.hold_min, len(g) - 1)
            exit_row = g.iloc[j]
            n = cfg.contracts
            cost = ask * n + _fee(ask, n)
            settled = j == len(g) - 1 and exit_row["t_left"] <= 0
            if settled:
                won = exit_row["home_win"] == 1 if side == "home" else exit_row["home_win"] == 0
                proceeds = n * float(won)
            else:
                bid = exit_row[f"{side}_yes_bid"]
                if not np.isfinite(bid):
                    i += 1
                    continue
                proceeds = bid * n - _fee(bid, n)
            p_side_k = r["p_kalshi"] if side == "home" else 1 - r["p_kalshi"]
            p_side_m = r["p_model"] if side == "home" else 1 - r["p_model"]
            later = exit_row["p_kalshi"] if side == "home" else 1 - exit_row["p_kalshi"]
            trades.append({"game_id": gid, "season": r["season"], "side": side,
                           "gap": abs(gap), "entry_ask": ask, "p_kalshi": p_side_k,
                           "p_model": p_side_m, "kalshi_move": later - p_side_k,
                           "pnl": proceeds - cost, "cost": cost, "settled": settled,
                           "t_left": r["t_left"]})
            i = j + 1
    return pd.DataFrame(trades)
