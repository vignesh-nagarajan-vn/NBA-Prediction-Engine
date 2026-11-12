"""Static figures for the reports (matplotlib, PNG).

Palette and mark rules follow the project's data-viz conventions: a fixed categorical order
(blue = our model, orange = market / Kalshi, aqua = third series), thin 2px lines, >= 8px
markers, recessive grid, legends whenever there are two or more series, and text in ink
colors rather than series colors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nba_engine.models.metrics import log_loss, reliability  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#a8a7a0"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "lines.linewidth": 2, "lines.markersize": 7, "text.color": INK,
    })


def _save(fig, out: Path, name: str, note: str | None = None) -> Path:
    if note:
        fig.text(0.01, -0.01, note, fontsize=8, color=INK2, ha="left", va="top")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- pregame ----------------------------------------------------------------------------
def accuracy_by_season(pred: pd.DataFrame, out: Path, holdout_from: int = 2023) -> Path:
    rows = []
    for s, d in pred.groupby("season"):
        d = d.dropna(subset=["p_spread"])
        if len(d) < 50:
            continue
        rows.append((s, log_loss(d["home_win"], d["p_model"]),
                     log_loss(d["home_win"], d["p_spread"]), len(d)))
    t = pd.DataFrame(rows, columns=["season", "model", "market", "n"])
    present = list(t["season"])
    # Reindex to every season so seasons without closing lines leave a gap in the lines.
    t = t.set_index("season").reindex(range(min(present), max(present) + 1)).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    lo = t["season"].min() - 0.5
    ax.axvspan(holdout_from - 0.5, t["season"].max() + 0.5, color="#f0efec", zorder=0)
    ax.text(holdout_from - 0.35, t[["model", "market"]].max().max() + 0.002, "holdout",
            color=INK2, fontsize=9)
    ax.plot(t["season"], t["model"], "-o", color=BLUE, label="Our model")
    ax.plot(t["season"], t["market"], "-o", color=ORANGE, label="Sportsbook closing line")
    ax.set_xlim(lo, t["season"].max() + 0.5)
    ax.set_xticks(present)
    ax.set_xticklabels([f"{s - 1 - 2000:02d}-{s - 2000:02d}" for s in present])
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Pregame accuracy by season")
    ax.legend(loc="upper left")
    return _save(fig, out, "accuracy_by_season",
                 "Each season predicted by a model trained only on earlier seasons. "
                 "Seasons without ESPN closing lines (2023-24, 2024-25) are omitted.")


def calibration_plot(pred: pd.DataFrame, out: Path, holdout_from: int = 2023) -> Path:
    d = pred[pred["season"] >= holdout_from]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1, label="Perfect calibration")
    r = np.array(reliability(d["home_win"], d["p_model"], 10))
    ax.plot(r[:, 0], r[:, 1], "-o", color=BLUE, label=f"Our model ({len(d):,} games)")
    k = d.dropna(subset=["kalshi_p_home"])
    if len(k) > 200:
        rk = np.array(reliability(k["home_win"], k["kalshi_p_home"], 10))
        ax.plot(rk[:, 0], rk[:, 1], "-o", color=ORANGE,
                label=f"Kalshi pre-tip ({len(k):,} games)")
    ax.set_xlabel("Predicted home win probability")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Calibration, 2022-23 onward")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    return _save(fig, out, "calibration")


def this_season(pred: pd.DataFrame, season: int, out: Path) -> Path:
    d = pred[pred["season"] == season].sort_values("tip_utc")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(1, len(d) + 1)

    def cum_ll(p):
        y = d["home_win"].to_numpy(float)
        p = np.clip(p.to_numpy(float), 1e-6, 1 - 1e-6)
        return np.cumsum(-(y * np.log(p) + (1 - y) * np.log(1 - p))) / x

    ax.plot(x, cum_ll(d["p_model"]), color=BLUE, label="Our model")
    k = d["kalshi_p_home"].notna()
    if k.sum() > 20:
        dk = d[k]
        xk = np.arange(1, len(dk) + 1)
        y = dk["home_win"].to_numpy(float)
        p = np.clip(dk["kalshi_p_home"].to_numpy(float), 1e-6, 1 - 1e-6)
        ax.plot(d.index.get_indexer(dk.index) + 1,
                np.cumsum(-(y * np.log(p) + (1 - y) * np.log(1 - p))) / xk,
                color=ORANGE, label="Kalshi pre-tip")
    s = d["p_spread"].notna()
    if s.sum() > 20:
        ds = d[s]
        xs = np.arange(1, len(ds) + 1)
        y = ds["home_win"].to_numpy(float)
        p = np.clip(ds["p_spread"].to_numpy(float), 1e-6, 1 - 1e-6)
        ax.plot(d.index.get_indexer(ds.index) + 1,
                np.cumsum(-(y * np.log(p) + (1 - y) * np.log(1 - p))) / xs,
                color=AQUA, label="Sportsbook closing line")
    ax.axhline(np.log(2), color=NEUTRAL, linewidth=1)
    ax.text(2, np.log(2) + 0.004, "coin flip", ha="left", color=INK2, fontsize=8)
    ax.set_ylim(0.45, 0.75)
    ax.set_xlabel("Games played this season")
    ax.set_ylabel("Running log loss")
    acc = ((d["p_model"] > 0.5) == (d["home_win"] == 1)).mean()
    ax.set_title(f"{season - 1}-{season - 2000} season so far: {len(d)} games, "
                 f"model picked {acc:.0%} of winners")
    ax.legend(loc="upper right")
    return _save(fig, out, "this_season",
                 "Model frozen at the start of the season (trained only on prior seasons).")


FEATURE_GROUPS = {
    "Team ratings (margin, efficiency, Elo)": ["rating_margin_pred", "eff_margin_pred",
                                               "elo_diff"],
    "Lineup strength (player ratings)": ["player_strength_diff"],
    "Regulars missing": ["home_out_regulars", "away_out_regulars"],
    "Back-to-backs": ["home_b2b", "away_b2b"],
    "Playoff game": ["playoffs"],
    "3 games in 4 nights": ["home_3in4", "away_3in4"],
    "Rest days": ["rest_diff"],
    "Time-zone shifts": ["home_tz_abs", "away_tz_abs"],
    "Altitude (DEN/UTA home)": ["altitude_home"],
    "Travel distance": ["home_travel_k", "away_travel_k"],
    "Home court": ["home_court"],
}


def feature_weights(coef: pd.DataFrame, X: pd.DataFrame, out: Path) -> Path:
    """How much each group of inputs moves the predicted margin: the standard deviation of
    the group's combined contribution across games. Correlated features (the three team
    ratings) are grouped so they don't appear to fight each other."""
    w = dict(zip(coef["feature"], coef["coef"], strict=True))
    rows = []
    for name, feats in FEATURE_GROUPS.items():
        feats = [f for f in feats if f in w and f in X]
        if not feats:
            continue
        contrib = sum(X[f].fillna(0) * w[f] for f in feats)
        rows.append((name, float(contrib.std())))
    t = pd.DataFrame(rows, columns=["group", "impact"]).sort_values("impact")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(t["group"], t["impact"], color=BLUE, height=0.55)
    for yv, v in enumerate(t["impact"]):
        ax.text(v + 0.05, yv, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Typical swing in predicted home margin, points (sd across games)")
    ax.set_title("What drives the pregame prediction")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "feature_weights", "Linear part of the ensemble, latest model. "
                 "Related inputs are grouped because they share credit with each other.")


def top_players(players: pd.DataFrame, out: Path, k: int = 20) -> Path:
    p = players.head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(p["player"], p["rating"], color=BLUE, height=0.6)
    ax.set_xlabel("Rating: points per 48 minutes vs an average player")
    ax.set_title("Top player ratings right now")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "top_players", "Lineup-stint plus-minus with a box-score prior, "
                 "time-weighted toward recent games.")


# ---- season ------------------------------------------------------------------------------
def season_projection(table: pd.DataFrame, out: Path, season_label: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharex=True)
    for ax, conf in zip(axes, ("East", "West"), strict=True):
        t = table[table["conference"] == conf].sort_values("proj_wins")
        y = np.arange(len(t))
        ax.hlines(y, t["wins_p10"], t["wins_p90"], color="#b7d3f6", linewidth=6)
        ax.plot(t["proj_wins"], y, "o", color=BLUE, label="Projected wins")
        ax.plot(t["wins_now"], y, "o", mfc=SURFACE, mec=INK2, label="Wins so far")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.team}  {r.p_playoffs:.0%}" for r in t.itertuples()])
        ax.set_title(f"{conf}")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Wins (bar = 80% range)")
    axes[0].legend(loc="lower right")
    fig.suptitle(f"{season_label} projected final wins and playoff odds", x=0.01, ha="left",
                 fontsize=14, fontweight="semibold", color=INK)
    return _save(fig, out, "season_projection", "Percent next to each team = chance of "
                 "making the playoffs (including via the play-in). 20,000 simulations.")


def title_odds(table: pd.DataFrame, out: Path, k: int = 12) -> Path:
    t = table.sort_values("p_title", ascending=False).head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(t["team"], t["p_title"] * 100, color=BLUE, height=0.6)
    for yv, v in enumerate(t["p_title"] * 100):
        ax.text(v + 0.3, yv, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Chance of winning the title, %")
    ax.set_title("Championship odds")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "title_odds")


def projection_backtest(bt: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(wspace=0.55)
    ax = axes[0]
    lim = (10, 70)
    ax.plot(lim, lim, color=NEUTRAL, linewidth=1)
    ax.plot(bt["proj_wins"], bt["actual_wins"], "o", color=BLUE, alpha=0.75,
            mec=SURFACE, mew=1)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Projected wins (from Nov 12)")
    ax.set_ylabel("Actual wins")
    ax.set_title("Season projections vs what happened")
    ax2 = axes[1]
    methods = [("proj_wins", "Our projection"), ("naive_pace_wins", "Current win % pace"),
               ("naive_last_season_wins", "Last season's record")]
    mae = [float((bt[c] - bt["actual_wins"]).abs().mean()) for c, _ in methods]
    ax2.barh([m for _, m in methods][::-1], mae[::-1], color=[NEUTRAL, NEUTRAL, BLUE],
             height=0.45)
    for yv, v in enumerate(mae[::-1]):
        ax2.text(v + 0.1, yv, f"{v:.1f}", va="center", fontsize=9)
    ax2.set_xlabel("Average miss, wins")
    ax2.set_title("Error vs simple baselines")
    ax2.grid(axis="y", visible=False)
    seasons = ", ".join(f"{s - 1}-{s - 2000}" for s in sorted(bt["season"].unique()))
    return _save(fig, out, "projection_backtest", f"Seasons projected from Nov 12 using only "
                 f"data available then: {seasons}.")


# ---- in-game -----------------------------------------------------------------------------
def ingame_accuracy(by_phase: pd.DataFrame, kal: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    phases = [p for p in ("Q1", "Q2", "Q3", "Q4") if p in set(kal["phase"])]
    k = kal.set_index("phase").reindex(phases)
    x = np.arange(len(phases))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, k["model_logloss"], w, color=BLUE, label="Our model")
    ax.bar(x + w / 2 + 0.01, k["kalshi_logloss"], w, color=ORANGE, label="Kalshi live price")
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("In-game accuracy by quarter, same moments")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    return _save(fig, out, "ingame_accuracy",
                 "Game state taken at each Kalshi minute; later quarters are easier for both.")


def ingame_calibration(states: pd.DataFrame, out: Path) -> Path:
    live = states[states["t_left"] > 0]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1, label="Perfect calibration")
    r = np.array(reliability(live["home_win"], live["p_model"], 20))
    ax.plot(r[:, 0], r[:, 1], "-o", color=BLUE, markersize=5, label="Our in-game model")
    ax.set_xlabel("Predicted home win probability")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("In-game calibration, 2022-23 onward")
    ax.legend(loc="upper left")
    return _save(fig, out, "ingame_calibration")


def example_game(minutes: pd.DataFrame, games: pd.DataFrame, out: Path) -> Path | None:
    if minutes.empty:
        return None
    swings = minutes.groupby("game_id")["p_kalshi"].agg(lambda s: s.max() - s.min())
    gid = swings.sort_values().index[-1]
    m = minutes[minutes["game_id"] == gid].sort_values("end_ts")
    g = games.set_index("game_id").loc[gid]
    t0 = m["end_ts"].min()
    x = (m["end_ts"] - t0) / 60
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, m["p_model"] * 100, color=BLUE, label="Our model")
    ax.plot(x, m["p_kalshi"] * 100, color=ORANGE, label="Kalshi")
    ax.axhline(50, color=NEUTRAL, linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Minutes since tip (wall clock)")
    ax.set_ylabel(f"{g['home']} win probability, %")
    winner = g["home"] if g["home_score"] > g["away_score"] else g["away"]
    ax.set_title(f"{g['away']} @ {g['home']}, {pd.Timestamp(g['tip_utc']):%b %d %Y}: "
                 f"{int(g['away_score'])}-{int(g['home_score'])}, {winner} won")
    ax.legend(loc="lower left")
    return _save(fig, out, "example_game", "The game with the largest Kalshi swing in the data.")


def upcoming_games(up: pd.DataFrame, out: Path, k: int = 16) -> Path | None:
    if up.empty:
        return None
    u = up.sort_values("tip_utc").head(k).iloc[::-1]
    et = [pd.Timestamp(t).tz_convert("America/New_York") for t in u["tip_utc"]]
    labels = [f"{a} @ {h}  {d:%a %m/%d}" for a, h, d in zip(u["away"], u["home"], et,
                                                           strict=True)]
    y = np.arange(len(u))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(u) + 1.6))
    ax.axvline(50, color=NEUTRAL, linewidth=1)
    ax.plot(u["p_home"] * 100, y, "o", color=BLUE, label="Our model")
    if u["kalshi_p_home"].notna().any():
        ax.plot(u["kalshi_p_home"] * 100, y, "o", color=ORANGE, label="Kalshi now")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Home team win probability, %")
    ax.set_title("Upcoming games")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, out, "upcoming_games", "Injured players listed as Out are removed from "
                 "the projected lineup.")


def make_all(data_dir: Path | str, reports: Path | str = "reports") -> list[Path]:
    _style()
    data_dir, reports = Path(data_dir), Path(reports)
    h = data_dir / "history"
    pdir = reports / "predictions"
    out = reports / "figures"
    made: list[Path | None] = []
    pred = pd.read_parquet(h / "pregame_predictions.parquet")
    pred["tip_utc"] = pd.to_datetime(pred["tip_utc"], utc=True)
    made.append(accuracy_by_season(pred, out))
    made.append(calibration_plot(pred, out))
    made.append(this_season(pred, int(pred["season"].max()), out))
    if (pdir / "model_coefficients.csv").exists():
        made.append(feature_weights(pd.read_csv(pdir / "model_coefficients.csv"), pred, out))
    if (pdir / "player_ratings.csv").exists():
        made.append(top_players(pd.read_csv(pdir / "player_ratings.csv"), out))
    season = int(pred["season"].max())
    label = f"{season - 1}-{season - 2000}"
    if (pdir / "season_projection.csv").exists():
        t = pd.read_csv(pdir / "season_projection.csv")
        made.append(season_projection(t, out, label))
        made.append(title_odds(t, out))
    if (pdir / "projection_backtest.csv").exists():
        made.append(projection_backtest(pd.read_csv(pdir / "projection_backtest.csv"), out))
    if (pdir / "ingame_by_phase.csv").exists() and (pdir / "ingame_vs_kalshi.csv").exists():
        made.append(ingame_accuracy(pd.read_csv(pdir / "ingame_by_phase.csv"),
                                    pd.read_csv(pdir / "ingame_vs_kalshi.csv"), out))
    if (h / "ingame_test_states.parquet").exists():
        made.append(ingame_calibration(pd.read_parquet(h / "ingame_test_states.parquet"), out))
    if (h / "ingame_kalshi_minutes.parquet").exists():
        games = pd.read_parquet(h / "games.parquet")
        made.append(example_game(pd.read_parquet(h / "ingame_kalshi_minutes.parquet"), games,
                                 out))
    if (pdir / "upcoming.csv").exists():
        up = pd.read_csv(pdir / "upcoming.csv")
        up["tip_utc"] = pd.to_datetime(up["tip_utc"], utc=True)
        made.append(upcoming_games(up, out))
    return [m for m in made if m is not None]
