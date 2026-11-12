"""Command line entry point: `uv run nba-engine --help`."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal

import typer

from nba_engine.config.settings import NBA_GAME_SERIES, get_settings

app = typer.Typer(add_completion=False, help="NBA x Kalshi prediction engine")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def markets(verbose: bool = typer.Option(False, "-v")) -> None:
    """Show upcoming/live NBA games matched to Kalshi markets with top of book and fees."""
    _setup_logging(verbose)
    from datetime import datetime, timedelta

    from nba_engine.data.espn import EspnClient
    from nba_engine.data.schedule import ET, match_games
    from nba_engine.kalshi.fees import order_fee
    from nba_engine.kalshi.orderbook import OrderBook
    from nba_engine.kalshi.rest import KalshiRest

    async def go() -> None:
        s = get_settings()
        async with KalshiRest(s.rest_base, max_rps=s.kalshi_max_rps) as rest:
            espn = EspnClient()
            try:
                today = datetime.now(ET).date()
                games = (await espn.scoreboard(today)
                         + await espn.scoreboard(today + timedelta(days=1)))
            finally:
                await espn.aclose()
            series = await rest.get_series(NBA_GAME_SERIES)
            events = await rest.get_events(NBA_GAME_SERIES, status="open")
            matched = [g for g in match_games(games, events) if g.game]
            books = await rest.get_orderbooks([t for g in matched for t in g.tickers])
        fee_type, mult = series.get("fee_type"), series.get("fee_multiplier", 1)
        typer.echo(f"Series {NBA_GAME_SERIES}: fee_type={fee_type} multiplier={mult}\n")
        hdr = (f"{'game':<12}{'tip (ET)':<16}{'state':<7}{'team':<6}{'bid':>6}{'ask':>6}"
               f"{'sprd':>6}{'taker fee/100':>15}")
        typer.echo(hdr)
        typer.echo("-" * len(hdr))
        for g in sorted(matched, key=lambda x: x.game.tip_utc):
            gm = g.game
            tip = gm.tip_utc.astimezone(ET).strftime("%a %m/%d %H:%M")
            state = gm.state if gm.state != "in" else f"Q{gm.period}"
            if not g.markets:
                typer.echo(f"{g.label:<12}{tip:<16}{state:<7}(no Kalshi market)")
                continue
            for team in (gm.away.code, gm.home.code):
                m = g.markets.get(team)
                if not m:
                    continue
                ob = OrderBook.from_rest(m["ticker"], books.get(m["ticker"], {}))
                bid, ask = ob.best_yes_bid(), ob.best_yes_ask()
                fee = (order_fee(ask, 100, maker=False, fee_type=fee_type, multiplier=mult)
                       if ask is not None else None)
                fmt = lambda x: f"{x:.2f}" if x is not None else "-"  # noqa: E731
                typer.echo(f"{g.label:<12}{tip:<16}{state:<7}{team:<6}{fmt(bid):>6}"
                           f"{fmt(ask):>6}{fmt(ob.spread()):>6}"
                           f"{('$' + fmt(fee)) if fee is not None else '-':>15}")

    asyncio.run(go())


@app.command()
def book(ticker: str, depth: int = 10) -> None:
    """Print the order book ladder for one market."""
    from nba_engine.kalshi.orderbook import OrderBook
    from nba_engine.kalshi.rest import KalshiRest

    async def go() -> None:
        s = get_settings()
        async with KalshiRest(s.rest_base) as rest:
            ob = OrderBook.from_rest(ticker, await rest.get_orderbook(ticker))
        typer.echo(f"{ticker}  mid={ob.mid()}  spread={ob.spread()}")
        typer.echo("  YES asks (buy YES)        YES bids (sell YES)")
        asks, bids = ob.asks("yes")[:depth], ob.bids("yes")[:depth]
        for i in range(max(len(asks), len(bids))):
            a = f"{asks[i][0]:.2f} x {asks[i][1]:>12,.2f}" if i < len(asks) else ""
            b = f"{bids[i][0]:.2f} x {bids[i][1]:>12,.2f}" if i < len(bids) else ""
            typer.echo(f"  {a:<26}{b}")

    asyncio.run(go())


@app.command()
def record(
    duration: float = typer.Option(0, help="Seconds to run (0 = until Ctrl+C)"),
    rest_only: bool = typer.Option(False, help="Force REST polling even with an API key"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Run the always-on recorder (Kalshi books/trades + ESPN play-by-play + odds)."""
    _setup_logging(verbose)
    from nba_engine.recorder.run import Recorder

    rec = Recorder(get_settings(), force_rest=rest_only)
    try:
        asyncio.run(rec.run(duration_s=duration or None))
    except KeyboardInterrupt:
        typer.echo("stopped")


@app.command()
def backfill(
    source: str = typer.Argument("all", help="espn | kalshi | all"),
    tables_only: bool = typer.Option(False, help="Rebuild tables from cache, no fetching"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Download history (ESPN games/box scores/lines, Kalshi settled markets/candles)."""
    _setup_logging(verbose)
    from nba_engine.history.espn_backfill import EspnBackfill
    from nba_engine.history.kalshi_backfill import KalshiBackfill
    from nba_engine.kalshi.rest import KalshiRest

    s = get_settings()

    async def go() -> None:
        if source in ("espn", "all"):
            eb = EspnBackfill(s.data_dir)
            if not tables_only:
                await eb.run()
            eb.build_tables()
        if source in ("kalshi", "all"):
            async with KalshiRest(s.rest_base, max_rps=s.kalshi_max_rps) as rest:
                await KalshiBackfill(rest, s.data_dir).run()

    asyncio.run(go())


@app.command()
def train(
    first_season: int = typer.Option(2017, help="First ESPN season year to evaluate"),
    report: str = typer.Option("reports/phase1_pregame.md", help="Report output path"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Build features, run the walk-forward pregame model and write the evaluation report."""
    _setup_logging(verbose)
    from pathlib import Path

    from nba_engine.models.pregame import build_dataset, load_history, walk_forward
    from nba_engine.models.report import build_report
    from nba_engine.models.tuning import load_params

    s = get_settings()
    hist = load_history(s.data_dir)
    if hist["games"].empty:
        typer.echo("No history found: run `nba-engine backfill` first.")
        raise typer.Exit(1)
    params = load_params()
    holder: list = []
    df = build_dataset(hist, player_kwargs=params["player"], team_kwargs=params["team"],
                       eff_kwargs=params["eff"], player_model_out=holder)
    pred, models = walk_forward(df, first_season=first_season,
                                gbm_weight=params["gbm_weight"])
    out = Path(s.data_dir) / "history" / "pregame_predictions.parquet"
    pred.drop(columns=[c for c in pred.columns if pred[c].dtype == object
                       and c not in ("game_id", "home", "away", "season_type", "venue_city")]
              ).to_parquet(out, index=False)
    pm = holder[0]
    summary = build_report(pred, models, df, pm.top_players(hist["player_box"]), Path(report))
    pdir = Path("reports/predictions")
    pdir.mkdir(parents=True, exist_ok=True)
    pm.top_players(hist["player_box"], k=30).to_csv(pdir / "player_ratings.csv", index=False)
    models[max(models)].coef_table().to_csv(pdir / "model_coefficients.csv", index=False)
    typer.echo(f"Report written to {report}")
    typer.echo(f"Model log loss {summary['pooled_model']['log_loss']:.4f} vs market "
               f"{summary['pooled_market']['log_loss']:.4f} (gap {summary['gap']:+.4f}); "
               f"calibration slope {summary['pooled_model']['calib_slope']:.3f}; "
               f"blend gain {summary['blend_gain']:+.4f}")


@app.command()
def backtest(
    report: str = typer.Option("reports/phase2_pregame_backtest.md", help="Report path"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Score fair-value variants against Kalshi and backtest the pregame strategies."""
    _setup_logging(verbose)
    from pathlib import Path

    from nba_engine.backtest.run import run

    res = run(get_settings().data_dir, Path(report))
    typer.echo(f"Report written to {report}")
    typer.echo(res["score"].to_string(index=False))


@app.command("build-pbp")
def build_pbp(verbose: bool = typer.Option(False, "-v")) -> None:
    """Extract play-by-play and lineup stints from the cached ESPN summaries."""
    _setup_logging(verbose)
    from pathlib import Path

    import pandas as pd

    from nba_engine.history.pbp import build_plays_table, build_stints, stint_quality

    s = get_settings()
    h = Path(s.data_dir) / "history"
    games = pd.read_parquet(h / "games.parquet")
    games = games[games["completed"]]
    plays = build_plays_table(Path(s.data_dir) / "cache" / "espn" / "summary",
                              list(games["game_id"]), h / "plays.parquet")
    stints = build_stints(plays, pd.read_parquet(h / "player_box.parquet"), games,
                          h / "stints.parquet")
    q = stint_quality(stints, games)
    typer.echo(f"{len(plays):,} plays, {len(stints):,} stints; mean valid lineup time "
               f"{q['valid_time'].mean():.1%}; stint points match final score in "
               f"{q['points_match'].mean():.1%} of games")


@app.command()
def ingame(
    report: str = typer.Option("reports/phase3_ingame.md", help="Report path"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Train/evaluate the in-game win-probability model and study Kalshi live prices."""
    _setup_logging(verbose)
    from pathlib import Path

    from nba_engine.backtest.ingame_run import run

    res = run(get_settings().data_dir, Path(report))
    pdir = Path("reports/predictions")
    pdir.mkdir(parents=True, exist_ok=True)
    res["by_phase"].to_csv(pdir / "ingame_by_phase.csv", index=False)
    res["kalshi"].to_csv(pdir / "ingame_vs_kalshi.csv", index=False)
    res["strategy"].to_csv(pdir / "ingame_fade_strategy.csv", index=False)
    typer.echo(f"Report written to {report}")
    typer.echo(res["by_phase"].round(4).to_string(index=False))
    typer.echo(res["kalshi"].round(4).to_string(index=False))
    typer.echo(res["strategy"].round(4).to_string(index=False))


@app.command()
def tune(verbose: bool = typer.Option(False, "-v")) -> None:
    """Pick model settings on validation seasons (2017-2022) and save them."""
    _setup_logging(verbose)
    from pathlib import Path

    from nba_engine.models.pregame import load_history
    from nba_engine.models.tuning import PARAMS_PATH
    from nba_engine.models.tuning import tune as run_tune

    s = get_settings()
    out = Path(s.data_dir) / "history" / "tuning_results.csv"
    params = run_tune(load_history(s.data_dir), out)
    typer.echo(f"Saved {PARAMS_PATH}: {params}")


@app.command()
def predict(
    days: int = typer.Option(7, help="Days ahead to predict"),
    out: str = typer.Option("reports/predictions/upcoming.csv", help="CSV output path"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Win probabilities for upcoming games (current rosters minus injured players),
    next to Kalshi's current price."""
    _setup_logging(verbose)
    from datetime import date, timedelta
    from pathlib import Path

    import pandas as pd

    from nba_engine.data.schedule import kalshi_event_ticker
    from nba_engine.models.pregame import load_history
    from nba_engine.models.tuning import load_params
    from nba_engine.predict.forecast import fetch_schedule, forecast_games, kalshi_open_prices
    from nba_engine.predict.lineups import fetch_injuries

    s = get_settings()
    hist = load_history(s.data_dir)
    now = pd.Timestamp.now(tz="UTC")
    sched = fetch_schedule(s.data_dir, date.today(), date.today() + timedelta(days=days))
    fut = sched[~sched["completed"] & (sched["tip_utc"] > now)]
    if fut.empty:
        typer.echo("No upcoming games found.")
        raise typer.Exit(0)
    inj = fetch_injuries()
    pred, model = forecast_games(hist, fut, now, load_params(), injuries=inj)
    pred["event_ticker"] = [kalshi_event_ticker(a, h, t) for a, h, t in
                            zip(pred["away"], pred["home"], pred["tip_utc"], strict=True)]
    try:
        k = kalshi_open_prices()
        km = k.merge(pred[["event_ticker", "home", "away"]], on="event_ticker")
        home_mid = km[km["team"] == km["home"]].set_index("event_ticker")["mid"]
        away_mid = km[km["team"] == km["away"]].set_index("event_ticker")["mid"]
        pred["kalshi_p_home"] = pd.concat([home_mid, 1 - away_mid], axis=1).mean(axis=1
                                                                                ).reindex(
            pred["event_ticker"]).to_numpy()
    except Exception as e:  # noqa: BLE001 - comparison column is optional
        typer.echo(f"(Kalshi prices unavailable: {e})")
        pred["kalshi_p_home"] = float("nan")
    out_injured = inj[inj["status"].isin({"Out", "Doubtful"})].groupby("team")["player"].apply(
        lambda x: ", ".join(x.dropna()[:4]))
    pred["home_out"] = pred["home"].map(out_injured).fillna("")
    pred["away_out"] = pred["away"].map(out_injured).fillna("")
    cols = ["tip_utc", "away", "home", "mu", "p_home", "kalshi_p_home", "away_out", "home_out"]
    res = pred[cols].sort_values("tip_utc")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        typer.echo(res.round({"mu": 1, "p_home": 3, "kalshi_p_home": 3}).to_string(index=False))
    typer.echo(f"\nSaved {out} (model sigma {model.sigma_:.1f} pts)")


@app.command()
def project(
    sims: int = typer.Option(20000, help="Monte Carlo simulations"),
    out: str = typer.Option("reports/predictions/season_projection.csv", help="CSV output"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Simulate the rest of the current season: wins, playoff, seed and title odds."""
    _setup_logging(verbose)
    from datetime import date
    from pathlib import Path

    import pandas as pd

    from nba_engine.models.pregame import load_history
    from nba_engine.models.tuning import load_params
    from nba_engine.predict.forecast import fetch_schedule, project_season
    from nba_engine.predict.lineups import fetch_injuries
    from nba_engine.predict.season import SimConfig

    s = get_settings()
    hist = load_history(s.data_dir)
    now = pd.Timestamp.now(tz="UTC")
    season = int(hist["games"]["season"].max())
    sched = fetch_schedule(s.data_dir, date(season - 1, 10, 1), date(season, 4, 30))
    table, _ = project_season(hist, sched, season, now, load_params(),
                              injuries=fetch_injuries(), cfg=SimConfig(n_sims=sims))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    show = ["team", "wins_now", "losses_now", "proj_wins", "wins_p10", "wins_p90",
            "p_playoffs", "p_seed1", "p_finals", "p_title"]
    with pd.option_context("display.width", 200):
        typer.echo(table[show].round(3).to_string(index=False))
    typer.echo(f"\nSaved {out}")


@app.command("validate-projections")
def validate_projections(
    seasons: str = typer.Option("2023,2024,2025,2026", help="ESPN season years"),
    month_day: str = typer.Option("11-12", help="Cutoff date within each season (MM-DD)"),
    out: str = typer.Option("reports/predictions/projection_backtest.csv", help="CSV output"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Backtest season projections: project past seasons from the same calendar date and
    compare with final records."""
    _setup_logging(verbose)
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from nba_engine.models.pregame import load_history
    from nba_engine.models.tuning import load_params
    from nba_engine.predict.forecast import backtest_projection
    from nba_engine.predict.season import SimConfig

    s = get_settings()
    hist = load_history(s.data_dir)
    params = load_params()
    mm, dd = (int(x) for x in month_day.split("-"))
    frames = []
    for yr in (int(x) for x in seasons.split(",")):
        cutoff = pd.Timestamp(year=yr - 1, month=mm, day=dd, tz="UTC")
        frames.append(backtest_projection(hist, yr, cutoff, params, SimConfig(n_sims=5000)))
    t = pd.concat(frames, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    for col in ("proj_wins", "naive_pace_wins", "naive_last_season_wins"):
        err = t[col] - t["actual_wins"]
        typer.echo(f"{col:<24} MAE {err.abs().mean():5.2f}  RMSE {np.sqrt((err**2).mean()):5.2f}")
    inside = ((t["actual_wins"] >= t["wins_p10"]) & (t["actual_wins"] <= t["wins_p90"])).mean()
    typer.echo(f"80% interval coverage: {inside:.0%}")
    typer.echo(f"Saved {out}")


@app.command()
def figures() -> None:
    """Render report figures to reports/figures/."""
    from nba_engine.viz import make_all

    for path in make_all(get_settings().data_dir):
        typer.echo(f"wrote {path}")


@app.command("check-auth")
def check_auth() -> None:
    """Verify the Kalshi API key by fetching your balance (read-only)."""
    from nba_engine.kalshi.auth import KalshiSigner
    from nba_engine.kalshi.rest import KalshiRest

    s = get_settings()
    if not s.has_kalshi_auth:
        typer.echo("No key configured: set NBA_KALSHI_KEY_ID and NBA_KALSHI_PRIVATE_KEY_PATH "
                   "in .env")
        raise typer.Exit(1)

    async def go() -> None:
        signer = KalshiSigner.from_file(s.kalshi_key_id, s.kalshi_private_key_path)
        async with KalshiRest(s.rest_base, signer=signer) as rest:
            bal = await rest.get_balance()
        cents = bal.get("balance")
        typer.echo(f"Auth OK ({s.kalshi_env}). Balance: "
                   f"{'$' + str(Decimal(cents) / 100) if cents is not None else bal}")

    asyncio.run(go())


if __name__ == "__main__":
    app()
