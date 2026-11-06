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
