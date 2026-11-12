# NBA-Prediction-Engine

An engine that estimates fair, sentiment-free win probabilities for NBA games and trades the
gap against Kalshi's `KXNBAGAME` markets. The full roadmap is in [docs/PLAN.md](docs/PLAN.md).

**Status: Phases 0-3 built.** The pregame and in-game models are validated on held-out
seasons and the current season. No trading strategy has passed its gate, so nothing trades.
Start with [reports/BRIEF.md](reports/BRIEF.md).

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # optional: add a Kalshi API key for WebSocket mode
```

## Commands

```bash
uv run nba-engine markets              # today's/tomorrow's games, Kalshi top of book, taker fees
uv run nba-engine book KXNBAGAME-26NOV06PORLAL-LAL
uv run nba-engine record               # always-on recorder (Ctrl+C to stop)
uv run nba-engine check-auth           # verify a configured API key (read-only)
```

## History and models

```bash
uv run nba-engine backfill espn     # 2014-15 to today: games, box scores, closing lines (~1.5h first run)
uv run nba-engine backfill kalshi   # every settled KXNBAGAME market + 1-minute candles (~25 min)
uv run nba-engine build-pbp         # play-by-play + lineup stints from the cached summaries (~10 min)
uv run nba-engine tune              # pick settings on 2016-17..2021-22 only (~45 min)
uv run nba-engine train             # walk-forward pregame model + reports/phase1_pregame.md
uv run nba-engine backtest          # pregame fair value vs Kalshi + reports/phase2_pregame_backtest.md
uv run nba-engine ingame            # in-game win probability vs Kalshi + reports/phase3_ingame.md
```

Raw responses are cached under `data/cache/` so reruns only fetch what is new
(`backfill espn --tables-only` rebuilds the tables without any network calls).

The **pregame model** predicts the home margin and turns it into a win probability. Its inputs:

- **Lineup strength:** player ratings from five-on-five lineup stints reconstructed from
  play-by-play (who was actually on the floor together), shrunk toward a box-score prior.
  Injured or resting players drop out of the projected lineup.
- **Team ratings:** Elo, decay-weighted margin ratings and pace-adjusted offense/defense
  efficiency, all updated game by game.
- **Schedule context:** rest, back-to-backs, 3-in-4, travel distance, time-zone shifts,
  altitude, playoffs.

A ridge regression and a small gradient-boosted tree model (pure numpy) are averaged. Each
season is predicted by a model trained only on earlier seasons, and settings were tuned on
2016-17 to 2021-22 only, so 2022-23 onward is a true holdout.

The **in-game model** is a fitted version of the random-walk win-probability formula
(score margin, time left, possession, pregame expected margin), trained on play-by-play.

## Forecasts

```bash
uv run nba-engine predict                # next 7 days: model vs Kalshi, injured players removed
uv run nba-engine project                # rest-of-season simulation: wins, seeds, playoff and title odds
uv run nba-engine validate-projections   # backtest projections from Nov 12 of past seasons
uv run nba-engine figures                # charts to reports/figures/
```

Outputs land in `reports/predictions/` (CSV) and `reports/figures/` (PNG). The brief with
the headline results is [reports/BRIEF.md](reports/BRIEF.md).

## Recorder

`record` writes Parquet files under `data/raw/<stream>/date=YYYY-MM-DD/`:

| Stream | What |
|---|---|
| `kalshi_books` | Order-book snapshots (REST mode): every 2s for live/near-tip games, 30s otherwise, unchanged books skipped |
| `kalshi_trades` | Public trades per market (REST mode) |
| `kalshi_ws` | Raw WebSocket messages: `orderbook_delta`, `ticker`, `trade` (key configured) |
| `book_checks` | WebSocket-rebuilt book vs REST snapshot every 60s (Phase 0 exit gate) |
| `espn_plays` | Play-by-play with ESPN wall clock, our receive time, and a dead-ball flag |
| `espn_state`, `games` | Game status and the ESPN-to-Kalshi ticker mapping |
| `espn_odds` | Free pregame sportsbook lines and injuries from ESPN, every 15 min within 12h of tip |
| `espn_summaries/` | Full ESPN game summary (box score, PBP, odds) saved once per finished game |

Without an API key the recorder polls Kalshi's public REST API at 4 requests per second at
most (unauthenticated traffic is rate-limited around 5/s). With a key it switches to the
WebSocket feed, which is real-time and has no polling gaps.

Query recordings with DuckDB, for example:
`SELECT * FROM read_parquet('data/raw/kalshi_books/**/*.parquet', hive_partitioning=true)`.

## Tests

```bash
uv run python -m pytest -m "not network"   # offline unit tests
uv run python -m pytest                    # plus live public-endpoint smoke tests
```
