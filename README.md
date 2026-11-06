# NBA-Prediction-Engine

An engine that estimates fair, sentiment-free win probabilities for NBA games and trades the
gap against Kalshi's `KXNBAGAME` markets. The full roadmap is in [docs/PLAN.md](docs/PLAN.md).

**Status: Phases 0-2 built; pregame strategies failed their gates.** No trading yet. Results
are in [reports/FINDINGS.md](reports/FINDINGS.md).

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

## History and the pregame model

```bash
uv run nba-engine backfill espn     # 2014-15 to today: games, box scores, closing lines (~1.5h first run)
uv run nba-engine backfill kalshi   # every settled KXNBAGAME market + 1-minute candles (~25 min)
uv run nba-engine train             # walk-forward model + reports/phase1_pregame.md
uv run nba-engine backtest          # fair value vs Kalshi + reports/phase2_pregame_backtest.md
```

Raw responses are cached under `data/cache/` so reruns only fetch what is new
(`backfill espn --tables-only` rebuilds the tables without any network calls).

The pregame model predicts the home margin from:

- **Elo** and **decay-weighted team margin ratings**, both updated game by game
- **Player-impact ratings** (game-level adjusted plus-minus with ridge shrinkage), so a
  team's strength drops when its best players are out
- **Schedule context:** rest, back-to-backs, 3-in-4, travel distance, time-zone shifts,
  altitude, playoffs

It converts that margin into a win probability. Each season is predicted by a model trained
only on earlier seasons, and the report compares it with sportsbook closing lines and
Kalshi's pre-tip prices.

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
uv run pytest -m "not network"   # offline unit tests
uv run pytest                    # plus live public-endpoint smoke tests
```
