# NBA-Prediction-Engine

An engine that estimates fair, sentiment-free win probabilities for NBA games and trades the
gap against Kalshi's `KXNBAGAME` markets. The full roadmap is in [docs/PLAN.md](docs/PLAN.md).

**Status: Phase 0 (foundation + recorder).** No models or trading yet.

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
