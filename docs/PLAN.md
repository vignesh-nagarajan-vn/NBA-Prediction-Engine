# NBA Prediction Engine for Kalshi: Build Plan

## Status (Nov 2026)

| Phase | State |
|---|---|
| 0. Foundation + recorder | Built. The REST-polling recorder is verified on live games. The WebSocket gate is pending a Kalshi API key. |
| 1. History + pregame model | Built. Gate **failed** on accuracy (log loss +0.019 vs closing line) and passed on calibration. |
| 2. Debias + blend + pregame backtest | Built. Gate **failed**: no variant beats Kalshi's own price, and every strategy loses after fees. |
| 3. In-game WP + overreaction study | Next. All required data is already cached. |

See [reports/FINDINGS.md](../reports/FINDINGS.md) for the evidence behind each verdict.

## Context

The repo (`NBA-Prediction-Engine`, cloned at `C:\Users\nrvig\Projects\NBA Predictor Engine`) is empty apart from a README. The goal is an engine that:

1. Estimates a **fair, sentiment-free probability** for NBA game-winner markets on Kalshi.
2. Compares it to the **executable** Kalshi price (ask/bid, not mid) after fees and slippage.
3. Trades only when the edge is real, both **pregame** and **in-game** (swing trades that fade crowd overreactions).

Decisions already made:
- **Staged rollout:** backtest, then paper, then you approve every trade, then automatic within hard caps.
- **Free live feeds first:** the NBA CDN and ESPN JSON lag the Kalshi book by about 10–30 seconds, so in-game edge has to come from overreaction and mean reversion, never from speed.
- **v1 = game winners (`KXNBAGAME`)**, with spreads and totals in a later phase on the same model.

**Honest framing:** beating sharp sportsbooks on pregame NBA moneylines with a homegrown model is very hard. The most realistic edge is that Kalshi is a retail-heavy venue whose prices **lag and drift away from** sharp consensus and react emotionally in-game. The plan is built so data decides which strategies survive. Any strategy that misses its go/no-go gate gets cut, not tuned until it passes.

---

## 1. What "removing sentiment" means, concretely

The Kalshi price is roughly: true probability + crowd bias + noise. We estimate the true probability three independent ways, then blend:

| Source | What it gives | Why it's sentiment-light |
|---|---|---|
| **A. Own model** | P(win) from team strength, injuries, rest, travel, home court | Built only on basketball data |
| **B. Sharp consensus** | De-vigged Pinnacle/Circa-style odds (power or Shin method) | Sharp books move on professional money and limit retail bettors |
| **C. Debiased Kalshi** | Kalshi mid passed through a learned bias-correction layer | Explicitly strips measured crowd biases |

**Debias layer (`models/debias.py`):** fit on historical Kalshi prices vs. realized outcomes, on the logit scale:
`logit(p_true) = f(logit(kalshi_mid)) + β·sentiment_features`
- `f`: isotonic or spline calibration, which captures **favorite–longshot bias** (longshots overpriced, favorites underpriced).
- Sentiment features: popular-team flag (LAL, GSW, BOS, NYK, etc.), national-TV flag, team's recent win/loss streak, last-game blowout, time-to-tip, volume surge vs. the market's baseline, retail-sized trade share (trade-size distribution from the `trade` feed), and price move without sharp-book confirmation.

**Blend (`models/blend.py`):** stacked logistic regression on logits of A, B, and C, fit on out-of-sample history with time-series cross-validation. The weights tell us how much each source is worth. Expect B to dominate pregame. The model's job is to confirm signals and fill gaps where no sharp line exists yet.

**Edge per contract (YES side):**
`edge = p_fair − ask − fee(ask) − slippage_buffer − model_uncertainty_haircut`
Mirror this for NO. Read the fee from the API per series (`fee_type` / multiplier) and never hardcode it. The current taker formula is `ceil(0.07 · C · P · (1−P))` cents per order, rounded **up per order**, so tiny orders are fee-inefficient. Sports may now carry maker fees (coefficient about 0.0175), so verify this live.

---

## 2. Strategies (in order of expected robustness)

### S1. Pregame value vs. sharp consensus (core)
- Trigger when Kalshi's executable price diverges from blended fair value by more than the threshold (start at 3¢ net after fees and tune in backtest).
- Watch the windows where Kalshi is likeliest to be stale: market open, injury-report drops, lineup confirmations (about 30 minutes before tip), and the last 10 minutes before tip when retail flow spikes.
- Exit: hold to settlement, **or** sell into convergence (S2).

### S2. Pregame CLV swing
- Enter when Kalshi lags a sharp-book move. Exit when Kalshi converges or at tipoff, whichever comes first.
- **Primary KPI: closing line value (CLV)**, meaning entry price vs. Kalshi price at tip. CLV is a much faster and less noisy signal than win/loss.

### S3. In-game overreaction fade (the swing-trading idea)
- In-game win-probability model (Section 3) gives `p_model(t)`. Signal when `kalshi_mid − p_model` exceeds a z-score threshold **after** a scoring run, star-player injury scare, or foul trouble.
- **Neutralizing the free-feed lag:** act **only during dead-ball stoppages** (timeouts, quarter and half breaks, free-throw sequences, replay reviews), when game state is frozen, so a 20-second-old feed is still accurate. The live-feed module flags these from play-by-play event types.
- Exit: revert to target (gap closed by 50–70%), time stop (N possessions or end of quarter), or stop on adverse model move. Never hold through the final 2 minutes of a close game, where the win-probability model's variance explodes.
- **Unproven until the Phase 3 study shows it works after fees with a simulated 30-second feed delay.**

### S4. Passive market making around fair value (Phase 6, research only)
Rest post-only limit orders at fair ± spread on both sides, cancel on feed events, and inventory-skew quotes. Adverse selection risk is high with a lagged feed, so this only goes ahead if Phases 3–5 show stable fair values.

---

## 3. Models

**Pregame (`models/pregame.py`)** predicts margin μ, then `P(win) = Φ(μ/σ)` with σ of about 12–13 points (fitted). The margin distribution reuses directly for spreads and totals later.
- Team strength: possession-level ridge regression (adjusted offensive and defensive rating) with exponential time decay, plus an Elo baseline for comparison.
- Player impact: minutes-weighted player ratings (box-score plus on/off regression, RAPM-lite) so injuries and rest days move μ. The lineup-adjusted team rating is the key feature.
- Context: home court (fit per season, since it has shrunk post-2020), rest days, back-to-backs, 3-in-4s, travel distance and time zones, altitude (DEN, UTA).
- Training: 2014–15 through the current season. Walk-forward validation only, with no random splits.

**In-game win probability (`models/ingame_wp.py`)**
- Baseline: a Stern-style Brownian model, `P = Φ((d + μ·τ) / (σ·√τ))`, with d = score diff, τ = fraction of game remaining, and μ = the pregame margin prior.
- Correction: gradient-boosted model (LightGBM) on historical play-by-play. Features: d, τ, possession, bonus/fouls, timeouts left, pregame μ, and star on-court status.
- Calibrate with isotonic regression by time bucket. Target: well-calibrated reliability curves in every quarter.

**Evaluation:** log loss, Brier score, calibration plots, and log loss **relative to the de-vigged sharp closing line**. Matching the close is the bar, and beating it is unlikely.

---

## 4. Data sources

| Need | Source | Notes |
|---|---|---|
| Kalshi markets/books | REST `/trade-api/v2` (markets, events?series_ticker=KXNBAGAME, orderbook, trades, candlesticks) and WebSocket channels `orderbook_delta` (snapshot+deltas), `ticker`, `trade`, `fill` | Auth headers `KALSHI-ACCESS-KEY/-TIMESTAMP/-SIGNATURE`; Ed25519 (preferred) or RSA-PSS-SHA256 over `timestamp+METHOD+path` (no query string). Use the demo env `external-api.demo.kalshi.co` for all integration testing |
| Historical games, box, PBP | `nba_api` (stats.nba.com), with Basketball-Reference as fallback | Throttle requests and cache everything to Parquet |
| Live PBP / box | `cdn.nba.com/static/json/liveData/...` (free), ESPN scoreboard/summary JSON as backup | Poll every 3–5 seconds and log both receive time and event time |
| Injuries / lineups | Official NBA injury report (PDF, multiple daily releases) plus team news via ESPN | **Store publication timestamps** to avoid lookahead bias |
| Sharp odds | The Odds API (includes Pinnacle in some regions; historical odds need a paid tier, about $30–120/month) | The only recommended paid item. Without it S1/S2 lose their best anchor |

**Phase 0 urgency:** Kalshi gives trades and candlesticks but **not historical order books**. Every game night not recorded is lost in-game data, and the 2026–27 season is already underway. Start the recorder first.

---

## 5. Architecture (Python 3.12)

```
nba_engine/
  config/        settings.py (pydantic-settings, .env), risk.yaml, strategies.yaml
  kalshi/        auth.py (signing), rest.py (httpx, retry, rate limit), ws.py (asyncio,
                 reconnect, seq-gap detect, snapshot resync), orderbook.py (local book),
                 fees.py, schemas.py
  data/          nba_stats.py, live_feed.py (dead-ball detection), injuries.py,
                 odds.py (de-vig), schedule.py (game ↔ Kalshi ticker mapping)
  storage/       Parquet + DuckDB for research history; SQLite (WAL) for live state
                 (orders, fills, positions, signals, approvals)
  recorder/      always-on process: Kalshi WS + live feed + odds snapshots → Parquet
  features/      team_ratings.py, player_impact.py, schedule_context.py, sentiment.py
  models/        pregame.py, ingame_wp.py, debias.py, blend.py, calibration.py, registry.py
  strategy/      base.py (Signal dataclass), pregame_value.py, clv_swing.py,
                 ingame_reversion.py
  execution/     sizing.py, risk.py, order_manager.py (idempotent client_order_id,
                 reconcile loop), brokers/{paper.py, kalshi_live.py}, approval.py
  backtest/      replay.py (event-driven replay of recorded books + feed),
                 fills.py (conservative fill model), metrics.py
  monitoring/    Streamlit dashboard (positions, P&L, CLV, signals, approve/reject),
                 alerts.py (Discord/Telegram webhook), structured logging
  cli.py         typer: record | backfill | train | backtest | paper | live | kill
tests/           unit + integration (demo env) + replay regression fixtures
```

Key design rules:
- **One code path for backtest, paper, and live.** Strategies emit `Signal`s, and only the broker layer differs. This keeps backtests honest.
- An event-driven core (asyncio) with a single in-process event bus; no microservices.
- Secrets: the Kalshi private key file lives outside the repo, and its path goes in `.env`, which `.gitignore` covers.

---

## 6. Execution, sizing, and risk

- **Orders:** limit orders by default. Post-only when the edge is modest, and cross the spread (taker) only when net edge after taker fee is at least 2× the threshold. Size from the visible book depth and never take more than X% of top-of-book.
- **Sizing:** fractional Kelly (start at 0.25×) on net edge, shrunk by model uncertainty (bootstrap std of p_fair), then capped.
- **Hard caps (`risk.yaml`):** max 2% of bankroll per trade, 5% per game, 25% total open exposure, 5% daily loss limit (halts trading for the day), and a 15% drawdown from peak (halts everything until manual reset). Same-game positions aggregate as one exposure.
- **Safety:** reconcile positions and orders with Kalshi every 30 seconds, and halt on mismatch. Cancel all resting orders on WS disconnect or stale feed (more than 60 seconds). Kill switch via `cli kill` and a dashboard button. Check clock skew at startup, since signing depends on the timestamp.
- **Approval mode (Phase 5):** signals appear on the dashboard and as a phone alert with fair value, price, edge, size, and reason. You approve or reject, and each signal expires after N seconds, since prices move.

---

## 7. Phased roadmap with go/no-go gates

| Phase | Build | Exit gate |
|---|---|---|
| **0. Foundation + recorder** (week 1) | Project scaffold, Kalshi auth/REST/WS client tested on demo, local order book, game↔ticker mapping, always-on recorder (Kalshi WS + NBA live feed + odds) | Recorder captures a full game night with no sequence gaps, and the rebuilt book matches the REST snapshot |
| **1. History + pregame model** (weeks 2–3) | Backfill seasons, injury history, Kalshi trades/candlesticks for KXNBAGAME, sharp odds history; team/player ratings; pregame model | Walk-forward log loss within about 0.005 of the de-vigged sharp close, and calibration slope 0.9–1.1 |
| **2. Debias + blend + pregame backtest** (weeks 3–4) | Bias study (favorite–longshot, popular teams, time-to-tip), debias layer, blend, S1/S2 backtests on recorded and historical prices with a conservative fill model | Positive mean CLV after fees, and positive ROI with a bootstrap 90% CI that excludes zero over at least 150 historical signals |
| **3. In-game WP + overreaction study** (weeks 4–6) | WP model, dead-ball detector, S3 replay backtest with an **injected 30-second feed delay** | WP calibration holds in every quarter, and S3 is net-positive after fees and delay over at least 100 events. Otherwise S3 is dropped |
| **4. Paper trading** (at least 4 weeks) | Paper broker fills against the live book (only if the book trades through our price), dashboard, alerts | Paper results within the backtest's expected range, CLV still positive, zero operational incidents |
| **5. Live, human-approved** (at least 4 weeks) | Live broker, approval flow, small bankroll (you set it), all caps on | Realized slippage and fill rates match paper, no risk-cap breaches, positive CLV |
| **6. Automatic + expansion** | Auto mode within caps; spreads/totals via the margin distribution; S4 market-making research; optional paid low-latency feed if S3 earns it | Ongoing monitoring: weekly CLV/ROI report, automatic strategy disable if rolling 100-trade CLV turns negative |

---

## 8. Backtest integrity checklist

- Every feature uses only data timestamped **before** the decision time (injury report publication time, odds snapshot time, feed receive time, not event time).
- Fills happen only when a later trade prints at or through our limit, or when we cross the visible book at the recorded depth. No fills at the mid.
- Fees use the exact per-order rounding. Include a latency of at least 1 second for order placement in replay.
- Hold out the most recent season and do not touch it until final evaluation.
- Pre-register thresholds per phase so they cannot be tuned on the evaluation set.

---

## 9. Compliance and practical notes

- Confirm your state allows Kalshi sports contracts (several states have contested them) and that API trading fits your account tier and Kalshi's rate limits.
- Kalshi issues tax forms for net gains, so the fills table doubles as a tax ledger and should export to CSV.
- This plan is a research and engineering framework, not financial advice. Treat early capital as tuition.

---

## 10. Verification

- **Unit:** fee calculator against Kalshi's published examples; signature generation against a known vector; de-vig math; order book delta application; risk-cap enforcement (property tests with Hypothesis).
- **Integration (demo env):** place, amend, and cancel orders; receive `fill` events; reconcile positions; force WS disconnect and confirm resync and cancel-all.
- **Replay regression:** one recorded game night saved as a fixture; the backtest on it must reproduce identical signals and P&L across runs.
- **Model:** walk-forward reports (log loss, Brier, reliability plots) generated by `cli train --report`.
- **End-to-end:** run `cli paper` through a full NBA slate. The dashboard shows signals, paper fills, CLV at tip, and settled P&L, and the kill switch works mid-game.

## First implementation steps after approval

1. Copy this plan into the repo as `docs/PLAN.md`. Scaffold the package with `pyproject.toml` (uv), ruff, pytest, `.env.example`, and `.gitignore` covering keys and data.
2. Build `kalshi/auth.py`, `rest.py`, `ws.py`, and `orderbook.py` against the demo env, with tests.
3. Ship the recorder and start it running nightly, since this is time-critical data.
4. In parallel, start the historical backfill (`nba_api`) and odds history pull.
