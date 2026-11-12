# Findings so far (Phases 1-3)

> **Update, Nov 12 2026:** model v2 and Phase 3 are done. The summary with charts is in
> [BRIEF.md](BRIEF.md). In short:
>
> - Lineup-stint player ratings and pace-adjusted efficiency narrowed the pregame gap to
>   the closing line from +0.019 to +0.016 (+0.0135 on the 2022-23+ holdout). That still
>   fails the 0.005 accuracy gate; calibration passes.
> - The in-game model beats the textbook formula (0.454 vs 0.485 log loss) but not Kalshi's
>   live price (0.443 vs 0.434). Fading in-game swings loses 8-10% after fees, so Phase 3's
>   gate also fails.
> - Rest-of-season projections miss by 6.5 wins on average in backtests, versus about 9
>   for naive baselines.
>
> The sections below are the original Phase 1-2 write-up and its v1 numbers.


Data: 15,600 completed NBA games from 2014-15 to Nov 2026 (ESPN: results, 403k player
box-score rows, closing lines) and every settled Kalshi `KXNBAGAME` market since April 2025
(1,634 games, 7.4M one-minute candles). Every number below is out of sample. Details are in
[phase1_pregame.md](phase1_pregame.md) and
[phase2_pregame_backtest.md](phase2_pregame_backtest.md).

## 1. Kalshi is sharp right before tip

On the 1,181 recent games where both exist, Kalshi's last pre-tip quote **beat the
sportsbook closing line** from ESPN's providers (log loss 0.576 vs 0.579) and was well
calibrated (slope 1.01). The two rarely disagree: mean absolute gap 1.6 points of
probability, and 90% of games within 3.2 points. The quoted spread at tip is usually 1¢.

Kalshi also gets steadily more accurate as tip approaches (log loss 0.589 at 24h, 0.580 at
3h, 0.577 at 15m), so information keeps arriving all day.

## 2. Phase 1 gate: the pregame model fails the accuracy bar

| | Log loss | Accuracy |
|---|---|---|
| Our model (10,077 games, 2016-17 onward) | 0.624 | 65.0% |
| Sportsbook closing spread | 0.605 | 67.3% |

- Accuracy gate (within 0.005 of the close): **FAIL**, gap +0.019.
- Calibration gate (slope 0.9 to 1.1): **PASS**, 1.02.
- An out-of-sample blend of model and market does **not** beat the market alone, so the
  model carries no information the closing line lacks.

The model is a reasonable public-data model (Elo, decayed team ratings, lineup-adjusted
player ratings, rest/travel/altitude). Closing the remaining gap needs information the
market has and the box score doesn't: timely injury and lineup news, minutes restrictions,
and sharp-money moves.

A sweep of player-rating settings (ridge 3/8/15/30, half-life 120-365 days) moved the gap
only between +0.019 and +0.021. The best setting (ridge 8, 365 days) is now the default.
It was chosen on the same evaluation seasons, so treat +0.019 as slightly optimistic.

## 3. Phase 2 gate: no pregame strategy survives

"Removing the crowd's sentiment" from Kalshi's own prices did not work. A bias-correction
layer for the favorite-longshot effect, popular teams, streaks, momentum and volume surges
was **worse than Kalshi's raw price at every horizon** out of sample (+0.003 log loss). The
apparent biases visible in-sample are noise at this sample size.

Nothing else beat Kalshi either: our model alone, Kalshi+model, Kalshi+sportsbook and all
three combined each scored worse than Kalshi's raw price from 24h out to 15m. Trading any
of them (pay the ask, taker fees, one trade per game) **lost money**, with hold-to-settlement
ROI between -4% and -30%.

One weak lead: model-only entries at a 5-point edge threshold beat the tip price by
+0.29¢ on average (90% CI 0.00 to 0.58¢, 740 trades). That is far below the ~1.5¢ taker fee,
selling at tip still loses 8%, and the threshold was picked from three candidates after the
fact. It is a signal that the model sometimes sees early what Kalshi prices later, but not
a tradeable edge.

**Gate result: FAIL.** Per the plan, pregame strategies S1 and S2 are dropped in their
current form rather than tuned until they pass.

## 4. What remains open

1. **Kalshi lagging sharp books intraday.** This is the plan's main pregame hypothesis, and
   it is still untested: we only have *closing* lines, which can't be used hours before
   tip. Testing it needs timestamped intraday sharp odds (e.g. The Odds API historical
   snapshots, a paid tier) lined up against Kalshi's minute candles.
2. **In-game overreaction (strategy S3).** This can be studied now without new data. The
   ESPN cache holds play-by-play with wall-clock times for every game, and Kalshi candles
   cover the in-game minutes of 1,634 games. That is Phase 3.
3. **Live recording.** The recorder captures order-book depth that candles lack, which any
   real execution study needs. It still needs a Kalshi API key for WebSocket mode.
