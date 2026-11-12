# Phase 3 report: in-game win probability and Kalshi live prices
_Generated 2026-11-12 08:49 UTC. Model trained on 2016-17 to 2021-22 play-by-play; everything below is scored on 2022-23 onward._

## Model accuracy by game phase (holdout seasons)

`stern` is the classic random-walk formula; `model` is our fitted version with possession and phase-specific weights. Both use the pregame expected margin.

| phase | states | model_logloss | stern_logloss |
|---|---|---|---|
| Q1 | 193009 | 0.5848 | 0.6037 |
| Q2 | 191912 | 0.5231 | 0.5572 |
| Q3 | 191282 | 0.4347 | 0.4745 |
| Q4 | 193635 | 0.2729 | 0.3047 |

Overall: model 0.4537, stern 0.4848 log loss over 769,838 game states.

Calibration (predicted vs observed home win rate):

| predicted | observed | states |
|---|---|---|
| 0.0343 | 0.0267 | 79057 |
| 0.1509 | 0.1442 | 51939 |
| 0.2510 | 0.2492 | 56825 |
| 0.3507 | 0.3460 | 61440 |
| 0.4510 | 0.4506 | 67433 |
| 0.5508 | 0.5429 | 75459 |
| 0.6506 | 0.6498 | 80094 |
| 0.7504 | 0.7468 | 82526 |
| 0.8500 | 0.8428 | 84691 |
| 0.9648 | 0.9701 | 130374 |

## Against Kalshi's live prices

207,992 game-minutes across 1548 games where Kalshi traded. Kalshi's price vs our model given the same moment's game state:

| phase | minutes | kalshi_logloss | model_logloss | mean_abs_gap |
|---|---|---|---|---|
| Q1 | 46073 | 0.5568 | 0.5704 | 0.0578 |
| Q2 | 61838 | 0.4922 | 0.5037 | 0.0494 |
| Q3 | 49455 | 0.4081 | 0.4173 | 0.0420 |
| Q4 | 50626 | 0.2757 | 0.2793 | 0.0282 |

Overall: Kalshi 0.4338 vs model 0.4433.

## Fading swings (strategy S3)

Trade only at dead balls, using game state from 30 seconds earlier (free-feed lag). Buy the side the model prefers when the gap exceeds the threshold, pay the ask, sell at the bid `hold_min` minutes later (or settle if the game ends), taker fees both ways, 100 contracts. `kalshi_moved_toward_model_c` is how far Kalshi's price moved in our direction while we held, in cents.

| threshold | hold_min | trades | roi | pnl_per_trade_$ | ci90_$ | kalshi_moved_toward_model_c |
|---|---|---|---|---|---|---|
| 0.0500 | 3 | 10622 | -0.0884 | -4.0586 | -4.16 to -3.96 | 0.0487 |
| 0.0500 | 10 | 6212 | -0.0849 | -3.8808 | -4.16 to -3.60 | 0.1278 |
| 0.0800 | 3 | 6008 | -0.0923 | -4.2086 | -4.35 to -4.07 | 0.0563 |
| 0.0800 | 10 | 3646 | -0.0842 | -3.8318 | -4.22 to -3.44 | 0.3211 |
| 0.1200 | 3 | 2469 | -0.0953 | -4.2103 | -4.45 to -3.97 | 0.1663 |
| 0.1200 | 10 | 1583 | -0.0955 | -4.2395 | -4.89 to -3.60 | 0.0182 |
| 0.1600 | 3 | 1017 | -0.1033 | -4.3886 | -4.78 to -4.00 | 0.0034 |
| 0.1600 | 10 | 664 | -0.0827 | -3.5295 | -4.57 to -2.49 | 0.7293 |
