# Phase 1 report: pregame model vs. the market
_Generated 2026-11-06 19:30 UTC from walk-forward predictions. Each season is predicted by a model trained only on earlier seasons._

## Data coverage

| season | games | with_spread | with_moneyline | with_kalshi |
|---|---|---|---|---|
| 2015 | 1311 | 1311 | 1311 | 0 |
| 2016 | 1316 | 1316 | 1316 | 0 |
| 2017 | 1309 | 1309 | 1309 | 0 |
| 2018 | 1312 | 1312 | 1312 | 0 |
| 2019 | 1312 | 1312 | 1312 | 0 |
| 2020 | 1143 | 1143 | 1143 | 0 |
| 2021 | 1171 | 1171 | 1171 | 0 |
| 2022 | 1323 | 1323 | 1323 | 0 |
| 2023 | 1320 | 1320 | 1320 | 0 |
| 2024 | 1319 | 0 | 0 | 0 |
| 2025 | 1321 | 0 | 0 | 86 |
| 2026 | 1322 | 1066 | 1061 | 1316 |
| 2027 | 121 | 121 | 121 | 121 |

## Model vs. sportsbook closing spread

The market probability converts the median closing spread across ESPN's odds providers with a normal margin distribution (sigma fit on training seasons). Lower log loss is better; `gap` is model minus market.

| season | games | model_logloss | market_logloss | gap | model_acc | market_acc | model_calib_slope |
|---|---|---|---|---|---|---|---|
| 2017 | 1309 | 0.6290 | 0.6073 | 0.0217 | 0.6486 | 0.6738 | 0.8236 |
| 2018 | 1312 | 0.6179 | 0.5924 | 0.0255 | 0.6616 | 0.6890 | 1.1257 |
| 2019 | 1312 | 0.6120 | 0.5972 | 0.0148 | 0.6532 | 0.6700 | 1.0735 |
| 2020 | 1143 | 0.6273 | 0.6118 | 0.0155 | 0.6500 | 0.6667 | 1.0411 |
| 2021 | 1171 | 0.6397 | 0.6202 | 0.0195 | 0.6200 | 0.6610 | 0.9767 |
| 2022 | 1323 | 0.6278 | 0.6053 | 0.0225 | 0.6508 | 0.6833 | 1.0580 |
| 2023 | 1320 | 0.6410 | 0.6257 | 0.0153 | 0.6402 | 0.6583 | 0.9730 |
| 2026 | 1066 | 0.5967 | 0.5810 | 0.0156 | 0.6839 | 0.6829 | 1.1042 |
| 2027 | 121 | 0.6106 | 0.5638 | 0.0468 | 0.6281 | 0.6942 | 1.1856 |

**Pooled (10077 games):** model log loss 0.6242 vs market 0.6049 (gap +0.0193); model accuracy 0.650 vs 0.673; model calibration slope 1.017, intercept -0.035.

### Phase 1 gate

- Log loss within 0.005 of the closing line: **FAIL** (gap +0.0193)
- Calibration slope in [0.9, 1.1]: **PASS** (1.017)

Note: ESPN's closing lines come from mainstream books, not Pinnacle, so this is a slightly softer benchmark than the plan's sharp close.

## Does the model add anything the market misses?

Out-of-sample logistic blend of logit(market) and logit(model), refit each season on earlier seasons.

- Games: 8768; market log loss 0.6045, blend 0.6049 (improvement -0.0004).
- A positive improvement means the model carries information the closing line does not. Near zero means the market already prices everything the model knows.

## Kalshi pre-tip prices

Kalshi P(home win) from the last 1-minute quote before tip (bid/ask mid, or last trade when the quote is wide), averaged across the two team markets.

| source | n | log_loss | brier | accuracy | calib_slope | calib_intercept |
|---|---|---|---|---|---|---|
| Kalshi pre-tip | 1181 | 0.5762 | 0.1975 | 0.6876 | 1.0127 | -0.0286 |
| Sportsbook close | 1181 | 0.5787 | 0.1984 | 0.6850 | 1.0088 | -0.0302 |
| Our model | 1181 | 0.5970 | 0.2056 | 0.6799 | 1.1160 | -0.0567 |

Kalshi minus sportsbook probability: mean -0.0005, mean absolute 0.0156, 90th percentile absolute 0.0318. Median quoted spread at tip: 0.010.

Reliability of Kalshi prices (a favorite-longshot bias would show longshot bins winning less often than priced, favorites more often):

| kalshi_price | home_win_rate | games |
|---|---|---|
| 0.0767 | 0.0000 | 23 |
| 0.1574 | 0.1594 | 69 |
| 0.2538 | 0.2235 | 85 |
| 0.3492 | 0.3885 | 139 |
| 0.4491 | 0.4540 | 163 |
| 0.5581 | 0.5257 | 175 |
| 0.6524 | 0.6725 | 171 |
| 0.7485 | 0.7378 | 164 |
| 0.8476 | 0.7987 | 149 |
| 0.9273 | 1.0000 | 43 |

## Model coefficients (season 2027 model)

| feature | coef |
|---|---|
| intercept | -1.3416 |
| elo_diff | 0.0059 |
| rating_margin_pred | 0.3640 |
| player_strength_diff | 0.4953 |
| home_court | 2.3546 |
| rest_diff | 0.1606 |
| home_b2b | -1.2961 |
| away_b2b | 1.6743 |
| home_3in4 | -0.4218 |
| away_3in4 | 0.6914 |
| home_travel_k | 0.0374 |
| away_travel_k | -0.0666 |
| home_tz_abs | -0.2492 |
| away_tz_abs | 0.2276 |
| altitude_home | 1.0252 |
| home_out_regulars | -0.4639 |
| away_out_regulars | 0.4080 |
| playoffs | 2.0500 |

Residual sigma: 13.36 points.

## Highest-rated players (latest ratings, points per 48 vs average)

| player | rating |
|---|---|
| Kawhi Leonard | 6.44 |
| Kevin Huerter | 6.23 |
| Kyrie Irving | 5.92 |
| Ajay Mitchell | 5.85 |
| Dylan Harper | 5.58 |
| Rudy Gobert | 5.38 |
| Stephen Curry | 5.16 |
| Jarrett Allen | 4.81 |
| VJ Edgecombe | 4.80 |
| Carter Bryant | 4.77 |
| Brook Lopez | 4.77 |
| Isaiah Joe | 4.67 |
| Victor Wembanyama | 4.66 |
| Cameron Johnson | 4.56 |
| Zach Edey | 4.56 |
