# Phase 1 report: pregame model vs. the market
_Generated 2026-11-12 08:46 UTC from walk-forward predictions. Each season is predicted by a model trained only on earlier seasons._

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
| 2027 | 159 | 159 | 159 | 159 |

## Model vs. sportsbook closing spread

The market probability converts the median closing spread across ESPN's odds providers with a normal margin distribution (sigma fit on training seasons). Lower log loss is better; `gap` is model minus market.

| season | games | model_logloss | market_logloss | gap | model_acc | market_acc | model_calib_slope |
|---|---|---|---|---|---|---|---|
| 2017 | 1309 | 0.6235 | 0.6073 | 0.0162 | 0.6524 | 0.6738 | 0.8033 |
| 2018 | 1312 | 0.6199 | 0.5924 | 0.0276 | 0.6562 | 0.6890 | 0.9431 |
| 2019 | 1312 | 0.6118 | 0.5972 | 0.0146 | 0.6608 | 0.6700 | 1.0218 |
| 2020 | 1143 | 0.6237 | 0.6118 | 0.0119 | 0.6667 | 0.6667 | 1.0415 |
| 2021 | 1171 | 0.6357 | 0.6202 | 0.0155 | 0.6388 | 0.6610 | 1.0590 |
| 2022 | 1323 | 0.6228 | 0.6053 | 0.0175 | 0.6576 | 0.6833 | 1.1074 |
| 2023 | 1320 | 0.6354 | 0.6257 | 0.0097 | 0.6492 | 0.6583 | 1.0238 |
| 2026 | 1066 | 0.5970 | 0.5810 | 0.0160 | 0.6792 | 0.6829 | 1.1852 |
| 2027 | 159 | 0.5750 | 0.5466 | 0.0284 | 0.6855 | 0.7170 | 1.6749 |

**Pooled (10115 games):** model log loss 0.6209 vs market 0.6045 (gap +0.0164); model accuracy 0.658 vs 0.674; model calibration slope 1.015, intercept -0.029.

### Phase 1 gate

- Log loss within 0.005 of the closing line: **FAIL** (gap +0.0164)
- Calibration slope in [0.9, 1.1]: **PASS** (1.015)

Note: ESPN's closing lines come from mainstream books, not Pinnacle, so this is a slightly softer benchmark than the plan's sharp close.

### Validation vs. holdout

Settings were chosen on the validation seasons only, so the holdout row is the fair estimate of live accuracy.

| split | games | model_logloss | market_logloss | gap | model_acc | market_acc |
|---|---|---|---|---|---|---|
| validation (2016-17 to 2021-22, used for tuning) | 7570 | 0.6227 | 0.6053 | 0.0174 | 0.6555 | 0.6744 |
| holdout (2022-23 onward, never used for tuning) | 2545 | 0.6155 | 0.6020 | 0.0135 | 0.6640 | 0.6723 |

## This season so far (2026-27)

The model is frozen at the start of the season: trained only on earlier seasons, with ratings updated game by game as results come in.

| source | n | log_loss | brier | accuracy |
|---|---|---|---|---|
| Our model | 159 | 0.5750 | 0.1966 | 0.6855 |
| Our model (games with Kalshi) | 159 | 0.5750 | 0.1966 | 0.6855 |
| Kalshi pre-tip (same games) | 159 | 0.5432 | 0.1835 | 0.7107 |
| Our model (games with a line) | 159 | 0.5750 | 0.1966 | 0.6855 |
| Sportsbook close (same games) | 159 | 0.5466 | 0.1847 | 0.7170 |

## Does the model add anything the market misses?

Out-of-sample logistic blend of logit(market) and logit(model), refit each season on earlier seasons.

- Games: 8806; market log loss 0.6041, blend 0.6046 (improvement -0.0005).
- A positive improvement means the model carries information the closing line does not. Near zero means the market already prices everything the model knows.

## Kalshi pre-tip prices

Kalshi P(home win) from the last 1-minute quote before tip (bid/ask mid, or last trade when the quote is wide), averaged across the two team markets.

| source | n | log_loss | brier | accuracy | calib_slope | calib_intercept |
|---|---|---|---|---|---|---|
| Kalshi pre-tip | 1219 | 0.5732 | 0.1962 | 0.6907 | 1.0325 | -0.0340 |
| Sportsbook close | 1219 | 0.5760 | 0.1972 | 0.6883 | 1.0276 | -0.0354 |
| Our model | 1219 | 0.5933 | 0.2039 | 0.6809 | 1.2435 | -0.0792 |

Kalshi minus sportsbook probability: mean -0.0005, mean absolute 0.0155, 90th percentile absolute 0.0318. Median quoted spread at tip: 0.010.

Reliability of Kalshi prices (a favorite-longshot bias would show longshot bins winning less often than priced, favorites more often):

| kalshi_price | home_win_rate | games |
|---|---|---|
| 0.0775 | 0.0000 | 24 |
| 0.1566 | 0.1528 | 72 |
| 0.2536 | 0.2308 | 91 |
| 0.3493 | 0.3750 | 144 |
| 0.4494 | 0.4524 | 168 |
| 0.5583 | 0.5251 | 179 |
| 0.6521 | 0.6761 | 176 |
| 0.7480 | 0.7396 | 169 |
| 0.8477 | 0.8039 | 153 |
| 0.9273 | 1.0000 | 43 |

## Model coefficients (season 2027 model)

| feature | coef |
|---|---|
| intercept | -1.4839 |
| elo_diff | 0.0048 |
| rating_margin_pred | 0.9020 |
| eff_margin_pred | -0.3745 |
| player_strength_diff | 0.5516 |
| home_court | 2.1333 |
| rest_diff | 0.1832 |
| home_b2b | -1.1846 |
| away_b2b | 1.6290 |
| home_3in4 | -0.3433 |
| away_3in4 | 0.6532 |
| home_travel_k | 0.0506 |
| away_travel_k | -0.0680 |
| home_tz_abs | -0.2546 |
| away_tz_abs | 0.2409 |
| altitude_home | 0.6618 |
| home_out_regulars | -0.6395 |
| away_out_regulars | 0.6061 |
| playoffs | 2.0550 |

Residual sigma: 13.34 points.

## Highest-rated players (latest ratings, points per 48 vs average)

| player | rating |
|---|---|
| Nikola Jokic | 5.94 |
| Victor Wembanyama | 5.28 |
| Shai Gilgeous-Alexander | 5.28 |
| Stephen Curry | 4.65 |
| Chet Holmgren | 4.36 |
| Kawhi Leonard | 4.29 |
| Giannis Antetokounmpo | 4.03 |
| Jrue Holiday | 3.84 |
| Kyrie Irving | 3.76 |
| Luka Doncic | 3.59 |
| Tyrese Haliburton | 3.57 |
| Alex Caruso | 3.52 |
| Anthony Davis | 3.46 |
| Kevin Durant | 3.46 |
| Donovan Mitchell | 3.46 |
