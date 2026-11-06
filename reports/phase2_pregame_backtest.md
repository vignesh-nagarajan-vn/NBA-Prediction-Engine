# Phase 2 report: pregame fair value vs. Kalshi
_Generated 2026-11-06 19:31 UTC. 1523 games with Kalshi quotes, 2025-04-15 to 2026-11-05. Every probability is out of sample (fit only on games at least a day earlier)._

## Which fair value predicts outcomes best?

`kalshi_raw` is Kalshi's own log loss at that horizon. Other columns are the **difference** from it: negative means more accurate than Kalshi. Horizons are time before tip. Sportsbook blends exist only within 1h of tip (we only have closing lines).

| horizon | games | kalshi_raw | kalshi_debiased | model | kalshi+model | kalshi+book | kalshi+book+model |
|---|---|---|---|---|---|---|---|
| 15m | 796 | 0.5361 | 0.0048 | 0.0324 | 0.0067 | 0.0120 | 0.0124 |
| 1h | 796 | 0.5383 | 0.0042 | 0.0303 | 0.0060 | 0.0104 | 0.0107 |
| 3h | 1187 | 0.5799 | 0.0033 | 0.0170 | 0.0036 | - | - |
| 6h | 1187 | 0.5826 | 0.0034 | 0.0143 | 0.0028 | - | - |
| 12h | 1186 | 0.5855 | 0.0033 | 0.0117 | 0.0024 | - | - |
| 24h | 1178 | 0.5889 | 0.0033 | 0.0088 | 0.0019 | - | - |

## Trading it

Buy the YES side whose fair value beats the ask by more than `threshold` after the taker fee, earliest horizon from 24h to 15m, one trade per game, 100 contracts. `hold` holds to settlement; `clv_exit` sells at the bid at tip. Fills are conservative (pay the ask, sell the bid) but ignore depth.

| fair_value | threshold | trades | win_rate | hold_roi | hold_pnl_$ | hold_ci90_$/trade | clv_exit_roi | mean_clv_c | clv_ci90_c |
|---|---|---|---|---|---|---|---|---|---|
| kalshi_debiased | 0.0200 | 314 | 0.3981 | -0.0608 | -809.6900 | -6.73 to 1.48 | -0.0939 | -0.6760 | -1.05 to -0.32 |
| kalshi_debiased | 0.0300 | 200 | 0.3800 | -0.1099 | -938.0600 | -9.70 to 0.38 | -0.0919 | -0.5950 | -1.08 to -0.11 |
| kalshi_debiased | 0.0500 | 66 | 0.3030 | -0.1832 | -448.6400 | -15.75 to 2.34 | -0.0991 | -0.2235 | -0.89 to 0.48 |
| model | 0.0200 | 1118 | 0.3676 | -0.0454 | -1953.1200 | -3.94 to 0.42 | -0.0832 | 0.0304 | -0.21 to 0.27 |
| model | 0.0300 | 994 | 0.3592 | -0.0483 | -1810.8700 | -4.05 to 0.56 | -0.0819 | 0.1459 | -0.10 to 0.39 |
| model | 0.0500 | 740 | 0.3486 | -0.0377 | -1009.4200 | -4.00 to 1.32 | -0.0809 | 0.2875 | 0.00 to 0.58 |
| kalshi+model | 0.0200 | 167 | 0.3114 | -0.1022 | -592.2400 | -9.07 to 1.96 | -0.0880 | -0.0015 | -0.54 to 0.53 |
| kalshi+model | 0.0300 | 76 | 0.2895 | -0.1681 | -444.6800 | -13.76 to 2.18 | -0.1033 | -0.4441 | -1.22 to 0.31 |
| kalshi+model | 0.0500 | 18 | 0.2778 | -0.1859 | -114.1500 | -22.58 to 10.72 | -0.1114 | -0.6389 | -1.75 to 0.60 |
| kalshi+book | 0.0200 | 255 | 0.1843 | -0.2416 | -1497.4000 | -9.69 to -1.87 | -0.1231 | -0.2206 | -0.37 to -0.06 |
| kalshi+book | 0.0300 | 196 | 0.1735 | -0.2639 | -1219.1600 | -10.46 to -1.85 | -0.1227 | -0.1492 | -0.32 to 0.04 |
| kalshi+book | 0.0500 | 105 | 0.1524 | -0.2997 | -684.6000 | -12.27 to -0.84 | -0.1215 | -0.0333 | -0.30 to 0.29 |
| kalshi+book+model | 0.0200 | 258 | 0.1938 | -0.2476 | -1645.7000 | -10.27 to -2.34 | -0.1173 | -0.1938 | -0.34 to -0.03 |
| kalshi+book+model | 0.0300 | 191 | 0.1885 | -0.2422 | -1150.6500 | -10.53 to -1.34 | -0.1168 | -0.1113 | -0.31 to 0.11 |
| kalshi+book+model | 0.0500 | 106 | 0.1604 | -0.2768 | -650.5800 | -12.05 to 0.04 | -0.1206 | -0.0613 | -0.33 to 0.23 |
