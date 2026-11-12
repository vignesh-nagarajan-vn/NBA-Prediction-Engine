# Phase 2 report: pregame fair value vs. Kalshi
_Generated 2026-11-12 09:15 UTC. 1561 games with Kalshi quotes, 2025-04-15 to 2026-11-12. Every probability is out of sample (fit only on games at least a day earlier)._

## Which fair value predicts outcomes best?

`kalshi_raw` is Kalshi's own log loss at that horizon. Other columns are the **difference** from it: negative means more accurate than Kalshi. Horizons are time before tip. Sportsbook blends exist only within 1h of tip (we only have closing lines).

| horizon | games | kalshi_raw | kalshi_debiased | model | kalshi+model | kalshi+book | kalshi+book+model |
|---|---|---|---|---|---|---|---|
| 15m | 834 | 0.5337 | 0.0046 | 0.0316 | 0.0056 | 0.0115 | 0.0118 |
| 1h | 834 | 0.5357 | 0.0040 | 0.0296 | 0.0050 | 0.0099 | 0.0103 |
| 3h | 1225 | 0.5770 | 0.0032 | 0.0161 | 0.0034 | - | - |
| 6h | 1225 | 0.5796 | 0.0032 | 0.0135 | 0.0027 | - | - |
| 12h | 1224 | 0.5824 | 0.0032 | 0.0110 | 0.0022 | - | - |
| 24h | 1216 | 0.5861 | 0.0033 | 0.0080 | 0.0019 | - | - |

## Trading it

Buy the YES side whose fair value beats the ask by more than `threshold` after the taker fee, earliest horizon from 24h to 15m, one trade per game, 100 contracts. `hold` holds to settlement; `clv_exit` sells at the bid at tip. Fills are conservative (pay the ask, sell the bid) but ignore depth.

| fair_value | threshold | trades | win_rate | hold_roi | hold_pnl_$ | hold_ci90_$/trade | clv_exit_roi | mean_clv_c | clv_ci90_c |
|---|---|---|---|---|---|---|---|---|---|
| kalshi_debiased | 0.0200 | 318 | 0.3994 | -0.0639 | -867.1200 | -6.76 to 1.42 | -0.0938 | -0.7052 | -1.06 to -0.34 |
| kalshi_debiased | 0.0300 | 200 | 0.3800 | -0.1099 | -938.0600 | -9.70 to 0.38 | -0.0919 | -0.5950 | -1.08 to -0.11 |
| kalshi_debiased | 0.0500 | 66 | 0.3030 | -0.1832 | -448.6400 | -15.75 to 2.34 | -0.0991 | -0.2235 | -0.89 to 0.48 |
| model | 0.0200 | 1237 | 0.3452 | -0.0365 | -1617.6100 | -3.39 to 0.75 | -0.0867 | 0.0994 | -0.11 to 0.31 |
| model | 0.0300 | 1080 | 0.3343 | -0.0400 | -1503.3100 | -3.60 to 0.77 | -0.0854 | 0.2148 | -0.02 to 0.45 |
| model | 0.0500 | 819 | 0.3297 | -0.0227 | -626.4800 | -3.28 to 1.77 | -0.0808 | 0.4570 | 0.18 to 0.73 |
| kalshi+model | 0.0200 | 172 | 0.3198 | -0.0844 | -507.2500 | -8.39 to 2.52 | -0.0750 | 0.3779 | -0.13 to 0.89 |
| kalshi+model | 0.0300 | 63 | 0.2857 | -0.2498 | -599.3800 | -17.69 to -1.00 | -0.0597 | 0.8889 | -0.07 to 1.88 |
| kalshi+model | 0.0500 | 10 | 0.3000 | -0.1058 | -35.4800 | -22.31 to 17.88 | -0.0852 | 0.3000 | -1.25 to 2.18 |
| kalshi+book | 0.0200 | 255 | 0.1843 | -0.2416 | -1497.4000 | -9.69 to -1.87 | -0.1231 | -0.2206 | -0.37 to -0.06 |
| kalshi+book | 0.0300 | 196 | 0.1735 | -0.2639 | -1219.1600 | -10.46 to -1.85 | -0.1227 | -0.1492 | -0.32 to 0.04 |
| kalshi+book | 0.0500 | 105 | 0.1524 | -0.2997 | -684.6000 | -12.27 to -0.84 | -0.1215 | -0.0333 | -0.30 to 0.29 |
| kalshi+book+model | 0.0200 | 259 | 0.2046 | -0.2265 | -1552.2700 | -9.85 to -2.01 | -0.1139 | -0.1641 | -0.31 to 0.00 |
| kalshi+book+model | 0.0300 | 204 | 0.1961 | -0.2297 | -1192.8400 | -10.04 to -1.26 | -0.1161 | -0.1213 | -0.30 to 0.08 |
| kalshi+book+model | 0.0500 | 103 | 0.1650 | -0.2443 | -549.4500 | -11.30 to 0.84 | -0.1219 | -0.0388 | -0.31 to 0.29 |
