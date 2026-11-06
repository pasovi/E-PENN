# Control: TL on ICs uniform inside the capturable shell

500 ICs per dimension in 0.4 ≤ |Δ0| ≤ 0.88 (IC seed 0), head seeds [0, 1, 2], same protocol, frozen pipeline bodies.

| | S^1 | T^2 | T^3 | T^4 |
|---|---|---|---|---|
| errors per seed (of 500) | 204 / 221 / 235 | 64 / 60 / 58 | 35 / 31 / 30 | 20 / 21 / 10 |
| error rate pooled (95 % CI) | 44.0 % [41.5, 46.5] | 12.1 % [10.6, 13.9] | 6.4 % [5.3, 7.8] | 3.4 % [2.6, 4.4] |
| danger-zone share of the ICs | 18.4 % | 15.2 % | 10.2 % | 9.0 % |
| errors / ICs inside the zone (rate) | 141 / 276 (51 %) | 96 / 228 (42 %) | 51 / 153 (33 %) | 29 / 135 (21 %) |
| errors / ICs outside the zone (rate) | 519 / 1224 (42.4 %) | 86 / 1272 (6.8 %) | 45 / 1347 (3.3 %) | 22 / 1365 (1.6 %) |

Pooled logistic fit over T2, T3, T4: logit p = 6.26 + (26.59) b + (-19.94) m; geometry vs null p = 4.6e-190; dimension term on top: LR = 1.35 (2 df), p = 0.509; odds ratios vs T²: T3 1.10, T4 0.87.

| | observed errors | expected (geometry only) | observed rate | standardised rate |
|---|---|---|---|---|
| T2 | 182 | 182.8 | 12.13 % | 12.19 % |
| T3 | 96 | 89.7 | 6.40 % | 5.98 % |
| T4 | 51 | 56.5 | 3.40 % | 3.77 % |
