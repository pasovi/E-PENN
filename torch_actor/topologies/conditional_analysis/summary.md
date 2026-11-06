# Conditional analysis of the TL errors

Danger zone: b < 0.1 (outer shell of the capturable ball) and m < 0.5 (a competing image).

| | S¹ | T² | T³ | T⁴ |
|---|---|---|---|---|
| TL sampler | grid 100 | grid 32² | grid 10³ | Sobol 1000 |
| capturable ICs / errors | 88 / 21 | 616 / 56 | 360 / 10 | 179 / 3 |
| error rate (95 % CI) | 23.9 % [16.2, 33.7] | 9.1 % [7.1, 11.6] | 2.8 % [1.5, 5.0] | 1.7 % [0.6, 4.8] |
| capturable fraction, uniform sampling | 47.9 % | 48.5 % | 32.5 % | 17.8 % |
| danger-zone share of capturable ICs, uniform | 20.8 % | 16.8 % | 12.2 % | 8.5 % |
| danger-zone share, actual sample | 11.4 % | 13.0 % | 0.0 % | 6.1 % |
| errors / ICs inside the zone | 5 / 10 | 27 / 80 | 0 / 0 | 3 / 11 |
| errors / ICs outside the zone | 16 / 78 | 29 / 536 | 10 / 360 | 0 / 168 |

## Cells (b bin × m bin): errors / ICs

b bins: [0, 0.05), [0.05, 0.1), [0.1, 0.2), [0.2, inf); m bins: [0, 0.5), [0.5, 0.7), [0.7, inf)

| b bin | m bin | S¹ | T² | T³ | T⁴ |
|---|---|---|---|---|---|
| 0 | 0 | 2 / 5 | 21 / 48 | 0 / 0 | 3 / 6 |
| 0 | 1 | 0 / 0 | 0 / 20 | 1 / 48 | 0 / 14 |
| 0 | 2 | 0 / 0 | 0 / 0 | 0 / 8 | 0 / 7 |
| 1 | 0 | 3 / 5 | 6 / 32 | 0 / 0 | 0 / 5 |
| 1 | 1 | 0 / 0 | 0 / 24 | 0 / 24 | 0 / 15 |
| 1 | 2 | 0 / 0 | 0 / 8 | 0 / 0 | 0 / 15 |
| 2 | 0 | 0 / 2 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2 | 1 | 5 / 8 | 19 / 72 | 8 / 72 | 0 / 13 |
| 2 | 2 | 0 / 0 | 0 / 36 | 0 / 48 | 0 / 29 |
| 3 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 3 | 1 | 1 / 2 | 9 / 24 | 0 / 0 | 0 / 0 |
| 3 | 2 | 10 / 66 | 1 / 352 | 1 / 160 | 0 / 75 |

## Pooled logistic regression (T², T³, T⁴)

logit p = 5.36 + (25.28) b + (-18.53) m  on 1155 capturable ICs, 69 errors.
Geometry vs null: LR = 183.0 (df 2), p = 1.8e-40.
Dimension term on top of geometry: LR = 1.07 (df 2), p = 0.585; odds ratios vs T²: T³ 0.75, T⁴ 0.58.

| | observed errors | expected from the geometry-only model | observed rate | standardised rate |
|---|---|---|---|---|
| T2 | 56 | 53.3 | 9.1 % | 8.7 % |
| T3 | 10 | 11.4 | 2.8 % | 3.2 % |
| T4 | 3 | 4.3 | 1.7 % | 2.4 % |
