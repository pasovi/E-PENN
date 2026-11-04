# R1 census (14,000 iterations)

Entries: median [Q1, Q3] (min-max) over runs.


## arm `grid` (n = 100)

| statistic | evader | pursuer |
|---|---|---|
| **Strategy ensemble (unit headings)** | | |
| Gram PR (Prop. 1 object; <= 2 if constant) | 2.47 [2.32, 2.70] (2.00-3.75) | 2.50 [2.30, 2.69] (1.90-3.57) |
| Gram PR, head-centred (legacy 'head PR') | 2.48 [2.31, 2.71] (2.04-3.57) | 2.49 [2.30, 2.65] (1.97-3.42) |
| Constancy eta (1 = constant headings) | 0.85 [0.81, 0.89] (0.63-0.94) | 0.85 [0.82, 0.89] (0.66-0.95) |
| Temporal PR of u (time-varying part) | 2.80 [2.39, 3.24] (1.66-4.34) | 3.35 [2.85, 3.78] (1.71-5.27) |
| Temporal PR of z (LEGACY headline 'PR(Z)'; gauge-dependent) | 2.43 [1.95, 2.74] (1.51-3.31) | 1.75 [1.57, 2.04] (1.21-2.78) |
| Gram PR, capturable heads only | 2.68 [2.46, 3.09] (2.10-4.16) | 2.70 [2.47, 3.11] (2.06-4.14) |
| Gram PR, uncapturable heads only | 1.85 [1.77, 1.92] (1.06-1.99) | 1.85 [1.76, 1.92] (1.06-1.99) |
| **Latent curve** | | |
| PR(H), time-centred (LEGACY) | 5.73 [5.29, 6.13] (3.80-6.82) | 6.51 [5.98, 7.28] (3.54-8.85) |
| PR(H), uncentred | 3.22 [2.76, 3.57] (2.18-4.44) | 3.97 [3.45, 4.73] (2.44-6.45) |
| Latent power fraction at k <= K | 0.98 [0.98, 0.99] (0.98-0.99) | 0.91 [0.91, 0.92] (0.89-0.94) |
| **Heads** | | |
| Gauge-fixed effective rank of W_stack | 2.75 [2.41, 3.11] (1.95-4.03) | 2.17 [1.91, 2.40] (1.57-3.38) |
| Unit-heading power fraction, k in [11,50] | 0.16 [0.12, 0.19] (0.05-0.35) | 0.28 [0.25, 0.33] (0.11-0.44) |
| **Within-run ratios** | | |
| PR(H)/PR(Z) (LEGACY) | 2.41 [2.07, 2.83] (1.49-4.34) | 3.76 [3.32, 4.11] (1.55-5.25) |
| PR(H)/tPR(u) (gauge-invariant) | 2.04 [1.77, 2.30] (1.30-2.97) | 1.96 [1.71, 2.29] (0.88-4.34) |

| game statistic | value |
|---|---|
| capturable heads (of 16) | 8.50 [7.00, 9.00] (4.00-12.00) |
| capture rate on capturable heads | 0.96 [0.87, 1.00] (0.71-1.00) |
| uncapturable heads captured by trained P | 0.00 [0.00, 0.00] (0.00-1.00) |
| heads captured by a best-response P (of 16) | 16.00 [16.00, 16.00] (16.00-16.00) |
| pursuer regret vs best response (time) | 2.86 [2.70, 2.94] (2.49-3.12) |
| Gram PR of the analytic saddle (this IC set) | 1.92 [1.90, 1.96] (1.83-2.00) |
| band fraction ratio P/E (asymmetry statistic) | 1.86 [1.59, 2.16] (0.92-3.78) |
| raw max singular value ratio P/E | 3.08 [2.69, 3.48] (2.13-4.69) |

## arm `chaos` (n = 10)

| statistic | evader | pursuer |
|---|---|---|
| **Strategy ensemble (unit headings)** | | |
| Gram PR (Prop. 1 object; <= 2 if constant) | 2.41 [2.31, 2.45] (2.21-2.50) | 2.42 [2.36, 2.44] (2.22-2.50) |
| Gram PR, head-centred (legacy 'head PR') | 2.39 [2.31, 2.49] (2.10-2.52) | 2.40 [2.37, 2.48] (2.10-2.51) |
| Constancy eta (1 = constant headings) | 0.89 [0.87, 0.91] (0.86-0.92) | 0.89 [0.87, 0.90] (0.86-0.92) |
| Temporal PR of u (time-varying part) | 2.02 [1.95, 2.31] (1.71-2.72) | 2.52 [2.43, 3.02] (2.18-3.26) |
| Temporal PR of z (LEGACY headline 'PR(Z)'; gauge-dependent) | 2.02 [1.91, 2.29] (1.83-2.46) | 1.96 [1.71, 2.16] (1.57-2.48) |
| Gram PR, capturable heads only | 2.30 [2.14, 2.43] (1.99-2.54) | 2.39 [2.26, 2.45] (2.04-2.50) |
| Gram PR, uncapturable heads only | 1.95 [1.91, 1.96] (1.86-1.97) | 1.95 [1.91, 1.96] (1.86-1.97) |
| **Latent curve** | | |
| PR(H), time-centred (LEGACY) | 6.05 [5.87, 6.90] (5.21-7.10) | 7.33 [6.84, 8.18] (6.11-8.95) |
| PR(H), uncentred | 3.60 [3.29, 3.73] (2.80-4.20) | 4.54 [3.97, 5.01] (3.32-6.76) |
| Latent power fraction at k <= K | 0.98 [0.98, 0.98] (0.98-0.99) | 0.91 [0.91, 0.92] (0.90-0.93) |
| **Heads** | | |
| Gauge-fixed effective rank of W_stack | 2.23 [2.06, 2.36] (2.02-2.43) | 1.87 [1.83, 2.05] (1.75-2.20) |
| Unit-heading power fraction, k in [11,50] | 0.17 [0.16, 0.19] (0.10-0.28) | 0.32 [0.27, 0.34] (0.20-0.40) |
| **Within-run ratios** | | |
| PR(H)/PR(Z) (LEGACY) | 2.97 [2.69, 3.34] (2.52-3.75) | 3.75 [3.34, 4.41] (2.87-5.32) |
| PR(H)/tPR(u) (gauge-invariant) | 2.92 [2.63, 3.34] (2.49-3.53) | 2.79 [2.59, 3.21] (1.87-3.83) |

| game statistic | value |
|---|---|
| capturable heads (of 16) | 4.00 [4.00, 4.00] (4.00-4.00) |
| capture rate on capturable heads | 1.00 [1.00, 1.00] (0.75-1.00) |
| uncapturable heads captured by trained P | 0.00 [0.00, 0.00] (0.00-0.00) |
| heads captured by a best-response P (of 16) | 16.00 [16.00, 16.00] (16.00-16.00) |
| pursuer regret vs best response (time) | 2.93 [2.93, 2.93] (2.92-3.00) |
| Gram PR of the analytic saddle (this IC set) | 1.96 [1.96, 1.96] (1.96-1.96) |
| band fraction ratio P/E (asymmetry statistic) | 1.75 [1.53, 1.84] (1.43-2.09) |
| raw max singular value ratio P/E | 2.85 [2.63, 2.97] (2.56-3.22) |

## arm `noreset` (n = 10)

| statistic | evader | pursuer |
|---|---|---|
| **Strategy ensemble (unit headings)** | | |
| Gram PR (Prop. 1 object; <= 2 if constant) | 2.39 [2.32, 2.44] (2.17-2.70) | 2.37 [2.33, 2.43] (2.22-2.61) |
| Gram PR, head-centred (legacy 'head PR') | 2.42 [2.22, 2.45] (2.12-2.73) | 2.39 [2.27, 2.46] (2.11-2.62) |
| Constancy eta (1 = constant headings) | 0.89 [0.88, 0.90] (0.83-0.95) | 0.89 [0.88, 0.90] (0.85-0.93) |
| Temporal PR of u (time-varying part) | 2.98 [2.30, 3.19] (1.62-3.46) | 3.25 [2.91, 3.63] (2.02-3.83) |
| Temporal PR of z (LEGACY headline 'PR(Z)'; gauge-dependent) | 1.97 [1.78, 2.33] (1.55-2.65) | 1.91 [1.58, 2.06] (1.38-2.58) |
| Gram PR, capturable heads only | 2.30 [2.07, 2.40] (2.02-2.62) | 2.28 [2.16, 2.37] (2.08-2.51) |
| Gram PR, uncapturable heads only | 1.94 [1.92, 1.96] (1.81-1.99) | 1.94 [1.92, 1.96] (1.81-1.98) |
| **Latent curve** | | |
| PR(H), time-centred (LEGACY) | 5.71 [5.37, 6.01] (4.59-6.43) | 7.23 [6.59, 7.94] (5.71-8.49) |
| PR(H), uncentred | 3.00 [2.72, 3.17] (2.31-3.61) | 4.63 [4.00, 5.52] (3.19-5.97) |
| Latent power fraction at k <= K | 0.99 [0.99, 0.99] (0.98-0.99) | 0.91 [0.91, 0.92] (0.90-0.92) |
| **Heads** | | |
| Gauge-fixed effective rank of W_stack | 2.54 [2.43, 2.64] (2.12-2.89) | 1.77 [1.73, 1.85] (1.71-2.37) |
| Unit-heading power fraction, k in [11,50] | 0.17 [0.14, 0.22] (0.05-0.24) | 0.31 [0.29, 0.34] (0.15-0.43) |
| **Within-run ratios** | | |
| PR(H)/PR(Z) (LEGACY) | 2.86 [2.33, 3.22] (2.20-3.83) | 3.76 [3.20, 4.95] (2.95-5.06) |
| PR(H)/tPR(u) (gauge-invariant) | 1.95 [1.80, 2.55] (1.57-3.27) | 2.28 [2.09, 2.71] (1.66-3.29) |

| game statistic | value |
|---|---|
| capturable heads (of 16) | 4.00 [4.00, 4.00] (4.00-4.00) |
| capture rate on capturable heads | 1.00 [1.00, 1.00] (1.00-1.00) |
| uncapturable heads captured by trained P | 0.00 [0.00, 0.00] (0.00-0.00) |
| heads captured by a best-response P (of 16) | 16.00 [16.00, 16.00] (16.00-16.00) |
| pursuer regret vs best response (time) | 2.93 [2.92, 2.94] (2.92-2.94) |
| Gram PR of the analytic saddle (this IC set) | 1.96 [1.96, 1.96] (1.96-1.96) |
| band fraction ratio P/E (asymmetry statistic) | 1.85 [1.59, 2.09] (1.21-3.59) |
| raw max singular value ratio P/E | 3.13 [3.08, 3.23] (2.79-3.37) |
