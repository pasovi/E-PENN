# R1 census (14,000 iterations)

Entries: median [Q1, Q3] (min-max) over runs.


## arm `grid` (n = 100)

| statistic | evader | pursuer |
|---|---|---|
| **Strategy ensemble (unit headings)** | | |
| Gram PR (Prop. 1 object; <= 2 if constant) | 2.47 [2.32, 2.67] (2.05-3.28) | 2.47 [2.32, 2.67] (2.06-3.07) |
| Gram PR, head-centred (legacy 'head PR') | 2.47 [2.31, 2.70] (1.99-3.25) | 2.46 [2.31, 2.67] (2.04-3.09) |
| Constancy eta (1 = constant headings) | 0.86 [0.79, 0.90] (0.69-0.96) | 0.86 [0.80, 0.90] (0.71-0.96) |
| Temporal PR of u (time-varying part) | 2.73 [2.29, 3.15] (1.44-4.29) | 3.25 [2.75, 3.59] (1.36-5.41) |
| Temporal PR of z (LEGACY headline 'PR(Z)'; gauge-dependent) | 2.26 [2.04, 2.61] (1.54-3.75) | 1.77 [1.55, 2.04] (1.26-3.58) |
| Gram PR, capturable heads only | 2.68 [2.42, 3.07] (1.97-3.92) | 2.69 [2.48, 3.05] (2.04-3.68) |
| Gram PR, uncapturable heads only | 1.85 [1.75, 1.91] (1.05-2.05) | 1.85 [1.75, 1.91] (1.05-2.10) |
| **Latent curve** | | |
| PR(H), time-centred (LEGACY) | 5.61 [5.23, 6.19] (4.09-7.13) | 6.77 [5.93, 7.47] (4.44-8.98) |
| PR(H), uncentred | 3.13 [2.75, 3.50] (2.08-4.89) | 3.98 [3.29, 4.71] (2.35-7.00) |
| Latent power fraction at k <= K | 0.98 [0.98, 0.99] (0.98-0.99) | 0.91 [0.91, 0.92] (0.89-0.93) |
| **Heads** | | |
| Gauge-fixed effective rank of W_stack | 2.70 [2.37, 3.11] (2.10-4.02) | 2.16 [1.94, 2.44] (1.52-2.94) |
| Unit-heading power fraction, k in [11,50] | 0.15 [0.11, 0.18] (0.05-0.29) | 0.29 [0.25, 0.33] (0.11-0.44) |
| **Within-run ratios** | | |
| PR(H)/PR(Z) (LEGACY) | 2.51 [2.15, 2.86] (1.61-4.23) | 3.68 [3.31, 4.20] (2.02-5.73) |
| PR(H)/tPR(u) (gauge-invariant) | 2.07 [1.84, 2.46] (1.33-3.39) | 2.07 [1.81, 2.50] (0.98-3.63) |

| game statistic | value |
|---|---|
| capturable heads (of 16) | 8.50 [7.00, 9.00] (4.00-12.00) |
| capture rate on capturable heads | 1.00 [0.88, 1.00] (0.62-1.00) |
| uncapturable heads captured by trained P | 0.00 [0.00, 0.00] (0.00-1.00) |
| heads captured by a best-response P (of 16) | 16.00 [16.00, 16.00] (16.00-16.00) |
| pursuer regret vs best response (time) | 2.86 [2.67, 2.94] (2.49-3.12) |
| Gram PR of the analytic saddle (this IC set) | 1.92 [1.90, 1.96] (1.83-2.00) |
| band fraction ratio P/E (asymmetry statistic) | 1.91 [1.69, 2.34] (0.96-3.32) |
| raw max singular value ratio P/E | 3.11 [2.80, 3.39] (1.96-4.97) |

## arm `chaos` (n = 10)

| statistic | evader | pursuer |
|---|---|---|
| **Strategy ensemble (unit headings)** | | |
| Gram PR (Prop. 1 object; <= 2 if constant) | 2.45 [2.28, 2.46] (2.18-2.57) | 2.40 [2.30, 2.47] (2.23-2.53) |
| Gram PR, head-centred (legacy 'head PR') | 2.47 [2.35, 2.51] (2.21-2.63) | 2.43 [2.36, 2.49] (2.26-2.72) |
| Constancy eta (1 = constant headings) | 0.88 [0.87, 0.91] (0.85-0.95) | 0.89 [0.88, 0.91] (0.85-0.93) |
| Temporal PR of u (time-varying part) | 2.23 [1.96, 2.42] (1.51-3.04) | 2.82 [2.64, 2.98] (2.16-3.82) |
| Temporal PR of z (LEGACY headline 'PR(Z)'; gauge-dependent) | 1.99 [1.84, 2.08] (1.57-2.18) | 1.69 [1.60, 1.89] (1.46-2.06) |
| Gram PR, capturable heads only | 2.17 [2.10, 2.22] (1.99-2.52) | 2.22 [2.13, 2.30] (2.11-2.42) |
| Gram PR, uncapturable heads only | 1.95 [1.95, 1.96] (1.93-1.96) | 1.96 [1.96, 1.97] (1.93-1.98) |
| **Latent curve** | | |
| PR(H), time-centred (LEGACY) | 6.77 [6.59, 6.97] (6.18-7.69) | 7.21 [6.33, 7.46] (5.97-8.47) |
| PR(H), uncentred | 3.89 [3.65, 4.22] (2.87-4.73) | 4.67 [4.24, 5.19] (3.43-6.09) |
| Latent power fraction at k <= K | 0.98 [0.98, 0.99] (0.98-0.99) | 0.91 [0.90, 0.92] (0.89-0.93) |
| **Heads** | | |
| Gauge-fixed effective rank of W_stack | 2.16 [2.09, 2.24] (2.02-2.42) | 1.78 [1.67, 1.84] (1.52-2.01) |
| Unit-heading power fraction, k in [11,50] | 0.15 [0.14, 0.20] (0.13-0.26) | 0.29 [0.28, 0.37] (0.21-0.40) |
| **Within-run ratios** | | |
| PR(H)/PR(Z) (LEGACY) | 3.51 [3.31, 3.60] (3.12-3.94) | 3.91 [3.63, 4.61] (3.03-5.08) |
| PR(H)/tPR(u) (gauge-invariant) | 3.03 [2.73, 3.50] (2.39-4.10) | 2.48 [2.26, 2.82] (1.92-3.22) |

| game statistic | value |
|---|---|
| capturable heads (of 16) | 4.00 [4.00, 4.00] (4.00-4.00) |
| capture rate on capturable heads | 1.00 [1.00, 1.00] (1.00-1.00) |
| uncapturable heads captured by trained P | 0.00 [0.00, 0.00] (0.00-0.00) |
| heads captured by a best-response P (of 16) | 16.00 [16.00, 16.00] (16.00-16.00) |
| pursuer regret vs best response (time) | 2.93 [2.93, 2.94] (2.93-2.94) |
| Gram PR of the analytic saddle (this IC set) | 1.96 [1.96, 1.96] (1.96-1.96) |
| band fraction ratio P/E (asymmetry statistic) | 1.70 [1.49, 2.06] (1.42-2.56) |
| raw max singular value ratio P/E | 2.77 [2.58, 2.82] (2.45-3.37) |

## arm `noreset` (n = 10)

| statistic | evader | pursuer |
|---|---|---|
| **Strategy ensemble (unit headings)** | | |
| Gram PR (Prop. 1 object; <= 2 if constant) | 2.45 [2.40, 2.48] (2.32-2.62) | 2.44 [2.38, 2.47] (2.32-2.62) |
| Gram PR, head-centred (legacy 'head PR') | 2.50 [2.38, 2.54] (2.26-2.78) | 2.43 [2.36, 2.53] (2.32-2.84) |
| Constancy eta (1 = constant headings) | 0.88 [0.87, 0.89] (0.83-0.92) | 0.88 [0.87, 0.89] (0.83-0.92) |
| Temporal PR of u (time-varying part) | 2.41 [2.07, 2.58] (1.56-3.18) | 2.94 [2.67, 3.19] (2.13-3.55) |
| Temporal PR of z (LEGACY headline 'PR(Z)'; gauge-dependent) | 2.24 [1.84, 2.33] (1.48-2.83) | 1.95 [1.74, 2.33] (1.47-2.86) |
| Gram PR, capturable heads only | 2.27 [2.25, 2.33] (1.90-2.75) | 2.30 [2.25, 2.34] (2.08-2.51) |
| Gram PR, uncapturable heads only | 1.96 [1.95, 1.96] (1.93-2.01) | 1.96 [1.95, 1.97] (1.93-2.13) |
| **Latent curve** | | |
| PR(H), time-centred (LEGACY) | 5.04 [4.70, 5.76] (4.21-5.89) | 7.32 [6.38, 7.76] (6.02-9.50) |
| PR(H), uncentred | 2.87 [2.75, 3.27] (2.62-3.73) | 4.54 [4.16, 5.18] (3.72-6.99) |
| Latent power fraction at k <= K | 0.99 [0.99, 0.99] (0.98-0.99) | 0.91 [0.90, 0.91] (0.89-0.92) |
| **Heads** | | |
| Gauge-fixed effective rank of W_stack | 2.64 [2.38, 2.81] (2.04-3.10) | 1.82 [1.77, 1.96] (1.72-2.29) |
| Unit-heading power fraction, k in [11,50] | 0.13 [0.12, 0.15] (0.07-0.20) | 0.30 [0.28, 0.33] (0.18-0.36) |
| **Within-run ratios** | | |
| PR(H)/PR(Z) (LEGACY) | 2.41 [2.06, 2.61] (1.75-3.93) | 3.66 [3.24, 4.06] (2.70-5.12) |
| PR(H)/tPR(u) (gauge-invariant) | 2.17 [1.85, 2.70] (1.67-2.98) | 2.66 [2.32, 2.90] (1.82-2.98) |

| game statistic | value |
|---|---|
| capturable heads (of 16) | 4.00 [4.00, 4.00] (4.00-4.00) |
| capture rate on capturable heads | 1.00 [1.00, 1.00] (1.00-1.00) |
| uncapturable heads captured by trained P | 0.00 [0.00, 0.00] (0.00-1.00) |
| heads captured by a best-response P (of 16) | 16.00 [16.00, 16.00] (16.00-16.00) |
| pursuer regret vs best response (time) | 2.93 [2.93, 2.93] (2.76-2.94) |
| Gram PR of the analytic saddle (this IC set) | 1.96 [1.96, 1.96] (1.96-1.96) |
| band fraction ratio P/E (asymmetry statistic) | 2.34 [1.83, 2.66] (1.45-2.80) |
| raw max singular value ratio P/E | 3.20 [3.01, 3.37] (2.83-3.87) |
