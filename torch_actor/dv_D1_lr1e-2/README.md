# `dv_D1_lr1e-2` — the 1-layer speed-difference sweep at 10× the learning rates (10 Nov 2026)

The [`dv_D1`](../dv_D1/README.md) sweep with the learning rates raised from lr_P = 1e-3, lr_E = 1e-4 to the
defaults **lr_P = 1e-2, lr_E = 1e-3**. Everything else is unchanged: 1 tanh layer of 16 units, K = 12,
capture-truncated loss, no warm-up, 3 P : 1 E, 14,000 iterations, Adam reset every 3,500.
Δv = v_P − v_E ∈ {0.05, 0.10, 0.15, **0.20**, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}, v_E = 1, the same ten points as
`dv_D1` and `dv_D4`. Each Δv is a full 100-run grid (10 init seeds × 10 IC sets). The seeds are those of `dv_D1`,
so the initial weights and the ICs are identical in the two sweeps, and only the learning rates differ. About 45 s
per grid on the Mac GPU (MPS, float32).

**Δv = 0.20 (v_P = 1.2) is the baseline speed** and is ringed in both figures. It is the configuration of the
existing grid `r1_trunc_w0_p3` (`checks/{ckpt,results}/r1_trunc_w0_p3/`), and its 100 runs are
bit-identical to it (max |difference| = 0 over every array of every checkpoint). That grid was trained under
Python 3.13.9 and this one under 3.11.18, both with torch 2.15.0 on the same Mac.

Geometry (as in `dv_D4`): R = Δv t_max + ε (t_max = 4, ε = 0.08) is the forced-capture radius and
ρ = R / (L/2) (L = 2). The capturable share of uniformly drawn ICs is f(R), the area of the radius-R disc clipped
to the torus cell. It has a break at ρ = 1 (Δv = 0.230), where the disc first touches the cell edges, and
saturates at ρ = √2 (Δv = 0.334), where it covers the whole cell. The IC sampler rejects |Δ0| < 0.4, so the
training ICs follow f(R | |Δ0| ≥ 0.4) = (4 f(R) − π 0.4²) / (4 − π 0.4²) instead.

## Contents

| | |
|---|---|
| `dv<dv>/ckpt/`, `dv<dv>/results/` | one 100-run grid per Δv, in the `baseline_D4` layout (rev2 records, checkpoints at 3,500 / 7,000 / 10,500 / 14,000) |
| `make_figures.py` | the two figures and `summary.json` (a copy of `dv_D1/make_figures.py`; it reads the body depth from the checkpoints and the learning rates from the records) |
| `figs/pr_vs_dv.png` | participation ratios vs Δv, evader and pursuer, median [IQR] over 100 runs: (a) the draft's strategy PR(Z) (`tpr_Z`) and centred latent PR(H) (`pr_H`); (b) the gauge-invariant Gram PR and uncentred PR(H). Δv = 0.2 ringed |
| `figs/capture_vs_dv.png` | capturable share of the training ICs, captured share, captured / capturable, against f(R) and f(R \| \|Δ0\| ≥ 0.4); the break at ρ = 1 (dashed), saturation (dotted, shaded beyond) |
| `summary.json` | every number in the figures, per Δv |

## Results (iteration 14,000; PRs are medians over 100 runs)

| Δv | ρ | f(R) | capturable (ICs) | captured | captured / capturable | runs capturing all | PR(Z) E / P | PR(H) E / P | Gram PR E / P | PR(H) unc E / P | η E / P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.28 | 0.062 | 0.000 | 0.003 | – | 100 | 1.43 / 1.06 | 4.52 / 2.34 | 1.93 / 1.93 | 1.51 / 1.35 | 1.00 / 1.00 |
| 0.10 | 0.48 | 0.181 | 0.050 | 0.050 | 1.000 | 100 | 1.41 / 1.32 | 4.79 / 6.96 | 1.93 / 1.93 | 1.70 / 4.22 | 1.00 / 1.00 |
| 0.15 | 0.68 | 0.363 | 0.231 | 0.231 | 0.997 | 99 | 1.53 / 1.62 | 5.13 / 7.59 | 1.95 / 1.95 | 1.93 / 4.94 | 1.00 / 1.00 |
| **0.20** | **0.88** | **0.608** | **0.512** | **0.464** | **0.904** | **44** | **1.62 / 1.92** | **5.09 / 7.52** | **1.99 / 2.00** | **2.18 / 4.55** | **0.98 / 0.98** |
| 0.25 | 1.08 | 0.872 | 0.844 | 0.657 | 0.779 | 3 | 1.86 / 2.06 | 5.31 / 7.63 | 2.10 / 2.16 | 2.52 / 4.59 | 0.94 / 0.92 |
| 0.30 | 1.28 | 0.981 | 0.975 | 0.809 | 0.830 | 8 | 1.82 / 2.08 | 5.31 / 7.60 | 2.23 / 2.33 | 2.66 / 4.89 | 0.92 / 0.89 |
| 0.35 | 1.48 | 1.000 | 1.000 | 0.879 | 0.879 | 16 | 2.05 / 2.14 | 5.49 / 7.72 | 2.40 / 2.53 | 2.71 / 4.84 | 0.89 / 0.83 |
| 0.40 | 1.68 | 1.000 | 1.000 | 0.911 | 0.911 | 25 | 1.98 / 1.96 | 5.52 / 7.65 | 2.52 / 2.82 | 2.80 / 5.11 | 0.84 / 0.74 |
| 0.45 | 1.88 | 1.000 | 1.000 | 0.949 | 0.949 | 42 | 2.14 / 2.31 | 5.68 / 8.14 | 2.71 / 3.09 | 2.94 / 5.50 | 0.80 / 0.69 |
| 0.50 | 2.08 | 1.000 | 1.000 | 0.963 | 0.963 | 57 | 2.42 / 2.31 | 5.94 / 8.33 | 2.76 / 3.39 | 2.97 / 5.99 | 0.78 / 0.65 |

### Against `dv_D1` (lr10 = this sweep, lr 1e-2 / 1e-3; lr1 = `dv_D1`, lr 1e-3 / 1e-4; same seeds, same ICs, same initial weights)

| Δv | captured / capturable lr10 / lr1 | runs capturing all lr10 / lr1 | PR(Z) E lr10 / lr1 | PR(Z) P lr10 / lr1 | PR(H) E lr10 / lr1 | PR(H) P lr10 / lr1 | PR(H) unc E lr10 / lr1 | PR(H) unc P lr10 / lr1 | Gram PR P lr10 / lr1 | η P lr10 / lr1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | – | 100 / 100 | 1.43 / 2.05 | 1.06 / 2.08 | 4.52 / 5.41 | 2.34 / 4.76 | 1.51 / 2.73 | 1.35 / 2.05 | 1.93 / 1.93 | 1.00 / 1.00 |
| 0.10 | 1.000 / 1.000 | 100 / 100 | 1.41 / 2.18 | 1.32 / 2.43 | 4.79 / 5.33 | 6.96 / 5.00 | 1.70 / 2.65 | 4.22 / 2.27 | 1.93 / 1.93 | 1.00 / 1.00 |
| 0.15 | 0.997 / 1.000 | 99 / 100 | 1.53 / 2.34 | 1.62 / 2.53 | 5.13 / 5.51 | 7.59 / 5.07 | 1.93 / 2.73 | 4.94 / 2.38 | 1.95 / 1.94 | 1.00 / 1.00 |
| **0.20** | **0.904 / 0.913** | **44 / 49** | **1.62 / 2.68** | **1.92 / 2.63** | **5.09 / 5.56** | **7.52 / 5.79** | **2.18 / 2.84** | **4.55 / 2.53** | **2.00 / 1.97** | **0.98 / 0.99** |
| 0.25 | 0.779 / 0.853 | 3 / 12 | 1.86 / 2.94 | 2.06 / 2.26 | 5.31 / 6.00 | 7.63 / 6.15 | 2.52 / 3.16 | 4.59 / 3.07 | 2.16 / 2.10 | 0.92 / 0.94 |
| 0.30 | 0.830 / 0.910 | 8 / 25 | 1.82 / 2.83 | 2.08 / 2.17 | 5.31 / 6.30 | 7.60 / 6.50 | 2.66 / 3.43 | 4.89 / 3.48 | 2.33 / 2.45 | 0.89 / 0.85 |
| 0.35 | 0.879 / 0.964 | 16 / 57 | 2.05 / 2.60 | 2.14 / 2.01 | 5.49 / 6.72 | 7.72 / 6.98 | 2.71 / 3.83 | 4.84 / 4.02 | 2.53 / 2.87 | 0.83 / 0.73 |
| 0.40 | 0.911 / 0.979 | 25 / 73 | 1.98 / 2.58 | 1.96 / 2.15 | 5.52 / 7.14 | 7.65 / 7.32 | 2.80 / 4.46 | 5.11 / 4.60 | 2.82 / 3.35 | 0.74 / 0.63 |
| 0.45 | 0.949 / 0.996 | 42 / 94 | 2.14 / 2.45 | 2.31 / 2.21 | 5.68 / 7.47 | 8.14 / 7.71 | 2.94 / 5.09 | 5.50 / 5.15 | 3.09 / 3.77 | 0.69 / 0.55 |
| 0.50 | 0.963 / 0.998 | 57 / 97 | 2.42 / 2.56 | 2.31 / 2.22 | 5.94 / 7.85 | 8.33 / 7.94 | 2.97 / 5.76 | 5.99 / 5.69 | 3.39 / 4.30 | 0.65 / 0.43 |

### Two checks on the checkpoints

**Headings held fixed after capture.** Each captured head's heading is held at its last value before T_c, and the
Gram PR is recomputed over the whole window. "After capture" is the share of head-time after capture (uncaptured
heads count as 0).

| Δv | after capture | Gram PR P: plotted / frozen | Gram PR E: plotted / frozen |
|---|---|---|---|
| 0.05 | 0% | 1.93 / 1.93 | 1.93 / 1.93 |
| 0.10 | 1% | 1.93 / 1.93 | 1.93 / 1.93 |
| 0.15 | 5% | 1.95 / 1.93 | 1.95 / 1.94 |
| **0.20** | **12%** | **2.00 / 1.93** | **1.99 / 1.96** |
| 0.25 | 20% | 2.16 / 1.93 | 2.10 / 2.00 |
| 0.30 | 27% | 2.33 / 1.95 | 2.23 / 2.06 |
| 0.35 | 33% | 2.53 / 1.95 | 2.40 / 2.14 |
| 0.40 | 36% | 2.82 / 1.94 | 2.52 / 2.23 |
| 0.45 | 42% | 3.09 / 1.96 | 2.71 / 2.30 |
| 0.50 | 45% | 3.39 / 1.97 | 2.76 / 2.41 |

**Missed capturable heads.** A head goes for the "wrong copy" when the copy of the evader it captures, or gets
closest to, is not the nearest one. The best-response pursuer, which knows the trained evader's path, catches every
missed head at every Δv.

| Δv | missed (of capturable) lr10 / lr1 | missed heads that went for another copy, lr10 | wrong copy (of capturable) lr10 / lr1 | wrong-copy heads that are missed lr10 / lr1 |
|---|---|---|---|---|
| **0.20** | **9.6% / 8.7%** | **77 of 79** | **9.4% / 8.8%** | **100% / 99%** |
| 0.25 | 22.1% / 14.7% | 297 of 299 | 25.9% / 18.4% | 85% / 80% |
| 0.30 | 17.0% / 9.0% | 264 of 265 | 32.6% / 24.4% | 52% / 37% |
| 0.35 | 12.1% / 3.6% | 188 of 194 | 40.4% / 28.4% | 29% / 12.5% |
| 0.40 | 8.9% / 2.1% | 140 of 142 | 47.4% / 30.9% | 18% / 7% |
| 0.50 | 3.8% / 0.2% | 57 of 60 | 49.8% / 31.0% | 7% / 0.6% |

Reading:

- **Δv = 0.20 is `r1_trunc_w0_p3`**, bit-identical (see above).
- **Capture matches `dv_D1` up to the break and is worse past it.**
  - Captured / capturable is 1.000 and 0.997 for Δv ≤ 0.15, and 0.904 at the baseline speed (`dv_D1`: 0.913).
  - It then drops to 0.779 at Δv = 0.25 (0.853) and recovers more slowly: 0.830, 0.879, 0.911, 0.949, 0.963
    at 0.30 to 0.50 (0.910 to 0.998).
  - Runs capturing every capturable head: 3 at 0.25 (12) and 57 at 0.50 (97).
- **The extra misses are wrong copies, more often and farther away.**
  - As in `dv_D1`, at least 94% of the missed capturable heads went for a copy of the evader other than the
    nearest, almost always one out of reach.
  - The faster-learning pursuer goes for a wrong copy more often: 26–50% of capturable heads past the break,
    against 18–31%.
  - It does so even for close starts: 18–41% of starts with |Δ0| < 0.6 at Δv = 0.35, 0.40 and 0.50, against 3–5%.
  - Its wrong copies are also farther: still out of reach for 29% of them at Δv = 0.35 and 7% at 0.50, against
    12.5% and 0.6%.
- **The played strategy is unchanged: constant headings before capture at every Δv.**
  - Holding each captured head's heading fixed after capture brings the pursuer's Gram PR back to 1.93–1.97 at
    every Δv. So the rise past the break (to 3.39 at 0.50) is again motion after capture, which the truncated
    loss does not constrain.
  - The rise is smaller than in `dv_D1` (4.30) because fewer heads are captured, and later: 45% of the head-time
    is after capture at Δv = 0.50, against 54%.
  - For the evader most of the rise goes too (2.76 → 2.41 at Δv = 0.50); the rest is variation that freezing at
    capture does not remove.
- **The pursuer's latent is saturated and large from Δv = 0.10 on, nearly flat in Δv.**
  - At lr_P = 1e-2 the pursuer's input weights move far from their initial values: |B − B0| / |B0| is 3.7, 5.1
    and 5.5 at Δv = 0.10, 0.20 and 0.50, against 0.8–1.0 in `dv_D1`.
  - 79–88% of its tanh pre-activations are saturated (|x| > 2) at those Δv, against 1–2% in `dv_D1`.
  - The saturated units switch sign during the episode: 57–70% of the latent's energy is time variation.
    So centred PR(H) is 7.0–8.3 and uncentred 4.2–6.0.
  - The headings before capture stay constant all the same, because the heads are blind to that motion, as in
    `dv_D1`.
  - At Δv = 0.05, where nothing is capturable, the pursuer's latent is small: PR(H) 2.34 centred and 1.35
    uncentred, input weights moved by 1.8× their initial norm, 45% saturated, and only 15% of the latent's energy
    is time variation. The jump between Δv = 0.05 and 0.10 coincides with the first capturable starts.
- **The evader's latent and both PR(Z) are smaller than in `dv_D1`.**
  - The evader's uncentred PR(H) is 1.5–3.0, against 2.7–5.8, and its centred PR(H) 4.5–5.9, against 5.3–7.9.
  - PR(Z) is 1.4–2.4 for the evader, against 2.0–2.9, and 1.1–2.3 for the pursuer, against 2.0–2.6.
  - With one layer, the learning rate moves every latent statistic in either direction, depending on the player.
- **Conclusion.** Raising the learning rates changes the latent representation and the pursuer's choice of copy,
  and through the latter costs capture past the break. It does not change the form of the played strategy. At both
  learning rates, as at both depths, every head plays a constant straight line until capture. The latent PRs move
  with depth and learning rate while the strategy stays the same, so they cannot be read as its dimensionality.

## Regenerate

```bash
python3 torch_actor/experiments.py dv_D1_lr1e-2
```
```bash
python3 torch_actor/dv_D1_lr1e-2/make_figures.py
```
