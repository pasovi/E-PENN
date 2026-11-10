# `dv_D1` — speed-difference sweep with a 1-layer body (10 Nov 2026)

The [`dv_D4`](../dv_D4/README.md) sweep with the body cut from 4 tanh layers to **1** (16 units). Everything
else is unchanged: K = 12, capture-truncated loss, no warm-up, 3 P : 1 E, lr 1e-3 / 1e-4, 14,000 iterations,
Adam reset every 3,500. Δv = v_P − v_E ∈ {0.05, 0.10, 0.15, **0.20**, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50},
v_E = 1, the same ten points as `dv_D4`. Each Δv is a full 100-run grid (10 init seeds × 10 IC sets). The seeds
are those of `dv_D4`, so the ICs, and with them the capturable shares, are identical in the two sweeps.
About 45 s per grid on the Mac GPU (MPS, float32).

**Δv = 0.20 (v_P = 1.2) is the baseline speed**, the one-layer counterpart of `baseline_D4`, and is ringed in
both figures. No earlier run has this exact configuration. The one-layer grids under `checks/`
(`r1_trunc`, `r1_trunc_w0_p3`) use the 10× larger learning rates 1e-2 / 1e-3; `r1_trunc_w0_p3` differs from
`dv_D1/dv0.20` only in that.

Geometry (as in `dv_D4`): R = Δv t_max + ε (t_max = 4, ε = 0.08) is the forced-capture radius and
ρ = R / (L/2) (L = 2). The capturable share of uniformly drawn ICs is f(R), the area of the radius-R disc clipped
to the torus cell. It has a break at ρ = 1 (Δv = 0.230), where the disc first touches the cell edges, and
saturates at ρ = √2 (Δv = 0.334), where it covers the whole cell. The IC sampler rejects |Δ0| < 0.4, so the
training ICs follow f(R | |Δ0| ≥ 0.4) = (4 f(R) − π 0.4²) / (4 − π 0.4²) instead.

## Contents

| | |
|---|---|
| `dv<dv>/ckpt/`, `dv<dv>/results/` | one 100-run grid per Δv, in the `baseline_D4` layout (rev2 records, checkpoints at 3,500 / 7,000 / 10,500 / 14,000) |
| `make_figures.py` | the two figures and `summary.json` (a copy of `dv_D4/make_figures.py`; it reads the body depth from the checkpoints) |
| `figs/pr_vs_dv.png` | participation ratios vs Δv, evader and pursuer, median [IQR] over 100 runs: (a) the draft's strategy PR(Z) (`tpr_Z`) and centred latent PR(H) (`pr_H`); (b) the gauge-invariant Gram PR and uncentred PR(H). Δv = 0.2 ringed |
| `figs/capture_vs_dv.png` | capturable share of the training ICs, captured share, captured / capturable, against f(R) and f(R \| \|Δ0\| ≥ 0.4); the break at ρ = 1 (dashed), saturation (dotted, shaded beyond) |
| `summary.json` | every number in the figures, per Δv |

## Results (iteration 14,000; PRs are medians over 100 runs)

| Δv | ρ | f(R) | capturable (ICs) | captured | captured / capturable | runs capturing all | PR(Z) E / P | PR(H) E / P | Gram PR E / P | PR(H) unc E / P | η E / P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.28 | 0.062 | 0.000 | 0.006 | – | 100 | 2.05 / 2.08 | 5.41 / 4.76 | 1.93 / 1.93 | 2.73 / 2.05 | 1.00 / 1.00 |
| 0.10 | 0.48 | 0.181 | 0.050 | 0.050 | 1.000 | 100 | 2.18 / 2.43 | 5.33 / 5.00 | 1.93 / 1.93 | 2.65 / 2.27 | 1.00 / 1.00 |
| 0.15 | 0.68 | 0.363 | 0.231 | 0.232 | 1.000 | 100 | 2.34 / 2.53 | 5.51 / 5.07 | 1.94 / 1.94 | 2.73 / 2.38 | 1.00 / 1.00 |
| **0.20** | **0.88** | **0.608** | **0.512** | **0.468** | **0.913** | **49** | **2.68 / 2.63** | **5.56 / 5.79** | **1.94 / 1.97** | **2.84 / 2.53** | **1.00 / 0.99** |
| 0.25 | 1.08 | 0.872 | 0.844 | 0.721 | 0.853 | 12 | 2.94 / 2.26 | 6.00 / 6.15 | 1.98 / 2.10 | 3.16 / 3.07 | 0.98 / 0.94 |
| 0.30 | 1.28 | 0.981 | 0.975 | 0.887 | 0.910 | 25 | 2.83 / 2.17 | 6.30 / 6.50 | 2.10 / 2.45 | 3.43 / 3.48 | 0.95 / 0.85 |
| 0.35 | 1.48 | 1.000 | 1.000 | 0.964 | 0.964 | 57 | 2.60 / 2.01 | 6.72 / 6.98 | 2.32 / 2.87 | 3.83 / 4.02 | 0.88 / 0.73 |
| 0.40 | 1.68 | 1.000 | 1.000 | 0.979 | 0.979 | 73 | 2.58 / 2.15 | 7.14 / 7.32 | 2.65 / 3.35 | 4.46 / 4.60 | 0.78 / 0.63 |
| 0.45 | 1.88 | 1.000 | 1.000 | 0.996 | 0.996 | 94 | 2.45 / 2.21 | 7.47 / 7.71 | 3.06 / 3.77 | 5.09 / 5.15 | 0.68 / 0.55 |
| 0.50 | 2.08 | 1.000 | 1.000 | 0.998 | 0.998 | 97 | 2.56 / 2.22 | 7.85 / 7.94 | 3.60 / 4.30 | 5.76 / 5.69 | 0.59 / 0.43 |

### Against `dv_D4` (1L = this sweep, 4L = `dv_D4`; same seeds, same ICs)

| Δv | captured / capturable 1L / 4L | runs capturing all 1L / 4L | PR(Z) P 1L / 4L | PR(H) E 1L / 4L | PR(H) P 1L / 4L | PR(H) unc P 1L / 4L | Gram PR P 1L / 4L | η P 1L / 4L |
|---|---|---|---|---|---|---|---|---|
| 0.05 | – | 100 / 100 | 2.08 / 1.15 | 5.41 / 1.58 | 4.76 / 1.19 | 2.05 / 1.00 | 1.93 / 1.92 | 1.00 / 1.00 |
| 0.10 | 1.000 / 1.000 | 100 / 100 | 2.43 / 1.18 | 5.33 / 1.72 | 5.00 / 1.21 | 2.27 / 1.00 | 1.93 / 1.92 | 1.00 / 1.00 |
| 0.15 | 1.000 / 1.000 | 100 / 100 | 2.53 / 1.17 | 5.51 / 1.95 | 5.07 / 1.23 | 2.38 / 1.00 | 1.94 / 1.94 | 1.00 / 1.00 |
| **0.20** | **0.913 / 0.917** | **49 / 49** | **2.63 / 1.12** | **5.56 / 1.78** | **5.79 / 1.21** | **2.53 / 1.00** | **1.97 / 1.92** | **0.99 / 1.00** |
| 0.25 | 0.853 / 0.836 | 12 / 8 | 2.26 / 1.15 | 6.00 / 1.76 | 6.15 / 1.24 | 3.07 / 1.01 | 2.10 / 1.94 | 0.94 / 1.00 |
| 0.30 | 0.910 / 0.890 | 25 / 21 | 2.17 / 1.21 | 6.30 / 2.06 | 6.50 / 1.35 | 3.48 / 1.03 | 2.45 / 1.94 | 0.85 / 1.00 |
| 0.35 | 0.964 / 0.949 | 57 / 41 | 2.01 / 1.25 | 6.72 / 2.27 | 6.98 / 1.65 | 4.02 / 1.06 | 2.87 / 1.98 | 0.73 / 0.98 |
| 0.40 | 0.979 / 0.967 | 73 / 59 | 2.15 / 1.32 | 7.14 / 2.35 | 7.32 / 1.79 | 4.60 / 1.15 | 3.35 / 2.13 | 0.63 / 0.90 |
| 0.45 | 0.996 / 0.986 | 94 / 81 | 2.21 / 1.30 | 7.47 / 2.24 | 7.71 / 1.86 | 5.15 / 1.25 | 3.77 / 2.36 | 0.55 / 0.78 |
| 0.50 | 0.998 / 0.990 | 97 / 86 | 2.22 / 1.38 | 7.85 / 2.42 | 7.94 / 2.07 | 5.69 / 1.36 | 4.30 / 2.52 | 0.43 / 0.71 |

### Before and after capture

Each captured head split at its capture time T_c, with η = |⟨u⟩|² over t < T_c and over t ≥ T_c (as in
`checks/analyze_r1.capture_windows`; 1,024 time points, windows of at least 16 points). Counts are heads of the
100 runs; "η < 0.99" marks a visibly time-varying heading.

| Δv | captured heads (1L) | pre-capture η median E / P, 1L | pre-capture η < 0.99, E / P: 1L, 4L | post-capture η median P, 1L / 4L | post-capture η < 0.99, P, 1L / 4L |
|---|---|---|---|---|---|
| 0.05 | 9 | 0.704 / 0.964 | 89% / 100%, 68% / 73% | 0.830 / 0.930 | 100% / 70% |
| 0.10 | 80 | 1.000 / 1.000 | 2% / 1%, 0% / 0% | 1.000 / 1.000 | 4% / 0% |
| 0.15 | 371 | 1.000 / 1.000 | 1% / 0%, 0% / 0% | 1.000 / 1.000 | 12% / 1% |
| **0.20** | **749** | **1.000 / 1.000** | **0% / 0%, 0% / 0%** | **1.000 / 1.000** | **28% / 2%** |
| 0.25 | 1153 | 1.000 / 1.000 | 1% / 0%, 1% / 0% | 0.976 / 1.000 | 53% / 2% |
| 0.30 | 1419 | 1.000 / 1.000 | 1% / 0%, 1% / 0% | 0.682 / 1.000 | 77% / 8% |
| 0.35 | 1543 | 1.000 / 1.000 | 1% / 0%, 1% / 0% | 0.471 / 0.999 | 91% / 35% |
| 0.40 | 1566 | 1.000 / 1.000 | 4% / 0%, 2% / 1% | 0.321 / 0.875 | 97% / 59% |
| 0.45 | 1594 | 1.000 / 1.000 | 6% / 0%, 6% / 1% | 0.236 / 0.612 | 98% / 77% |
| 0.50 | 1597 | 1.000 / 1.000 | 9% / 1%, 9% / 2% | 0.147 / 0.508 | 99% / 88% |

Reading:

- **Capture follows the geometry, as in `dv_D4`.** The training ICs are those of `dv_D4` (checked: identical
  x_E0 and run configuration in every run), so the capturable shares are the same and sit on the conditioned
  curve f(R | |Δ0| ≥ 0.4). The captured share tracks it from below, with the f(R) kink at ρ = 1. Every
  training IC is capturable from Δv = 0.35 on.
- **The one-layer pursuer captures as well as the four-layer one, and slightly better past the break.**
  Captured / capturable is 1.000 for Δv ≤ 0.15 and 0.913 at the baseline speed (`dv_D4`: 0.917; 49 runs
  capture every capturable head in both). It dips to 0.853 at Δv = 0.25 (0.836), then climbs faster: 0.910,
  0.964, 0.979, 0.996, 0.998 at Δv = 0.30 to 0.50 (0.890 to 0.990). At Δv = 0.05 no IC is capturable, yet
  9 of the 1,600 heads are captured (`dv_D4`: 37); in those few heads the evader's heading varies before
  capture.
- **The latent is several times higher-dimensional at every Δv, without changing the strategy below the
  break.** Up to Δv = 0.20 the centred PR(H) is 4.8–5.8 and the uncentred one 2.0–2.8 (`dv_D4`: 1.2–2.0 and
  1.00). Yet the headings are constant there (η ≥ 0.99), and the Gram PR is 1.93–1.97, under the
  constant-heading bound of 2. Constant headings do not need the latent to sit on one line. They need each
  head's readout across its own heading to be blind to the directions the latent moves in. That is what the
  trained heads do. Each W^j (2 × 16) is far from rank 1 as a matrix (ratio of its singular values
  0.09–0.36). Restricted to the latent it actually sees (√(λ2/λ1) of W^j ⟨H Hᵀ⟩_t W^jᵀ), it is rank 1
  (0.0003–0.0065 at Δv = 0.05–0.20). The latent's motion therefore changes only the lengths |z^j|.
  PR(Z) (2.0–2.7, against 1.1–1.6 in `dv_D4`) and the centred PR(H) count that motion; the Gram PR does not.
  In `dv_D4` the same constancy comes from the latent itself sitting on a line.
- **The headings start to vary around the ρ = 1 break, about 0.1 in Δv earlier than in `dv_D4`.** The
  pursuer's η falls from 0.99 at Δv = 0.20 to 0.94, 0.85, 0.73, 0.63, 0.55 and 0.43 at 0.25 to 0.50. Its
  Gram PR crosses 2 between Δv = 0.20 and 0.25 and reaches 4.30 at 0.50 (`dv_D4`: crosses between 0.35
  and 0.40, 2.52 at 0.50). The evader follows (its Gram PR crosses 2 between 0.25 and 0.30, 3.60 at 0.50),
  and the latent PRs climb with both.
- **All of that variation is after capture.** Split at T_c, the headings before capture are constant at
  every Δv ≥ 0.10 in both sweeps: median η 1.000, at most 2% of the pursuer's and 9% of the evader's
  captured heads with η < 0.99. The headings after capture are not. The share of the pursuer's captured heads
  whose post-capture heading varies grows from 28% at Δv = 0.20 to 99% at 0.50 (`dv_D4`: 2% to 88%). This
  settles the hypothesis left open in `dv_D4`: under the truncated loss nothing constrains the headings after
  capture. Both bodies use that freedom; the one-layer body uses it from lower Δv and more. The strategy
  actually played, the heading up to capture, is the constant straight-line one at every Δv and both depths.
- **PR(Z) does not track it.** The pursuer's PR(Z) falls past the break (2.63 at Δv = 0.20 to 2.01 at
  0.35) while its headings start to vary. PR(Z) mixes length and direction changes of z, so it is not a
  dimension of the strategy (the same caveat as in `dv_D4`).
- **Learning rate.** At the baseline speed, the same one-layer grid at the 10× larger learning rates
  (`r1_trunc_w0_p3`, lr 1e-2 / 1e-3) has:
  - centred PR(H) 5.09 / 7.52 and uncentred 2.18 / 4.55;
  - Gram PR 1.99 / 2.00 and η 0.98 / 0.98;
  - captured / capturable 0.904.

  Lowering the learning rate shrinks the pursuer's latent (7.5 to 5.8 centred, 4.6 to 2.5 uncentred) and makes
  the headings more constant. The one-layer latent still stays far above the four-layer one.

## Regenerate

```bash
python3 torch_actor/experiments.py dv_D1
```
```bash
python3 torch_actor/dv_D1/make_figures.py
```
