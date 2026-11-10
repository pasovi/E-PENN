# `dv_D4` — speed-difference sweep on the `baseline_D4` configuration (5 Nov 2026)

The [`baseline_D4`](../baseline_D4/README.md) architecture and protocol held fixed (4 tanh layers of 16 units,
K = 12, capture-truncated loss, no warm-up, 3 P : 1 E, lr 1e-3 / 1e-4, 14,000 iterations, Adam reset every
3,500), with only the pursuer speed changed: Δv = v_P − v_E ∈ {0.05, 0.10, 0.15, **0.20**, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50},
v_E = 1. Each Δv is a full 100-run grid (10 init seeds × 10 IC sets, the same seeds at every Δv, so the ICs are
identical across the sweep), 70 s per grid on the Mac GPU. **Δv = 0.20 is `baseline_D4`**, and its 100 runs are
bit-identical to it.

Geometry: R = Δv t_max + ε (t_max = 4, ε = 0.08) is the forced-capture radius and ρ = R / (L/2) (L = 2). The
capturable share of uniformly drawn ICs is f(R), the area of the radius-R disc clipped to the torus cell. It has a
break at ρ = 1 (Δv = 0.230), where the disc first touches the cell edges, and saturates at ρ = √2 (Δv = 0.334),
where it covers the whole cell. The IC sampler rejects |Δ0| < 0.4, so the training ICs follow
f(R | |Δ0| ≥ 0.4) = (4 f(R) − π 0.4²) / (4 − π 0.4²) instead. Both curves are drawn in the capture figure.

## Contents

| | |
|---|---|
| `dv<dv>/ckpt/`, `dv<dv>/results/` | one 100-run grid per Δv, in the `baseline_D4` layout (rev2 records, checkpoints at 3,500 / 7,000 / 10,500 / 14,000) |
| `make_figures.py` | the two figures and `summary.json` |
| `figs/pr_vs_dv.png` | participation ratios vs Δv, evader and pursuer, median [IQR] over 100 runs: (a) the draft's strategy PR(Z) (`tpr_Z`) and centred latent PR(H) (`pr_H`); (b) the gauge-invariant Gram PR and uncentred PR(H). Δv = 0.2 ringed |
| `figs/capture_vs_dv.png` | capturable share of the training ICs, captured share, captured / capturable, against f(R) and f(R \| \|Δ0\| ≥ 0.4); the break at ρ = 1 (dashed), saturation (dotted, shaded beyond) |
| `summary.json` | every number in the figures, per Δv |

## Results (iteration 14,000; PRs are medians over 100 runs)

| Δv | ρ | f(R) | capturable (ICs) | captured | captured / capturable | runs capturing all | PR(Z) E / P | PR(H) E / P | Gram PR E / P | PR(H) unc E / P | η E / P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.28 | 0.062 | 0.000 | 0.023 | – | 100 | 1.46 / 1.15 | 1.58 / 1.19 | 1.92 / 1.92 | 1.01 / 1.00 | 1.00 / 1.00 |
| 0.10 | 0.48 | 0.181 | 0.050 | 0.050 | 1.000 | 100 | 1.49 / 1.18 | 1.72 / 1.21 | 1.92 / 1.92 | 1.00 / 1.00 | 1.00 / 1.00 |
| 0.15 | 0.68 | 0.363 | 0.231 | 0.231 | 1.000 | 100 | 1.47 / 1.17 | 1.95 / 1.23 | 1.94 / 1.94 | 1.01 / 1.00 | 1.00 / 1.00 |
| **0.20** | **0.88** | **0.608** | **0.512** | **0.470** | **0.917** | **49** | **1.63 / 1.12** | **1.78 / 1.21** | **1.92 / 1.92** | **1.01 / 1.00** | **1.00 / 1.00** |
| 0.25 | 1.08 | 0.872 | 0.844 | 0.706 | 0.836 | 8 | 1.52 / 1.15 | 1.76 / 1.24 | 1.94 / 1.94 | 1.01 / 1.01 | 1.00 / 1.00 |
| 0.30 | 1.28 | 0.981 | 0.975 | 0.868 | 0.890 | 21 | 1.52 / 1.21 | 2.06 / 1.35 | 1.93 / 1.94 | 1.02 / 1.03 | 1.00 / 1.00 |
| 0.35 | 1.48 | 1.000 | 1.000 | 0.949 | 0.949 | 41 | 1.65 / 1.25 | 2.27 / 1.65 | 1.95 / 1.98 | 1.04 / 1.06 | 1.00 / 0.98 |
| 0.40 | 1.68 | 1.000 | 1.000 | 0.967 | 0.967 | 59 | 1.56 / 1.32 | 2.35 / 1.79 | 1.98 / 2.13 | 1.06 / 1.15 | 0.99 / 0.91 |
| 0.45 | 1.88 | 1.000 | 1.000 | 0.986 | 0.986 | 81 | 1.41 / 1.30 | 2.24 / 1.86 | 2.04 / 2.36 | 1.10 / 1.25 | 0.95 / 0.78 |
| 0.50 | 2.08 | 1.000 | 1.000 | 0.990 | 0.990 | 86 | 1.42 / 1.38 | 2.42 / 2.07 | 2.17 / 2.52 | 1.18 / 1.36 | 0.88 / 0.71 |

Reading:

- **Capture follows the geometry.** The capturable share of the training ICs sits on the conditioned analytic
  curve f(R | |Δ0| ≥ 0.4), not on f(R) itself (the gap is the IC sampler's exclusion disc). The captured share
  tracks it from below, with the f(R) kink at ρ = 1. Every training IC is capturable from Δv = 0.35 on.
- **The pursuer is perfect well below the break and near-perfect past saturation. The losses sit in between.**
  Captured / capturable is 1.000 for Δv ≤ 0.15, drops to 0.917 at Δv = 0.20, to 0.836 at Δv = 0.25 (just past
  ρ = 1) and 0.890 at 0.30. It then climbs steadily once every IC is capturable: 0.949, 0.967, 0.986, 0.990 at
  Δv = 0.35 to 0.50. Runs that capture every capturable head fall from 100/100 to 8/100 at Δv = 0.25, then recover
  to 41, 59, 81 and 86.
- At Δv = 0.05 no IC is capturable, yet 37 of the 1,600 heads (2.3%) are captured: the trained evader fails to
  escape in those.
- **The embedding is flat across the ρ = 1 break.** Up to Δv = 0.30 the headings are exactly constant
  (η = 1.00), the Gram PR stays at 1.92–1.94 (below the constant-heading bound of 2) and the uncentred PR(H) at
  1.00–1.03. The draft's PR(Z) and centred PR(H) fluctuate (evader PR(H) 1.6–2.1) without a trend at the break.
- **Dimensionality grows from saturation on, steadily, the pursuer's the most.** The pursuer's centred PR(H)
  starts to lift at Δv = 0.30 (1.35, against 1.19–1.24 before) and climbs through 1.65, 1.79, 1.86 to 2.07 at 0.50.
  Its headings stop being constant (η 0.98, 0.91, 0.78, 0.71 at Δv = 0.35 to 0.50), and its Gram PR crosses the
  constant-heading bound of 2 between Δv = 0.35 (1.98) and 0.40 (2.13), reaching 2.52 at 0.50. The evader follows
  the same pattern more weakly (Gram PR 2.17 and η 0.88 at 0.50).
  The time variation is all after capture (checked in
  [`dv_D1/README.md`](../dv_D1/README.md#before-and-after-capture)). Before T_c the headings are constant at
  every Δv ≥ 0.10: median η 1.000, at most 2% of the pursuer's captured heads with η < 0.99. After T_c the
  share of the pursuer's captured heads whose heading varies grows from 2% at Δv = 0.20 to 35% at 0.35 and 88%
  at 0.50. Under the truncated loss nothing constrains the headings after capture.

## Regenerate

```bash
python3 torch_actor/experiments.py dv_D4
```
```bash
python3 torch_actor/dv_D4/make_figures.py
```
