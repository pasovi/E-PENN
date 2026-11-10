# `dv_D4_precap`: the `dv_D4` sweep with the participation ratios taken before capture (10 Nov 2026)

A re-run of [`dv_D4`](../dv_D4/README.md). The architecture and protocol are those of [`baseline_D4`](../baseline_D4/README.md):
4 tanh layers of 16 units, K = 12, capture-truncated loss, no warm-up, 3 P : 1 E, lr 1e-3 / 1e-4, 14,000 iterations,
Adam reset every 3,500. Only the pursuer speed changes: Δv = v_P − v_E ∈ {0.05, 0.10, 0.15, **0.20**, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50},
v_E = 1. These are the eight requested values plus 0.35 and 0.40, as in the other sweeps.

Each Δv is a full 100-run grid (10 init seeds × 10 IC sets), with the same seeds as `dv_D4`, at about 90 s per grid on the
Mac GPU. **All 1,000 runs are bit-identical to `dv_D4`'s:** every array of every checkpoint is equal. Δv = 0.20 is
`baseline_D4`.

What is new is how the participation ratios are computed: over the time before capture only. After capture nothing a
head does enters the game, nor the capture-truncated loss.

Geometry: R = Δv t_max + ε (t_max = 4, ε = 0.08) is the forced-capture radius and ρ = R / (L/2) (L = 2).
- The capturable share of uniformly drawn ICs is f(R), the area of the radius-R disc clipped to the torus cell.
- f(R) has a break at ρ = 1 (Δv = 0.230) and saturates at ρ = √2 (Δv = 0.334).
- The IC sampler rejects |Δ0| < 0.4, so the training ICs follow f(R | |Δ0| ≥ 0.4) = (4 f(R) − π 0.4²) / (4 − π 0.4²) instead.

## The participation ratios with the time after capture removed

Head j, one evader initial condition, is **in play** on A_j = {t ≤ T_c^j}. T_c^j is its capture time: the first grid
time with d ≤ ε, `game.Tc` of the records. A head that is never captured is in play on the whole of [0, t_max]. Every
time average is restricted to the time in play ([`precap.py`](precap.py)):

- **Gram PR:** G_jk = ⟨u^j · u^k⟩ over A_j ∩ A_k, the time both heads are in play, and PR = (tr G)² / tr(G²).
  - The diagonal is 1, so each head counts once, as in the whole-episode Gram.
  - Headings that are constant while in play give G_jk = cos(θ_j − θ_k) exactly, so the constant-heading bound,
    Gram PR = 2 / (1 + r²) ≤ 2, still applies.
  - Simply setting each heading to zero after capture would count differences in capture time as extra dimensions.
    On synthetic constant-heading ensembles with random capture times, that version overshoots 2 / (1 + r²) by 0.55 in
    the median, and by up to 1.2.
- **η:** (1/n_h) Σ_j |⟨u^j⟩ over A_j|². It equals 1 iff every heading is constant while in play.
- **PR(Z)** (the draft's `tpr_Z`): the PR of the covariance of the in-play head outputs. Head j contributes
  z^j(t) − ⟨z^j⟩ over A_j for t in A_j, and nothing after capture.
- **PR(H), centred and uncentred:** the PR of the latent's covariance (or second moment) over the in-play samples
  (j, t), t ∈ A_j. These are the time points, each weighted by the number of heads still in play.

All four equal the whole-episode statistics when no head is captured. This was checked on the 92 runs at Δv = 0.05
without a capture, where they agree to 1e-8. The whole-episode values of the same runs, which are those of `dv_D4`, are
drawn faintly in the figure.

## Contents

| | |
|---|---|
| `dv<dv>/ckpt/`, `dv<dv>/results/` | one 100-run grid per Δv, in the `dv_D4` layout (rev2 records, checkpoints at 3,500 / 7,000 / 10,500 / 14,000) |
| `dv<dv>/precap.json` | the pre-capture statistics of every run of the grid |
| `precap.py` | the definitions above, plus the alternatives of the last table; writes `precap.json` |
| `make_figures.py` | the two figures and `summary.json` (runs `precap.py` where `precap.json` is missing) |
| `figs/pr_vs_dv.png` | participation ratios before capture vs Δv, median [IQR] over 100 runs, both players, with the same runs over the whole episode drawn faintly. Top row, the strategy ensemble: (a) PR(Z), (b) Gram PR. Bottom row, the latent space: (c) centred PR(H), (d) uncentred PR(H). Δv = 0.2 ringed |
| `figs/pr_vs_dv_before_capture.png` | the before-capture lines alone, in the two-panel layout of [`dv_D4/figs/pr_vs_dv.png`](../dv_D4/figs/pr_vs_dv.png). (a) the draft's pair: strategy PR(Z) (circles, solid) and centred latent PR(H) (squares, dashed). (b) their gauge-invariant counterparts: Gram PR and uncentred PR(H). Δv = 0.2 ringed |
| `figs/capture_vs_dv.png` | capturable share of the training ICs, captured share, and captured / capturable, against f(R) and f(R \| \|Δ0\| ≥ 0.4). The break at ρ = 1 is dashed and saturation dotted, with the region beyond it shaded. Identical to `dv_D4`'s |
| `summary.json` | every number in the figures and tables, per Δv |

## Results (iteration 14,000)

Capture is that of `dv_D4`, since these are the same runs:

| Δv | ρ | f(R) | capturable (ICs) | captured | captured / capturable | runs capturing all | head-time after capture |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.28 | 0.062 | 0.000 | 0.023 | – | 100 | 1% |
| 0.10 | 0.48 | 0.181 | 0.050 | 0.050 | 1.000 | 100 | 1% |
| 0.15 | 0.68 | 0.363 | 0.231 | 0.231 | 1.000 | 100 | 5% |
| **0.20** | **0.88** | **0.608** | **0.512** | **0.470** | **0.917** | **49** | **12%** |
| 0.25 | 1.08 | 0.872 | 0.844 | 0.706 | 0.836 | 8 | 20% |
| 0.30 | 1.28 | 0.981 | 0.975 | 0.868 | 0.890 | 21 | 29% |
| 0.35 | 1.48 | 1.000 | 1.000 | 0.949 | 0.949 | 41 | 36% |
| 0.40 | 1.68 | 1.000 | 1.000 | 0.967 | 0.967 | 59 | 41% |
| 0.45 | 1.88 | 1.000 | 1.000 | 0.986 | 0.986 | 81 | 46% |
| 0.50 | 2.08 | 1.000 | 1.000 | 0.990 | 0.990 | 86 | 51% |

"Head-time after capture" is the share of all (head, time) pairs that come after the head's capture, i.e. what the
pre-capture statistics leave out.

Participation ratios before capture, medians over 100 runs, with the whole episode in brackets:

| Δv | PR(Z) E / P | PR(H) E / P | Gram PR E / P | PR(H) unc E / P | η E / P |
|---|---|---|---|---|---|
| 0.05 | 1.47 / 1.17 (1.46 / 1.15) | 1.58 / 1.19 (1.58 / 1.19) | 1.92 / 1.92 (1.92 / 1.92) | 1.01 / 1.00 (1.01 / 1.00) | 1.000 / 1.000 (1.00 / 1.00) |
| 0.10 | 1.49 / 1.19 (1.49 / 1.18) | 1.72 / 1.21 (1.72 / 1.21) | 1.92 / 1.92 (1.92 / 1.92) | 1.00 / 1.00 (1.00 / 1.00) | 1.000 / 1.000 (1.00 / 1.00) |
| 0.15 | 1.58 / 1.18 (1.47 / 1.17) | 1.90 / 1.22 (1.95 / 1.23) | 1.94 / 1.94 (1.94 / 1.94) | 1.01 / 1.00 (1.01 / 1.00) | 1.000 / 1.000 (1.00 / 1.00) |
| **0.20** | **1.70 / 1.15 (1.63 / 1.12)** | **1.75 / 1.22 (1.78 / 1.21)** | **1.92 / 1.92 (1.92 / 1.92)** | **1.01 / 1.00 (1.01 / 1.00)** | **1.000 / 1.000 (1.00 / 1.00)** |
| 0.25 | 1.76 / 1.26 (1.52 / 1.15) | 1.69 / 1.22 (1.76 / 1.24) | 1.93 / 1.93 (1.94 / 1.94) | 1.01 / 1.01 (1.01 / 1.01) | 1.000 / 1.000 (1.00 / 1.00) |
| 0.30 | 2.02 / 1.48 (1.52 / 1.21) | 1.94 / 1.32 (2.06 / 1.35) | 1.92 / 1.92 (1.93 / 1.94) | 1.02 / 1.02 (1.02 / 1.03) | 1.000 / 1.000 (1.00 / 1.00) |
| 0.35 | 2.00 / 1.85 (1.65 / 1.25) | 2.20 / 1.62 (2.27 / 1.65) | 1.93 / 1.93 (1.95 / 1.98) | 1.02 / 1.03 (1.04 / 1.06) | 1.000 / 1.000 (1.00 / 0.98) |
| 0.40 | 2.20 / 1.97 (1.56 / 1.32) | 2.09 / 1.75 (2.35 / 1.79) | 1.91 / 1.91 (1.98 / 2.13) | 1.04 / 1.08 (1.06 / 1.15) | 0.999 / 1.000 (0.99 / 0.90) |
| 0.45 | 2.50 / 2.12 (1.41 / 1.30) | 2.37 / 2.00 (2.24 / 1.86) | 1.92 / 1.91 (2.04 / 2.36) | 1.05 / 1.11 (1.10 / 1.25) | 0.998 / 1.000 (0.95 / 0.78) |
| 0.50 | 2.96 / 2.20 (1.42 / 1.38) | 2.64 / 2.14 (2.42 / 2.07) | 1.92 / 1.92 (2.17 / 2.52) | 1.07 / 1.14 (1.18 / 1.36) | 0.996 / 0.999 (0.88 / 0.71) |

The latent's motion while heads are in play (medians). The first column is the in-play time variation as a share of
the latent's second moment, tr C_H / (tr C_H + |⟨H⟩|²). The second is the centred PR(H) over the first quarter of the
episode, t ≤ 1:

| Δv | latent time variation E / P | centred PR(H) over t ≤ 1, E / P |
|---|---|---|
| 0.05 | 0.3% / 0.1% | 1.35 / 1.15 |
| 0.10 | 0.3% / 0.1% | 1.43 / 1.14 |
| 0.15 | 0.7% / 0.2% | 1.33 / 1.11 |
| **0.20** | **0.5% / 0.2%** | **1.26 / 1.12** |
| 0.25 | 0.7% / 0.5% | 1.21 / 1.10 |
| 0.30 | 1.0% / 1.1% | 1.23 / 1.12 |
| 0.35 | 1.6% / 2.2% | 1.26 / 1.16 |
| 0.40 | 2.6% / 5.2% | 1.44 / 1.17 |
| 0.45 | 3.6% / 6.7% | 1.56 / 1.21 |
| 0.50 | 4.8% / 8.7% | 1.75 / 1.25 |

Reading:

- **The strategy played before capture is the same at every Δv.**
  - Before capture the Gram PR is 1.91–1.94 for both players at every Δv, below the constant-heading bound of 2, and
    η ≥ 0.996.
  - The whole-episode rise of the Gram PR (to 2.52 for the pursuer and 2.17 for the evader at Δv = 0.5) disappears
    completely, and so does the drop of η (to 0.71 / 0.88). Both were heading motion after capture.
  - Each head plays a constant heading until it is caught, across the ρ = 1 break and past saturation.
- **The latent moves a little before capture at high Δv, in directions the headings do not read.**
  - The pre-capture uncentred PR(H) stays at 1.00–1.02 up to Δv = 0.30. It then rises to 1.14 for the pursuer and 1.07
    for the evader at Δv = 0.5, against 1.36 / 1.18 over the whole episode. So removing the post-capture time takes away
    about 60% of the rise.
  - The rest is motion while heads are in play. The latent's time variation is at most 0.7% of its second moment up to
    Δv = 0.25, and grows to 8.7% (pursuer) and 4.8% (evader) at Δv = 0.5.
  - The headings still in play stay constant through it (η ≥ 0.996), so they do not read this motion.
- **The centred PR(H) is not changed by the removal:** 2.14 for the pursuer and 2.64 for the evader at Δv = 0.5, against
  2.07 / 2.42 over the whole episode.
  - It is a scale-free measure: it describes the shape of the latent's motion however small that motion is.
  - Over the first quarter of the episode, the pursuer's latent moves along essentially one direction at every Δv
    (centred PR(H) 1.10–1.25). The evader's centred PR(H) over the same quarter is 1.2–1.75.
  - So the extra directions come in later in the episode, as heads are caught. A likely reason, not checked: with fewer
    heads in play, there are fewer headings for the latent to hold constant.
- **PR(Z) rises before capture, more than over the whole episode:** 1.47 → 2.96 for the evader and 1.17 → 2.20 for the
  pursuer, against 1.42 / 1.38 over the whole episode at Δv = 0.5. This is not strategy:
  - The headings are constant while in play, so all of z^j's in-play variation is in its length |z^j|, which the
    normalisation u = z / |z| removes.
  - Each head contributes that variation over its own time in play, which differs from head to head, so a single shared
    modulation of |z| shows up as several modes.
  - How large the rise is depends on how the different windows are combined (next table). At Δv = 0.5 the pursuer's
    pre-capture PR(Z) ranges from 1.44 to 2.40 depending on that choice, and the evader's from 1.64 to 3.33. The four
    definitions agree only where almost nothing is captured (Δv ≤ 0.10).

How much the result depends on the way the post-capture time is removed (medians; the first entry of each group is the
plotted one):

| Δv | Gram PR P: in play / own clock / frozen / mean headings | Gram PR E: same | PR(Z) P: plotted / pairwise / own clock / latent-projected | PR(Z) E: same |
|---|---|---|---|---|
| 0.05 | 1.92 / 1.92 / 1.92 / 1.92 | 1.92 / 1.92 / 1.92 / 1.92 | 1.17 / 1.16 / 1.18 / 1.15 | 1.47 / 1.46 / 1.47 / 1.46 |
| 0.10 | 1.92 / 1.92 / 1.92 / 1.92 | 1.92 / 1.92 / 1.92 / 1.92 | 1.19 / 1.17 / 1.20 / 1.18 | 1.49 / 1.49 / 1.57 / 1.49 |
| 0.15 | 1.94 / 1.94 / 1.94 / 1.94 | 1.94 / 1.94 / 1.94 / 1.94 | 1.18 / 1.13 / 1.26 / 1.16 | 1.58 / 1.48 / 1.71 / 1.45 |
| **0.20** | **1.92 / 1.92 / 1.92 / 1.92** | **1.92 / 1.92 / 1.92 / 1.92** | **1.15 / 1.07 / 1.22 / 1.10** | **1.70 / 1.41 / 2.21 / 1.58** |
| 0.25 | 1.93 / 1.93 / 1.93 / 1.93 | 1.93 / 1.93 / 1.94 / 1.93 | 1.26 / 1.12 / 1.37 / 1.16 | 1.76 / 1.37 / 2.18 / 1.50 |
| 0.30 | 1.92 / 1.92 / 1.92 / 1.92 | 1.92 / 1.92 / 1.93 / 1.92 | 1.48 / 1.30 / 1.55 / 1.19 | 2.02 / 1.71 / 2.90 / 1.59 |
| 0.35 | 1.93 / 1.93 / 1.92 / 1.92 | 1.93 / 1.93 / 1.94 / 1.92 | 1.85 / 1.60 / 2.22 / 1.34 | 2.00 / 1.72 / 2.85 / 1.67 |
| 0.40 | 1.91 / 1.92 / 1.92 / 1.91 | 1.91 / 1.92 / 1.95 / 1.91 | 1.97 / 1.71 / 2.23 / 1.42 | 2.20 / 2.01 / 2.64 / 1.68 |
| 0.45 | 1.91 / 1.91 / 1.91 / 1.91 | 1.92 / 1.93 / 1.97 / 1.91 | 2.12 / 1.80 / 2.46 / 1.43 | 2.50 / 2.20 / 2.99 / 1.55 |
| 0.50 | 1.92 / 1.92 / 1.92 / 1.91 | 1.92 / 1.93 / 2.01 / 1.90 | 2.20 / 1.92 / 2.40 / 1.44 | 2.96 / 2.46 / 3.33 / 1.64 |

- *Own clock:* each head's time in play is rescaled to [0, 1] before the whole-window statistic is taken.
- *Frozen:* each captured heading is held at its value at capture for the rest of the episode, as in the check of
  [`dv_D1/README.md`](../dv_D1/README.md#before-and-after-capture).
- *Mean headings:* the Gram PR of the in-play mean headings, 2 / (1 + r²).
- *Pairwise:* the covariance of z^j and z^k over the time both heads are in play.
- *Latent-projected:* PR(W C_H Wᵀ), with C_H the in-play latent covariance of PR(H). That is, the stacked outputs over
  the same samples as PR(H); it keeps the outputs of heads already captured.

Notes on this table:

- **The Gram PR is robust.** The four constructions agree to ±0.01 for the pursuer at every Δv, and for the evader up to
  Δv = 0.35.
  - The frozen variant reads the evader higher at Δv ≥ 0.40 (up to 2.01 at 0.50). Holding a heading at its value at
    capture stretches a last-moment turn of the evader over the rest of the episode.
  - The in-play Gram need not be positive semidefinite. Its lowest eigenvalue is within 0.08 of zero (out of a trace of
    16) in the median run at every Δv.
  - It reaches −1.5 only in the few runs at Δv = 0.05 where the evader's training failed: 11–12 of the 16 heads are
    caught although none is capturable, and the headings swing while in play.
- **PR(Z) is not robust,** for the reasons given above.

## Conclusion

Removing the time after capture leaves the strategy at every Δv as it is at the baseline speed: constant headings, with
a Gram PR of 1.92 across the ρ = 1 break and past saturation.

The whole-episode rise of the Gram PR, the drop of η, and about 60% of the rise of the uncentred PR(H) were all motion
after capture. What remains of the latent's rise is body motion while heads are in play, which the headings do not
read. PR(Z)'s pre-capture rise is variation of |z| (the gauge) counted over heads in play for different lengths of time.

Of the four statistics, only the Gram PR (with η) measures the strategy, and before capture it is flat in Δv.

## Regenerate

```bash
python3 torch_actor/experiments.py dv_D4_precap
```
```bash
python3 torch_actor/dv_D4_precap/make_figures.py
```

`make_figures.py` runs `precap.py` for any grid without a `precap.json`. To recompute the pre-capture statistics,
delete the `precap.json` files, or run `python3 torch_actor/dv_D4_precap/precap.py` after deleting them.
