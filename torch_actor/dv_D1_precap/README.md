# `dv_D1_precap`: the `dv_D1` sweep with the participation ratios taken before capture (10 Nov 2026)

[`dv_D4_precap`](../dv_D4_precap/README.md) with a 1-layer body. This is a re-run of [`dv_D1`](../dv_D1/README.md):
1 tanh layer of 16 units, with everything else as in `dv_D4_precap`. That is K = 12, capture-truncated loss, no
warm-up, 3 P : 1 E, lr 1e-3 / 1e-4, 14,000 iterations and Adam reset every 3,500.

Δv = v_P − v_E ∈ {0.05, 0.10, 0.15, **0.20**, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}, with v_E = 1. These are the eight
requested values plus 0.35 and 0.40, as in the other sweeps.

Each Δv is a full 100-run grid (10 init seeds × 10 IC sets), with the same seeds as `dv_D1` and `dv_D4`. **All 1,000
runs are bit-identical to `dv_D1`'s:** every array of every checkpoint is equal. Δv = 0.20 (v_P = 1.2) is the
baseline speed and is ringed in every figure.

The participation ratios are taken over the time before capture only, exactly as in `dv_D4_precap` (the same
[`precap.py`](precap.py); definitions in
[`dv_D4_precap/README.md`](../dv_D4_precap/README.md#the-participation-ratios-with-the-time-after-capture-removed)).
In short, head j is in play until its capture time T_c^j, or for the whole episode if never caught:

- **Gram PR:** each overlap ⟨u^j · u^k⟩ is averaged over the time both heads are in play.
- **η:** each head's mean heading is taken over its own time in play.
- **PR(Z):** each head contributes its centred output up to its own capture.
- **Centred and uncentred PR(H):** computed from the latent samples (j, t) with t ≤ T_c^j.

All four equal the whole-episode statistics when no head is captured.

Geometry (as in `dv_D4`): R = Δv t_max + ε (t_max = 4, ε = 0.08) and ρ = R / (L/2) (L = 2).
- f(R) has its break at ρ = 1 (Δv = 0.230) and saturates at ρ = √2 (Δv = 0.334).
- The training ICs follow f(R | |Δ0| ≥ 0.4).

## Contents

| | |
|---|---|
| `dv<dv>/ckpt/`, `dv<dv>/results/` | one 100-run grid per Δv, in the `dv_D1` layout (rev2 records, checkpoints at 3,500 / 7,000 / 10,500 / 14,000) |
| `dv<dv>/precap.json` | the pre-capture statistics of every run of the grid |
| `precap.py` | the definitions, plus the alternatives of the robustness table; writes `precap.json` (a copy of `dv_D4_precap/precap.py`) |
| `make_figures.py` | the three figures and `summary.json` (runs `precap.py` where `precap.json` is missing; a copy of `dv_D4_precap/make_figures.py`) |
| `figs/pr_vs_dv.png` | participation ratios before capture vs Δv, median [IQR] over 100 runs, both players, with the same runs over the whole episode drawn faintly. Top row, the strategy ensemble: (a) PR(Z), (b) Gram PR. Bottom row, the latent space: (c) centred PR(H), (d) uncentred PR(H). Δv = 0.2 ringed |
| `figs/pr_vs_dv_before_capture.png` | the before-capture lines alone, in the two-panel layout of [`dv_D1/figs/pr_vs_dv.png`](../dv_D1/figs/pr_vs_dv.png). (a) the draft's pair: strategy PR(Z) (circles, solid) and centred latent PR(H) (squares, dashed). (b) their gauge-invariant counterparts: Gram PR and uncentred PR(H). Δv = 0.2 ringed |
| `figs/capture_vs_dv.png` | capturable share of the training ICs, captured share, and captured / capturable, against f(R) and f(R \| \|Δ0\| ≥ 0.4). The break at ρ = 1 is dashed and saturation dotted, with the region beyond it shaded. Identical to `dv_D1`'s |
| `summary.json` | every number in the figures and tables, per Δv |

## Results (iteration 14,000)

Capture is that of `dv_D1`, since these are the same runs:

| Δv | ρ | f(R) | capturable (ICs) | captured | captured / capturable | runs capturing all | head-time after capture |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.28 | 0.062 | 0.000 | 0.006 | – | 100 | 0% |
| 0.10 | 0.48 | 0.181 | 0.050 | 0.050 | 1.000 | 100 | 1% |
| 0.15 | 0.68 | 0.363 | 0.231 | 0.232 | 1.000 | 100 | 5% |
| **0.20** | **0.88** | **0.608** | **0.512** | **0.468** | **0.913** | **49** | **12%** |
| 0.25 | 1.08 | 0.872 | 0.844 | 0.721 | 0.853 | 12 | 21% |
| 0.30 | 1.28 | 0.981 | 0.975 | 0.887 | 0.910 | 25 | 30% |
| 0.35 | 1.48 | 1.000 | 1.000 | 0.964 | 0.964 | 57 | 38% |
| 0.40 | 1.68 | 1.000 | 1.000 | 0.979 | 0.979 | 73 | 44% |
| 0.45 | 1.88 | 1.000 | 1.000 | 0.996 | 0.996 | 94 | 50% |
| 0.50 | 2.08 | 1.000 | 1.000 | 0.998 | 0.998 | 97 | 54% |

"Head-time after capture" is the share of all (head, time) pairs that come after the head's capture, i.e. what the
pre-capture statistics leave out.

Participation ratios before capture, medians over 100 runs, with the whole episode in brackets:

| Δv | PR(Z) E / P | PR(H) E / P | Gram PR E / P | PR(H) unc E / P | η E / P |
|---|---|---|---|---|---|
| 0.05 | 2.05 / 2.08 (2.05 / 2.08) | 5.41 / 4.76 (5.41 / 4.76) | 1.93 / 1.93 (1.93 / 1.93) | 2.73 / 2.05 (2.73 / 2.05) | 1.000 / 1.000 (1.00 / 1.00) |
| 0.10 | 2.18 / 2.42 (2.18 / 2.43) | 5.33 / 4.99 (5.33 / 5.00) | 1.93 / 1.93 (1.93 / 1.93) | 2.66 / 2.26 (2.65 / 2.27) | 1.000 / 1.000 (1.00 / 1.00) |
| 0.15 | 2.32 / 2.77 (2.34 / 2.53) | 5.48 / 5.00 (5.51 / 5.07) | 1.93 / 1.93 (1.94 / 1.94) | 2.70 / 2.34 (2.73 / 2.38) | 1.000 / 1.000 (1.00 / 1.00) |
| **0.20** | **2.56 / 3.05 (2.68 / 2.63)** | **5.51 / 5.53 (5.56 / 5.79)** | **1.93 / 1.93 (1.94 / 1.97)** | **2.79 / 2.41 (2.84 / 2.53)** | **1.000 / 1.000 (1.00 / 0.99)** |
| 0.25 | 2.93 / 3.46 (2.94 / 2.26) | 5.76 / 5.85 (6.00 / 6.15) | 1.93 / 1.92 (1.98 / 2.10) | 2.94 / 2.69 (3.16 / 3.07) | 1.000 / 1.000 (0.98 / 0.94) |
| 0.30 | 3.29 / 3.67 (2.83 / 2.17) | 5.75 / 5.87 (6.30 / 6.50) | 1.93 / 1.93 (2.10 / 2.45) | 2.94 / 2.76 (3.43 / 3.48) | 1.000 / 1.000 (0.95 / 0.85) |
| 0.35 | 3.60 / 3.87 (2.60 / 2.01) | 5.82 / 5.81 (6.72 / 6.98) | 1.92 / 1.92 (2.32 / 2.87) | 2.94 / 2.74 (3.83 / 4.02) | 1.000 / 1.000 (0.88 / 0.73) |
| 0.40 | 3.89 / 4.17 (2.58 / 2.15) | 5.83 / 5.79 (7.14 / 7.32) | 1.94 / 1.93 (2.65 / 3.35) | 2.95 / 2.80 (4.46 / 4.60) | 0.999 / 1.000 (0.78 / 0.63) |
| 0.45 | 4.03 / 3.96 (2.45 / 2.21) | 5.74 / 5.56 (7.47 / 7.71) | 1.93 / 1.93 (3.06 / 3.77) | 2.98 / 2.80 (5.09 / 5.15) | 0.998 / 1.000 (0.68 / 0.55) |
| 0.50 | 4.06 / 3.95 (2.56 / 2.22) | 5.51 / 5.54 (7.85 / 7.94) | 1.94 / 1.93 (3.60 / 4.30) | 3.06 / 2.76 (5.76 / 5.69) | 0.995 / 1.000 (0.59 / 0.43) |

The latent's motion while heads are in play (medians). The first column is the in-play time variation as a share of
the latent's second moment, tr C_H / (tr C_H + |⟨H⟩|²). The second is the centred PR(H) over the first quarter of the
episode, t ≤ 1:

| Δv | latent time variation E / P | centred PR(H) over t ≤ 1, E / P |
|---|---|---|
| 0.05 | 44.2% / 33.5% | 2.64 / 2.31 |
| 0.10 | 43.0% / 36.4% | 2.69 / 2.30 |
| 0.15 | 43.6% / 38.3% | 2.53 / 2.15 |
| **0.20** | **44.8% / 39.4%** | **2.56 / 2.13** |
| 0.25 | 46.8% / 44.8% | 2.48 / 2.09 |
| 0.30 | 46.7% / 45.7% | 2.49 / 2.12 |
| 0.35 | 47.3% / 45.8% | 2.47 / 2.20 |
| 0.40 | 48.0% / 47.0% | 2.51 / 2.23 |
| 0.45 | 48.8% / 47.4% | 2.52 / 2.26 |
| 0.50 | 50.4% / 47.3% | 2.64 / 2.31 |

Reading:

- **The strategy played before capture is the same at every Δv, as with 4 layers.**
  - Before capture the Gram PR is 1.92–1.94 for both players at every Δv, below the constant-heading bound of 2, and
    η ≥ 0.995.
  - The whole-episode rise of the Gram PR (to 4.30 for the pursuer and 3.60 for the evader at Δv = 0.5) disappears
    completely, and so does the drop of η (to 0.43 / 0.59). Both were heading motion after capture.
  - Each head plays a constant heading until it is caught, across the ρ = 1 break and past saturation.
- **Removing the post-capture time also removes almost all of the Δv dependence of the latent PRs.**
  - Before capture, the centred PR(H) is 4.8–5.9 for the pursuer and 5.3–5.8 for the evader. Over the whole episode
    it rises to 7.9.
  - The uncentred PR(H) before capture is 2.0–2.8 for the pursuer and 2.7–3.1 for the evader. Over the whole episode
    it rises to 5.7–5.8.
  - What remains is a modest rise of the pursuer's latent up to the break (uncentred 2.05 → 2.69 at Δv = 0.25), and
    it is flat after that. This is unlike the 4-layer body, whose centred PR(H) keeps its whole rise before capture.
- **The 1-layer latent moves a lot before capture, at every Δv, and the headings do not read it.**
  - Its time variation is 34–50% of its second moment, against 0.1–9% with 4 layers. It grows only slowly with Δv
    (pursuer 34% → 47%).
  - Over the first quarter of the episode its centred PR(H) is already 2.1–2.7.
  - The headings stay constant through it (η ≥ 0.995): each head's sideways readout is blind to the latent's motion.
    So the latent PRs here measure the body, not the strategy.
- **PR(Z) rises before capture, more than over the whole episode:** 2.05 → 4.06 for the evader and 2.08 → 3.95 for the
  pursuer (4.17 at Δv = 0.40), against 2.56 / 2.22 over the whole episode at Δv = 0.5.
  - As with 4 layers, the headings are constant while in play, so this is variation of the length |z^j|, which the
    normalisation removes. It is counted over heads in play for different lengths of time.
  - Its value depends on how those windows are combined (next table): at Δv = 0.5 it ranges over 2.46–3.95 for the
    pursuer and 2.76–4.52 for the evader. The four definitions agree only where almost nothing is captured
    (Δv ≤ 0.10).

How much the result depends on the way the post-capture time is removed (medians; the first entry of each group is the
plotted one; the alternatives are defined in
[`dv_D4_precap/README.md`](../dv_D4_precap/README.md#results-iteration-14000)):

| Δv | Gram PR P: in play / own clock / frozen / mean headings | Gram PR E: same | PR(Z) P: plotted / pairwise / own clock / latent-projected | PR(Z) E: same |
|---|---|---|---|---|
| 0.05 | 1.93 / 1.93 / 1.93 / 1.93 | 1.93 / 1.93 / 1.93 / 1.93 | 2.08 / 2.09 / 2.08 / 2.08 | 2.05 / 2.05 / 2.06 / 2.05 |
| 0.10 | 1.93 / 1.93 / 1.93 / 1.93 | 1.93 / 1.93 / 1.93 / 1.93 | 2.42 / 2.42 / 2.45 / 2.42 | 2.18 / 2.16 / 2.25 / 2.17 |
| 0.15 | 1.93 / 1.93 / 1.93 / 1.93 | 1.93 / 1.93 / 1.93 / 1.93 | 2.77 / 2.65 / 2.93 / 2.54 | 2.32 / 2.26 / 2.79 / 2.32 |
| **0.20** | **1.93 / 1.93 / 1.93 / 1.93** | **1.93 / 1.93 / 1.93 / 1.93** | **3.05 / 2.78 / 3.25 / 2.64** | **2.56 / 2.36 / 3.71 / 2.60** |
| 0.25 | 1.92 / 1.92 / 1.92 / 1.92 | 1.93 / 1.93 / 1.93 / 1.92 | 3.46 / 2.95 / 3.34 / 2.38 | 2.93 / 2.62 / 5.16 / 2.94 |
| 0.30 | 1.93 / 1.93 / 1.93 / 1.93 | 1.93 / 1.93 / 1.94 / 1.93 | 3.67 / 3.08 / 3.48 / 2.32 | 3.29 / 2.81 / 5.46 / 3.02 |
| 0.35 | 1.92 / 1.93 / 1.92 / 1.92 | 1.92 / 1.93 / 1.93 / 1.92 | 3.87 / 3.27 / 3.29 / 2.19 | 3.60 / 3.02 / 5.55 / 2.80 |
| 0.40 | 1.93 / 1.93 / 1.93 / 1.93 | 1.94 / 1.94 / 1.96 / 1.93 | 4.17 / 3.54 / 3.57 / 2.39 | 3.89 / 3.30 / 5.15 / 2.95 |
| 0.45 | 1.93 / 1.93 / 1.93 / 1.92 | 1.93 / 1.93 / 2.07 / 1.92 | 3.96 / 3.30 / 3.63 / 2.36 | 4.03 / 3.41 / 4.95 / 2.74 |
| 0.50 | 1.93 / 1.93 / 1.93 / 1.93 | 1.94 / 1.95 / 2.16 / 1.93 | 3.95 / 3.37 / 3.79 / 2.46 | 4.06 / 3.38 / 4.52 / 2.76 |

Notes on this table:

- **The Gram PR is robust.** The four constructions agree to ±0.01 for the pursuer at every Δv, and for the evader up to
  Δv = 0.35.
  - The frozen variant reads the evader higher at Δv ≥ 0.40 (up to 2.16 at 0.50): holding a heading at its value at
    capture stretches a last-moment turn of the evader over the rest of the episode.
  - The lowest eigenvalue of the in-play Gram is within 0.06 of zero (out of a trace of 16) in the median run at every
    Δv, and −0.27 in the worst run.
- **PR(Z) is not robust.**

## Against `dv_D4_precap` (4-layer body), before capture

Medians, each entry E / P:

| Δv | PR(Z): 1 layer, 4 layers | PR(H): 1 layer, 4 layers | Gram PR: 1 layer, 4 layers | PR(H) unc: 1 layer, 4 layers | latent time variation: 1 layer, 4 layers |
|---|---|---|---|---|---|
| 0.05 | 2.05 / 2.08, 1.47 / 1.17 | 5.41 / 4.76, 1.58 / 1.19 | 1.93 / 1.93, 1.92 / 1.92 | 2.73 / 2.05, 1.01 / 1.00 | 44.2% / 33.5%, 0.3% / 0.1% |
| 0.10 | 2.18 / 2.42, 1.49 / 1.19 | 5.33 / 4.99, 1.72 / 1.21 | 1.93 / 1.93, 1.92 / 1.92 | 2.66 / 2.26, 1.00 / 1.00 | 43.0% / 36.4%, 0.3% / 0.1% |
| 0.15 | 2.32 / 2.77, 1.58 / 1.18 | 5.48 / 5.00, 1.90 / 1.22 | 1.93 / 1.93, 1.94 / 1.94 | 2.70 / 2.34, 1.01 / 1.00 | 43.6% / 38.3%, 0.7% / 0.2% |
| **0.20** | **2.56 / 3.05, 1.70 / 1.15** | **5.51 / 5.53, 1.75 / 1.22** | **1.93 / 1.93, 1.92 / 1.92** | **2.79 / 2.41, 1.01 / 1.00** | **44.8% / 39.4%, 0.5% / 0.2%** |
| 0.25 | 2.93 / 3.46, 1.76 / 1.26 | 5.76 / 5.85, 1.69 / 1.22 | 1.93 / 1.92, 1.93 / 1.93 | 2.94 / 2.69, 1.01 / 1.01 | 46.8% / 44.8%, 0.7% / 0.5% |
| 0.30 | 3.29 / 3.67, 2.02 / 1.48 | 5.75 / 5.87, 1.94 / 1.32 | 1.93 / 1.93, 1.92 / 1.92 | 2.94 / 2.76, 1.02 / 1.02 | 46.7% / 45.7%, 1.0% / 1.1% |
| 0.35 | 3.60 / 3.87, 2.00 / 1.85 | 5.82 / 5.81, 2.20 / 1.62 | 1.92 / 1.92, 1.93 / 1.93 | 2.94 / 2.74, 1.02 / 1.03 | 47.3% / 45.8%, 1.6% / 2.2% |
| 0.40 | 3.89 / 4.17, 2.20 / 1.97 | 5.83 / 5.79, 2.09 / 1.75 | 1.94 / 1.93, 1.91 / 1.91 | 2.95 / 2.80, 1.04 / 1.08 | 48.0% / 47.0%, 2.6% / 5.2% |
| 0.45 | 4.03 / 3.96, 2.50 / 2.12 | 5.74 / 5.56, 2.37 / 2.00 | 1.93 / 1.93, 1.92 / 1.91 | 2.98 / 2.80, 1.05 / 1.11 | 48.8% / 47.4%, 3.6% / 6.7% |
| 0.50 | 4.06 / 3.95, 2.96 / 2.20 | 5.51 / 5.54, 2.64 / 2.14 | 1.94 / 1.93, 1.92 / 1.92 | 3.06 / 2.76, 1.07 / 1.14 | 50.4% / 47.3%, 4.8% / 8.7% |

- **The strategy before capture does not depend on depth:** the Gram PR is 1.91–1.94 with either body at every Δv.
- **The latent does.** With one layer, before capture:
  - the uncentred PR(H) is 2.0–3.1, against 1.00–1.14 with four layers;
  - the centred PR(H) is 4.8–5.9, against 1.2–2.6;
  - the time variation is 34–50% of the latent's second moment, against 0.1–9%;
  - PR(Z) is larger too: 2.0–4.2, against 1.1–3.0.
- **How the latent changes with Δv differs between the depths.**
  - With 4 layers the latent is nearly still. It moves more before capture at high Δv (time variation 0.1% → 8.7% for
    the pursuer), so its PRs keep a small rise.
  - With 1 layer the motion is large already at Δv = 0.05, and its PRs hardly change once the post-capture time is
    removed.

## Conclusion

With one layer as with four, the strategy played before capture is the same at every Δv: constant headings, with a
Gram PR of 1.93 across the ρ = 1 break and past saturation.

The whole-episode rises of the Gram PR, of η and, with one layer, of both latent PRs, are motion after capture.

Before capture, the 1-layer latent has large PRs at every Δv (centred 5–6, uncentred 2–3). A third to a half of its
second moment is time variation that the headings do not read. These PRs therefore measure the body, which changes with depth, and not
the strategy, which does not. PR(Z) before capture is |z| (gauge) variation over unequal windows, as with four layers.

## Regenerate

```bash
python3 torch_actor/experiments.py dv_D1_precap
```
```bash
python3 torch_actor/dv_D1_precap/make_figures.py
```

`make_figures.py` runs `precap.py` for any grid without a `precap.json`. To recompute the pre-capture statistics,
delete the `precap.json` files first.
