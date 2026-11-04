# The draft's experiments, re-run with the torch port

Everything here was produced by `experiments.py`, `analysis.py`, `figures.py` and the
`checks/` tools on 29 Oct 2026 (Apple M4 Max, torch 2.15, 22.5 min wall time for all
training, see `run.log`). Nothing was tuned; the arms are the draft's arms under one
deterministic protocol (`rev2/core.py`'s: warm-up 1,500, 14,000 iterations of 3 P + 1 E Adam
steps at 1e-2 / 1e-3, Adam reset every 3,500), with 10 init seeds per arm instead of the
draft's single seed wherever that was free.

| | |
|---|---|
| `ckpt/<arm>/*.npz` | parameters (rev2 key layout at the checkpoints, plus legacy keys for the final state) |
| `results/*.json`, `*.jsonl` | per-run light metrics, the draft's JSON-line logs for Figs. 4, 6, 7, 8, `table1.json` |
| `figs/fig1..8_*.png/.pdf` | the draft's eight figures; `figA_c7_collapse` is Table 1's C7 as a figure |
| `../checks/results/r1*`, `../checks/figs/r1_*` | the R1 ensemble (100 + 10 + 10 runs) analysed by the ported `rev2/analyze_r1.py` |
| `../checks/figs/baseline/`, `grid_i3_c5/`, `e1b_s0/` | trajectory, lifted-plane, torus, heading and latent figures from the ported `viz/visualize_actor.py` |

Regenerate: `python3 torch_actor/experiments.py` (all stages, or a subset by name), then
`python3 torch_actor/analysis.py`, `python3 torch_actor/figures.py`,
`python3 torch_actor/checks/analyze_r1.py`, `python3 torch_actor/checks/visualize_actor.py [ckpt] [out]`.

## What was run

| arm | draft | here | wall |
|---|---|---|---|
| baseline + C1 | 1 run; 8 init seeds at 10k, 5 IC resamples | R1 grid: 10 init x 10 IC seeds at 14k (GPU, f32), + chaos arm (10 x 1e-10, CPU f64) + no-reset arm (10) | 43 s + 147 s + 145 s |
| C3 bandwidth | K = 6, 24, seed 0 | K = 6, 24, 10 seeds each | 2 x 146 s |
| C4, C5, E1, E1b, E1b-reflected | seed 0 each | 10 seeds each, one 60-run batch (GPU) | 72 s |
| E7 shaping anneal | seed 0, lam_d -> 0 at 4k | 10 seeds, checkpoint at 4k and 14k | 145 s |
| C2 symmetrised | 4 (8) seeds, 28k, snapshots 20k-28k | 7 seeds, 28k, snapshots 20k-28k, exact zero-sum, 1:1, lr 3e-3, no warm-up | 109 s |
| C7 warm start | 1 run | 10 seeds: 20k supervised fit to the curved family (MSE 6e-6), then 14k game, PR every 250 | 47 + 142 s |
| Δv sweep | 8 v_P x 3 seeds | same, one 24-run batch (GPU) | 78 s |
| Fig. 6 protocols | oracle / selfplay / simul / ABR, 14k steps, 3 seeds | same | 62 s |
| Fig. 7 honest | gda / eg / ogda / avg / ogda+EMA / oracle, 6k, 3 seeds | same | 58 s |
| Fig. 8 Shapley | 3 seeds, 5k | same | 11 s |

Not re-run: the critics of Section 9 (not part of the actor; the companion paper's code).

## Reproduced numbers against the draft

Single-run numbers cannot match the draft (training is chaotic; see the top-level README),
so the comparison is claim by claim. "seed 0" is the same initial state and IC set as the
draft's baseline; ranges are over the seeds run here.

| claim | draft | here |
|---|---|---|
| strategy PR(Z_E) / PR(Z_P), baseline | 2.99 / 1.91 | seed 0: 1.78 / 2.23; 100 runs: 1.5-3.8 / 1.3-3.6 |
| latent PR(H_E), baseline; over init seeds | 6.75; 5.4 ± 0.9 | seed 0: 6.90; 6.1 ± 0.8 (8 seeds), 100 runs 4.1-7.1 |
| PR(H)/PR(Z) ≥ 1.69 in every run, both players | yes (43 ckpts) | min 1.61 (E), 2.02 (P) over 100 runs |
| 90% of ensemble variance in 4 / 3 of 32 components | 4 / 3 | 3 / 2 |
| gauge-fixed effective rank E / P | 2.53 / 1.91 | 2.15 / 1.66 (seed 0); 2.70 / 2.16 median of 100 |
| raw head σ_max E / P | 5.6 / 18.1 | 6.0 / 17.2 |
| init PR(H) | ≈ 10.4 | 10.38 |
| latent power at k ≤ K | 98% | 98% (E), 90% (P), every arm |
| spectral break follows K (1% rule) | 6 → 6, 12 → 10, 24 → 22 | 6 → 11, 12 → 19 ± 2, 24 → 37 ± 5 (seed 0: 11, 17, 46) |
| drop k = 12 → 16, E / P | ×27 / ×9 | ×31 / ×6 |
| C1 variance ratio init : IC | 4.7 : 1 | 0.8 : 1 (as rev2's R1 already found: the IC set dominates) |
| C4 classes 12 → 4, PR(H_E) at the seed mean | 5.22 vs 5.4 ± 0.9 | 4 classes; 6.32 (seed 0), 6.0 ± 0.7 vs 6.1 ± 0.8 |
| C5 PR(Z) in [1.6, 3.0] in all arms | yes | lam_d = 1, 5: 2.6, 2.2 (means); lam_d = 0: 4.5 ± 1.2 and only 1 of 16 captured: the unshaped game barely trains here |
| C7: planted rank, collapse within 8k | 9.3 (unit headings); 6.5 → 2.5, 8.0 → 1.6 | PR(U) 9.3, PR(Z) 6.7 / 7.7 → 2.2 ± 0.4 / 2.0 ± 0.3; median PR(Z_E) < 3 from iteration 3,250 |
| C2 run-mean band gaps; snapshot range | ×1.25-1.99; [0.66, 3.29] | ×1.32-2.14 (7 seeds); [0.64, 3.08], 14% of snapshots < 1 |
| Sec. 7 mid-band share P vs E, baseline | 30.3% vs 15.8% | 23.9% vs 15.9% (seed 0); ratio 2.0 ± 0.5 over 100 runs, 1.5-2.5 over the 8 init seeds |
| E7 annealed mid-band share | 22.7% vs 11.5% | 15.7% vs 7.2% at 14k (seed 0) |
| E1b reflection: mean w_x | +1.05 → -0.86 | +0.78 → -0.42 |
| Δv sweep | PR(Z_E) 1.4-3.1, PR(H_E) 5-7, no feature at ρ = 1 | PR(Z_E) 1.5-3.3, PR(H_E) 4.5-7.2, rising gently with Δv, no feature at ρ = 1 |
| Fig. 6 oscillation: oracle ≪ 3:1 < 1:1 ≪ ABR | 5e-4, 0.015, 0.019, 0.17 | 5e-4, 0.008, 0.034, 0.16 |
| Fig. 7: no honest correction enters the oracle basin | none | none (GDA 0.92 capture / 0.034 osc., OGDA 1.0 / 0.039, EG 0.83 / 0.072, averaging 0.67 / 0.024, OGDA+EMA 0.50 / 0.044) |
| Fig. 8: oracle capture in the k = 1-4 band; self-play ≈ 0 | 68%; 0.17 capture | 65% ± 7%; self-play 0.25 capture, all bands ≈ 0 |

Everything the draft attributes to the game (constant headings at rank ≈ 2, the collapse of a
planted rank-9 family, the chart choice), to the featurisation (latent PR 5-7 with 98% of
power inside the input band, a break that moves with K, PR 10.4 at initialisation) and to
the protocol (oscillation ordering, honest corrections, Shapley) reproduces. Two numbers do
not: the C1 variance split (reversed, as rev2's R1 found with the NumPy code) and the absolute
position of the 1% break index, which lands above K rather than at it in every arm (the
*ratio* across K is as in the draft). The unshaped C5 arm (lam_d = 0) does not train to a
constant-heading ensemble here (one capture, PR(Z_E) 4.5): the draft's "trains despite having
no value to converge to" rests on one run and should be re-checked with the NumPy code.

The R1 census of the torch ensemble (`../checks/results/r1_census.md`) matches rev2's NumPy
census line by line: Gram PR 2.47 [2.32, 2.67] vs 2.47 [2.32, 2.70], constancy 0.86 vs 0.85,
PR(H_E) 5.61 vs 5.73, chart errors falling from 46% at the cut locus to 0% far from it,
99.9% capture in the nearest chart vs 9% in any other.
