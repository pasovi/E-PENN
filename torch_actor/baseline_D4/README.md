# `baseline_D4` — the benchmark from 4 November 2026 on

**This run is the baseline for all new work.** It replaces the draft's one-layer baseline
(`paper/ckpt/baseline`, `checks/results/r1`) and the one-layer truncated-loss baseline
(`checks/results/r1_trunc`), both of which are kept for comparison and are not deleted.

## Configuration

Everything is the draft's baseline except the five items in bold. The full list is in
[`config.json`](config.json) (the `RunConfig` of run `grid_i0_c0`; `truncate` and `depth` added).

| | |
|---|---|
| game | flat torus [0, 2)², v_E = 1, v_P = 1.2, ε = 0.08, τ = 0.005, t_max = 4, 64 time steps; 16 heads, IC sets with min distance 0.4 |
| actor | Fourier features of time, K = 12 (25 inputs); **4 tanh layers of 16 units** (`depth=4`); 16 linear heads to unit headings |
| loss | Eq. (1) of the draft with λ_nc = 2, λ_d = 5, and the **capture-truncated distance term** `<S(t) d(t)>_t` (`truncate=True`) |
| protocol | **no pursuer warm-up** (`warmup=0`); 14,000 iterations of **3 pursuer + 1 evader** Adam steps; Adam reset every 3,500 iterations |
| learning rates | **lr_P = 1e-3, lr_E = 1e-4** (the draft's 1e-2 / 1e-3 divided by 10: a 4-layer body saturates at the draft's rates) |
| ensemble | 10 init seeds x 10 IC sets = 100 runs, trained as one batch on the Mac GPU (MPS, float32) in 90 s |

## Contents

| | |
|---|---|
| `make_figures.py` | the figures and `summary.json` below (the training stage itself is `experiments.py baseline_D4`) |
| `ckpt/grid_i<init>_c<ic>.npz` | parameters of all 100 runs at iterations 3,500 / 7,000 / 10,500 / 14,000 (`BE_<it>, bE_<it>, BE2_<it>, ..., WE_<it>`, same for P) plus legacy keys; read with `pe_torch.load_ckpt` or `checks/core.load_ckpt` |
| `results/grid_*.json` | one record per run: full rev2 metrics at every checkpoint (`"full"`), the light-metric snapshots every 1,000 iterations, the analytic saddle, the ICs, the config |
| `summary.json` | ensemble medians [IQR] of the headline numbers, and the same for the one-layer truncated baseline |
| `figs/game.png` | capture time vs initial distance (trained pursuer vs best response) and capture fractions |
| `figs/dimensionalities.png` | Gram PR, temporal PR of u and z, PR(H) uncentred and centred, both players; diamonds = one-layer truncated baseline |
| `figs/fig_torus_trajectories.png`, `fig_lifted.png`, `fig_torus3d.png`, `fig_headings.png`, `fig_latent.png`, `head_table.md` | run `grid_i0_c0` in detail |
| `figs/training_evolution_flat_6heads.mp4` | training evolution of run (init 0, IC 0): flat torus, embedded torus, loss history; heads 1, 4, 11 (captured) and 0, 2, 14 (escape) |
| `figs/training_snapshots.npz` | the parameter snapshots and the per-step loss history behind the video |

## Headline numbers (100 runs, median [IQR])

| | baseline_D4 | one-layer truncated baseline (`r1_trunc`) |
|---|---|---|
| capture rate on capturable heads | 91.7 % (49/100 runs capture all) | 90.7 % (48/100) |
| T_c / V* on captured heads | 1.011 [1.005, 1.015] | 1.010 [1.004, 1.015] |
| Gram PR, E / P | 1.92 / 1.92 | 1.99 / 2.00 |
| constancy η, E / P | 1.00 / 1.00 | 0.98 / 0.98 |
| temporal PR of u, E / P | 1.52 / 1.09 | 1.54 / 1.71 |
| PR(H) uncentred, E / P | 1.01 / 1.00 | 2.08 / 4.51 |
| PR(H) centred, E / P | 1.78 / 1.21 | 5.01 / 7.39 |
| mid-band share ratio P / E | 0.69 [0.45, 0.98] | 2.15 [1.66, 3.69] |

Reading: the game is played as well as by the one-layer actor (same capture rate, T_c at V*),
with headings that are exactly constant (η = 1.00) and a strategy ensemble of rank 2 (Gram PR
1.92, below the constant-heading bound of 2). The latent curve is now essentially one direction
(uncentred PR(H) 1.0): the PR(H) of 5–7 of the one-layer baselines was a one-layer artefact, not
a property of the game. The band-ratio asymmetry between pursuer and evader also disappears.

## Regenerate

```bash
python3 torch_actor/experiments.py baseline_D4
```
```bash
python3 torch_actor/baseline_D4/make_figures.py
```
```bash
python3 torch_actor/checks/animate_training.py --truncate --warmup 0 --depth 4 --lr-scale 0.1 --left flat --legend middle --heads 1,4,11,14,0,2 --outdir torch_actor/baseline_D4/figs --title baseline_D4
```

Runs are chaotic, so a re-run reproduces the ensemble statistics, not individual runs.
