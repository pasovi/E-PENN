# `baseline_D4_crosscheck` — an independent re-run of `baseline_D4` (5 Nov 2026)

Same configuration as [`baseline_D4`](../baseline_D4/README.md), field for field ([`config.json`](config.json)):
4 tanh layers of 16 units, K = 12, capture-truncated loss, no warm-up, 3 P : 1 E Adam steps,
lr_P = 1e-3, lr_E = 1e-4, 14,000 iterations, Adam reset every 3,500; 10 init seeds x 10 IC sets = 100 runs
on the Mac GPU (MPS, float32), trained in 75 s.

Environment: a fresh conda env (`/opt/anaconda3/envs/torch`, Python 3.11.18, torch 2.15.0); `baseline_D4`
was run under Python 3.13.9 with the same torch 2.15.0 on the same arm64 Mac.

## Result

All 100 runs are **bit-identical** to `baseline_D4` (max |parameter difference| = 0 at every checkpoint,
same capture pattern on every head). The pipeline is therefore deterministic on a fixed machine, torch
version and device; the chaotic sensitivity noted in the top-level README matters across machines or torch
versions, not between repeated runs here.

| | baseline_D4_crosscheck | baseline_D4 |
|---|---|---|
| capture rate on capturable heads | 91.7 % (49/100 runs capture all) | 91.7 % (49/100) |
| T_c / V* on captured heads | 1.011 | 1.011 |
| Gram PR, E / P | 1.92 / 1.92 | 1.92 / 1.92 |
| constancy η, E / P | 1.00 / 1.00 | 1.00 / 1.00 |
| PR(H) uncentred, E / P | 1.01 / 1.00 | 1.01 / 1.00 |
| PR(H) centred, E / P | 1.78 / 1.21 | 1.78 / 1.21 |
| mid-band share ratio P / E | 0.69 | 0.69 |

`summary.json` holds the medians [IQR] for this run, `baseline_D4` and `r1_trunc`.
`figs/` has the same figures as `baseline_D4/figs` (game, dimensionalities, and the trajectory views of
run `grid_i0_c0`), plus `figs/training_evolution_flat_6heads.gif`: the training evolution of run
(init 0, IC 0), heads 1, 4, 11 (captured) and 0, 2, 14 (escape), 141 frames, as a gif instead of the
mp4 of `baseline_D4`. Its parameter snapshots (`figs/training_snapshots.npz`) are also bit-identical
to those of `baseline_D4`.

## Regenerate

```bash
python3 torch_actor/experiments.py baseline_D4_crosscheck
```
```bash
python3 torch_actor/baseline_D4_crosscheck/make_figures.py
```
```bash
python3 torch_actor/checks/animate_training.py --truncate --warmup 0 --depth 4 --lr-scale 0.1 --left flat --legend middle --heads 1,4,11,14,0,2 --outdir torch_actor/baseline_D4_crosscheck/figs --title baseline_D4_crosscheck --format gif --out training_evolution_flat_6heads
```
