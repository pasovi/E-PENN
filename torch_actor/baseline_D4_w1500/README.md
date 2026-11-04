# `baseline_D4_w1500` — the benchmark with the pursuer warm-up

The same run as [`baseline_D4`](../baseline_D4/README.md) (4-layer body of 16 tanh units, K = 12,
capture-truncated loss, 3 pursuer : 1 evader steps, lr_P = 1e-3 / lr_E = 1e-4, Adam reset every
3,500 iterations, 100 runs) with the draft's **1,500-step pursuer-only warm-up** before self-play
(`warmup=1500`). Everything else is identical; `config.json` lists it all. `baseline_D4` (no warm-up)
is the benchmark; this folder is its warm-up control.

Same layout as `baseline_D4/`: `make_figures.py`, `ckpt/`, `results/`, `summary.json`, `config.json`,
`figs/` (`game`, `dimensionalities`, the static views of run `grid_i0_c0`, and
`training_evolution_flat_6heads.mp4` with heads 1, 4, 11, 14, 0, 2).

## Headline numbers (100 runs, median)

| | baseline_D4_w1500 | baseline_D4 (no warm-up) | one-layer truncated (`r1_trunc`) |
|---|---|---|---|
| capture rate on capturable heads | 92.2 % (48/100 runs capture all) | 91.7 % (49/100) | 90.7 % (48/100) |
| T_c / V* on captured heads | 1.011 | 1.011 | 1.010 |
| Gram PR, E / P | 1.93 / 1.93 | 1.92 / 1.92 | 1.99 / 2.00 |
| constancy η, E / P | 1.00 / 1.00 | 1.00 / 1.00 | 0.98 / 0.98 |
| PR(H) uncentred, E / P | 1.01 / 1.00 | 1.01 / 1.00 | 2.08 / 4.51 |
| PR(H) centred, E / P | 1.78 / 1.22 | 1.78 / 1.21 | 5.01 / 7.39 |
| mid-band share ratio P / E | 0.62 | 0.69 | 2.15 |

The warm-up changes nothing at the ensemble level, as in the one-layer protocol split.

## Regenerate

```bash
python3 torch_actor/experiments.py baseline_D4_w1500
```
```bash
python3 torch_actor/baseline_D4_w1500/make_figures.py
```
```bash
python3 torch_actor/checks/animate_training.py --truncate --warmup 1500 --depth 4 --lr-scale 0.1 --left flat --legend middle --heads 1,4,11,14,0,2 --outdir torch_actor/baseline_D4_w1500/figs --title baseline_D4_w1500
```
