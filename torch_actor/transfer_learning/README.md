# `transfer_learning/` — new heads on a frozen body

Transfer learning (TL) in the multi-head actor: load the trained actor of both players, freeze
their bodies (B, b, and B2, b2, ... for deep bodies: no gradient reaches them), and train one new
linear, bias-free head W^j (2 x 16) per new initial condition with the same capture-truncated loss
and the same self-play protocol as training (pursuer-head warm-up 1,500, 14,000 iterations of
3 P + 1 E Adam steps at 1e-2 / 1e-3, Adam reset every 3,500), gradients with respect to the new
heads only. The heads *are* linear in the latent: z^j(t) = W^j H(t), u^j = z^j / |z^j|, no bias
(`pe_torch.forward_dirs`). All new heads train together as one batched program.

`tl.py` does it; `results/<tag>.json` holds every number (per-head game statistics as in the
rev2 records, ensemble statistics of the new heads, timing), `results/<tag>.npz` the new heads
and the frozen bodies, `figs/<tag>_game` the capture-time figure and `figs/<tag>_torus_map` the
outcome per IC on the flat torus.

```bash
python3 torch_actor/transfer_learning/tl.py --ckpt torch_actor/paper/ckpt/baseline_trunc/grid_i0_c0.npz --grid 10 --seed 0
```

## D1_grid100_seed0 (4 Nov 2026)

Frozen actor: the one-layer truncated baseline, init seed 0 / IC seed 0
(`paper/ckpt/baseline_trunc/grid_i0_c0.npz`, final state). New ICs: the 10 x 10 cell-centred grid
of [0, 2)² (x, y in {0.1, 0.3, ..., 1.9}), pursuer at the origin; 60 of the 100 are capturable
(geodesic distance <= R = 0.88). One initialisation seed of the new heads (seed 0).

| | TL heads (100 new ICs) | training heads of the same run (16 ICs) |
|---|---|---|
| time to TL 100 heads | 93 s CPU float64 (1.6 ms per gradient step); 32 s MPS float32 | (a full training run: 30 s) |
| capture on capturable heads | 91.7 % (55 of 60) | 100 % (4 of 4) |
| capture on uncapturable heads | 0 % (0 of 40) | 0 % |
| T_c / V* on captured heads | 1.017 [1.009, 1.018] | |
| capturable heads that escaped | 5, all at |Δ0| = 0.707, all chasing a non-nearest periodic image | 0 |
| best-response pursuer on the same evader heads | captures 100 % | |
| Gram PR of the new heads, E / P | 2.24 / 2.24 | |
| constancy η of the new heads, E / P | 0.94 / 0.94 | |
| PR(H) of the (frozen) body, E / P | 5.0-class run: centred 6.48 / 7.36, uncentred 2.01 / 3.99 | same body |

The torus map shows the capturable region reproduced as a bicolour map; the five errors are not
at the cut locus but at the eight grid points with displacement (±0.7, ±0.1) or (±0.1, ±0.7) from the
origin's images, where the pursuer head of five of them locked onto the wrong image (it ends at
distance 0.50 from the evader instead of capturing). The other three such points are captured, so
this is a head-initialisation effect, to be checked over seeds.
