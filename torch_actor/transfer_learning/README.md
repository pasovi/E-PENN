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

## D1_grid1024_seed0 (4 Nov 2026): 32 x 32 grid, 1,024 new heads

Same frozen actor, same protocol and seed, on the 32 x 32 cell-centred grid (616 capturable ICs);
MPS float32, 34 s for the 1,024 heads (0.59 ms per gradient step: the batch is free on the GPU).
Figures are PDF only; in the torus map a capturable head that escaped carries a ring coloured by
the periodic image of the pursuer that it chased (blue (0,0), orange (L,0), purple (0,L), brown (L,L);
the corner squares use the same colours).

| | 1,024 new ICs |
|---|---|
| capture on capturable heads | 90.9 % (560 of 616) |
| capture on uncapturable heads | 0 % (0 of 408) |
| T_c / V* on captured heads | 1.015 [1.008, 1.026] |
| capturable heads that escaped | 56, at geodesic distance 0.60-0.87 (median 0.77); 54 of them chased a non-nearest image (closest approach 0.36-0.55) |
| Gram PR / constancy of the new heads, E / P | 2.23 / 2.24, 0.94 / 0.94 |

The errors sit in the outer band of each capturable quarter-disc, never near the pursuer, and
in 54 of 56 cases the head chases an *adjacent* image (the one across the nearer cut-locus line),
not the diagonal one; the two remaining heads chase the right image and still miss.

## D4_grid1024_seed0 (4 Nov 2026): the D4 baseline body, 32 x 32 grid

Frozen actor: `baseline_D4/ckpt/grid_i0_c0.npz` (4-layer body), with the D4 protocol (no warm-up,
lr 1e-3 / 1e-4); 1,024 new heads, seed 0, MPS float32, 45 s. Corner colours of the torus map are now
dark blue (0,0), orange (L,0), yellow (0,L), brown (L,L).

| | D4 body | D1 body (previous section) |
|---|---|---|
| capture on capturable heads | 87.2 % (537 of 616) | 90.9 % |
| capture on uncapturable heads | 16.2 % (66 of 408) | 0 % |
| T_c / V* on captured heads | 1.008 [0.55, 1.02] | 1.015 [1.008, 1.026] |
| capturable heads chasing a non-nearest image | 194 | 55 |
| evader heads whose mean heading points *toward* the pursuer | 314 of 1,024 | — |
| Gram PR / constancy of the new heads, E / P | 2.25 / 2.01, 0.94 / 1.00 | 2.23 / 2.24, 0.94 / 0.94 |

The D4 transfer is worse on both sides: 66 uncapturable ICs get captured (the evader head runs toward
the pursuer or stands still: 314 evader heads have a mean heading with negative projection on Δ0), 167
capturable heads are captured faster than V* for the same reason, and 194 pursuer heads chase a wrong
image. The frozen D4 body is a near-constant latent (PR(H) uncentred 1.0), so a new head can only pick a
constant heading, which is what the saddle needs; the failures are heads that did not move far enough
from their random initialisation. The D4 protocol's learning rates (1e-3 / 1e-4) were chosen to keep the
4-layer body from saturating during training; with the body frozen that constraint is gone, so the
natural check is the same TL with the draft's rates (1e-2 / 1e-3).

### D4_grid1024_seed0_lr1: the same with the draft's learning rates (1e-2 / 1e-3)

| | D4 body, lr 1e-3 / 1e-4 | D4 body, lr 1e-2 / 1e-3 | D1 body, lr 1e-2 / 1e-3 |
|---|---|---|---|
| capture on capturable heads | 87.2 % | **75.2 %** | 90.9 % |
| uncapturable ICs captured | 66 | 14 | 0 |
| T_c / V* on captured heads | 1.008 [0.55, 1.02] | 1.016 [1.01, 1.03] | 1.015 [1.01, 1.03] |
| capturable pursuer heads locked on a non-nearest image | 194 | 168 (97 of them escape) | 55 (54 escape) |
| Gram PR / constancy of the new heads, E / P | 2.25 / 2.01, 0.94 / 1.00 | 2.06 / 2.00, 0.99 / 1.00 | 2.23 / 2.23, 0.94 / 0.94 |

The higher rates fix the evader side (uncapturable captures 66 -> 14, T_c back on V*) but not the
pursuer side: capture on capturable ICs falls to 75 %, and the escapes now fill bands around the
lines x, y = 0.7 and 1.3 inside the quarter-discs, wherever a pursuer head has locked onto the
adjacent image. So the learning rate was only part of the story. The frozen D4 latent is one
direction, so a new head is a constant heading and the pursuer cannot curve; a wrong-image start
is then a basin it cannot leave under the min-image distance term. The D1 body's richer latent
(PR(H) 5-7) lets a new head bend its path and recover, which is why the one-layer body transfers
better on the pursuer side.

## A D4 body trained on 100 heads (`train_body.py`, 4 Nov 2026)

`train_body.py` trains one multi-head actor with n_h random ICs under the baseline_D4 pipeline and
saves it under `bodies/` (checkpoint, record, and `figs/<tag>_game.pdf` of its training heads).
`D4_h100_i0_c0`: 100 heads, init seed 0, IC seed 0, MPS float32, 67 s. Training heads: capture
90.9 % of the 55 capturable, T_c / V* 1.008, Gram PR 2.06 / 2.00, η 0.99 / 1.00, PR(H) uncentred
1.11 / 1.01: the same rank-2 game and near-constant latent as with 16 heads.

TL of 1,024 heads on it (`D4h100_grid1024_seed0`, D4 protocol, 42 s):

| frozen body | D4, 100 training heads | D4, 16 training heads | D1, 16 training heads |
|---|---|---|---|
| capture on capturable heads | 86.9 % | 87.2 % | 90.9 % |
| uncapturable ICs captured | 87 | 66 | 0 |
| T_c / V* on captured heads | 1.006 [0.61, 1.02] | 1.008 [0.55, 1.02] | 1.015 [1.01, 1.03] |
| capturable pursuer heads on a non-nearest image | 205 | 194 | 55 |
| PR(H) of the frozen body, uncentred E / P | 1.11 / 1.01 | 1.01 / 1.00 | 2.01 / 3.99 |

Six times more training heads do not change the transfer: the body still collapses to one latent
direction because the 100 training heads need nothing else, and the new heads fail in the same way.
What the transfer needs is time variation in the latent, which more constant-heading training ICs
cannot provide.
