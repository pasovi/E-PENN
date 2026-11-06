# `topologies/` — the baseline D1 pipeline on S¹ and T³

The T² baseline (Fourier features of time, one tanh layer of 16 units, one linear bias-free head per
IC, capture-truncated loss, pursuer warm-up 1,500, 14,000 iterations of 3 P + 1 E Adam steps at
1e-2 / 1e-3, Adam reset every 3,500) on the flat torus T^d = [0, 2)^d for d = 1 (the circle S¹) and
d = 3. `pe_nd.py` is the dimension-generic model and metrics (it reuses `pe_torch.rollout` and
`pe_torch.game_loss`, which never depended on the dimension); `run_topology.py` trains the multi-head
actor (MH), then transfers new heads onto the frozen bodies (TL) and draws the figures;
`animate_s1.py` renders the S¹ training video from the saved snapshots.

One difference by necessity: for d = 1 the unit sphere is {−1, +1} and the normalisation u = z/|z|
has no gradient, so the S¹ heading is u = tanh(z) in [−1, 1] (full speed in either direction is
reachable, and is what both players use: mean |u| = 0.99). The game constants are the same
(v_E = 1, v_P = 1.2, ε = 0.08, t_max = 4), so the reach is R = 0.88 in every dimension; on S¹ the
farthest point is at distance 1 (the antipode is the cut locus), on T³ at √3 = 1.73. ICs are drawn
uniformly on T^d at geodesic distance ≥ 0.4 from the pursuer at the origin. TL uses cell-centred
uniform grids: 100 points on S¹, 10 × 10 × 10 on T³, one head-initialisation seed.

```bash
python3 torch_actor/topologies/run_topology.py --dim 1 --n-heads 16 --tl-grid 100
```
```bash
python3 torch_actor/topologies/run_topology.py --dim 3 --n-heads 64 --tl-grid 10 --device mps --dtype float32
```
```bash
python3 torch_actor/topologies/animate_s1.py
```
```bash
python3 torch_actor/topologies/animate_t3.py
```
```bash
python3 torch_actor/topologies/tl_map_t3_html.py
```

Outputs (6 Nov 2026): `results/<S1|T3>_{mh,tl}.json/.npz` (every number; per-head game statistics;
PR evolution every 500 iterations), `results/S1_mh_snapshots.npz`, and in `figs/`: `_game_mh`,
`_game_tl` (capture time vs |Δ0| against V*, capture fractions), `_pr` (MH and TL statistics),
`_pr_evolution`, `_latent` (components, PCA profiles, spectrum), `_tl_map` (outcome per new IC:
the circle for S¹, a 3D scatter plus a histogram by geodesic distance for T³), and
`S1_training_evolution.mp4` (two captured and two escaped heads as space–time spirals on the circle,
with the loss history), `T3_training_evolution.mp4` (three captured and three escaped heads in the
periodic cube, with the loss history) and `T3_tl_map.html` (the T³ TL outcome map as a rotatable,
self-contained plotly page: hover gives each IC's distance, V*, T_c and the chased image).

## Results

| | S¹, MH (16 heads) | S¹, TL (100 heads) | T³, MH (64 heads) | T³, TL (1,000 heads) | T² reference (MH 16 / TL 1,024) |
|---|---|---|---|---|---|
| capturable heads | 14 | 88 | 19 | 360 | 4 / 616 |
| capture on capturable | 100 % | 76.1 % | 100 % | 97.2 % | 100 % / 90.9 % |
| capture on uncapturable | 0 % | 0 % | 0 % | 0 % | 0 % / 0 % |
| T_c / V* on captured | 1.009 [1.004, 1.013] | 1.013 [1.000, 1.024] | 1.008 [1.005, 1.012] | 1.016 [1.002, 1.018] | 1.01 / 1.015 |
| capturable heads escaped | 0 | 21 (13 ran the wrong way) | 0 | 10 (8 chased a non-nearest image) | 0 / 56 |
| Gram PR, E / P (bound d) | 1.00 / 1.00 (1) | 1.01 / 1.11 | 2.98 / 2.94 (3) | 3.17 / 3.16 | 1.99 / 2.00 (2) |
| constancy η, E / P | 1.00 / 1.00 | 0.99 / 0.94 | 0.98 / 0.99 | 0.97 / 0.97 | 0.98 / 0.98 |
| temporal PR of u, E / P | 1.01 / 1.00 | 1.61 / 3.76 | 1.65 / 1.56 | 2.34 / 3.95 | 1.5 / 1.7 |
| PR(H) centred, E / P | 2.67 / 4.70 | same body | 3.44 / 6.91 | same body | 5.0 / 7.4 |
| PR(H) uncentred, E / P | 1.00 / 1.33 | same body | 1.95 / 3.41 | same body | 2.1 / 4.5 |
| wall time | 24 s (CPU f64) | 41 s | 116 s (MPS f32) | 86 s | |

**The rank of the strategy ensemble follows the dimension.** Constant headings on T^d span at most
d directions, and the trained ensembles sit at the bound in every case: Gram PR 1.00 on S¹, 1.99 on
T², 2.98 on T³, with η ≈ 1. The PR-evolution figures show the ensembles collapsing onto the bound
within the first 1,000 to 4,000 iterations and staying there. The game is solved exactly (T_c = V*,
all capturable training heads captured, none of the uncapturable) in every dimension.

**The latent overhead is a featurisation residue, as on T².** PR(H) centred is 2.7 / 4.7 on S¹ and
3.4 / 6.9 on T³ against 5.0 / 7.4 on T²: the one-layer body keeps time variation the heads do not
read, more for the pursuer (trained at the higher rate) than for the evader. On T³ the pursuer's
latent components are square waves saturated at ±1, the lr 1e-2 saturation seen in the deep T² bodies.

**Transfer.** On T³ the transfer is better than on T² (97.2 % against 90.9 % of capturable new ICs,
10 escapes of 360, 8 of them chasing a non-nearest image, scattered over the outer shell of the
capturable balls). On S¹ it is worse (76.1 %): 21 of 88 capturable new ICs escape, in two arcs at
geodesic distance 0.45 to 0.85 on either side of the pursuer, where the new pursuer head runs the
wrong way round the circle at full speed. With one dimension the only wrong choice is the opposite
direction, and a head that starts that way has nothing to steer it back: the smooth normalisation
that lets a T² or T³ head rotate is replaced by a sign.

### S¹ without the pursuer warm-up (`--warmup 0 --name S1_w0`)

| | S¹ MH, warm-up 1500 | S¹ MH, no warm-up | S¹ TL, warm-up 1500 | S¹ TL, no warm-up |
|---|---|---|---|---|
| capture on capturable | 100 % (14) | 100 % (14) | 76.1 % (88) | 79.5 % (88) |
| T_c / V* on captured | 1.009 | 1.009 | 1.013 | 1.012 |
| Gram PR, E / P | 1.00 / 1.00 | 1.00 / 1.00 | 1.01 / 1.11 | 1.02 / 1.07 |
| constancy η, E / P | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.94 | 0.98 / 0.95 |
| PR(H) centred, E / P | 2.67 / 4.70 | 3.71 / 4.03 | same body | same body |

The warm-up has no effect on the game (identical capture and T_c in the MH run, 3 heads' difference
in the TL run, inside the single-seed noise); the latent PR of the unread directions differs because
it is never trained toward anything, as on T². `S1_w0_training_evolution.mp4` is the corresponding video.
