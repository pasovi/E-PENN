# Summary of the `torch_actor/` work: tests done and conclusions (as of 10 Nov 2026)

For collaborators and their assistants. This file is the entry point; every number below is backed by a
README, a results JSON and a script named in the last column of each table. Paths are relative to
`torch_actor/`. All work is on branch `pablo`. The pre-existing material from `main` lives unchanged in
`original_paper_reproduction/`.

## 1. Setting (what is being trained)

Pursuit–evasion on the flat torus [0, 2)^d, v_E = 1, v_P = 1.2, capture radius ε = 0.08, t_max = 4, so the
pursuer's reach is R = ε + (v_P − v_E) t_max = 0.88 and an initial condition (IC) is *capturable* iff its
geodesic distance to the pursuer is ≤ R. The analytic value is V* = (|Δ0| − ε)/0.2 with constant headings.

Each player is an open-loop **multi-head actor**: Fourier features of time (K = 12) → tanh body (16 units per
layer) → one linear, bias-free head per IC, normalised to a unit heading. The latent is H(t) (the body's last
layer); head j outputs z_j = W_j H(t). Training is alternating self-play with Adam: 14,000 iterations of
3 pursuer + 1 evader steps, Adam reset every 3,500, optional pursuer-only warm-up of 1,500 steps.

Metrics used everywhere: capture rate on capturable heads, T_c/V* on captured heads, **Gram PR** (participation
ratio of the ensemble of mean headings; bounded by d for constant headings), **constancy η** (1 = headings
constant in time), **PR(H)** of the latent curve (centred and uncentred).

Individual runs are chaotic (a 1e-15 perturbation diverges in 500 iterations): only ensemble statistics over
the 100-run seed grid (10 init seeds × 10 IC sets) are reproducible. One grid trains in about a minute on an
Apple-silicon GPU (MPS, float32).

## 2. Tests done and what each one showed

### 2.1 Loss and architecture on T² (100-run grids)

| test | result | where |
|---|---|---|
| Port of the draft's NumPy actor to PyTorch (`r1`) | reproduces the draft's ensemble | `PORT_NOTES.md`, `paper/README.md` |
| **Capture-truncated loss** `<S(t) d(t)>`: the distance term stops at capture (`r1_trunc`) | Gram PR 2.47 → 1.99, η 0.86 → 0.98, same capture (90.7 %), T_c/V* 1.01. PR(H) 5 / 7 (E / P) unchanged | `README.md`, `checks/` |
| Body fed `t` only, no Fourier features (`r1_trunc_k0`) | PR(H) → 1.0, capture 80 % | `README.md` |
| 4-layer body, at the draft's lr and at lr/10 (`r1_trunc_d4*`) | saturates at the draft's lr; at lr/10 capture 92 %, PR(H) 1.8 / 1.2 | `README.md` |
| **Protocol split**: warm-up {1500, 0} × step ratio {3:1, 1:1, 1:3} | warm-up irrelevant in every arm; capture 90 % at 3:1, 55–65 % at 1:1, 0.5 % at 1:3 | `protocol_split.py`, `paper/results/protocol_split.json` |
| **`baseline_D4`** (4 layers, lr 1e-3 / 1e-4, no warm-up, 3:1, truncated loss) | capture 91.7 %, T_c/V* 1.011, Gram PR 1.92, η 1.00, PR(H) uncentred 1.0 | `baseline_D4/README.md` |
| `baseline_D4_w1500` (same with warm-up) | identical ensemble (capture 92.2 %) | `baseline_D4_w1500/README.md` |

### 2.2 Transfer learning (TL) on T²

TL = freeze both bodies, train one new linear head per new IC with the same loss and protocol. All new heads
train as one batch; 1,024 heads cost 35–45 s. One head-initialisation seed unless stated.

| frozen body | TL result on 1,024 new ICs (32 × 32 grid, 616 capturable) | where |
|---|---|---|
| D1 (one-layer truncated baseline, 16 training heads) | 90.9 % of capturable captured, 0 % of uncapturable; 54 of the 56 escapes chase an adjacent periodic image | `transfer_learning/README.md` |
| D4 baseline body, D4 learning rates | 87.2 %; 66 uncapturable ICs captured (evader heads that barely moved from initialisation) | same |
| D4 body, draft learning rates | evader side fixed (14 uncapturable captured), pursuer side worse: 75.2 % | same |
| D4 body trained on 100 heads instead of 16 (`train_body.py`) | 86.9 %: no change | same |

### 2.3 Other topologies with the D1 pipeline (`topologies/`)

| | S¹ | T² | T³ | T⁴ |
|---|---|---|---|---|
| MH training heads | 16 | 16 | 64 | 96 |
| MH capture on capturable / uncapturable | 100 % / 0 % | 100 % / 0 % | 100 % / 0 % | 100 % / 0 % |
| MH T_c / V* | 1.009 | 1.01 | 1.008 | 1.007 |
| MH Gram PR, E / P (bound d) | 1.00 / 1.00 | 1.99 / 2.00 | 2.98 / 2.94 | 3.94 / 3.93 |
| TL heads (sampler) | 100 (grid) | 1,024 (grid) | 1,000 (grid 10³) | 1,000 (Sobol) |
| TL capture on capturable | 76.1 % | 90.9 % | 97.2 % | 98.3 % |

S¹ uses u = tanh(z) because normalisation has no gradient in one dimension. S¹ was also run without warm-up:
same game (100 % capture, T_c/V* 1.009), TL 79.5 % against 76.1 %, inside single-seed noise.
Figures per topology: game (MH and TL), PR bars, PR evolution against iteration, latent components and PCA,
TL outcome map; videos for S¹ and T³; a rotatable HTML map for T³. See `topologies/README.md`.

### 2.4 Is the TL improvement with d real? (`topologies/conditional_analysis/`)

Two coordinates per capturable IC: reach margin b = R − |Δ0| and **image margin m = d₁ − d₀** (distance to the
second-nearest periodic image of the pursuer minus the distance to the nearest).

1. *Re-analysis of the existing TL runs* (`conditional.py`): pooled logistic regression of escape on (b, m)
   over T², T³, T⁴ (1,155 ICs, 69 escapes). A per-dimension term on top of the geometry is not significant
   (likelihood-ratio p = 0.59). The 10³ grid on T³ contains no IC with b < 0.1 and m < 0.5.
2. *Control run* (`control_ball.py`): 500 ICs per dimension drawn uniformly inside the capturable shell
   0.4 ≤ |Δ0| ≤ R, three head seeds, 12 TL runs.

| control | S¹ | T² | T³ | T⁴ |
|---|---|---|---|---|
| escapes per seed (of 500) | 204 / 221 / 235 | 64 / 60 / 58 | 35 / 31 / 30 | 20 / 21 / 10 |
| escape rate, pooled (95 % CI) | 44.0 % [41.5, 46.5] | 12.1 % [10.6, 13.9] | 6.4 % [5.3, 7.8] | 3.4 % [2.6, 4.4] |
| escapes expected from geometry alone | not pooled | 183 | 90 | 57 |
| escapes observed | 660 | 182 | 96 | 51 |

   Dimension term: p = 0.51 (4,500 ICs, 329 escapes). The escape rate against m is one curve for the three
   tori: about 40–60 % for m < 0.4, 10–30 % for m in [0.4, 0.6), essentially zero for m > 0.8.

## 3. Conclusions

1. **The truncated loss is the right loss.** Letting the game end at capture removes the post-capture
   wandering that inflated the ensemble rank: the strategy ensemble becomes rank d with constant headings,
   without changing how well the game is played. It is the default in everything after 29 Oct 2026.
2. **The step ratio is load-bearing, the warm-up is not.** 3 pursuer steps per evader step are needed for the
   pursuer to win the capturable heads; the 1,500-step warm-up changes nothing on T² (D1 and D4) or on S¹.
3. **The "latent overhead" PR(H) ≈ 5–7 of the draft is a one-layer featurisation residue, not a property of
   the game.** A one-layer body is a fixed random mix of Fourier features that the loss never asks to remove;
   a 4-layer body collapses to essentially one latent direction (PR(H) uncentred 1.0) with the same capture
   rate and exactly constant headings. The game needs a constant latent and nothing else.
4. **`baseline_D4` is the benchmark for new work** (config in `baseline_D4/config.json`). The older baselines
   are kept for comparison.
5. **The ensemble rank equals the dimension of the space.** Gram PR 1.00 / 1.99 / 2.98 / 3.94 on S¹ / T² /
   T³ / T⁴, η ≈ 1, T_c = V*, reached within the first few thousand iterations.
6. **Transfer works, and its failures are one mechanism.** New heads on a frozen body capture about 90 % of
   the capturable new ICs on T² and none of the uncapturable ones. The escapes are pursuer heads that lock
   onto a non-nearest periodic image, a basin of the head's random initialisation.
7. **A collapsed latent transfers worse, and more training heads do not help.** The D4 body's constant latent
   lets a new head choose only a constant heading, so a head that starts toward the wrong image cannot curve
   back; the one-layer body's unused time variation is what lets heads recover. Training the body on 100
   heads instead of 16 changes nothing, because constant-heading ICs never ask the body for more.
8. **The apparent improvement of TL with dimension is geometry of the sampled ICs, not a better mechanism.**
   The escape probability depends on the image margin m and is the same function of m on T², T³ and T⁴. In
   higher d a smaller share of the capturable ball lies near a cut-locus wall, so fewer new heads start with
   a competing image. The first T³ and T⁴ numbers were also flattered by their samplers.
9. **S¹ is a separate case.** Its failures are heads running the wrong way round the circle, at about 44 % of
   the shell ICs regardless of position: with a one-dimensional heading there is no rotation to recover with.

## 4. Caveats a reader should keep in mind

- TL results other than the control in 2.4 use one head-initialisation seed and one frozen body (init 0, IC 0).
- The S¹, T³ and T⁴ multi-head trainings are single runs, not 100-run grids.
- "Danger zone" (b < 0.1, m < 0.5) in the conditional analysis is a descriptive cut read off the data; the
  logistic fit and its test use no threshold. The fitted coefficient on b is positive only because b and m
  are correlated inside the ball; m carries the signal.
- The logistic model is statistical, not a model of the game. The proposed mechanism (random initial heading
  falling in the basin of a competing image) is consistent with the data but has not been tested directly.
- MPS is float32; CPU runs are float64. Ensemble statistics agree; individual runs do not.

## 5. Not done / open

- Head-initialisation seed sweep for the T² TL grids, and a seed grid of 100-head bodies (both deferred).
- Matching the capturable volume across dimensions by changing Δv: judged unnecessary after 2.4, since it
  would not equalise the distribution of m.
- A direct test of the basin mechanism: e.g. initialising new pursuer heads toward the nearest image, or
  restarting escaped heads, which should remove most TL escapes if conclusion 6 is right.
- Games whose saddle is not a constant heading (obstacles, curvature). Conclusions 3 and 7 pull in opposite
  directions there: a deep body is wanted for expressivity, but on this game it collapses and transfers
  worse. Whether a body trained on a game that *requires* time variation keeps a transferable latent is the
  open question that motivates the next step.

## 6. Where to start reading

`README.md` (pipeline and layout) → `baseline_D4/README.md` (benchmark) → `transfer_learning/README.md` →
`topologies/README.md` → `topologies/conditional_analysis/README.md`.

```bash
python3 torch_actor/experiments.py baseline_D4
```
```bash
python3 torch_actor/transfer_learning/tl.py --ckpt torch_actor/paper/ckpt/baseline_trunc/grid_i0_c0.npz --grid 32 --seed 0 --device mps --dtype float32
```
```bash
python3 torch_actor/topologies/run_topology.py --dim 4 --n-heads 96 --tl-n 1000 --device mps --dtype float32
```
```bash
python3 torch_actor/topologies/conditional_analysis/control_ball.py --n 500 --seeds 0,1,2 --device mps --dtype float32
```
