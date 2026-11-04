# `torch_actor/` — the PyTorch pipeline (working directory)

PyTorch rewrite of the open-loop multi-head actor of the draft and of its training protocol,
plus everything that has been run with it. Dependencies: `torch`, `numpy`, `matplotlib`, `scipy`
(`imageio-ffmpeg` for the mp4 videos). Everything runs from the repository root, e.g.
`python3 torch_actor/experiments.py baseline_D4`.

**The benchmark from now on is [`baseline_D4/`](baseline_D4/README.md)** (4-layer body, capture-truncated
loss, no warm-up, 3:1 steps, lr 1e-3 / 1e-4). The older one-layer baselines (`paper/ckpt/baseline*`,
`checks/results/r1*`) are kept for reference and are not deleted.

## The model and the training loop in one paragraph

Each player is an open-loop actor: the time `t` on a 64-point grid enters as Fourier features
(K = 12 harmonics, 25 inputs), goes through a tanh body (16 units per layer; one layer in the draft,
four in `baseline_D4`), and 16 linear heads read the last layer and are normalised to unit headings,
one head per initial condition of the evader. Both players' headings are integrated on the flat
torus [0, 2)², and the loss of Eq. (1) of the draft (soft capture time via per-step capture
probabilities, a one-sided no-capture penalty, and the distance-shaping term) is minimised by the
pursuer and maximised by the evader with Adam. The **capture-truncated loss** (`Config.truncate=True`)
weights the distance term by the survival probability, `<S(t) d(t)>`, so time steps after capture
do not contribute: that is the loss used in every experiment since 29 Oct 2026. Training alternates
`n_P_steps` pursuer and `n_E_steps` evader Adam steps per iteration (3:1 by default), 14,000
iterations, Adam reset every 3,500, optionally preceded by a pursuer-only warm-up. A whole seed grid
(10 init seeds x 10 IC sets = 100 runs) trains as one batched program on the Mac GPU in about a
minute. Individual runs are chaotic (a 1e-15 perturbation diverges within 500 iterations), so only
ensemble statistics over the grid are reproducible across machines.

## Layout

| | |
|---|---|
| `pe_torch.py` | the port: `Config`, parameter init and IC sampling (NumPy streams of the original, verbatim), features, body + heads, rollout, loss (plain and truncated), `train`, checkpoint I/O (rev2 key layout) |
| `experiments.py` | every training stage, by name: the draft's arms (`r1`, `controls`, `c2`, `c7`, `dv`, `protocols`, `honest`, `shapley`), the truncated-loss grids (`r1_trunc*`), and `baseline_D4` |
| `protocols.py`, `analysis.py`, `figures.py` | the draft's Section 8 protocol arms, its statistics and its eight figures (into `paper/`) |
| `protocol_split.py` | warm-up {1500, 0} x step ratio {3:1, 1:1, 1:3} under the truncated loss: capture and PR figures |
| `checks/` | the checking layer: `core.py` + `metrics.py` (rev2 metrics, read the torch checkpoints), `analyze_r1.py` (R1 census), `visualize_actor.py` (torus / lifted-plane / 3D / headings / latent figures of one run), `animate_training.py` (training-evolution videos), `dimensionalities.py` (Gram PR / temporal PR / PR(H) panel of any grid) |
| `checks/{ckpt,results}/<grid>/` | the 100-run grids of every experiment so far (rev2 record format, one JSON per run); `checks/figs/` their figures and videos |
| `baseline_D4/` | **the current benchmark**: driver, 100-run grid (`ckpt/`, `results/`), figures, video, `summary.json`, `config.json` |
| `paper/` | the draft's experiments re-run with the port (29 Oct 2026); `paper/README.md` compares the numbers with the draft |
| `test_parity.py`, `bench.py`, `run_grid.py`, `compare_r1.py`, `numpy_reference.py`, `results/` | the port's gate against the NumPy backward pass, timing, and the first grid comparisons (`PORT_NOTES.md`) |

## Experiments done with the port (all 100-run grids, truncated loss unless stated)

| stage | what | headline |
|---|---|---|
| `r1` | the draft's baseline (draft loss, 1 layer, warm-up 1500, 3:1) | matches the NumPy R1 ensemble |
| `r1_trunc` | same with the truncated loss | Gram PR 2.47 -> 1.99, constancy 0.86 -> 0.98; PR(H) 5 / 7 unchanged |
| `r1_trunc_k0` | body fed only `t` (no Fourier features) | PR(H) -> 1.0, capture 80% |
| `r1_trunc_k0_d4`, `..._lr0.1` | 4-layer body on `t` alone, at lr and lr/10 | saturates at lr; at lr/10 capture 86%, PR(H) 1.1 |
| `r1_trunc_d4`, `r1_trunc_d4_lr0.1` | 4-layer body with Fourier features | PR(H) 1.8 / 1.2 at lr/10, capture 92%: the PR(H) overhead is a one-layer effect |
| `r1_trunc_w{1500,0}_p{3,1}`, `..._p1_e3` | protocol split (warm-up x step ratio) | warm-up irrelevant; 3:1 -> 90% capture, 1:1 -> 55–65%, 1:3 -> 0.5% |
| `baseline_D4` | 4 layers, K = 12, lr 1e-3 / 1e-4, no warm-up, 3:1 | see `baseline_D4/README.md` |

## Typical commands

```bash
python3 torch_actor/experiments.py baseline_D4
```
```bash
python3 torch_actor/baseline_D4/make_figures.py
```
```bash
python3 torch_actor/checks/animate_training.py --truncate --warmup 0 --depth 4 --lr-scale 0.1 --left flat --legend middle --heads 1,4,11,14,0,2 --outdir torch_actor/baseline_D4/figs --title baseline_D4
```
```bash
python3 torch_actor/checks/visualize_actor.py torch_actor/baseline_D4/ckpt/grid_i3_c5.npz torch_actor/checks/figs/D4_i3_c5
```
```bash
python3 torch_actor/checks/dimensionalities.py torch_actor/baseline_D4/results r1_trunc
```

`SMOKE=1 python3 torch_actor/experiments.py <stage>` runs a stage with a tiny budget into
`checks_smoke/` and `paper_smoke/` to test a change in seconds.
