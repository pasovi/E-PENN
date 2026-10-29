# `torch_actor/` — PyTorch port of the multi-head actor

Self-contained PyTorch rewrite of the open-loop multi-head actor and its published training
protocol (`pursuit_evasion_numpy.py` + `rev2/core.py`). Everything needed to train runs from
this folder; the only link back to the rest of the repository is optional (checkpoints are
written in `rev2`'s key layout so `rev2/metrics.py` reads them, and `compare_r1.py` reads the
R1 results if they exist).

Dependencies: `torch`, `numpy` (and `matplotlib`, imported by the copied NumPy reference).
Written and tested with torch 2.15, Python 3.13, on an Apple M4 Max (29 Oct 2026).

| file | what |
|---|---|
| `pe_torch.py` | the port: `Config` (field for field `rev2/core.RunConfig`), NumPy-stream init and IC sampler (verbatim), Fourier features, body + heads, cumsum rollout, loss, one `step`, `train` with warm-up / 3:1 steps / Adam resets, `light_metrics`, checkpoint I/O |
| `test_parity.py` | gate: losses and autograd gradients vs the hand-written NumPy backward pass, then how fast two bit-different runs separate |
| `bench.py` | ms per step for NumPy, torch CPU, torch MPS/CUDA, batched runs; ATen op count per step |
| `run_grid.py` | train an init-seed x IC-seed grid in one batched program; writes `.npz` per run + `summary.json` |
| `compare_r1.py` | ensemble statistics of a grid vs the NumPy R1 grid (`rev2/results/r1/`) |
| `numpy_reference.py` | verbatim copy of `pursuit_evasion_numpy.py`, used only by the test, the benchmark and `checks/` |
| `results/` | outputs of the timing runs reported below (not needed to run anything) |
| `experiments.py` | every actor experiment of the draft (R1 grid, C2-C7, E1/E1b/E7, Δv sweep, protocols, honest corrections, Shapley) -> `paper/` |
| `protocols.py` | Section 8 arms: oracle, simultaneous GDA, ABR, extragradient, optimistic GDA, averaging, grouped Shapley |
| `analysis.py`, `figures.py` | the draft's statistics (`paper/results/table1.json`) and its eight figures (`paper/figs/`) |
| `checks/` | the checking layer ported: `core.py`, `metrics.py`, `analyze_r1.py` (from `rev2/`) and `visualize_actor.py` (from `viz/`), with their R1 outputs and figures |
| `paper/` | checkpoints, results, figures and `README.md` with the reproduced numbers against the draft |

The reproduction of the draft's experiments is described in [paper/README.md](paper/README.md).

```bash
python3 torch_actor/test_parity.py 500
python3 torch_actor/bench.py 100
python3 torch_actor/run_grid.py --init-seeds 0 --ic-seeds 0 --device cpu --dtype float64 --out torch_actor/results/baseline_cpu_f64
python3 torch_actor/run_grid.py --init-seeds 0-9 --ic-seeds 0-9 --device mps --dtype float32 --out torch_actor/results/grid100_mps_f32
python3 torch_actor/compare_r1.py torch_actor/results/grid100_mps_f32
```

## What changed, what did not

Same model (Fourier features of time, one tanh layer, 16 linear heads to unit headings), same
loss (soft capture time, one-sided no-capture penalty, distance shaping), same protocol
(1,500 pursuer warm-up, 14,000 iterations of 3 P + 1 E Adam steps at 1e-2 / 1e-3, Adam reset
every 3,500), same initial parameters and IC sets (the NumPy `default_rng` streams are reused
exactly). Three implementation changes, none of which changes the mathematics:

1. **Autograd** replaces the hand-written backward pass.
2. **The 64-step rollout loop is a `cumsum`** on the universal cover. The original wraps
   positions mod L at every step; the loss only ever uses the minimum-image displacement, for
   which the wrap is a no-op, so the lifted positions give the same loss and gradient.
3. **A leading run axis.** All tensors are `(R, ...)`; the batched loss is the *sum* over runs,
   so each run's gradient is exactly what it would be alone, and Adam is elementwise. R = 1 is
   one published run; R = 100 is the whole R1 grid in one program.

Only the player being updated carries gradients (the other's headings are detached), which is
what the original does by calling `backward_dirs` for one player only.

## Parity (float64, CPU, baseline initial state)

```
L_E, L_P                 agree to 2e-16 (rel)
grad B, b, W  (P step)   1.4e-14, 5.5e-15, 2.9e-14   vs NumPy backward pass
grad B, b, W  (E step)   1.1e-14, 8.8e-15, 1.6e-14
```

Following the same training trajectory in both implementations, the parameters separate:
1.6e-5 after 200 warm-up steps, 5e-2 after 50 self-play iterations, order 1 after 500. A NumPy
run against the same NumPy run with a 1e-15 perturbation of the initial weights separates at the
same rate (1.6e-4 at 10, 2.5e-2 at 50, 1.0 at 500 iterations). So the drift is the chaos of the
adversarial dynamics that `REVISION_PLAN.md` §2.2 documents, not a port error, and **no
implementation other than the bit-identical original can reproduce `v2_actor.npz` or any R1
checkpoint individually.** What is reproducible is the ensemble.

## Ensemble check: 100-run R1 grid, torch on the Mac GPU (float32) vs NumPy R1

Median [IQR] of the final light metrics over the same 10 x 10 (init, IC) seeds:

| | torch mps f32 | NumPy R1 |
|---|---|---|
| Gram PR, E / P | 2.47 [2.32, 2.67] / 2.47 [2.32, 2.67] | 2.47 [2.32, 2.70] / 2.50 [2.30, 2.69] |
| constancy, E / P | 0.86 [0.79, 0.90] / 0.86 [0.80, 0.90] | 0.85 [0.81, 0.89] / 0.85 [0.82, 0.89] |
| PR(H), E / P | 5.61 [5.23, 6.19] / 6.77 [5.93, 7.47] | 5.73 [5.29, 6.13] / 6.51 [5.98, 7.28] |
| temporal PR of u, E / P | 2.73 [2.29, 3.15] / 3.24 [2.74, 3.59] | 2.80 [2.39, 3.23] / 3.35 [2.85, 3.77] |
| captured heads | 7 [6, 9] | 8 [6, 9] |

The same 10 runs (init 0–9, IC 0) on CPU float64, on MPS float32 and in NumPy also agree within
their IQRs. `rev2/metrics.full_metrics` reads the torch checkpoints unchanged.

## Timing (M4 Max, 12 threads; budget = 57,500 gradient steps per run)

| setting | ms per batched step | full budget | per run |
|---|---|---|---|
| NumPy original, 1 run | 0.97 | 56 s | 56 s |
| torch CPU f64, 1 run | 0.53 | 30 s | 30 s |
| torch CPU f64, 10 runs batched | 2.5 | 146 s | 14.6 s |
| torch CPU f64, 100 runs batched | 31.7 | 1,822 s | 18.2 s |
| torch MPS f32, 1 run | 0.62 | 36 s | 36 s |
| torch MPS f32, 10 runs | 0.64 | 37 s | 3.7 s |
| torch MPS f32, 100 runs | 0.69 | 40 s | 0.40 s |
| torch MPS f32, 1,000 runs | 3.6 | 208 s | 0.21 s |

End to end with checkpoints and light metrics: the published baseline in 31 s on CPU f64; the
100-run R1 grid in 41 s on the Mac GPU (NumPy R1 took 59 min on an M3 with 8 workers; about
9 min on this machine with 16 processes). One gradient step is 495 ATen operator calls, so on
a CUDA GPU in eager mode expect roughly 495 x (10–20 µs) = 5–10 ms per batched step, i.e.
5–10 min for the whole budget whatever the batch size up to a few thousand runs; CUDA graphs
(`torch.compile(mode="reduce-overhead")`) would remove most of that. Neither was measured here.

## Potential difficulties

- **Chaos.** Individual runs are not reproducible across implementations, precisions, devices
  or torch versions (see above). Any comparison with the NumPy results must be at the ensemble
  level, with the seed grid of R1. Per-run numbers quoted in the draft (`v2_actor.npz`) cannot
  be regenerated with this code.
- **MPS is float32 only.** The ensemble statistics match float64 within the spread, but the
  revision's chaos arm (1e-10 perturbations) is meaningless in float32, where the perturbation
  is below the rounding error. Use CPU float64 (or a CUDA GPU in float64) for that arm.
- **`eigvalsh` and friends are not implemented on MPS.** `light_metrics` moves the tiny
  Gram matrices to CPU; anything else from `rev2/metrics.py` should run on the saved
  checkpoints with NumPy, as before.
- **CPU batching does not scale.** Batched runs on CPU are only 2–4x more efficient than one
  run at a time, and 100 batched runs are slower per run than 10; for CPU-only machines (a
  cluster node without a GPU) use one process per run, as `rev2/run_r1.py` does, now at
  about 30 s per run per core.
- **Adam state semantics.** The Adam reset recreates both optimisers, as in `rev2/core.py`.
  `torch.optim.Adam` with default `foreach` matches the NumPy Adam formula exactly (verified
  implicitly by the 1.6e-5 agreement after 200 steps, which is rounding growth, not a bias).
- **The step is overhead-bound, not compute-bound.** 0.5 ms per step is Python plus 495
  operator launches; the arithmetic is microseconds. Larger heads, denser time grids or more
  runs are nearly free; a faster single run needs fewer operators (fusion via `torch.compile`),
  not a faster device.
- **Extending the loss** (e.g. R2's capture-truncated shaping `<S(t) d(t)>`) is now a one-line
  change in `game_loss` with no backward pass to derive, which was the point of the port. The
  finite-difference gradient checks of the original are replaced by autograd; keep
  `test_parity.py` as the gate for any change that is meant to preserve the published loss.
- **Not ported:** the critics (`pursuit_evasion_v2/v4.py`, `honest_critic_geom.py`,
  `geometries.py`), the protocol study (`run_game_protocols.py`, `run_honest_protocols.py`:
  extragradient, optimistic GDA, averaging), the Shapley attribution, and the distillation /
  obstacle variants of v3. All of them wrap the same forward functions, so each is a short
  driver on top of `pe_torch.py`.
