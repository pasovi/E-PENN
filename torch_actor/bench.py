"""Timing of the torch port against the NumPy original, and of batched runs.

For each setting: time 4 x N_IT gradient steps of the published protocol after a
short warm-up, report ms per step (per batch of R runs) and the extrapolated
wall time of the full baseline budget (warm-up 1,500 + 14,000 x 4 = 57,500 steps).
Also counts the ATen operator calls in one step (what a GPU would launch).
Usage: python3 torch_actor/bench.py [n_iters_timed]
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy_reference as ref                      # noqa: E402
import pe_torch as pt                              # noqa: E402

cfg = pt.Config()
BUDGET = cfg.warmup + cfg.n_iters * (cfg.n_P_steps + cfg.n_E_steps)
N_IT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rows = []


def report(name, R, per_step):
    full = BUDGET * per_step
    rows.append((name, R, per_step * 1e3, full, full / R))
    print(f"{name:34s} R={R:4d}  {per_step*1e3:8.3f} ms/step  full budget {full:7.1f} s"
          f"  = {full/R:7.2f} s per run", flush=True)


# ------------------------------------------------------------ NumPy original
def bench_numpy():
    pE, pP, x0 = pt.initial_state_np(cfg, 0, 0)
    pE = {"B": pE["B"], "b_body": pE["b"], "W": pE["W"]}
    pP = {"B": pP["B"], "b_body": pP["b"], "W": pP["W"]}
    phi = ref.fourier_features(np.linspace(0, cfg.t_max, cfg.n_t + 1), cfg.t_max, cfg.K)
    optE, optP = ref.Adam(pE, lr=cfg.lr_E), ref.Adam(pP, lr=cfg.lr_P)

    def step(who):
        uE, cE = ref.forward_dirs(pE, phi)
        uP, cP = ref.forward_dirs(pP, phi)
        xE, xP = ref.rollout(uE, uP, x0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        _, _, cL = ref.forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max, cfg.n_t,
                                    cfg.lam_nc, lam_d=cfg.lam_d)
        is_p = who == "P"
        dxE, dxP = ref.backward_loss(1.0, cL, cfg.n_h, cfg.lam_nc, cfg.t_max, is_pursuer=is_p)
        duE, duP = ref.backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        if is_p:
            optP.step(pP, ref.backward_dirs(pP, cP, duP), sign=-1.0)
        else:
            optE.step(pE, ref.backward_dirs(pE, cE, duE), sign=-1.0)

    for _ in range(20):
        step("P")
    t0 = time.perf_counter()
    for _ in range(N_IT):
        for _ in range(3):
            step("P")
        step("E")
    report("numpy original (hand-written bwd)", 1, (time.perf_counter() - t0) / (4 * N_IT))


# ------------------------------------------------------------------- torch
def bench_torch(R, device, dtype, label):
    runs = [(i % 10, i // 10) for i in range(R)]
    states = [pt.initial_state_np(cfg, *r) for r in runs]
    pE, pP, x0 = pt.stack_runs(states, device, dtype)
    t = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=device, dtype=dtype)
    phi = pt.fourier_features(t, cfg.t_max, cfg.K)
    optE, optP = pt.make_adam(pE, cfg.lr_E), pt.make_adam(pP, cfg.lr_P)
    sync = (torch.mps.synchronize if device == "mps" else
            torch.cuda.synchronize if device == "cuda" else (lambda: None))
    for _ in range(20):
        pt.step("P", pE, pP, x0, phi, cfg, optE, optP)
    sync()
    t0 = time.perf_counter()
    for _ in range(N_IT):
        for _ in range(3):
            pt.step("P", pE, pP, x0, phi, cfg, optE, optP)
        pt.step("E", pE, pP, x0, phi, cfg, optE, optP)
    sync()
    report(label, R, (time.perf_counter() - t0) / (4 * N_IT))
    return pE, pP, x0, phi, optE, optP


def count_ops():
    pE, pP, x0, phi, optE, optP = bench_torch(1, "cpu", torch.float64, "torch cpu f64 (for op count)")
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        pt.step("P", pE, pP, x0, phi, cfg, optE, optP)
    n = sum(e.count for e in prof.key_averages() if e.key.startswith("aten::"))
    print(f"\nATen operator calls in one gradient step (fwd + bwd + Adam): {n}\n")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, "
          f"mps {torch.backends.mps.is_available()}, cuda {torch.cuda.is_available()}; "
          f"timing {4*N_IT} steps per setting\n")
    bench_numpy()
    count_ops()
    bench_torch(1, "cpu", torch.float64, "torch cpu f64")
    bench_torch(1, "cpu", torch.float32, "torch cpu f32")
    for R in (10, 100):
        bench_torch(R, "cpu", torch.float64, "torch cpu f64 batched")
    if torch.backends.mps.is_available():
        for R in (1, 10, 100, 1000):
            bench_torch(R, "mps", torch.float32, "torch mps f32 batched")
    if torch.cuda.is_available():
        for R in (1, 10, 100, 1000):
            bench_torch(R, "cuda", torch.float32, "torch cuda f32 batched")
            bench_torch(R, "cuda", torch.float64, "torch cuda f64 batched")
    print("\nsummary (ms per batched step; seconds for the 57,500-step budget; seconds per run)")
    for name, R, ms, full, per in rows:
        print(f"  {name:34s} R={R:4d}  {ms:8.3f}  {full:8.1f}  {per:8.2f}")
