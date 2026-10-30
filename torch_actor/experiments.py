"""Run every actor experiment of the draft with the torch port.

Outputs (all inside torch_actor/):
  paper/ckpt/<arm>/<name>.npz     parameters (rev2 key layout + legacy keys)
  paper/results/<arm>.json(l)     per-run records; protocol/Shapley logs in the
                                  draft's JSON-line formats
  checks/ckpt/r1, checks/results/r1   the R1 ensemble in rev2's record format,
                                  for checks/analyze_r1.py

Stages (python3 torch_actor/experiments.py [stage ...]; default: all):
  r1        10 init x 10 IC grid, chaos arm (1e-10 perturbations, float64),
            no-reset arm                                   (rev2 R1; contains C1)
  controls  baseline protocol, 14k iters, 10 init seeds each:
            C3 K in {6, 24}; C4 diagonal sector; C5 lam_d in {0, 1};
            E1 sector, E1b one-sided, E1b reflected; E7 shaping anneal at 4k
  c2        fully symmetrised protocol, 28k iters, 7 seeds, snapshots 20k..28k
  c7        warm start: fit a curved heading family, then the game, PR every 250
  dv        v_P in {1.05 .. 1.5} x 3 init seeds, 14k iters
  protocols Fig. 6: oracle / selfplay / simul / abr, 14k steps, 3 seeds
  honest    Fig. 7: gda / eg / ogda / avg / ogda_ema / oracle, 6k iters, 3 seeds
  shapley   Fig. 8: grouped Shapley, oracle vs self-play pursuer, 3 seeds
  r1_trunc  the 10 x 10 grid with the capture-truncated distance term <S d> (rev2's R2);
            not part of the default "all" run
  r1_trunc_k0  same, with the body fed only t / t_max (no Fourier features, K = 0)
  r1_trunc_k0_d4  same with a 4-layer tanh body (16 units each) instead of 1 layer
  r1_trunc_k0_d4_lr0.1  same with both learning rates 10x smaller (1e-3 / 1e-4)
  r1_trunc_d4, r1_trunc_d4_lr0.1  the 4-layer body with the Fourier features (K = 12)
Devices: batches of >= 20 runs go to the Mac GPU in float32 when available; the
chaos arm and everything else run on CPU in float64.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "checks"))
import pe_torch as pt                              # noqa: E402
import protocols as pr                             # noqa: E402
from core import RunConfig                         # noqa: E402  (checks/core.py)
import metrics as M                                # noqa: E402  (checks/metrics.py)

SMOKE = bool(os.environ.get("SMOKE"))          # SMOKE=1: tiny budgets, separate folders
PAPER = os.path.join(HERE, "paper_smoke" if SMOKE else "paper")
CKPT = os.path.join(PAPER, "ckpt")
RES = os.path.join(PAPER, "results")
CHECKS = os.path.join(HERE, "checks_smoke" if SMOKE else "checks")
BUDGET = {"protocols": 200, "honest": 120, "shapley": 50, "c7_fit": 50} if SMOKE else \
         {"protocols": 14000, "honest": 6000, "shapley": 5000, "c7_fit": 20000}


def smoke(cfg: pt.Config) -> pt.Config:
    """Shrink a config's budget for the smoke test (keeps the ckpt_at structure)."""
    if not SMOKE:
        return cfg
    n = len(cfg.ckpt_at)
    return replace(cfg, warmup=min(cfg.warmup, 10), n_iters=20 * n, snapshot_every=10,
                   adam_reset_every=min(cfg.adam_reset_every, 15) if cfg.adam_reset_every else 0,
                   ckpt_at=tuple(20 * (k + 1) for k in range(n)))
GPU = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else None)
SEEDS10 = list(range(10))
ENV = {"torch": torch.__version__, "python": platform.python_version(),
       "machine": platform.machine(), "system": platform.system()}


def device_for(R: int):
    if GPU and R >= 20:
        return GPU, torch.float32
    return "cpu", torch.float64


def np_params(p: dict, r: int) -> dict:
    """Run r of batched NumPy params, in the key names of the original code."""
    out = {"B": p["B"][r], "b_body": p["b"][r], "W": p["W"][r]}
    out.update({k: p[k][r] for k in p if k not in ("B", "b", "W")})
    return out


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def dump_lines(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- helpers
def train_arm(arm: str, cfg: pt.Config, names: list, runs=None, states=None,
              rcfg: RunConfig | None = None, **kw) -> dict:
    """Train a batch, save one .npz per run under paper/ckpt/<arm>/, and a JSON of
    light metrics; return the raw result."""
    R = len(names)
    dev, dt = device_for(R)
    log(f"{arm}: {R} runs on {dev} {dt} ...")
    cfg = smoke(cfg)
    if SMOKE and "ckpt_at" in kw:
        kw["ckpt_at"] = cfg.ckpt_at
    out = pt.train(cfg, runs=runs, states=states, device=dev, dtype=dt,
                   snapshot_fn=pt.light_metrics, **kw)
    for r, name in enumerate(names):
        pt.save_run_ckpt(os.path.join(CKPT, arm, name + ".npz"), cfg, out, r)
    recs = [{"name": names[r], "run": (runs[r] if runs else None),
             "snapshots": [s[r] for s in out["snapshots"]]} for r in range(R)]
    dump({"arm": arm, "config": asdict(cfg), "device": dev, "dtype": str(dt),
          "seconds": out["seconds"], "env": ENV, "runs": recs},
         os.path.join(RES, arm + ".json"))
    log(f"{arm}: done in {out['seconds']:.0f} s")
    return out


# --------------------------------------------------------------------- R1
def r1_record(name, arm, rcfg: RunConfig, out, r, snapshots):
    """One record in the format of rev2/run_r1.py (full metrics at the checkpoints)."""
    x0 = out["x0"][r]
    full = {str(it): M.full_metrics(np_params(pE, r), np_params(pP, r), x0, rcfg)
            for it, (pE, pP) in sorted(out["ckpts"].items())}
    h = hashlib.sha1()
    for p in (out["pE"], out["pP"]):
        for k in sorted(p):
            h.update(np.ascontiguousarray(p[k][r]).tobytes())
    return {"name": name, "arm": arm, "config": asdict(rcfg), "digest": h.hexdigest()[:16],
            "saddle": M.saddle_reference(x0, rcfg), "x_E0": x0.tolist(),
            "full": full, "snapshots": snapshots, "seconds": out["seconds"] / out["x0"].shape[0],
            "env": ENV}


def stage_r1(truncate: bool = False, K: int | None = None, depth: int = 1, lr_scale: float = 1.0):
    """truncate=True: the 10 x 10 grid only, with the capture-truncated distance term,
    under checks/{ckpt,results}/r1_trunc and paper/ckpt/baseline_trunc.
    K: input harmonics (K = 0: the body sees only t / t_max); tag gets a _k<K> suffix."""
    cfg = pt.Config(truncate=truncate, depth=depth) if K is None else pt.Config(truncate=truncate, K=K, depth=depth)
    cfg = replace(cfg, lr_P=cfg.lr_P * lr_scale, lr_E=cfg.lr_E * lr_scale)
    base = replace(RunConfig(), lr_P=cfg.lr_P, lr_E=cfg.lr_E) if K is None else replace(RunConfig(), K=K, lr_P=cfg.lr_P, lr_E=cfg.lr_E)
    tag = (("r1_trunc" if truncate else "r1") + ("" if K is None else f"_k{K}") + ("" if depth == 1 else f"_d{depth}")
           + ("" if lr_scale == 1.0 else f"_lr{lr_scale:g}"))
    arms = {
        "grid": ([(i, c) for c in range(10) for i in range(10)],
                 [f"grid_i{i}_c{c}" for c in range(10) for i in range(10)],
                 [replace(base, init_seed=i, ic_seed=c) for c in range(10) for i in range(10)],
                 {}),
        "chaos": ([(0, 0, p, 1e-10) for p in range(1, 11)],
                  [f"chaos_p{p}" for p in range(1, 11)],
                  [replace(base, perturb_seed=p, perturb_scale=1e-10) for p in range(1, 11)],
                  {}),
        "noreset": ([(i, 0) for i in range(10)],
                    [f"noreset_i{i}_c0" for i in range(10)],
                    [replace(base, init_seed=i, adam_reset_every=0) for i in range(10)],
                    {"adam_reset_every": 0}),
    }
    if truncate or K is not None or depth != 1 or lr_scale != 1.0:
        arms = {"grid": arms["grid"]}
    os.makedirs(os.path.join(CHECKS, "results", tag), exist_ok=True)
    for arm, (runs, names, rcfgs, over) in arms.items():
        c = smoke(replace(cfg, **over))
        if SMOKE:
            rcfgs = [replace(rc, warmup=c.warmup, n_iters=c.n_iters, ckpt_at=c.ckpt_at,
                             adam_reset_every=c.adam_reset_every, snapshot_every=c.snapshot_every)
                     for rc in rcfgs]
        R = len(runs)
        dev, dt = ("cpu", torch.float64) if arm == "chaos" else device_for(R)
        log(f"{tag}/{arm}: {R} runs on {dev} {dt} ...")
        out = pt.train(c, runs=runs, device=dev, dtype=dt, snapshot_fn=pt.light_metrics)
        log(f"{tag}/{arm}: trained in {out['seconds']:.0f} s; scoring with rev2 metrics ...")
        for r, name in enumerate(names):
            pt.save_run_ckpt(os.path.join(CHECKS, "ckpt", tag, name + ".npz"), c, out, r)
            rec = r1_record(name, arm, rcfgs[r], out, r, [s[r] for s in out["snapshots"]])
            rec["device"] = f"{dev} {dt}"; rec["truncate"] = truncate
            dump(rec, os.path.join(CHECKS, "results", tag, name + ".json"))
        if arm == "grid":   # the published baseline configuration, also under paper/
            pt.save_run_ckpt(os.path.join(CKPT, ("baseline_trunc" if truncate else "baseline") + ("" if K is None else f"_k{K}")
                                          + ("" if depth == 1 else f"_d{depth}") + ("" if lr_scale == 1.0 else f"_lr{lr_scale:g}"),
                                          "grid_i0_c0.npz"), c, out, 0)
        log(f"{tag}/{arm}: done")


# --------------------------------------------------------------- controls
def sector_ICs(n_h, seed=0, r_max=1.35, reflect=False):
    """run_e17.sector_ICs / run_e1b_reflected.sector_ICs_reflected (verbatim draws)."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.45, r_max, size=n_h)
    phi = rng.uniform(-np.pi / 12, np.pi / 12, size=n_h)
    pts = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
    if reflect:
        pts[:, 0] = -pts[:, 0]
    return pts % 2.0


def c4_ICs(cfg: pt.Config):
    """run_c4c5.c4_ICs: baseline radius multiset rotated into a +-15 deg diagonal sector."""
    base = pt.sample_evader_ICs(cfg.n_h, cfg.L, cfg.ic_min_dist, 0)
    d = (base + cfg.L / 2) % cfg.L - cfg.L / 2
    r = np.linalg.norm(d, axis=1)
    rng = np.random.default_rng(0)
    ang = np.pi / 4 + rng.uniform(-np.pi / 12, np.pi / 12, size=cfg.n_h)
    pts = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    assert np.abs(pts).max() < cfg.L / 2
    return pts % cfg.L


def stage_controls():
    cfg = pt.Config()
    # C3: bandwidth sweep (separate batches: the input dimension changes)
    for K in (6, 24):
        c = replace(cfg, K=K)
        train_arm(f"c3_k{K}", c, [f"c3_k{K}_s{s}" for s in SEEDS10], runs=[(s, 0) for s in SEEDS10])
    # C4, C5, E1, E1b, E1b-reflected: one batch, per-run ICs and lam_d
    variants = {
        "c4": (c4_ICs(cfg), 5.0),
        "c5_lam0": (pt.sample_evader_ICs(cfg.n_h, cfg.L, cfg.ic_min_dist, 0), 0.0),
        "c5_lam1": (pt.sample_evader_ICs(cfg.n_h, cfg.L, cfg.ic_min_dist, 0), 1.0),
        "e1": (sector_ICs(cfg.n_h, 0, 1.35), 5.0),
        "e1b": (sector_ICs(cfg.n_h, 0, 0.90), 5.0),
        "e1b_refl": (sector_ICs(cfg.n_h, 0, 0.90, reflect=True), 5.0),
    }
    states, names, lam = [], [], []
    for v, (x0, ld) in variants.items():
        for s in SEEDS10:
            pE, pP, _ = pt.initial_state_np(cfg, s, 0)
            states.append((pE, pP, x0)); names.append(f"{v}_s{s}"); lam.append(ld)
    train_arm("variants", cfg, names, states=states, lam_d=np.array(lam))
    # E7: shaping anneal lam_d = 5 through warm-up and the first 4,000 iterations, then 0
    train_arm("e7", cfg, [f"e7_s{s}" for s in SEEDS10], runs=[(s, 0) for s in SEEDS10],
              lam_d_schedule=lambda it: 5.0 if it <= 4000 else 0.0,
              ckpt_at=(4000, 7000, 10500, 14000))


# --------------------------------------------------------------------- C2
def stage_c2():
    cfg = replace(pt.Config(), warmup=0, n_iters=28000, n_P_steps=1, n_E_steps=1,
                  lr_P=3e-3, lr_E=3e-3, adam_reset_every=2500,
                  ckpt_at=(20000, 22000, 24000, 26000, 28000))
    seeds = list(range(7))
    train_arm("c2", cfg, [f"c2_s{s}" for s in seeds], runs=[(s, 0) for s in seeds], symmetric=True)


# --------------------------------------------------------------------- C7
def curved_targets(cfg: pt.Config, seed: int) -> np.ndarray:
    """run_c7.curved_targets: random band-limited heading curves (n_h, n_t+1, 2)."""
    tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    rng = np.random.default_rng(seed)
    T = tg / cfg.t_max
    th = rng.uniform(0, 2 * np.pi, size=(cfg.n_h, 1)) * np.ones((1, tg.size))
    for k in range(1, 6):
        a = rng.normal(0, 1.8 / k, size=(cfg.n_h, 1))
        b = rng.normal(0, 1.8 / k, size=(cfg.n_h, 1))
        th = th + a * np.cos(2 * np.pi * k * T)[None, :] + b * np.sin(2 * np.pi * k * T)[None, :]
    return np.stack([np.cos(th), np.sin(th)], axis=-1)


def stage_c7(n_fit: int | None = None):
    n_fit = BUDGET["c7_fit"] if n_fit is None else n_fit
    cfg = pt.Config()
    seeds = SEEDS10
    dev, dt = "cpu", torch.float64
    states = [pt.initial_state_np(cfg, s, 0) for s in seeds]
    pE, pP, x0 = pt.stack_runs(states, dev, dt)
    t = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=dev, dtype=dt)
    phi = pt.fourier_features(t, cfg.t_max, cfg.K)
    tgt = {"E": torch.tensor(curved_targets(cfg, 100), device=dev, dtype=dt)[None],
           "P": torch.tensor(curved_targets(cfg, 101), device=dev, dtype=dt)[None]}
    log(f"c7/fit: {len(seeds)} runs, {n_fit} supervised iterations ...")
    t0 = time.time()
    opts = {"E": pt.make_adam(pE, 3e-3), "P": pt.make_adam(pP, 3e-3)}
    hist = []
    for it in range(1, n_fit + 1):
        mse = {}
        for a, p in (("E", pE), ("P", pP)):
            u, _, _ = pt.forward_dirs(p, phi)
            loss = ((u - tgt[a]) ** 2).mean(dim=(1, 2, 3))          # per run
            opts[a].zero_grad(set_to_none=True)
            loss.sum().backward()
            opts[a].step()
            mse[a] = loss.detach().cpu().numpy()
        if it % 1000 == 0 or it == n_fit:
            hist.append({"it": it, "mse_E": mse["E"].tolist(), "mse_P": mse["P"].tolist()})
    fit = {"pE": pt.to_numpy(pE), "pP": pt.to_numpy(pP), "x0": x0.cpu().numpy(),
           "ckpts": {0: (pt.to_numpy(pE), pt.to_numpy(pP))}, "snapshots": [], "seconds": time.time() - t0}
    snap = pt.light_metrics(pE, pP, x0, 0, cfg)
    for r, s in enumerate(seeds):
        pt.save_run_ckpt(os.path.join(CKPT, "c7", f"c7_fit_s{s}.npz"), replace(cfg, n_iters=0), fit, r)
    dump({"arm": "c7_fit", "n_fit": n_fit, "mse_hist": hist, "fit_metrics": snap,
          "seconds": fit["seconds"], "env": ENV}, os.path.join(RES, "c7_fit.json"))
    log(f"c7/fit: done in {fit['seconds']:.0f} s; final MSE E {mse['E'].mean():.2e} P {mse['P'].mean():.2e}")
    # game phase from the fitted state: no warm-up, 3:1, lr 1e-2 / 1e-3, PR tracked every 250
    cg = replace(cfg, warmup=0, ckpt_at=(250, 1000, 3000, 8000, 14000))
    states = [({"B": fit["pE"]["B"][r], "b": fit["pE"]["b"][r], "W": fit["pE"]["W"][r]},
               {"B": fit["pP"]["B"][r], "b": fit["pP"]["b"][r], "W": fit["pP"]["W"][r]},
               fit["x0"][r]) for r in range(len(seeds))]
    train_arm("c7", cg, [f"c7_game_s{s}" for s in seeds], states=states, snapshot_every=250)


# --------------------------------------------------------------------- dv
def stage_dv():
    cfg = pt.Config()
    vPs = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50]
    seeds = [0, 1, 2]
    runs = [(s, 0) for v in vPs for s in seeds]
    vP = np.array([v for v in vPs for s in seeds])
    names = [f"dvsweep_vP{v:.2f}_s{s}" for v in vPs for s in seeds]
    R = len(runs)
    dev, dt = device_for(R)
    cfg = smoke(cfg)
    log(f"dv: {R} runs on {dev} {dt} ...")
    out = pt.train(cfg, runs=runs, device=dev, dtype=dt, v_P=vP)
    rows = []
    for r in range(R):
        c = replace(cfg, v_P=float(vP[r]))
        pt.save_run_ckpt(os.path.join(CKPT, "dv", names[r] + ".npz"), c, out, r)
        rc = replace(RunConfig(), v_P=float(vP[r]), init_seed=runs[r][0])
        m = M.full_metrics(np_params(out["pE"], r), np_params(out["pP"], r), out["x0"][r], rc)
        g = m["game"]
        rows.append({"v_P": round(float(vP[r]), 3), "dv": round(float(vP[r]) - cfg.v_E, 3),
                     "pseed": runs[r][0], "iters": cfg.n_iters, "R": round(rc.reach, 4),
                     "rho": round(rc.reach / (cfg.L / 2), 4),
                     "latPR_E": m["E"]["lat"]["pr_H"], "latPR_P": m["P"]["lat"]["pr_H"],
                     "zPR_E": m["E"]["ens"]["tpr_Z"], "zPR_P": m["P"]["ens"]["tpr_Z"],
                     "gramPR_E": m["E"]["ens"]["gram_pr"], "gramPR_P": m["P"]["ens"]["gram_pr"],
                     "captured_frac": g["n_captured"] / cfg.n_h,
                     "cap_share_ICs": g["n_capturable"] / cfg.n_h})
    dump_lines(rows, os.path.join(RES, "dv_sweep_results_full.jsonl"))
    log(f"dv: done in {out['seconds']:.0f} s")


# ------------------------------------------------------- protocols / Shapley
def stage_protocols(truncate: bool = False):
    rows = []
    cfg = pt.Config(truncate=truncate)
    for p in ("oracle", "selfplay", "simul", "abr"):
        t0 = time.time()
        rows += pr.run_game_protocol(p, [0, 1, 2], total=BUDGET["protocols"], cfg=cfg)
        log(f"protocols{'_trunc' if truncate else ''}/{p}: {time.time() - t0:.0f} s")
    dump_lines(rows, os.path.join(RES, "game_protocols_results_full" + ("_trunc" if truncate else "") + ".jsonl"))


def stage_honest():
    rows = []
    for p in ("oracle", "gda", "ogda", "eg", "avg", "ogda_ema"):
        t0 = time.time()
        rows += pr.run_honest_protocol(p, [0, 1, 2], total=BUDGET["honest"])
        log(f"honest/{p}: {time.time() - t0:.0f} s")
    dump_lines(rows, os.path.join(RES, "honest_protocols_results.jsonl"))


def stage_shapley():
    t0 = time.time()
    outs = pr.run_shapley([0, 1, 2], n_iters=BUDGET["shapley"])
    for o in outs:
        dump(o, os.path.join(RES, f"shap_seed{o['seed']}.json"))
    log(f"shapley: {time.time() - t0:.0f} s")


STAGES = {"r1": stage_r1, "r1_trunc": lambda: stage_r1(truncate=True),
          "r1_trunc_k0": lambda: stage_r1(truncate=True, K=0),
          "r1_trunc_k0_d4": lambda: stage_r1(truncate=True, K=0, depth=4),
          "r1_trunc_k0_d4_lr0.1": lambda: stage_r1(truncate=True, K=0, depth=4, lr_scale=0.1),
          "r1_trunc_d4": lambda: stage_r1(truncate=True, depth=4),
          "r1_trunc_d4_lr0.1": lambda: stage_r1(truncate=True, depth=4, lr_scale=0.1),
          "protocols_trunc": lambda: stage_protocols(truncate=True), "controls": stage_controls, "c2": stage_c2, "c7": stage_c7,
          "dv": stage_dv, "protocols": stage_protocols, "honest": stage_honest,
          "shapley": stage_shapley}

if __name__ == "__main__":
    stages = sys.argv[1:] or [k for k in STAGES if "trunc" not in k]
    T0 = time.time()
    for s in stages:
        STAGES[s]()
    log(f"all done: {', '.join(stages)} in {time.time() - T0:.0f} s")
