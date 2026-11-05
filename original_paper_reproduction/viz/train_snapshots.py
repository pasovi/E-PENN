"""Train the published baseline protocol and save parameter snapshots along the way.

Same protocol as rev2/core.py train_actor (RunConfig defaults = the published
baseline: 1500 pursuer-only warm-up steps, then 14,000 self-play iterations
at 3 P : 1 E steps, Adam 1e-2 / 1e-3 reset every 3,500, lam_d = 5,
lam_nc = 2), with (init_seed, ic_seed) = (0, 0) = the published configuration.
The step function is copied from core.train_actor so that the warm-up can be
snapshotted too; nothing analytic enters training.

Snapshots are dense early (where the headings change fastest) and sparse
later.  Output: one compressed npz holding the stacked parameters of both
players at every snapshot plus the schedule, read by viz/animate_training.py.

Usage: python3 viz/train_snapshots.py [--init-seed 0] [--ic-seed 0] [--out viz/snapshots_baseline]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rev2"))
from pursuit_evasion_numpy import (forward_dirs, backward_dirs, rollout,  # noqa: E402
                                   backward_rollout, forward_loss, backward_loss, Adam)
from core import RunConfig, initial_state, train_grid  # noqa: E402

WARMUP_SNAPS = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1250, 1500]
SELFPLAY_SNAPS = sorted(set([5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 225, 250,
                             275, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1100, 1250,
                             1500, 1750, 2000, 2250, 2500, 2750, 3000, 3500]
                            + list(range(4000, 14001, 250))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--ic-seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "viz", "snapshots_baseline"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = RunConfig(init_seed=a.init_seed, ic_seed=a.ic_seed)
    pE, pP, x0 = initial_state(cfg)
    phi = train_grid(cfg)

    def losses():
        uE, _ = forward_dirs(pE, phi); uP, _ = forward_dirs(pP, phi)
        xE, xP = rollout(uE, uP, x0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        LE, LP, _ = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                 cfg.n_t, cfg.lam_nc, lam_d=cfg.lam_d)
        return LE, LP

    def step(who, optE, optP):
        uE, cE = forward_dirs(pE, phi)
        uP, cP = forward_dirs(pP, phi)
        xE, xP = rollout(uE, uP, x0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        _, _, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                cfg.n_t, cfg.lam_nc, lam_d=cfg.lam_d)
        is_p = who == "P"
        dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.lam_nc, cfg.t_max, is_pursuer=is_p)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        if is_p:
            optP.step(pP, backward_dirs(pP, cP, duP), sign=-1.0)
        else:
            optE.step(pE, backward_dirs(pE, cE, duE), sign=-1.0)

    snaps = []   # (phase, it, LE, LP, pE copy, pP copy)
    def record(phase, it):
        LE, LP = losses()
        snaps.append((phase, it, LE, LP, {k: v.copy() for k, v in pE.items()},
                      {k: v.copy() for k, v in pP.items()}))
        print(f"[snap] {phase:8s} it {it:5d}  L_E={LE:+.4f}  L_P={LP:+.4f}  [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time()
    optE, optP = Adam(pE, lr=cfg.lr_E), Adam(pP, lr=cfg.lr_P)
    record("warmup", 0)
    for w in range(1, cfg.warmup + 1):
        step("P", optE, optP)
        if w in WARMUP_SNAPS:
            record("warmup", w)
    for it in range(1, cfg.n_iters + 1):
        if cfg.adam_reset_every and it > 1 and (it - 1) % cfg.adam_reset_every == 0:
            optE, optP = Adam(pE, lr=cfg.lr_E), Adam(pP, lr=cfg.lr_P)
        for _ in range(cfg.n_P_steps):
            step("P", optE, optP)
        for _ in range(cfg.n_E_steps):
            step("E", optE, optP)
        if it in SELFPLAY_SNAPS:
            record("selfplay", it)

    arrs = {"x_E0": x0,
            "phase": np.array([s[0] for s in snaps]),
            "it": np.array([s[1] for s in snaps]),
            "L_E": np.array([s[2] for s in snaps]),
            "L_P": np.array([s[3] for s in snaps])}
    for key, name in (("B", "B"), ("b_body", "b"), ("W", "W")):
        arrs[f"{name}E"] = np.stack([s[4][key] for s in snaps])
        arrs[f"{name}P"] = np.stack([s[5][key] for s in snaps])
    np.savez_compressed(os.path.join(a.out, "snapshots.npz"), **arrs)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({"init_seed": a.init_seed, "ic_seed": a.ic_seed, "n_snapshots": len(snaps),
                   "warmup_snaps": WARMUP_SNAPS, "selfplay_snaps": SELFPLAY_SNAPS,
                   "wall_s": time.time() - t0}, f, indent=1)
    print(f"[snap] wrote {len(snaps)} snapshots to {a.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
