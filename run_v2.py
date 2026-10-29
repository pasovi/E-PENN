"""Driver for the v2 paper: train V, do all verification, save figures.

Splits into phases so each phase fits inside one bash call:
  PHASE 1: train (or resume) the multi-head open-loop actor for longer.
  PHASE 2: train the HJI value-function critic V(Δ).
  PHASE 3: train V on T^3 (higher-dim demo).
  PHASE 4: verification suite (alignment check, capture-time maps,
           V vs analytic comparison) + assembled figures.

Run as:
    python3 run_v2.py 1        # phase 1 only
    python3 run_v2.py 1 2 3 4  # all phases
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

import pursuit_evasion_numpy as pen
import pursuit_evasion_v2 as v2
from pursuit_evasion_numpy import (
    TrainConfig, train, forward_dirs, fourier_features, rollout,
    sample_evader_ICs, init_params, Adam, forward_loss, backward_loss,
    backward_rollout, backward_dirs,
)

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"

# ---------------- Config ----------------
def make_actor_cfg():
    return TrainConfig(
        n_iters=2500, log_every=200, n_h=16,
        n_P_steps=3, n_E_steps=1,
        lr_P=1e-2, lr_E=1e-3, lam_d=5.0,
        pretrain_P_iters=1500, no_capture_penalty=2.0,
        seed=0,
    )

ACTOR_CKPT = OUT + "v2_actor.npz"
V_CKPT = OUT + "v2_V.npz"
V3D_CKPT = OUT + "v2_V3d.npz"

# =============================================================================
# Phase 1: longer training of the actor (with checkpoint chaining)
# =============================================================================

def _train_actor_chunk(cfg: TrainConfig, n_chunk: int,
                       resume: bool = False) -> dict:
    """Run n_chunk iterations of adversarial training; resume from ckpt if asked."""
    rng = np.random.default_rng(cfg.seed)
    D = 2 * cfg.n_freqs + 1
    paramsE = init_params(cfg.n_h, cfg.n_b, D, rng)
    paramsP = init_params(cfg.n_h, cfg.n_b, D, rng)
    optE = Adam(paramsE, lr=cfg.lr_E)
    optP = Adam(paramsP, lr=cfg.lr_P)
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, 0.4, cfg.seed)

    pretrain_done = False
    history = {"iter": [], "L_E": [], "L_P": [], "ET": [],
               "final_d": [], "no_cap_frac": []}
    iter0 = 0
    if resume and os.path.exists(ACTOR_CKPT):
        npz = np.load(ACTOR_CKPT)
        paramsE = {"B": npz["BE"], "b_body": npz["bE"], "W": npz["WE"]}
        paramsP = {"B": npz["BP"], "b_body": npz["bP"], "W": npz["WP"]}
        x_E0 = npz["x_E0"]
        # warmup is done in the first run, so skip subsequent times
        pretrain_done = True
        if "history" in npz.files:
            history = json.loads(str(npz["history"]))
        iter0 = int(npz["iter0"]) if "iter0" in npz.files else 0
        # reinit Adam states fresh (slight info loss but simpler)
        optE = Adam(paramsE, lr=cfg.lr_E)
        optP = Adam(paramsP, lr=cfg.lr_P)
        print(f"[actor] resumed at iter {iter0}")

    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)

    def one_step(who):
        uE, cE = forward_dirs(paramsE, phi)
        uP, cP = forward_dirs(paramsP, phi)
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max,
                          cfg.n_t)
        LE, LP, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                  cfg.n_t, cfg.no_capture_penalty,
                                  lam_d=cfg.lam_d)
        diag = {"ET_mean": float(cL["ET"].mean()),
                "final_dist_mean": float(np.sqrt(cL["d2"][:, -1]).mean()),
                "no_capture_frac": float((cL["S"][:, -1] > 0.5).mean())}
        is_p = (who == "P")
        dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty,
                                 cfg.t_max, is_pursuer=is_p)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                    cfg.t_max, cfg.n_t)
        if is_p:
            gP = backward_dirs(paramsP, cP, duP)
            optP.step(paramsP, gP, sign=-1.0)
        else:
            gE = backward_dirs(paramsE, cE, duE)
            optE.step(paramsE, gE, sign=-1.0)
        return LE, LP, diag

    if not pretrain_done:
        for it in range(cfg.pretrain_P_iters):
            LE, LP, diag = one_step("P")
            if it % 400 == 0:
                print(f"[pretrain-P] it {it:4d}  L_P={LP:.3f}  "
                      f"final_d={diag['final_dist_mean']:.3f}  "
                      f"no_cap={diag['no_capture_frac']:.2f}")

    t0 = time.time()
    for it in range(n_chunk):
        for _ in range(cfg.n_P_steps):
            LE, LP, diag = one_step("P")
        for _ in range(cfg.n_E_steps):
            LE, LP, diag = one_step("E")
        if it % cfg.log_every == 0 or it == n_chunk - 1:
            history["iter"].append(iter0 + it)
            history["L_E"].append(float(LE))
            history["L_P"].append(float(LP))
            history["ET"].append(diag["ET_mean"])
            history["final_d"].append(diag["final_dist_mean"])
            history["no_cap_frac"].append(diag["no_capture_frac"])
            print(f"it {iter0+it:5d}  L_P={LP:.3f}  L_E={LE:.3f}  "
                  f"E[T]={diag['ET_mean']:.3f}  "
                  f"final_d={diag['final_dist_mean']:.3f}  "
                  f"no_cap={diag['no_capture_frac']:.2f}  "
                  f"[{time.time()-t0:.0f}s]")

    np.savez(ACTOR_CKPT,
             BE=paramsE["B"], bE=paramsE["b_body"], WE=paramsE["W"],
             BP=paramsP["B"], bP=paramsP["b_body"], WP=paramsP["W"],
             x_E0=x_E0, iter0=iter0 + n_chunk,
             history=json.dumps(history))
    return {"paramsE": paramsE, "paramsP": paramsP, "x_E0": x_E0,
            "history": history, "cfg": make_actor_cfg()}


def phase1(n_chunk=3500, resume=False):
    cfg = make_actor_cfg()
    return _train_actor_chunk(cfg, n_chunk=n_chunk, resume=resume)


# =============================================================================
# Phase 2: train HJI critic V(Δ)
# =============================================================================
def phase2(n_iters=4000):
    params, hist = v2.train_V(
        L_torus=2.0, v_E=1.0, v_P=1.2, eps=0.08,
        n_dim=2, n_freqs=6, hidden=64,
        n_iters=n_iters, n_samples=512, lr=3e-3, seed=0, log_every=400)
    np.savez(V_CKPT, **params, hist=json.dumps(hist))
    v2.plot_V_loss(hist, OUT + "fig_v2_V_loss.png")
    return params, hist


# =============================================================================
# Phase 3: T^3 demo
# =============================================================================
def phase3(n_iters=1500):
    params, hist = v2.demo_3d_torus(L=2.0, v_E=1.0, v_P=1.2, eps=0.1,
                                     n_iters=n_iters, n_freqs=4, hidden=48)
    np.savez(V3D_CKPT, **params, hist=json.dumps(hist))
    return params, hist


# =============================================================================
# Phase 4: verification + figures
# =============================================================================
def _load_actor():
    npz = np.load(ACTOR_CKPT)
    paramsE = {"B": npz["BE"], "b_body": npz["bE"], "W": npz["WE"]}
    paramsP = {"B": npz["BP"], "b_body": npz["bP"], "W": npz["WP"]}
    x_E0 = npz["x_E0"]
    return paramsE, paramsP, x_E0


def _load_V(path=V_CKPT):
    npz = np.load(path)
    keys = ("B1", "b1", "B2", "b2", "w", "c")
    return {k: npz[k] for k in keys}


def phase4():
    cfg = make_actor_cfg()
    paramsE, paramsP, x_E0 = _load_actor()

    # 1. alignment check
    v2.plot_alignment_check(paramsE, paramsP, x_E0, cfg,
                             OUT + "fig_v2_alignment.png")

    # 2. capture time map -- open-loop actors (nearest head)
    print("computing capture-time map for open-loop actors ...")
    xs, ys, Tc = v2.capture_time_map(paramsE, paramsP, cfg, n_grid=24)
    v2.plot_capture_time_map(xs, ys, Tc, cfg,
                              OUT + "fig_v2_capture_time_actor.png",
                              title="Open-loop multi-head actor")

    # 3. capture time map -- HJI feedback policy
    print("computing capture-time map for V-feedback policy ...")
    paramsV = _load_V()
    xs2, ys2, Tc2 = v2.capture_time_map_feedback(
        paramsV, cfg, n_grid=24, n_freqs_V=6, h_fd=5e-3)
    v2.plot_capture_time_map(xs2, ys2, Tc2, cfg,
                              OUT + "fig_v2_capture_time_feedback.png",
                              title="HJI saddle-point feedback policy")

    # 4. analytic capture time
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)
    Tc_analytic = v2.analytic_V(pts, cfg.L, cfg.v_E, cfg.v_P, cfg.eps)\
        .reshape(len(xs), len(ys))
    v2.plot_capture_time_map(xs, ys, Tc_analytic, cfg,
                              OUT + "fig_v2_capture_time_analytic.png",
                              title="Analytic $|\\Delta|_T / (v_P - v_E)$")

    # 5. V vs analytic plot
    v2.plot_value_function(paramsV, cfg,
                            OUT + "fig_v2_value_function.png",
                            n_grid=80, n_freqs_V=6)

    # 6. dump a small JSON of summary diagnostics
    valid = ~np.isnan(Tc)
    valid_fb = ~np.isnan(Tc2)
    summary = {
        "n_train_iter_actor": int(np.load(ACTOR_CKPT)["iter0"]),
        "actor_capture_fraction": float(valid.mean()),
        "actor_mean_Tc_when_captured": float(np.nanmean(Tc)),
        "feedback_capture_fraction": float(valid_fb.mean()),
        "feedback_mean_Tc_when_captured": float(np.nanmean(Tc2)),
        "analytic_mean_Tc": float(Tc_analytic.mean()),
        "actor_Tc_min": float(np.nanmin(Tc)) if valid.any() else None,
        "feedback_Tc_min": float(np.nanmin(Tc2)) if valid_fb.any() else None,
    }
    print("\nSUMMARY:\n", json.dumps(summary, indent=2))
    with open(OUT + "v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    phases = sys.argv[1:] or ["1", "2", "3", "4"]
    for ph in phases:
        ph = ph.strip()
        if ph == "1":
            phase1(n_chunk=3500, resume=False)
        elif ph == "1b":
            phase1(n_chunk=3500, resume=True)
        elif ph == "2":
            phase2(n_iters=4000)
        elif ph == "3":
            phase3(n_iters=1500)
        elif ph == "4":
            phase4()
        else:
            print(f"Unknown phase: {ph}")
