"""Reflection control for Oleksii's question about the winding figure.

Claim under test: the left/right asymmetry of the E1b winding cloud comes
from the *one-sided* initial conditions (a +x sector), not from any cyclic
topology of the winding axis.  Test: reflect the ICs across the y-axis
(x -> -x), keep EVERYTHING else identical (same seed-0 init, same optimiser,
same schedule), retrain, and check that the winding cloud mirrors
(w_x -> -w_x, w_y unchanged).

This reuses the exact training machinery of run_e17.py / v2_actor.npz; no
loss-code changes, no analytic V*/u* anywhere.  Self-timed checkpoint
chaining (same as run_e17): repeated calls extend the run.

Usage: python3 run_e1b_reflected.py [max_iters]
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
    backward_dirs, rollout, backward_rollout, forward_loss, backward_loss,
    init_params, Adam)

TRAIN_SECONDS = 40.0
LAM_D, LAM_NC = 5.0, 2.0
LR_P, LR_E = 1e-2, 1e-3
WARMUP = 1500
CKPT = os.path.join(HERE, "e1b_refl_actor.npz")


def sector_ICs_reflected(n_h, seed=0, r_max=0.90):
    """Exactly the E1b +x sector (same seed-0 draw), reflected across the
    y-axis: x -> -x.  Same r, same |phi|, mirrored side."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.45, r_max, size=n_h)
    phi = rng.uniform(-np.pi / 12, np.pi / 12, size=n_h)
    pts = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
    pts[:, 0] = -pts[:, 0]                     # reflect across the y-axis
    return pts % 2.0


def winding(pE, phi, dt, vE, L):
    u, _ = forward_dirs(pE, phi)
    return (dt * vE * (0.5 * (u[:, :-1] + u[:, 1:])).sum(1)) / L


def main(max_iters=14000):
    cfg = TrainConfig(n_h=16, n_b=16, n_freqs=12, t_max=4.0, n_t=64, L=2.0,
                      v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
                      no_capture_penalty=LAM_NC, lam_d=LAM_D, seed=0)
    rng = np.random.default_rng(cfg.seed); D = 2 * cfg.n_freqs + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)   # SAME seed-0 init as e1b
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)
    x_E0 = sector_ICs_reflected(cfg.n_h, seed=0, r_max=0.90)

    it0, warm_done = 0, False
    if os.path.exists(CKPT):
        z = np.load(CKPT)
        pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
        pP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
        x_E0 = z["x_E0"]; it0 = int(z["iter0"]); warm_done = True
        print(f"[e1b_refl] resumed at iter {it0}")
    if it0 >= max_iters:
        print(f"[e1b_refl] DONE at {it0} iters"); return

    optE, optP = Adam(pE, lr=LR_E), Adam(pP, lr=LR_P)
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs); dt = cfg.t_max / cfg.n_t

    def step(who):
        uE, cE = forward_dirs(pE, phi); uP, cP = forward_dirs(pP, phi)
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        LE, LP, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                  cfg.n_t, cfg.no_capture_penalty, lam_d=LAM_D)
        isP = (who == "P")
        dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty, cfg.t_max, is_pursuer=isP)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        if isP: optP.step(pP, backward_dirs(pP, cP, duP), sign=-1.0)
        else:   optE.step(pE, backward_dirs(pE, cE, duE), sign=-1.0)
        return LP

    t0 = time.time()
    if not warm_done:
        for i in range(WARMUP): step("P")
        print("[e1b_refl] warmup done")
    it = it0
    while it < max_iters and time.time() - t0 < TRAIN_SECONDS:
        for _ in range(3): step("P")
        step("E"); it += 1
    np.savez(CKPT, BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
             BP=pP["B"], bP=pP["b_body"], WP=pP["W"], x_E0=x_E0, iter0=it)
    w = winding(pE, phi, dt, cfg.v_E, cfg.L)
    print(f"[e1b_refl] saved at iter {it} ({time.time()-t0:.0f}s)  "
          f"mean w_x={w[:,0].mean():+.2f}  heads w_x<0: {(w[:,0]<0).sum()}/{len(w)}  "
          f"w_x=[{w[:,0].min():+.2f},{w[:,0].max():+.2f}]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14000)
