"""
C1: seed reproducibility with DECOUPLED randomness.

The released pipeline uses one seed for both the parameter init and the
IC sample; here the IC set is held fixed (ic seed 0, identical to the
baseline) and only the parameter-initialisation seed varies.  Protocol
otherwise identical to the baseline (pursuer warm-up 1500, 3:1 steps,
lr 1e-2 / 1e-3, lam_nc = 2, lam_d = 5, lam_orth = 0 as in the actual
NumPy pipeline).  Runs to 10,000 iterations (baseline diagnostics
plateau by ~5,000).

Usage: python3 run_c1.py <init_seed> [max_iters]
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
                                   backward_dirs, rollout, backward_rollout,
                                   forward_loss, backward_loss, init_params,
                                   sample_evader_ICs, Adam)

TRAIN_SECONDS = 33.0
cfg = TrainConfig(n_h=16, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
                  L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
                  no_capture_penalty=2.0, lam_d=5.0, seed=0)
tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
phig = fourier_features(tg, cfg.t_max, cfg.n_freqs)
X_E0 = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.4, seed=0)  # FIXED ICs


def main(init_seed, max_iters=10000):
    ckpt = os.path.join(HERE, f"c1_s{init_seed}.npz")
    rng = np.random.default_rng(init_seed)
    D = 2 * cfg.n_freqs + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)
    it0, warm = 0, False
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
        pP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
        it0 = int(z["iter0"]); warm = True
        print(f"[c1 s{init_seed}] resumed at {it0}")
    if it0 >= max_iters:
        print(f"[c1 s{init_seed}] DONE at {it0}"); return
    optE, optP = Adam(pE, lr=1e-3), Adam(pP, lr=1e-2)

    def step(who):
        uE, cE = forward_dirs(pE, phig); uP, cP = forward_dirs(pP, phig)
        xE, xP = rollout(uE, uP, X_E0, cfg.L, cfg.v_E, cfg.v_P,
                         cfg.t_max, cfg.n_t)
        L_E, L_P, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau,
                                    cfg.t_max, cfg.n_t,
                                    cfg.no_capture_penalty, lam_d=cfg.lam_d)
        is_p = who == "P"
        dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty,
                                 cfg.t_max, is_pursuer=is_p)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                    cfg.t_max, cfg.n_t)
        if is_p:
            optP.step(pP, backward_dirs(pP, cP, duP), sign=-1.0)
        else:
            optE.step(pE, backward_dirs(pE, cE, duE), sign=-1.0)
        return L_E, L_P, float((cL["S"][:, -1] > 0.5).mean())

    t0 = time.time()
    if not warm:
        for i in range(1500):
            step("P")
    it = it0
    while it < max_iters and time.time() - t0 < TRAIN_SECONDS:
        for _ in range(3):
            step("P")
        L_E, L_P, nc = step("E")
        it += 1
        if it % 2000 == 0:
            print(f"[c1 s{init_seed}] it {it} L_P={L_P:.2f} no_cap={nc:.2f}")
    np.savez(ckpt, BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
             BP=pP["B"], bP=pP["b_body"], WP=pP["W"], x_E0=X_E0, iter0=it)
    print(f"[c1 s{init_seed}] saved at {it} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 10000)
