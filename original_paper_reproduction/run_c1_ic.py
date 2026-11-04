"""C1 second arm: IC-resampling variance (committed; decoupled randomness).

Complement of run_c1.py: parameter init FIXED at seed 0 (identical to the
baseline), evader-IC sample seed varies.  Protocol otherwise identical
(pursuer warm-up 1500, 3:1 steps, lr 1e-2/1e-3, lam_nc=2, lam_d=5, 10k
iters), chunked with a fresh Adam every 2,500 iterations to reproduce the
checkpoint-chaining protocol.

Usage: python3 run_c1_ic.py <ic_seed>     (writes c1_ic<seed>.npz)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
                                   backward_dirs, rollout, backward_rollout,
                                   forward_loss, backward_loss, init_params,
                                   sample_evader_ICs, Adam)

cfg = TrainConfig(n_h=16, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
                  L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
                  no_capture_penalty=2.0, lam_d=5.0, seed=0)
tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
phig = fourier_features(tg, cfg.t_max, cfg.n_freqs)
MAX_ITERS, CHUNK, WARMUP = 10000, 2500, 1500


def main(ic_seed):
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.4, seed=ic_seed)
    rng = np.random.default_rng(0)                      # FIXED init
    D = 2 * cfg.n_freqs + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)

    def step(who, optE, optP):
        uE, cE = forward_dirs(pE, phig); uP, cP = forward_dirs(pP, phig)
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P,
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

    optE, optP = Adam(pE, lr=1e-3), Adam(pP, lr=1e-2)
    for _ in range(WARMUP):
        step("P", optE, optP)
    it = 0
    while it < MAX_ITERS:
        optE, optP = Adam(pE, lr=1e-3), Adam(pP, lr=1e-2)  # chaining protocol
        for _ in range(CHUNK):
            for _ in range(3):
                step("P", optE, optP)
            L_E, L_P, nc = step("E", optE, optP)
            it += 1
            if it % 2000 == 0:
                print(f"[c1_ic{ic_seed}] it {it} L_P={L_P:.2f} "
                      f"no_cap={nc:.2f}", flush=True)
    np.savez(os.path.join(HERE, f"c1_ic{ic_seed}.npz"),
             BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
             BP=pP["B"], bP=pP["b_body"], WP=pP["W"], x_E0=x_E0, iter0=it)
    print(f"[c1_ic{ic_seed}] DONE at {it}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
