"""C3: input-bandwidth sweep K in {6, 24} (committed; K = 12 is the baseline).

Pre-registered signature (S 'fourier'): the latent spectral break tracks the
input Fourier bandwidth K (architectural reading, predicted) rather than
sitting at a game-set scale.  Protocol identical to the baseline v2 actor
(warm-up 1500 pursuer-only, 3:1 steps, lr 1e-2/1e-3, lam_nc=2, lam_d=5,
seed 0, 14k iters), chunked with a fresh Adam every 2,500 iterations to
reproduce the checkpoint-chaining protocol.  Only n_freqs changes; the body
input dimension D = 2K+1 follows.

Usage: python3 run_c3.py <K>      (writes c3_k<K>_actor.npz)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
                                   backward_dirs, rollout, backward_rollout,
                                   forward_loss, backward_loss, init_params,
                                   sample_evader_ICs, Adam)

MAX_ITERS, CHUNK, WARMUP = 14000, 2500, 1500


def main(K):
    cfg = TrainConfig(n_h=16, n_b=16, n_freqs=K, t_max=4.0, n_t=64,
                      L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
                      no_capture_penalty=2.0, lam_d=5.0, seed=0)
    tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phig = fourier_features(tg, cfg.t_max, cfg.n_freqs)
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.4, seed=0)
    rng = np.random.default_rng(0)
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
        optE, optP = Adam(pE, lr=1e-3), Adam(pP, lr=1e-2)
        for _ in range(CHUNK):
            for _ in range(3):
                step("P", optE, optP)
            L_E, L_P, nc = step("E", optE, optP)
            it += 1
            if it % 2000 == 0:
                print(f"[c3 K={K}] it {it} L_P={L_P:.2f} no_cap={nc:.2f}",
                      flush=True)
    np.savez(os.path.join(HERE, f"c3_k{K}_actor.npz"),
             BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
             BP=pP["B"], bP=pP["b_body"], WP=pP["W"], x_E0=x_E0, iter0=it)
    print(f"[c3 K={K}] DONE at {it}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
