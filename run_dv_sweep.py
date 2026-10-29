"""Delta-v (speed-difference) sweep for embedding structure.

Vary the primary dimensionless group Dv = v_P - v_E, holding architecture
(K=12), horizon (t_max=4), geometry (L=2), capture radius (eps=0.08) and the
evader IC set (seed 0) fixed; re-train the actor from scratch at each Dv with
several init seeds.  Protocol matches the baseline/C3 actor (warm-up 1500
pursuer-only, 3:1 P:E steps, lr 1e-2/1e-3, lam_nc=2, lam_d=5, chunked fresh
Adam every 2500 iters, 14000 iters).

Reach radius R = Dv*t_max + eps on T^2=[0,L)^2 (min-image distance; cut locus
at d=L/2).  Ridge activation rho = R/(L/2) = 1 <=> Dv ~ 0.23; saturation
R = L/sqrt(2) <=> Dv ~ 0.335.

Env: SWEEP_ITERS (default 14000), SWEEP_WARMUP (default 1500),
     SWEEP_OUT (npz dir, default /tmp/sweep_ckpts).
Usage: python3 run_dv_sweep.py <v_P> <init_seed>   (prints one JSON line)
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
                                   backward_dirs, rollout, backward_rollout,
                                   forward_loss, backward_loss, init_params,
                                   sample_evader_ICs, Adam)

MAX_ITERS = int(os.environ.get("SWEEP_ITERS", 14000))
WARMUP    = int(os.environ.get("SWEEP_WARMUP", 1500))
CHUNK     = 2500
OUTDIR    = os.environ.get("SWEEP_OUT", "/tmp/sweep_ckpts")

def PR(eigs):
    eigs = np.clip(eigs, 0, None); s1 = eigs.sum(); s2 = (eigs**2).sum()
    return float(s1*s1/max(s2,1e-12))

def latent_PR(params, K, t_max, n_grid=1024):
    t = np.linspace(0.0, t_max, n_grid); phi = fourier_features(t, t_max, K)
    _, c = forward_dirs(params, phi); H = c["H"]
    Hc = H - H.mean(0, keepdims=True); C = Hc.T @ Hc / Hc.shape[0]
    return PR(np.linalg.eigvalsh(C))

def strat_PR(params, K, t_max, n_grid=1024):
    t = np.linspace(0.0, t_max, n_grid); phi = fourier_features(t, t_max, K)
    _, c = forward_dirs(params, phi); Z = c["z_head"]; n_h, T, _ = Z.shape
    M = np.transpose(Z,(1,0,2)).reshape(T, 2*n_h); Mc = M - M.mean(0, keepdims=True)
    return PR(np.linalg.eigvalsh(Mc.T @ Mc / T))

def main(v_P, pseed):
    cfg = TrainConfig(n_h=16, n_b=16, n_freqs=12, t_max=4.0, n_t=64, L=2.0,
                      v_E=1.0, v_P=v_P, eps=0.08, tau=5e-3,
                      no_capture_penalty=2.0, lam_d=5.0, seed=0)
    tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phig = fourier_features(tg, cfg.t_max, cfg.n_freqs)
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.4, seed=0)
    rng = np.random.default_rng(pseed)
    D = 2 * cfg.n_freqs + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)

    def step(who, optE, optP):
        uE, cE = forward_dirs(pE, phig); uP, cP = forward_dirs(pP, phig)
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        L_E, L_P, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                    cfg.n_t, cfg.no_capture_penalty, lam_d=cfg.lam_d)
        ip = who == "P"
        dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty,
                                 cfg.t_max, is_pursuer=ip)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                    cfg.t_max, cfg.n_t)
        if ip: optP.step(pP, backward_dirs(pP, cP, duP), sign=-1.0)
        else:  optE.step(pE, backward_dirs(pE, cE, duE), sign=-1.0)

    optE, optP = Adam(pE, 1e-3), Adam(pP, 1e-2)
    for _ in range(WARMUP): step("P", optE, optP)
    it = 0
    while it < MAX_ITERS:
        optE, optP = Adam(pE, 1e-3), Adam(pP, 1e-2)
        for _ in range(CHUNK):
            for _ in range(3): step("P", optE, optP)
            step("E", optE, optP); it += 1
            if it >= MAX_ITERS: break

    uE, _ = forward_dirs(pE, phig); uP, _ = forward_dirs(pP, phig)
    xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    disp = (xE - xP + cfg.L/2) % cfg.L - cfg.L/2
    dmin = np.sqrt((disp**2).sum(-1)).min(axis=1)
    captured = float((dmin <= cfg.eps).mean())
    R = (cfg.v_P - cfg.v_E) * cfg.t_max + cfg.eps
    d0 = np.sqrt(((( x_E0 + cfg.L/2) % cfg.L - cfg.L/2)**2).sum(-1))
    cap_share_ICs = float((d0 <= R).mean())

    res = {"v_P": round(v_P,3), "dv": round(cfg.v_P-cfg.v_E,3), "pseed": pseed,
           "iters": it, "R": round(R,4), "rho": round(R/(cfg.L/2),4),
           "latPR_E": latent_PR(pE, cfg.n_freqs, cfg.t_max),
           "latPR_P": latent_PR(pP, cfg.n_freqs, cfg.t_max),
           "zPR_E": strat_PR(pE, cfg.n_freqs, cfg.t_max),
           "zPR_P": strat_PR(pP, cfg.n_freqs, cfg.t_max),
           "captured_frac": captured, "cap_share_ICs": cap_share_ICs}
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(os.path.join(OUTDIR, f"dvsweep_vP{v_P:.2f}_s{pseed}.npz"),
             BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
             BP=pP["B"], bP=pP["b_body"], WP=pP["W"], x_E0=x_E0, iter0=it)
    print(json.dumps(res), flush=True)

if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]))
