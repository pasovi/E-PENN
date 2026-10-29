"""
Drivers for the two config-only revision experiments (no loss-code changes,
no V*/u* anywhere; identical training machinery to run_v2 / v2_actor.npz):

  E1  single-winding-sector ICs : 16 evader ICs restricted to an angular
      sector (phi ~ U[-15deg, +15deg], r ~ U[0.45, 1.35], seed 0); tests
      whether latent PR(H) tracks sector diversity or stays ~7.
  E7  shaping anneal            : identical ICs to v2_actor (seed 0), but
      lambda_d = 5 only through warm-up + first 4000 adversarial iters,
      then lambda_d = 0; tests which embedding observables are held up by
      the distance-shaping term.

Self-timed chaining: each invocation trains for ~TRAIN_SECONDS and writes a
checkpoint (same key layout as v2_actor.npz), so repeated calls extend the
run.  Mirrors run_v2's protocol: warm-up 1500 pursuer-only steps on fresh
start; 3:1 P:E adversarial steps; Adam lr_P=1e-2, lr_E=1e-3 (the values
actually used to produce v2_actor.npz); fresh Adam state per invocation
(same as the checkpoint-chaining protocol used throughout the paper).

Usage: python3 run_e17.py {e1|e7} [max_iters]
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
ANNEAL_AT = 4000          # E7: lambda_d -> 0 from this adversarial iter on
LAM_D = 5.0
LAM_NC = 2.0
LR_P, LR_E = 1e-2, 1e-3   # as in run_v2 (which produced v2_actor.npz)
WARMUP = 1500


def sector_ICs(n_h: int, seed: int = 0, r_max: float = 1.35) -> np.ndarray:
    """E1/E1b: ICs in one angular sector around +x.

    E1  (r_max=1.35): radii mixed as in v2; NOTE radii > L/2 = 1 place the
        evader across the antipodal cut locus, so the minimum-image
        displacement points along -x for ~39% of ICs: the ensemble spans
        both sides of the cut (reported as such in the paper).
    E1b (r_max=0.90): strictly single-sided, all ICs on the +x side of the
        cut and all within the t_max-capturability reach 0.88 (marginal at
        the edge).  This is the clean single-sector design.
    """
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.45, r_max, size=n_h)
    phi = rng.uniform(-np.pi / 12, np.pi / 12, size=n_h)
    pts = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
    return pts % 2.0


def main(exp: str, max_iters: int = 14000):
    cfg = TrainConfig(n_h=16, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
                      L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
                      no_capture_penalty=LAM_NC, lam_d=LAM_D, seed=0)
    ckpt = os.path.join(HERE, f"{exp}_actor.npz")
    rng = np.random.default_rng(cfg.seed)
    D = 2 * cfg.n_freqs + 1
    paramsE = init_params(cfg.n_h, cfg.n_b, D, rng)
    paramsP = init_params(cfg.n_h, cfg.n_b, D, rng)

    if exp == "e1":
        x_E0 = sector_ICs(cfg.n_h, seed=0)
    elif exp == "e1b":
        x_E0 = sector_ICs(cfg.n_h, seed=0, r_max=0.90)
    elif exp == "e7":
        x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.4, seed=cfg.seed)
    else:
        raise SystemExit("exp must be e1 or e7")

    it0, warm_done = 0, False
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        paramsE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
        paramsP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
        x_E0 = z["x_E0"]
        it0 = int(z["iter0"]); warm_done = True
        print(f"[{exp}] resumed at iter {it0}")
    if it0 >= max_iters:
        print(f"[{exp}] DONE at {it0} iters"); return

    optE, optP = Adam(paramsE, lr=LR_E), Adam(paramsP, lr=LR_P)
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)

    def lam_d_now(it):
        if exp == "e7":
            return LAM_D if it < ANNEAL_AT else 0.0
        return LAM_D

    def step(update_who: str, lam_d: float):
        uE, cacheE = forward_dirs(paramsE, phi)
        uP, cacheP = forward_dirs(paramsP, phi)
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P,
                         cfg.t_max, cfg.n_t)
        L_E_val, L_P_val, cacheL = forward_loss(
            xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max, cfg.n_t,
            cfg.no_capture_penalty, lam_d=lam_d)
        diag = {"ET": float(cacheL["ET"].mean()),
                "no_cap": float((cacheL["S"][:, -1] > 0.5).mean())}
        is_pursuer = (update_who == "P")
        dxE, dxP = backward_loss(1.0, cacheL, cfg.n_h, cfg.no_capture_penalty,
                                 cfg.t_max, is_pursuer=is_pursuer)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                    cfg.t_max, cfg.n_t)
        if is_pursuer:
            optP.step(paramsP, backward_dirs(paramsP, cacheP, duP), sign=-1.0)
        else:
            optE.step(paramsE, backward_dirs(paramsE, cacheE, duE), sign=-1.0)
        return L_E_val, L_P_val, diag

    t0 = time.time()
    if not warm_done:
        for i in range(WARMUP):
            _, LP, diag = step("P", LAM_D)
            if i % 500 == 0:
                print(f"[{exp} warmup] {i} L_P={LP:.3f} ET={diag['ET']:.3f}")

    it = it0
    while it < max_iters and time.time() - t0 < TRAIN_SECONDS:
        ld = lam_d_now(it)
        for _ in range(3):
            L_E_val, L_P_val, diag = step("P", ld)
        L_E_val, L_P_val, diag = step("E", ld)
        it += 1
        if it % 500 == 0:
            print(f"[{exp}] it {it} lam_d={ld} L_P={L_P_val:.3f} "
                  f"L_E={L_E_val:.3f} ET={diag['ET']:.3f} "
                  f"no_cap={diag['no_cap']:.3f}")

    np.savez(ckpt, BE=paramsE["B"], bE=paramsE["b_body"], WE=paramsE["W"],
             BP=paramsP["B"], bP=paramsP["b_body"], WP=paramsP["W"],
             x_E0=x_E0, iter0=it)
    print(f"[{exp}] saved at iter {it}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 14000)
