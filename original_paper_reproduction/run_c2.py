"""
C2: fully symmetrised training.

Removes ALL four player-asymmetric training choices of the baseline:
  1. loss:      evader receives the mirrored survival bonus
                L_E = -E[T] - lam_nc * mean(S_last) * t_max - lam_d <d>
                (exactly zero-sum: L_P + L_E = 0, checked every log step);
  2. warm-up:   none (the baseline's pursuer-only warm-up is asymmetric);
  3. steps:     1:1 alternation;
  4. lr:        equal (3e-3 both).

The symmetric losses are implemented LOCALLY (the verified module is not
modified); the full parameter gradient of each player's loss is verified
against central finite differences before training starts.

Pre-registered decision rule (paper, C2): the unit-heading mid-band
fraction gap persists at >= 1.5x  => attributable to the inf-sup
structure; collapse of the gap => optimizer-made, retract the
game-attribution.  Raw head-SV scale gap is predicted to collapse either
way.  The comparison must be read alongside the equilibrium's capture
mass (saturation mechanism).

Usage: python3 run_c2.py [max_iters]
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
                                   backward_dirs, rollout, backward_rollout,
                                   init_params, sample_evader_ICs, Adam)

TRAIN_SECONDS = 33.0
CKPT = os.path.join(HERE, "c2_actor.npz")
LAM_NC, LAM_D, LR = 2.0, 5.0, 3e-3

cfg = TrainConfig(n_h=16, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
                  L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
                  no_capture_penalty=LAM_NC, lam_d=LAM_D, seed=0)
tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
phig = fourier_features(tg, cfg.t_max, cfg.n_freqs)
x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.4, seed=0)


# ---------------- symmetric loss (local; verified below) --------------------
def sym_forward_loss(xE, xP):
    dt = cfg.t_max / cfg.n_t
    disp = (xE - xP + cfg.L / 2.0) % cfg.L - cfg.L / 2.0
    d2 = (disp ** 2).sum(axis=-1)
    d = np.sqrt(d2 + 1e-12)
    arg = (cfg.eps ** 2 - d2) / cfg.tau
    log_1mp = -np.logaddexp(0.0, arg)
    p = 1.0 - np.exp(log_1mp)
    S = np.exp(np.cumsum(log_1mp, axis=1))
    ET = dt * (0.5 * S[:, 0] + S[:, 1:-1].sum(axis=1) + 0.5 * S[:, -1])
    nc = LAM_NC * S[:, -1] * cfg.t_max
    d_mean = d.mean(axis=1)
    L_P = float((ET + nc + LAM_D * d_mean).mean())
    L_E = float((-ET - nc - LAM_D * d_mean).mean())     # exact zero-sum
    cache = {"disp": disp, "d": d, "p": p, "S": S, "dt": dt}
    return L_E, L_P, cache


def sym_backward_loss(cache, is_pursuer):
    S, p, disp, d = cache["S"], cache["p"], cache["disp"], cache["d"]
    dt = cache["dt"]; n_h, T = S.shape
    sign = +1.0 if is_pursuer else -1.0
    dET = sign * np.ones(n_h) / n_h
    dS = np.zeros_like(S)
    dS[:, 0] = 0.5 * dt * dET
    if T > 2:
        dS[:, 1:-1] = (dt * dET)[:, None]
    dS[:, -1] = 0.5 * dt * dET
    dS[:, -1] += sign * LAM_NC * cfg.t_max / n_h          # symmetric bonus
    d_logS = dS * S
    d_log1mp = np.cumsum(d_logS[:, ::-1], axis=1)[:, ::-1]
    d_arg = -p * d_log1mp
    d_d2 = -d_arg / cfg.tau
    d_d2 = d_d2 + (sign * LAM_D / (n_h * T)) * 0.5 / d
    d_disp = 2.0 * disp * d_d2[..., None]
    return d_disp.copy(), -d_disp.copy()


def player_loss_and_grads(pE, pP, who):
    uE, cE = forward_dirs(pE, phig); uP, cP = forward_dirs(pP, phig)
    xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    L_E, L_P, cL = sym_forward_loss(xE, xP)
    dxE, dxP = sym_backward_loss(cL, is_pursuer=(who == "P"))
    duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                cfg.t_max, cfg.n_t)
    if who == "P":
        return L_P, backward_dirs(pP, cP, duP), (L_E, L_P, cL)
    return L_E, backward_dirs(pE, cE, duE), (L_E, L_P, cL)


def fd_check(pE, pP):
    rng = np.random.default_rng(11)
    max_rel = 0.0
    for who, params in (("P", pP), ("E", pE)):
        L0, g0, _ = player_loss_and_grads(pE, pP, who)
        for key in ("B", "b_body", "W"):
            arr = params[key]
            for _ in range(2):
                idx = tuple(rng.integers(0, s) for s in arr.shape)
                h = 1e-6
                arr[idx] += h; Lp, _, _ = player_loss_and_grads(pE, pP, who)
                arr[idx] -= 2 * h; Lm, _, _ = player_loss_and_grads(pE, pP, who)
                arr[idx] += h
                fd = (Lp - Lm) / (2 * h)
                an = g0[key][idx]
                rel = abs(fd - an) / max(abs(fd), abs(an), 1e-12)
                max_rel = max(max_rel, rel)
    print("[c2] FD gradient check (both players, sym loss): max rel = %.2e"
          % max_rel)
    assert max_rel < 1e-6, "FD check failed"


def main(max_iters=14000):
    rng = np.random.default_rng(0)
    D = 2 * cfg.n_freqs + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)
    it0 = 0
    if os.path.exists(CKPT):
        z = np.load(CKPT)
        pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
        pP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
        it0 = int(z["iter0"])
        print("[c2] resumed at %d" % it0)
    else:
        fd_check(pE, pP)
    if it0 >= max_iters:
        print("[c2] DONE at %d" % it0); return
    optE, optP = Adam(pE, lr=LR), Adam(pP, lr=LR)

    t0 = time.time(); it = it0
    while it < max_iters and time.time() - t0 < TRAIN_SECONDS:
        _, gP, _ = player_loss_and_grads(pE, pP, "P")
        optP.step(pP, gP, sign=-1.0)
        L_E, gE, (le, lp, cL) = player_loss_and_grads(pE, pP, "E")
        optE.step(pE, gE, sign=-1.0)
        it += 1
        if it % 500 == 0:
            nc = float((cL["S"][:, -1] > 0.5).mean())
            print("[c2] it %d  L_P=%.3f  L_E=%.3f  zero-sum=%.2e  no_cap=%.2f"
                  % (it, lp, le, lp + le, nc))
    np.savez(CKPT, BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
             BP=pP["B"], bP=pP["b_body"], WP=pP["W"], x_E0=x_E0, iter0=it)
    print("[c2] saved at %d (%.0fs)" % (it, time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14000)
