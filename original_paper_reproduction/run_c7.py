"""
C7: architecture-default control (two arms in one driver).

Arm (a) expressivity/sanity  = the supervised fit phase itself.
Arm (b) warm-start collapse  = game training FROM the fitted state.

Phase "fit":  fit the identical multi-head architecture by gradient
  descent to a CURVED, band-limited family of unit-heading targets
  (random Fourier heading curves, k <= 3, no game information of any
  kind: no V*, no u*, no Delta, no opponent).  Reports the target
  ensemble PR and the achieved PR(Z)/fit error.  Includes a
  finite-difference check of the supervised gradient path.

Phase "game": load the fitted state for BOTH players and run the
  standard adversarial loss (lam_d = 5, lam_nc = 2, 3:1 steps,
  lr 1e-2/1e-3) with NO pursuer warm-up (the warm start replaces it),
  tracking PR(Z_E), PR(Z_P), PR(H_E) every 250 iterations.

Pre-registered decision rule (paper, committed-experiments C7):
  collapse of PR(Z_E) to ~2-3  -> shaped-game objective actively
                                  selects near-constant strategies;
  persistence near the fitted ~8 -> architecture/optimiser-path default,
                                  Sec. 6.2(i) must be weakened.

Usage:  python3 run_c7.py fit
        python3 run_c7.py game [max_iters]
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
FIT_CKPT = os.path.join(HERE, "c7_fit.npz")
GAME_CKPT = os.path.join(HERE, "c7_actor.npz")

cfg = TrainConfig(n_h=16, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
                  L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
                  no_capture_penalty=2.0, lam_d=5.0, seed=0)
tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
phig = fourier_features(tg, cfg.t_max, cfg.n_freqs)
x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.4, seed=0)   # same ICs as baseline


def PR_of(X):
    Xc = X - X.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(Xc.T @ Xc / X.shape[0])[::-1]
    ev = np.maximum(ev, 0)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300))


def curved_targets(seed):
    """Random band-limited heading curves, k <= 3.  No game information."""
    rng = np.random.default_rng(seed)
    T = tg / cfg.t_max
    th = rng.uniform(0, 2 * np.pi, size=(cfg.n_h, 1)) * np.ones((1, tg.size))
    for k in range(1, 6):
        a = rng.normal(0, 1.8 / k, size=(cfg.n_h, 1))
        b = rng.normal(0, 1.8 / k, size=(cfg.n_h, 1))
        th = th + a * np.cos(2 * np.pi * k * T)[None, :] \
                + b * np.sin(2 * np.pi * k * T)[None, :]
    u = np.stack([np.cos(th), np.sin(th)], axis=-1)      # (n_h, T, 2)
    return u


def fit_loss_and_grads(params, u_tgt):
    u, cache = forward_dirs(params, phig)
    diff = u - u_tgt
    L = float((diff ** 2).mean())
    du = 2.0 * diff / diff.size * diff.shape[-1] * 0  # placeholder (replaced below)
    # dL/du for mean over all entries: 2*diff / N_total
    du = 2.0 * diff / diff.size
    grads = backward_dirs(params, cache, du)
    return L, grads, u


def phase_fit():
    rng = np.random.default_rng(0)
    D = 2 * cfg.n_freqs + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)
    uE_t = curved_targets(100)
    uP_t = curved_targets(101)
    zt = uE_t.transpose(1, 0, 2).reshape(tg.size, -1)
    print("[c7 fit] target ensemble PR (E) = %.2f" % PR_of(zt))

    # ---- finite-difference check of the supervised gradient path ----
    L0, g0, _ = fit_loss_and_grads(pE, uE_t)
    rng2 = np.random.default_rng(7)
    max_rel = 0.0
    for key in ("B", "b_body", "W"):
        arr = pE[key]
        for _ in range(3):
            idx = tuple(rng2.integers(0, s) for s in arr.shape)
            h = 1e-6
            arr[idx] += h; Lp, _, _ = fit_loss_and_grads(pE, uE_t)
            arr[idx] -= 2 * h; Lm, _, _ = fit_loss_and_grads(pE, uE_t)
            arr[idx] += h
            fd = (Lp - Lm) / (2 * h)
            an = g0[key][idx]
            rel = abs(fd - an) / max(abs(fd), abs(an), 1e-12)
            max_rel = max(max_rel, rel)
    print("[c7 fit] FD gradient check: max rel err = %.2e" % max_rel)
    assert max_rel < 1e-6, "FD check failed"

    optE, optP = Adam(pE, lr=3e-3), Adam(pP, lr=3e-3)
    t0 = time.time()
    it = 0
    while time.time() - t0 < TRAIN_SECONDS and it < 20000:
        LE, gE, _ = fit_loss_and_grads(pE, uE_t)
        optE.step(pE, gE, sign=-1.0)
        LP, gP, _ = fit_loss_and_grads(pP, uP_t)
        optP.step(pP, gP, sign=-1.0)
        it += 1
        if it % 2000 == 0:
            print("[c7 fit] it %d  L_E=%.5f  L_P=%.5f" % (it, LE, LP))
    # achieved ensembles
    for tag, p in (("E", pE), ("P", pP)):
        u, cache = forward_dirs(p, phig)
        z = np.einsum("hab,tb->tha", p["W"], cache["H"])
        print("[c7 fit] %s: fit MSE=%.5f  PR(Z)=%.2f  PR(H)=%.2f  PR(u)=%.2f" %
              (tag, float(((u - (uE_t if tag == "E" else uP_t)) ** 2).mean()),
               PR_of(z.reshape(tg.size, -1)), PR_of(cache["H"]),
               PR_of(u.transpose(1, 0, 2).reshape(tg.size, -1))))
    np.savez(FIT_CKPT, BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
             BP=pP["B"], bP=pP["b_body"], WP=pP["W"], x_E0=x_E0,
             iter0=it)
    print("[c7 fit] saved after %d iters" % it)


def measure(pE, pP):
    _, cE = forward_dirs(pE, phig); _, cP = forward_dirs(pP, phig)
    zE = np.einsum("hab,tb->tha", pE["W"], cE["H"])
    zP = np.einsum("hab,tb->tha", pP["W"], cP["H"])
    return (PR_of(zE.reshape(tg.size, -1)), PR_of(zP.reshape(tg.size, -1)),
            PR_of(cE["H"]))


def phase_game(max_iters=14000):
    z = np.load(GAME_CKPT if os.path.exists(GAME_CKPT) else FIT_CKPT)
    pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
    pP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
    it0 = int(z["iter0"]) if os.path.exists(GAME_CKPT) else 0
    hist = json.loads(str(z["hist"])) if "hist" in z.files else \
        {"iter": [], "PR_ZE": [], "PR_ZP": [], "PR_HE": [], "no_cap": []}
    if it0 >= max_iters:
        print("[c7 game] DONE at %d" % it0); return
    optE, optP = Adam(pE, lr=1e-3), Adam(pP, lr=1e-2)

    def step(who):
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

    t0 = time.time(); it = it0
    while it < max_iters and time.time() - t0 < TRAIN_SECONDS:
        for _ in range(3):
            step("P")
        L_E, L_P, nc = step("E")
        it += 1
        if it % 250 == 0 or it == 1:
            pze, pzp, phe = measure(pE, pP)
            hist["iter"].append(it); hist["PR_ZE"].append(pze)
            hist["PR_ZP"].append(pzp); hist["PR_HE"].append(phe)
            hist["no_cap"].append(nc)
            if it % 1000 == 0 or it == 1:
                print("[c7 game] it %d  PR_ZE=%.2f PR_ZP=%.2f PR_HE=%.2f "
                      "no_cap=%.2f L_P=%.2f" % (it, pze, pzp, phe, nc, L_P))
    np.savez(GAME_CKPT, BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
             BP=pP["B"], bP=pP["b_body"], WP=pP["W"], x_E0=x_E0,
             iter0=it, hist=json.dumps(hist))
    print("[c7 game] saved at iter %d (%.0fs)" % (it, time.time() - t0))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        phase_fit()
    else:
        phase_game(int(sys.argv[2]) if len(sys.argv) > 2 else 14000)
