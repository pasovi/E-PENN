"""Pilot: the simplest exactly zero-sum open-loop game, same architecture.

Loss (single term, exactly zero-sum, no capture mechanics at all):
    J = mean_j  <d_T(x_E^j(t), x_P^j(t))>_t        (integrated torus distance)
    L_P = +J  (pursuer minimises),   L_E = -J  (evader maximises)
or, with --terminal, J = mean_j d_T(t_max) (terminal distance).

Protocol: no warm-up, 1:1 alternating steps, equal Adam learning rates,
one Adam state for the whole run.  Same body/heads/Fourier features and the
same evader IC set (seed 0) as the published baseline.

Analytic open-loop saddle when no head is reached within the horizon
(|Delta0| > v_P t_max): both players move at constant heading along the
min-image direction Delta0-hat, value d0 - (v_P - v_E) t_max.  So the
strategy ensemble should be the constant-heading family (Gram PR <= 2,
constancy eta = 1), with nothing left over to argue about.

Optional latent regulariser (--lam-pr LAM > 0): each player's loss gains
    R_a = LAM * (n_b - PR(<H_a H_a^T>_t))         (pursuit_evasion_numpy.pr_latent_reg)
which pushes the latent second moment towards isotropy (PR(H)_unc -> n_b = 16)
so one can ask whether the Gram PR <= 2 reduction of the heading ensemble
survives when the latent is *not* allowed to collapse.  --pr-centred uses
Cov_t[H] instead (leaves the time-mean free; see the docstring of
pr_latent_reg for why that is a loophole).

Usage: python3 viz/minimal_game_pilot.py [--tmax 1.5] [--iters 3000]
                                         [--terminal] [--seed 0] [--lr 3e-3]
                                         [--lam-pr 0] [--pr-centred] [--pr-target n_b]
                                         [--pr-form linear|hinge2]
Writes viz/pilot_<tag>/{pilot.json, fig_torus_trajectories.png, fig_headings.png}
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
sys.path.insert(0, os.path.join(ROOT, "viz"))
from pursuit_evasion_numpy import (fourier_features, forward_dirs, backward_dirs,  # noqa: E402
                                   rollout, backward_rollout, init_params,
                                   sample_evader_ICs, Adam, pr_latent_reg)
from core import RunConfig  # noqa: E402
import metrics as M  # noqa: E402
import visualize_actor as V  # noqa: E402


def dist_loss(xE, xP, L, terminal):
    disp = (xE - xP + L / 2.0) % L - L / 2.0
    d = np.sqrt((disp ** 2).sum(-1) + 1e-12)                 # (n_h, T)
    n_h, T = d.shape
    if terminal:
        J = float(d[:, -1].mean())
        w = np.zeros_like(d); w[:, -1] = 1.0 / n_h
    else:
        J = float(d.mean())
        w = np.full_like(d, 1.0 / (n_h * T))
    dJ_ddisp = (w / d)[..., None] * disp                      # dJ/d(disp)
    return J, dJ_ddisp


def player_grads(pE, pP, x0, phi, cfg, terminal, who, reg=None):
    """Player loss, its parameter gradient, and the game value J.

    reg : None or dict(lam=, target=, centred=) -> adds pr_latent_reg on the
          player's own latent H (training grid) to that player's loss.
    """
    uE, cE = forward_dirs(pE, phi); uP, cP = forward_dirs(pP, phi)
    xE, xP = rollout(uE, uP, x0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    J, g = dist_loss(xE, xP, cfg.L, terminal)
    sign = 1.0 if who == "P" else -1.0                        # L_P = J, L_E = -J
    dxE, dxP = sign * g, -sign * g
    duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    params, cache, du = (pP, cP, duP) if who == "P" else (pE, cE, duE)
    R, dH = 0.0, None
    if reg is not None and reg["lam"] > 0.0:
        R, _, dH = pr_latent_reg(cache["H"], reg["lam"], reg["target"], reg["centred"],
                                 reg.get("form", "linear"))
    return sign * J + R, backward_dirs(params, cache, du, dH_extra=dH), J


def fd_check(pE, pP, x0, phi, cfg, terminal, reg=None):
    rng = np.random.default_rng(3)
    worst = 0.0
    for who, params in (("P", pP), ("E", pE)):
        L0, g0, _ = player_grads(pE, pP, x0, phi, cfg, terminal, who, reg)
        for key in ("B", "b_body", "W"):
            for _ in range(2):
                idx = tuple(rng.integers(0, s) for s in params[key].shape)
                h = 1e-6
                params[key][idx] += h; Lp, _, _ = player_grads(pE, pP, x0, phi, cfg, terminal, who, reg)
                params[key][idx] -= 2 * h; Lm, _, _ = player_grads(pE, pP, x0, phi, cfg, terminal, who, reg)
                params[key][idx] += h
                fd = (Lp - Lm) / (2 * h); an = g0[key][idx]
                worst = max(worst, abs(fd - an) / max(abs(fd), abs(an), 1e-3))
    print(f"[pilot] finite-difference gradient check: max rel err = {worst:.2e}")
    assert worst < 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=1.5)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--terminal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam-pr", type=float, default=0.0,
                    help="weight of the latent PR regulariser lam*(target - PR(<HH^T>_t)) per player")
    ap.add_argument("--pr-centred", action="store_true", help="regularise PR of Cov_t[H] instead")
    ap.add_argument("--pr-target", type=float, default=None, help="PR target (default n_b)")
    ap.add_argument("--pr-form", choices=("linear", "hinge2"), default="linear",
                    help="linear: lam*(target-PR); hinge2: lam*max(0, target-PR)^2")
    a = ap.parse_args()
    tag = f"{'term' if a.terminal else 'integ'}_tmax{a.tmax:g}_s{a.seed}"
    if a.lam_pr > 0.0:
        tag += f"_lampr{a.lam_pr:g}" + ("c" if a.pr_centred else "")
        if a.pr_form != "linear":
            tag += f"_{a.pr_form}"
        if a.pr_target is not None:
            tag += f"_tgt{a.pr_target:g}"
    reg = ({"lam": a.lam_pr, "target": a.pr_target, "centred": a.pr_centred, "form": a.pr_form}
           if a.lam_pr > 0.0 else None)
    out = os.path.join(ROOT, "viz", f"pilot_{tag}")
    os.makedirs(out, exist_ok=True)

    cfg = RunConfig(t_max=a.tmax, n_t=max(16, int(round(16 * a.tmax))), init_seed=a.seed)
    rng = np.random.default_rng(a.seed)
    D = 2 * cfg.K + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)
    x0 = sample_evader_ICs(cfg.n_h, cfg.L, cfg.ic_min_dist, cfg.ic_seed)
    tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(tg, cfg.t_max, cfg.K)
    fd_check(pE, pP, x0, phi, cfg, a.terminal, reg)

    d0 = np.linalg.norm(M.min_image(x0, cfg.L), axis=1)
    print(f"[pilot] tag={tag}  min |Delta0|={d0.min():.2f}, v_P t_max={cfg.v_P * cfg.t_max:.2f} "
          f"(no head reached within horizon iff min|Delta0| > v_P t_max: {d0.min() > cfg.v_P * cfg.t_max})")

    optE, optP = Adam(pE, lr=a.lr), Adam(pP, lr=a.lr)
    t0 = time.time(); hist = []
    def snap(it):
        _, phid = M.dense_phi(cfg, 512)
        row = {"it": it}
        for name, p in (("E", pE), ("P", pP)):
            u, z, H = M.curves(p, phid)
            e = M.ensemble_stats(u, z); l = M.latent_stats(H, cfg.K)
            row[name] = {"gram_pr": e["gram_pr"], "constancy": e["constancy"],
                         "tpr_U": e["tpr_U"], "pr_H": l["pr_H"], "pr_H_unc": l["pr_H_unc"]}
        return row
    hist.append(snap(0))
    for it in range(1, a.iters + 1):
        LP, gP, J = player_grads(pE, pP, x0, phi, cfg, a.terminal, "P", reg)
        optP.step(pP, gP, sign=-1.0)
        LE, gE, J = player_grads(pE, pP, x0, phi, cfg, a.terminal, "E", reg)
        optE.step(pE, gE, sign=-1.0)
        if it % 250 == 0 or it == a.iters:
            r = snap(it); r["J"] = J; r["R_P"] = LP - J; r["R_E"] = LE + J; hist.append(r)
            print(f"[pilot] it {it:5d}  J={J:.4f}  gramPR E/P={r['E']['gram_pr']:.2f}/{r['P']['gram_pr']:.2f}  "
                  f"eta E/P={r['E']['constancy']:.3f}/{r['P']['constancy']:.3f}  "
                  f"PR(H) E/P={r['E']['pr_H']:.2f}/{r['P']['pr_H']:.2f}  "
                  f"PR(H)unc E/P={r['E']['pr_H_unc']:.2f}/{r['P']['pr_H_unc']:.2f}  [{time.time()-t0:.0f}s]")

    # analytic saddle value and alignment
    Delta0 = M.min_image(x0, cfg.L)
    sad = M.saddle_reference(x0, cfg)
    _, phid = M.dense_phi(cfg, 512)
    uE, _, _ = M.curves(pE, phid); uP, _, _ = M.curves(pP, phid)
    alignE = M.static_alignment(uE, Delta0); alignP = M.static_alignment(uP, Delta0)
    J_saddle = float(np.mean(np.clip(d0 - cfg.dv * cfg.t_max, 0, None))) if a.terminal else \
        float(np.mean(np.clip(d0[:, None] - cfg.dv * tg[None, :], 0, None).mean(1)))
    res = {"tag": tag, "config": {"t_max": cfg.t_max, "n_t": cfg.n_t, "iters": a.iters,
                                  "lr": a.lr, "terminal": a.terminal, "seed": a.seed,
                                  "lam_pr": a.lam_pr, "pr_centred": a.pr_centred, "pr_form": a.pr_form,
                                  "pr_target": a.pr_target if a.pr_target is not None else cfg.n_b},
           "J_final": hist[-1]["J"], "J_saddle_analytic": J_saddle,
           "saddle_gram_pr": sad, "mean_align_E": float(np.mean(alignE)),
           "mean_align_P": float(np.mean(alignP)), "min_align_E": float(np.min(alignE)),
           "min_align_P": float(np.min(alignP)), "history": hist}
    with open(os.path.join(out, "pilot.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(out, "pilot_actor.npz"), BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
             BP=pP["B"], bP=pP["b_body"], WP=pP["W"], x_E0=x0, iter0=a.iters)
    S = V.simulate(pE, pP, x0, cfg)
    V.fig_torus(S, cfg, os.path.join(out, "fig_torus_trajectories.png"))
    V.fig_headings(S, cfg, pE, pP, os.path.join(out, "fig_headings.png"))
    V.fig_latent(S, cfg, pE, pP, os.path.join(out, "fig_latent.png"))
    print(f"[pilot] J_final={res['J_final']:.4f} vs analytic saddle {J_saddle:.4f}; "
          f"alignment with Delta0-hat: E mean {res['mean_align_E']:.5f} (min {res['min_align_E']:.4f}), "
          f"P mean {res['mean_align_P']:.5f} (min {res['min_align_P']:.4f}); "
          f"saddle Gram PR {sad['gram_pr_saddle']:.2f}")
    print("[pilot] wrote", out)


if __name__ == "__main__":
    main()
