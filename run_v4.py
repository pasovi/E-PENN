"""
v4 driver -- honest pipeline.

Phases:
  V    Train the honest time-augmented critic V(Δ, τ).
       Loss = HJI residual + terminal BC (τ=0) + capture-set BC.
       NO V* values used as supervision.  Run repeatedly to chain.
  Vb   Same, resuming from the v4 checkpoint.
  A    The v1 ADVERSARIAL ACTOR is reused as-is.  We do not retrain it,
       since the v1 training was already an honest self-play.
  W    Run the verification: alignment, capture-time maps, V comparison.

Usage:
  python3 run_v4.py V          # first 3000 iters of honest V
  python3 run_v4.py Vb         # resume +3000
  python3 run_v4.py W          # verification figures + summary
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

import pursuit_evasion_numpy as pen
import pursuit_evasion_v4 as v4
from pursuit_evasion_numpy import (
    TrainConfig, fourier_features, forward_dirs, rollout,
)

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
V_CKPT = OUT + "v4_honestV.npz"

# v1 actor checkpoint (from the original adversarial training)
V1_ACTOR_CKPT = OUT + "v2_actor.npz"   # this is the v1-style actor


# =============================================================================
# Phase V / Vb -- honest critic training
# =============================================================================
def phase_V(resume: bool = False, n_iters: int = 3000):
    cfg = v4.HonestVConfig(n_iters=n_iters, log_every=200,
                            hidden=96, n_freqs=6,
                            n_res=384, n_term=96, n_cap=96,
                            lam_bc_term=5.0, lam_bc_cap=5.0,
                            t_max=8.0, lr=3e-3)
    return v4.train_honest_V(cfg, V_CKPT, resume=resume)


# =============================================================================
# Phase W -- verification (after-the-fact, V* used only as yardstick)
# =============================================================================
def _load_V_params():
    npz = np.load(V_CKPT)
    return {k: npz[k] for k in ("B1", "b1", "B2", "b2", "w", "c")}, int(npz["iter0"])


def _load_actor():
    npz = np.load(V1_ACTOR_CKPT)
    pE = {"B": npz["BE"], "b_body": npz["bE"], "W": npz["WE"]}
    pP = {"B": npz["BP"], "b_body": npz["bP"], "W": npz["WP"]}
    return pE, pP, npz["x_E0"]


def phase_W():
    cfg_V = v4.HonestVConfig(hidden=96, n_freqs=6, t_max=8.0)
    params_V, train_iters = _load_V_params()

    # 1. V_NN vs V* visual comparison (V* is verification-only)
    v4.plot_V_vs_analytic(params_V, cfg_V, OUT + "fig_v4_V_vs_analytic.png",
                          tau_use=cfg_V.t_max, n_grid=80)

    # 2. mean gradient norm  -- should approach 1/(v_P - v_E) outside capture
    rng = np.random.default_rng(1)
    Δ = rng.uniform(-cfg_V.L / 2 + 0.05, cfg_V.L / 2 - 0.05,
                     size=(2000, cfg_V.n_dim))
    Δ = Δ[v4.torus_dist0(Δ, cfg_V.L) > 1.5 * cfg_V.eps][:1500]
    tau_arr = np.full(Δ.shape[0], cfg_V.t_max)
    phi = v4.encode_inputs(Δ, tau_arr, cfg_V.L, cfg_V.n_freqs, cfg_V.t_max)
    # estimate |∇V| at each Δ
    gnorms = []
    for i in range(len(Δ)):
        g = v4.grad_V(params_V, Δ[i], cfg_V.t_max, cfg_V)
        gnorms.append(np.linalg.norm(g))
    gnorms = np.array(gnorms)
    target_norm = 1.0 / (cfg_V.v_P - cfg_V.v_E)
    grad_stat = {
        "mean": float(gnorms.mean()),
        "median": float(np.median(gnorms)),
        "std": float(gnorms.std()),
        "target": float(target_norm),
    }

    # 3. capture-time map using the HONEST feedback policy
    cfg_actor = TrainConfig(
        n_h=12, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
        L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
        no_capture_penalty=2.0, lam_d=5.0, seed=0)
    xs, ys, Tc = v4.capture_time_map_honest(params_V, cfg_actor, cfg_V,
                                              n_grid=24, tau_use=cfg_V.t_max)
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    Tplot = np.where(np.isnan(Tc), cfg_actor.t_max + 0.1, Tc)
    im = ax.imshow(Tplot.T, origin="lower",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]],
                   cmap="plasma", vmin=0, vmax=cfg_actor.t_max)
    plt.colorbar(im, ax=ax, label=r"$T_c$  (NaN -> no capture)")
    ax.scatter([0.0], [0.0], s=70, c="cyan", marker="s",
               edgecolor="k", label="pursuer")
    ax.set_xlabel(r"$x_E(0)_x$"); ax.set_ylabel(r"$x_E(0)_y$")
    ax.set_title("Capture time -- honestly-trained feedback")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT + "fig_v4_capture_honest_feedback.png", dpi=140,
                bbox_inches="tight"); plt.close(fig)
    honest_cap = float((~np.isnan(Tc)).mean())

    # 4. compare v1 actor vs honest critic (honest verification)
    pE, pP, x_E0 = _load_actor()
    # v1 actor has n_h=12 — load TrainConfig accordingly
    cfg_a = TrainConfig(
        n_h=12, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
        L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
        no_capture_penalty=2.0, lam_d=5.0, seed=0)
    align = v4.alignment_actor_vs_honest_critic(
        pE, pP, x_E0, cfg_a, params_V, cfg_V, tau_use=cfg_V.t_max)
    t_grid = np.linspace(0.0, cfg_a.t_max, cfg_a.n_t + 1)
    v4.plot_honest_alignment(align, t_grid,
                              OUT + "fig_v4_alignment_honest.png")

    # 5. summary
    summary = {
        "n_train_iter_honest_V": int(train_iters),
        "honest_V_grad_norm": grad_stat,
        "honest_feedback_capture_fraction": honest_cap,
        "actor_vs_honest_critic_alignment_mean": float(align.mean()),
        "v1_adversarial_actor_train_iters": int(np.load(V1_ACTOR_CKPT)["iter0"]),
        "comment_v2_supervised_loss": "RETRACTED (was supervised on V*)",
        "comment_v3_distillation_loss": "RETRACTED (was supervised on u*=Delta/|Delta|)",
    }
    print("\nHONEST SUMMARY:")
    print(json.dumps(summary, indent=2))
    with open(OUT + "v4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# =============================================================================
# Entry
# =============================================================================
if __name__ == "__main__":
    phases = sys.argv[1:] or ["V"]
    for ph in phases:
        if ph == "V":   phase_V(resume=False)
        elif ph == "Vb": phase_V(resume=True)
        elif ph == "W":  phase_W()
        else: print("unknown:", ph)
