"""
v5 long honest training.

Increases the honest critic's capacity (hidden = 160, n_freqs = 10),
adds a cosine LR schedule, and runs many chained 45 s calls to drive
the loss below ~10^{-3} where physically possible.  Also collects a
detailed loss-vs-iteration history for plotting.

Honesty is preserved: NO V* values used as supervision -- only PDE
residual + terminal BC + capture-set BC.

Phases:
  L0   Train fresh large critic (warm start).
  L    Resume training (call repeatedly).
  P    Make loss-curve and trajectory plots from the saved history.
"""
from __future__ import annotations
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

import pursuit_evasion_numpy as pen
import pursuit_evasion_v4 as v4
from pursuit_evasion_numpy import (
    TrainConfig, fourier_features, forward_dirs, rollout, sample_evader_ICs,
)

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
V5_CKPT = OUT + "v5_honestV.npz"
V1_ACTOR_CKPT = OUT + "v2_actor.npz"


# =============================================================================
# Larger honest critic config
# =============================================================================
def make_cfg(n_iters: int) -> v4.HonestVConfig:
    return v4.HonestVConfig(
        L=2.0, v_E=1.0, v_P=1.2, eps=0.08,
        n_dim=2, n_freqs=8, hidden=128,
        t_max=8.0,
        n_iters=n_iters,
        n_res=384, n_term=96, n_cap=96,
        lr=3e-3,
        lam_bc_term=5.0, lam_bc_cap=5.0,
        seed=0, log_every=200,
    )


# =============================================================================
# Train with cosine LR schedule (kept inside this script so we don't
# disturb the v4 training loop)
# =============================================================================
def train_with_cosine_schedule(cfg: v4.HonestVConfig,
                                ckpt_path: str,
                                resume: bool,
                                lr_min: float = 5e-5,
                                total_iters_for_schedule: int = 50000):
    rng = np.random.default_rng(cfg.seed)
    D = v4.encoded_dim(cfg.n_dim, cfg.n_freqs)
    params = v4.init_VT_params(D, cfg.hidden, rng)
    opt = pen.Adam(params, lr=cfg.lr)
    hist = {"iter": [], "loss": [], "loss_hji": [], "loss_term": [],
            "loss_cap": [], "mean_grad_norm": [], "mean_dV_dt": [], "lr": []}
    iter0 = 0
    if resume and os.path.exists(ckpt_path):
        npz = np.load(ckpt_path)
        for k in params:
            params[k] = npz[k]
        opt = pen.Adam(params, lr=cfg.lr)
        iter0 = int(npz["iter0"])
        if "hist" in npz.files:
            hist = json.loads(str(npz["hist"]))
        # Use a different RNG stream so the resumed run sees fresh batches
        rng = np.random.default_rng(int(cfg.seed + iter0))
        print(f"[v5] resumed at iter {iter0}")

    target_norm = 1.0 / (cfg.v_P - cfg.v_E)
    t0 = time.time()
    for it in range(cfg.n_iters):
        global_it = iter0 + it
        # cosine schedule
        progress = min(global_it / total_iters_for_schedule, 1.0)
        lr_now = lr_min + 0.5 * (cfg.lr - lr_min) * (1 + math.cos(math.pi * progress))
        opt.lr = lr_now

        Δr, τr, Δb, τb, Δc, τc = v4.sample_honest_batch(
            rng, cfg.L, cfg.eps, cfg.t_max, cfg.n_dim,
            cfg.n_res, cfg.n_term, cfg.n_cap)
        loss, grads, diag = v4.honest_loss(
            params, Δr, τr, Δb, τb, Δc, τc,
            cfg.L, cfg.v_E, cfg.v_P, cfg.eps,
            cfg.n_freqs, cfg.t_max,
            lam_bc_term=cfg.lam_bc_term, lam_bc_cap=cfg.lam_bc_cap)
        opt.step(params, grads, sign=-1.0)
        if it % cfg.log_every == 0 or it == cfg.n_iters - 1:
            hist["iter"].append(global_it)
            hist["loss"].append(diag["loss"])
            hist["loss_hji"].append(diag["loss_hji"])
            hist["loss_term"].append(diag["loss_term"])
            hist["loss_cap"].append(diag["loss_cap"])
            hist["mean_grad_norm"].append(diag["mean_grad_norm"])
            hist["mean_dV_dt"].append(diag["mean_dV_dt"])
            hist["lr"].append(lr_now)
            print(f"[v5] it {global_it:5d}  loss={diag['loss']:.4e}  "
                  f"hji={diag['loss_hji']:.4e}  term={diag['loss_term']:.4e}  "
                  f"cap={diag['loss_cap']:.4e}  "
                  f"|∇V|≈{diag['mean_grad_norm']:.3f}/{target_norm:.2f}  "
                  f"lr={lr_now:.2e}  [{time.time()-t0:.0f}s]")
    npz_out = {k: v for k, v in params.items()}
    npz_out["iter0"] = iter0 + cfg.n_iters
    npz_out["hist"] = json.dumps(hist)
    np.savez(ckpt_path, **npz_out)
    return params, hist


def phase_L(resume: bool, n_iters: int = 4000):
    cfg = make_cfg(n_iters)
    return train_with_cosine_schedule(cfg, V5_CKPT, resume=resume,
                                       lr_min=5e-5,
                                       total_iters_for_schedule=50000)


# =============================================================================
# Phase P -- detailed loss and trajectory plots
# =============================================================================
def _load_v5_V():
    npz = np.load(V5_CKPT)
    return {k: npz[k] for k in ("B1", "b1", "B2", "b2", "w", "c")}, int(npz["iter0"]), json.loads(str(npz["hist"]))


def _load_actor():
    npz = np.load(V1_ACTOR_CKPT)
    pE = {"B": npz["BE"], "b_body": npz["bE"], "W": npz["WE"]}
    pP = {"B": npz["BP"], "b_body": npz["bP"], "W": npz["WP"]}
    return pE, pP, npz["x_E0"]


def plot_loss_curves(hist: dict, path: str) -> None:
    it = np.array(hist["iter"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(it, hist["loss"], color="black", lw=1.6, label="total")
    ax[0].semilogy(it, hist["loss_hji"], color="tab:red", lw=1.3,
                    label=r"HJI residual")
    ax[0].semilogy(it, hist["loss_term"], color="tab:blue", lw=1.3,
                    label="terminal BC")
    ax[0].semilogy(it, hist["loss_cap"], color="tab:green", lw=1.3,
                    label="capture-set BC")
    ax[0].axhline(1e-3, color="grey", ls="--", lw=0.7, label=r"$10^{-3}$")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss")
    ax[0].set_title("Honest critic loss components")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    target = 1.0 / 0.2
    ax[1].plot(it, hist["mean_grad_norm"], color="tab:purple", lw=1.5,
                label=r"$\langle\|\nabla_\Delta V\|\rangle$")
    ax[1].axhline(target, color="black", ls="--", lw=0.7,
                   label=fr"target $1/(v_P-v_E)={target:.2f}$")
    ax[1].plot(it, hist["lr"], color="tab:orange", lw=1.0, label="lr (right)")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("magnitude")
    ax[1].set_title("Mean spatial gradient norm and learning rate")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_trajectories(out_dir: str, n_show: int = 9) -> dict:
    pE, pP, x_E0 = _load_actor()
    cfg = TrainConfig(
        n_h=12, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
        L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
        no_capture_penalty=2.0, lam_d=5.0, seed=0)
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)
    uE, _ = forward_dirs(pE, phi)
    uP, _ = forward_dirs(pP, phi)
    xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)

    # also do feedback rollouts using v5 critic
    params_V, _, _ = _load_v5_V()
    cfg_V = make_cfg(1)
    dt = cfg.t_max / cfg.n_t
    xE_fb = np.empty_like(xE); xP_fb = np.empty_like(xP)
    xE_fb[:, 0] = x_E0; xP_fb[:, 0] = 0.0
    for j in range(cfg.n_h):
        xe = x_E0[j].copy(); xp = np.zeros(2)
        for k in range(cfg.n_t):
            delta = ((xe - xp) + cfg.L / 2) % cfg.L - cfg.L / 2
            g = v4.grad_V(params_V, delta, cfg_V.t_max, cfg_V)
            u = g / max(np.linalg.norm(g), 1e-12)
            xe = (xe + dt * cfg.v_E * u) % cfg.L
            xp = (xp + dt * cfg.v_P * u) % cfg.L
            xE_fb[j, k + 1] = xe; xP_fb[j, k + 1] = xp

    def _plot_torus_path(ax, traj, L, color, label=None, lw=1.0):
        xs, ys = traj[:, 0], traj[:, 1]
        seg_x, seg_y = [], []; first = True
        for k in range(len(xs)):
            if k > 0 and (abs(xs[k] - xs[k-1]) > L/2 or abs(ys[k] - ys[k-1]) > L/2):
                if seg_x:
                    ax.plot(seg_x, seg_y, color=color, lw=lw,
                            label=label if first else None)
                    first = False; seg_x, seg_y = [], []
            seg_x.append(xs[k]); seg_y.append(ys[k])
        if seg_x:
            ax.plot(seg_x, seg_y, color=color, lw=lw,
                    label=label if first else None)

    # ACTOR trajectories
    cols = 3; rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3.5*rows))
    axes = np.array(axes).reshape(-1)
    captures = []
    for j in range(min(n_show, cfg.n_h)):
        ax = axes[j]
        _plot_torus_path(ax, xE[j], cfg.L, "tab:blue", "Evader", lw=1.2)
        _plot_torus_path(ax, xP[j], cfg.L, "tab:red", "Pursuer", lw=1.2)
        ax.scatter([xE[j, 0, 0]], [xE[j, 0, 1]], s=45, c="tab:blue",
                   marker="o", edgecolor="k", zorder=5)
        ax.scatter([xP[j, 0, 0]], [xP[j, 0, 1]], s=45, c="tab:red",
                   marker="s", edgecolor="k", zorder=5)
        ax.scatter([xE[j, -1, 0]], [xE[j, -1, 1]], s=30, c="tab:blue",
                   marker="x", zorder=5)
        ax.scatter([xP[j, -1, 0]], [xP[j, -1, 1]], s=30, c="tab:red",
                   marker="x", zorder=5)
        diff = (xE[j] - xP[j] + cfg.L/2) % cfg.L - cfg.L/2
        d = np.sqrt((diff**2).sum(-1))
        cap_idx = np.where(d <= cfg.eps)[0]
        Tc = (cap_idx[0]/cfg.n_t)*cfg.t_max if len(cap_idx) else float("nan")
        if not math.isnan(Tc):
            captures.append(Tc)
        ax.set_xlim(0, cfg.L); ax.set_ylim(0, cfg.L); ax.set_aspect("equal")
        ax.set_title(f"IC {j}: $T_c$≈{Tc:.2f}" if not math.isnan(Tc)
                      else f"IC {j}: no capture")
        if j == 0: ax.legend(fontsize=7, loc="upper right")
    for j in range(n_show, len(axes)): axes[j].axis("off")
    fig.suptitle("Adversarial-actor trajectories on $\\mathbb{T}^2$", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir + "fig_traj_actor.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # FEEDBACK trajectories
    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3.5*rows))
    axes = np.array(axes).reshape(-1)
    captures_fb = []
    for j in range(min(n_show, cfg.n_h)):
        ax = axes[j]
        _plot_torus_path(ax, xE_fb[j], cfg.L, "tab:blue", "Evader", lw=1.2)
        _plot_torus_path(ax, xP_fb[j], cfg.L, "tab:red", "Pursuer", lw=1.2)
        ax.scatter([xE_fb[j, 0, 0]], [xE_fb[j, 0, 1]], s=45, c="tab:blue",
                   marker="o", edgecolor="k", zorder=5)
        ax.scatter([xP_fb[j, 0, 0]], [xP_fb[j, 0, 1]], s=45, c="tab:red",
                   marker="s", edgecolor="k", zorder=5)
        ax.scatter([xE_fb[j, -1, 0]], [xE_fb[j, -1, 1]], s=30, c="tab:blue",
                   marker="x", zorder=5)
        ax.scatter([xP_fb[j, -1, 0]], [xP_fb[j, -1, 1]], s=30, c="tab:red",
                   marker="x", zorder=5)
        diff = (xE_fb[j] - xP_fb[j] + cfg.L/2) % cfg.L - cfg.L/2
        d = np.sqrt((diff**2).sum(-1))
        cap_idx = np.where(d <= cfg.eps)[0]
        Tc = (cap_idx[0]/cfg.n_t)*cfg.t_max if len(cap_idx) else float("nan")
        if not math.isnan(Tc): captures_fb.append(Tc)
        ax.set_xlim(0, cfg.L); ax.set_ylim(0, cfg.L); ax.set_aspect("equal")
        ax.set_title(f"IC {j}: $T_c$≈{Tc:.2f}" if not math.isnan(Tc)
                      else f"IC {j}: no capture")
        if j == 0: ax.legend(fontsize=7, loc="upper right")
    for j in range(n_show, len(axes)): axes[j].axis("off")
    fig.suptitle("HJI-feedback trajectories on $\\mathbb{T}^2$  (using v5 critic)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir + "fig_traj_feedback.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "actor_n_captured": len(captures),
        "actor_n_total": min(n_show, cfg.n_h),
        "actor_mean_Tc": float(np.mean(captures)) if captures else None,
        "feedback_n_captured": len(captures_fb),
        "feedback_n_total": min(n_show, cfg.n_h),
        "feedback_mean_Tc": float(np.mean(captures_fb)) if captures_fb else None,
    }
    print(json.dumps(summary, indent=2))
    return summary


def phase_P():
    _, iters, hist = _load_v5_V()
    plot_loss_curves(hist, OUT + "fig_v5_loss_curves.png")
    trajs = plot_trajectories(OUT, n_show=9)
    # update v5 summary
    summary = {
        "n_train_iter": iters,
        "final_loss": hist["loss"][-1],
        "final_loss_hji": hist["loss_hji"][-1],
        "final_loss_term": hist["loss_term"][-1],
        "final_loss_cap": hist["loss_cap"][-1],
        "final_mean_grad_norm": hist["mean_grad_norm"][-1],
        "target_grad_norm": 1.0 / 0.2,
        "min_loss_seen": float(min(hist["loss"])),
        "trajectories": trajs,
    }
    with open(OUT + "v5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


# =============================================================================
# Entry
# =============================================================================
if __name__ == "__main__":
    phases = sys.argv[1:] or ["L0"]
    for ph in phases:
        if ph == "L0":   phase_L(resume=False, n_iters=2500)
        elif ph == "L":  phase_L(resume=True,  n_iters=2500)
        elif ph == "P":  phase_P()
        else: print("unknown:", ph)
