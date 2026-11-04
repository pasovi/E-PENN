"""Driver for the v3 open questions.

Phases (each ≤ 45 s; use a/b/c suffixes to resume from checkpoints):

  D1   Train the distilled actor (warm-up + ~2500 distilled iters)
  D1b  Resume distilled actor training (no warm-up)
  D2   Train the obstacle actor (warm-up + ~2000 iters)
  D2b  Resume obstacle actor training
  N4   N_h scaling — train one head count at a time (Nh4, Nh8, Nh16, Nh32)
  V    Make all verification figures and dump v3_summary.json
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

import pursuit_evasion_numpy as pen
import pursuit_evasion_v2 as v2
import pursuit_evasion_v3 as v3
from pursuit_evasion_numpy import (
    TrainConfig, fourier_features, forward_dirs, rollout, sample_evader_ICs,
)

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
DISTILL_CKPT = OUT + "v3_distill.npz"
OBSTACLE_CKPT = OUT + "v3_obstacle.npz"
NH_PREFIX = OUT + "v3_nh"

# -------- helpers --------
def _to_train_config(d: v3.DistillConfig) -> TrainConfig:
    return TrainConfig(
        n_iters=0, log_every=200, n_h=d.n_h, n_b=d.n_b, n_freqs=d.n_freqs,
        t_max=d.t_max, n_t=d.n_t, v_E=d.v_E, v_P=d.v_P, eps=d.eps,
        tau=d.tau, L=d.L, lr_E=d.lr_E, lr_P=d.lr_P,
        no_capture_penalty=d.no_capture_penalty, lam_d=d.lam_d,
        n_P_steps=d.n_P_steps, n_E_steps=d.n_E_steps,
        pretrain_P_iters=d.pretrain_P_iters, seed=d.seed,
    )


# =============================================================================
# Phase D1 (and D1b): distilled actor
# =============================================================================
def phase_D1(resume: bool = False, n_iters: int = 2500):
    cfg = v3.DistillConfig(lam_dist_P=5.0, lam_dist_E=0.5)
    return v3.train_distill(cfg, n_iters=n_iters, ckpt_path=DISTILL_CKPT,
                            log_every=200, resume=resume)


# =============================================================================
# Phase D2 (and D2b): obstacle actor
# =============================================================================
def phase_D2(resume: bool = False, n_iters: int = 2000):
    cfg = v3.DistillConfig()
    return v3.train_obstacle(cfg, n_iters=n_iters, ckpt_path=OBSTACLE_CKPT,
                              obs_center=np.array([1.0, 1.0]),
                              obs_sigma=0.25, obs_strength=8.0,
                              log_every=200, resume=resume)


# =============================================================================
# Phase N4: train one N_h at a time (call once per N_h)
# =============================================================================
def phase_Nh(n_h: int, n_iters: int = 1500):
    cfg = v3.DistillConfig(n_h=n_h, n_b=max(16, n_h), pretrain_P_iters=900)
    ckpt = f"{NH_PREFIX}{n_h}.npz"
    pE, pP, x0, hist = v3.train_distill(cfg, n_iters=n_iters,
                                         ckpt_path=ckpt, log_every=300,
                                         resume=False)
    # measure final alignment
    tc = _to_train_config(cfg)
    align = v3.mean_pursuer_alignment(pE, pP, x0, tc)
    cap_frac = v3.capture_fraction_on_grid(pE, pP, x0, tc, n_grid=16)
    print(f"[Nh={n_h}] mean alignment = {align:.3f}   capture frac = {cap_frac:.3f}")
    return {"n_h": n_h, "align": align, "cap_frac": cap_frac}


# =============================================================================
# Phase V: produce figures and summary
# =============================================================================
def _load(ckpt):
    npz = np.load(ckpt)
    pE = {"B": npz["BE"], "b_body": npz["bE"], "W": npz["WE"]}
    pP = {"B": npz["BP"], "b_body": npz["bP"], "W": npz["WP"]}
    return pE, pP, npz["x_E0"], npz


def phase_V():
    # -- (a) distilled actor: alignment + capture-time map -------------------
    cfg_dist = v3.DistillConfig(lam_dist_P=5.0, lam_dist_E=0.5)
    tc_dist = _to_train_config(cfg_dist)
    pE_d, pP_d, x0_d, npz_d = _load(DISTILL_CKPT)
    v2.plot_alignment_check(pE_d, pP_d, x0_d, tc_dist,
                             OUT + "fig_v3_align_distill.png")
    xs, ys, Tc = v2.capture_time_map(pE_d, pP_d, tc_dist, n_grid=24)
    v2.plot_capture_time_map(xs, ys, Tc, tc_dist,
                              OUT + "fig_v3_capture_distill.png",
                              title="Distilled multi-head actor")
    cap_distill = float((~np.isnan(Tc)).mean())

    # -- (b) v2 baseline (open-loop, no distillation), for comparison -------
    npz_v2 = np.load(OUT + "v2_actor.npz")
    pE_v2 = {"B": npz_v2["BE"], "b_body": npz_v2["bE"], "W": npz_v2["WE"]}
    pP_v2 = {"B": npz_v2["BP"], "b_body": npz_v2["bP"], "W": npz_v2["WP"]}
    x0_v2 = npz_v2["x_E0"]
    tc_v2 = TrainConfig(n_h=12, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
                        L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
                        no_capture_penalty=2.0, lam_d=5.0, seed=0)
    xs2, ys2, Tc2 = v2.capture_time_map(pE_v2, pP_v2, tc_v2, n_grid=24)
    cap_baseline = float((~np.isnan(Tc2)).mean())

    # -- (c) obstacle actor: capture-time map ------------------------------
    cap_obs = None
    if os.path.exists(OBSTACLE_CKPT):
        pE_o, pP_o, x0_o, npz_o = _load(OBSTACLE_CKPT)
        cfg_obs = v3.DistillConfig()
        tc_obs = _to_train_config(cfg_obs)
        xs_o, ys_o, Tc_o = v2.capture_time_map(pE_o, pP_o, tc_obs, n_grid=24)
        cap_obs = float((~np.isnan(Tc_o)).mean())

        # Plot the obstacle setup
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        no_cap = np.isnan(Tc_o)
        Tplot = np.where(no_cap, cfg_obs.t_max + 0.1, Tc_o)
        im = ax.imshow(Tplot.T, origin="lower",
                       extent=[xs_o[0], xs_o[-1], ys_o[0], ys_o[-1]],
                       cmap="plasma", vmin=0, vmax=cfg_obs.t_max)
        plt.colorbar(im, ax=ax, label=r"$T_c$  (NaN $\to$ no capture)")
        obs_center = npz_o["obs_center"]; obs_sigma = float(npz_o["obs_sigma"])
        circ = plt.Circle((obs_center[0], obs_center[1]), 2 * obs_sigma,
                          color="white", fill=False, lw=2, ls="--",
                          label="obstacle ($2\\sigma$)")
        ax.add_patch(circ)
        ax.scatter([0.0], [0.0], s=70, c="cyan", marker="s",
                   edgecolor="k", label="pursuer $x_P(0)$")
        ax.set_xlabel(r"$x_E(0)_x$"); ax.set_ylabel(r"$x_E(0)_y$")
        ax.set_title("Capture time with Gaussian obstacle")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT + "fig_v3_capture_obstacle.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

        # Plot trajectories for the obstacle setting
        v2.plot_capture_time_map(xs_o, ys_o, Tc_o, tc_obs,
                                  OUT + "fig_v3_obstacle_Tc_plain.png",
                                  title="Obstacle scenario")

    # -- (d) Nh scaling --------------------------------------------------
    nh_results = []
    for n_h in (4, 8, 16, 32):
        ck = f"{NH_PREFIX}{n_h}.npz"
        if not os.path.exists(ck):
            continue
        pE, pP, x0, _ = _load(ck)
        cfg = v3.DistillConfig(n_h=n_h, n_b=max(16, n_h), pretrain_P_iters=0)
        tc = _to_train_config(cfg)
        align = v3.mean_pursuer_alignment(pE, pP, x0, tc)
        cap = v3.capture_fraction_on_grid(pE, pP, x0, tc, n_grid=16)
        nh_results.append({"n_h": n_h, "align": align, "cap_frac": cap})
        print(f"[Nh={n_h:3d}] align={align:.3f} cap_frac={cap:.3f}")

    if nh_results:
        nhs = [r["n_h"] for r in nh_results]
        als = [r["align"] for r in nh_results]
        cfs = [r["cap_frac"] for r in nh_results]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(nhs, als, "o-", color="tab:blue")
        ax[0].set_xscale("log", base=2)
        ax[0].set_xlabel(r"$N_h$"); ax[0].set_ylabel("mean alignment")
        ax[0].set_title(r"Pursuer alignment vs $N_h$ (1500 iter, distilled)")
        ax[0].axhline(1.0, color="k", ls="--", lw=0.6)
        ax[0].set_ylim(0, 1.05)
        ax[1].plot(nhs, cfs, "s-", color="tab:red")
        ax[1].set_xscale("log", base=2)
        ax[1].set_xlabel(r"$N_h$"); ax[1].set_ylabel("capture fraction")
        ax[1].set_title(r"Test-grid capture fraction vs $N_h$")
        ax[1].set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(OUT + "fig_v3_Nh_scaling.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    summary = {
        "baseline_v2_capture_fraction": cap_baseline,
        "distilled_capture_fraction": cap_distill,
        "v2_feedback_capture_fraction": 0.582,  # from v2_summary.json
        "obstacle_capture_fraction": cap_obs,
        "Nh_scaling": nh_results,
        "n_train_iter_distill": int(npz_d["iter0"]),
    }
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))
    with open(OUT + "v3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# =============================================================================
# Entry
# =============================================================================
if __name__ == "__main__":
    phases = sys.argv[1:] or ["D1", "D2", "Nh4", "Nh8", "Nh16", "Nh32", "V"]
    for ph in phases:
        if ph == "D1":   phase_D1(resume=False)
        elif ph == "D1b": phase_D1(resume=True)
        elif ph == "D2":  phase_D2(resume=False)
        elif ph == "D2b": phase_D2(resume=True)
        elif ph.startswith("Nh"):
            n_h = int(ph[2:])
            phase_Nh(n_h)
        elif ph == "V":   phase_V()
        else: print("unknown:", ph)
