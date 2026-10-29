"""Animate how the baseline's trajectories evolve during training.

Retrains the published baseline configuration (init seed 0, IC seed 0, the protocol of
rev2/core.py) and stores a parameter snapshot every `--every` gradient steps of the
pursuer warm-up and every `--every` self-play iterations.  Then renders one frame per
snapshot: left, the lifted paths on the universal cover R^2; right, the embedded torus;
one colour per head as in visualize_actor.py.

Outputs (under checks/figs/<name>/):
  training_snapshots.npz   all snapshots (BE, bE, WE, BP, bP, WP stacked, with labels)
  training_evolution.gif   the animation
Usage: python3 torch_actor/checks/animate_training.py [--every 100] [--fps 8] [--heads3d 1,4,11,0,2,3]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import pe_torch as pt  # noqa: E402
from core import RunConfig  # noqa: E402
from visualize_actor import simulate, head_color  # noqa: E402


def train_with_snapshots(cfg: pt.Config, every: int, device="cpu", dtype=torch.float64):
    states = [pt.initial_state_np(cfg, 0, 0)]
    pE, pP, x0 = pt.stack_runs(states, device, dtype)
    t = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=device, dtype=dtype)
    phi = pt.fourier_features(t, cfg.t_max, cfg.K)
    optE, optP = pt.make_adam(pE, cfg.lr_E), pt.make_adam(pP, cfg.lr_P)
    snaps, labels = [], []

    def snap(label):
        snaps.append((pt.to_numpy(pE), pt.to_numpy(pP))); labels.append(label)

    snap("init")
    for k in range(1, cfg.warmup + 1):
        pt.step("P", pE, pP, x0, phi, cfg, optE, optP)
        if k % every == 0:
            snap(f"warm-up step {k}")
    for it in range(1, cfg.n_iters + 1):
        if cfg.adam_reset_every and it > 1 and (it - 1) % cfg.adam_reset_every == 0:
            optE, optP = pt.make_adam(pE, cfg.lr_E), pt.make_adam(pP, cfg.lr_P)
        for _ in range(cfg.n_P_steps):
            pt.step("P", pE, pP, x0, phi, cfg, optE, optP)
        for _ in range(cfg.n_E_steps):
            pt.step("E", pE, pP, x0, phi, cfg, optE, optP)
        if it % every == 0:
            snap(f"iteration {it}")
    return snaps, labels, x0.cpu().numpy()[0]


def legacy(p, r=0):
    return {"B": p["B"][r], "b_body": p["b"][r], "W": p["W"][r]}


def emb(x, L, R0=1.6, r=0.6):
    a = 2 * np.pi * x[..., 0] / L
    b = 2 * np.pi * x[..., 1] / L
    return np.stack([(R0 + r * np.cos(b)) * np.cos(a), (R0 + r * np.cos(b)) * np.sin(a), r * np.sin(b)], -1)


def render(snaps, labels, x0, cfg_r: RunConfig, heads3d, out_gif, fps):
    n_h = x0.shape[0]
    sims = [simulate(legacy(pE), legacy(pP), x0, cfg_r) for pE, pP in snaps]
    L = cfg_r.L
    fig = plt.figure(figsize=(15, 7.2))
    axL = fig.add_subplot(1, 2, 1)
    axR = fig.add_subplot(1, 2, 2, projection="3d")

    # static scenery, left
    for m in range(-2, 3):
        for n in range(-2, 3):
            axL.scatter(m * L, n * L, s=14, c="0.6", zorder=1)
            axL.add_patch(plt.Rectangle((m * L, n * L), L, L, fill=False, ec="0.85", lw=0.6))
    th = np.linspace(0, 2 * np.pi, 200)
    axL.plot(cfg_r.reach * np.cos(th), cfg_r.reach * np.sin(th), color="0.5", ls=":", lw=1)
    axL.set_aspect("equal"); axL.set_xlim(-2.2 * L, 2.2 * L); axL.set_ylim(-2.2 * L, 2.2 * L)
    axL.set_xlabel("x"); axL.set_ylabel("y")
    axL.set_title("Lifted paths on R² (solid = evader, dashed = pursuer, ○ = capture)", fontsize=10)
    linesE, linesP, caps = [], [], []
    for j in range(n_h):
        c = head_color(j)
        (lE,) = axL.plot([], [], color=c, lw=1.4, label=f"head {j}")
        (lP,) = axL.plot([], [], color=c, lw=1.0, ls="--")
        axL.scatter(*x0[j], s=30, color=c, edgecolor="k", zorder=5)
        axL.annotate(str(j), x0[j], fontsize=7, xytext=(3, 3), textcoords="offset points")
        cp = axL.scatter([], [], s=70, facecolors="none", edgecolors=c, lw=1.5)
        linesE.append(lE); linesP.append(lP); caps.append(cp)
    axL.legend(fontsize=6.5, loc="upper right", ncol=2, framealpha=0.9, title="colour = head")

    # static scenery, right
    A, B = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(0, 2 * np.pi, 30))
    R0, r = 1.6, 0.6
    axR.plot_surface((R0 + r * np.cos(B)) * np.cos(A), (R0 + r * np.cos(B)) * np.sin(A), r * np.sin(B),
                     color="0.9", alpha=0.25, linewidth=0, shade=True)
    axR.plot_wireframe((R0 + r * np.cos(B)) * np.cos(A), (R0 + r * np.cos(B)) * np.sin(A), r * np.sin(B),
                       color="0.7", lw=0.3, rstride=3, cstride=6, alpha=0.5)
    axR.scatter(*emb(np.zeros(2), L), color="tab:red", marker="s", s=60, edgecolor="k", label="pursuer start")
    lines3E, lines3P = {}, {}
    for j in heads3d:
        c = head_color(j)
        (lE,) = axR.plot([], [], [], color=c, lw=2, label=f"head {j}")
        (lP,) = axR.plot([], [], [], color=c, lw=1.4, ls="--")
        e0 = emb(x0[j], L)
        axR.scatter(*e0, color=c, s=40, edgecolor="k")
        lines3E[j] = lE; lines3P[j] = lP
    axR.set_box_aspect((1, 1, 0.4)); axR.set_axis_off()
    axR.legend(fontsize=8, loc="upper left")
    axR.set_title("Embedded torus (same colours; a few heads)", fontsize=10)
    sup = fig.suptitle("", fontsize=12)

    def update(i):
        S = sims[i]
        for j in range(n_h):
            linesE[j].set_data(S["xE_l"][j, :, 0], S["xE_l"][j, :, 1])
            linesP[j].set_data(S["xP_l"][j, :, 0], S["xP_l"][j, :, 1])
            if S["captured"][j]:
                caps[j].set_offsets(S["xP_l"][j, S["kc"][j]][None, :])
            else:
                caps[j].set_offsets(np.empty((0, 2)))
        for j in heads3d:
            eE = emb(S["xE"][j], L); eP = emb(S["xP"][j], L)
            lines3E[j].set_data(eE[:, 0], eE[:, 1]); lines3E[j].set_3d_properties(eE[:, 2])
            lines3P[j].set_data(eP[:, 0], eP[:, 1]); lines3P[j].set_3d_properties(eP[:, 2])
        n_cap = int(S["captured"].sum()); n_capable = int(S["capturable"].sum())
        n_cap_c = int((S["captured"] & S["capturable"]).sum())
        sup.set_text(f"Baseline training (init seed 0, IC seed 0): {labels[i]}   |   "
                     f"captured {n_cap} of {n_h} heads ({n_cap_c} of the {n_capable} capturable)   |   "
                     f"frame {i + 1}/{len(sims)}")
        return linesE + linesP + caps + list(lines3E.values()) + list(lines3P.values()) + [sup]

    anim = FuncAnimation(fig, update, frames=len(sims), blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--heads3d", default="1,4,11,14,0,2")
    ap.add_argument("--name", default="baseline")
    a = ap.parse_args()
    out = os.path.join(HERE, "figs", a.name)
    os.makedirs(out, exist_ok=True)
    cfg = pt.Config()
    t0 = time.time()
    snaps, labels, x0 = train_with_snapshots(cfg, a.every)
    print(f"trained with {len(snaps)} snapshots in {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(os.path.join(out, "training_snapshots.npz"),
                        labels=np.array(labels), x_E0=x0,
                        BE=np.stack([s[0]["B"][0] for s in snaps]), bE=np.stack([s[0]["b"][0] for s in snaps]),
                        WE=np.stack([s[0]["W"][0] for s in snaps]), BP=np.stack([s[1]["B"][0] for s in snaps]),
                        bP=np.stack([s[1]["b"][0] for s in snaps]), WP=np.stack([s[1]["W"][0] for s in snaps]))
    t0 = time.time()
    heads3d = [int(h) for h in a.heads3d.split(",")]
    gif = os.path.join(out, "training_evolution.gif")
    render(snaps, labels, x0, RunConfig(), heads3d, gif, a.fps)
    print(f"rendered {len(snaps)} frames to {gif} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
