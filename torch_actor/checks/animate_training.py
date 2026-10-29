"""Animate how the baseline's trajectories and losses evolve during training.

Retrains the published baseline configuration (init seed 0, IC seed 0, the protocol of
rev2/core.py), records both players' losses at every gradient step, and stores a
parameter snapshot every `--every` gradient steps of the pursuer warm-up and every
`--every` self-play iterations.  Then renders one frame per snapshot:
  top left     lifted paths on the universal cover R^2, or (--left flat) the flat torus
               [0, L)^2 with periodic boundary conditions
  top right    the embedded torus
  bottom       the full loss history L_P (pursuer) and L_E (evader) against the
               gradient-step count, with a marker at the current snapshot
One colour per head, as in visualize_actor.py.

Outputs (under checks/figs/<name>/):
  training_snapshots.npz    snapshots (BE, bE, WE, BP, bP, WP stacked), labels, step
                            index of each snapshot, and the per-step loss history
  <out>.mp4 (or .gif)       the animation
Usage: python3 torch_actor/checks/animate_training.py [--every 100] [--fps 3]
       [--heads 4,2] (restrict both panels to these heads) [--heads3d 1,4,11,14,0,2]
       [--snapshots <npz>] (reuse saved snapshots instead of retraining)
       [--out training_evolution] [--format mp4|gif]
       [--left lifted|flat]  left panel: lifted paths on R^2, or the flat torus [0,L)^2
                             with periodic boundaries (paths wrap to the opposite side)
mp4 needs an ffmpeg binary: the `imageio-ffmpeg` package provides one.
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
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import pe_torch as pt  # noqa: E402
from core import RunConfig  # noqa: E402
from visualize_actor import simulate, head_color, cut  # noqa: E402


# ---------------------------------------------------------------- training
def step_with_losses(who, pE, pP, x0, phi, cfg, optE, optP):
    """pe_torch.step, returning both players' losses at the pre-update parameters."""
    uE, _, _ = pt.forward_dirs(pE, phi)
    uP, _, _ = pt.forward_dirs(pP, phi)
    if who == "P":
        uE = uE.detach()
    else:
        uP = uP.detach()
    xE, xP = pt.rollout(uE, uP, x0, cfg)
    L_E, L_P, _, _ = pt.game_loss(xE, xP, cfg)
    loss = L_P.sum() if who == "P" else L_E.sum()
    opt = optP if who == "P" else optE
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(L_E.item()), float(L_P.item())


def train_with_snapshots(cfg: pt.Config, every: int, device="cpu", dtype=torch.float64):
    states = [pt.initial_state_np(cfg, 0, 0)]
    pE, pP, x0 = pt.stack_runs(states, device, dtype)
    t = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=device, dtype=dtype)
    phi = pt.fourier_features(t, cfg.t_max, cfg.K)
    optE, optP = pt.make_adam(pE, cfg.lr_E), pt.make_adam(pP, cfg.lr_P)
    snaps, labels, snap_step, hist = [], [], [], []      # hist: (step, L_E, L_P)
    gstep = 0

    def snap(label):
        snaps.append((pt.to_numpy(pE), pt.to_numpy(pP))); labels.append(label); snap_step.append(gstep)

    snap("initialisation")
    for k in range(1, cfg.warmup + 1):
        le, lp = step_with_losses("P", pE, pP, x0, phi, cfg, optE, optP)
        hist.append((gstep, le, lp)); gstep += 1
        if k % every == 0:
            snap(f"warm-up step {k} of {cfg.warmup}")
    for it in range(1, cfg.n_iters + 1):
        if cfg.adam_reset_every and it > 1 and (it - 1) % cfg.adam_reset_every == 0:
            optE, optP = pt.make_adam(pE, cfg.lr_E), pt.make_adam(pP, cfg.lr_P)
        for _ in range(cfg.n_P_steps):
            le, lp = step_with_losses("P", pE, pP, x0, phi, cfg, optE, optP)
            hist.append((gstep, le, lp)); gstep += 1
        for _ in range(cfg.n_E_steps):
            le, lp = step_with_losses("E", pE, pP, x0, phi, cfg, optE, optP)
            hist.append((gstep, le, lp)); gstep += 1
        if it % every == 0:
            snap(f"self-play iteration {it} of {cfg.n_iters}")
    return snaps, labels, np.array(snap_step), np.array(hist), x0.cpu().numpy()[0]


def legacy(p, r=0):
    return {"B": p["B"][r], "b_body": p["b"][r], "W": p["W"][r]}


def emb(x, L, R0=1.6, r=0.6):
    a = 2 * np.pi * x[..., 0] / L
    b = 2 * np.pi * x[..., 1] / L
    return np.stack([(R0 + r * np.cos(b)) * np.cos(a), (R0 + r * np.cos(b)) * np.sin(a), r * np.sin(b)], -1)


def wrapped_segments(traj, L):
    """Split a wrapped path (n, 2) into pieces at the seams, for a LineCollection."""
    xs, ys = traj[:, 0], traj[:, 1]
    brk = np.where((np.abs(np.diff(xs)) > L / 2) | (np.abs(np.diff(ys)) > L / 2))[0] + 1
    return [traj[idx] for idx in np.split(np.arange(len(xs)), brk) if len(idx) > 1]


# ----------------------------------------------------------------- render
def render(snaps, labels, snap_step, hist, x0, cfg_r: RunConfig, cfg: pt.Config,
           heads3d, out_path, fps, heads_left=None, fmt="mp4", left="lifted"):
    n_h = x0.shape[0]
    heads_left = list(range(n_h)) if heads_left is None else list(heads_left)
    sims = [simulate(legacy(pE), legacy(pP), x0, cfg_r) for pE, pP in snaps]
    S0 = sims[0]
    L = cfg_r.L
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.3], hspace=0.28, wspace=0.08)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], projection="3d")
    axB = fig.add_subplot(gs[1, :])

    # static scenery, left
    linesE, linesP, caps = {}, {}, {}
    th = np.linspace(0, 2 * np.pi, 200)
    if left == "lifted":
        for m in range(-2, 3):
            for n in range(-2, 3):
                axL.scatter(m * L, n * L, s=14, c="0.6", zorder=1)
                axL.add_patch(plt.Rectangle((m * L, n * L), L, L, fill=False, ec="0.85", lw=0.6))
        axL.plot(cfg_r.reach * np.cos(th), cfg_r.reach * np.sin(th), color="0.5", ls=":", lw=1)
        axL.set_aspect("equal"); axL.set_xlim(-2.2 * L, 2.2 * L); axL.set_ylim(-2.2 * L, 2.2 * L)
        axL.set_xlabel("x"); axL.set_ylabel("y")
        axL.set_title("Lifted paths on R², cut at capture (solid = evader, dashed = pursuer, ○ = capture)", fontsize=10)
        for j in heads_left:
            c = head_color(j)
            tag = "capturable" if S0["capturable"][j] else "uncapturable"
            (lE,) = axL.plot([], [], color=c, lw=1.4, label=f"head {j} ({tag}, |Δ0|={S0['d0'][j]:.2f})")
            (lP,) = axL.plot([], [], color=c, lw=1.0, ls="--")
            axL.scatter(*x0[j], s=30, color=c, edgecolor="k", zorder=5)
            axL.annotate(str(j), x0[j], fontsize=7, xytext=(3, 3), textcoords="offset points")
            cp = axL.scatter([], [], s=70, facecolors="none", edgecolors=c, lw=1.5)
            linesE[j] = lE; linesP[j] = lP; caps[j] = cp
    else:
        # the flat torus: fundamental domain [0, L)^2, opposite sides identified
        for cx in (0, L):
            for cy in (0, L):
                axL.plot(cx + cfg_r.reach * np.cos(th), cy + cfg_r.reach * np.sin(th), color="0.6", lw=0.8, ls=":")
        axL.scatter(0, 0, s=60, c="tab:red", marker="s", edgecolor="k", zorder=5, label="pursuer start (all heads)")
        axL.set_aspect("equal"); axL.set_xlim(0, L); axL.set_ylim(0, L)
        axL.set_xticks([0, L / 2, L]); axL.set_yticks([0, L / 2, L])
        for sp in axL.spines.values():
            sp.set_linewidth(1.6); sp.set_edgecolor("0.3")
        axL.set_xlabel("x  (x = 0 and x = L identified)"); axL.set_ylabel("y  (y = 0 and y = L identified)")
        axL.set_title("Flat torus [0, L)², periodic, paths cut at capture (solid = evader, dashed = pursuer, ○ = capture)",
                      fontsize=10)
        for j in heads_left:
            c = head_color(j)
            tag = "capturable" if S0["capturable"][j] else "uncapturable"
            lE = LineCollection([], colors=[c], linewidths=1.4, label=f"head {j} ({tag}, |Δ0|={S0['d0'][j]:.2f})")
            lP = LineCollection([], colors=[c], linewidths=1.0, linestyles="--")
            axL.add_collection(lE); axL.add_collection(lP)
            axL.scatter(*(x0[j] % L), s=30, color=c, edgecolor="k", zorder=5)
            axL.annotate(str(j), x0[j] % L, fontsize=7, xytext=(3, 3), textcoords="offset points")
            cp = axL.scatter([], [], s=70, facecolors="none", edgecolors=c, lw=1.5)
            linesE[j] = lE; linesP[j] = lP; caps[j] = cp
    axL.legend(fontsize=6.5 if len(heads_left) > 6 else 8, loc="upper right",
               ncol=2 if len(heads_left) > 6 else 1, framealpha=0.9, title="colour = head")

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
        axR.scatter(*emb(x0[j], L), color=c, s=40, edgecolor="k")
        lines3E[j] = lE; lines3P[j] = lP
    axR.set_box_aspect((1, 1, 0.4)); axR.set_axis_off()
    axR.legend(fontsize=8, loc="upper left")
    axR.set_title("Embedded torus (same colours)", fontsize=10)

    # loss history, bottom: full curves once; a marker moves along them
    st, LE, LP = hist[:, 0], hist[:, 1], hist[:, 2]
    axB.axvspan(0, cfg.warmup, color="0.92", lw=0)
    axB.text(cfg.warmup / 2, 0.97, "pursuer-only\nwarm-up", ha="center", va="top", fontsize=8,
             transform=axB.get_xaxis_transform())
    for k in range(1, cfg.n_iters // cfg.adam_reset_every + 1) if cfg.adam_reset_every else []:
        axB.axvline(cfg.warmup + k * cfg.adam_reset_every * (cfg.n_P_steps + cfg.n_E_steps),
                    color="0.75", ls=":", lw=0.8)
    axB.plot(st, LP, color="tab:red", lw=0.8, label=r"$L_P$ (pursuer, minimises)")
    axB.plot(st, LE, color="tab:blue", lw=0.8, label=r"$L_E$ (evader, minimises)")
    mP = axB.scatter([], [], s=70, color="tab:red", edgecolor="k", zorder=6)
    mE = axB.scatter([], [], s=70, color="tab:blue", edgecolor="k", zorder=6)
    vline = axB.axvline(0, color="k", lw=1)
    ltxt = axB.text(0.995, 0.95, "", ha="right", va="top", fontsize=9, transform=axB.transAxes,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    axB.set_xlim(0, st[-1]); axB.set_xlabel("gradient step (warm-up: 1 per step; self-play: 3 P + 1 E per iteration)")
    axB.set_ylabel("loss"); axB.legend(fontsize=8, loc="center right")
    axB.set_title("Loss history of Eq. (1); dotted = Adam resets; marker = current frame", fontsize=10)
    axB.grid(alpha=0.25)
    sup = fig.suptitle("", fontsize=12, y=0.95)

    def update(i):
        S = sims[i]
        for j in heads_left:
            if left == "lifted":
                pE_, pP_ = cut(S, j, S["xE_l"][j])[0], cut(S, j, S["xP_l"][j])[0]
                linesE[j].set_data(pE_[:, 0], pE_[:, 1]); linesP[j].set_data(pP_[:, 0], pP_[:, 1])
                caps[j].set_offsets(S["xP_l"][j, S["kc"][j]][None, :] if S["captured"][j] else np.empty((0, 2)))
            else:
                linesE[j].set_segments(wrapped_segments(cut(S, j, S["xE"][j])[0] % L, L))
                linesP[j].set_segments(wrapped_segments(cut(S, j, S["xP"][j])[0] % L, L))
                caps[j].set_offsets((S["xP"][j, S["kc"][j]] % L)[None, :] if S["captured"][j] else np.empty((0, 2)))
        for j in heads3d:
            eE = emb(cut(S, j, S["xE"][j])[0], L); eP = emb(cut(S, j, S["xP"][j])[0], L)
            lines3E[j].set_data(eE[:, 0], eE[:, 1]); lines3E[j].set_3d_properties(eE[:, 2])
            lines3P[j].set_data(eP[:, 0], eP[:, 1]); lines3P[j].set_3d_properties(eP[:, 2])
        k = min(int(snap_step[i]), len(st) - 1)
        mP.set_offsets([[st[k], LP[k]]]); mE.set_offsets([[st[k], LE[k]]])
        vline.set_xdata([st[k], st[k]])
        ltxt.set_text(f"step {int(st[k])}:  L_P = {LP[k]:.3f},  L_E = {LE[k]:.3f}")
        n_cap = int(S["captured"].sum()); n_capable = int(S["capturable"].sum())
        n_cap_c = int((S["captured"] & S["capturable"]).sum())
        sup.set_text(f"Baseline{' (capture-truncated loss)' if cfg.truncate else ''} training "
                     f"(init seed 0, IC seed 0): {labels[i]}   |   "
                     f"captured {n_cap} of {n_h} heads ({n_cap_c} of the {n_capable} capturable)   |   "
                     f"frame {i + 1}/{len(sims)}")
        return []

    anim = FuncAnimation(fig, update, frames=len(sims), blit=False)
    if fmt == "mp4":
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    else:
        writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--heads3d", default="1,4,11,14,0,2")
    ap.add_argument("--heads", default=None, help="restrict both panels to these heads, e.g. 4,2")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--snapshots", default=None, help="reuse a saved training_snapshots.npz")
    ap.add_argument("--out", default="training_evolution")
    ap.add_argument("--format", default="mp4", choices=["mp4", "gif"])
    ap.add_argument("--left", default="lifted", choices=["lifted", "flat"])
    ap.add_argument("--truncate", action="store_true", help="train with the capture-truncated distance term")
    a = ap.parse_args()
    out = os.path.join(HERE, "figs", a.name)
    os.makedirs(out, exist_ok=True)
    cfg = pt.Config(truncate=a.truncate)
    if a.snapshots:
        z = np.load(a.snapshots)
        if "loss_hist" not in z.files:
            sys.exit("these snapshots have no loss history; retrain without --snapshots")
        labels = [str(l) for l in z["labels"]]; x0 = z["x_E0"]
        snap_step, hist = z["snap_step"], z["loss_hist"]
        snaps = [({"B": z["BE"][i][None], "b": z["bE"][i][None], "W": z["WE"][i][None]},
                  {"B": z["BP"][i][None], "b": z["bP"][i][None], "W": z["WP"][i][None]})
                 for i in range(len(labels))]
        print(f"loaded {len(snaps)} snapshots from {a.snapshots}", flush=True)
    else:
        t0 = time.time()
        snaps, labels, snap_step, hist, x0 = train_with_snapshots(cfg, a.every)
        print(f"trained with {len(snaps)} snapshots in {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(os.path.join(out, "training_snapshots.npz"),
                            labels=np.array(labels), x_E0=x0, snap_step=snap_step, loss_hist=hist,
                            BE=np.stack([s[0]["B"][0] for s in snaps]), bE=np.stack([s[0]["b"][0] for s in snaps]),
                            WE=np.stack([s[0]["W"][0] for s in snaps]), BP=np.stack([s[1]["B"][0] for s in snaps]),
                            bP=np.stack([s[1]["b"][0] for s in snaps]), WP=np.stack([s[1]["W"][0] for s in snaps]))
    t0 = time.time()
    heads_left = [int(h) for h in a.heads.split(",")] if a.heads else None
    heads3d = heads_left if heads_left is not None else [int(h) for h in a.heads3d.split(",")]
    path = os.path.join(out, f"{a.out}.{a.format}")
    render(snaps, labels, snap_step, hist, x0, RunConfig(), cfg, heads3d, path, a.fps,
           heads_left=heads_left, fmt=a.format, left=a.left)
    print(f"rendered {len(snaps)} frames to {path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
