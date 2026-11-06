"""Training-evolution video of the T^3 multi-head run from the snapshots saved by run_topology.py.

Left: the fundamental cube [0, L)^3 with the paths of six heads (three captured, three escaped at
the end of training), wrapped periodically (a path leaving one face re-enters at the opposite one),
evader solid, pursuer dashed, one colour per head, paths cut at capture, a ring at the capture point,
a square at the pursuer start.  Middle: a legend naming each head and its outcome.  Bottom: the loss
history of both players with a marker at the current snapshot.

Usage: python3 torch_actor/topologies/animate_t3.py [--snapshots results/T3_mh_snapshots.npz] [--fps 3]
       [--heads j1,...,j6] [--out figs/T3_training_evolution.mp4] [--elev 22] [--azim -55]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe_nd as nd  # noqa: E402

COLORS = plt.get_cmap("tab10").colors


def wrapped_segments(x, L):
    """Split a lifted path (T, 3) into pieces that do not cross a face of [0, L)^3."""
    xm = x % L
    jumps = np.where(np.abs(np.diff(xm, axis=0)).max(1) > L / 2)[0] + 1
    return np.split(xm, jumps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default=os.path.join(HERE, "results", "T3_mh_snapshots.npz"))
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--heads", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "T3_training_evolution.mp4"))
    ap.add_argument("--elev", type=float, default=22); ap.add_argument("--azim", type=float, default=-55)
    a = ap.parse_args()
    z = np.load(a.snapshots)
    labels, steps, hist, x0 = [str(s) for s in z["labels"]], z["snap_step"], z["loss_hist"], z["x_E0"]
    n = len(labels)
    cfg = nd.Config(truncate=True, n_h=x0.shape[0]); L, tmax = cfg.L, cfg.t_max
    snaps = [({"B": z["BE"][i], "b": z["bE"][i], "W": z["WE"][i]}, {"B": z["BP"][i], "b": z["bP"][i], "W": z["WP"][i]}) for i in range(n)]
    sims = [nd.game_stats(pE, pP, x0, cfg) for pE, pP in snaps]
    g_end = sims[-1]
    cap, capt = np.array(g_end["capturable"]), np.array(g_end["captured"])
    if a.heads:
        heads = [int(h) for h in a.heads.split(",")]
    else:
        heads = list(np.where(cap & capt)[0][:3]) + list(np.where(~cap & ~capt)[0][:3])
    dt = tmax / cfg.n_t

    fig = plt.figure(figsize=(15, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.2], width_ratios=[1.6, 1], hspace=0.25, wspace=0.0)
    ax = fig.add_subplot(gs[0, 0], projection="3d"); axM = fig.add_subplot(gs[0, 1]); axM.axis("off")
    axB = fig.add_subplot(gs[1, :])
    ax.view_init(elev=a.elev, azim=a.azim)
    for c in np.array(np.meshgrid([0, L], [0, L], [0, L])).reshape(3, -1).T:
        ax.scatter(*c, s=40, color="k", marker="s", zorder=5)
    for h, j in enumerate(heads):
        ax.scatter(*(x0[j] % L), s=35, color=COLORS[h], zorder=5)
        ax.text(*(x0[j] % L), f" {j}", fontsize=8, color=COLORS[h])
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Fundamental cube [0, L)³, periodic; paths cut at capture (solid = evader, dashed = pursuer, ○ = capture)", fontsize=9)
    st = np.arange(1, len(hist) + 1)
    axB.plot(st, hist[:, 1], color="#c0392b", lw=0.8, label="$L_P$ (pursuer, minimises)")
    axB.plot(st, hist[:, 0], color="#1f6f8b", lw=0.8, label="$L_E$ (evader, minimises)")
    if cfg.warmup:
        axB.axvspan(0, cfg.warmup, color="0.92", lw=0)
        axB.text(cfg.warmup / 2, 0.97, "pursuer-only\nwarm-up", ha="center", va="top", fontsize=8, transform=axB.get_xaxis_transform())
    for k in range(1, cfg.n_iters // cfg.adam_reset_every + 1):
        axB.axvline(cfg.warmup + k * cfg.adam_reset_every * (cfg.n_P_steps + cfg.n_E_steps), color="0.6", lw=0.7, ls=":")
    mP, = axB.plot([], [], "o", color="#c0392b", ms=7); mE, = axB.plot([], [], "o", color="#1f6f8b", ms=7)
    txt = axB.text(0.99, 0.95, "", transform=axB.transAxes, ha="right", va="top", fontsize=9, bbox=dict(fc="white", ec="0.7"))
    axB.set_xlim(0, st[-1]); axB.set_xlabel(f"gradient step (warm-up: 1 per step; self-play: {cfg.n_P_steps} P + {cfg.n_E_steps} E per iteration)")
    axB.set_ylabel("loss"); axB.legend(fontsize=8, loc="center right"); axB.grid(alpha=0.3)
    axB.set_title("Loss history of Eq. (1); dotted = Adam resets; marker = current frame", fontsize=10)
    sup = fig.suptitle("", fontsize=11, y=0.97)
    drawn = []

    def update(i):
        for art in drawn:
            art.remove()
        drawn.clear()
        g = sims[i]; xE, xP = np.array(g["xE"]), np.array(g["xP"])
        handles = []
        for h, j in enumerate(heads):
            k = int(np.rint(g["Tc"][j] / dt)) if g["captured"][j] else cfg.n_t
            k = min(max(k, 1), cfg.n_t)
            for seg in wrapped_segments(xE[j, :k + 1], L):
                drawn.append(ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=COLORS[h], lw=1.8)[0])
            for seg in wrapped_segments(xP[j, :k + 1], L):
                drawn.append(ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=COLORS[h], lw=1.3, ls="--")[0])
            if g["captured"][j]:
                p = xE[j, k] % L
                drawn.append(ax.scatter(*p, s=90, facecolor="none", edgecolor="k", lw=1.5, zorder=6))
                out = f"captured, $T_c$ = {g['Tc'][j]:.2f} (|Δ0| = {g['d0'][j]:.2f})"
            else:
                out = f"escapes (|Δ0| = {g['d0'][j]:.2f}, {'capturable' if g['capturable'][j] else 'uncapturable'})"
            handles.append(Line2D([], [], color=COLORS[h], lw=2, label=f"head {j}: {out}"))
        handles += [Line2D([], [], color="k", lw=1.5, label="evader (solid)"), Line2D([], [], color="k", lw=1.2, ls="--", label="pursuer (dashed)"),
                    Line2D([], [], marker="o", mfc="none", mec="k", ls="", label="capture"), Line2D([], [], marker="s", color="k", ls="", label="pursuer start and images")]
        drawn.append(axM.legend(handles=handles, loc="center left", fontsize=8.5, title="outcome at the end of training", title_fontsize=9, frameon=True))
        s = steps[i]
        mP.set_data([s], [hist[max(s - 1, 0), 1]]); mE.set_data([s], [hist[max(s - 1, 0), 0]])
        txt.set_text(f"step {s}:  L_P = {hist[max(s - 1, 0), 1]:.3f},  L_E = {hist[max(s - 1, 0), 0]:.3f}")
        sup.set_text(f"$T^3$ baseline (capture-truncated loss, warm-up {cfg.warmup}, {cfg.n_P_steps}:{cfg.n_E_steps} steps), {x0.shape[0]} heads: {labels[i]}   |   "
                     f"captured {int(np.sum(g['captured']))} of {x0.shape[0]} heads ({int(np.sum(np.array(g['captured']) & cap))} of the {int(cap.sum())} capturable)   |   frame {i + 1}/{n}")
        return []

    t0 = time.time()
    anim = FuncAnimation(fig, update, frames=n, blit=False)
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    anim.save(a.out, writer=FFMpegWriter(fps=a.fps, codec="libx264", bitrate=-1, extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]), dpi=90)
    plt.close(fig)
    print(f"rendered {n} frames (heads {[int(h) for h in heads]}) to {a.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
