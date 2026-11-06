"""Training-evolution video of the S^1 multi-head run, from the snapshots saved by run_topology.py.

Top row: four heads (two captured, two escaped at the end of training), each drawn on the circle
S^1 = [0, L) with angle 2*pi*x/L.  A path on a circle overlaps itself, so each trajectory is drawn as a
space-time spiral: the radius grows with time, r(t) = 1 + a t, so position x at time t sits at
(r(t) cos(2*pi*x/L), r(t) sin(2*pi*x/L)).  Evader solid, pursuer dashed, paths cut at capture, a
circle at the capture point; the unit circle is t = 0, the outer dotted circle t = t_max.
Bottom: the loss history of both players with a marker at the current snapshot.

Usage: python3 torch_actor/topologies/animate_s1.py [--snapshots results/S1_mh_snapshots.npz] [--fps 3]
       [--heads 2 captured + 2 escaped chosen automatically, or --heads j1,j2,j3,j4] [--out figs/S1_training_evolution.mp4]
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe_nd as nd  # noqa: E402

C_E, C_P = "#2a78d6", "#eb6834"
A = 0.12          # spiral growth per unit time


def paths(pE, pP, x0, cfg):
    g = nd.game_stats(pE, pP, x0, cfg)
    return np.array(g["xE"]), np.array(g["xP"]), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default=os.path.join(HERE, "results", "S1_mh_snapshots.npz"))
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--heads", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "S1_training_evolution.mp4"))
    a = ap.parse_args()
    z = np.load(a.snapshots)
    labels, steps, hist, x0 = [str(s) for s in z["labels"]], z["snap_step"], z["loss_hist"], z["x_E0"]
    n = len(labels)
    snaps = [({"B": z["BE"][i], "b": z["bE"][i], "W": z["WE"][i]}, {"B": z["BP"][i], "b": z["bP"][i], "W": z["WP"][i]}) for i in range(n)]
    cfg = nd.Config(truncate=True, n_h=x0.shape[0])
    L, tmax = cfg.L, cfg.t_max
    sims = [paths(pE, pP, x0, cfg) for pE, pP in snaps]
    g_end = sims[-1][2]
    cap, capt = np.array(g_end["capturable"]), np.array(g_end["captured"])
    if a.heads:
        heads = [int(h) for h in a.heads.split(",")]
    else:
        heads = list(np.where(cap & capt)[0][:2]) + list(np.where(~capt)[0][:2])
    tg = np.linspace(0, tmax, cfg.n_t + 1)
    r = 1 + A * tg

    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.3], hspace=0.3, wspace=0.15)
    axes = [fig.add_subplot(gs[0, k]) for k in range(4)]
    axB = fig.add_subplot(gs[1, :])
    th = np.linspace(0, 2 * np.pi, 300)
    arts = []
    for ax, j in zip(axes, heads):
        ax.plot(np.cos(th), np.sin(th), color="0.75", lw=1)
        ax.plot((1 + A * tmax) * np.cos(th), (1 + A * tmax) * np.sin(th), color="0.85", lw=0.8, ls=":")
        aR = 2 * np.pi * g_end["reach"] / L
        ax.plot(np.cos(np.linspace(-aR, aR, 100)), np.sin(np.linspace(-aR, aR, 100)), color="0.45", lw=3, ls=":", alpha=0.6)
        ax.plot(1, 0, "s", color="k", ms=7, zorder=5)
        ax.plot(np.cos(2 * np.pi * x0[j, 0] / L), np.sin(2 * np.pi * x0[j, 0] / L), "o", color=C_E, ms=6, zorder=5)
        lE, = ax.plot([], [], color=C_E, lw=1.8)
        lP, = ax.plot([], [], color=C_P, lw=1.5, ls="--")
        mk, = ax.plot([], [], "o", ms=11, mfc="none", mec="k", mew=1.5)
        ttl = ax.set_title("", fontsize=9)
        ax.set_aspect("equal"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.axis("off")
        arts.append((lE, lP, mk, ttl))
    axes[0].text(0, -1.52, "unit circle = t = 0, radius grows with t; dotted thick arc = capturable region; square = pursuer start",
                 fontsize=7.5, ha="left", color="0.3", transform=axes[0].transData, clip_on=False)
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

    def update(i):
        xE, xP, g = sims[i]
        for (lE, lP, mk, ttl), j in zip(arts, heads):
            k = int(np.rint(g["Tc"][j] / (tmax / cfg.n_t))) if g["captured"][j] else cfg.n_t
            k = min(max(k, 1), cfg.n_t)
            thE, thP = 2 * np.pi * xE[j, :k + 1, 0] / L, 2 * np.pi * xP[j, :k + 1, 0] / L
            lE.set_data(r[:k + 1] * np.cos(thE), r[:k + 1] * np.sin(thE))
            lP.set_data(r[:k + 1] * np.cos(thP), r[:k + 1] * np.sin(thP))
            if g["captured"][j]:
                mk.set_data([r[k] * np.cos(thE[k])], [r[k] * np.sin(thE[k])])
                out = f"captured, $T_c$ = {g['Tc'][j]:.2f} (V* = {g['Vstar'][j]:.2f})"
            else:
                mk.set_data([], []); out = "escapes"
            ttl.set_text(f"head {j}: |Δ0| = {g['d0'][j]:.2f} ({'capturable' if g['capturable'][j] else 'uncapturable'})\n{out}")
        s = steps[i]
        mP.set_data([s], [hist[s - 1, 1]] if s > 0 else [hist[0, 1]]); mE.set_data([s], [hist[s - 1, 0]] if s > 0 else [hist[0, 0]])
        txt.set_text(f"step {s}:  L_P = {hist[max(s - 1, 0), 1]:.3f},  L_E = {hist[max(s - 1, 0), 0]:.3f}")
        sup.set_text(f"$S^1$ baseline (capture-truncated loss, warm-up {cfg.warmup}, {cfg.n_P_steps}:{cfg.n_E_steps} steps), 16 heads: {labels[i]}   |   "
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
    print(f"rendered {n} frames (heads {heads}) to {a.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
