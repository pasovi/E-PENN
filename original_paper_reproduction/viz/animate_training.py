"""Animate how the open-loop headings u(t) evolve during training.

Reads viz/snapshots_baseline/snapshots.npz (from viz/train_snapshots.py) and
writes, in the same folder:

  training_evolution.gif     one frame per snapshot.  Top row: all 16 heads
                             on the torus; heading deviation from the saddle
                             angle, theta(t) - theta_saddle, for every evader
                             head and every pursuer head.  Bottom row: two
                             individual E-P pairs (torus path + headings + d(t)).
  fig_evolution_grid.png     the top row at eight chosen snapshots.
  fig_evolution_pairs.png    the two pairs at the same eight snapshots.
  fig_training_metrics.png   constancy, Gram PR, PR(H), captures and losses
                             against training iteration.
  frames/frame_XX.png        every frame, for the record.

Usage: python3 viz/animate_training.py [--snap viz/snapshots_baseline] [--heads A B]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rev2"))
sys.path.insert(0, os.path.join(ROOT, "viz"))
from pursuit_evasion_numpy import fourier_features, forward_dirs  # noqa: E402
from core import RunConfig  # noqa: E402
import metrics as M  # noqa: E402
import visualize_actor as V  # noqa: E402

GRID_PICKS = [("warmup", 0), ("warmup", 100), ("warmup", 1500), ("selfplay", 100),
              ("selfplay", 500), ("selfplay", 1000), ("selfplay", 3500), ("selfplay", 14000)]


def load(snapdir):
    z = np.load(os.path.join(snapdir, "snapshots.npz"))
    n = len(z["it"])
    snaps = []
    for s in range(n):
        pE = {"B": z["BE"][s], "b_body": z["bE"][s], "W": z["WE"][s]}
        pP = {"B": z["BP"][s], "b_body": z["bP"][s], "W": z["WP"][s]}
        snaps.append(dict(phase=str(z["phase"][s]), it=int(z["it"][s]),
                          L_E=float(z["L_E"][s]), L_P=float(z["L_P"][s]), pE=pE, pP=pP))
    return snaps, z["x_E0"]


def label(s):
    if s["phase"] == "warmup":
        return f"warm-up step {s['it']} / 1500 (pursuer only)"
    return f"self-play iteration {s['it']} / 14000"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def headings(p, td, cfg):
    phid = fourier_features(td, cfg.t_max, cfg.K)
    u, _ = forward_dirs(p, phid)
    return np.arctan2(u[:, :, 1], u[:, :, 0])


# ------------------------------------------------------------------ panels
def panel_torus_all(ax, S, cfg, title=True):
    n_h = S["xE"].shape[0]
    colors = cm.viridis(np.linspace(0, 1, n_h))
    for j in range(n_h):
        dh = S["Delta0"][j] / S["d0"][j]
        tt = np.linspace(0, cfg.t_max, 100)[:, None]
        V._torus_path(ax, (S["xE"][j, 0] + cfg.v_E * tt * dh) % cfg.L, cfg.L, "0.75", lw=0.6, ls="--")
        V._torus_path(ax, (cfg.v_P * tt * dh) % cfg.L, cfg.L, "0.75", lw=0.6, ls="--")
    for j in range(n_h):
        V._torus_path(ax, S["xE"][j], cfg.L, colors[j], lw=1.3)
        V._torus_path(ax, S["xP"][j], cfg.L, colors[j], lw=1.3, ls=(0, (2, 1)))
        ax.scatter(*S["xE"][j, 0], s=22, color=colors[j], edgecolor="k", lw=0.5, zorder=5)
        if S["captured"][j]:
            ax.scatter(*S["xP"][j, S["kc"][j]], s=60, facecolors="none", edgecolors="k", lw=1.2, zorder=6)
    ax.scatter(0, 0, s=50, c="k", marker="s", zorder=7)
    ax.set_xlim(0, cfg.L); ax.set_ylim(0, cfg.L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(f"all 16 heads on T²\n(solid E, dotted P, grey dashed = saddle)\n"
                     f"captured: {int(S['captured'].sum())}/16", fontsize=8.5)


def panel_dev(ax, th, sad, td, who, title=True):
    n_h = th.shape[0]
    colors = cm.viridis(np.linspace(0, 1, n_h))
    for j in range(n_h):
        ax.plot(td, wrap(th[j] - sad[j]), color=colors[j], lw=1.1)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylim(-np.pi, np.pi); ax.set_xlim(td[0], td[-1])
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]); ax.set_yticklabels(["−π", "−π/2", "0", "π/2", "π"])
    ax.set_xlabel("t", fontsize=8); ax.tick_params(labelsize=7)
    if title:
        ax.set_title(f"{who}: heading θ(t) − saddle angle, all heads", fontsize=9)


def panel_pair(ax_t, ax_h, S, cfg, thE, thP, sad, td, j):
    # torus
    dh = S["Delta0"][j] / S["d0"][j]
    tt = np.linspace(0, cfg.t_max, 100)[:, None]
    V._torus_path(ax_t, (S["xE"][j, 0] + cfg.v_E * tt * dh) % cfg.L, cfg.L, "tab:blue", lw=0.8, ls="--", alpha=0.6)
    V._torus_path(ax_t, (cfg.v_P * tt * dh) % cfg.L, cfg.L, "tab:red", lw=0.8, ls="--", alpha=0.6)
    V._torus_path(ax_t, S["xE"][j], cfg.L, "tab:blue", lw=1.5)
    V._torus_path(ax_t, S["xP"][j], cfg.L, "tab:red", lw=1.5)
    ax_t.scatter(*S["xE"][j, 0], s=40, c="tab:blue", edgecolor="k", zorder=5)
    ax_t.scatter(0, 0, s=40, c="tab:red", marker="s", edgecolor="k", zorder=5)
    if S["captured"][j]:
        ax_t.scatter(*S["xP"][j, S["kc"][j]], s=90, facecolors="none", edgecolors="k", lw=1.5, zorder=6)
    ax_t.set_xlim(0, cfg.L); ax_t.set_ylim(0, cfg.L); ax_t.set_aspect("equal")
    ax_t.set_xticks([]); ax_t.set_yticks([])
    cap = "capturable" if S["capturable"][j] else "uncapturable"
    tc = f"T_c={S['Tc'][j]:.2f} (V*={S['Vstar'][j]:.2f})" if S["captured"][j] else "no capture"
    ax_t.set_title(f"head {j}: |Δ0|={S['d0'][j]:.2f}, {cap}\n{tc}", fontsize=8.5)
    # headings
    e = np.unwrap(thE[j]); p = np.unwrap(thP[j])
    e -= 2 * np.pi * np.round((e[0] - sad[j]) / (2 * np.pi))
    p -= 2 * np.pi * np.round((p[0] - sad[j]) / (2 * np.pi))
    ax_h.plot(td, e, color="tab:blue", lw=1.3, label="θ_E")
    ax_h.plot(td, p, color="tab:red", lw=1.3, label="θ_P")
    ax_h.axhline(sad[j], color="k", ls="--", lw=0.8, label="saddle")
    ax2 = ax_h.twinx()
    ax2.plot(S["tg"], S["d"][j], color="0.4", lw=0.9, alpha=0.8)
    ax2.axhline(cfg.eps, color="0.4", ls=":", lw=0.8)
    ax2.set_ylim(0, 1.6); ax2.tick_params(labelsize=6, colors="0.4")
    if S["captured"][j]:
        ax_h.axvline(S["Tc"][j], color="g", lw=0.8, alpha=0.7)
    ax_h.set_ylim(sad[j] - np.pi, sad[j] + np.pi); ax_h.set_xlim(td[0], td[-1])
    ax_h.tick_params(labelsize=7); ax_h.set_xlabel("t", fontsize=8)
    ax_h.set_title(f"head {j}: θ_E, θ_P (left), d(t) grey (right)", fontsize=8.5)
    ax_h.legend(fontsize=6, loc="upper left")


# ------------------------------------------------------------------ frame
def render_frame(s, x0, cfg, heads, td):
    S = V.simulate(s["pE"], s["pP"], x0, cfg)
    thE = headings(s["pE"], td, cfg); thP = headings(s["pP"], td, cfg)
    sad = np.arctan2(S["Delta0"][:, 1], S["Delta0"][:, 0])
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.38, wspace=0.3)
    ax_all = fig.add_subplot(gs[0, 0])
    panel_torus_all(ax_all, S, cfg)
    panel_dev(fig.add_subplot(gs[0, 1:3]), thE, sad, td, "evader")
    panel_dev(fig.add_subplot(gs[0, 3]), thP, sad, td, "pursuer")
    for k, j in enumerate(heads):
        panel_pair(fig.add_subplot(gs[1, 2 * k]), fig.add_subplot(gs[1, 2 * k + 1]), S, cfg, thE, thP, sad, td, j)
    fig.suptitle(f"{label(s)}     L_P = {s['L_P']:+.3f}   L_E = {s['L_E']:+.3f}", fontsize=13, y=0.985)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    return img, S


def render_frame_compact(s, x0, cfg, head, td):
    """All heads (torus + heading deviations) on top, one E-P pair below."""
    S = V.simulate(s["pE"], s["pP"], x0, cfg)
    thE = headings(s["pE"], td, cfg); thP = headings(s["pP"], td, cfg)
    sad = np.arctan2(S["Delta0"][:, 1], S["Delta0"][:, 0])
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.4, wspace=0.3)
    panel_torus_all(fig.add_subplot(gs[0, 0]), S, cfg)
    panel_dev(fig.add_subplot(gs[0, 1]), thE, sad, td, "evader")
    panel_dev(fig.add_subplot(gs[0, 2]), thP, sad, td, "pursuer")
    panel_pair(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:3]), S, cfg, thE, thP, sad, td, head)
    fig.suptitle(f"{label(s)}     L_P = {s['L_P']:+.3f}   L_E = {s['L_E']:+.3f}", fontsize=13, y=0.985)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    return img


def write_gif(frames, path, frame_ms):
    dur = [frame_ms] * len(frames); dur[0] = 1200; dur[-1] = 2500
    small = [f.resize((f.width * 2 // 3, f.height * 2 // 3), Image.LANCZOS) for f in frames]
    small[0].save(path, save_all=True, append_images=small[1:], duration=dur, loop=0, optimize=False)


def fig_grid(snaps, x0, cfg, td, path):
    picks = [next(s for s in snaps if (s["phase"], s["it"]) == pk) for pk in GRID_PICKS]
    n = len(picks)
    fig, axes = plt.subplots(3, n, figsize=(2.9 * n, 8.2))
    for c, s in enumerate(picks):
        S = V.simulate(s["pE"], s["pP"], x0, cfg)
        thE = headings(s["pE"], td, cfg); thP = headings(s["pP"], td, cfg)
        sad = np.arctan2(S["Delta0"][:, 1], S["Delta0"][:, 0])
        panel_torus_all(axes[0, c], S, cfg, title=False)
        axes[0, c].set_title(label(s).replace(" (pursuer only)", "\n(pursuer only)").replace(" / ", "/") +
                             f"\ncaptured {int(S['captured'].sum())}/16", fontsize=8)
        panel_dev(axes[1, c], thE, sad, td, "evader", title=False)
        panel_dev(axes[2, c], thP, sad, td, "pursuer", title=False)
        if c > 0:
            for r in (1, 2):
                axes[r, c].set_yticklabels([])
    axes[1, 0].set_ylabel("evader: θ_E(t) − saddle angle", fontsize=9)
    axes[2, 0].set_ylabel("pursuer: θ_P(t) − saddle angle", fontsize=9)
    fig.suptitle("Evolution of the open-loop headings during training (all 16 heads; one colour per head)",
                 y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_pairs(snaps, x0, cfg, td, heads, path):
    picks = [next(s for s in snaps if (s["phase"], s["it"]) == pk) for pk in GRID_PICKS]
    n = len(picks)
    fig, axes = plt.subplots(2 * len(heads), n, figsize=(2.9 * n, 3.0 * 2 * len(heads)))
    for c, s in enumerate(picks):
        S = V.simulate(s["pE"], s["pP"], x0, cfg)
        thE = headings(s["pE"], td, cfg); thP = headings(s["pP"], td, cfg)
        sad = np.arctan2(S["Delta0"][:, 1], S["Delta0"][:, 0])
        for k, j in enumerate(heads):
            panel_pair(axes[2 * k, c], axes[2 * k + 1, c], S, cfg, thE, thP, sad, td, j)
            axes[2 * k + 1, c].get_legend().remove()
            if c == 0:
                axes[2 * k + 1, c].legend(fontsize=6, loc="upper left")
        axes[0, c].set_title(label(s).replace(" (pursuer only)", "\n(pursuer only)").replace(" / ", "/") +
                             "\n" + axes[0, c].get_title().split("\n")[1], fontsize=8)
    fig.suptitle(f"Two E–P pairs through training: heads {heads[0]} and {heads[1]} "
                 "(dashed = analytic saddle, ○ = capture, green = capture time)", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_metrics(snaps, x0, cfg, path):
    _, phid = M.dense_phi(cfg, 512)
    rows = []
    for s in snaps:
        S = V.simulate(s["pE"], s["pP"], x0, cfg)
        r = {"x": s["it"] - 1500 if s["phase"] == "warmup" else s["it"], "cap": int(S["captured"].sum()),
             "L_P": s["L_P"], "L_E": s["L_E"]}
        for name, p in (("E", s["pE"]), ("P", s["pP"])):
            u, z, H = M.curves(p, phid)
            e = M.ensemble_stats(u, z); l = M.latent_stats(H, cfg.K)
            r[name] = (e["gram_pr"], e["constancy"], l["pr_H"], e["tpr_Z"])
        rows.append(r)
    x = np.array([r["x"] for r in rows])
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for i, (name, ttl) in enumerate((("gram_pr", "Gram PR of the strategy ensemble"),
                                     ("constancy", "heading constancy η (1 = straight lines)"),
                                     ("pr_H", "PR(H) of the latent curve"))):
        ax = axes[i]
        for name2, col in (("E", "tab:blue"), ("P", "tab:red")):
            ax.plot(x, [r[name2][i] for r in rows], "o-", ms=3, color=col, label=name2)
        ax.set_title(ttl, fontsize=9); ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xscale("symlog", linthresh=100); ax.tick_params(labelsize=7); ax.legend(fontsize=7)
        if name == "gram_pr":
            for name2, col in (("E", "tab:blue"), ("P", "tab:red")):
                ax.plot(x, [r[name2][3] for r in rows], ":", lw=1, color=col, label=f"{name2}, paper's PR(Z)")
            ax.axhline(2, color="k", ls="--", lw=0.8); ax.legend(fontsize=7)
            ax.set_title("Gram PR of û (solid) and the paper's PR(Z) (dotted)", fontsize=9)
    ax = axes[3]
    ax.plot(x, [r["cap"] for r in rows], "o-", ms=3, color="k", label="captured heads /16")
    ax.set_title("captured heads; losses (right)", fontsize=9); ax.axvline(0, color="k", ls=":", lw=0.8)
    ax2 = ax.twinx(); ax2.plot(x, [r["L_P"] for r in rows], color="tab:red", lw=0.9, label="L_P")
    ax2.plot(x, [r["L_E"] for r in rows], color="tab:blue", lw=0.9, label="L_E"); ax2.tick_params(labelsize=7)
    ax.set_xscale("symlog", linthresh=100); ax.tick_params(labelsize=7); ax.legend(fontsize=7, loc="center left")
    ax2.legend(fontsize=7, loc="center right")
    for a in axes:
        a.set_xlabel("iteration (negative = warm-up, symlog)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default=os.path.join(ROOT, "viz", "snapshots_baseline"))
    ap.add_argument("--heads", type=int, nargs=2, default=None)
    ap.add_argument("--frame-ms", type=int, default=150, help="GIF frame duration in ms")
    ap.add_argument("--skip-static", action="store_true")
    a = ap.parse_args()
    cfg = RunConfig()
    snaps, x0 = load(a.snap)
    td = np.linspace(0, cfg.t_max, 300)
    S_final = V.simulate(snaps[-1]["pE"], snaps[-1]["pP"], x0, cfg)
    if a.heads is None:
        cap = np.where(S_final["captured"])[0]
        esc = np.where(~S_final["captured"])[0]
        hA = int(cap[np.argmin(S_final["d0"][cap])]) if len(cap) else int(np.argmin(S_final["d0"]))
        hB = int(esc[np.argmax(S_final["d0"][esc])]) if len(esc) else int(np.argmax(S_final["d0"]))
        heads = [hA, hB]
    else:
        heads = a.heads
    print(f"[anim] {len(snaps)} snapshots; pair heads {heads} "
          f"(|Δ0| = {S_final['d0'][heads[0]]:.2f}, {S_final['d0'][heads[1]]:.2f})")

    fdir = os.path.join(a.snap, "frames"); os.makedirs(fdir, exist_ok=True)
    frames, compact = [], []
    for i, s in enumerate(snaps):
        img, _ = render_frame(s, x0, cfg, heads, td)
        img.save(os.path.join(fdir, f"frame_{i:03d}.png"))
        frames.append(img)
        compact.append(render_frame_compact(s, x0, cfg, heads[0], td))
        print(f"[anim] frame {i:3d}: {label(s)}", flush=True)
    write_gif(frames, os.path.join(a.snap, "training_evolution.gif"), a.frame_ms)
    write_gif(compact, os.path.join(a.snap, f"training_evolution_compact_head{heads[0]}.gif"), a.frame_ms)
    if a.skip_static:
        return
    fig_grid(snaps, x0, cfg, td, os.path.join(a.snap, "fig_evolution_grid.png"))
    fig_pairs(snaps, x0, cfg, td, heads, os.path.join(a.snap, "fig_evolution_pairs.png"))
    fig_metrics(snaps, x0, cfg, os.path.join(a.snap, "fig_training_metrics.png"))
    print("[anim] wrote", a.snap)


if __name__ == "__main__":
    main()
