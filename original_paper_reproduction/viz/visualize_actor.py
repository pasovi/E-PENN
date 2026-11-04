"""Visualise what a trained actor checkpoint actually does.

Rolls out the open-loop multi-head actor (evader + pursuer) on the torus
and draws, for every head (= evader initial condition):

  fig_torus_trajectories.png  both players on the fundamental domain [0,L)^2,
                              pursuer starts at the origin, with the analytic
                              saddle (straight line along the min-image
                              direction) dashed, the reach circle R, capture
                              markers and the capture time vs V*.
  fig_lifted.png              the same paths lifted to the universal cover R^2
                              (no wrap).  The grey dots are the periodic images
                              of the pursuer's start; the image the pursuer
                              actually chases is the "chart" of rev2/metrics.py.
  fig_torus3d.png             a few heads on the embedded donut.
  fig_headings.png            heading angle theta(t) of both players per head
                              against the saddle angle, plus the torus
                              distance d(t) with the capture radius eps.
  fig_latent.png              the 16 latent components H_E(t), H_P(t) and
                              their leading PCA time profiles.
  head_table.md               per-head numbers (|Delta0|, capturable, T_c,
                              V*, chart played, heading constancy before and
                              after capture).

Usage:
    python3 viz/visualize_actor.py [checkpoint.npz] [out_dir]
Default checkpoint: v2_actor.npz (the published baseline).  Also accepts the
rev2 checkpoint layout (keys BE_14000, ...).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rev2"))
from pursuit_evasion_numpy import fourier_features, forward_dirs, rollout  # noqa: E402
from core import RunConfig  # noqa: E402


# ----------------------------------------------------------------- loading
def load_actor(path: str, it: int = 14000):
    z = np.load(path)
    if "BE" in z.files:
        pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
        pP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
    else:
        pE = {"B": z[f"BE_{it}"], "b_body": z[f"bE_{it}"], "W": z[f"WE_{it}"]}
        pP = {"B": z[f"BP_{it}"], "b_body": z[f"bP_{it}"], "W": z[f"WP_{it}"]}
    K = (pE["B"].shape[1] - 1) // 2
    return pE, pP, z["x_E0"], K


def min_image(x, L):
    return (x + L / 2.0) % L - L / 2.0


# ----------------------------------------------------------------- rollout
def simulate(pE, pP, x0, cfg: RunConfig):
    tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(tg, cfg.t_max, cfg.K)
    uE, cE = forward_dirs(pE, phi)
    uP, cP = forward_dirs(pP, phi)
    xE, xP = rollout(uE, uP, x0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    dt = cfg.t_max / cfg.n_t
    # lifted paths (same increments as rollout, no wrap)
    incE = dt * cfg.v_E * 0.5 * (uE[:, :-1] + uE[:, 1:])
    incP = dt * cfg.v_P * 0.5 * (uP[:, :-1] + uP[:, 1:])
    zero = np.zeros((x0.shape[0], 1, 2))
    xE_l = x0[:, None, :] + np.concatenate([zero, np.cumsum(incE, 1)], 1)
    xP_l = np.concatenate([zero, np.cumsum(incP, 1)], 1)
    d = np.linalg.norm(min_image(xE - xP, cfg.L), axis=-1)
    captured = d.min(1) <= cfg.eps
    kc = np.where(captured, np.argmax(d <= cfg.eps, axis=1), -1)
    Tc = np.where(captured, kc * dt, np.nan)
    Delta0 = min_image(x0, cfg.L)
    d0 = np.linalg.norm(Delta0, axis=1)
    Vstar = (d0 - cfg.eps) / cfg.dv
    capturable = d0 <= cfg.reach
    chart_opt = np.rint((Delta0 - x0) / cfg.L).astype(int)
    D = xE_l - xP_l
    chart_cap = np.zeros_like(chart_opt)
    for j in np.where(captured)[0]:
        chart_cap[j] = np.rint(-D[j, kc[j]] / cfg.L).astype(int)
    return dict(tg=tg, uE=uE, uP=uP, HE=cE["H"], HP=cP["H"], xE=xE, xP=xP,
                xE_l=xE_l, xP_l=xP_l, d=d, captured=captured, kc=kc, Tc=Tc,
                Delta0=Delta0, d0=d0, Vstar=Vstar, capturable=capturable,
                chart_opt=chart_opt, chart_cap=chart_cap)


def constancy(u):
    """eta = |<u>_t|^2 per head (1 iff constant heading)."""
    ub = u.mean(axis=1)
    return (ub ** 2).sum(-1)


# ----------------------------------------------------------------- plots
def _torus_path(ax, traj, L, color, lw=1.2, label=None, **kw):
    xs, ys = traj[:, 0], traj[:, 1]
    brk = np.where((np.abs(np.diff(xs)) > L / 2) | (np.abs(np.diff(ys)) > L / 2))[0] + 1
    first = True
    for seg in np.split(np.arange(len(xs)), brk):
        ax.plot(xs[seg], ys[seg], color=color, lw=lw, label=label if first else None, **kw)
        first = False


def fig_torus(S, cfg, path):
    n_h = S["xE"].shape[0]
    cols = 4
    rows = int(np.ceil(n_h / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    axes = np.array(axes).reshape(-1)
    th = np.linspace(0, 2 * np.pi, 200)
    for j in range(n_h):
        ax = axes[j]
        # reach circle around the pursuer origin, drawn on all four corners
        for cx in (0, cfg.L):
            for cy in (0, cfg.L):
                ax.plot(cx + cfg.reach * np.cos(th), cy + cfg.reach * np.sin(th),
                        color="0.75", lw=0.8, ls=":")
        # analytic saddle: both move along Delta0-hat from their starts
        dh = S["Delta0"][j] / S["d0"][j]
        tt = np.linspace(0, cfg.t_max, 200)[:, None]
        _torus_path(ax, (S["xE"][j, 0] + cfg.v_E * tt * dh) % cfg.L, cfg.L, "tab:blue",
                    lw=0.8, ls="--", alpha=0.6)
        _torus_path(ax, (cfg.v_P * tt * dh) % cfg.L, cfg.L, "tab:red", lw=0.8, ls="--", alpha=0.6)
        # learned paths
        _torus_path(ax, S["xE"][j], cfg.L, "tab:blue", label="evader")
        _torus_path(ax, S["xP"][j], cfg.L, "tab:red", label="pursuer")
        ax.scatter(*S["xE"][j, 0], s=40, c="tab:blue", edgecolor="k", zorder=5)
        ax.scatter(0, 0, s=40, c="tab:red", marker="s", edgecolor="k", zorder=5)
        if S["captured"][j]:
            k = S["kc"][j]
            ax.scatter(*S["xP"][j, k], s=90, facecolors="none", edgecolors="k", lw=1.5, zorder=6)
        ax.set_xlim(0, cfg.L); ax.set_ylim(0, cfg.L); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        cap = "capturable" if S["capturable"][j] else "uncapturable"
        tc = f"T_c={S['Tc'][j]:.2f}, V*={S['Vstar'][j]:.2f}" if S["captured"][j] else "no capture"
        ax.set_title(f"head {j}  |Δ0|={S['d0'][j]:.2f} ({cap})\n{tc}", fontsize=8.5)
        if j == 0:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    for j in range(n_h, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Learned open-loop trajectories on T² (solid) vs the analytic constant-heading "
                 "saddle (dashed); dotted = reach circle R; ○ = capture", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_lifted(S, cfg, path):
    n_h = S["xE"].shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    # periodic images of the pursuer start and of the fundamental domain
    for m in range(-2, 3):
        for n in range(-2, 3):
            ax.scatter(m * cfg.L, n * cfg.L, s=14, c="0.6", zorder=1)
            ax.add_patch(plt.Rectangle((m * cfg.L, n * cfg.L), cfg.L, cfg.L, fill=False,
                                       ec="0.85", lw=0.6))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cfg.reach * np.cos(th), cfg.reach * np.sin(th), color="0.5", ls=":", lw=1)
    cmap = plt.get_cmap("tab20")
    for j in range(n_h):
        c = cmap(j % 20)
        ax.plot(*S["xE_l"][j].T, color=c, lw=1.4)
        ax.plot(*S["xP_l"][j].T, color=c, lw=1.0, ls="--")
        ax.scatter(*S["xE_l"][j, 0], s=30, color=c, edgecolor="k", zorder=5)
        ax.annotate(str(j), S["xE_l"][j, 0], fontsize=7, xytext=(3, 3), textcoords="offset points")
        if S["captured"][j]:
            ax.scatter(*S["xP_l"][j, S["kc"][j]], s=70, facecolors="none", edgecolors=c, lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2 * cfg.L, 2.2 * cfg.L); ax.set_ylim(-2.2 * cfg.L, 2.2 * cfg.L)
    ax.set_title("Lifted paths on the universal cover R² (solid = evader, dashed = pursuer,\n"
                 "grey dots = images of the pursuer start, ○ = capture; the image the pursuer "
                 "reaches is its 'chart')", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_torus3d(S, cfg, path, heads=None, R0=1.6, r=0.6):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    n_h = S["xE"].shape[0]
    if heads is None:
        cap = np.where(S["captured"])[0][:3].tolist()
        unc = np.where(~S["captured"])[0][:3].tolist()
        heads = cap + unc

    def emb(x):
        a = 2 * np.pi * x[..., 0] / cfg.L
        b = 2 * np.pi * x[..., 1] / cfg.L
        return np.stack([(R0 + r * np.cos(b)) * np.cos(a),
                         (R0 + r * np.cos(b)) * np.sin(a),
                         r * np.sin(b)], -1)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    A, B = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(0, 2 * np.pi, 30))
    ax.plot_surface((R0 + r * np.cos(B)) * np.cos(A), (R0 + r * np.cos(B)) * np.sin(A),
                    r * np.sin(B), color="0.9", alpha=0.25, linewidth=0, shade=True)
    ax.plot_wireframe((R0 + r * np.cos(B)) * np.cos(A), (R0 + r * np.cos(B)) * np.sin(A),
                      r * np.sin(B), color="0.7", lw=0.3, rstride=3, cstride=6, alpha=0.5)
    cmap = plt.get_cmap("tab10")
    for i, j in enumerate(heads):
        c = cmap(i % 10)
        eE = emb(S["xE"][j]); eP = emb(S["xP"][j])
        ax.plot(*eE.T, color=c, lw=2, label=f"head {j} ({'captured' if S['captured'][j] else 'escapes'})")
        ax.plot(*eP.T, color=c, lw=1.4, ls="--")
        ax.scatter(*eE[0], color=c, s=40, edgecolor="k")
    ax.scatter(*emb(np.zeros(2)), color="tab:red", marker="s", s=60, edgecolor="k", label="pursuer start")
    ax.set_box_aspect((1, 1, 0.4)); ax.set_axis_off()
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Embedded torus: evader (solid) and pursuer (dashed) for a few heads")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_headings(S, cfg, pE, pP, path):
    n_h = S["xE"].shape[0]
    td = np.linspace(0, cfg.t_max, 600)
    phid = fourier_features(td, cfg.t_max, cfg.K)
    uE, _ = forward_dirs(pE, phid); uP, _ = forward_dirs(pP, phid)
    cols = 4
    rows = int(np.ceil(n_h / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.6 * rows))
    axes = np.array(axes).reshape(-1)
    for j in range(n_h):
        ax = axes[j]
        sad = np.arctan2(S["Delta0"][j, 1], S["Delta0"][j, 0])
        thE = np.unwrap(np.arctan2(uE[j, :, 1], uE[j, :, 0]))
        thP = np.unwrap(np.arctan2(uP[j, :, 1], uP[j, :, 0]))
        # bring both into the branch nearest the saddle angle
        thE -= 2 * np.pi * np.round((thE[0] - sad) / (2 * np.pi))
        thP -= 2 * np.pi * np.round((thP[0] - sad) / (2 * np.pi))
        ax.plot(td, thE, color="tab:blue", lw=1.2, label="θ_E")
        ax.plot(td, thP, color="tab:red", lw=1.2, label="θ_P")
        ax.axhline(sad, color="k", ls="--", lw=0.8, label="saddle angle")
        ax2 = ax.twinx()
        ax2.plot(S["tg"], S["d"][j], color="0.4", lw=0.9, alpha=0.8)
        ax2.axhline(cfg.eps, color="0.4", ls=":", lw=0.8)
        ax2.set_ylim(0, 1.6); ax2.tick_params(labelsize=6, colors="0.4")
        if S["captured"][j]:
            ax.axvline(S["Tc"][j], color="g", lw=0.8, alpha=0.7)
        ax.set_ylim(sad - np.pi, sad + np.pi)
        ax.set_title(f"head {j}" + (" (captured)" if S["captured"][j] else " (escapes)"), fontsize=8.5)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=6, loc="upper left")
    for j in range(n_h, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Heading angles θ(t) (left axis) vs the saddle angle; grey = torus distance d(t) "
                 "(right axis, dotted = ε); green = capture time", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_latent(S, cfg, pE, pP, path):
    td = np.linspace(0, cfg.t_max, 1024)
    phid = fourier_features(td, cfg.t_max, cfg.K)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
    for row, (p, name) in enumerate(((pE, "evader"), (pP, "pursuer"))):
        _, c = forward_dirs(p, phid)
        H = c["H"]
        Hc = H - H.mean(0)
        C = Hc.T @ Hc / H.shape[0]
        ev, V = np.linalg.eigh(C); ev, V = ev[::-1], V[:, ::-1]
        pr = ev.sum() ** 2 / (ev ** 2).sum()
        ax = axes[row, 0]
        ax.plot(td, H, lw=0.8)
        ax.set_title(f"{name}: the {H.shape[1]} latent components H(t)", fontsize=10)
        ax.set_xlabel("t")
        ax = axes[row, 1]
        ax.plot(td, Hc @ V[:, :4], lw=1.2)
        ax.legend([f"PC{k+1}" for k in range(4)], fontsize=7)
        ax.set_title(f"{name}: leading PCA time profiles (time-centred), PR(H)={pr:.2f}", fontsize=10)
        ax.set_xlabel("t")
        ax = axes[row, 2]
        ax.bar(np.arange(1, len(ev) + 1), ev / ev.sum(), color="0.5")
        ax.set_title(f"{name}: PCA eigenvalue spectrum of Cov_t[H]", fontsize=10)
        ax.set_xlabel("PC index"); ax.set_ylabel("fraction of variance")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def head_table(S, cfg, path):
    n_h = S["xE"].shape[0]
    lines = ["| head | |Δ0| | capturable | captured | T_c | V* | chart opt | chart played | "
             "η_E pre | η_E post | η_P pre | η_P post |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in range(n_h):
        k = S["kc"][j] if S["captured"][j] else cfg.n_t
        def eta(u, a, b):
            return float((u[j, a:b].mean(0) ** 2).sum()) if b - a >= 2 else float("nan")
        lines.append(
            f"| {j} | {S['d0'][j]:.2f} | {'yes' if S['capturable'][j] else 'no'} | "
            f"{'yes' if S['captured'][j] else 'no'} | "
            f"{S['Tc'][j]:.2f} | {S['Vstar'][j]:.2f} | {tuple(S['chart_opt'][j])} | "
            f"{tuple(S['chart_cap'][j]) if S['captured'][j] else '-'} | "
            f"{eta(S['uE'], 0, k + 1):.3f} | {eta(S['uE'], k, cfg.n_t + 1):.3f} | "
            f"{eta(S['uP'], 0, k + 1):.3f} | {eta(S['uP'], k, cfg.n_t + 1):.3f} |")
    txt = "\n".join(lines)
    with open(path, "w") as f:
        f.write("η = |<u>_t|² over the window (1 = perfectly constant heading). "
                "'chart' = integer image of the pursuer start, as in rev2/metrics.py.\n\n" + txt + "\n")
    print(txt)


def main(ckpt, out):
    os.makedirs(out, exist_ok=True)
    pE, pP, x0, K = load_actor(ckpt)
    cfg = RunConfig(K=K)
    S = simulate(pE, pP, x0, cfg)
    print(f"checkpoint {ckpt}: {x0.shape[0]} heads, K={K}, reach R={cfg.reach:.2f}; "
          f"capturable {int(S['capturable'].sum())}, captured {int(S['captured'].sum())}")
    fig_torus(S, cfg, os.path.join(out, "fig_torus_trajectories.png"))
    fig_lifted(S, cfg, os.path.join(out, "fig_lifted.png"))
    fig_torus3d(S, cfg, os.path.join(out, "fig_torus3d.png"))
    fig_headings(S, cfg, pE, pP, os.path.join(out, "fig_headings.png"))
    fig_latent(S, cfg, pE, pP, os.path.join(out, "fig_latent.png"))
    head_table(S, cfg, os.path.join(out, "head_table.md"))
    print("wrote figures to", out)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "v2_actor.npz")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "viz", "baseline")
    main(ckpt, out)
