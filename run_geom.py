"""Phased driver for the three-geometry honest training."""
from __future__ import annotations
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

import geometries as geo
import honest_critic_geom as hc

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
CKPTS = {"disk":   OUT + "geom_disk.npz",
          "ball":   OUT + "geom_ball.npz",
          "mobius": OUT + "geom_mobius.npz"}


def make_geom(name: str):
    if name == "disk":
        return geo.Disk(R=1.0, n_freqs=4, t_max=4.0)
    if name == "ball":
        return geo.Ball3D(R=1.0, t_max=4.0)
    if name == "mobius":
        return geo.Mobius(L=2.0, w=0.5, n_freqs=4, t_max=4.0)
    raise ValueError(name)


def cfg_for(name: str, n_iters: int) -> hc.GeomTrainConfig:
    return hc.GeomTrainConfig(
        v_E=1.0, v_P=1.2, eps=0.1,
        hidden=96 if name != "mobius" else 128,
        n_iters=n_iters,
        n_res=192 if name != "ball" else 160,
        n_term=48, n_cap=48,
        lr_max=3e-3, lr_min=1e-4,
        schedule_total=15000,
        lam_term=5.0, lam_cap=5.0,
        log_every=200, seed=0,
    )


def phase_train(name: str, resume: bool, n_iters: int):
    G = make_geom(name)
    cfg = cfg_for(name, n_iters)
    return hc.train_geom(G, cfg, CKPTS[name], resume=resume)


def _load(name: str):
    npz = np.load(CKPTS[name])
    params = {k: npz[k] for k in ("B1","b1","B2","b2","w","c")}
    iters = int(npz["iter0"])
    hist = json.loads(str(npz["hist"]))
    return params, iters, hist


def plot_loss_curves(out_path: str):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, name in enumerate(("disk", "ball", "mobius")):
        if not os.path.exists(CKPTS[name]):
            ax[j].text(0.5, 0.5, "(no checkpoint)", ha="center", va="center")
            ax[j].set_title(name.capitalize()); continue
        _, _, hist = _load(name)
        it = np.array(hist["iter"])
        ax[j].semilogy(it, hist["loss"], color="black", lw=1.4, label="total")
        ax[j].semilogy(it, hist["loss_hji"], color="tab:red", lw=1.0,
                        label="HJI")
        ax[j].semilogy(it, hist["loss_term"], color="tab:blue", lw=1.0,
                        label="term BC")
        ax[j].semilogy(it, hist["loss_cap"], color="tab:green", lw=1.0,
                        label="cap BC")
        ax[j].axhline(1e-3, color="grey", ls="--", lw=0.6)
        ax[j].set_xlabel("iteration"); ax[j].set_ylabel("loss")
        ax[j].set_title({"disk":  "Disk ($\\mathbb{D}^2_{R=1}$)",
                          "ball":  "3D Ball ($\\mathbb{B}^3_{R=1}$)",
                          "mobius":"Möbius strip"}[name])
        ax[j].legend(fontsize=8, loc="upper right")
        ax[j].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_disk_V(out_path: str):
    """V on a 2D slice for the disk (fix x_P at origin, vary x_E)."""
    if not os.path.exists(CKPTS["disk"]): return
    params, _, _ = _load("disk")
    G = make_geom("disk")
    n_grid = 70
    xs = np.linspace(-0.95, 0.95, n_grid)
    ys = np.linspace(-0.95, 0.95, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)
    inside = np.linalg.norm(pts, axis=-1) < 0.95
    xP = np.zeros_like(pts)
    tau = np.full(pts.shape[0], G.t_max)
    phi = G.encode_state(pts, xP, tau)
    V, _ = hc.forward(params, phi)
    V_pred = V.reshape(n_grid, n_grid)
    V_pred[~inside.reshape(n_grid, n_grid)] = np.nan

    # Reference baseline: pure-eikonal value V_ref = |x_E|/(v_P-v_E)
    # (assumes evader never uses the boundary)
    V_ref = (np.linalg.norm(pts, axis=-1) - 0.1).clip(min=0) / 0.2
    V_ref = V_ref.reshape(n_grid, n_grid)
    V_ref_plot = V_ref.copy()
    V_ref_plot[~inside.reshape(n_grid, n_grid)] = np.nan

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for axj, data, title in zip(ax, [V_pred, V_ref_plot, V_pred - V_ref_plot],
                                  ["Honest $V_{NN}(x_E, x_P\\!=\\!0)$",
                                   "Pure-eikonal $|x_E|/(v_P-v_E)$",
                                   "Difference (network minus eikonal)"]):
        im = axj.imshow(data.T, origin="lower",
                         extent=[xs[0], xs[-1], ys[0], ys[-1]],
                         cmap="viridis" if "Difference" not in title else "RdBu_r")
        plt.colorbar(im, ax=axj)
        axj.set_title(title); axj.set_xlabel("$x_E$ (x)"); axj.set_ylabel("$x_E$ (y)")
        circ = plt.Circle((0, 0), 1.0, color="black", fill=False, lw=1)
        axj.add_patch(circ)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_mobius_V(out_path: str):
    """V on a (u, v) slice for the Möbius strip, fixing x_P at (0, 0)."""
    if not os.path.exists(CKPTS["mobius"]): return
    params, _, _ = _load("mobius")
    G = make_geom("mobius")
    n_u, n_v = 60, 30
    us = np.linspace(-G.L/2 + 0.02, G.L/2 - 0.02, n_u)
    vs = np.linspace(-G.w + 0.02, G.w - 0.02, n_v)
    U, V = np.meshgrid(us, vs, indexing="ij")
    pts = np.stack([U.flatten(), V.flatten()], axis=1)
    xP = np.zeros_like(pts)
    tau = np.full(pts.shape[0], G.t_max)
    phi = G.encode_state(pts, xP, tau)
    Vp, _ = hc.forward(params, phi)
    Vp = Vp.reshape(n_u, n_v)

    # Reference: torus-distance-based (no flip)
    dE = G.distance(pts, xP).reshape(n_u, n_v)
    Vref = (dE - 0.1).clip(min=0) / 0.2

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for axj, data, title in zip(ax,
                                  [Vp, Vref, Vp - Vref],
                                  ["Honest $V_{NN}$",
                                   "Möbius distance $/(v_P-v_E)$",
                                   "Difference"]):
        im = axj.imshow(data.T, origin="lower",
                         extent=[us[0], us[-1], vs[0], vs[-1]],
                         cmap="viridis" if "Diff" not in title else "RdBu_r",
                         aspect="auto")
        plt.colorbar(im, ax=axj); axj.set_xlabel("u"); axj.set_ylabel("v")
        axj.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_ball_V(out_path: str):
    """V on the equatorial slice of the 3-ball, fixing x_P at (0, 0, 0)."""
    if not os.path.exists(CKPTS["ball"]): return
    params, _, _ = _load("ball")
    G = make_geom("ball")
    n_grid = 60
    xs = np.linspace(-0.95, 0.95, n_grid)
    ys = np.linspace(-0.95, 0.95, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.zeros_like(X)
    pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    inside = np.linalg.norm(pts, axis=-1) < 0.95
    xP = np.zeros_like(pts)
    tau = np.full(pts.shape[0], G.t_max)
    phi = G.encode_state(pts, xP, tau)
    Vp, _ = hc.forward(params, phi)
    Vp = Vp.reshape(n_grid, n_grid)
    Vp[~inside.reshape(n_grid, n_grid)] = np.nan
    V_ref = (np.linalg.norm(pts, axis=-1) - 0.1).clip(min=0) / 0.2
    V_ref = V_ref.reshape(n_grid, n_grid)
    V_ref[~inside.reshape(n_grid, n_grid)] = np.nan

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for axj, data, title in zip(ax, [Vp, V_ref, Vp - V_ref],
        ["Honest $V_{NN}$ on $z\\!=\\!0$ slice",
         "Eikonal baseline $|x_E|/(v_P-v_E)$",
         "Difference"]):
        im = axj.imshow(data.T, origin="lower",
                         extent=[xs[0], xs[-1], ys[0], ys[-1]],
                         cmap="viridis" if "Diff" not in title else "RdBu_r")
        plt.colorbar(im, ax=axj)
        axj.set_xlabel("$x_E$ (x)"); axj.set_ylabel("$x_E$ (y)")
        axj.set_title(title)
        circ = plt.Circle((0, 0), 1.0, color="black", fill=False, lw=1)
        axj.add_patch(circ)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)


def phase_P():
    plot_loss_curves(OUT + "fig_geom_losses.png")
    plot_disk_V(OUT + "fig_geom_disk_V.png")
    plot_ball_V(OUT + "fig_geom_ball_V.png")
    plot_mobius_V(OUT + "fig_geom_mobius_V.png")
    # summary JSON
    summary = {}
    for name in ("disk", "ball", "mobius"):
        if not os.path.exists(CKPTS[name]): continue
        _, iters, hist = _load(name)
        summary[name] = {
            "n_train_iter": iters,
            "min_loss":      float(min(hist["loss"])),
            "final_loss":    float(hist["loss"][-1]),
            "final_loss_hji": float(hist["loss_hji"][-1]),
            "final_mean_gE":  float(hist["mean_gE"][-1]),
            "final_mean_gP":  float(hist["mean_gP"][-1]),
        }
    summary["target_grad_norm_steady"] = 1.0 / (1.2 - 1.0)
    with open(OUT + "geom_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    args = sys.argv[1:] or ["P"]
    for a in args:
        if a in ("disk", "ball", "mobius"):
            phase_train(a, resume=os.path.exists(CKPTS[a]), n_iters=2500)
        elif a == "P":
            phase_P()
        else:
            print("unknown:", a)
