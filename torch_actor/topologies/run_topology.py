"""Baseline D1 pipeline on S^1 or T^3: multi-head (MH) training, then transfer learning (TL) of new
heads on the frozen bodies, with the figures used for T^2.

Usage:  python3 torch_actor/topologies/run_topology.py --dim 1 --n-heads 16 --tl-grid 100
        python3 torch_actor/topologies/run_topology.py --dim 3 --n-heads 64 --tl-grid 10 --device mps --dtype float32
(--tl-grid n: n^dim cell-centred new ICs.)  Outputs under topologies/{results,figs}/<name>_... with
name = S1 or T3:
  results/<name>_mh.json, _mh.npz          training heads: game and PR statistics, PR evolution, params
  results/<name>_mh_snapshots.npz          parameter snapshots every --snapshot-every iterations + loss history (video)
  results/<name>_tl.json, _tl.npz          TL heads
  figs/<name>_game_mh.pdf, _game_tl.pdf    capture time vs |Delta0| against V*, capture fractions
  figs/<name>_pr.pdf                       PR statistics of the MH heads and of the TL heads
  figs/<name>_pr_evolution.pdf             PR statistics of the MH run against the training iteration
  figs/<name>_latent.pdf                   latent components H(t), leading PCA profiles, spectrum (MH bodies)
  figs/<name>_tl_map.pdf                   outcome per new IC (circle for S^1, 3D scatter for T^3)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe_nd as nd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figs")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
C_E, C_P = "#2a78d6", "#eb6834"
NAMES = {1: "S1", 2: "T2", 3: "T3"}
PRETTY = {1: "$S^1$", 2: "$T^2$", 3: "$T^3$"}


# ------------------------------------------------------------------ figures
def figure_game(g, cfg, path, label, who="heads"):
    d0 = np.array(g["d0"]); Tc = np.array([np.nan if t is None else t for t in g["Tc"]])
    Tbr = np.array([np.nan if t is None else t for t in g["T_BR"]])
    cap, capt = np.array(g["capturable"]), np.array(g["captured"])
    s_pt = 16 if len(d0) <= 200 else 5
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.scatter(d0, Tc, s=s_pt, color=C_P, alpha=0.75, label=f"{who}: $T_c$", linewidths=0)
    ax.scatter(d0, Tbr, s=s_pt, color=C_E, alpha=0.75, label="best response $T_{BR}$", linewidths=0)
    dv = cfg.v_P - cfg.v_E; R = g["reach"]
    xx = np.linspace(cfg.eps, R, 50); ax.plot(xx, (xx - cfg.eps) / dv, "k-", lw=1.2, label=r"$V^*(\Delta_0)$")
    ax.axvline(R, color="0.5", lw=1); ax.text(R + 0.02, 1.7, "reach R", fontsize=8, color="0.4")
    ax.set_ylim(0, cfg.t_max + 0.1); ax.set_xlim(0, max(1.45, d0.max() + 0.1))
    ax.set_xlabel(r"initial geodesic distance $|\Delta_0|$"); ax.set_ylabel("capture time")
    ax.set_title(f"{label}: capture times", fontsize=9.5); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    fr = [capt[cap].mean() if cap.any() else 0, (~np.isnan(Tbr[cap])).mean() if cap.any() else 0,
          capt[~cap].mean() if (~cap).any() else 0, (~np.isnan(Tbr[~cap])).mean() if (~cap).any() else 0]
    x = np.array([0, 0.38, 1.1, 1.48])
    bars = ax.bar(x, fr, width=0.36, color=[C_P, C_E, C_P, C_E])
    for b, v in zip(bars, fr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{100 * v:.1f}%", ha="center", fontsize=8)
    ax.set_xticks([0.19, 1.29]); ax.set_xticklabels([f"capturable\n(n={int(cap.sum())})", f"uncapturable\n(n={int((~cap).sum())})"])
    ax.set_ylim(0, 1.12); ax.set_ylabel("fraction of heads captured"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("trained pursuer (orange) vs best response (blue)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


PR_KEYS = [("gram_pr", "Gram PR"), ("tpr_U", "temporal PR of $u$"), ("constancy", "constancy $\\eta$"),
           ("pr_H_unc", "PR($H$) uncentred"), ("pr_H", "PR($H$) centred")]


def figure_pr(mh: dict, tl: dict, dim: int, path: str, label: str):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.19
    for i, (k, _) in enumerate(PR_KEYS):
        for j, (src, name, hatch) in enumerate(((mh, "MH", ""), (tl, "TL", "//"))):
            for a, col in (("E", C_E), ("P", C_P)):
                x = i + (-1.5 + 2 * j + (0 if a == "E" else 1)) * w
                v = src[f"{k}_{a}"]
                ax.bar(x, v, width=w, color=col, alpha=0.95 if name == "MH" else 0.5, hatch=hatch, edgecolor="white",
                       label=(f"{name} {'evader' if a == 'E' else 'pursuer'}" if i == 0 else None))
                ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
    ax.axhline(dim, color="0.4", ls=":", lw=1, label=f"constant-heading bound ({dim})")
    ax.set_xticks(range(len(PR_KEYS))); ax.set_xticklabels([n for _, n in PR_KEYS])
    ax.set_ylabel("value"); ax.set_ylim(0, max(8, 1.25 * max(max(mh[f"{k}_{a}"], tl[f"{k}_{a}"]) for k, _ in PR_KEYS for a in "EP")))
    ax.set_title(f"{label}: ensemble and latent statistics of the MH training heads and of the TL heads", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def figure_pr_evolution(ev: list, dim: int, path: str, label: str):
    its = np.array([e["iteration"] for e in ev])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, keys, title in ((axes[0], [("gram_pr", "Gram PR"), ("tpr_U", "temporal PR of $u$")], "strategy ensemble"),
                            (axes[1], [("pr_H", "PR($H$) centred"), ("pr_H_unc", "PR($H$) uncentred")], "latent curve"),
                            (axes[2], [("constancy", "constancy $\\eta$")], "heading constancy")):
        for k, name in keys:
            for a, col in (("E", C_E), ("P", C_P)):
                ax.plot(its, [e[a][k] for e in ev], color=col, ls="-" if k in ("gram_pr", "pr_H", "constancy") else "--",
                        lw=1.4, label=f"{name}, {'E' if a == 'E' else 'P'}")
        if title != "heading constancy":
            ax.axhline(dim, color="0.4", ls=":", lw=1, label=f"bound ({dim})")
        ax.set_xlabel("self-play iteration"); ax.set_title(f"{label}: {title}", fontsize=9.5)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def figure_latent(pE, pP, cfg, path, label):
    td = np.linspace(0, cfg.t_max, 1024); phid = nd.fourier_np(td, cfg.t_max, cfg.K)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
    for row, (p, name) in enumerate(((pE, "evader"), (pP, "pursuer"))):
        _, _, H = nd.curves_np(p, phid)
        Hc = H - H.mean(0); C = Hc.T @ Hc / H.shape[0]
        ev, V = np.linalg.eigh(C); ev, V = ev[::-1], V[:, ::-1]
        pr = ev.sum() ** 2 / (ev ** 2).sum()
        ax = axes[row, 0]; ax.plot(td, H, lw=0.8); ax.set_xlabel("t")
        ax.set_title(f"{label} {name}: the {H.shape[1]} latent components H(t)", fontsize=10)
        ax = axes[row, 1]; ax.plot(td, Hc @ V[:, :4], lw=1.2); ax.legend([f"PC{k + 1}" for k in range(4)], fontsize=7); ax.set_xlabel("t")
        ax.set_title(f"{name}: leading PCA time profiles (time-centred), PR(H)={pr:.2f}", fontsize=10)
        ax = axes[row, 2]; ax.bar(np.arange(1, len(ev) + 1), ev / ev.sum(), color="0.5")
        ax.set_title(f"{name}: PCA eigenvalue spectrum of Cov_t[H]", fontsize=10); ax.set_xlabel("PC index"); ax.set_ylabel("fraction of variance")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def figure_tl_map(x0, g, cfg, dim, path, label):
    L, R = cfg.L, g["reach"]
    cap, capt = np.array(g["capturable"]), np.array(g["captured"])
    wrong, wrong2 = cap & ~capt, ~cap & capt
    if dim == 1:
        th = 2 * np.pi * x0[:, 0] / L
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.plot(np.cos(np.linspace(0, 2 * np.pi, 400)), np.sin(np.linspace(0, 2 * np.pi, 400)), color="0.8", lw=1)
        a = 2 * np.pi * R / L
        ax.plot(1.08 * np.cos(np.linspace(-a, a, 200)), 1.08 * np.sin(np.linspace(-a, a, 200)), color="0.4", lw=2.5, ls=":",
                label=f"capturable arc (geodesic distance <= R = {R:.2f})")
        ax.plot([-1.15, -1.0], [0, 0], color="0.6", lw=1, ls="--", label="cut locus (antipode, x = L/2)")
        ax.scatter(np.cos(th[capt]), np.sin(th[capt]), s=60, color="#d62728", zorder=3, label=f"captured ({int(capt.sum())})")
        ax.scatter(np.cos(th[~capt]), np.sin(th[~capt]), s=60, color="#2ca02c", zorder=3, label=f"escaped ({int((~capt).sum())})")
        if wrong.any():
            ax.scatter(np.cos(th[wrong]), np.sin(th[wrong]), s=150, facecolor="none", edgecolor="k", lw=1.3, zorder=4,
                       label=f"capturable but escaped ({int(wrong.sum())})")
        if wrong2.any():
            ax.scatter(np.cos(th[wrong2]), np.sin(th[wrong2]), s=150, facecolor="none", edgecolor="k", lw=1.3, ls="--", zorder=4,
                       label=f"uncapturable but captured ({int(wrong2.sum())})")
        ax.plot(1, 0, "s", color="k", ms=9, zorder=5, label="pursuer start (x = 0)")
        ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.axis("off")
        ax.set_title(f"{label}: outcome per new IC on $S^1$ (angle = 2$\\pi$ x / L)", fontsize=10)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    else:
        fig = plt.figure(figsize=(12, 5.5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(*x0[capt].T, s=10, color="#d62728", alpha=0.8, label=f"captured ({int(capt.sum())})", depthshade=False)
        ax.scatter(*x0[~capt].T, s=4, color="#2ca02c", alpha=0.25, label=f"escaped ({int((~capt).sum())})", depthshade=False)
        if wrong.any():
            ax.scatter(*x0[wrong].T, s=50, facecolor="none", edgecolor="k", lw=1, label=f"capturable but escaped ({int(wrong.sum())})", depthshade=False)
        if wrong2.any():
            ax.scatter(*x0[wrong2].T, s=50, facecolor="none", edgecolor="b", lw=1, label=f"uncapturable but captured ({int(wrong2.sum())})", depthshade=False)
        for c in np.array(np.meshgrid([0, L], [0, L], [0, L])).reshape(3, -1).T:
            ax.scatter(*c, s=40, color="k", marker="s")
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"{label}: outcome per new IC on $T^3$ (squares = pursuer start and images)", fontsize=9.5)
        ax.legend(fontsize=7.5, loc="upper left")
        ax = fig.add_subplot(1, 2, 2)
        d0 = np.array(g["d0"])
        bins = np.linspace(0, d0.max() + 0.01, 25)
        ax.hist(d0[capt], bins=bins, color="#d62728", alpha=0.8, label="captured")
        ax.hist(d0[~capt], bins=bins, color="#2ca02c", alpha=0.6, label="escaped")
        ax.axvline(R, color="0.3", ls=":", lw=1.5, label=f"reach R = {R:.2f}")
        ax.set_xlabel(r"initial geodesic distance $|\Delta_0|$"); ax.set_ylabel("new ICs"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title("the same ICs by geodesic distance to the pursuer", fontsize=9.5)
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--tl-grid", type=int, default=10, help="new ICs per axis for TL (n^dim heads)")
    ap.add_argument("--init-seed", type=int, default=0); ap.add_argument("--ic-seed", type=int, default=0)
    ap.add_argument("--tl-seed", type=int, default=0)
    ap.add_argument("--snapshot-every", type=int, default=100)
    ap.add_argument("--pr-every", type=int, default=500, help="iterations between PR-evolution points")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--dtype", default="float64")
    ap.add_argument("--tl-device", default=None); ap.add_argument("--tl-dtype", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--warmup", type=int, default=None, help="pursuer-only warm-up steps (default 1500)")
    a = ap.parse_args()
    dim, name = a.dim, a.name or NAMES[a.dim]
    label = PRETTY[dim]
    dtype = {"float64": torch.float64, "float32": torch.float32}[a.dtype]
    tl_dev, tl_dt = a.tl_device or a.device, {"float64": torch.float64, "float32": torch.float32}[a.tl_dtype or a.dtype]
    cfg = nd.Config(truncate=True, n_h=a.n_heads)
    if a.warmup is not None:
        cfg = replace(cfg, warmup=a.warmup)

    # ---------------------------------------------------------------- MH training
    pE0, pP0, x0 = nd.initial_state(cfg, dim, a.init_seed, a.ic_seed)
    print(f"[{name}] MH: {cfg.n_h} heads on T^{dim}, warm-up {cfg.warmup}, {cfg.n_iters} x ({cfg.n_P_steps} P + {cfg.n_E_steps} E), "
          f"lr {cfg.lr_P} / {cfg.lr_E}, truncated loss, {a.device} {a.dtype}", flush=True)
    pE, pP, hist, snaps, secs = nd.train(cfg, pE0, pP0, x0, a.device, dtype, snapshot_every=a.snapshot_every, log_every=2000)
    print(f"[{name}] MH trained in {secs:.0f} s", flush=True)
    g = nd.game_stats(pE, pP, x0, cfg)
    prE, prP = nd.pr_stats(pE, cfg), nd.pr_stats(pP, cfg)
    mh_summary = nd.summarise(g, prE, prP); mh_summary["seconds"] = secs
    evol = []
    for lab, gstep, sE, sP in snaps:
        if lab.startswith("self-play") or lab in ("init",) or lab.startswith("warm-up step " + str(cfg.warmup)):
            it = int(lab.split()[2]) if lab.startswith("self-play") else 0
            if it % a.pr_every == 0:
                evol.append({"iteration": it, "label": lab, "E": {k: v for k, v in nd.pr_stats(sE, cfg, 256).items() if not k.startswith("eig")},
                             "P": {k: v for k, v in nd.pr_stats(sP, cfg, 256).items() if not k.startswith("eig")}})
    json.dump({"name": name, "dim": dim, "config": asdict(cfg), "init_seed": a.init_seed, "ic_seed": a.ic_seed, "x0": x0.tolist(),
               "summary": mh_summary, "game": g, "pr_E": prE, "pr_P": prP, "pr_evolution": evol, "device": f"{a.device} {a.dtype}"},
              open(os.path.join(RES, f"{name}_mh.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(RES, f"{name}_mh.npz"), x_E0=x0, **{k + "E": v for k, v in pE.items()}, **{k + "P": v for k, v in pP.items()})
    np.savez_compressed(os.path.join(RES, f"{name}_mh_snapshots.npz"), x_E0=x0, labels=np.array([s[0] for s in snaps]),
                        snap_step=np.array([s[1] for s in snaps]), loss_hist=hist,
                        **{k + "E": np.stack([s[2][k] for s in snaps]) for k in pE}, **{k + "P": np.stack([s[3][k] for s in snaps]) for k in pP})
    figure_game(g, cfg, os.path.join(FIG, f"{name}_game_mh.pdf"), f"{label} MH, {cfg.n_h} training heads", who="trained heads")
    figure_pr_evolution(evol, dim, os.path.join(FIG, f"{name}_pr_evolution.pdf"), f"{label} MH ({cfg.n_h} heads)")
    figure_latent(pE, pP, cfg, os.path.join(FIG, f"{name}_latent.pdf"), label)
    print(f"[{name}] MH summary: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in mh_summary.items()}), flush=True)

    # ---------------------------------------------------------------- TL
    x0_new = nd.grid_ICs(a.tl_grid, cfg.L, dim)
    rng = np.random.default_rng(a.tl_seed)
    pE_t = {"B": pE["B"], "b": pE["b"], "W": rng.standard_normal((len(x0_new), dim, cfg.n_b)) * np.sqrt(1.0 / cfg.n_b)}
    pP_t = {"B": pP["B"], "b": pP["b"], "W": rng.standard_normal((len(x0_new), dim, cfg.n_b)) * np.sqrt(1.0 / cfg.n_b)}
    print(f"[{name}] TL: {len(x0_new)} new heads on the frozen bodies, same protocol, {tl_dev} {tl_dt}", flush=True)
    pE_t, pP_t, _, _, secs_t = nd.train(cfg, pE_t, pP_t, x0_new, tl_dev, tl_dt, frozen_body=True, log_every=2000)
    assert np.allclose(pE_t["B"], pE["B"]) and np.allclose(pP_t["b"], pP["b"])
    print(f"[{name}] TL trained in {secs_t:.0f} s", flush=True)
    g_t = nd.game_stats(pE_t, pP_t, x0_new, cfg)
    prE_t, prP_t = nd.pr_stats(pE_t, cfg), nd.pr_stats(pP_t, cfg)
    tl_summary = nd.summarise(g_t, prE_t, prP_t); tl_summary["seconds"] = secs_t
    json.dump({"name": name, "dim": dim, "config": asdict(cfg), "tl_seed": a.tl_seed, "x0_new": x0_new.tolist(),
               "summary": tl_summary, "game": {k: v for k, v in g_t.items() if k not in ("xE", "xP")}, "pr_E": prE_t, "pr_P": prP_t,
               "device": f"{tl_dev} {tl_dt}"}, open(os.path.join(RES, f"{name}_tl.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(RES, f"{name}_tl.npz"), x_E0=x0_new, WE=pE_t["W"], WP=pP_t["W"], BE=pE["B"], bE=pE["b"], BP=pP["B"], bP=pP["b"])
    figure_game(g_t, cfg, os.path.join(FIG, f"{name}_game_tl.pdf"), f"{label} TL, {len(x0_new)} new heads on the frozen bodies", who="TL heads")
    figure_pr(mh_summary, tl_summary, dim, os.path.join(FIG, f"{name}_pr.pdf"), label)
    figure_tl_map(x0_new, g_t, cfg, dim, os.path.join(FIG, f"{name}_tl_map.pdf"), f"{label} TL")
    print(f"[{name}] TL summary: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in tl_summary.items()}), flush=True)
    print("wrote", os.path.join(RES, name + "_*"), "and", os.path.join(FIG, name + "_*.pdf"))


if __name__ == "__main__":
    main()
