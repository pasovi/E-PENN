"""The r1_dimensionalities figure for any grid of runs in the rev2 record format.

Same panel as checks/analyze_r1.py's Fig R1-a: Gram PR, temporal PR of u, temporal PR of z,
PR(H) uncentred and PR(H) centred, evader and pursuer, every run as a dot with median and
IQR, the constant-heading bound and the analytic saddle's Gram PR.  The diamonds mark a
reference: by default the medians of the draft-loss grid (checks/results/r1).

Usage: python3 torch_actor/checks/dimensionalities.py [tag=r1_trunc] [ref_tag=r1] [out_name]
Writes checks/figs/<out_name>.png/.pdf (default <tag>_dimensionalities).
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C_E, C_P, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.6})
CATS = [("gram_pr", "Gram PR\n(Prop. 1)"), ("tpr_U", "temporal PR\nof $u$"), ("tpr_Z", "temporal PR\nof $z$ (legacy)"),
        ("pr_H_unc", "PR$(H)$\nuncentred"), ("pr_H", "PR$(H)$\ncentred (legacy)")]
LABEL = {"r1": "draft loss", "r1_trunc": "capture-truncated loss", "r1_trunc_w0_p3": "capture-truncated loss",
         "r1_trunc_w0_p1_e3": "capture-truncated loss"}


def load(tag):
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "results", tag, "grid_*.json")))]
    G = {}
    for a in "EP":
        for k, _ in CATS:
            src = "lat" if k.startswith("pr_H") else "ens"
            G[f"{a}.{k}"] = np.array([r["full"]["14000"][a][src][k] for r in recs])
    G["gram_pr_saddle"] = np.array([r["saddle"]["gram_pr_saddle"] for r in recs])
    G["warmup"] = recs[0]["config"]["warmup"]; G["steps"] = recs[0]["config"]["n_P_steps"]
    G["esteps"] = recs[0]["config"].get("n_E_steps", 1)
    G["truncate"] = recs[0].get("truncate", False)
    return G


def strip(ax, x, vals, color, label=None, w=0.28):
    rng = np.random.default_rng(int(1000 * x) % 2 ** 31)
    vals = np.asarray(vals)[np.isfinite(vals)]
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    ax.scatter(x + rng.uniform(-w / 2, w / 2, len(vals)), vals, s=9, color=color, alpha=0.45, linewidths=0, label=label, zorder=2)
    ax.plot([x - w / 1.6, x + w / 1.6], [med, med], color=INK, lw=1.4, zorder=3)
    ax.plot([x, x], [q1, q3], color=INK, lw=0.9, zorder=3)


def main(tag="r1_trunc", ref_tag="r1", out=None):
    G, R = load(tag), (load(ref_tag) if ref_tag else None)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for i, (k, _) in enumerate(CATS):
        for a, col, off in (("E", C_E, -0.19), ("P", C_P, 0.19)):
            strip(ax, i + off, G[f"{a}.{k}"], col, label=({"E": "evader", "P": "pursuer"}[a] if i == 0 else None))
            if R is not None:
                ax.scatter(i + off, np.median(R[f"{a}.{k}"]), marker="D", s=22, facecolor="white", edgecolor=INK, lw=0.9,
                           zorder=4, label=(f"{LABEL.get(ref_tag, ref_tag)} grid, median" if (i == 0 and a == "E") else None))
    ax.plot([-0.45, 0.45], [2, 2], color=MUTED, lw=1.0, zorder=1, label="bound for constant headings (2)")
    ax.plot([-0.45, 0.45], [np.median(G["gram_pr_saddle"])] * 2, color=C_3, lw=1.2, zorder=1,
            label="analytic saddle (median over IC sets)")
    ax.axvline(2.5, color=AXIS, lw=0.8)
    ax.text(1.0, 0.99, "strategy ensemble", transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK2, fontsize=8)
    ax.text(3.5, 0.99, "latent curve", transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK2, fontsize=8)
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels([c[1] for c in CATS])
    ax.set_ylabel("participation ratio"); ax.set_xlim(-0.6, len(CATS) - 0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), ncol=1)
    proto = f"warm-up {G['warmup']}, {G['steps']}:{G['esteps']}"
    ax.set_title(f"Ensemble and latent dimensionality over {len(G['E.gram_pr'])} runs (10 init seeds x 10 IC sets), "
                 f"{LABEL.get(tag, tag)}, {proto}", loc="left", fontsize=8.5)
    out = out or f"{tag}_dimensionalities"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"{out}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(HERE, "figs", out + ".png/.pdf"))


if __name__ == "__main__":
    main(*sys.argv[1:])
