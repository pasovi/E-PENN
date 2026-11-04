"""Reconstruct fig_embed_sectors.png (sector experiments) from data.

Left:  latent PR(H_E) (blue) vs control-ensemble PR(Z_E) (green) for the baseline
       and two sector-restricted runs, with the constant-heading linear-dimension
       bound (= 2, Prop. "constant curves are linearly cheap") dotted.
Right: per-head net winding w = dispE/L in R^2 for the baseline (12 integer
       classes, circles) vs the one-sided restriction E1b (3 classes, triangles).

PR values are read from revision/e1e7_analysis.json; the winding is recomputed
from the actors with the exact pipeline of revision/analyze_e1e7.py.
Run from the bundle root:  python3 make_fig_sectors.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pursuit_evasion_numpy as pen

HERE = os.path.dirname(os.path.abspath(__file__))
L, V_E, T_MAX, NF, N_T = 2.0, 1.0, 4.0, 12, 64
BOUND = 2.0  # constant-heading bound: dim span{constant R^2 curves} = 2


def winding(npz):
    """Per-head net evader winding w = dispE / L in R^2 (shape n_h x 2)."""
    z = np.load(os.path.join(HERE, npz))
    pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
    phig = pen.fourier_features(np.linspace(0, T_MAX, N_T + 1), T_MAX, NF)
    uEg, _ = pen.forward_dirs(pE, phig)
    dt = T_MAX / N_T
    dispE = dt * V_E * (0.5 * (uEg[:, :-1] + uEg[:, 1:])).sum(1)
    return dispE / L


ana = json.load(open(os.path.join(HERE, "revision", "e1e7_analysis.json")))
runs = [("baseline_v2", "baseline"), ("E1", "E1 angular"), ("E1b", "E1b one-sided")]
PR_H = [ana[k]["PR_HE"] for k, _ in runs]
PR_Z = [ana[k]["PR_ZE"] for k, _ in runs]
labels = [f"{lab}\n({ana[k]['n_sectors']} classes)" for k, lab in runs]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)

# ---- left: participation ratios ----
x = np.arange(len(runs)); bw = 0.38
axL.bar(x - bw / 2, PR_H, bw, color="#1f77b4", label=r"latent curve  $\mathrm{PR}(\mathbf{H}_E)$")
axL.bar(x + bw / 2, PR_Z, bw, color="#2ca02c", label=r"control ensemble  $\mathrm{PR}(Z_E)$")
axL.axhline(BOUND, ls=":", c="k", lw=1.6, label="constant-heading bound (=2)")
for xi, h, zv in zip(x, PR_H, PR_Z):
    axL.text(xi - bw / 2, h + 0.08, f"{h:.1f}", ha="center", fontsize=8)
    axL.text(xi + bw / 2, zv + 0.08, f"{zv:.1f}", ha="center", fontsize=8)
axL.set_xticks(x); axL.set_xticklabels(labels)
axL.set_ylabel("participation ratio")
axL.set_title("Sector restriction: PR falls far less than class count")
axL.legend(fontsize=9, frameon=False); axL.grid(axis="y", alpha=0.25)
axL.set_ylim(0, max(PR_H) + 1.0)

# ---- right: per-head winding clusters ----
wb, we = winding("v2_actor.npz"), winding("e1b_actor.npz")
nb = len({tuple(np.round(p).astype(int)) for p in wb})
ne = len({tuple(np.round(p).astype(int)) for p in we})
axR.axhline(0, c="0.85", lw=0.8); axR.axvline(0, c="0.85", lw=0.8)
axR.scatter(wb[:, 0], wb[:, 1], s=80, color="#1f77b4", label="baseline ICs")
axR.scatter(we[:, 0], we[:, 1], s=80, marker="^", color="#ff7f0e", label="E1b one-sided ICs")
axR.set_xlabel(r"evader winding $w_x$"); axR.set_ylabel(r"evader winding $w_y$")
axR.set_title("Per-head net winding (integer cells = classes)")
axR.legend(fontsize=9, frameon=False); axR.grid(alpha=0.25)
axR.set_aspect("equal", "datalim")

out = os.path.join(HERE, "fig_embed_sectors.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}  (baseline {nb} classes, E1b {ne} classes)")
