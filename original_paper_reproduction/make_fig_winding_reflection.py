"""Reconstruct fig_winding_reflection.png (reflection control) from data.

Per-head evader winding w = dispE/L in R^2 for the one-sided +x sector (E1b,
circles) and the same construction reflected to the -x sector and retrained
identically (E1b-reflected, triangles). The cloud mirrors about w_x = 0: the
asymmetry tracks the deliberately one-sided initial conditions and the signed
winding does not wrap.

Winding is recomputed from the actors (e1b_actor.npz, e1b_refl_actor.npz) with
the exact pipeline of revision/analyze_e1e7.py.
Run from the bundle root:  python3 make_fig_winding_reflection.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pursuit_evasion_numpy as pen

HERE = os.path.dirname(os.path.abspath(__file__))
L, V_E, T_MAX, NF, N_T = 2.0, 1.0, 4.0, 12, 64


def winding(npz):
    z = np.load(os.path.join(HERE, npz))
    pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
    phig = pen.fourier_features(np.linspace(0, T_MAX, N_T + 1), T_MAX, NF)
    uEg, _ = pen.forward_dirs(pE, phig)
    dt = T_MAX / N_T
    return dt * V_E * (0.5 * (uEg[:, :-1] + uEg[:, 1:])).sum(1) / L


w1 = winding("e1b_actor.npz")
w2 = winding("e1b_refl_actor.npz")

fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
ax.axvline(0, ls="--", c="k", lw=1.3)
ax.axhline(0, c="0.85", lw=0.8)
ax.scatter(w1[:, 0], w1[:, 1], s=85, color="#1f77b4", label=r"$+x$ ICs (E1b)")
ax.scatter(w2[:, 0], w2[:, 1], s=85, marker="^", color="#ff7f0e", label=r"$-x$ ICs (reflected)")
ax.set_xlabel(r"evader winding $w_x$"); ax.set_ylabel(r"evader winding $w_y$")
ax.legend(fontsize=10, frameon=True, loc="upper center", ncol=2); ax.grid(alpha=0.25)
ax.set_aspect("equal", "datalim")
lim = np.abs(np.concatenate([w1, w2])).max() * 1.15
ax.set_xlim(-lim, lim)

out = os.path.join(HERE, "fig_winding_reflection.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}  (+x mean wx={w1[:,0].mean():+.2f}, reflected mean wx={w2[:,0].mean():+.2f})")
