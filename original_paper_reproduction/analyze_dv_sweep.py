"""Aggregate the Delta-v sweep and render the parameter-dependence figure.

Panel (a): analytic capturable share f(R), R = Dv*t_max + eps, over the flat
torus (exact, piecewise-smooth with a curvature break at rho = R/(L/2) = 1 and
saturation at R = L/sqrt(2)); overlaid with the discrete capturable share of
the fixed 16-IC set and the realized capture fraction of the trained actor.

Panel (b): latent participation ratio PR(H_E) and strategy-ensemble PR(Z_E)
vs Dv (mean +- s.d. over 3 init seeds), with the rho=1 break marked.  Tests
whether the learned embedding tracks the game's structural transition.
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L, t_max, eps = 2.0, 4.0, 0.08
HERE = os.path.dirname(os.path.abspath(__file__))
DV_IN = os.environ.get("DV_IN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dv_sweep_results_full.jsonl"))
rows = [json.loads(l) for l in open(DV_IN)]

# aggregate by dv
dvs = sorted(set(r["dv"] for r in rows))
def agg(key, dv):
    v = [r[key] for r in rows if r["dv"] == dv]
    return np.mean(v), np.std(v)

def f_reach(R):
    if R <= 0: return 0.0
    if R >= math.sqrt(2): return 1.0
    if R <= 1.0: return math.pi*R*R/4.0
    seg = R*R*math.acos(1.0/R) - math.sqrt(R*R - 1.0)
    return (math.pi*R*R - 4.0*seg)/4.0

# analytic break points in Dv
dv_ridge = (1.0 - eps)/t_max          # R = L/2 = 1  -> rho = 1
dv_sat   = (math.sqrt(2) - eps)/t_max # R = L/sqrt(2) -> whole domain

# summary table
summary = {"budget_iters": rows[0]["iters"], "n_seeds": 3,
           "dv_ridge": round(dv_ridge,3), "dv_sat": round(dv_sat,3), "by_dv": {}}
for dv in dvs:
    le_m, le_s = agg("latPR_E", dv); ze_m, ze_s = agg("zPR_E", dv)
    lp_m, lp_s = agg("latPR_P", dv); zp_m, zp_s = agg("zPR_P", dv)
    cap_m, _   = agg("captured_frac", dv); ics_m,_ = agg("cap_share_ICs", dv)
    R = dv*t_max + eps
    summary["by_dv"][dv] = dict(R=round(R,3), rho=round(R/(L/2),3),
        latPR_E=round(le_m,2), latPR_E_sd=round(le_s,2),
        zPR_E=round(ze_m,2), zPR_E_sd=round(ze_s,2),
        latPR_P=round(lp_m,2), zPR_P=round(zp_m,2),
        captured=round(cap_m,3), cap_share_ICs=round(ics_m,3),
        f_analytic=round(f_reach(R),3))

with open(os.path.join(HERE,"dv_sweep_summary.json"),"w") as fh: json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))

# ---- figure ----
dv_arr = np.array(dvs)
Rgrid = np.linspace(0.05, dv_arr.max()*t_max+eps+0.05, 400)
dvgrid = (Rgrid - eps)/t_max
fana = np.array([f_reach(R) for R in Rgrid])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))

# panel a
ax[0].plot(dvgrid, fana, "-", color="tab:green", lw=2,
           label=r"analytic capturable share $f(R)$")
ics = [summary["by_dv"][d]["cap_share_ICs"] for d in dvs]
cap = [summary["by_dv"][d]["captured"] for d in dvs]
ax[0].plot(dv_arr, ics, "s", color="tab:gray", ms=6,
           label="capturable share, 16-IC set")
ax[0].plot(dv_arr, cap, "o", color="tab:blue", ms=6,
           label="realized capture (trained)")
ax[0].axvline(dv_ridge, color="k", ls="--", lw=1)
ax[0].axvline(dv_sat, color="k", ls=":", lw=1)
ax[0].text(dv_ridge-0.006, 0.30, r"$\rho=1$ (ridge)", rotation=90,
           va="center", ha="right", fontsize=8)
ax[0].text(dv_sat+0.012, 0.30, "saturation", rotation=90,
           va="center", ha="left", fontsize=8)
ax[0].set_xlabel(r"$\Delta v = v_P - v_E$")
ax[0].set_ylabel("capturable / captured share")
ax[0].set_title("(a) Game geometry: a break at $\\rho=1$")
ax[0].legend(fontsize=7.5, loc="lower right"); ax[0].set_ylim(-0.03, 1.05)

# panel b
le = np.array([summary["by_dv"][d]["latPR_E"] for d in dvs])
les = np.array([summary["by_dv"][d]["latPR_E_sd"] for d in dvs])
ze = np.array([summary["by_dv"][d]["zPR_E"] for d in dvs])
zes = np.array([summary["by_dv"][d]["zPR_E_sd"] for d in dvs])
ax[1].errorbar(dv_arr, le, yerr=les, fmt="o-", color="tab:red", capsize=3,
               label=r"latent $\mathrm{PR}(H_E)$ (overhead)")
ax[1].errorbar(dv_arr, ze, yerr=zes, fmt="s-", color="tab:blue", capsize=3,
               label=r"strategy $\mathrm{PR}(Z_E)$ (ensemble)")
ax[1].axhline(2.0, color="tab:blue", ls=":", lw=1,
              label="constant-heading bound $=2$")
ax[1].axvline(dv_ridge, color="k", ls="--", lw=1)
ax[1].set_xlabel(r"$\Delta v = v_P - v_E$")
ax[1].set_ylabel("participation ratio")
ax[1].set_title("(b) Embedding is flat across the break")
ax[1].legend(fontsize=7.5, loc="upper left"); ax[1].set_ylim(0, 9.5)

fig.tight_layout()
for p in [os.path.join(HERE,"fig_dv_sweep.png"),
          os.path.join(HERE,"fig_dv_sweep.png")]:
    fig.savefig(p, dpi=150, bbox_inches="tight")
print("wrote figure")
