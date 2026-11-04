"""Gate: rev2 metrics must reproduce the published legacy numbers on the
published baseline checkpoint (v2_actor.npz) before any new run is analysed.

Usage: python3 rev2/test_metrics.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import ROOT, RunConfig, initial_state  # noqa: E402
import metrics as M  # noqa: E402

z = np.load(os.path.join(ROOT, "v2_actor.npz"))
pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
pP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
x0 = z["x_E0"]
cfg = RunConfig()
m = M.full_metrics(pE, pP, x0, cfg)
g = m["game"]

# (value computed here, published value, tolerance, source)
checks = {
    "PR(H_E)":       (m["E"]["lat"]["pr_H"], 6.748, 5e-3, "embedding_summary.json"),
    "PR(H_P)":       (m["P"]["lat"]["pr_H"], 7.080, 5e-3, "embedding_summary.json"),
    "PR(Z_E)":       (m["E"]["ens"]["tpr_Z"], 2.987, 5e-3, "c1_seeds.json s0"),
    "PR(Z_P)":       (m["P"]["ens"]["tpr_Z"], 1.915, 5e-3, "c1_seeds.json s0"),
    "erank_gf E":    (m["E"]["heads"]["erank_gf"], 2.53, 1e-2, "paper Fig. 5"),
    "erank_gf P":    (m["P"]["heads"]["erank_gf"], 1.91, 1e-2, "paper Fig. 5"),
    "sv_raw_max E":  (m["E"]["heads"]["sv_raw_max"], 5.58, 1e-2, "paper Fig. 5"),
    "sv_raw_max P":  (m["P"]["heads"]["sv_raw_max"], 18.07, 1e-2, "paper Fig. 5"),
    "band11-50 U_E": (m["E"]["band"]["band_11_50_U"], 0.158, 1e-3, "paper Sec. 7"),
    "band11-50 U_P": (m["P"]["band"]["band_11_50_U"], 0.303, 1e-3, "paper Sec. 7"),
    "frac<=K H_E":   (m["E"]["lat"]["frac_le_K_H"], 0.9838, 2e-3, "c3_break_analysis.json"),
    "n capturable":  (g["n_capturable"], 4, 0, "pr_capturable_scrutiny.py"),
    "n winding cls": (len({tuple(np.rint(w).astype(int)) for w in g["wind_E"]}), 12, 0, "paper Sec. 5"),
}
ok_all = True
for k, (val, ref, tol, src) in checks.items():
    ok = abs(val - ref) <= tol
    ok_all &= ok
    print(f"{'OK  ' if ok else 'FAIL'} {k:15s} rev2={val:9.4f}  published={ref:9.4f}  [{src}]")

# internal consistency of the best-response computation
Tc = np.array([np.nan if v is None else v for v in g["Tc"]])
Tbr = np.array([np.nan if v is None else v for v in g["T_BR"]])
both = ~np.isnan(Tc)
assert np.all(~np.isnan(Tbr[both])), "BR must capture wherever the trained pursuer does"
assert np.all(Tbr[both] <= Tc[both] + 1e-12), "T_BR must not exceed Tc"
print("OK   T_BR <= Tc on every captured head")

# Proposition 1 sanity: constant headings give gram_pr == saddle PR (<= 2)
_, phid = M.dense_phi(cfg)
dh = M.min_image(x0, cfg.L); dh /= np.linalg.norm(dh, axis=1, keepdims=True)
u_const = np.repeat(dh[:, None, :], phid.shape[0], axis=1)
e = M.ensemble_stats(u_const, u_const)
ref = M.saddle_reference(x0, cfg)["gram_pr_saddle"]
assert abs(e["gram_pr"] - ref) < 1e-10 and e["gram_pr"] <= 2 and abs(e["constancy"] - 1) < 1e-12
print(f"OK   constant saddle headings: gram_pr={e['gram_pr']:.4f} (<=2), constancy=1")

# the gauge dependence of the legacy statistic: time-dependent per-head
# rescaling leaves the policy unchanged but moves tpr_Z
u, zz, _ = M.curves(pP, phid)
c = np.exp(np.sin(np.linspace(0, 6, phid.shape[0])))[None, :, None] * np.ones((u.shape[0], 1, 1))
e0, e1 = M.ensemble_stats(u, zz), M.ensemble_stats(u, zz * c)
print(f"INFO gauge test (pursuer): tpr_Z {e0['tpr_Z']:.3f} -> {e1['tpr_Z']:.3f} under z->c(t)z; "
      f"tpr_U {e0['tpr_U']:.3f} -> {e1['tpr_U']:.3f}; gram_pr {e0['gram_pr']:.3f} -> {e1['gram_pr']:.3f}")

# reproduce the baseline's initial state from (init_seed, ic_seed) = (0, 0)
_, _, x0r = initial_state(cfg)
assert np.array_equal(x0r, x0), "IC set of (0,0) must equal the published baseline ICs"
print("OK   (init_seed, ic_seed)=(0,0) regenerates the published IC set")

print("\nnew (corrected) statistics on the published baseline:")
for a in "EP":
    e = m[a]["ens"]
    print(f"  {a}: gram_pr={e['gram_pr']:.3f} gram_pr_hc={e['gram_pr_hc']:.3f} constancy={e['constancy']:.3f} "
          f"tpr_U={e['tpr_U']:.3f} tpr_Z={e['tpr_Z']:.3f} | pr_H={m[a]['lat']['pr_H']:.3f} pr_H_unc={m[a]['lat']['pr_H_unc']:.3f}")
print("  saddle reference:", M.saddle_reference(x0, cfg))
print(f"  captured {g['n_captured_capturable']}/{g['n_capturable']} capturable, "
      f"{g['n_captured_uncapturable']} uncapturable; BR pursuer captures {g['n_br_captured']}/16 "
      f"({g['n_br_captured_uncapturable']} uncapturable)")
sys.exit(0 if ok_all else 1)
