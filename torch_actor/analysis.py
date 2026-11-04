"""The draft's statistics (Table 1, Sections 3-7) recomputed on the torch checkpoints.

Formulas follow the original analysis scripts (revision/analyze_e1e7.py,
revision/post_hoc_checks.py, run_c4c5.py): dense grid of 1,024 time points,
time-centred PR of H and of the stacked head outputs Z, integer winding classes,
unit-heading mid-band power fraction k in [11, 50], gauge-fixed head-matrix PR,
integer-k latent power with the 1% break index.

Writes paper/results/table1.json.  `figures.py` imports the functions.
Usage: python3 torch_actor/analysis.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from numpy_reference import fourier_features, forward_dirs, rollout, init_params  # noqa: E402

PAPER = os.path.join(HERE, "paper")
CKPT = os.path.join(PAPER, "ckpt")
RES = os.path.join(PAPER, "results")
R1 = os.path.join(HERE, "checks", "ckpt", "r1")
T_MAX, L, V_E, EPS, N_T, T = 4.0, 2.0, 1.0, 0.08, 64, 1024
td = np.linspace(0, T_MAX, T)
tg = np.linspace(0, T_MAX, N_T + 1)


# ------------------------------------------------------------------ basics
def PR(ev):
    ev = np.maximum(np.asarray(ev, float), 0)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300))


def cov_eigs(X):
    Xc = X - X.mean(0, keepdims=True)
    return np.linalg.eigvalsh(Xc.T @ Xc / X.shape[0])[::-1]


def pcs_for(ev, thr):
    return int(np.searchsorted(np.cumsum(ev) / ev.sum(), thr) + 1)


def unit(z):
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


def load(path, it=None):
    """(pE, pP, x0) in the original key names; it=None -> final parameters."""
    z = np.load(path)
    if it is None:
        pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
        pP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
    else:
        pE = {"B": z[f"BE_{it}"], "b_body": z[f"bE_{it}"], "W": z[f"WE_{it}"]}
        pP = {"B": z[f"BP_{it}"], "b_body": z[f"bP_{it}"], "W": z[f"WP_{it}"]}
    return pE, pP, z["x_E0"]


def K_of(p):
    return (p["B"].shape[1] - 1) // 2


def int_k_power(H):
    Hc = H - H.mean(0, keepdims=True)
    return (np.abs(np.fft.rfft(Hc, axis=0)) ** 2).sum(1) / Hc.shape[0] ** 2


def break_idx(pk, frac=1e-2):
    plateau = pk[1:9].mean()
    for k in range(1, len(pk)):
        if pk[k] < frac * plateau and np.all(pk[k:k + 5] < frac * plateau):
            return k
    return len(pk)


def gauge_fixed_svs(W):
    Wg = np.stack([w / np.linalg.norm(w) for w in W]).reshape(-1, W.shape[2])
    return np.linalg.svd(Wg, compute_uv=False)


def winding(pE, K):
    phig = fourier_features(tg, T_MAX, K)
    uEg, _ = forward_dirs(pE, phig)
    dt = T_MAX / N_T
    return dt * V_E * (0.5 * (uEg[:, :-1] + uEg[:, 1:])).sum(1) / L


# ------------------------------------------------------------------- suite
def suite(pE, pP, x0, v_P=1.2):
    """analyze_e1e7.full_suite plus spectra, windings and head singular values."""
    K = K_of(pE)
    phid = fourier_features(td, T_MAX, K)
    phig = fourier_features(tg, T_MAX, K)
    n_h = pE["W"].shape[0]
    _, cE = forward_dirs(pE, phid); HE = cE["H"]
    _, cP = forward_dirs(pP, phid); HP = cP["H"]
    zE = np.einsum("hab,tb->tha", pE["W"], HE)
    zP = np.einsum("hab,tb->tha", pP["W"], HP)
    evHE, evHP = cov_eigs(HE), cov_eigs(HP)
    evZE, evZP = cov_eigs(zE.reshape(T, -1)), cov_eigs(zP.reshape(T, -1))
    uEg, _ = forward_dirs(pE, phig); uPg, _ = forward_dirs(pP, phig)
    xE, xP = rollout(uEg, uPg, x0, L, V_E, v_P, T_MAX, N_T)
    d = np.linalg.norm((xE - xP + L / 2) % L - L / 2, axis=-1)
    wE = winding(pE, K)
    sec = [tuple(s) for s in np.round(wE).astype(int)]

    def bandfrac(z):
        X = unit(z).reshape(T, -1); X = X - X.mean(0, keepdims=True)
        P = (np.abs(np.fft.rfft(X, axis=0)) ** 2).sum(1)
        return float(P[11:51].sum() / P[1:].sum())

    pkE, pkP = int_k_power(HE), int_k_power(HP)
    sE = np.linalg.svd(pE["W"].reshape(2 * n_h, -1), compute_uv=False)
    sP = np.linalg.svd(pP["W"].reshape(2 * n_h, -1), compute_uv=False)
    gE, gP = gauge_fixed_svs(pE["W"]), gauge_fixed_svs(pP["W"])
    return {
        "K": K,
        "PR_HE": PR(evHE), "PR_HP": PR(evHP),
        "pc90_HE": pcs_for(evHE, .90), "pc90_HP": pcs_for(evHP, .90),
        "PR_ZE": PR(evZE), "PR_ZP": PR(evZP),
        "pc90_ZE": pcs_for(evZE, .90), "pc90_ZP": pcs_for(evZP, .90),
        "PR_UE": PR(cov_eigs(unit(zE).reshape(T, -1))), "PR_UP": PR(cov_eigs(unit(zP).reshape(T, -1))),
        "n_sectors": len(set(sec)), "sectors": [str(s) for s in sec],
        "winding_E": wE.tolist(),
        "n_captured": int((d.min(1) <= EPS).sum()), "n_h": n_h,
        "midband_frac_uE": bandfrac(zE), "midband_frac_uP": bandfrac(zP),
        "sv_raw_E": sE.tolist(), "sv_raw_P": sP.tolist(),
        "svPR_raw_E": PR(sE ** 2), "svPR_raw_P": PR(sP ** 2),
        "sv_gf_E": gE.tolist(), "sv_gf_P": gP.tolist(),
        "gf_svPR_E": PR(gE ** 2), "gf_svPR_P": PR(gP ** 2),
        "kpow_E": pkE[:129].tolist(), "kpow_P": pkP[:129].tolist(),
        "break_E": break_idx(pkE), "break_P": break_idx(pkP),
        "frac_le_K_E": float(pkE[1:K + 1].sum() / pkE[1:].sum()),
        "frac_le_K_P": float(pkP[1:K + 1].sum() / pkP[1:].sum()),
        "drop_K_to_K4_E": float(pkE[K] / max(pkE[K + 4], 1e-300)),
        "drop_K_to_K4_P": float(pkP[K] / max(pkP[K + 4], 1e-300)),
    }


def init_suite(seed=0, K=12, n_h=16, n_b=16):
    rng = np.random.default_rng(seed)
    iE = init_params(n_h, n_b, 2 * K + 1, rng)
    iP = init_params(n_h, n_b, 2 * K + 1, rng)
    phid = fourier_features(td, T_MAX, K)
    _, c0 = forward_dirs(iE, phid); H0 = c0["H"]
    z0 = np.einsum("hab,tb->tha", iE["W"], H0)
    return {"PR_H": PR(cov_eigs(H0)), "PR_Z": PR(cov_eigs(z0.reshape(T, -1))),
            "sv_raw_E": np.linalg.svd(iE["W"].reshape(2 * n_h, n_b), compute_uv=False).tolist(),
            "sv_raw_P": np.linalg.svd(iP["W"].reshape(2 * n_h, n_b), compute_uv=False).tolist(),
            "sv_gf_E": gauge_fixed_svs(iE["W"]).tolist(),
            "W_E": iE["W"], "W_P": iP["W"], "B_E": iE["B"], "B_P": iP["B"]}


def ms(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def ckpts(arm, pattern):
    return sorted(glob.glob(os.path.join(CKPT, arm, pattern)))


# -------------------------------------------------------------------- main
def main():
    OUT = {}
    base = suite(*load(os.path.join(CKPT, "baseline", "grid_i0_c0.npz")))
    ini = init_suite(0)
    OUT["baseline"] = {k: v for k, v in base.items() if not k.startswith(("sv_", "kpow", "winding", "sectors"))}
    OUT["init_seed0"] = {"PR_H": ini["PR_H"], "PR_Z": ini["PR_Z"]}
    OUT["sec7_asymmetry_baseline"] = {
        "midband_P": base["midband_frac_uP"], "midband_E": base["midband_frac_uE"],
        "ratio_P_over_E": base["midband_frac_uP"] / base["midband_frac_uE"],
        "sv_max_raw_E": base["sv_raw_E"][0], "sv_max_raw_P": base["sv_raw_P"][0],
        "erank_gf_E": base["gf_svPR_E"], "erank_gf_P": base["gf_svPR_P"]}

    # C1: init seeds at IC 0, IC resamples at init 0 (from the R1 grid, 14k, same protocol)
    s_init = [suite(*load(os.path.join(R1, f"grid_i{i}_c0.npz"))) for i in range(8)]
    s_ic = [suite(*load(os.path.join(R1, f"grid_i0_c{c}.npz"))) for c in range(1, 6)]
    OUT["C1"] = {"PR_HE_init_seeds": ms([s["PR_HE"] for s in s_init]),
                 "PR_HE_ic_resamples": ms([s["PR_HE"] for s in s_ic]),
                 "variance_ratio_init_over_ic": float(np.var([s["PR_HE"] for s in s_init], ddof=1) /
                                                      np.var([s["PR_HE"] for s in s_ic], ddof=1)),
                 "PR_ZE_init_seeds": ms([s["PR_ZE"] for s in s_init]),
                 "midband_ratio_init_seeds": ms([s["midband_frac_uP"] / s["midband_frac_uE"] for s in s_init]),
                 "PR_HE_over_PR_ZE_min_E": float(min(s["PR_HE"] / s["PR_ZE"] for s in s_init)),
                 "PR_HP_over_PR_ZP_min_P": float(min(s["PR_HP"] / s["PR_ZP"] for s in s_init))}
    # all 100 grid runs: population ranges
    allg = [suite(*load(f)) for f in sorted(glob.glob(os.path.join(R1, "grid_*.npz")))]
    OUT["grid100"] = {"PR_ZE": ms([s["PR_ZE"] for s in allg]), "PR_ZP": ms([s["PR_ZP"] for s in allg]),
                      "PR_HE": ms([s["PR_HE"] for s in allg]), "PR_HP": ms([s["PR_HP"] for s in allg]),
                      "ratio_min_E": float(min(s["PR_HE"] / s["PR_ZE"] for s in allg)),
                      "ratio_min_P": float(min(s["PR_HP"] / s["PR_ZP"] for s in allg)),
                      "n_sectors": ms([s["n_sectors"] for s in allg]),
                      "midband_ratio": ms([s["midband_frac_uP"] / s["midband_frac_uE"] for s in allg])}

    # C2: symmetrised protocol, snapshots 20k..28k, 7 seeds
    c2 = {}
    for f in ckpts("c2", "c2_s*.npz"):
        name = os.path.basename(f)[:-4]
        gaps, prs = {}, {}
        for it in (20000, 22000, 24000, 26000, 28000):
            s = suite(*load(f, it))
            gaps[it] = s["midband_frac_uP"] / s["midband_frac_uE"]
            prs[it] = (s["PR_ZE"], s["PR_ZP"], s["PR_HE"], s["PR_HP"], s["sv_raw_E"][0], s["sv_raw_P"][0])
        c2[name] = {"gap_by_snapshot": gaps, "gap_run_mean": float(np.mean(list(gaps.values()))),
                    "PR_ZE_ZP_HE_HP_svE_svP_by_snapshot": prs}
    allgaps = [g for r in c2.values() for g in r["gap_by_snapshot"].values()]
    OUT["C2"] = {"runs": c2, "run_means": sorted(r["gap_run_mean"] for r in c2.values()),
                 "snapshot_range": [float(min(allgaps)), float(max(allgaps))],
                 "frac_snapshots_below_1": float(np.mean(np.array(allgaps) < 1))}

    # C3: bandwidth sweep
    c3 = {}
    for K in (6, 12, 24):
        files = ckpts(f"c3_k{K}", "*.npz") if K != 12 else [os.path.join(R1, f"grid_i{i}_c0.npz") for i in range(10)]
        ss = [suite(*load(f)) for f in files]
        c3[K] = {"break_E_seed0": ss[0]["break_E"], "break_P_seed0": ss[0]["break_P"],
                 "break_E": ms([s["break_E"] for s in ss]), "break_P": ms([s["break_P"] for s in ss]),
                 "frac_le_K_E": ms([s["frac_le_K_E"] for s in ss]), "frac_le_K_P": ms([s["frac_le_K_P"] for s in ss]),
                 "PR_HE": ms([s["PR_HE"] for s in ss]), "PR_ZE": ms([s["PR_ZE"] for s in ss])}
    OUT["C3"] = c3

    # C4, C5, E1, E1b, E1b reflected (variants batch), E7
    var = {}
    for v in ("c4", "c5_lam0", "c5_lam1", "e1", "e1b", "e1b_refl"):
        ss = [suite(*load(f)) for f in ckpts("variants", f"{v}_s*.npz")]
        var[v] = {"seed0": {k: ss[0][k] for k in ("PR_HE", "PR_HP", "PR_ZE", "PR_ZP", "n_sectors", "n_captured",
                                                 "midband_frac_uE", "midband_frac_uP")},
                  "PR_HE": ms([s["PR_HE"] for s in ss]), "PR_ZE": ms([s["PR_ZE"] for s in ss]),
                  "PR_ZP": ms([s["PR_ZP"] for s in ss]), "n_sectors": ms([s["n_sectors"] for s in ss]),
                  "mean_wx_seed0": float(np.mean(np.array(ss[0]["winding_E"])[:, 0])),
                  "heads_wx_pos_seed0": int((np.array(ss[0]["winding_E"])[:, 0] > 0).sum())}
    var["c5_lam5"] = {"seed0": {k: base[k] for k in ("PR_HE", "PR_HP", "PR_ZE", "PR_ZP", "n_sectors", "n_captured",
                                                     "midband_frac_uE", "midband_frac_uP")},
                      "PR_HE": OUT["C1"]["PR_HE_init_seeds"], "PR_ZE": OUT["C1"]["PR_ZE_init_seeds"]}
    OUT["variants"] = var
    e7 = {}
    for it in (4000, 14000):
        ss = [suite(*load(f, it)) for f in ckpts("e7", "e7_s*.npz")]
        e7[it] = {"seed0": {k: ss[0][k] for k in ("PR_HE", "PR_ZE", "PR_ZP", "midband_frac_uE", "midband_frac_uP")},
                  "midband_ratio": ms([s["midband_frac_uP"] / s["midband_frac_uE"] for s in ss]),
                  "PR_ZE": ms([s["PR_ZE"] for s in ss]), "PR_HE": ms([s["PR_HE"] for s in ss])}
    OUT["E7"] = e7

    # C7: planted rank, collapse trajectory (tpr_Z from the 250-iteration snapshots)
    fit = [suite(*load(f)) for f in ckpts("c7", "c7_fit_s*.npz")]
    rec = json.load(open(os.path.join(RES, "c7.json")))
    traj = {a: np.array([[s[a]["tpr_Z"] for s in r["snapshots"]] for r in rec["runs"]]) for a in "EP"}
    its = [s["it"] for s in rec["runs"][0]["snapshots"]]
    game = [suite(*load(f)) for f in ckpts("c7", "c7_game_s*.npz")]
    OUT["C7"] = {"fit_PR_ZE": ms([s["PR_ZE"] for s in fit]), "fit_PR_ZP": ms([s["PR_ZP"] for s in fit]),
                 "fit_PR_UE": ms([s["PR_UE"] for s in fit]),
                 "final_PR_ZE": ms([s["PR_ZE"] for s in game]), "final_PR_ZP": ms([s["PR_ZP"] for s in game]),
                 "final_PR_HE": ms([s["PR_HE"] for s in game]),
                 "PR_ZE_at_iters": {str(i): ms(traj["E"][:, k]) for k, i in enumerate(its) if i in (0, 250, 1000, 3000, 8000, 14000)},
                 "PR_ZP_at_iters": {str(i): ms(traj["P"][:, k]) for k, i in enumerate(its) if i in (0, 250, 1000, 3000, 8000, 14000)},
                 "first_iter_PR_ZE_below_3_median": int(its[int(np.argmax(np.median(traj["E"], 0) < 3))])}

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "table1.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=float)
    print(json.dumps({k: OUT[k] for k in ("baseline", "C1", "C3", "E7", "C7")}, indent=1, default=float)[:6000])
    print("wrote", os.path.join(RES, "table1.json"))


if __name__ == "__main__":
    main()
