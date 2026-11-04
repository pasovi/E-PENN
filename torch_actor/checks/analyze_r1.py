"""Analysis of R1 (baseline ensemble). Reads checks/results/r1/*.json and
checks/ckpt/r1/*.npz; writes

  checks/results/r1_summary.json    every number quoted from R1
  checks/results/r1_census.md       census tables (median [IQR] (min-max))
  checks/figs/r1_*.png / .pdf       figures

Usage: python3 torch_actor/checks/analyze_r1.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import RunConfig, ROOT, load_ckpt  # noqa: E402
import metrics as M  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

# ------------------------------------------------------------------ style
C_E, C_P, C_3 = "#2a78d6", "#eb6834", "#1baf7a"     # evader, pursuer, third
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "lines.linewidth": 1.6,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


# ------------------------------------------------------------------- load
def load_runs():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "r1", "*.json")))]
    return runs


def scalars(rec, it="14000"):
    f = rec["full"][it]
    g = f["game"]
    out = {}
    cap = np.array(g["capturable"])
    for a in "EP":
        e, l, b, h = f[a]["ens"], f[a]["lat"], f[a]["band"], f[a]["heads"]
        for k in ("gram_pr", "gram_pr_hc", "constancy", "tpr_U", "tpr_Z"):
            out[f"{a}.{k}"] = e[k]
        for k in ("pr_H", "pr_H_unc", "frac_le_K_H"):
            out[f"{a}.{k}"] = l[k]
        out[f"{a}.band_11_50_U"] = b["band_11_50_U"]
        out[f"{a}.erank_gf"] = h["erank_gf"]
        out[f"{a}.sv_raw_max"] = h["sv_raw_max"]
        ec, eu = f[a]["ens_cap"], f[a]["ens_uncap"]
        out[f"{a}.gram_pr_cap"] = ec.get("gram_pr", np.nan)
        out[f"{a}.gram_pr_uncap"] = eu.get("gram_pr", np.nan)
        out[f"{a}.constancy_cap"] = ec.get("constancy", np.nan)
        out[f"{a}.constancy_uncap"] = eu.get("constancy", np.nan)
        al = np.array(f[a]["static_align"])
        out[f"{a}.align_cap"] = float(al[cap].mean()) if cap.any() else np.nan
        out[f"{a}.align_uncap"] = float(al[~cap].mean()) if (~cap).any() else np.nan
        out[f"{a}.ratio_legacy"] = l["pr_H"] / e["tpr_Z"]
        out[f"{a}.ratio_U"] = l["pr_H"] / e["tpr_U"]
        out[f"{a}.ratio_unc"] = l["pr_H_unc"] / e["gram_pr"]
    out["band_ratio_PE"] = f["P"]["band"]["band_11_50_U"] / f["E"]["band"]["band_11_50_U"]
    out["sv_ratio_PE"] = f["P"]["heads"]["sv_raw_max"] / f["E"]["heads"]["sv_raw_max"]
    out["n_capturable"] = g["n_capturable"]
    out["n_captured_capturable"] = g["n_captured_capturable"]
    out["frac_captured_capturable"] = g["n_captured_capturable"] / max(g["n_capturable"], 1)
    out["n_captured_uncapturable"] = g["n_captured_uncapturable"]
    out["n_br_captured"] = g["n_br_captured"]
    out["n_br_captured_uncapturable"] = g["n_br_captured_uncapturable"]
    out["pursuer_regret_mean"] = g["pursuer_regret_mean"]
    out["gram_pr_saddle"] = rec["saddle"]["gram_pr_saddle"]
    out["gram_pr_saddle_cap"] = (rec["saddle"]["gram_pr_saddle_cap"]
                                 if rec["saddle"]["gram_pr_saddle_cap"] is not None else np.nan)
    return out


def table(rows, keys):
    return {k: np.array([r[k] for r in rows], dtype=float) for k in keys}


def describe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    q = np.percentile(x, [0, 25, 50, 75, 100])
    return {"n": int(len(x)), "median": q[2], "q1": q[1], "q3": q[3], "min": q[0], "max": q[4],
            "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def fmt(d, nd=2):
    if d.get("n", 0) == 0:
        return "-"
    return (f"{d['median']:.{nd}f} [{d['q1']:.{nd}f}, {d['q3']:.{nd}f}] "
            f"({d['min']:.{nd}f}-{d['max']:.{nd}f})")


# --------------------------------------------------------- variance comps
def two_way(Y):
    """Two-way random-effects ANOVA without replication. Y: (n_init, n_ic)."""
    a, b = Y.shape
    m = Y.mean()
    ri, cj = Y.mean(1), Y.mean(0)
    ss_i = b * ((ri - m) ** 2).sum()
    ss_c = a * ((cj - m) ** 2).sum()
    res = Y - ri[:, None] - cj[None, :] + m
    ss_r = (res ** 2).sum()
    df_i, df_c, df_r = a - 1, b - 1, (a - 1) * (b - 1)
    ms_i, ms_c, ms_r = ss_i / df_i, ss_c / df_c, ss_r / df_r
    return {
        "F_init": ms_i / ms_r, "p_init": float(stats.f.sf(ms_i / ms_r, df_i, df_r)),
        "F_ic": ms_c / ms_r, "p_ic": float(stats.f.sf(ms_c / ms_r, df_c, df_r)),
        "var_init": max(0.0, (ms_i - ms_r) / b), "var_ic": max(0.0, (ms_c - ms_r) / a),
        "var_res": ms_r, "var_total": float(Y.var(ddof=1)),
        "ic_means": cj.tolist(), "init_means": ri.tolist(),
    }


# ------------------------------------------------------- topology helpers
def head_curves(name, cfg):
    pE, pP, x0 = load_ckpt(os.path.join(HERE, "ckpt", "r1", name + ".npz"))
    tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phig = M.fourier_features(tg, cfg.t_max, cfg.K)
    uE, _ = M.forward_dirs(pE, phig)
    uP, _ = M.forward_dirs(pP, phig)
    return uE, uP, x0


def class_ratio(u, labels):
    """mean within-class / mean across-class L2 distance of heading curves
    (sqrt of time-mean squared difference, as in the paper's Sec. 5)."""
    D = np.sqrt(((u[:, None] - u[None]) ** 2).sum(-1).mean(-1))
    n = len(labels)
    iu = np.triu_indices(n, 1)
    same = np.array([[labels[i] == labels[j] for j in range(n)] for i in range(n)])[iu]
    d = D[iu]
    return d[same], d[~same]


# =================================================================== main
def main():
    runs = load_runs()
    by = defaultdict(list)
    for r in runs:
        by[r["arm"]].append(r)
    print({k: len(v) for k, v in by.items()})
    S = {"n_runs": {k: len(v) for k, v in by.items()}}
    cfg = RunConfig()

    keys = list(scalars(runs[0]).keys())
    tabs = {arm: table([scalars(r) for r in rs], keys) for arm, rs in by.items()}

    # ---------------------------------------------------------- census
    S["census"] = {arm: {k: describe(v) for k, v in t.items()} for arm, t in tabs.items()}
    pub = {"E.pr_H": 6.748, "P.pr_H": 7.080, "E.tpr_Z": 2.987, "P.tpr_Z": 1.915,
           "E.gram_pr": 2.543, "P.gram_pr": 2.477, "E.tpr_U": 2.849, "P.tpr_U": 3.145,
           "E.erank_gf": 2.53, "P.erank_gf": 1.91, "band_ratio_PE": 1.915,
           "n_capturable": 4, "E.pr_H_unc": 4.214, "P.pr_H_unc": 4.015}
    G = tabs["grid"]
    S["published_percentile"] = {k: float(stats.percentileofscore(G[k][np.isfinite(G[k])], v))
                                 for k, v in pub.items()}

    # within-run separation: latent vs ensemble, every grid run, both players
    # Both ratios compare time-indexed, time-centred spectra (temporal modes of the
    # latent vs temporal modes of the head outputs), the comparison the rank chain
    # is about.  The Gram PR is head-indexed and is not compared with PR(H).
    S["separation"] = {}
    for a in "EP":
        for num, den in (("pr_H", "tpr_Z"), ("pr_H", "tpr_U")):
            allr = np.concatenate([tabs[arm][f"{a}.{num}"] / tabs[arm][f"{a}.{den}"] for arm in tabs])
            S["separation"][f"{a}.{num}/{den}"] = {"min_all_arms": float(allr.min()),
                                                   "median_grid": float(np.median(G[f"{a}.{num}"] / G[f"{a}.{den}"])),
                                                   "frac_gt1": float((allr > 1).mean())}
    # population overlap (the paper's "the two populations do not overlap")
    for a in "EP":
        S["separation"][f"{a}.overlap_prH_vs_tprZ"] = bool(G[f"{a}.pr_H"].min() < G[f"{a}.tpr_Z"].max())
        S["separation"][f"{a}.overlap_prH_vs_tprU"] = bool(G[f"{a}.pr_H"].min() < G[f"{a}.tpr_U"].max())

    # ------------------------------------------------- variance components
    grid = {(r["config"]["init_seed"], r["config"]["ic_seed"]): scalars(r) for r in by["grid"]}
    var_keys = ["E.pr_H", "P.pr_H", "E.pr_H_unc", "P.pr_H_unc", "E.gram_pr", "P.gram_pr",
                "E.tpr_U", "P.tpr_U", "E.tpr_Z", "P.tpr_Z", "E.constancy", "P.constancy",
                "band_ratio_PE", "frac_captured_capturable"]
    S["variance"] = {}
    if len(grid) == 100:
        chaos = {k: np.array([scalars(r)[k] for r in by.get("chaos", [])] + [grid[(0, 0)][k]])
                 for k in var_keys}
        ic_ncap = np.array([grid[(0, c)]["n_capturable"] for c in range(10)])
        ic_sad = np.array([grid[(0, c)]["gram_pr_saddle"] for c in range(10)])
        for k in var_keys:
            Y = np.array([[grid[(i, c)][k] for c in range(10)] for i in range(10)])
            v = two_way(Y)
            v["var_chaos"] = float(np.var(chaos[k], ddof=1)) if len(chaos[k]) > 2 else np.nan
            v["n_chaos"] = int(len(chaos[k]))
            # is the residual (interaction + chaos) consistent with pure chaos?
            if np.isfinite(v["var_chaos"]) and v["var_chaos"] > 0:
                F = v["var_res"] / v["var_chaos"]
                v["F_res_vs_chaos"] = F
                v["p_res_vs_chaos_2s"] = float(2 * min(stats.f.sf(F, 81, len(chaos[k]) - 1),
                                                       stats.f.cdf(F, 81, len(chaos[k]) - 1)))
            cm = np.array(v["ic_means"])
            v["spearman_icmean_vs_ncap"] = float(stats.spearmanr(cm, ic_ncap)[0])
            v["spearman_icmean_vs_saddlePR"] = float(stats.spearmanr(cm, ic_sad)[0])
            S["variance"][k] = v

        # How stable is the published C1 design (8 inits at one IC vs 5 ICs at one init)?
        rng = np.random.default_rng(0)
        Y = np.array([[grid[(i, c)]["E.pr_H"] for c in range(10)] for i in range(10)])
        ratios = []
        for _ in range(20000):
            c = rng.integers(10); i = rng.integers(10)
            a = Y[rng.choice(10, 8, replace=False), c]
            b = Y[i, rng.choice(10, 5, replace=False)]
            ratios.append(np.var(a, ddof=1) / np.var(b, ddof=1))
        ratios = np.array(ratios)
        S["c1_design_resampled_ratio_PRHE"] = {
            "median": float(np.median(ratios)), "p05": float(np.percentile(ratios, 5)),
            "p95": float(np.percentile(ratios, 95)), "frac_gt_1": float((ratios > 1).mean()),
            "frac_ge_4.7": float((ratios >= 4.7).mean())}

    # ---------------------------------------------- noreset vs grid c = 0
    if "noreset" in by and len(grid) == 100:
        nr = {r["config"]["init_seed"]: scalars(r) for r in by["noreset"]}
        S["noreset"] = {}
        for k in var_keys + ["n_captured_uncapturable", "n_br_captured"]:
            a = np.array([nr[i][k] for i in sorted(nr)])
            b = np.array([grid[(i, 0)][k] for i in sorted(nr)])
            d = a - b
            p = float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
            S["noreset"][k] = {"median_noreset": float(np.median(a)), "median_reset": float(np.median(b)),
                               "median_diff": float(np.median(d)), "wilcoxon_p": p}

    # ---------------------------------------------------- stationarity
    S["stationarity"] = {}
    for k in ["E.pr_H", "P.pr_H", "E.gram_pr", "P.gram_pr", "E.tpr_Z", "P.tpr_Z", "band_ratio_PE"]:
        a = np.array([scalars(r, "10500")[k] for r in by["grid"]])
        b = np.array([scalars(r, "14000")[k] for r in by["grid"]])
        iqr = np.subtract(*np.percentile(b, [75, 25]))
        S["stationarity"][k] = {"spearman_10.5k_14k": float(stats.spearmanr(a, b)[0]),
                                "median_abs_change": float(np.median(np.abs(b - a))),
                                "between_run_IQR_14k": float(iqr)}

    # --------------------------------------------------------- game
    heads = defaultdict(list)
    for r in by["grid"]:
        g = r["full"]["14000"]["game"]
        for j in range(len(g["d0"])):
            heads["d0"].append(g["d0"][j]); heads["cap"].append(g["capturable"][j])
            heads["Vstar"].append(g["Vstar"][j])
            heads["Tc"].append(np.nan if g["Tc"][j] is None else g["Tc"][j])
            heads["Tbr"].append(np.nan if g["T_BR"][j] is None else g["T_BR"][j])
            heads["captured"].append(g["captured"][j])
            heads["chart_ok"].append(tuple(g["chart_cap"][j]) == tuple(g["chart_opt"][j]))
            heads["chart_br_ok"].append(tuple(g["chart_br"][j]) == tuple(g["chart_opt"][j]))
            heads["alignP"].append(r["full"]["14000"]["P"]["static_align"][j])
            heads["alignE"].append(r["full"]["14000"]["E"]["static_align"][j])
            ub = np.array(r["full"]["14000"]["E"]["ubar"][j])
            # for near-constant headings w = (v_E t_max / L) * ubar = 2 ubar
            heads["wind_eq_2ubar"].append(tuple(np.rint(g["wind_E"][j]).astype(int)) ==
                                          tuple(np.rint(cfg.v_E * cfg.t_max / cfg.L * ub).astype(int)))
    H = {k: np.array(v) for k, v in heads.items()}
    cap, captured = H["cap"].astype(bool), H["captured"].astype(bool)
    brc = ~np.isnan(H["Tbr"])
    ccap = cap & captured
    S["game"] = {
        "n_heads": int(len(cap)), "frac_capturable": float(cap.mean()),
        "capture_rate_capturable": float(captured[cap].mean()),
        "capture_rate_uncapturable": float(captured[~cap].mean()),
        "br_capture_rate_capturable": float(brc[cap].mean()),
        "br_capture_rate_uncapturable": float(brc[~cap].mean()),
        "T_BR_max_uncapturable": float(np.nanmax(H["Tbr"][~cap])),
        "Tc_over_Vstar_captured_capturable": describe(H["Tc"][ccap] / H["Vstar"][ccap]),
        "Tbr_over_Vstar_capturable": describe(H["Tbr"][cap & brc] / H["Vstar"][cap & brc]),
        "chart_agreement_captured": float(H["chart_ok"][captured].mean()),
        "chart_agreement_br_capturable": float(H["chart_br_ok"][cap & brc].mean()),
        "chart_agreement_br_uncapturable": float(H["chart_br_ok"][~cap & brc].mean()),
        "static_align_P_cap": describe(H["alignP"][cap]), "static_align_P_uncap": describe(H["alignP"][~cap]),
        "static_align_E_cap": describe(H["alignE"][cap]), "static_align_E_uncap": describe(H["alignE"][~cap]),
        "winding_equals_round_2ubar": float(H["wind_eq_2ubar"].mean()),
    }

    # ------------------------------------------- topology: label nulls
    lab_stats = defaultdict(lambda: {"within": [], "across": []})
    rng = np.random.default_rng(1)
    for r in by["grid"]:
        uE, uP, x0 = head_curves(r["name"], cfg)
        g = r["full"]["14000"]["game"]
        D0 = np.array(g["Delta0"]); dh = D0 / np.linalg.norm(D0, axis=1, keepdims=True)
        labels = {
            "winding (paper)": [tuple(np.rint(w).astype(int)) for w in g["wind_E"]],
            "IC heading bin round(2*Delta0hat)": [tuple(np.rint(2 * d).astype(int)) for d in dh],
            "nearest-image chart m*": [tuple(m) for m in g["chart_opt"]],
        }
        perm = list(labels["winding (paper)"]); rng.shuffle(perm)
        labels["permuted winding (null)"] = perm
        for name, lab in labels.items():
            w, a = class_ratio(uE, lab)
            lab_stats[name]["within"] += w.tolist(); lab_stats[name]["across"] += a.tolist()
    S["topology"] = {}
    for name, d in lab_stats.items():
        w, a = np.array(d["within"]), np.array(d["across"])
        S["topology"][name] = {"n_within_pairs": int(len(w)), "mean_within": float(w.mean()),
                               "mean_across": float(a.mean()), "median_within": float(np.median(w)),
                               "median_across": float(np.median(a)),
                               "ratio_means": float(a.mean() / w.mean()),
                               "ratio_medians": float(np.median(a) / np.median(w))}

    CW = capture_windows(by, cfg)
    S["capture_windows"] = CW["summary"]
    CH = played_charts(by, cfg)
    S["charts"] = CH["summary"]

    with open(os.path.join(RES, "r1_summary.json"), "w") as f:
        json.dump(S, f, indent=1, default=float)
    write_census(S, tabs)
    figures(by, tabs, S, H, cfg)
    figure_capture_windows(CW)
    figure_charts(CH)
    print(json.dumps({k: S[k] for k in ("n_runs", "published_percentile", "separation", "game")},
                     indent=1, default=float))
    return S


def capture_windows(by, cfg, min_pts=16):
    """Where does the time variation of the headings live?

    Each captured head is split at its capture time T_c into a pre-capture and a
    post-capture window; eta = |<u>_window|^2 per window (1 = constant heading).
    After capture the survival term no longer depends on the headings, but the
    shaping term lambda_d <d> still does, so post-capture motion is close-range
    pursuit.  The asymmetry statistic (band fraction ratio P/E) is also computed
    separately over the captured and the uncaptured heads of each run.
    """
    t, phid = M.dense_phi(cfg)
    vals = defaultdict(list)
    for r in by["grid"]:
        pE, pP, _ = load_ckpt(os.path.join(HERE, "ckpt", "r1", r["name"] + ".npz"))
        Tc = r["full"]["14000"]["game"]["Tc"]
        cap_idx = [j for j, v in enumerate(Tc) if v is not None]
        unc_idx = [j for j, v in enumerate(Tc) if v is None]
        band = {}
        for a, p in (("E", pE), ("P", pP)):
            u, _ = M.forward_dirs(p, phid)
            for j, tc in enumerate(Tc):
                if tc is None:
                    vals[f"{a}.eta_uncaptured"].append(float((u[j].mean(0) ** 2).sum()))
                    continue
                pre, post = t < tc, t >= tc
                if pre.sum() >= min_pts:
                    vals[f"{a}.eta_pre"].append(float((u[j, pre].mean(0) ** 2).sum()))
                if post.sum() >= min_pts:
                    vals[f"{a}.eta_post"].append(float((u[j, post].mean(0) ** 2).sum()))
            for sub, idx in (("captured", cap_idx), ("uncaptured", unc_idx)):
                if len(idx) >= 1:
                    band[(a, sub)] = M.unit_band_stats(u[idx], cfg.K)["band_11_50_U"]
        for sub in ("captured", "uncaptured"):
            if ("E", sub) in band:
                vals[f"band_ratio_PE_{sub}"].append(band[("P", sub)] / band[("E", sub)])
                vals[f"E.band_{sub}"].append(band[("E", sub)])
                vals[f"P.band_{sub}"].append(band[("P", sub)])
    summary = {k: describe(v) for k, v in vals.items()}
    return {"summary": summary, "vals": {k: np.array(v) for k, v in vals.items()}}


def figure_capture_windows(CW):
    V = CW["vals"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.42})
    ax = axs[0]
    groups = [("eta_uncaptured", "uncaptured\nheads"), ("eta_pre", "captured:\npre-capture"),
              ("eta_post", "captured:\npost-capture")]
    for i, (k, _) in enumerate(groups):
        for a, col, off in (("E", C_E, -0.19), ("P", C_P, 0.19)):
            strip(ax, i + off, 1 - V[f"{a}.{k}"], col,
                  label=({"E": "evader", "P": "pursuer"}[a] if i == 0 else None), w=0.3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[1] for g in groups])
    ax.set_ylabel("1 - $\\eta$ (time-varying energy, per head)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("time variation of the headings", loc="left")
    ax = axs[1]
    for i, sub in enumerate(("uncaptured", "captured")):
        strip(ax, i, V[f"band_ratio_PE_{sub}"], C_3, w=0.35)
    ax.axhline(1.0, color=MUTED, lw=1.0)
    ax.axhline(1.5, color=AXIS, lw=1.0)
    ax.text(1.45, 1.5, "paper's 1.5\nthreshold", color=INK2, fontsize=7, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["uncaptured\nheads", "captured\nheads"])
    ax.set_xlim(-0.5, 1.9)
    ax.set_yscale("log")
    ax.set_ylabel("band fraction ratio P/E, $k \\in [11, 50]$")
    ax.set_title("asymmetry statistic", loc="left")
    save(fig, "r1_capture_windows")


IMAGES = np.array([(mx, my) for mx in (-2, -1, 0, 1) for my in (-2, -1, 0, 1)])


def played_charts(by, cfg):
    """Which image chart does each head play?

    In chart m the saddle heading of both players is the direction of
    Delta_m = x_E0 + m L (pursuer at the origin).  The played chart of a head is
    argmax_m cos(<u>_t, Delta_m); m* (nearest image) is the optimal chart.
    Chart ambiguity: gap = |Delta_(2)| - |Delta_(1)|, the difference between the
    second-nearest and nearest image distances (0 on the cut locus).
    The mean heading is taken over the PRE-capture window [0, T_c) for captured
    heads (post-capture motion is not play) and over [0, t_max] otherwise.
    """
    t, phid = M.dense_phi(cfg)
    rows = defaultdict(list)
    for r in by["grid"]:
        f = r["full"]["14000"]
        g = f["game"]
        x0 = np.array(r["x_E0"])
        pE, pP, _ = load_ckpt(os.path.join(HERE, "ckpt", "r1", r["name"] + ".npz"))
        uu = {"E": M.forward_dirs(pE, phid)[0], "P": M.forward_dirs(pP, phid)[0]}
        for j in range(len(x0)):
            D = x0[j][None, :] + cfg.L * IMAGES
            dist = np.linalg.norm(D, axis=1)
            order = np.argsort(dist)
            m_star = tuple(IMAGES[order[0]])
            assert m_star == tuple(g["chart_opt"][j])
            Dh = D / dist[:, None]
            win = t < g["Tc"][j] if g["Tc"][j] is not None else np.ones_like(t, dtype=bool)
            for a in "EP":
                ub = uu[a][j, win].mean(0); ub = ub / np.linalg.norm(ub)
                c = Dh @ ub
                # images on (nearly) the same ray share a direction; moving along
                # that ray reaches the nearest of them first, so ties go to it
                tied = np.where(c >= c.max() - 5e-3)[0]
                k = tied[np.argmin(dist[tied])]
                rows[f"{a}.chart"].append(tuple(IMAGES[k]))
                rows[f"{a}.cos_played"].append(float(c[k]))
            rows["m_star"].append(m_star)
            rows["gap"].append(float(dist[order[1]] - dist[order[0]]))
            rows["capturable"].append(bool(g["capturable"][j]))
            rows["captured"].append(bool(g["captured"][j]))
    V = {k: (np.array(v) if not isinstance(v[0], tuple) else v) for k, v in rows.items()}
    okP = np.array([a == b for a, b in zip(V["P.chart"], V["m_star"])])
    okE = np.array([a == b for a, b in zip(V["E.chart"], V["m_star"])])
    same = np.array([a == b for a, b in zip(V["P.chart"], V["E.chart"])])
    cap, captured, gap = V["capturable"], V["captured"], V["gap"]
    summary = {
        "n_heads": int(len(gap)),
        "frac_P_plays_mstar": float(okP.mean()), "frac_E_plays_mstar": float(okE.mean()),
        "frac_P_E_same_chart": float(same.mean()),
        "cos_played_P": describe(V["P.cos_played"]), "cos_played_E": describe(V["E.cos_played"]),
        "capturable_frac_mstar": float(okP[cap].mean()),
        "capture_rate_capturable_mstar": float(captured[cap & okP].mean()),
        "capture_rate_capturable_other": float(captured[cap & ~okP].mean()) if (cap & ~okP).any() else None,
        "n_capturable_other": int((cap & ~okP).sum()),
        "n_missed_capturable": int((cap & ~captured).sum()),
        "n_missed_capturable_other_chart": int((cap & ~captured & ~okP).sum()),
        "gap_median_mstar": float(np.median(gap[okP])), "gap_median_other": float(np.median(gap[~okP])),
    }
    bins = np.quantile(gap, np.linspace(0, 1, 9))
    idx = np.clip(np.digitize(gap, bins[1:-1]), 0, 7)
    summary["error_rate_by_gap_octile"] = [
        {"gap_lo": float(bins[b]), "gap_hi": float(bins[b + 1]), "n": int((idx == b).sum()),
         "chart_error_rate": float((~okP[idx == b]).mean())} for b in range(8)]
    return {"summary": summary, "V": V, "okP": okP}


def figure_charts(CH):
    V, okP = CH["V"], CH["okP"]
    gap, cap, captured = V["gap"], V["capturable"], V["captured"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"wspace": 0.35})
    ax = axs[0]
    bins = np.quantile(gap, np.linspace(0, 1, 11))
    mids, rates, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (gap >= lo) & (gap <= hi)
        mids.append(np.median(gap[m])); rates.append((~okP[m]).mean()); ns.append(m.sum())
    ax.plot(mids, rates, "o-", color=C_P, ms=4.5, lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("chart gap $|\\Delta_{(2)}| - |\\Delta_{(1)}|$ (0 = cut locus)")
    ax.set_ylabel("fraction of heads in a non-nearest chart")
    ax.set_title("pursuer chart errors vs cut locus", loc="left")
    ax = axs[1]
    xs = np.arange(2)
    r = [captured[cap & okP].mean(), captured[cap & ~okP].mean()]
    n = [(cap & okP).sum(), (cap & ~okP).sum()]
    ax.bar(xs, r, width=0.55, color=C_P)
    for x, v, k in zip(xs, r, n):
        ax.text(x, v + 0.03, f"{100 * v:.1f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels([f"nearest chart $m^*$\n(n={n[0]})", f"other chart\n(n={n[1]})"])
    ax.set_ylim(0, 1.12); ax.set_ylabel("capture rate (capturable heads)")
    ax.set_title("trained pursuer's capture rate", loc="left")
    save(fig, "r1_charts")


def write_census(S, tabs):
    rows = [
        ("Strategy ensemble (unit headings)", None),
        ("Gram PR (Prop. 1 object; <= 2 if constant)", "gram_pr"),
        ("Gram PR, head-centred (legacy 'head PR')", "gram_pr_hc"),
        ("Constancy eta (1 = constant headings)", "constancy"),
        ("Temporal PR of u (time-varying part)", "tpr_U"),
        ("Temporal PR of z (LEGACY headline 'PR(Z)'; gauge-dependent)", "tpr_Z"),
        ("Gram PR, capturable heads only", "gram_pr_cap"),
        ("Gram PR, uncapturable heads only", "gram_pr_uncap"),
        ("Latent curve", None),
        ("PR(H), time-centred (LEGACY)", "pr_H"),
        ("PR(H), uncentred", "pr_H_unc"),
        ("Latent power fraction at k <= K", "frac_le_K_H"),
        ("Heads", None),
        ("Gauge-fixed effective rank of W_stack", "erank_gf"),
        ("Unit-heading power fraction, k in [11,50]", "band_11_50_U"),
        ("Within-run ratios", None),
        ("PR(H)/PR(Z) (LEGACY)", "ratio_legacy"),
        ("PR(H)/tPR(u) (gauge-invariant)", "ratio_U"),
    ]
    out = ["# R1 census (14,000 iterations)\n",
           "Entries: median [Q1, Q3] (min-max) over runs.\n"]
    for arm in ("grid", "chaos", "noreset"):
        if arm not in S["census"]:
            continue
        c = S["census"][arm]
        out.append(f"\n## arm `{arm}` (n = {c['E.gram_pr']['n']})\n")
        out.append("| statistic | evader | pursuer |\n|---|---|---|")
        for label, k in rows:
            if k is None:
                out.append(f"| **{label}** | | |")
            else:
                out.append(f"| {label} | {fmt(c['E.' + k])} | {fmt(c['P.' + k])} |")
        out.append("")
        out.append("| game statistic | value |\n|---|---|")
        for k, lab in (("n_capturable", "capturable heads (of 16)"),
                       ("frac_captured_capturable", "capture rate on capturable heads"),
                       ("n_captured_uncapturable", "uncapturable heads captured by trained P"),
                       ("n_br_captured", "heads captured by a best-response P (of 16)"),
                       ("pursuer_regret_mean", "pursuer regret vs best response (time)"),
                       ("gram_pr_saddle", "Gram PR of the analytic saddle (this IC set)"),
                       ("band_ratio_PE", "band fraction ratio P/E (asymmetry statistic)"),
                       ("sv_ratio_PE", "raw max singular value ratio P/E")):
            out.append(f"| {lab} | {fmt(c[k])} |")
    with open(os.path.join(RES, "r1_census.md"), "w") as f:
        f.write("\n".join(out) + "\n")


# ================================================================= figures
def strip(ax, x, vals, color, label=None, w=0.28):
    rng = np.random.default_rng(int(1000 * x) % 2 ** 31)
    vals = np.asarray(vals)[np.isfinite(vals)]
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    ax.scatter(x + rng.uniform(-w / 2, w / 2, len(vals)), vals, s=9, color=color, alpha=0.45,
               linewidths=0, label=label, zorder=2)
    ax.plot([x - w / 1.6, x + w / 1.6], [med, med], color=INK, lw=1.4, zorder=3)
    ax.plot([x, x], [q1, q3], color=INK, lw=0.9, zorder=3)


def figures(by, tabs, S, H, cfg):
    G = tabs["grid"]
    # ---- Fig R1-a: the two dimensionalities, every statistic, every run
    cats = [("gram_pr", "Gram PR\n(Prop. 1)"), ("tpr_U", "temporal PR\nof $u$"),
            ("tpr_Z", "temporal PR\nof $z$ (legacy)"), ("pr_H_unc", "PR$(H)$\nuncentred"),
            ("pr_H", "PR$(H)$\ncentred (legacy)")]
    pub = {("E", "gram_pr"): 2.543, ("P", "gram_pr"): 2.477, ("E", "tpr_U"): 2.849, ("P", "tpr_U"): 3.145,
           ("E", "tpr_Z"): 2.987, ("P", "tpr_Z"): 1.915, ("E", "pr_H_unc"): 4.214, ("P", "pr_H_unc"): 4.015,
           ("E", "pr_H"): 6.748, ("P", "pr_H"): 7.080}
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for i, (k, _) in enumerate(cats):
        for a, col, off in (("E", C_E, -0.19), ("P", C_P, 0.19)):
            strip(ax, i + off, G[f"{a}.{k}"], col,
                  label=({"E": "evader", "P": "pursuer"}[a] if i == 0 else None))
            ax.scatter(i + off, pub[(a, k)], marker="D", s=22, facecolor="white", edgecolor=INK,
                       lw=0.9, zorder=4, label=("published baseline run" if (i == 0 and a == "E") else None))
    sad = G["gram_pr_saddle"]
    ax.plot([-0.45, 0.45], [2, 2], color=MUTED, lw=1.0, zorder=1,
            label="bound for constant headings (2)")
    ax.plot([-0.45, 0.45], [np.median(sad)] * 2, color=C_3, lw=1.2, zorder=1,
            label="analytic saddle (median over IC sets)")
    ax.axvline(2.5, color=AXIS, lw=0.8)
    ax.text(1.0, 0.99, "strategy ensemble", transform=ax.get_xaxis_transform(), ha="center",
            va="top", color=INK2, fontsize=8)
    ax.text(3.5, 0.99, "latent curve", transform=ax.get_xaxis_transform(), ha="center",
            va="top", color=INK2, fontsize=8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c[1] for c in cats])
    ax.set_ylabel("participation ratio")
    ax.set_xlim(-0.6, len(cats) - 0.4)
    hs, ls = ax.get_legend_handles_labels()
    order = [ls.index(x) for x in ("evader", "pursuer", "published baseline run",
                                    "bound for constant headings (2)",
                                    "analytic saddle (median over IC sets)")]
    ax.legend([hs[i] for i in order], [ls[i] for i in order], loc="upper left",
              bbox_to_anchor=(0.0, 0.93), ncol=1)
    ax.set_title(f"Ensemble and latent dimensionality over {len(G['E.gram_pr'])} runs "
                 "(10 init seeds x 10 IC sets)", loc="left")
    save(fig, "r1_dimensionalities")

    # ---- Fig R1-b: integer-k spectra, median and IQR over the grid
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    for ax, key, title in ((axs[0], "kpow_H", "latent $H(t)$"), (axs[1], "kpow_U", "unit headings $u(t)$")):
        for a, col, lab in (("E", C_E, "evader"), ("P", C_P, "pursuer")):
            sub = "lat" if key == "kpow_H" else "band"
            P = np.array([r["full"]["14000"][a][sub][key] for r in by["grid"]])
            P = P / P[:, 1:].sum(1, keepdims=True)
            k = np.arange(P.shape[1])
            q1, med, q3 = np.percentile(P[:, 1:], [25, 50, 75], axis=0)
            ax.fill_between(k[1:], q1, q3, color=col, alpha=0.2, lw=0)
            ax.plot(k[1:], med, color=col, lw=1.3, label=lab)
        ax.axvline(12, color=MUTED, lw=1.0)
        ax.text(12.6, 0.04, "input $K=12$", transform=ax.get_xaxis_transform(), color=INK2, fontsize=7.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("integer time frequency $k$")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("fraction of temporal power")
    axs[0].legend(loc="lower left")
    save(fig, "r1_spectra")

    # ---- Fig R1-c: variance components (standard deviations)
    if S.get("variance"):
        ks = ["E.pr_H", "P.pr_H", "E.gram_pr", "P.gram_pr", "E.tpr_Z", "P.tpr_Z", "band_ratio_PE"]
        labels = ["PR(H) evader", "PR(H) pursuer", "Gram PR evader", "Gram PR pursuer",
                  "legacy PR(Z) evader", "legacy PR(Z) pursuer", "band ratio P/E"]
        comps = [("var_init", "init seed", C_E), ("var_ic", "IC set", C_P),
                 ("var_res", "residual (interaction + chaos)", C_3)]
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        y = np.arange(len(ks))
        for k_i, k in enumerate(ks):
            v = S["variance"][k]
            tot = v["var_init"] + v["var_ic"] + v["var_res"]
            left = 0.0
            for ck, cl, col in comps:
                w = v[ck] / tot
                ax.barh(k_i, w, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.5,
                        label=cl if k_i == 0 else None)
                left += w
            ch = v["var_chaos"] / tot
            ax.plot([ch, ch], [k_i - 0.4, k_i + 0.4], color=INK, lw=1.6,
                    label="chaos-arm variance (1e-10 perturbation)" if k_i == 0 else None)
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
        ax.set_xlim(0, max(1.05, ax.get_xlim()[1])); ax.grid(axis="y", visible=False)
        ax.set_xlabel("fraction of run-to-run variance (two-way random effects, 10 x 10 grid)")
        hs, ls = ax.get_legend_handles_labels()
        order = [ls.index(c[1]) for c in comps] + [ls.index("chaos-arm variance (1e-10 perturbation)")]
        ax.legend([hs[i] for i in order], [ls[i] for i in order], loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=2)
        ax.set_title("Where run-to-run variation comes from", loc="left")
        save(fig, "r1_variance")

    # ---- Fig R1-d: game performance per head
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axs[0]
    d0, cap = H["d0"], H["cap"].astype(bool)
    rng = np.random.default_rng(0)
    sel = rng.random(len(d0)) < 1.0
    ax.scatter(d0[sel & ~np.isnan(H["Tc"])], H["Tc"][sel & ~np.isnan(H["Tc"])], s=8, color=C_P,
               alpha=0.5, lw=0, label="trained $T_c$")
    ax.scatter(d0[sel], H["Tbr"][sel], s=8, color=C_E, alpha=0.5, lw=0, label="best response $T_{BR}$")
    xx = np.linspace(cfg.eps, d0.max(), 100)
    ax.plot(xx, (xx - cfg.eps) / cfg.dv, color=INK, lw=1.0, label="$V^*(\\Delta_0)$")
    ax.axvline(cfg.reach, color=MUTED, lw=1.0)
    ax.text(cfg.reach + 0.015, 0.47, "reach $R$", transform=ax.get_xaxis_transform(), color=INK2,
            fontsize=7.5, va="top")
    ax.set_ylim(0, cfg.t_max + 0.1)
    ax.set_xlabel("initial distance $|\\Delta_0|$"); ax.set_ylabel("capture time")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("capture times on the training ICs", loc="left")
    ax = axs[1]
    rates = [(H["captured"][cap].mean(), H["captured"][~cap].mean()),
             ((~np.isnan(H["Tbr"]))[cap].mean(), (~np.isnan(H["Tbr"]))[~cap].mean())]
    xs = np.arange(2)
    ax.bar(xs - 0.18, [rates[0][0], rates[0][1]], width=0.34, color=C_P, label="trained pursuer")
    ax.bar(xs + 0.18, [rates[1][0], rates[1][1]], width=0.34, color=C_E, label="best response")
    for x0_, v in zip(np.r_[xs - 0.18, xs + 0.18], [rates[0][0], rates[0][1], rates[1][0], rates[1][1]]):
        ax.text(x0_, v + 0.02, f"{100 * v:.0f}%", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels([f"capturable\n(n={cap.sum()})", f"uncapturable\n(n={(~cap).sum()})"])
    ax.set_ylim(0, 1.12); ax.set_ylabel("fraction of heads captured")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("against the co-trained evader", loc="left")
    save(fig, "r1_game")

    # ---- Fig R1-e: training dynamics of the light metrics
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    its = np.array([s["it"] for s in by["grid"][0]["snapshots"]])
    for ax, k, title in ((axs[0], "gram_pr", "Gram PR"), (axs[1], "tpr_Z", "legacy PR(Z)"),
                         (axs[2], "pr_H", "PR(H) centred")):
        for a, col, lab in (("E", C_E, "evader"), ("P", C_P, "pursuer")):
            V = np.array([[s[a][k] for s in r["snapshots"]] for r in by["grid"]])
            q1, med, q3 = np.percentile(V, [25, 50, 75], axis=0)
            ax.fill_between(its, q1, q3, color=col, alpha=0.2, lw=0)
            ax.plot(its, med, color=col, lw=1.3, label=lab)
        for x in (3500, 7000, 10500):
            ax.axvline(x, color=GRID, lw=0.8)
        ax.set_title(title, loc="left"); ax.set_xlabel("self-play iteration")
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("participation ratio")
    save(fig, "r1_dynamics")


if __name__ == "__main__":
    main()
