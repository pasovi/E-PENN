"""The draft's eight figures from the torch runs, plus the C7 collapse figure.

  fig1_spectra.png        integer-k latent power spectra (baseline, both players) and the
                          K-sweep (C3) break
  fig2_sectors.png        sector experiments: PR bars and the winding cloud (baseline vs E1b)
  fig3_reflection.png     reflection control: E1b vs E1b reflected
  fig4_dv_sweep.png       speed-difference sweep
  fig5_heads.png          head singular values, raw vs gauge-fixed, with the initialisation
  fig6_protocols.png      training-protocol comparison
  fig7_honest.png         honest self-play corrections, quality-stability plane
  fig8_shap.png           grouped Shapley attribution
  figA_c7_collapse.png    warm-start collapse of the planted ensemble (Table 1, C7)
Written to paper/figs/ as .png and .pdf.  Usage: python3 torch_actor/figures.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as A  # noqa: E402

FIG = os.path.join(A.PAPER, "figs")
os.makedirs(FIG, exist_ok=True)
C_E, C_P = "#1f77b4", "#d62728"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def base_suite():
    return A.suite(*A.load(os.path.join(A.CKPT, "baseline", "grid_i0_c0.npz")))


# ------------------------------------------------------------------ Fig 1
def fig1(base):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
    k = np.arange(1, 129)
    for a, c, name in (("E", C_E, "evader"), ("P", C_P, "pursuer")):
        pk = np.array(base[f"kpow_{a}"])[1:129]
        ax[0].plot(k, pk / pk[:8].mean(), "-", color=c, lw=1.5, label=f"{name} (break at k={base['break_' + a]})")
    ax[0].axvline(12, color="k", ls="--", lw=1, label="input bandwidth K = 12")
    ax[0].axvspan(11, 50, color="0.85", alpha=0.5, label="intermediate band k ∈ [11, 50]")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("integer frequency k"); ax[0].set_ylabel("latent power (normalised to the k ≤ 8 plateau)")
    ax[0].set_title(f"(a) Baseline: drop k=12→16 is ×{base['drop_K_to_K4_E']:.0f} (E), ×{base['drop_K_to_K4_P']:.0f} (P)", fontsize=10)
    ax[0].legend(fontsize=7.5, loc="lower left"); ax[0].set_ylim(1e-7, 10)
    for K, c in ((6, "tab:green"), (12, C_E), (24, "tab:purple")):
        f = (os.path.join(A.CKPT, f"c3_k{K}", f"c3_k{K}_s0.npz") if K != 12
             else os.path.join(A.CKPT, "baseline", "grid_i0_c0.npz"))
        s = A.suite(*A.load(f))
        pk = np.array(s["kpow_E"])[1:129]
        ax[1].plot(k, pk / pk[:max(K // 2, 3)].mean(), "-", color=c, lw=1.5,
                   label=f"K = {K}: break at k = {s['break_E']}, {100 * s['frac_le_K_E']:.0f}% of power at k ≤ K")
        ax[1].axvline(K, color=c, ls="--", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-7, 10)
    ax[1].set_xlabel("integer frequency k"); ax[1].set_ylabel("evader latent power (normalised)")
    ax[1].set_title("(b) Bandwidth sweep (C3): the break follows K", fontsize=10)
    ax[1].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, "fig1_spectra")


# ------------------------------------------------------------------ Fig 2
def fig2(base):
    e1 = A.suite(*A.load(os.path.join(A.CKPT, "variants", "e1_s0.npz")))
    e1b = A.suite(*A.load(os.path.join(A.CKPT, "variants", "e1b_s0.npz")))
    runs = [("baseline", base), ("E1 angular", e1), ("E1b one-sided", e1b)]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    x = np.arange(3); bw = 0.38
    PR_H = [s["PR_HE"] for _, s in runs]; PR_Z = [s["PR_ZE"] for _, s in runs]
    axL.bar(x - bw / 2, PR_H, bw, color=C_E, label=r"latent curve  $\mathrm{PR}(\mathbf{H}_E)$")
    axL.bar(x + bw / 2, PR_Z, bw, color="#2ca02c", label=r"control ensemble  $\mathrm{PR}(Z_E)$")
    axL.axhline(2.0, ls=":", c="k", lw=1.6, label="constant-heading bound (=2)")
    for xi, h, zv in zip(x, PR_H, PR_Z):
        axL.text(xi - bw / 2, h + 0.08, f"{h:.1f}", ha="center", fontsize=8)
        axL.text(xi + bw / 2, zv + 0.08, f"{zv:.1f}", ha="center", fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels([f"{lab}\n({s['n_sectors']} classes)" for lab, s in runs])
    axL.set_ylabel("participation ratio"); axL.set_title("Sector restriction: PR falls far less than class count")
    axL.legend(fontsize=9, frameon=False); axL.grid(axis="y", alpha=0.25); axL.set_ylim(0, max(PR_H) + 1.0)
    wb, we = np.array(base["winding_E"]), np.array(e1b["winding_E"])
    axR.axhline(0, c="0.85", lw=0.8); axR.axvline(0, c="0.85", lw=0.8)
    for g in np.arange(-2.5, 3, 1):
        axR.axhline(g, c="0.92", lw=0.6); axR.axvline(g, c="0.92", lw=0.6)
    axR.scatter(wb[:, 0], wb[:, 1], s=80, color=C_E, label=f"baseline ICs ({base['n_sectors']} classes)")
    axR.scatter(we[:, 0], we[:, 1], s=80, marker="^", color="#ff7f0e", label=f"E1b one-sided ICs ({e1b['n_sectors']} classes)")
    axR.set_xlabel(r"evader winding $w_x$"); axR.set_ylabel(r"evader winding $w_y$")
    axR.set_title("Per-head net winding (integer cells = classes)")
    axR.legend(fontsize=9, frameon=False); axR.set_aspect("equal", "datalim")
    save(fig, "fig2_sectors")


# ------------------------------------------------------------------ Fig 3
def fig3():
    w1 = np.array(A.suite(*A.load(os.path.join(A.CKPT, "variants", "e1b_s0.npz")))["winding_E"])
    w2 = np.array(A.suite(*A.load(os.path.join(A.CKPT, "variants", "e1b_refl_s0.npz")))["winding_E"])
    fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    ax.axvline(0, ls="--", c="k", lw=1.3); ax.axhline(0, c="0.85", lw=0.8)
    ax.scatter(w1[:, 0], w1[:, 1], s=85, color=C_E, label=rf"$+x$ ICs (E1b), mean $w_x$={w1[:, 0].mean():+.2f}")
    ax.scatter(w2[:, 0], w2[:, 1], s=85, marker="^", color="#ff7f0e", label=rf"$-x$ ICs (reflected), mean $w_x$={w2[:, 0].mean():+.2f}")
    ax.set_xlabel(r"evader winding $w_x$"); ax.set_ylabel(r"evader winding $w_y$")
    ax.legend(fontsize=9, loc="upper center", ncol=1); ax.grid(alpha=0.25); ax.set_aspect("equal", "datalim")
    lim = np.abs(np.concatenate([w1, w2])).max() * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_title("Reflection control: the winding cloud mirrors about $w_x = 0$", fontsize=10)
    save(fig, "fig3_reflection")


# ------------------------------------------------------------------ Fig 4
def fig4():
    L, t_max, eps = 2.0, 4.0, 0.08
    rows = [json.loads(l) for l in open(os.path.join(A.RES, "dv_sweep_results_full.jsonl"))]
    dvs = sorted(set(r["dv"] for r in rows))

    def agg(key, dv):
        v = [r[key] for r in rows if r["dv"] == dv]
        return np.mean(v), np.std(v)

    def f_reach(R):
        if R <= 0: return 0.0
        if R >= math.sqrt(2): return 1.0
        if R <= 1.0: return math.pi * R * R / 4.0
        seg = R * R * math.acos(1.0 / R) - math.sqrt(R * R - 1.0)
        return (math.pi * R * R - 4.0 * seg) / 4.0

    dv_ridge = (1.0 - eps) / t_max; dv_sat = (math.sqrt(2) - eps) / t_max
    summary = {"budget_iters": rows[0]["iters"], "n_seeds": 3, "dv_ridge": dv_ridge, "dv_sat": dv_sat, "by_dv": {}}
    for dv in dvs:
        R = dv * t_max + eps
        summary["by_dv"][dv] = dict(R=R, rho=R / (L / 2), latPR_E=agg("latPR_E", dv), zPR_E=agg("zPR_E", dv),
                                    latPR_P=agg("latPR_P", dv), zPR_P=agg("zPR_P", dv),
                                    captured=agg("captured_frac", dv)[0], cap_share_ICs=agg("cap_share_ICs", dv)[0],
                                    f_analytic=f_reach(R))
    json.dump(summary, open(os.path.join(A.RES, "dv_sweep_summary.json"), "w"), indent=2)
    dv_arr = np.array(dvs)
    Rgrid = np.linspace(0.05, dv_arr.max() * t_max + eps + 0.05, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
    ax[0].plot((Rgrid - eps) / t_max, [f_reach(R) for R in Rgrid], "-", color="tab:green", lw=2, label=r"analytic capturable share $f(R)$")
    ax[0].plot(dv_arr, [summary["by_dv"][d]["cap_share_ICs"] for d in dvs], "s", color="tab:gray", ms=6, label="capturable share, 16-IC set")
    ax[0].plot(dv_arr, [summary["by_dv"][d]["captured"] for d in dvs], "o", color=C_E, ms=6, label="realised capture (trained)")
    ax[0].axvline(dv_ridge, color="k", ls="--", lw=1); ax[0].axvline(dv_sat, color="k", ls=":", lw=1)
    ax[0].text(dv_ridge - 0.006, 0.30, r"$\rho=1$ (ridge)", rotation=90, va="center", ha="right", fontsize=8)
    ax[0].text(dv_sat + 0.012, 0.30, "saturation", rotation=90, va="center", ha="left", fontsize=8)
    ax[0].set_xlabel(r"$\Delta v = v_P - v_E$"); ax[0].set_ylabel("capturable / captured share")
    ax[0].set_title("(a) Game geometry: a break at $\\rho=1$"); ax[0].legend(fontsize=7.5, loc="lower right"); ax[0].set_ylim(-0.03, 1.05)
    for a, c, m, lab in (("E", C_P, "o-", "E"), ("P", "tab:brown", "^-", "P")):
        le = [summary["by_dv"][d][f"latPR_{a}"][0] for d in dvs]; les = [summary["by_dv"][d][f"latPR_{a}"][1] for d in dvs]
        ax[1].errorbar(dv_arr, le, yerr=les, fmt=m, color=c, capsize=3, label=rf"latent $\mathrm{{PR}}(H_{lab})$")
    for a, c, m, lab in (("E", C_E, "s-", "E"), ("P", "tab:cyan", "v-", "P")):
        ze = [summary["by_dv"][d][f"zPR_{a}"][0] for d in dvs]; zes = [summary["by_dv"][d][f"zPR_{a}"][1] for d in dvs]
        ax[1].errorbar(dv_arr, ze, yerr=zes, fmt=m, color=c, capsize=3, label=rf"strategy $\mathrm{{PR}}(Z_{lab})$")
    ax[1].axhline(2.0, color=C_E, ls=":", lw=1, label="constant-heading bound $=2$")
    ax[1].axvline(dv_ridge, color="k", ls="--", lw=1)
    ax[1].set_xlabel(r"$\Delta v = v_P - v_E$"); ax[1].set_ylabel("participation ratio")
    ax[1].set_title("(b) Embedding is flat across the break (mean ± s.d., 3 seeds)")
    ax[1].legend(fontsize=7, loc="upper left", ncol=2); ax[1].set_ylim(0, 9.5)
    fig.tight_layout()
    save(fig, "fig4_dv_sweep")


# ------------------------------------------------------------------ Fig 5
def fig5(base):
    ini = A.init_suite(0)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    i = np.arange(1, 17)
    ax[0].plot(i, base["sv_raw_P"], "s-", color=C_P, label=f"pursuer (max {base['sv_raw_P'][0]:.1f}, PR {base['svPR_raw_P']:.2f})")
    ax[0].plot(i, base["sv_raw_E"], "o-", color=C_E, label=f"evader (max {base['sv_raw_E'][0]:.1f}, PR {base['svPR_raw_E']:.2f})")
    ax[0].plot(i, ini["sv_raw_E"], "--", color="0.5", label="initialisation (seed 0)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("singular-value index"); ax[0].set_ylabel("singular value of $W_{\\rm stack}$")
    ax[0].set_title("(a) Raw head spectra"); ax[0].legend(fontsize=8)
    ax[1].plot(i, base["sv_gf_P"], "s-", color=C_P, label=f"pursuer (effective rank {base['gf_svPR_P']:.2f})")
    ax[1].plot(i, base["sv_gf_E"], "o-", color=C_E, label=f"evader (effective rank {base['gf_svPR_E']:.2f})")
    ax[1].plot(i, ini["sv_gf_E"], "--", color="0.5", label="initialisation (seed 0)")
    ax[1].set_yscale("log"); ax[1].set_xlabel("singular-value index"); ax[1].set_ylabel("singular value, unit-Frobenius heads")
    ax[1].set_title("(b) Gauge-fixed (Lemma 2)"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_heads")


# ------------------------------------------------------------------ Fig 6
def fig6(suffix=""):
    rows = [json.loads(l) for l in open(os.path.join(A.RES, f"game_protocols_results_full{suffix}.jsonl"))]
    protos = ["oracle", "selfplay", "simul", "abr"]
    labels = {"oracle": "oracle (fixed optimal evader)", "selfplay": "self-play, 3:1 (paper baseline)",
              "simul": "simultaneous 1:1 (GDA)", "abr": "alternating best-response"}
    col = {"oracle": "tab:green", "selfplay": "tab:blue", "simul": "tab:orange", "abr": "tab:red"}
    summary, out = {}, {}
    for p in protos:
        T = [r["trace"] for r in rows if r["protocol"] == p]
        Lm = min(len(t) for t in T)
        gap = np.array([[t[i][1] for i in range(Lm)] for t in T])
        summary[p] = dict(frac=np.array([t[0] for t in T[0][:Lm]]), gap_mean=gap.mean(0), gap_sd=gap.std(0),
                          osc=[r["osc_sd_2ndhalf"] for r in rows if r["protocol"] == p],
                          cap=[r["final_cap_of_capturable"] for r in rows if r["protocol"] == p],
                          fgap=[r["final_gap_capturable"] for r in rows if r["protocol"] == p])
        out[p] = {"final_cap_of_capturable_mean": float(np.mean(summary[p]["cap"])),
                  "final_gap_capturable_mean": float(np.mean(summary[p]["fgap"])),
                  "osc_sd_2ndhalf_mean": float(np.mean(summary[p]["osc"])),
                  "osc_sd_2ndhalf_by_seed": summary[p]["osc"], "cap_by_seed": summary[p]["cap"]}
    json.dump(out, open(os.path.join(A.RES, f"game_protocols_summary{suffix}.json"), "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
    for p in protos:
        s = summary[p]
        ax[0].plot(s["frac"], s["gap_mean"], "-", color=col[p], lw=2, label=labels[p])
        ax[0].fill_between(s["frac"], np.clip(s["gap_mean"] - s["gap_sd"], 1e-3, None), s["gap_mean"] + s["gap_sd"], color=col[p], alpha=0.15)
    ax[0].axhline(0.08, color="k", ls=":", lw=1, label=r"capture radius $\varepsilon$")
    ax[0].set_yscale("log"); ax[0].set_xlabel("training fraction"); ax[0].set_ylabel("mean pursuit gap on capturable ICs")
    ax[0].set_title("(a) Convergence vs co-adaptation (3 seeds" + (", capture-truncated loss)" if suffix else ")"))
    ax[0].legend(fontsize=7.2, loc="upper right")
    for i, p in enumerate(protos):
        o = summary[p]["osc"]
        ax[1].bar(i, np.mean(o), color=col[p], alpha=0.75); ax[1].plot([i] * len(o), o, "o", color="k", ms=4, zorder=5)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["oracle", "self-play\n3:1", "simul\n1:1", "ABR"], fontsize=8)
    ax[1].set_ylabel("late-training oscillation (s.d. of gap)"); ax[1].set_title("(b) Cycling by protocol (dots = seeds)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig6_protocols" + suffix)


# ------------------------------------------------------------------ Fig 7
def fig7():
    rows = [json.loads(l) for l in open(os.path.join(A.RES, "honest_protocols_results.jsonl"))]
    by = defaultdict(list)
    for r in rows:
        by[r["protocol"]].append(r)
    lab = {"oracle": "oracle (non-honest ceiling)", "gda": "GDA (simultaneous)", "ogda": "optimistic GDA",
           "eg": "extragradient", "avg": "Cesàro averaging", "ogda_ema": "optimistic GDA + EMA"}
    col = {"oracle": "tab:green", "gda": "tab:red", "ogda": "tab:blue", "eg": "tab:orange", "avg": "tab:purple", "ogda_ema": "tab:brown"}
    order = ["oracle", "gda", "ogda", "eg", "avg", "ogda_ema"]
    summary = {p: dict(cap=float(np.mean([r["final_cap_of_capturable"] for r in by[p]])),
                       gap=float(np.mean([r["final_gap_capturable"] for r in by[p]])),
                       osc=float(np.mean([r["osc_sd_2ndhalf"] for r in by[p]])),
                       cap_by_seed=[r["final_cap_of_capturable"] for r in by[p]],
                       osc_by_seed=[r["osc_sd_2ndhalf"] for r in by[p]], n=len(by[p])) for p in order if by[p]}
    json.dump(summary, open(os.path.join(A.RES, "honest_protocols_summary.json"), "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for p in order:
        rs = by.get(p, [])
        if not rs:
            continue
        caps = [r["final_cap_of_capturable"] for r in rs]; oscs = [max(r["osc_sd_2ndhalf"], 1e-4) for r in rs]
        ax[0].scatter(caps, oscs, s=28, color=col[p], alpha=0.55, edgecolor="none")
        ax[0].scatter([np.mean(caps)], [np.mean(oscs)], s=150, color=col[p], edgecolor="k", zorder=5, label=lab[p])
    ax[0].set_yscale("log"); ax[0].set_xlabel("final capture of capturable set  (→ 1 better)")
    ax[0].set_ylabel("late-training oscillation  (↓ better)")
    ax[0].set_title(f"(a) Quality–stability plane ({rows[0]['iters']} iters, {max(len(v) for v in by.values())} seeds)")
    ax[0].axhspan(1e-4, 2e-3, color="tab:green", alpha=0.06)
    ax[0].annotate("oracle basin\n(stable + full capture)", (0.62, 0.0009), fontsize=7.5, color="tab:green", va="center")
    ax[0].legend(fontsize=7, loc="upper left"); ax[0].set_xlim(0.0, 1.05); ax[0].set_ylim(5e-5, 3e-1)
    for p in ["oracle", "gda", "ogda", "avg"]:
        rs = [r for r in by.get(p, []) if r["seed"] == 0]
        if rs:
            t = rs[0]["trace"]
            ax[1].plot([x[0] for x in t], [x[1] for x in t], "-", color=col[p], lw=1.8, label=lab[p], alpha=0.9)
    ax[1].axhline(0.08, color="k", ls=":", lw=1, label=r"capture radius $\varepsilon$")
    ax[1].set_yscale("log"); ax[1].set_xlabel("training fraction"); ax[1].set_ylabel("pursuit gap on capturable ICs")
    ax[1].set_title("(b) Representative dynamics (seed 0)"); ax[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "fig7_honest")


# ------------------------------------------------------------------ Fig 8
def fig8():
    bands = ["linear", "low", "mid", "high"]
    S = [json.load(open(os.path.join(A.RES, f"shap_seed{s}.json"))) for s in (0, 1, 2)]
    M = lambda role: np.array([[d[role]["shap_cap"][b] for b in bands] for d in S])  # noqa: E731
    o, sp = M("oracle"), M("selfplay")
    capO = np.mean([d["oracle"]["cap_full"] for d in S]); capS = np.mean([d["selfplay"]["cap_full"] for d in S])
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w / 2, o.mean(0), w, yerr=o.std(0), capsize=3, color="tab:green", alpha=0.85, label=f"oracle pursuer (capture {capO:.2f})")
    ax.bar(x + w / 2, sp.mean(0), w, yerr=sp.std(0), capsize=3, color="tab:red", alpha=0.85, label=f"self-play pursuer (capture {capS:.2f})")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x)
    ax.set_xticklabels(["linear\n(t)", "low\n(k1-4)", "mid\n(k5-8)", "high\n(k9-12)"])
    ax.set_ylabel("Shapley value for capture")
    ax.set_title("Shapley attribution of pursuit capability to input bands\n(vs the same optimal evader; 3 seeds)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "fig8_shap")


# ------------------------------------------------------------- C7 collapse
def figA_c7():
    rec = json.load(open(os.path.join(A.RES, "c7.json")))
    its = np.array([s["it"] for s in rec["runs"][0]["snapshots"]])
    fig, ax = plt.subplots(figsize=(7, 4))
    for a, c, lab in (("E", C_E, "evader PR(Z_E)"), ("P", C_P, "pursuer PR(Z_P)")):
        V = np.array([[s[a]["tpr_Z"] for s in r["snapshots"]] for r in rec["runs"]])
        ax.plot(its, np.median(V, 0), "-", color=c, lw=2, label=lab + " (median, 10 seeds)")
        ax.fill_between(its, np.percentile(V, 25, 0), np.percentile(V, 75, 0), color=c, alpha=0.2)
    V = np.array([[s["E"]["pr_H"] for s in r["snapshots"]] for r in rec["runs"]])
    ax.plot(its, np.median(V, 0), "--", color="0.4", lw=1.5, label="latent PR(H_E)")
    ax.axhline(2, color="k", ls=":", lw=1, label="constant-heading bound")
    ax.set_xlabel("game-training iteration (from the fitted state, no warm-up)"); ax.set_ylabel("participation ratio")
    ax.set_title("C7: game training collapses the planted curved ensemble", fontsize=10)
    ax.legend(fontsize=8); ax.set_ylim(0, 12)
    fig.tight_layout()
    save(fig, "figA_c7_collapse")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fig6_trunc":
        fig6("_trunc"); sys.exit()
    base = base_suite()
    fig1(base); fig2(base); fig3(); fig4(); fig5(base); fig6(); fig7(); fig8(); figA_c7()
