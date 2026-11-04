"""Figures and summary numbers of a benchmark folder (run first: python3 torch_actor/experiments.py <folder name>).
The folder name (baseline_D4, baseline_D4_w1500, ...) is the stage name and the label in the figures.

Reads  <folder>/results/grid_*.json (100 runs, rev2 record format) and <folder>/ckpt/grid_i0_c0.npz
Writes <folder>/figs/
         game.png/.pdf               capture time vs initial distance and capture fractions (the r1_game figure)
         dimensionalities.png/.pdf   Gram PR, temporal PR, PR(H) strips; diamonds = the 1-layer truncated baseline
         fig_torus_trajectories.png, fig_lifted.png, fig_torus3d.png, fig_headings.png, fig_latent.png,
         head_table.md               the static views of run (init seed 0, IC seed 0)
       <folder>/summary.json      ensemble medians [IQR] and the same numbers for the 1-layer truncated baseline
       <folder>/config.json       every hyperparameter of the run
The evolution video (figs/training_evolution_flat_6heads.mp4) is made by checks/animate_training.py; see README.md.
Usage: python3 torch_actor/<folder>/make_figures.py
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
NAME = os.path.basename(HERE)          # baseline_D4, baseline_D4_w1500, ...
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "checks"))
import dimensionalities as D  # noqa: E402
import visualize_actor as V  # noqa: E402

RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figs")
REF = os.path.join(ROOT, "checks", "results", "r1_trunc")     # 1-layer body, same loss and protocol except warm-up and lr
C_E, C_P = "#2a78d6", "#eb6834"
os.makedirs(FIG, exist_ok=True)


def load(d):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "grid_*.json")))]


def q(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))]


def game_arrays(recs):
    d0, Tc, Tbr, cap, capt, Vs = [], [], [], [], [], []
    for r in recs:
        g = r["full"]["14000"]["game"]
        d0 += g["d0"]; Vs += g["Vstar"]; cap += g["capturable"]; capt += g["captured"]
        Tc += [np.nan if t is None else t for t in g["Tc"]]
        Tbr += [np.nan if t is None else t for t in g["T_BR"]]
    return (np.array(d0), np.array(Tc), np.array(Tbr), np.array(cap, bool), np.array(capt, bool), np.array(Vs))


def figure_game(recs, label):
    d0, Tc, Tbr, cap, capt, Vs = game_arrays(recs)
    c = recs[0]["config"]; R = (c["v_P"] - c["v_E"]) * c["t_max"] + c["eps"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.scatter(d0, Tc, s=9, color=C_P, alpha=0.5, label="trained $T_c$")
    ax.scatter(d0, Tbr, s=9, color=C_E, alpha=0.5, label="best response $T_{BR}$")
    xx = np.linspace(c["eps"], R, 50); ax.plot(xx, (xx - c["eps"]) / (c["v_P"] - c["v_E"]), "k-", lw=1.2, label=r"$V^*(\Delta_0)$")
    ax.axvline(R, color="0.5", lw=1); ax.text(R + 0.02, 1.7, "reach R", fontsize=8, color="0.4")
    ax.set_ylim(0, c["t_max"] + 0.1); ax.set_xlim(0.1, 1.45)
    ax.set_xlabel(r"initial distance $|\Delta_0|$"); ax.set_ylabel("capture time")
    ax.set_title(f"{label}: capture times on the training ICs", fontsize=9.5); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    fr = [capt[cap].mean(), (~np.isnan(Tbr[cap])).mean(), capt[~cap].mean(), (~np.isnan(Tbr[~cap])).mean()]
    x = np.array([0, 0.38, 1.1, 1.48])
    bars = ax.bar(x, fr, width=0.36, color=[C_P, C_E, C_P, C_E])
    for b, v in zip(bars, fr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{100 * v:.1f}%", ha="center", fontsize=8)
    ax.set_xticks([0.19, 1.29]); ax.set_xticklabels([f"capturable\n(n={int(cap.sum())})", f"uncapturable\n(n={int((~cap).sum())})"])
    ax.set_ylim(0, 1.12); ax.set_ylabel("fraction of heads captured"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("trained pursuer (orange) vs best response (blue)", fontsize=9.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"game.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarise(recs):
    d0, Tc, Tbr, cap, capt, Vs = game_arrays(recs)
    F = [r["full"]["14000"] for r in recs]
    g = lambda a, s, k: [f[a][s][k] for f in F]  # noqa: E731
    out = {"n_runs": len(recs), "capture_rate_capturable": float(capt[cap].mean()),
           "capture_rate_uncapturable": float(capt[~cap].mean()),
           "runs_all_capturable_captured": int(sum(f["game"]["n_captured_capturable"] == f["game"]["n_capturable"] for f in F)),
           "Tc_over_Vstar_captured": q(Tc[cap & capt] / Vs[cap & capt]),
           "nonnearest_chart_capturable": int(sum(1 for f in F for j, c in enumerate(f["game"]["capturable"])
                                                  if c and f["game"]["chart_closest"][j] != f["game"]["chart_opt"][j])),
           "band_ratio_P_over_E": q([f["P"]["band"]["band_11_50_U"] / f["E"]["band"]["band_11_50_U"] for f in F])}
    for a in "EP":
        for s, k in (("ens", "gram_pr"), ("ens", "constancy"), ("ens", "tpr_U"), ("ens", "tpr_Z"), ("lat", "pr_H"), ("lat", "pr_H_unc")):
            out[f"{k}_{a}"] = q(g(a, s, k))
    return out


def main():
    recs = load(RES)
    if not recs:
        sys.exit(f"no records in {NAME}/results: run `python3 torch_actor/experiments.py {NAME}` first")
    cfg = dict(recs[0]["config"]); cfg["truncate"] = recs[0].get("truncate", True); cfg["depth"] = 4
    json.dump(cfg, open(os.path.join(HERE, "config.json"), "w"), indent=1)
    figure_game(recs, NAME)
    D.LABEL["r1_trunc"] = "1-layer truncated baseline (warm-up 1500, lr 1e-2/1e-3)"
    D.main(RES, "r1_trunc", "dimensionalities", label=f"{NAME} (4-layer body, lr 1e-3/1e-4)", out_dir=FIG)
    V.main(os.path.join(HERE, "ckpt", "grid_i0_c0.npz"), FIG)
    summary = {NAME: summarise(recs)}
    ref = load(REF)
    if ref:
        summary["r1_trunc (1-layer truncated baseline, for comparison)"] = summarise(ref)
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    for name, o in summary.items():
        print(f"{name}: capture {100 * o['capture_rate_capturable']:.1f}%  all-captured runs {o['runs_all_capturable_captured']}/{o['n_runs']}  "
              f"Tc/V* {o['Tc_over_Vstar_captured'][0]:.3f}  Gram PR E/P {o['gram_pr_E'][0]:.2f}/{o['gram_pr_P'][0]:.2f}  "
              f"eta {o['constancy_E'][0]:.2f}/{o['constancy_P'][0]:.2f}  PR(H) E/P {o['pr_H_E'][0]:.2f}/{o['pr_H_P'][0]:.2f}  "
              f"PR(H) unc {o['pr_H_unc_E'][0]:.2f}/{o['pr_H_unc_P'][0]:.2f}  band P/E {o['band_ratio_P_over_E'][0]:.2f}")


if __name__ == "__main__":
    main()
