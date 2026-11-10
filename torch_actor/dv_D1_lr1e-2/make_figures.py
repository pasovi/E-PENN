"""Summary figures of the speed-difference sweep with a 1-layer body at lr_P = 1e-2, lr_E = 1e-3 (dv_D1 at 10x
its learning rates; run first: python3 torch_actor/experiments.py dv_D1_lr1e-2). A copy of dv_D1/make_figures.py;
the body depth is read from the checkpoints and the learning rates from the records.

Reads  dv_D1_lr1e-2/dv<dv>/results/grid_*.json   (one 100-run grid per dv = v_P - v_E, rev2 record format)
       dv_D1_lr1e-2/dv<dv>/ckpt/grid_i0_c0.npz   (to check the body depth)
Writes dv_D1_lr1e-2/figs/
         pr_vs_dv.png/.pdf        participation ratios vs dv, both players, median [IQR] over the 100 runs:
                                  (a) the draft's pair, strategy PR(Z) (tpr_Z) and latent PR(H) (centred pr_H)
                                  (b) their gauge-invariant counterparts, Gram PR and uncentred PR(H)
         capture_vs_dv.png/.pdf   capturable and captured shares vs dv against the analytic capturable share
                                  f(R), R = dv t_max + eps, with the break at rho = R / (L/2) = 1 and saturation
       dv_D1_lr1e-2/summary.json         every number in the figures
dv = 0.20 is the baseline speed (v_P = 1.2) and is highlighted in both figures.
Usage: python3 torch_actor/dv_D1_lr1e-2/make_figures.py
"""
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = os.path.basename(HERE)
FIG = os.path.join(HERE, "figs")
C_E, C_P = "#2a78d6", "#eb6834"          # evader / pursuer, as in every other figure of the repo
INK, MUTED = "#222222", "#8a8a8a"
BASE_DV = 0.20
BASE_LABEL = "baseline speed"            # the dv = 0.2 point (v_P = 1.2)
os.makedirs(FIG, exist_ok=True)


# ------------------------------------------------------------------ analytic geometry
def f_reach(R, L=2.0):
    """Share of the flat torus [0, L)^2 within torus distance R of the pursuer start: the capturable share for
    uniformly drawn ICs (area of the radius-R disc clipped to the L x L square, over L^2). Written for L = 2."""
    h = L / 2
    if R <= 0:
        return 0.0
    if R >= math.sqrt(2) * h:
        return 1.0
    if R <= h:
        return math.pi * R * R / (L * L)
    seg = R * R * math.acos(h / R) - h * math.sqrt(R * R - h * h)
    return (math.pi * R * R - 4.0 * seg) / (L * L)


def f_reach_ic(R, r0, L=2.0):
    """The same for the IC sampler actually used (uniform, rejected if |Delta0| < r0): f(R | |Delta0| >= r0)."""
    if R <= r0:
        return 0.0
    return (f_reach(R, L) * L * L - math.pi * r0 * r0) / (L * L - math.pi * r0 * r0)


# ------------------------------------------------------------------ data
def q(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))]


def body_depth(d, it):
    """Number of tanh layers in the body of run grid_i0_c0 (B, then B2, B3, ... in the checkpoint)."""
    z = np.load(os.path.join(d, "ckpt", "grid_i0_c0.npz"))
    return 1 + sum(1 for l in range(2, 50) if f"BE{l}_{it}" in z.files)


def load_point(d):
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "results", "grid_*.json")))]
    if not recs:
        return None
    c = recs[0]["config"]
    last = str(max(int(k) for k in recs[0]["full"]))
    F = [r["full"][last] for r in recs]
    G = [f["game"] for f in F]
    n_h = c["n_h"]
    n_cap = np.array([g["n_capturable"] for g in G]); n_got = np.array([g["n_captured"] for g in G])
    n_got_cap = np.array([g["n_captured_capturable"] for g in G])
    out = {"n_runs": len(recs), "iteration": int(last), "depth": body_depth(d, last), "v_P": c["v_P"],
           "dv": round(c["v_P"] - c["v_E"], 4),
           "R": (c["v_P"] - c["v_E"]) * c["t_max"] + c["eps"],
           "capturable_share": float(n_cap.sum() / (n_h * len(recs))),
           "captured_share": float(n_got.sum() / (n_h * len(recs))),
           "captured_share_runs": q(n_got / n_h),
           "captured_of_capturable": float(n_got_cap.sum() / n_cap.sum()) if n_cap.sum() else None,
           "captured_uncapturable": int((n_got - n_got_cap).sum()),
           "runs_all_capturable_captured": int(sum((g["n_captured_capturable"] == g["n_capturable"]) for g in G))}
    out["rho"] = out["R"] / (c["L"] / 2)
    out["f_R"] = f_reach(out["R"], c["L"]); out["f_R_ic"] = f_reach_ic(out["R"], c["ic_min_dist"], c["L"])
    for a in "EP":
        out[f"PR_Z_{a}"] = q([f[a]["ens"]["tpr_Z"] for f in F])
        out[f"PR_H_{a}"] = q([f[a]["lat"]["pr_H"] for f in F])
        out[f"gram_PR_{a}"] = q([f[a]["ens"]["gram_pr"] for f in F])
        out[f"PR_H_unc_{a}"] = q([f[a]["lat"]["pr_H_unc"] for f in F])
        out[f"constancy_{a}"] = q([f[a]["ens"]["constancy"] for f in F])
    return out, c


# ------------------------------------------------------------------ figures
def marks(ax, c, ytxt):
    """The break at rho = 1 and the saturation (every IC capturable), in dv units."""
    dv_ridge = (c["L"] / 2 - c["eps"]) / c["t_max"]
    dv_sat = (math.sqrt(2) * c["L"] / 2 - c["eps"]) / c["t_max"]
    xlim = ax.get_xlim()
    ax.axvspan(dv_sat, 1.0, color="0.93", zorder=0, lw=0)
    ax.set_xlim(*xlim)
    ax.axvline(dv_ridge, color=INK, ls="--", lw=1, zorder=1)
    ax.axvline(dv_sat, color=INK, ls=":", lw=1.2, zorder=1)
    if ytxt is None:                      # short horizontal labels in a free strip along the top of the panel
        yt = ax.get_ylim()[1] - 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(dv_ridge - 0.005, yt, rf"break $\rho=1$ ($\Delta v={dv_ridge:.3f}$)", ha="right", va="top", fontsize=7.5, color=INK)
        ax.text(dv_sat + 0.006, yt, rf"saturation $\rho=\sqrt{{2}}$ ($\Delta v={dv_sat:.3f}$)", ha="left", va="top",
                fontsize=7.5, color=INK)
        return dv_ridge, dv_sat
    ax.text(dv_ridge - 0.005, ytxt, rf"break $\rho=1$  ($\Delta v={dv_ridge:.3f}$)", rotation=90, va="center", ha="right",
            fontsize=8, color=INK)
    ax.text(dv_sat + 0.006, ytxt, rf"saturation $\rho=\sqrt{{2}}$  ($\Delta v={dv_sat:.3f}$)", rotation=90, va="center",
            ha="left", fontsize=8, color=INK)
    return dv_ridge, dv_sat


def rho_axis(ax, c):
    sec = ax.secondary_xaxis("top", functions=(lambda dv: (dv * c["t_max"] + c["eps"]) / (c["L"] / 2),
                                               lambda r: (r * c["L"] / 2 - c["eps"]) / c["t_max"]))
    sec.set_xlabel(r"$\rho = R/(L/2)$, $R = \Delta v\,t_{\max} + \varepsilon$", fontsize=9)
    sec.tick_params(labelsize=8)


def series(ax, dvs, P, key, color, marker, ls, label, dx=0.0):
    med = np.array([P[d][key][0] for d in dvs]); lo = np.array([P[d][key][1] for d in dvs]); hi = np.array([P[d][key][2] for d in dvs])
    x = np.array(dvs) + dx
    ax.errorbar(x, med, yerr=[med - lo, hi - med], color=color, marker=marker, ls=ls, lw=1.6, ms=6.5, capsize=2.5,
                elinewidth=1, mec="white", mew=0.8, label=label, zorder=3)
    if BASE_DV in dvs:                    # the baseline point: ringed and enlarged
        i = dvs.index(BASE_DV)
        ax.plot(x[i], med[i], marker=marker, ms=12, mfc=color, mec=INK, mew=1.6, ls="none", zorder=4)


def lr_text(c):
    """lr_P / lr_E as 1e-2 / 1e-3."""
    def e(x):
        m, p = f"{x:.0e}".split("e")
        return f"{m}e{int(p)}"
    return f"lr {e(c['lr_P'])} / {e(c['lr_E'])}"


def body_text(depth):
    return "1-layer body (16 tanh units)" if depth == 1 else f"{depth}-layer body (16 tanh units each)"


def figure_pr(dvs, P, c, depth):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    xlim = (0.0, max(dvs) + 0.04)
    panels = (("PR_Z", "PR_H", "(a) the draft's statistics: strategy PR(Z) (circles), centred latent PR(H) (squares)"),
              ("gram_PR", "PR_H_unc", "(b) gauge-invariant: strategy Gram PR (circles), uncentred latent PR(H) (squares)"))
    for ax, (kz, kh, title) in zip(axes, panels):
        ax.set_xlim(*xlim)
        series(ax, dvs, P, f"{kz}_E", C_E, "o", "-", "strategy ensemble, evader", dx=-0.0075)
        series(ax, dvs, P, f"{kz}_P", C_P, "o", "-", "strategy ensemble, pursuer", dx=-0.0025)
        series(ax, dvs, P, f"{kh}_E", C_E, "s", "--", "latent space, evader", dx=0.0025)
        series(ax, dvs, P, f"{kh}_P", C_P, "s", "--", "latent space, pursuer", dx=0.0075)
        if kz == "gram_PR":
            ax.axhline(2.0, color=MUTED, ls=":", lw=1, zorder=1)
            ax.text(xlim[1] - 0.004, 1.97, "constant-heading bound 2", fontsize=7.5, color=MUTED, ha="right", va="top")
        ymax = max(P[d][k][2] for d in dvs for k in (f"{kz}_E", f"{kz}_P", f"{kh}_E", f"{kh}_P"))
        ax.set_ylim(0.8, max(2.6, ymax * 1.16))
        marks(ax, c, None)
        rho_axis(ax, c)
        ax.set_xlabel(r"$\Delta v = v_P - v_E$"); ax.set_ylabel("participation ratio")
        ax.set_title(title, fontsize=9.5); ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", ms=11, mfc="white", mec=INK, mew=1.6, ls="none"))
    l.append(rf"{BASE_LABEL} ($\Delta v = 0.2$)")
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Speed-difference sweep, {body_text(depth)}, truncated loss, {lr_text(c)}: median [IQR] over "
                 f"{P[dvs[0]]['n_runs']} runs per $\\Delta v$, iteration {P[dvs[0]]['iteration']}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"pr_vs_dv.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_capture(dvs, P, c, depth):
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    xlim = (0.0, max(dvs) + 0.04); ax.set_xlim(*xlim)
    xx = np.linspace(0, xlim[1], 600); RR = xx * c["t_max"] + c["eps"]
    ax.plot(xx, [f_reach(R, c["L"]) for R in RR], color=INK, lw=2, label=r"analytic capturable share $f(R)$ (uniform ICs)", zorder=2)
    ax.plot(xx, [f_reach_ic(R, c["ic_min_dist"], c["L"]) for R in RR], color=INK, lw=1.3, ls=(0, (5, 3)),
            label=rf"analytic $f(R\,|\,|\Delta_0| \geq {c['ic_min_dist']:g})$ (the IC sampler)", zorder=2)
    x = np.array(dvs)
    cs = [P[d]["capturable_share"] for d in dvs]
    ax.plot(x, cs, "s", color=MUTED, ms=8, mec="white", mew=0.8, label="capturable share of the training ICs", zorder=3)
    got = np.array([P[d]["captured_share"] for d in dvs])
    ax.plot(x, got, "o-", color=C_P, ms=8, lw=1.6, mec="white", mew=0.8, label="captured share (trained pursuer)", zorder=4)
    rat = [P[d]["captured_of_capturable"] for d in dvs]
    xr = [d for d, r in zip(dvs, rat) if r is not None]; rr = [r for r in rat if r is not None]
    ax.plot(xr, rr, "^:", color=C_E, ms=8, lw=1.3, mec="white", mew=0.8, label="captured / capturable", zorder=4)
    if BASE_DV in dvs:
        i = dvs.index(BASE_DV)
        ax.plot(BASE_DV, got[i], "o", ms=14, mfc=C_P, mec=INK, mew=1.8, zorder=5)
        ax.plot(BASE_DV, rat[i], "^", ms=14, mfc=C_E, mec=INK, mew=1.8, zorder=5)
        ax.annotate(rf"{BASE_LABEL} ($\Delta v = 0.2$)", (BASE_DV, got[i]), xytext=(BASE_DV - 0.12, got[i] + 0.2),
                    fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    dv_ridge, dv_sat = marks(ax, c, 0.18)
    ax.text(dv_sat + 0.05, 0.5, "every IC\ncapturable", fontsize=8, color=MUTED, ha="left", va="center")
    rho_axis(ax, c)
    ax.set_ylim(-0.03, 1.07)
    ax.set_xlabel(r"$\Delta v = v_P - v_E$"); ax.set_ylabel("share of heads")
    ax.set_title(f"Capturable and captured shares vs $\\Delta v$, {body_text(depth)}, {lr_text(c)}\n"
                 f"({P[dvs[0]]['n_runs']} runs x {c['n_h']} heads per point)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6); ax.spines[["right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"capture_vs_dv.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dv_ridge, dv_sat


def main():
    P, c = {}, None
    for d in sorted(glob.glob(os.path.join(HERE, "dv*"))):
        if not os.path.isdir(d) or d.endswith("_smoke"):
            continue
        got = load_point(d)
        if got:
            P[got[0]["dv"]], c = got
    if not P:
        sys.exit(f"no results under {NAME}/dv*/results: run `python3 torch_actor/experiments.py {NAME}` first")
    dvs = sorted(P)
    depths = {P[d]["depth"] for d in dvs}
    if len(depths) != 1:
        sys.exit(f"mixed body depths across the sweep: {sorted(depths)}")
    depth = depths.pop()
    figure_pr(dvs, P, c, depth)
    dv_ridge, dv_sat = figure_capture(dvs, P, c, depth)
    json.dump({"depth": depth, "dv_break_rho1": dv_ridge, "dv_saturation": dv_sat, "baseline_dv": BASE_DV,
               "by_dv": {f"{d:.2f}": P[d] for d in dvs}}, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(f"{NAME}: {body_text(depth)}, {len(dvs)} dv points")
    print(f"{'dv':>5} {'rho':>5} {'f(R)':>5} {'f_ic':>5} {'capbl':>5} {'captd':>5} {'c/c':>5} {'all':>3} | "
          f"{'PRZ_E':>5} {'PRZ_P':>5} {'PRH_E':>5} {'PRH_P':>5} | {'Gram_E':>6} {'Gram_P':>6} {'Hu_E':>5} {'Hu_P':>5} | "
          f"{'eta_E':>5} {'eta_P':>5}")
    for d in dvs:
        o = P[d]; r = o["captured_of_capturable"]
        print(f"{d:5.2f} {o['rho']:5.2f} {o['f_R']:5.3f} {o['f_R_ic']:5.3f} {o['capturable_share']:5.3f} {o['captured_share']:5.3f} "
              f"{'  -  ' if r is None else f'{r:5.3f}'} {o['runs_all_capturable_captured']:3d} | {o['PR_Z_E'][0]:5.2f} {o['PR_Z_P'][0]:5.2f} "
              f"{o['PR_H_E'][0]:5.2f} {o['PR_H_P'][0]:5.2f} | {o['gram_PR_E'][0]:6.2f} {o['gram_PR_P'][0]:6.2f} "
              f"{o['PR_H_unc_E'][0]:5.2f} {o['PR_H_unc_P'][0]:5.2f} | {o['constancy_E'][0]:5.3f} {o['constancy_P'][0]:5.3f}")


if __name__ == "__main__":
    main()
