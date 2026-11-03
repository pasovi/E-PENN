"""Protocol split under the capture-truncated loss: warm-up {1500, 0} x steps {3:1, 1:1}.

Reads the four 100-run grids (checks/results/r1_trunc*, rev2 record format) and writes
  paper/figs/protocol_split_game.png/.pdf   capture time vs initial distance and capture
                                            fractions (the r1_game figure, one row per setup)
  paper/figs/protocol_split_pr.png/.pdf     PR(Z) and PR(H) of evader and pursuer per setup
  paper/results/protocol_split.json         the numbers behind both
Usage: python3 torch_actor/protocol_split.py
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "checks", "results")
FIG = os.path.join(HERE, "paper", "figs")
os.makedirs(FIG, exist_ok=True)
SETUPS = [("r1_trunc", "warm-up 1500, 3:1 (benchmark)"), ("r1_trunc_w1500_p1", "warm-up 1500, 1:1"),
          ("r1_trunc_w0_p3", "no warm-up, 3:1"), ("r1_trunc_w0_p1", "no warm-up, 1:1")]
C_E, C_P, C_BR = "#2a78d6", "#eb6834", "#1baf7a"


def load(tag):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, tag, "grid_*.json")))]


def q(v):
    v = np.asarray(v, float)
    return [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))]


def main():
    data = {tag: load(tag) for tag, _ in SETUPS}
    out = {}
    # ---------------------------------------------------------------- game figure
    fig, axes = plt.subplots(len(SETUPS), 2, figsize=(10, 3.4 * len(SETUPS)))
    for row, (tag, label) in enumerate(SETUPS):
        recs = data[tag]
        d0, Tc, Tbr, cap, capt, brc, Vs = [], [], [], [], [], [], []
        for r in recs:
            g = r["full"]["14000"]["game"]
            d0 += g["d0"]; Vs += g["Vstar"]; cap += g["capturable"]; capt += g["captured"]
            Tc += [np.nan if t is None else t for t in g["Tc"]]
            Tbr += [np.nan if t is None else t for t in g["T_BR"]]
        d0, Tc, Tbr, Vs = map(np.array, (d0, Tc, Tbr, Vs)); cap = np.array(cap); capt = np.array(capt)
        R = recs[0]["config"]["v_P"] - recs[0]["config"]["v_E"]; R = R * 4.0 + 0.08
        ax = axes[row, 0]
        ax.scatter(d0, Tc, s=9, color=C_P, alpha=0.5, label="trained $T_c$")
        ax.scatter(d0, Tbr, s=9, color=C_E, alpha=0.5, label="best response $T_{BR}$")
        xx = np.linspace(0.08, R, 50); ax.plot(xx, (xx - 0.08) / 0.2, "k-", lw=1.2, label=r"$V^*(\Delta_0)$")
        ax.axvline(R, color="0.5", lw=1); ax.text(R + 0.02, 1.7, "reach R", fontsize=8, color="0.4")
        ax.set_ylim(0, 4.1); ax.set_xlim(0.1, 1.45); ax.set_xlabel(r"initial distance $|\Delta_0|$"); ax.set_ylabel("capture time")
        ax.set_title(f"{label}: capture times on the training ICs", fontsize=9.5); ax.grid(alpha=0.3)
        if row == 0:
            ax.legend(fontsize=8, loc="upper left")
        ax = axes[row, 1]
        fr = [capt[cap].mean(), (~np.isnan(Tbr[cap])).mean(), capt[~cap].mean(), (~np.isnan(Tbr[~cap])).mean()]
        x = np.array([0, 0.38, 1.1, 1.48])
        bars = ax.bar(x, fr, width=0.36, color=[C_P, C_E, C_P, C_E])
        for b, v in zip(bars, fr):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{100 * v:.0f}%", ha="center", fontsize=8)
        ax.set_xticks([0.19, 1.29]); ax.set_xticklabels([f"capturable\n(n={int(cap.sum())})", f"uncapturable\n(n={int((~cap).sum())})"])
        ax.set_ylim(0, 1.12); ax.set_ylabel("fraction of heads captured"); ax.grid(alpha=0.3, axis="y")
        ax.set_title("trained pursuer (orange) vs best response (blue)", fontsize=9.5)
        ratio = Tc[cap & capt] / Vs[cap & capt]
        out[tag] = {"label": label, "n_runs": len(recs),
                    "capture_rate_capturable": float(capt[cap].mean()), "capture_rate_uncapturable": float(capt[~cap].mean()),
                    "Tc_over_Vstar_captured": q(ratio),
                    "runs_all_capturable_captured": int(sum(r["full"]["14000"]["game"]["n_captured_capturable"] == r["full"]["14000"]["game"]["n_capturable"] for r in recs)),
                    "nonnearest_chart_capturable": int(sum(1 for r in recs for j, c in enumerate(r["full"]["14000"]["game"]["capturable"])
                                                           if c and r["full"]["14000"]["game"]["chart_closest"][j] != r["full"]["14000"]["game"]["chart_opt"][j]))}
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"protocol_split_game.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ PR figure
    keys = [("PR(Z)", "E", lambda f: f["E"]["ens"]["tpr_Z"]), ("PR(Z)", "P", lambda f: f["P"]["ens"]["tpr_Z"]),
            ("PR(H)", "E", lambda f: f["E"]["lat"]["pr_H"]), ("PR(H)", "P", lambda f: f["P"]["lat"]["pr_H"])]
    extra = [("Gram PR", "E", lambda f: f["E"]["ens"]["gram_pr"]), ("Gram PR", "P", lambda f: f["P"]["ens"]["gram_pr"]),
             ("constancy", "E", lambda f: f["E"]["ens"]["constancy"]), ("constancy", "P", lambda f: f["P"]["ens"]["constancy"]),
             ("tPR(u)", "E", lambda f: f["E"]["ens"]["tpr_U"]), ("tPR(u)", "P", lambda f: f["P"]["ens"]["tpr_U"]),
             ("PR(H) unc", "E", lambda f: f["E"]["lat"]["pr_H_unc"]), ("PR(H) unc", "P", lambda f: f["P"]["lat"]["pr_H_unc"]),
             ("band ratio P/E", "", lambda f: f["P"]["band"]["band_11_50_U"] / f["E"]["band"]["band_11_50_U"])]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    rng = np.random.default_rng(0)
    xpos, xlab = [], []
    for s_i, (tag, label) in enumerate(SETUPS):
        recs = [r["full"]["14000"] for r in data[tag]]
        for k_i, (name, pl, fn) in enumerate(keys):
            x = s_i * 5 + k_i
            v = np.array([fn(f) for f in recs])
            col = C_E if pl == "E" else C_P
            ax.scatter(x + rng.uniform(-0.22, 0.22, len(v)), v, s=9, color=col, alpha=0.35, edgecolor="none")
            m, lo, hi = q(v)
            ax.plot([x - 0.32, x + 0.32], [m, m], color="k", lw=1.8)
            ax.plot([x, x], [lo, hi], color="k", lw=1.0)
            xpos.append(x); xlab.append(f"{name}\n{pl}")
            out[tag][f"{name}_{pl}"] = [m, lo, hi]
        for name, pl, fn in extra:
            out[tag][f"{name}_{pl}".rstrip("_")] = q([fn(f) for f in recs])
        ax.text(s_i * 5 + 1.5, 9.4, label, ha="center", fontsize=9, fontweight="bold")
        if s_i:
            ax.axvline(s_i * 5 - 0.5, color="0.8", lw=0.8)
    ax.axhline(2, color="0.4", ls=":", lw=1, label="constant-heading bound (2)")
    ax.set_xticks(xpos); ax.set_xticklabels(xlab, fontsize=8); ax.set_ylim(0.8, 10); ax.set_ylabel("participation ratio")
    ax.set_title("Strategy PR(Z) and latent PR(H), evader (blue) and pursuer (orange); 100 runs per setup, "
                 "median and IQR; capture-truncated loss", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"protocol_split_pr.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(HERE, "paper", "results", "protocol_split.json"), "w") as f:
        json.dump(out, f, indent=1)
    for tag, label in SETUPS:
        o = out[tag]
        print(f"{label:30s} capture {100*o['capture_rate_capturable']:.1f}%  all-captured runs {o['runs_all_capturable_captured']}/100  "
              f"wrong chart {o['nonnearest_chart_capturable']:3d}  Tc/V* {o['Tc_over_Vstar_captured'][0]:.3f}  "
              f"PR(Z) E/P {o['PR(Z)_E'][0]:.2f}/{o['PR(Z)_P'][0]:.2f}  PR(H) E/P {o['PR(H)_E'][0]:.2f}/{o['PR(H)_P'][0]:.2f}  "
              f"Gram {o['Gram PR_E'][0]:.2f}/{o['Gram PR_P'][0]:.2f}  eta {o['constancy_E'][0]:.2f}/{o['constancy_P'][0]:.2f}  "
              f"band P/E {o['band ratio P/E'][0]:.2f}")


if __name__ == "__main__":
    main()
