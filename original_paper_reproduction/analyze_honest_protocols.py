"""Honest convergent-protocol comparison: quality-stability plane + traces.

Reads matched-budget (6,000-iter, 3 seeds) results for the honest self-play
corrections vs the oracle ceiling, and renders the finding that none of the
standard game-optimisation fixes reaches the oracle's stability while keeping
full capture on this non-convex game.
"""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
IN6 = os.environ.get("HP_IN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "honest_protocols_results.jsonl"))

rows = [json.loads(l) for l in open(IN6)]
by = defaultdict(list)
for r in rows: by[r["protocol"]].append(r)

lab = {"oracle":"oracle (non-honest ceiling)","gda":"GDA (simultaneous)",
       "ogda":"optimistic GDA","eg":"extragradient","avg":"Cesaro averaging",
       "ogda_ema":"optimistic GDA + EMA"}
col = {"oracle":"tab:green","gda":"tab:red","ogda":"tab:blue",
       "eg":"tab:orange","avg":"tab:purple","ogda_ema":"tab:brown"}
order = ["oracle","gda","ogda","eg","avg","ogda_ema"]

summary = {}
for p in order:
    rs = by.get(p, [])
    if not rs: continue
    summary[p] = dict(
        cap=round(float(np.mean([r["final_cap_of_capturable"] for r in rs])),3),
        gap=round(float(np.mean([r["final_gap_capturable"] for r in rs])),4),
        osc=round(float(np.mean([r["osc_sd_2ndhalf"] for r in rs])),4),
        n=len(rs))
json.dump(summary, open(os.path.join(HERE,"honest_protocols_summary.json"),"w"), indent=2)
print(json.dumps(summary, indent=2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
# (a) quality-stability plane
for p in order:
    rs = by.get(p, [])
    if not rs: continue
    caps=[r["final_cap_of_capturable"] for r in rs]; oscs=[r["osc_sd_2ndhalf"] for r in rs]
    ax[0].scatter(caps, oscs, s=28, color=col[p], alpha=0.55, edgecolor="none")
    ax[0].scatter([np.mean(caps)],[np.mean(oscs)], s=150, color=col[p],
                  edgecolor="k", zorder=5, label=lab[p], marker="o")
ax[0].set_yscale("log"); ax[0].set_xlabel("final capture of capturable set  (→ 1 better)")
ax[0].set_ylabel("late-training oscillation  (↓ better)")
budget=rows[0]["iters"]; nseed=max(len(v) for v in by.values())
ax[0].set_title(f"(a) Quality–stability plane ({budget} iters, {nseed} seeds)")
ax[0].axhspan(1e-4, 2e-3, color="tab:green", alpha=0.06)
ax[0].annotate("oracle basin\n(stable + full capture)", (0.62, 0.0009), fontsize=7.5,
               color="tab:green", va="center")
ax[0].legend(fontsize=7, loc="lower left"); ax[0].set_xlim(0.2, 1.05); ax[0].set_ylim(4e-4, 2e-1)
# (b) representative gap traces (seed 0)
def trace0(p):
    rs=[r for r in by.get(p,[]) if r["seed"]==0]
    return rs[0]["trace"] if rs else None
for p in ["oracle","gda","ogda","avg"]:
    t=trace0(p)
    if not t: continue
    fr=[x[0] for x in t]; gp=[x[1] for x in t]
    ax[1].plot(fr, gp, "-", color=col[p], lw=1.8, label=lab[p], alpha=0.9)
ax[1].axhline(0.08, color="k", ls=":", lw=1, label=r"capture radius $\varepsilon$")
ax[1].set_yscale("log"); ax[1].set_xlabel("training fraction")
ax[1].set_ylabel("pursuit gap on capturable ICs")
ax[1].set_title("(b) Representative dynamics (seed 0)")
ax[1].legend(fontsize=7, loc="upper right")
fig.tight_layout()
for pth in [os.path.join(HERE,"fig_honest.png"), os.path.join(HERE,"fig_honest.png")]:
    fig.savefig(pth, dpi=150, bbox_inches="tight")
print("wrote fig_honest.png + honest_protocols_summary.json")
