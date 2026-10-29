"""Aggregate the actor training-protocol experiment and render the figure."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

import os
HERE = os.path.dirname(os.path.abspath(__file__))
GP_IN = os.environ.get("GP_IN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_protocols_results_full.jsonl"))
rows = [json.loads(l) for l in open(GP_IN)]
protos = ["oracle", "selfplay", "simul", "abr"]
labels = {"oracle":"oracle (fixed optimal evader)",
          "selfplay":"self-play, 3:1 (paper baseline)",
          "simul":"simultaneous 1:1 (GDA)",
          "abr":"alternating best-response"}
col = {"oracle":"tab:green","selfplay":"tab:blue","simul":"tab:orange","abr":"tab:red"}

# aggregate traces (seeds share the checkpoint schedule)
def traces(p): return [r["trace"] for r in rows if r["protocol"]==p]
summary = {}
for p in protos:
    T = traces(p)
    L = min(len(t) for t in T)
    fr = np.array([t[0] for t in T[0][:L]])
    gap = np.array([[t[i][1] for i in range(L)] for t in T])   # (seeds, L)
    summary[p] = dict(frac=fr, gap_mean=gap.mean(0), gap_sd=gap.std(0),
        osc=[r["osc_sd_2ndhalf"] for r in rows if r["protocol"]==p],
        cap=[r["final_cap_of_capturable"] for r in rows if r["protocol"]==p],
        fgap=[r["final_gap_capturable"] for r in rows if r["protocol"]==p])

# save numeric summary
out = {p: {"final_cap_of_capturable_mean": round(float(np.mean(summary[p]["cap"])),3),
           "final_gap_capturable_mean": round(float(np.mean(summary[p]["fgap"])),4),
           "osc_sd_2ndhalf_mean": round(float(np.mean(summary[p]["osc"])),4),
           "osc_sd_2ndhalf_by_seed": [round(x,4) for x in summary[p]["osc"]]}
       for p in protos}
json.dump(out, open(os.path.join(HERE,"game_protocols_summary.json"),"w"), indent=2)
print(json.dumps(out, indent=2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
# (a) gap trace
for p in protos:
    s = summary[p]
    ax[0].plot(s["frac"], s["gap_mean"], "-", color=col[p], lw=2, label=labels[p])
    ax[0].fill_between(s["frac"], np.clip(s["gap_mean"]-s["gap_sd"],1e-3,None),
                       s["gap_mean"]+s["gap_sd"], color=col[p], alpha=0.15)
ax[0].axhline(0.08, color="k", ls=":", lw=1, label=r"capture radius $\varepsilon$")
ax[0].set_yscale("log"); ax[0].set_xlabel("training fraction")
ax[0].set_ylabel(r"mean pursuit gap on capturable ICs")
ax[0].set_title("(a) Convergence vs co-adaptation")
ax[0].legend(fontsize=7.2, loc="upper right")
# (b) oscillation amplitude
xs = np.arange(len(protos))
for i,p in enumerate(protos):
    o = summary[p]["osc"]
    ax[1].bar(i, np.mean(o), color=col[p], alpha=0.75)
    ax[1].plot([i]*len(o), o, "o", color="k", ms=4, zorder=5)
ax[1].set_xticks(xs); ax[1].set_xticklabels(["oracle","self-play\n3:1","simul\n1:1","ABR"], fontsize=8)
ax[1].set_ylabel(r"late-training oscillation (s.d. of gap)")
ax[1].set_title("(b) Cycling: absent under a fixed opponent, undamped under best-response", fontsize=10)
fig.tight_layout()
for pth in [os.path.join(HERE,"fig_protocols.png"),
            os.path.join(HERE,"fig_protocols.png")]:
    fig.savefig(pth, dpi=150, bbox_inches="tight")
# persist raw results into repo for the audit trail
print("figure + summary written to", HERE)
