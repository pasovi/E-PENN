"""Regenerate fig_shap.png from shap_seed{0,1,2}.json (grouped-Shapley
attribution of capture to input frequency bands, oracle vs self-play pursuer)."""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); bands=["linear","low","mid","high"]
S=[json.load(open(os.path.join(HERE,f"shap_seed{s}.json"))) for s in (0,1,2)]
def M(role): return np.array([[d[role]["shap_cap"][b] for b in bands] for d in S])
o=M("oracle"); sp=M("selfplay")
capO=np.mean([d["oracle"]["cap_full"] for d in S]); capS=np.mean([d["selfplay"]["cap_full"] for d in S])
x=np.arange(len(bands)); w=0.38
fig,ax=plt.subplots(figsize=(7,4.2))
ax.bar(x-w/2,o.mean(0),w,yerr=o.std(0),capsize=3,color="tab:green",alpha=0.85,label=f"oracle pursuer (capture {capO:.2f})")
ax.bar(x+w/2,sp.mean(0),w,yerr=sp.std(0),capsize=3,color="tab:red",alpha=0.85,label=f"self-play pursuer (capture {capS:.2f})")
ax.axhline(0,color="k",lw=0.8); ax.set_xticks(x)
ax.set_xticklabels(["linear\n(t)","low\n(k1-4)","mid\n(k5-8)","high\n(k9-12)"])
ax.set_ylabel("Shapley value for capture")
ax.set_title("Shapley attribution of pursuit capability to input bands\n(vs the same optimal evader; 3 seeds)")
ax.legend(fontsize=9,loc="upper right")
ax.annotate("oracle's capture lives in the\nlow-frequency band; the self-play\npursuer never learns it",
            (1,0.68),(1.4,0.5),fontsize=8,color="dimgray",arrowprops=dict(arrowstyle="->",color="gray",lw=1))
fig.tight_layout()
for p in [os.path.join(HERE,"fig_shap.png"), os.path.join(HERE,"fig_shap.png")]:
    fig.savefig(p,dpi=150,bbox_inches="tight")
print("wrote fig_shap.png")
