"""Control TL run: new ICs sampled uniformly INSIDE the capturable shell, the same number in every dimension.

The TL runs of the pipeline sample the whole torus (grids, Sobol), so the capturable heads are a different
share of the sample in every dimension and the samplers carry their own biases (conditional.py).  Here, for
d = 1, 2, 3, 4, N ICs are drawn uniformly in {x : 0.4 <= |Delta_0| <= R} (uniform in volume), transferred onto
the frozen MH bodies of the pipeline (T^2: the D1 truncated baseline checkpoint, S^1/T^3/T^4: results/*_mh.npz)
with the same protocol, for several head-initialisation seeds; then the same conditional analysis (reach
margin b, image margin m, pooled logistic fit with and without a dimension term).

Usage: python3 torch_actor/topologies/conditional_analysis/control_ball.py [--n 500] [--seeds 0,1,2] [--dims 1,2,3,4]
       [--device mps --dtype float32] [--analyse-only]
Outputs (this folder): results/ball_T<d>_seed<s>.json, results_ball.json, summary_ball.md, figs/conditional_ball.pdf
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(HERE); TA = os.path.dirname(TOP)
sys.path.insert(0, TOP); sys.path.insert(0, HERE)
import pe_nd as nd  # noqa: E402
from conditional import margins, wilson, logistic_fit, L, R, ZONE, COLORS, NAME  # noqa: E402

RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figs")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def body(d):
    if d == 2:
        z = np.load(os.path.join(TA, "paper", "ckpt", "baseline_trunc", "grid_i0_c0.npz"))
    else:
        z = np.load(os.path.join(TOP, "results", f"{'S1' if d == 1 else f'T{d}'}_mh.npz"))
    return {"B": z["BE"], "b": z["bE"]}, {"B": z["BP"], "b": z["bP"]}


def shell_ICs(n, d, seed, r_in=0.4, r_out=R):
    """Uniform in volume on the shell r_in <= |x| <= r_out around the pursuer, mapped to [0, L)^d."""
    rng = np.random.default_rng(seed); pts = []
    while len(pts) < n:
        x = rng.uniform(-r_out, r_out, size=(4 * n, d)); r = np.linalg.norm(x, axis=1)
        pts += list(x[(r >= r_in) & (r <= r_out)])
    return np.array(pts[:n]) % L


def run(d, n, seeds, device, dtype):
    bE, bP = body(d); x0 = shell_ICs(n, d, 0)
    cfg = nd.Config(truncate=True, n_h=n)
    for s in seeds:
        out = os.path.join(RES, f"ball_T{d}_seed{s}.json")
        if os.path.exists(out):
            print("exists", out); continue
        rng = np.random.default_rng(s)
        pE = dict(bE, W=rng.standard_normal((n, d, cfg.n_b)) * np.sqrt(1.0 / cfg.n_b))
        pP = dict(bP, W=rng.standard_normal((n, d, cfg.n_b)) * np.sqrt(1.0 / cfg.n_b))
        t0 = time.time()
        pE, pP, _, _, secs = nd.train(cfg, pE, pP, x0, device, dtype, frozen_body=True, log_every=0)
        g = nd.game_stats(pE, pP, x0, cfg)
        prE, prP = nd.pr_stats(pE, cfg), nd.pr_stats(pP, cfg)
        summ = nd.summarise(g, prE, prP); summ["seconds"] = secs
        json.dump({"dim": d, "n": n, "ic_seed": 0, "head_seed": s, "sampling": f"uniform in 0.4 <= |Delta0| <= {R}", "x0_new": x0.tolist(),
                   "summary": summ, "game": {k: v for k, v in g.items() if k not in ("xE", "xP")}}, open(out, "w"), indent=1)
        print(f"T^{d} seed {s}: {summ['n_capturable']} capturable, {summ['n_capturable_escaped']} escaped "
              f"({100 * summ['capture_rate_capturable']:.1f} % captured), {secs:.0f} s", flush=True)


def analyse(dims, seeds):
    data, per = {}, {}
    for d in dims:
        b_, m_, e_, s_ = [], [], [], []
        for s in seeds:
            r = json.load(open(os.path.join(RES, f"ball_T{d}_seed{s}.json"))); g = r["game"]; x = np.array(r["x0_new"])
            cap, capt = np.array(g["capturable"]), np.array(g["captured"]); assert cap.all()
            b, m, _ = margins(x, d); b_ += list(b); m_ += list(m); e_ += list(~capt); s_ += [s] * len(x)
        b, m, e, s_ = map(np.array, (b_, m_, e_, s_))
        data[d] = dict(b=b, m=m, err=e, seed=s_)
        zone = (b < ZONE["b"]) & (m < ZONE["m"])
        per[d] = dict(n_per_seed=int(len(b) / len(seeds)), seeds=list(seeds), errors_per_seed=[int(e[s_ == s].sum()) for s in seeds],
                      error_rate_pooled=wilson(int(e.sum()), int(len(e))), zone_share=float(zone.mean()),
                      zone=dict(errors=int(e[zone].sum()), n=int(zone.sum()), rate=wilson(int(e[zone].sum()), int(zone.sum()))),
                      outside=dict(errors=int(e[~zone].sum()), n=int((~zone).sum()), rate=wilson(int(e[~zone].sum()), int((~zone).sum()))),
                      nonnearest_share_of_errors=None)
    tori = [d for d in dims if d >= 2]
    Xb = np.concatenate([data[d]["b"] for d in tori]); Xm = np.concatenate([data[d]["m"] for d in tori])
    y = np.concatenate([data[d]["err"] for d in tori]).astype(float); dd = np.concatenate([[d] * len(data[d]["b"]) for d in tori])
    X0 = np.column_stack([np.ones_like(Xb), Xb, Xm]); X1 = np.column_stack([X0] + [(dd == d).astype(float) for d in tori[1:]])
    (_, lln), (b0, ll0), (b1, ll1) = logistic_fit(np.ones((len(y), 1)), y), logistic_fit(X0, y), logistic_fit(X1, y)
    lr_g, lr_d = 2 * (ll0 - lln), 2 * (ll1 - ll0)
    std = {}
    for d in tori:
        Xd = np.column_stack([np.ones(len(data[d]["b"])), data[d]["b"], data[d]["m"]])
        std[d] = dict(observed_rate=float(data[d]["err"].mean()), standardised_rate=float(np.mean(1 / (1 + np.exp(-Xd @ b0)))),
                      observed_errors=int(data[d]["err"].sum()), expected_errors=float(np.sum(1 / (1 + np.exp(-Xd @ b0)))))
    out = dict(sampling=f"uniform in 0.4 <= |Delta0| <= {R}", per_dim=per, logistic=dict(
        beta_geometry=dict(a=float(b0[0]), beta_b=float(b0[1]), beta_m=float(b0[2])),
        dim_odds_ratios_vs_T2={f"T{d}": float(np.exp(b1[3 + i])) for i, d in enumerate(tori[1:])},
        LR_geometry_vs_null=dict(stat=float(lr_g), df=2, p=float(chi2.sf(lr_g, 2))),
        LR_dim_vs_geometry=dict(stat=float(lr_d), df=len(tori) - 1, p=float(chi2.sf(lr_d, len(tori) - 1))), standardised=std))
    json.dump(out, open(os.path.join(HERE, "results_ball.json"), "w"), indent=1)

    # figure
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for k, d in enumerate(dims):
        ax = axes[0, k]; D = data[d]; first = D["seed"] == seeds[0]
        ax.scatter(D["b"][first & ~D["err"]], D["m"][first & ~D["err"]], s=9, color=COLORS[d], alpha=0.45, linewidths=0, label=f"captured (seed {seeds[0]})")
        ax.scatter(D["b"][D["err"]], D["m"][D["err"]], s=40, facecolor="none", edgecolor="k", lw=1.0, label=f"escaped, all seeds ({int(D['err'].sum())})")
        ax.axvline(ZONE["b"], color="0.4", ls=":", lw=1); ax.axhline(ZONE["m"], color="0.4", ls=":", lw=1)
        ax.set_xlabel("reach margin $b = R - |\\Delta_0|$"); ax.set_ylabel("image margin $m = d_1 - d_0$"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{NAME[d]}: {per[d]['n_per_seed']} ICs in the capturable shell × {len(seeds)} seeds", fontsize=9.5); ax.set_xlim(-0.01, 0.5); ax.set_ylim(0, 1.7)
    for k, (key, edges, lab) in enumerate((("b", [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5], "reach margin $b$"), ("m", [0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.7], "image margin $m$"))):
        ax = axes[1, k]
        for d in dims:
            D = data[d]; xs, ps, lo, hi = [], [], [], []
            for i in range(len(edges) - 1):
                sel = (D[key] >= edges[i]) & (D[key] < edges[i + 1])
                if sel.sum() >= 5:
                    p, l, h = wilson(int(D["err"][sel].sum()), int(sel.sum())); xs.append(0.5 * (edges[i] + edges[i + 1])); ps.append(p); lo.append(l); hi.append(h)
            xs, ps, lo, hi = map(np.array, (xs, ps, lo, hi))
            ax.errorbar(xs + 0.004 * (d - 2.5), ps, yerr=[ps - lo, hi - ps], fmt="o-", color=COLORS[d], ms=5, lw=1.2, capsize=2, label=NAME[d])
        ax.set_xlabel(lab); ax.set_ylabel("error rate"); ax.set_ylim(-0.02, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"error rate vs {lab}\n(bins with ≥ 5 ICs, 95 % Wilson intervals)", fontsize=9)
    ax = axes[1, 2]; w = 0.38
    ax.bar(np.array(dims) - w / 2, [per[d]["zone_share"] for d in dims], w, color=[COLORS[d] for d in dims], label="danger-zone share of the ICs")
    ax.bar(np.array(dims) + w / 2, [per[d]["error_rate_pooled"][0] for d in dims], w, color=[COLORS[d] for d in dims], alpha=0.4, hatch="//", label="error rate (pooled over seeds)")
    for d in dims:
        lo, hi = per[d]["error_rate_pooled"][1:]; ax.plot([d + w / 2] * 2, [lo, hi], color="k", lw=1)
        ax.text(d + w / 2, hi + 0.004, "/".join(str(e) for e in per[d]["errors_per_seed"]), ha="center", fontsize=7.5)
    ax.set_xticks(dims); ax.set_xticklabels([NAME[d] for d in dims]); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    ax.set_title(f"uniform in the capturable shell: zone share (b < {ZONE['b']}, m < {ZONE['m']})\nand error rate (errors per seed above the bars)", fontsize=9)
    ax = axes[1, 3]
    for d in tori:
        s = std[d]
        ax.bar(d - 0.18, s["observed_rate"], 0.34, color=COLORS[d], label="observed" if d == tori[0] else None)
        ax.bar(d + 0.18, s["standardised_rate"], 0.34, color=COLORS[d], alpha=0.4, hatch="//", label="pooled geometry model (no dimension term)" if d == tori[0] else None)
        ax.text(d - 0.18, s["observed_rate"] + 0.003, f"{s['observed_errors']}", ha="center", fontsize=8); ax.text(d + 0.18, s["standardised_rate"] + 0.003, f"{s['expected_errors']:.1f}", ha="center", fontsize=8)
    ax.set_xticks(tori); ax.set_xticklabels([NAME[d] for d in tori]); ax.set_ylabel("error rate"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    ax.set_title(f"geometry-standardised error rates\n(LR test of a dimension term: p = {out['logistic']['LR_dim_vs_geometry']['p']:.2f})", fontsize=9)
    fig.suptitle("Control: TL heads on ICs uniform inside the capturable shell, same count in every dimension (S¹ shown, not pooled)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "conditional_ball.pdf"), bbox_inches="tight"); plt.close(fig)

    lines = ["# Control: TL on ICs uniform inside the capturable shell\n", f"{per[dims[0]]['n_per_seed']} ICs per dimension in 0.4 ≤ |Δ0| ≤ {R} (IC seed 0), head seeds {list(seeds)}, same protocol, frozen pipeline bodies.\n",
             "| | " + " | ".join(NAME[d].replace("$", "") for d in dims) + " |", "|---|" + "---|" * len(dims),
             "| errors per seed (of " + str(per[dims[0]]["n_per_seed"]) + ") | " + " | ".join(" / ".join(str(e) for e in per[d]["errors_per_seed"]) for d in dims) + " |",
             "| error rate pooled (95 % CI) | " + " | ".join("%.1f %% [%.1f, %.1f]" % tuple(100 * v for v in per[d]["error_rate_pooled"]) for d in dims) + " |",
             "| danger-zone share of the ICs | " + " | ".join(f"{100 * per[d]['zone_share']:.1f} %" for d in dims) + " |",
             "| errors / ICs inside the zone (rate) | " + " | ".join(f"{per[d]['zone']['errors']} / {per[d]['zone']['n']} ({100 * per[d]['zone']['rate'][0]:.0f} %)" for d in dims) + " |",
             "| errors / ICs outside the zone (rate) | " + " | ".join(f"{per[d]['outside']['errors']} / {per[d]['outside']['n']} ({100 * per[d]['outside']['rate'][0]:.1f} %)" for d in dims) + " |",
             "", f"Pooled logistic fit over {', '.join('T' + str(d) for d in tori)}: logit p = {b0[0]:.2f} + ({b0[1]:.2f}) b + ({b0[2]:.2f}) m; geometry vs null p = {out['logistic']['LR_geometry_vs_null']['p']:.1e}; "
             f"dimension term on top: LR = {lr_d:.2f} ({len(tori) - 1} df), p = {out['logistic']['LR_dim_vs_geometry']['p']:.3f}; odds ratios vs T²: " +
             ", ".join(f"{k} {v:.2f}" for k, v in out["logistic"]["dim_odds_ratios_vs_T2"].items()) + ".", "",
             "| | observed errors | expected (geometry only) | observed rate | standardised rate |", "|---|---|---|---|---|"]
    for d in tori:
        s = std[d]; lines.append(f"| T{d} | {s['observed_errors']} | {s['expected_errors']:.1f} | {100 * s['observed_rate']:.2f} % | {100 * s['standardised_rate']:.2f} % |")
    open(os.path.join(HERE, "summary_ball.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--dims", default="1,2,3,4")
    ap.add_argument("--device", default="mps"); ap.add_argument("--dtype", default="float32"); ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    dims, seeds = [int(d) for d in a.dims.split(",")], [int(s) for s in a.seeds.split(",")]
    if not a.analyse_only:
        for d in dims:
            run(d, a.n, seeds, a.device, {"float64": torch.float64, "float32": torch.float32}[a.dtype])
    analyse(dims, seeds)


if __name__ == "__main__":
    main()
