"""Conditional (geometry-controlled) analysis of the transfer-learning errors on S^1, T^2, T^3, T^4.

Question: does the TL success rate improve with the dimension because of the dimension itself, or because the
capturable set is a smaller share of the sampled ICs for larger d (fixed reach R = 0.88 and box L = 2)?

For every capturable new IC x (pursuer at the origin) two geometric coordinates are computed:
  b = R - d0            margin to the reach boundary (d0 = geodesic distance = distance to the nearest image),
  m = d1 - d0           image margin (d1 = distance to the second-nearest periodic image of the pursuer);
an IC with small b sits in the outer shell of the capturable ball, an IC with small m has a competing image
(it sits near a cut-locus wall).  The TL errors on T^d (d >= 2) are heads chasing a non-nearest image, so
their rate should be a function of (b, m); whether it still depends on d after conditioning on (b, m) is the
question.  Three views:
  1. error counts and rates in coarse (b, m) cells, per dimension (nonparametric);
  2. a pooled logistic regression of error on (b, m) over d = 2, 3, 4, with and without a per-dimension
     intercept, and the likelihood-ratio test of the dimension term;
  3. direct standardisation: the error rate each dimension would show if its ICs had the T^2 conditional
     rates, and the share of the capturable ball in the "danger zone" (b < 0.1, m < 0.5) under uniform
     sampling (Monte Carlo), which is what a fair sampler would see, against what the grids actually sampled.
S^1 is reported but kept out of the pooled fit: its errors are heads running the wrong way, which has
nothing to do with m (every IC on S^1 has its competing image at d1 = L - d0).

Inputs: topologies/results/{S1,T3,T4}_tl.json and transfer_learning/results/D1_grid1024_seed0.json (T^2).
Outputs (this folder): results.json, summary.md, figs/conditional.pdf, figs/zone_shares.pdf.
"""
import itertools
import json
import os

import numpy as np
from scipy.stats import chi2, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(HERE)
TA = os.path.dirname(TOP)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
L, R = 2.0, 0.88
B_EDGES, M_EDGES = [0, 0.05, 0.1, 0.2, np.inf], [0, 0.5, 0.7, np.inf]
ZONE = dict(b=0.1, m=0.5)
COLORS = {1: "#7f7f7f", 2: "#1f77b4", 3: "#2ca02c", 4: "#d62728"}
NAME = {1: "$S^1$", 2: "$T^2$", 3: "$T^3$", 4: "$T^4$"}
SAMPLER = {1: "grid 100", 2: "grid 32²", 3: "grid 10³", 4: "Sobol 1000"}


def load(d):
    if d == 2:
        r = json.load(open(os.path.join(TA, "transfer_learning", "results", "D1_grid1024_seed0.json")))
        g, x = r["full"]["game"], np.array(r["x0_new"])
    else:
        r = json.load(open(os.path.join(TOP, "results", f"{'S1' if d == 1 else f'T{d}'}_tl.json")))
        g, x = r["game"], np.array(r["x0_new"])
    return x, np.array(g["capturable"]), np.array(g["captured"]), np.array(g["d0"])


def margins(x, d):
    offs = np.array(list(itertools.product([-1, 0, 1], repeat=d))) * L
    dist = np.sort(np.linalg.norm(x[:, None, :] - offs[None], axis=-1), axis=1)
    return R - dist[:, 0], dist[:, 1] - dist[:, 0], dist[:, 0]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, c - h, c + h)


def logistic_fit(X, y, iters=200):
    """Newton–Raphson logistic regression; X includes the intercept column. Returns beta, log-likelihood."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-X @ beta)), 1e-12, 1 - 1e-12)
    return beta, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mc_zone_share(d, n=400_000, seed=0):
    """Share of the capturable set (uniform on T^d, d0 in [0.4, R]) lying in the danger zone, by Monte Carlo."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, L, size=(n, d))
    b, m, d0 = margins(x, d)
    cap = (d0 >= 0.4) & (d0 <= R)
    zone = cap & (b < ZONE["b"]) & (m < ZONE["m"])
    return float(zone.sum() / cap.sum()), float(cap.mean())


def main():
    data, out = {}, {"reach": R, "L": L, "zone": ZONE, "b_edges": [e if np.isfinite(e) else None for e in B_EDGES],
                     "m_edges": [e if np.isfinite(e) else None for e in M_EDGES], "per_dim": {}}
    for d in (1, 2, 3, 4):
        x, cap, capt, d0 = load(d)
        b, m, d0c = margins(x, d)
        assert np.allclose(d0c, d0, atol=1e-4)
        err = cap & ~capt
        data[d] = dict(b=b[cap], m=m[cap], err=err[cap], n=int(cap.sum()))
        cells = []
        for i in range(len(B_EDGES) - 1):
            for j in range(len(M_EDGES) - 1):
                sel = (b[cap] >= B_EDGES[i]) & (b[cap] < B_EDGES[i + 1]) & (m[cap] >= M_EDGES[j]) & (m[cap] < M_EDGES[j + 1])
                cells.append(dict(b_bin=i, m_bin=j, n=int(sel.sum()), errors=int(err[cap][sel].sum())))
        zone = (b[cap] < ZONE["b"]) & (m[cap] < ZONE["m"])
        p_in, p_out = wilson(int(err[cap][zone].sum()), int(zone.sum())), wilson(int(err[cap][~zone].sum()), int((~zone).sum()))
        mc_share, mc_capfrac = mc_zone_share(d)
        out["per_dim"][d] = dict(sampler=SAMPLER[d], n_capturable=int(cap.sum()), n_errors=int(err.sum()),
                                 error_rate=wilson(int(err.sum()), int(cap.sum())),
                                 zone_share_sampled=float(zone.mean()), zone_share_uniform_mc=mc_share, capturable_fraction_uniform_mc=mc_capfrac,
                                 zone_n=int(zone.sum()), zone_errors=int(err[cap][zone].sum()), zone_error_rate=p_in,
                                 outside_n=int((~zone).sum()), outside_errors=int(err[cap][~zone].sum()), outside_error_rate=p_out,
                                 errors_b=b[err].round(4).tolist(), errors_m=m[err].round(4).tolist(), cells=cells)

    # ---- pooled logistic regression over d = 2, 3, 4
    Xb, Xm, y, dd = [], [], [], []
    for d in (2, 3, 4):
        Xb += list(data[d]["b"]); Xm += list(data[d]["m"]); y += list(data[d]["err"].astype(float)); dd += [d] * data[d]["n"]
    Xb, Xm, y, dd = map(np.array, (Xb, Xm, y, dd))
    X0 = np.column_stack([np.ones_like(Xb), Xb, Xm])
    X1 = np.column_stack([X0, (dd == 3).astype(float), (dd == 4).astype(float)])
    Xn = np.ones((len(y), 1))
    (bn, lln), (b0, ll0), (b1, ll1) = logistic_fit(Xn, y), logistic_fit(X0, y), logistic_fit(X1, y)
    lr_geom = 2 * (ll0 - lln); lr_dim = 2 * (ll1 - ll0)
    # per-dimension intercepts as odds ratios relative to T^2, and the standardised rates
    std = {}
    for d in (2, 3, 4):
        Xd = np.column_stack([np.ones(data[d]["n"]), data[d]["b"], data[d]["m"]])
        std[d] = dict(predicted_rate_T2_model=float(np.mean(1 / (1 + np.exp(-Xd @ b0)))), observed_rate=float(data[d]["err"].mean()),
                      expected_errors_T2_model=float(np.sum(1 / (1 + np.exp(-Xd @ b0)))), observed_errors=int(data[d]["err"].sum()))
    out["logistic"] = dict(features="logit p(error) = a + beta_b * b + beta_m * m", pooled_dims=[2, 3, 4], n=int(len(y)), errors=int(y.sum()),
                           beta_geometry=dict(a=float(b0[0]), beta_b=float(b0[1]), beta_m=float(b0[2])),
                           beta_with_dim=dict(a=float(b1[0]), beta_b=float(b1[1]), beta_m=float(b1[2]), gamma_T3=float(b1[3]), gamma_T4=float(b1[4]),
                                              odds_ratio_T3_vs_T2=float(np.exp(b1[3])), odds_ratio_T4_vs_T2=float(np.exp(b1[4]))),
                           loglik=dict(null=lln, geometry=ll0, geometry_plus_dim=ll1),
                           LR_geometry_vs_null=dict(stat=float(lr_geom), df=2, p=float(chi2.sf(lr_geom, 2))),
                           LR_dim_vs_geometry=dict(stat=float(lr_dim), df=2, p=float(chi2.sf(lr_dim, 2))),
                           standardised=std)
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)

    # ---- figure: (b, m) scatter per dimension, error rate vs b, vs m
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for k, d in enumerate((1, 2, 3, 4)):
        ax = axes[0, k]; D = data[d]
        ax.scatter(D["b"][~D["err"]], D["m"][~D["err"]], s=9, color=COLORS[d], alpha=0.45, linewidths=0, label="captured")
        ax.scatter(D["b"][D["err"]], D["m"][D["err"]], s=45, facecolor="none", edgecolor="k", lw=1.1, label=f"escaped ({int(D['err'].sum())})")
        ax.axvline(ZONE["b"], color="0.4", ls=":", lw=1); ax.axhline(ZONE["m"], color="0.4", ls=":", lw=1)
        ax.set_xlabel("reach margin $b = R - |\\Delta_0|$"); ax.set_ylabel("image margin $m = d_1 - d_0$")
        ax.set_title(f"{NAME[d]} TL ({SAMPLER[d]}): {D['n']} capturable ICs", fontsize=9.5); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(-0.01, 0.5); ax.set_ylim(0, 1.7)
    for k, (key, edges, lab) in enumerate((("b", [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5], "reach margin $b$"), ("m", [0.25, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.7], "image margin $m$"))):
        ax = axes[1, k]
        for d in (2, 3, 4):
            D = data[d]; xs, ps, lo, hi = [], [], [], []
            for i in range(len(edges) - 1):
                sel = (D[key] >= edges[i]) & (D[key] < edges[i + 1])
                if sel.sum() >= 3:
                    p, l, h = wilson(int(D["err"][sel].sum()), int(sel.sum())); xs.append(0.5 * (edges[i] + edges[i + 1])); ps.append(p); lo.append(l); hi.append(h)
            xs, ps, lo, hi = map(np.array, (xs, ps, lo, hi))
            ax.errorbar(xs + 0.004 * (d - 3), ps, yerr=[ps - lo, hi - ps], fmt="o-", color=COLORS[d], ms=5, lw=1.2, capsize=2, label=NAME[d])
        ax.set_xlabel(lab); ax.set_ylabel("error rate among capturable ICs"); ax.set_ylim(-0.02, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"error rate vs {lab}\n(bins with ≥ 3 ICs, 95 % Wilson intervals)", fontsize=9)
    ax = axes[1, 2]
    dims = [2, 3, 4]; w = 0.26
    ax.bar(np.array(dims) - w, [out["per_dim"][d]["zone_share_uniform_mc"] for d in dims], w, color="0.65", label="uniform sampling (Monte Carlo)")
    ax.bar(np.array(dims), [out["per_dim"][d]["zone_share_sampled"] for d in dims], w, color=[COLORS[d] for d in dims], label="actual TL sample")
    ax.bar(np.array(dims) + w, [out["per_dim"][d]["error_rate"][0] for d in dims], w, color=[COLORS[d] for d in dims], alpha=0.4, hatch="//", label="observed error rate")
    ax.set_xticks(dims); ax.set_xticklabels([NAME[d] for d in dims]); ax.set_ylabel("share of capturable ICs"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    ax.set_title(f"danger zone (b < {ZONE['b']}, m < {ZONE['m']}):\nshare of capturable ICs", fontsize=9)
    ax = axes[1, 3]
    for d in dims:
        s = std[d]
        ax.bar(d - 0.18, s["observed_rate"], 0.34, color=COLORS[d], label="observed" if d == 2 else None)
        ax.bar(d + 0.18, s["predicted_rate_T2_model"], 0.34, color=COLORS[d], alpha=0.4, hatch="//", label="pooled geometry model (no dimension term)" if d == 2 else None)
        ax.text(d - 0.18, s["observed_rate"] + 0.004, f"{s['observed_errors']}", ha="center", fontsize=8)
        ax.text(d + 0.18, s["predicted_rate_T2_model"] + 0.004, f"{s['expected_errors_T2_model']:.1f}", ha="center", fontsize=8)
    ax.set_xticks(dims); ax.set_xticklabels([NAME[d] for d in dims]); ax.set_ylabel("error rate among capturable ICs"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    ax.set_title(f"geometry-standardised error rates\n(LR test of a dimension term: p = {out['logistic']['LR_dim_vs_geometry']['p']:.2f})", fontsize=9)
    fig.suptitle("TL errors conditioned on geometry: reach margin b and image margin m, per dimension (S¹ shown, not pooled)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "conditional.pdf"), bbox_inches="tight"); plt.close(fig)

    # ---- markdown summary
    lines = ["# Conditional analysis of the TL errors\n", f"Danger zone: b < {ZONE['b']} (outer shell of the capturable ball) and m < {ZONE['m']} (a competing image).\n",
             "| | S¹ | T² | T³ | T⁴ |", "|---|---|---|---|---|",
             "| TL sampler | " + " | ".join(SAMPLER[d] for d in (1, 2, 3, 4)) + " |",
             "| capturable ICs / errors | " + " | ".join(f"{out['per_dim'][d]['n_capturable']} / {out['per_dim'][d]['n_errors']}" for d in (1, 2, 3, 4)) + " |",
             "| error rate (95 % CI) | " + " | ".join("%.1f %% [%.1f, %.1f]" % tuple(100 * v for v in out['per_dim'][d]['error_rate']) for d in (1, 2, 3, 4)) + " |",
             "| capturable fraction, uniform sampling | " + " | ".join(f"{100 * out['per_dim'][d]['capturable_fraction_uniform_mc']:.1f} %" for d in (1, 2, 3, 4)) + " |",
             "| danger-zone share of capturable ICs, uniform | " + " | ".join(f"{100 * out['per_dim'][d]['zone_share_uniform_mc']:.1f} %" for d in (1, 2, 3, 4)) + " |",
             "| danger-zone share, actual sample | " + " | ".join(f"{100 * out['per_dim'][d]['zone_share_sampled']:.1f} %" for d in (1, 2, 3, 4)) + " |",
             "| errors / ICs inside the zone | " + " | ".join(f"{out['per_dim'][d]['zone_errors']} / {out['per_dim'][d]['zone_n']}" for d in (1, 2, 3, 4)) + " |",
             "| errors / ICs outside the zone | " + " | ".join(f"{out['per_dim'][d]['outside_errors']} / {out['per_dim'][d]['outside_n']}" for d in (1, 2, 3, 4)) + " |",
             "", "## Cells (b bin × m bin): errors / ICs\n", "b bins: " + ", ".join(f"[{B_EDGES[i]}, {B_EDGES[i + 1]})" for i in range(len(B_EDGES) - 1)) +
             "; m bins: " + ", ".join(f"[{M_EDGES[j]}, {M_EDGES[j + 1]})" for j in range(len(M_EDGES) - 1)) + "\n",
             "| b bin | m bin | S¹ | T² | T³ | T⁴ |", "|---|---|---|---|---|---|"]
    for c in range(len(out["per_dim"][2]["cells"])):
        cc = out["per_dim"][2]["cells"][c]
        lines.append(f"| {cc['b_bin']} | {cc['m_bin']} | " + " | ".join(f"{out['per_dim'][d]['cells'][c]['errors']} / {out['per_dim'][d]['cells'][c]['n']}" for d in (1, 2, 3, 4)) + " |")
    lg = out["logistic"]
    lines += ["", "## Pooled logistic regression (T², T³, T⁴)\n",
              f"logit p = {lg['beta_geometry']['a']:.2f} + ({lg['beta_geometry']['beta_b']:.2f}) b + ({lg['beta_geometry']['beta_m']:.2f}) m  on {lg['n']} capturable ICs, {lg['errors']} errors.",
              f"Geometry vs null: LR = {lg['LR_geometry_vs_null']['stat']:.1f} (df 2), p = {lg['LR_geometry_vs_null']['p']:.1e}.",
              f"Dimension term on top of geometry: LR = {lg['LR_dim_vs_geometry']['stat']:.2f} (df 2), p = {lg['LR_dim_vs_geometry']['p']:.3f}; "
              f"odds ratios vs T²: T³ {lg['beta_with_dim']['odds_ratio_T3_vs_T2']:.2f}, T⁴ {lg['beta_with_dim']['odds_ratio_T4_vs_T2']:.2f}.", "",
              "| | observed errors | expected from the geometry-only model | observed rate | standardised rate |", "|---|---|---|---|---|"]
    for d in (2, 3, 4):
        s = std[d]; lines.append(f"| T{d} | {s['observed_errors']} | {s['expected_errors_T2_model']:.1f} | {100 * s['observed_rate']:.1f} % | {100 * s['predicted_rate_T2_model']:.1f} % |")
    open(os.path.join(HERE, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
