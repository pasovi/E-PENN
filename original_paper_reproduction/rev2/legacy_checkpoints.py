"""Re-score every published actor checkpoint with the corrected metrics.

Writes rev2/results/legacy_corrected.json and prints a table of legacy vs
corrected statistics, so each number in the current draft can be traced to
its gauge-invariant replacement.

Usage: python3 rev2/legacy_checkpoints.py
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import ROOT, RunConfig  # noqa: E402
import metrics as M  # noqa: E402

# checkpoint -> (config overrides, iterations as trained)
CKPTS = {
    "v2_actor.npz": {},
    **{f"c1_s{i}.npz": {} for i in range(1, 8)},
    **{f"c1_ic{i}.npz": {} for i in range(1, 6)},
    "c3_k6_actor.npz": {"K": 6}, "c3_k24_actor.npz": {"K": 24},
    "c4_actor.npz": {},
    "c5_lam0_actor.npz": {"lam_d": 0.0}, "c5_lam1_actor.npz": {"lam_d": 1.0},
    "c5_lam5_actor.npz": {},
    "c7_fit.npz": {}, "c7_actor.npz": {},
    "e1_actor.npz": {}, "e1b_actor.npz": {}, "e1b_refl_actor.npz": {},
    "e7_actor.npz": {},
}


def main():
    out = {}
    print(f"{'checkpoint':20s} {'iters':>6s} | {'PR(Z)E':>6s} {'PR(Z)P':>6s} | {'GramE':>5s} {'GramP':>5s} "
          f"{'etaE':>5s} {'etaP':>5s} {'tPRuE':>5s} {'tPRuP':>5s} | {'PR(H)E':>6s} {'PR(H)P':>6s} "
          f"{'HuncE':>5s} {'HuncP':>5s} | ncap capt BR")
    for f, ov in CKPTS.items():
        z = np.load(os.path.join(ROOT, f))
        cfg = replace(RunConfig(), **ov)
        pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
        pP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
        m = M.full_metrics(pE, pP, z["x_E0"], cfg)
        g = m["game"]
        it = int(z["iter0"]) if "iter0" in z.files else -1
        row = {"iters": it, "config_overrides": ov, "saddle": M.saddle_reference(z["x_E0"], cfg)}
        for a in "EP":
            row[a] = {k: m[a]["ens"][k] for k in ("gram_pr", "gram_pr_hc", "constancy", "tpr_U", "tpr_Z")}
            row[a].update({k: m[a]["lat"][k] for k in ("pr_H", "pr_H_unc", "frac_le_K_H")})
            row[a]["erank_gf"] = m[a]["heads"]["erank_gf"]
            row[a]["band_11_50_U"] = m[a]["band"]["band_11_50_U"]
        row["game"] = {k: g[k] for k in ("n_capturable", "n_captured_capturable",
                                         "n_captured_uncapturable", "n_br_captured")}
        out[f] = row
        E, P = row["E"], row["P"]
        print(f"{f:20s} {it:6d} | {E['tpr_Z']:6.2f} {P['tpr_Z']:6.2f} | {E['gram_pr']:5.2f} {P['gram_pr']:5.2f} "
              f"{E['constancy']:5.2f} {P['constancy']:5.2f} {E['tpr_U']:5.2f} {P['tpr_U']:5.2f} | "
              f"{E['pr_H']:6.2f} {P['pr_H']:6.2f} {E['pr_H_unc']:5.2f} {P['pr_H_unc']:5.2f} | "
              f"{g['n_capturable']:4d} {g['n_captured_capturable']:4d} {g['n_br_captured']:3d}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "legacy_corrected.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
