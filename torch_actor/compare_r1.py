"""Compare the final light metrics of a torch grid with the NumPy R1 ensemble.

The runs are chaotic, so individual runs are not expected to match; the ensemble
distributions are.  Reads <grid>/summary.json (from run_grid.py) and, if present,
the repository's rev2/results/r1/grid_*.json.
Usage: python3 torch_actor/compare_r1.py torch_actor/results/grid100
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.join(os.path.dirname(HERE), "rev2", "results", "r1")
KEYS = [("E", "gram_pr"), ("E", "constancy"), ("E", "pr_H"), ("E", "tpr_U"),
        ("P", "gram_pr"), ("P", "constancy"), ("P", "pr_H"), ("P", "tpr_U")]


def q(v):
    v = np.asarray(v, float)
    return f"{np.median(v):6.3f} [{np.percentile(v, 25):6.3f}, {np.percentile(v, 75):6.3f}]"


def main(grid_dir):
    s = json.load(open(os.path.join(grid_dir, "summary.json")))
    torch_final = [r["snapshots"][-1] for r in s["runs"]]
    r1 = [json.load(open(f))["snapshots"][-1] for f in sorted(glob.glob(os.path.join(R1, "grid_*.json")))]
    print(f"torch grid: {len(torch_final)} runs ({s['device']} {s['dtype']}, "
          f"{s['seconds_total']:.0f} s total);  R1 numpy grid: {len(r1)} runs")
    print(f"{'metric':16s} {'torch median [IQR]':28s} {'R1 median [IQR]':28s}")
    for a, k in KEYS:
        tv = [x[a][k] for x in torch_final]
        line = f"{a}.{k:14s} {q(tv):28s}"
        if r1:
            line += f" {q([x[a][k] for x in r1]):28s}"
        print(line)
    tv = [x["n_captured"] for x in torch_final]
    line = f"{'n_captured':16s} {q(tv):28s}"
    if r1:
        line += f" {q([x['n_captured'] for x in r1]):28s}"
    print(line)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "grid100"))
