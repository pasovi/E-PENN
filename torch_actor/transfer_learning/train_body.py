"""Train one multi-head actor with n_h heads (default 100 random ICs) under the baseline_D4 pipeline
(4-layer body, K = 12, capture-truncated loss, no warm-up, 3:1 steps, lr 1e-3 / 1e-4, Adam reset every
3,500), to be used as a frozen body for transfer learning.

Usage: python3 torch_actor/transfer_learning/train_body.py [--n-heads 100] [--init-seed 0] [--ic-seed 0]
            [--depth 4] [--lr-scale 0.1] [--warmup 0] [--device cpu] [--dtype float64] [--tag D4_h100_i0_c0]
Writes transfer_learning/bodies/<tag>.npz (rev2 key layout, loadable by pe_torch.load_ckpt and the TL script),
transfer_learning/bodies/<tag>.json (game and ensemble statistics of the training heads, rev2 record format)
and transfer_learning/figs/<tag>_game.pdf.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "checks"))
import pe_torch as pt  # noqa: E402
from core import RunConfig  # noqa: E402
import metrics as M  # noqa: E402
from tl import figure_game  # noqa: E402

BODIES = os.path.join(HERE, "bodies"); FIG = os.path.join(HERE, "figs")
os.makedirs(BODIES, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, default=100)
    ap.add_argument("--init-seed", type=int, default=0); ap.add_argument("--ic-seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=4); ap.add_argument("--lr-scale", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--dtype", default="float64")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"D{a.depth}_h{a.n_heads}_i{a.init_seed}_c{a.ic_seed}"
    dtype = {"float64": torch.float64, "float32": torch.float32}[a.dtype]
    cfg = pt.Config(truncate=True, depth=a.depth, n_h=a.n_heads, warmup=a.warmup)
    cfg = replace(cfg, lr_P=cfg.lr_P * a.lr_scale, lr_E=cfg.lr_E * a.lr_scale)
    print(f"{tag}: {cfg.n_h} heads, {cfg.depth}-layer body, warm-up {cfg.warmup}, {cfg.n_iters} x ({cfg.n_P_steps} P + {cfg.n_E_steps} E), "
          f"lr {cfg.lr_P:g} / {cfg.lr_E:g}, truncated loss; {a.device} {a.dtype}", flush=True)
    out = pt.train(cfg, runs=[(a.init_seed, a.ic_seed)], device=a.device, dtype=dtype)
    print(f"trained in {out['seconds']:.0f} s", flush=True)
    path = os.path.join(BODIES, tag + ".npz")
    pt.save_run_ckpt(path, cfg, out, 0)
    rcfg = replace(RunConfig(), n_h=cfg.n_h, warmup=cfg.warmup, lr_P=cfg.lr_P, lr_E=cfg.lr_E)
    pE = {"B": out["pE"]["B"][0], "b_body": out["pE"]["b"][0], "W": out["pE"]["W"][0]}
    pP = {"B": out["pP"]["B"][0], "b_body": out["pP"]["b"][0], "W": out["pP"]["W"][0]}
    for p, q in ((pE, out["pE"]), (pP, out["pP"])):
        for k in q:
            if k not in ("B", "b", "W"):
                p[k] = q[k][0]
    full = M.full_metrics(pE, pP, out["x0"][0], rcfg)
    g = full["game"]
    cap, capt = np.array(g["capturable"]), np.array(g["captured"])
    Tc, Vs = np.array([np.nan if t is None else t for t in g["Tc"]]), np.array(g["Vstar"])
    summary = {"n_heads": cfg.n_h, "n_capturable": int(cap.sum()), "capture_rate_capturable": float(capt[cap].mean()),
               "capture_rate_uncapturable": float(capt[~cap].mean()) if (~cap).any() else None,
               "Tc_over_Vstar_captured_median_q1_q3": [float(np.median(Tc[cap & capt] / Vs[cap & capt])),
                                                       float(np.percentile(Tc[cap & capt] / Vs[cap & capt], 25)),
                                                       float(np.percentile(Tc[cap & capt] / Vs[cap & capt], 75))],
               "nonnearest_chart_capturable": int(sum(1 for j, c in enumerate(cap) if c and g["chart_closest"][j] != g["chart_opt"][j])),
               "gram_pr_E": full["E"]["ens"]["gram_pr"], "gram_pr_P": full["P"]["ens"]["gram_pr"],
               "constancy_E": full["E"]["ens"]["constancy"], "constancy_P": full["P"]["ens"]["constancy"],
               "pr_H_E": full["E"]["lat"]["pr_H"], "pr_H_P": full["P"]["lat"]["pr_H"],
               "pr_H_unc_E": full["E"]["lat"]["pr_H_unc"], "pr_H_unc_P": full["P"]["lat"]["pr_H_unc"],
               "seconds": out["seconds"]}
    with open(os.path.join(BODIES, tag + ".json"), "w") as f:
        json.dump({"tag": tag, "config": asdict(cfg), "x_E0": out["x0"][0].tolist(), "summary": summary, "full": full,
                   "device": f"{a.device} {a.dtype}"}, f, indent=1)
    figure_game(g, rcfg, os.path.join(FIG, tag + "_game"), f"{tag} (training heads)")
    print(json.dumps(summary, indent=1)); print("wrote", path)


if __name__ == "__main__":
    main()
