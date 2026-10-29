"""R1: baseline ensemble under one deterministic protocol.

Arms (120 runs, all at the published baseline configuration otherwise):
  grid     init_seed in 0..9  x  ic_seed in 0..9          (100 runs)
  chaos    init 0, ic 0, init params + 1e-10 * N(0,1),
           perturbation seeds 1..10                       (10 runs)
  noreset  init 0..9, ic 0, a single Adam for the whole run
           (the grid uses a fresh Adam every 3,500 iterations,
           as the published baseline did)                 (10 runs)

Outputs (resumable; finished runs are skipped):
  rev2/results/r1/<name>.json   config, full metrics at 3.5k/7k/10.5k/14k,
                                light metrics every 1,000 iterations
  rev2/ckpt/r1/<name>.npz       parameters at 3.5k/7k/10.5k/14k

Usage: python3 rev2/run_r1.py [n_workers]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import RunConfig, train_actor, params_digest, save_ckpts  # noqa: E402
import metrics as M  # noqa: E402

RES = os.path.join(HERE, "results", "r1")
CKPT = os.path.join(HERE, "ckpt", "r1")


def jobs():
    base = RunConfig()
    out = [("grid_i0_c0", "grid", base)]
    out += [(f"chaos_p{r}", "chaos", replace(base, perturb_seed=r, perturb_scale=1e-10))
            for r in range(1, 11)]
    out += [(f"noreset_i{i}_c0", "noreset", replace(base, init_seed=i, adam_reset_every=0))
            for i in range(10)]
    out += [(f"grid_i{i}_c{c}", "grid", replace(base, init_seed=i, ic_seed=c))
            for c in range(10) for i in range(10) if (i, c) != (0, 0)]
    return out


def run_one(name, arm, cfg):
    t0 = time.time()
    snap = lambda pE, pP, x0, it: M.light_metrics(pE, pP, x0, cfg, it)  # noqa: E731
    out = train_actor(cfg, snapshot_fn=snap)
    full = {str(it): M.full_metrics(pE, pP, out["x0"], cfg)
            for it, (pE, pP) in sorted(out["ckpts"].items())}
    rec = {
        "name": name, "arm": arm, "config": asdict(cfg),
        "digest": params_digest(out["pE"], out["pP"]),
        "saddle": M.saddle_reference(out["x0"], cfg),
        "x_E0": out["x0"].tolist(),
        "full": full, "snapshots": out["snapshots"],
        "seconds": time.time() - t0,
        "env": {"numpy": np.__version__, "python": platform.python_version(),
                "machine": platform.machine(), "system": platform.system()},
    }
    save_ckpts(os.path.join(CKPT, name + ".npz"), cfg, out)
    tmp = os.path.join(RES, name + ".json.tmp")
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, os.path.join(RES, name + ".json"))
    return name, rec["seconds"]


def main(n_workers):
    os.makedirs(RES, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    todo = [j for j in jobs() if not os.path.exists(os.path.join(RES, j[0] + ".json"))]
    print(f"[r1] {len(todo)} runs to do, {n_workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(run_one, *j) for j in todo]
        for k, f in enumerate(as_completed(futs), 1):
            name, sec = f.result()
            print(f"[r1] {k}/{len(todo)} {name} {sec:.0f}s (wall {time.time()-t0:.0f}s)", flush=True)
    print("[r1] DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
