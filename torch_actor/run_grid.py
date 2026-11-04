"""Train a grid of baseline runs (init seeds x IC seeds) in one batched torch program.

Writes, under --out:
  <name>.npz        per run, parameters at cfg.ckpt_at in the rev2/core.save_ckpts
                    key layout (readable by rev2/core.load_ckpt and rev2/metrics.py)
  summary.json      config, device/dtype, wall time, and the light metrics of every
                    run after warm-up and every 1,000 iterations
Examples:
  python3 torch_actor/run_grid.py --init-seeds 0 --ic-seeds 0 --device cpu --dtype float64 --out torch_actor/results/baseline
  python3 torch_actor/run_grid.py --init-seeds 0-9 --ic-seeds 0-9 --device mps --dtype float32 --out torch_actor/results/grid100
"""
import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe_torch as pt                              # noqa: E402


def parse_range(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-seeds", default="0")
    ap.add_argument("--ic-seeds", default="0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "run"))
    ap.add_argument("--n-iters", type=int, default=None)
    ap.add_argument("--adam-reset-every", type=int, default=None)
    ap.add_argument("--no-ckpt", action="store_true", help="do not write the .npz files")
    ap.add_argument("--log-every", type=int, default=1000)
    a = ap.parse_args()

    cfg = pt.Config()
    if a.n_iters is not None:
        cfg = replace(cfg, n_iters=a.n_iters,
                      ckpt_at=tuple(c for c in cfg.ckpt_at if c <= a.n_iters) or (a.n_iters,))
    if a.adam_reset_every is not None:
        cfg = replace(cfg, adam_reset_every=a.adam_reset_every)
    dtype = {"float32": torch.float32, "float64": torch.float64}[a.dtype]
    if a.device == "mps" and dtype is torch.float64:
        sys.exit("MPS has no float64; use --dtype float32 or --device cpu")

    runs = [(i, c) for c in parse_range(a.ic_seeds) for i in parse_range(a.init_seeds)]
    names = [f"grid_i{i}_c{c}" for i, c in runs]
    print(f"[grid] {len(runs)} runs on {a.device} {a.dtype}, budget "
          f"{cfg.warmup + cfg.n_iters * (cfg.n_P_steps + cfg.n_E_steps)} steps", flush=True)
    t0 = time.time()
    out = pt.train(cfg, runs, device=a.device, dtype=dtype,
                   snapshot_fn=pt.light_metrics, log_every=a.log_every)
    wall = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    if not a.no_ckpt:
        for r, name in enumerate(names):
            pt.save_run_ckpt(os.path.join(a.out, name + ".npz"), cfg, out, r)
    per_run = [{"name": names[r], "init_seed": runs[r][0], "ic_seed": runs[r][1],
                "snapshots": [snap[r] for snap in out["snapshots"]]}
               for r in range(len(runs))]
    summary = {"config": asdict(cfg), "device": a.device, "dtype": a.dtype,
               "n_runs": len(runs), "seconds_total": wall, "seconds_per_run": wall / len(runs),
               "env": {"torch": torch.__version__, "python": platform.python_version(),
                       "machine": platform.machine(), "system": platform.system()},
               "runs": per_run}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[grid] done: {len(runs)} runs in {wall:.1f} s ({wall/len(runs):.2f} s per run) -> {a.out}")


if __name__ == "__main__":
    main()
