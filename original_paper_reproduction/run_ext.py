"""
Long-run extensions of the honest critics (referee critique: 'run longer,
see if the residual floors hold').

Torus  : v5 checkpoint (50k iters) -> extend with a cosine warm restart
         (lr 1e-3 -> 5e-5 over the extension window) to +20k.
Mobius : 15k -> 45k (x3), warm restart via schedule_total = 45000.
Disk   : 12.5k -> 25k;  Ball: 10k -> 20k (controls).

All extensions write *_ext.npz checkpoints; the original checkpoints
analysed in the paper are never overwritten.  No V*/u* anywhere.

Usage: python3 run_ext.py {torus|disk|ball|mobius} [target_total_iters]
"""
import os, sys, json, math, time, shutil, dataclasses
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pursuit_evasion_v4 as v4
import pursuit_evasion_numpy as pen
import honest_critic_geom as hc
import run_geom

TRAIN_SECONDS = 33.0
TORUS_SRC = os.path.join(HERE, "v5_honestV.npz")
TORUS_EXT = os.path.join(HERE, "v5_ext.npz")


def torus(target_total=70000):
    if not os.path.exists(TORUS_EXT):
        shutil.copy(TORUS_SRC, TORUS_EXT)
    z = np.load(TORUS_EXT)
    params = {k: z[k] for k in ("B1", "b1", "B2", "b2", "w", "c")}
    it0 = int(z["iter0"])
    hist = json.loads(str(z["hist"])) if "hist" in z.files else {}
    ext_hist = json.loads(str(z["ext_hist"])) if "ext_hist" in z.files else \
        {"iter": [], "loss": [], "loss_hji": [], "min_loss": 1e9}
    if it0 >= target_total:
        print(f"[torus ext] DONE at {it0}"); return
    START, TOTAL = 50000, target_total - 50000       # warm-restart window
    lr_max, lr_min = 1e-3, 5e-5
    opt = pen.Adam(params, lr=lr_max)
    rng = np.random.default_rng(1000 + it0)
    t0 = time.time(); it = it0
    while it < target_total and time.time() - t0 < TRAIN_SECONDS:
        p = min((it - START) / TOTAL, 1.0)
        opt.lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))
        b = v4.sample_honest_batch(rng, 2.0, 0.08, 8.0, 2, 384, 96, 96)
        loss, grads, diag = v4.honest_loss(params, *b, L=2.0, v_E=1.0,
                                           v_P=1.2, eps=0.08, n_freqs=8,
                                           t_max=8.0)
        opt.step(params, grads, sign=-1.0)
        it += 1
        ext_hist["min_loss"] = min(ext_hist["min_loss"], float(loss))
        if it % 500 == 0:
            ext_hist["iter"].append(it)
            ext_hist["loss"].append(float(loss))
            ext_hist["loss_hji"].append(float(diag["loss_hji"]))
            print(f"[torus ext] it {it} lr={opt.lr:.2e} loss={loss:.3e} "
                  f"hji={diag['loss_hji']:.3e} min={ext_hist['min_loss']:.3e}")
    out = dict(params); out["iter0"] = it
    out["hist"] = json.dumps(hist); out["ext_hist"] = json.dumps(ext_hist)
    np.savez(TORUS_EXT, **out)
    print(f"[torus ext] saved at {it} ({time.time()-t0:.0f}s)")


def torus_validate():
    """Fixed 6k-point validation batch, mirroring the paper's protocol."""
    for tag, f in (("v5@50k", TORUS_SRC), ("ext", TORUS_EXT)):
        z = np.load(f)
        params = {k: z[k] for k in ("B1", "b1", "B2", "b2", "w", "c")}
        rng = np.random.default_rng(999)
        b = v4.sample_honest_batch(rng, 2.0, 0.08, 8.0, 2, 4800, 600, 600)
        loss, _, diag = v4.honest_loss(params, *b, L=2.0, v_E=1.0, v_P=1.2,
                                       eps=0.08, n_freqs=8, t_max=8.0)
        print("%s (iter %d): val loss %.3e  hji %.3e  term %.1e  cap %.1e  "
              "<|gradV|> %.3f" % (tag, int(z["iter0"]), loss,
              diag["loss_hji"], diag["loss_term"], diag["loss_cap"],
              diag["mean_grad_norm"]))


def geom(name, target_total):
    src = run_geom.CKPTS[name]
    ext = src.replace(".npz", "_ext.npz")
    if not os.path.exists(ext):
        shutil.copy(src, ext)
    z = np.load(ext); it0 = int(z["iter0"])
    if it0 >= target_total:
        print(f"[{name} ext] DONE at {it0}"); return
    G = run_geom.make_geom(name)
    cfg = run_geom.cfg_for(name, n_iters=min(2500, target_total - it0))
    cfg = dataclasses.replace(cfg, schedule_total=target_total)
    hc.train_geom(G, cfg, ext, resume=True)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "torus":
        torus(int(sys.argv[2]) if len(sys.argv) > 2 else 70000)
    elif which == "val":
        torus_validate()
    else:
        tgt = {"disk": 25000, "ball": 20000, "mobius": 45000}[which]
        geom(which, int(sys.argv[2]) if len(sys.argv) > 2 else tgt)
