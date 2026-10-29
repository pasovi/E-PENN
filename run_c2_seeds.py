"""C2 replicate seeds (referee condition A1 / committed 'seed replicates').

Identical symmetrised protocol to run_c2.py (whose local loss functions are
imported and whose FD check is re-run per seed), but with the parameter-init
seed as an argument and snapshots saved at 20k/22k/24k/26k/28k for the
six-snapshot gap statistics.  Training is chunked with a fresh Adam every
2,500 iterations to reproduce the checkpoint-chaining protocol under which
the original c2 run and all actor runs were produced.

Usage: python3 run_c2_seeds.py <init_seed>
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_c2 as c2
from pursuit_evasion_numpy import init_params, Adam

SNAPSHOTS = (20000, 22000, 24000, 26000, 28000)
CHUNK = 2500


def main(seed):
    rng = np.random.default_rng(seed)
    D = 2 * c2.cfg.n_freqs + 1
    pE = init_params(c2.cfg.n_h, c2.cfg.n_b, D, rng)
    pP = init_params(c2.cfg.n_h, c2.cfg.n_b, D, rng)
    c2.fd_check(pE, pP)
    it = 0
    while it < max(SNAPSHOTS):
        optE, optP = Adam(pE, lr=c2.LR), Adam(pP, lr=c2.LR)  # chaining protocol
        for _ in range(CHUNK):
            _, gP, _ = c2.player_loss_and_grads(pE, pP, "P")
            optP.step(pP, gP, sign=-1.0)
            _, gE, diag = c2.player_loss_and_grads(pE, pP, "E")
            optE.step(pE, gE, sign=-1.0)
            it += 1
            if it in SNAPSHOTS:
                np.savez(os.path.join(HERE, f"c2_s{seed}_{it//1000}k.npz"),
                         BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
                         BP=pP["B"], bP=pP["b_body"], WP=pP["W"],
                         x_E0=c2.x_E0, iter0=it)
                L_E, L_P, cL = diag
                print(f"[c2 s{seed}] snapshot {it}: L_P={L_P:.3f} "
                      f"zero-sum check {abs(L_P + L_E):.2e} "
                      f"no_cap={float((cL['S'][:, -1] > 0.5).mean()):.2f}",
                      flush=True)
    print(f"[c2 s{seed}] DONE at {it}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
