import os
"""Follow-up: fixed critic loader (H), normalization-invariant asymmetry
observables, sign-sector robustness check.  Post-hoc only."""
import sys, json, dataclasses
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, ROOT)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
                                   rollout)
import pursuit_evasion_v4 as v4

OUT = {}
npz = np.load(ROOT + "v2_actor.npz")
pE = {"B": npz["BE"], "b_body": npz["bE"], "W": npz["WE"]}
pP = {"B": npz["BP"], "b_body": npz["bP"], "W": npz["WP"]}
x0 = npz["x_E0"]; n_h = pE["W"].shape[0]
t_max, L, v_E, v_P, n_freqs, n_t = 4.0, 2.0, 1.0, 1.2, 12, 64

T = 1024
td = np.linspace(0, t_max, T)
phid = fourier_features(td, t_max, n_freqs)
_, cE = forward_dirs(pE, phid); HE = cE["H"]
_, cP = forward_dirs(pP, phid); HP = cP["H"]

def PR(ev):
    ev = np.maximum(ev, 0)
    return float(ev.sum()**2 / max((ev**2).sum(), 1e-300))
def cov_eigs(X):
    Xc = X - X.mean(0, keepdims=True)
    return np.linalg.eigvalsh(Xc.T @ Xc / X.shape[0])[::-1]
def unit(z):
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    return z / np.maximum(n, 1e-12)

zE = np.einsum("hab,tb->tha", pE["W"], HE)
zP = np.einsum("hab,tb->tha", pP["W"], HP)
uE_d, uP_d = unit(zE), unit(zP)

# --- normalization-invariant spectral asymmetry: unit headings -----------
def band_power(X):        # X: (T, n_h, 2) -> integer-k total power
    Xc = X.reshape(T, -1)
    Xc = Xc - Xc.mean(0, keepdims=True)
    P = (np.abs(np.fft.rfft(Xc, axis=0))**2).sum(1) / T**2
    return P
PuE, PuP = band_power(uE_d), band_power(uP_d)
b = lambda P, lo, hi: float(P[lo:hi].sum())
OUT["unit_heading_spectra"] = {
    "frac_band_11_50_E": b(PuE, 11, 51) / b(PuE, 1, len(PuE)),
    "frac_band_11_50_P": b(PuP, 11, 51) / b(PuP, 1, len(PuP)),
    "ratio_P_over_E_band_11_50_unit": b(PuP, 11, 51) / max(b(PuE, 11, 51), 1e-300),
    "ratio_P_over_E_band_1_10_unit": b(PuP, 1, 11) / max(b(PuE, 1, 11), 1e-300),
    "total_E": b(PuE, 1, len(PuE)), "total_P": b(PuP, 1, len(PuP)),
}

# --- sign-quadrant sectors (coarser, bigger groups) -----------------------
tg = np.linspace(0, t_max, n_t + 1)
phig = fourier_features(tg, t_max, n_freqs)
uEg, _ = forward_dirs(pE, phig); uPg, _ = forward_dirs(pP, phig)
dt = t_max / n_t
dispE = dt * v_E * (0.5 * (uEg[:, :-1] + uEg[:, 1:])).sum(1)
quad = [tuple(np.sign(w).astype(int)) for w in dispE]
res = {}
for q in sorted(set(quad)):
    idx = [j for j, l in enumerate(quad) if l == q]
    if len(idx) >= 2:
        ev = cov_eigs(zE[:, idx, :].reshape(T, -1))
        res[str(q)] = {"n_heads": len(idx), "PR": PR(ev)}
    else:
        res[str(q)] = {"n_heads": len(idx)}
OUT["sign_quadrant_sectors_ZE"] = res
OUT["pooled_ZE_PR"] = PR(cov_eigs(zE.reshape(T, -1)))

# --- H: critic bridge ------------------------------------------------------
def load_critic(path):
    z = np.load(path)
    params = {k: z[k] for k in ("B1", "b1", "B2", "b2", "w", "c")}
    hid, D = params["B1"].shape
    nf = next(c for c in range(1, 20) if v4.encoded_dim(2, c) == D)
    cfg = dataclasses.replace(v4.HonestVConfig(), hidden=hid, n_freqs=nf)
    return params, cfg, int(z["iter0"])

xE, xP = rollout(uEg, uPg, x0, L, v_E, v_P, t_max, n_t)
for tag, path in (("v4", "v4_honestV.npz"), ("v5", "v5_honestV.npz")):
    pV, cfgV, it = load_critic(ROOT + path)
    al = v4.alignment_actor_vs_honest_critic(pE, pP, x0,
            TrainConfig(n_h=n_h), pV, cfgV)
    A, B, Vt = [], [], np.zeros((n_h, n_t + 1))
    for j in range(n_h):
        for k in range(n_t + 1):
            delta = (xE[j, k] - xP[j, k] + L / 2) % L - L / 2
            g = v4.grad_V(pV, delta, cfgV.t_max, cfgV)
            B.append(g / max(np.linalg.norm(g), 1e-12))
            A.append(uPg[j, k])
            phiV = v4.encode_inputs(delta[None, :], np.array([cfgV.t_max]),
                                    cfgV.L, cfgV.n_freqs, cfgV.t_max)
            Vt[j, k] = v4.forward_VT(pV, phiV)[0][0]
    A = np.array(A); Bm = np.array(B)
    Ac = A - A.mean(0); Bc = Bm - Bm.mean(0)
    Ca = Ac.T@Ac/len(A); Cb = Bc.T@Bc/len(A); Cab = Ac.T@Bc/len(A)
    wa = np.linalg.inv(np.linalg.cholesky(Ca + 1e-9*np.eye(2)))
    wb = np.linalg.inv(np.linalg.cholesky(Cb + 1e-9*np.eye(2)))
    cc = np.linalg.svd(wa @ Cab @ wb.T, compute_uv=False)
    _, cg = forward_dirs(pE, phig); HEg = cg["H"]
    _, cg = forward_dirs(pP, phig); HPg = cg["H"]
    def r2_on(Hg):
        X = np.hstack([np.ones((n_t + 1, 1)), Hg - Hg.mean(0)])
        out = []
        for j in range(n_h):
            beta, *_ = np.linalg.lstsq(X, Vt[j], rcond=None)
            out.append(1 - (Vt[j] - X@beta).var() / max(Vt[j].var(), 1e-300))
        return float(np.mean(out))
    OUT[f"H_{tag}"] = {
        "iters": it, "hidden": cfgV.hidden, "n_freqs": cfgV.n_freqs,
        "tau_max": cfgV.t_max,
        "align_mean": float(al.mean()),
        "cca": [float(c) for c in cc],
        "R2_V_on_HE": r2_on(HEg), "R2_V_on_HP": r2_on(HPg),
        "V_range_on_traj": [float(Vt.min()), float(Vt.max())],
    }

print(json.dumps(OUT, indent=1))
json.dump(OUT, open("./post_hoc_checks2.json", "w"), indent=1)
