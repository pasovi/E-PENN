import os
"""E1/E7 checkpoint analysis + referee-proofing nulls. Post-hoc only."""
import sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, ROOT)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
                                   backward_dirs, rollout, backward_rollout,
                                   forward_loss, backward_loss, init_params)

t_max, L, v_E, v_P, eps_c, nf, n_t, T = 4.0, 2.0, 1.0, 1.2, 0.08, 12, 64, 1024
td = np.linspace(0, t_max, T); phid = fourier_features(td, t_max, nf)
tg = np.linspace(0, t_max, n_t + 1); phig = fourier_features(tg, t_max, nf)
OUT = {}

def PR(ev):
    ev = np.maximum(ev, 0)
    return float(ev.sum()**2 / max((ev**2).sum(), 1e-300))
def cov_eigs(X):
    Xc = X - X.mean(0, keepdims=True)
    return np.linalg.eigvalsh(Xc.T @ Xc / X.shape[0])[::-1]
def pcs_for(ev, thr):
    return int(np.searchsorted(np.cumsum(ev)/ev.sum(), thr) + 1)
def unit(z):
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
def load(name):
    z = np.load(ROOT + name)
    return ({"B": z["BE"], "b_body": z["bE"], "W": z["WE"]},
            {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}, z["x_E0"])

def full_suite(pE, pP, x0):
    n_h = pE["W"].shape[0]
    _, cE = forward_dirs(pE, phid); HE = cE["H"]
    _, cP = forward_dirs(pP, phid); HP = cP["H"]
    zE = np.einsum("hab,tb->tha", pE["W"], HE)
    zP = np.einsum("hab,tb->tha", pP["W"], HP)
    evHE, evHP = cov_eigs(HE), cov_eigs(HP)
    evZE = cov_eigs(zE.reshape(T, -1)); evZP = cov_eigs(zP.reshape(T, -1))
    # winding sectors on training grid
    uEg, _ = forward_dirs(pE, phig); uPg, _ = forward_dirs(pP, phig)
    xE, xP = rollout(uEg, uPg, x0, L, v_E, v_P, t_max, n_t)
    dt = t_max / n_t
    dispE = dt * v_E * (0.5 * (uEg[:, :-1] + uEg[:, 1:])).sum(1)
    sec = [tuple(s) for s in np.round(dispE / L).astype(int)]
    d = np.linalg.norm((xE - xP + L/2) % L - L/2, axis=-1)
    # unit-heading band fractions
    def bandfrac(z):
        X = unit(z).reshape(T, -1); X = X - X.mean(0, keepdims=True)
        P = (np.abs(np.fft.rfft(X, axis=0))**2).sum(1)
        return float(P[11:51].sum() / P[1:].sum())
    # gauge-fixed head svPR
    def gf_svpr(p):
        Wg = np.stack([w/np.linalg.norm(w) for w in p["W"]]).reshape(-1, p["W"].shape[2])
        s2 = np.linalg.svd(Wg, compute_uv=False)**2
        return PR(s2)
    return {
        "PR_HE": PR(evHE), "PR_HP": PR(evHP),
        "pc95_HE": pcs_for(evHE, .95), "pc95_HP": pcs_for(evHP, .95),
        "PR_ZE": PR(evZE), "PR_ZP": PR(evZP),
        "pc95_ZE": pcs_for(evZE, .95), "pc95_ZP": pcs_for(evZP, .95),
        "n_sectors": len(set(sec)), "sectors": [str(s) for s in sec],
        "n_captured": int((d.min(1) <= eps_c).sum()), "n_h": n_h,
        "midband_frac_uE": bandfrac(zE), "midband_frac_uP": bandfrac(zP),
        "gf_svPR_E": gf_svpr(pE), "gf_svPR_P": gf_svpr(pP),
    }

for tag, f in (("baseline_v2", "v2_actor.npz"), ("E1", "e1_actor.npz"),
               ("E7_4k", "e7_4k.npz"), ("E7_14k", "e7_actor.npz")):
    pE, pP, x0 = load(f)
    OUT[tag] = full_suite(pE, pP, x0)

# ---- E7 drift 4k -> 14k and evader gradient with lam_d = 0 ----------------
pE4, pP4, x04 = load("e7_4k.npz")
pE14, pP14, _ = load("e7_actor.npz")
OUT["E7_drift"] = {
    f"rel_{k}_{who}": float(np.linalg.norm(p14[k]-p4[k])/np.linalg.norm(p4[k]))
    for who, p4, p14 in (("E", pE4, pE14), ("P", pP4, pP14))
    for k in ("B", "W")}
def gnorms(pE, pP, x0, lam_d, who):
    uE, cE = forward_dirs(pE, phig); uP, cP = forward_dirs(pP, phig)
    xE, xP = rollout(uE, uP, x0, L, v_E, v_P, t_max, n_t)
    _, _, cL = forward_loss(xE, xP, L, eps_c, 5e-3, t_max, n_t, 2.0, lam_d=lam_d)
    dxE, dxP = backward_loss(1.0, cL, pE["W"].shape[0], 2.0, t_max,
                             is_pursuer=(who == "P"))
    duE, duP = backward_rollout(dxE, dxP, L, v_E, v_P, t_max, n_t)
    g = backward_dirs(pP if who == "P" else pE, cP if who == "P" else cE,
                      duP if who == "P" else duE)
    return float(np.sqrt(sum((v**2).sum() for v in g.values())))
OUT["E7_gradnorms_at_14k"] = {
    "E_lam0": gnorms(pE14, pP14, x04, 0.0, "E"),
    "E_lam5": gnorms(pE14, pP14, x04, 5.0, "E"),
    "P_lam0": gnorms(pE14, pP14, x04, 0.0, "P"),
    "P_lam5": gnorms(pE14, pP14, x04, 5.0, "P")}
OUT["baseline_gradnorm_E_full"] = 0.0833  # from post_hoc_checks.json (G)

# ---- nulls and Omega(k) on the baseline ------------------------------------
import pursuit_evasion_v4 as v4
import dataclasses
pE, pP, x0 = load("v2_actor.npz")
n_h = 16
z5 = np.load(ROOT + "v5_honestV.npz")
pV = {k: z5[k] for k in ("B1", "b1", "B2", "b2", "w", "c")}
hid, D = pV["B1"].shape
nfv = next(c for c in range(1, 20) if v4.encoded_dim(2, c) == D)
cfgV = dataclasses.replace(v4.HonestVConfig(), hidden=hid, n_freqs=nfv)
# dense rollout (257 pts) for regression dof
n_t2 = 256
tg2 = np.linspace(0, t_max, n_t2 + 1); phig2 = fourier_features(tg2, t_max, nf)
uE2, _ = forward_dirs(pE, phig2); uP2, _ = forward_dirs(pP, phig2)
xE2, xP2 = rollout(uE2, uP2, x0, L, v_E, v_P, t_max, n_t2)
_, cg2 = forward_dirs(pE, phig2); HE2 = cg2["H"]
Vt = np.zeros((n_h, n_t2 + 1))
for j in range(n_h):
    dlt = (xE2[j] - xP2[j] + L/2) % L - L/2
    phiV = v4.encode_inputs(dlt, np.full(n_t2 + 1, cfgV.t_max),
                            cfgV.L, cfgV.n_freqs, cfgV.t_max)
    Vt[j] = v4.forward_VT(pV, phiV)[0]
def cv_r2(X, y, k=5):
    n = len(y); idx = np.arange(n); folds = np.array_split(idx, k)
    press, sst = 0.0, ((y - y.mean())**2).sum()
    for f in folds:
        tr = np.setdiff1d(idx, f)
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        press += ((y[f] - X[f] @ beta)**2).sum()
    return 1 - press / sst
# Omega(k): CV R^2 of V on top-k latent PCs (per head, averaged)
mu = HE2.mean(0, keepdims=True); Hc = HE2 - mu
ev, V = np.linalg.eigh(Hc.T @ Hc / len(Hc)); V = V[:, ::-1]
Omega = []
for k in (1, 2, 3, 4, 6, 8, 12, 16):
    Xk = np.hstack([np.ones((n_t2 + 1, 1)), Hc @ V[:, :k]])
    Omega.append((k, float(np.mean([cv_r2(Xk, Vt[j]) for j in range(n_h)]))))
OUT["Omega_k_cvR2_HE_v5"] = Omega
# NULL 1: raw input features phi(t) (25-dim)
Xphi = np.hstack([np.ones((n_t2 + 1, 1)), phig2 - phig2.mean(0)])
OUT["null_cvR2_on_phi25"] = float(np.mean([cv_r2(Xphi, Vt[j]) for j in range(n_h)]))
# NULL 2: random tanh body (untrained, same arch), 16-dim
rng = np.random.default_rng(123)
p0 = init_params(16, 16, 2 * nf + 1, rng)
_, c0 = forward_dirs(p0, phig2); H0 = c0["H"]
X0 = np.hstack([np.ones((n_t2 + 1, 1)), H0 - H0.mean(0)])
OUT["null_cvR2_on_random_body16"] = float(np.mean([cv_r2(X0, Vt[j]) for j in range(n_h)]))
# trained latent, all 16 dims, CV (upgrade of the in-sample 0.90)
X16 = np.hstack([np.ones((n_t2 + 1, 1)), Hc])
OUT["cvR2_on_trained_HE16"] = float(np.mean([cv_r2(X16, Vt[j]) for j in range(n_h)]))

# ---- init ranks: is the head/control compression learned? ------------------
rng = np.random.default_rng(0)
iE = init_params(16, 16, 2 * nf + 1, rng); iP = init_params(16, 16, 2 * nf + 1, rng)
s2 = np.linalg.svd(iE["W"].reshape(32, 16), compute_uv=False)**2
OUT["init_svPR_raw_E"] = PR(s2)
Wg = np.stack([w/np.linalg.norm(w) for w in iE["W"]]).reshape(32, 16)
OUT["init_svPR_gauge_E"] = PR(np.linalg.svd(Wg, compute_uv=False)**2)
_, c0 = forward_dirs(iE, phid); H0 = c0["H"]
OUT["init_PR_H"] = PR(cov_eigs(H0))
z0 = np.einsum("hab,tb->tha", iE["W"], H0)
OUT["init_PR_Z"] = PR(cov_eigs(z0.reshape(T, -1)))

# ---- bootstrap CI for PR(Z_E) over heads (baseline) ------------------------
_, cE = forward_dirs(pE, phid); HE = cE["H"]
zE = np.einsum("hab,tb->tha", pE["W"], HE)
rng = np.random.default_rng(7)
prs = []
for _ in range(400):
    idx = rng.integers(0, 16, 16)
    prs.append(PR(cov_eigs(zE[:, idx, :].reshape(T, -1))))
OUT["bootstrap_PR_ZE"] = {"mean": float(np.mean(prs)),
                          "p5": float(np.percentile(prs, 5)),
                          "p95": float(np.percentile(prs, 95))}
# winding-class robustness: fraction of heads whose class flips if rounding
# threshold moves by +-0.1
uEg, _ = forward_dirs(pE, phig); uPg, _ = forward_dirs(pP, phig)
dt = t_max / n_t
dispE = dt * v_E * (0.5 * (uEg[:, :-1] + uEg[:, 1:])).sum(1) / L
flips = np.mean([not (np.all(np.round(w) == np.round(w + 0.1)) and
                      np.all(np.round(w) == np.round(w - 0.1))) for w in dispE])
OUT["winding_class_flip_frac_pm0.1"] = float(flips)

print(json.dumps(OUT, indent=1))
json.dump(OUT, open("./e1e7_analysis.json", "w"), indent=1)
