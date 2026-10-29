import os
"""
Post-hoc diagnostics on EXISTING checkpoints for the revision of
'Learning Embeddings of Pursuit-Evasion Strategies on Compact Manifolds'.

Everything here is post-training analysis: no training loss is touched,
no V* or u* enters any optimisation.  V*/u* appear only as verification
yardsticks, exactly as in the paper's honesty contract.

Sections
  A. Harness validation: reproduce embedding_summary.json numbers.
  B. Normalisation sensitivity: covariance-PCA vs correlation-PCA (PDE
     paper standardises latents; the draft does not).
  C. Strategy-ensemble observables: PR of the control family Z=W^j H(t)
     and of the unit-heading family, vs latent PR.
  D. Winding-sector analysis: per-head winding numbers, per-sector
     control-family PR, within/across sector distances.
  E. Head-matrix diagnostics: singular values, sv participation ratio,
     distance from (reproduced) init per player.
  F. Integer-k Fourier power of H_c: locate the spectral break, band
     power asymmetry P vs E.
  G. Loss-term & gradient decomposition at the checkpoint: how much of
     each player's learning signal comes from the lambda_d shaping term.
  H. Actor-critic bridge: reproduce the 0.43 alignment (v4 critic),
     recompute with v5 critic; CCA between pursuer controls and critic
     gradient; per-head linear R^2 of V along trajectories on H(t).
"""
import sys, json, dataclasses
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, ROOT)

import pursuit_evasion_numpy as pen
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
                                   backward_dirs, rollout, backward_rollout,
                                   forward_loss, backward_loss, init_params,
                                   sample_evader_ICs)
import pursuit_evasion_v4 as v4

OUT = {}

def PR(ev):
    ev = np.maximum(ev, 0.0)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300))

def pcs_for(ev, thr):
    c = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(c, thr) + 1)

def cov_eigs(X):
    """X: (N, d) -> eigenvalues of centred covariance, descending."""
    Xc = X - X.mean(0, keepdims=True)
    C = Xc.T @ Xc / X.shape[0]
    ev = np.linalg.eigvalsh(C)[::-1]
    return ev, Xc

# ---------------------------------------------------------------- load actor
npz = np.load(ROOT + "v2_actor.npz")
pE = {"B": npz["BE"], "b_body": npz["bE"], "W": npz["WE"]}
pP = {"B": npz["BP"], "b_body": npz["bP"], "W": npz["WP"]}
x0 = npz["x_E0"]
n_h = pE["W"].shape[0]; n_b = pE["W"].shape[2]
OUT["checkpoint"] = {"iters": int(npz["iter0"]), "n_h": int(n_h),
                     "n_b": int(n_b), "x_E0_shape": list(x0.shape)}

T_DENSE = 1024
t_max, L, v_E, v_P, eps_c, n_freqs = 4.0, 2.0, 1.0, 1.2, 0.08, 12
td = np.linspace(0.0, t_max, T_DENSE)
phid = fourier_features(td, t_max, n_freqs)
_, cE = forward_dirs(pE, phid); HE = cE["H"]
_, cP = forward_dirs(pP, phid); HP = cP["H"]

# ---------------------------------------------------------------- A
evE, HcE = cov_eigs(HE); evP, HcP = cov_eigs(HP)
OUT["A_reproduce"] = {
    "PR_E": PR(evE), "PR_P": PR(evP),
    "pc95_E": pcs_for(evE, 0.95), "pc95_P": pcs_for(evP, 0.95),
    "pc99_E": pcs_for(evE, 0.99), "pc99_P": pcs_for(evP, 0.99),
}

# ---------------------------------------------------------------- B
def corr_eigs(X):
    Xc = X - X.mean(0, keepdims=True)
    s = Xc.std(0, keepdims=True); s = np.maximum(s, 1e-12)
    return np.linalg.eigvalsh((Xc / s).T @ (Xc / s) / X.shape[0])[::-1]
OUT["B_normalisation"] = {
    "PR_E_corr": PR(corr_eigs(HE)), "PR_P_corr": PR(corr_eigs(HP)),
    "pc95_E_corr": pcs_for(corr_eigs(HE), 0.95),
    "pc95_P_corr": pcs_for(corr_eigs(HP), 0.95),
}

# ---------------------------------------------------------------- C
def controls(p, H):
    return np.einsum("hab,tb->tha", p["W"], H)        # (T, n_h, 2)
zE = controls(pE, HE); zP = controls(pP, HP)
def fam_stats(z):
    X = z.reshape(z.shape[0], -1)                      # (T, 2 n_h)
    ev, _ = cov_eigs(X)
    return {"PR": PR(ev), "pc95": pcs_for(ev, 0.95), "pc99": pcs_for(ev, 0.99)}
def unit(z):
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    return z / np.maximum(n, 1e-12)
OUT["C_control_family"] = {
    "Z_E": fam_stats(zE), "Z_P": fam_stats(zP),
    "U_E": fam_stats(unit(zE)), "U_P": fam_stats(unit(zP)),
}

# ---------------------------------------------------------------- D
n_t = 64
tg = np.linspace(0.0, t_max, n_t + 1)
phig = fourier_features(tg, t_max, n_freqs)
uEg, _ = forward_dirs(pE, phig); uPg, _ = forward_dirs(pP, phig)
xE, xP = rollout(uEg, uPg, x0, L, v_E, v_P, t_max, n_t)
dt = t_max / n_t
dispE_net = dt * v_E * (0.5 * (uEg[:, :-1] + uEg[:, 1:])).sum(axis=1)  # (n_h,2)
dispP_net = dt * v_P * (0.5 * (uPg[:, :-1] + uPg[:, 1:])).sum(axis=1)
wE = dispE_net / L
w_rel = (dispE_net - dispP_net) / L
sec_int = np.round(wE).astype(int)                   # integer winding class
sec_sign = np.sign(np.round(wE * 2) / 2 + 0.0)       # sign class (coarse)
disp = (xE - xP + L / 2) % L - L / 2
dmin = np.linalg.norm(disp, axis=-1).min(axis=1)
captured = dmin <= eps_c
OUT["D_winding"] = {
    "winding_E_per_head": [list(map(float, w)) for w in wE],
    "winding_rel_per_head": [list(map(float, w)) for w in w_rel],
    "sector_int_labels": [f"({a},{b})" for a, b in sec_int],
    "captured_mask": [bool(c) for c in captured],
    "n_captured": int(captured.sum()),
}
# per-sector control-family PR (evader), sectors with >= 3 heads
labels = [tuple(s) for s in sec_int]
sector_prs = {}
for lab in sorted(set(labels)):
    idx = [j for j, l in enumerate(labels) if l == lab]
    if len(idx) < 2:
        sector_prs[str(lab)] = {"n_heads": len(idx), "PR": None}
        continue
    zsub = zE[:, idx, :]
    sector_prs[str(lab)] = {"n_heads": len(idx), **fam_stats(zsub)}
OUT["D_winding"]["per_sector_ZE"] = sector_prs
# within vs across sector mean L2 distance between unit heading curves
uEd = unit(zE)                                        # (T, n_h, 2) dense grid
flat = uEd.transpose(1, 0, 2).reshape(n_h, -1) / np.sqrt(T_DENSE)
D2 = ((flat[:, None, :] - flat[None, :, :]) ** 2).sum(-1)
same = np.array([[labels[i] == labels[j] for j in range(n_h)]
                 for i in range(n_h)])
iu = np.triu_indices(n_h, 1)
OUT["D_winding"]["mean_L2_within_sector"] = float(np.sqrt(D2[iu][same[iu]]).mean()) if same[iu].any() else None
OUT["D_winding"]["mean_L2_across_sector"] = float(np.sqrt(D2[iu][~same[iu]]).mean()) if (~same[iu]).any() else None

# ---------------------------------------------------------------- E
rng = np.random.default_rng(0)
D_in = 2 * n_freqs + 1
initE = init_params(n_h, n_b, D_in, rng)
initP = init_params(n_h, n_b, D_in, rng)
def svs(p):
    return np.linalg.svd(p["W"].reshape(2 * n_h, n_b), compute_uv=False)
sE, sP = svs(pE), svs(pP)
s0E, s0P = svs(initE), svs(initP)
def rel_change(p, p0, key):
    return float(np.linalg.norm(p[key] - p0[key]) / np.linalg.norm(p0[key]))
OUT["E_heads"] = {
    "sv_max_E": float(sE[0]), "sv_max_P": float(sP[0]),
    "sv_min_E": float(sE[-1]), "sv_min_P": float(sP[-1]),
    "svPR_E": PR(sE ** 2), "svPR_P": PR(sP ** 2),
    "sv_max_init_E": float(s0E[0]), "sv_max_init_P": float(s0P[0]),
    "relW_change_E": rel_change(pE, initE, "W"),
    "relW_change_P": rel_change(pP, initP, "W"),
    "relB_change_E": rel_change(pE, initE, "B"),
    "relB_change_P": rel_change(pP, initP, "B"),
    "head_norms_E": [float(np.linalg.norm(pE["W"][j])) for j in range(n_h)],
    "head_norms_P": [float(np.linalg.norm(pP["W"][j])) for j in range(n_h)],
}

# ---------------------------------------------------------------- F
def int_k_power(Hc):
    Ht = np.fft.rfft(Hc, axis=0)
    return (np.abs(Ht) ** 2).sum(axis=1) / Hc.shape[0] ** 2   # (K,)
pkE, pkP = int_k_power(HcE), int_k_power(HcP)
plateau_E = pkE[1:9].mean(); plateau_P = pkP[1:9].mean()
def break_idx(pk, plateau, frac=1e-2):
    for k in range(1, len(pk)):
        if pk[k] < frac * plateau and all(pk[k:k+5] < frac * plateau):
            return k
    return len(pk)
band = lambda pk, lo, hi: float(pk[lo:hi].sum())
OUT["F_fourier"] = {
    "break_idx_E_1pc": break_idx(pkE, plateau_E),
    "break_idx_P_1pc": break_idx(pkP, plateau_P),
    "band_1_10_E": band(pkE, 1, 11), "band_1_10_P": band(pkP, 1, 11),
    "band_11_50_E": band(pkE, 11, 51), "band_11_50_P": band(pkP, 11, 51),
    "ratio_P_over_E_band_11_50": band(pkP, 11, 51) / max(band(pkE, 11, 51), 1e-300),
    "ratio_P_over_E_band_1_10": band(pkP, 1, 11) / max(band(pkE, 1, 11), 1e-300),
    "input_K": n_freqs,
    "power_at_k": {str(k): [float(pkE[k]), float(pkP[k])] for k in
                   [1, 4, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128]},
}

# ---------------------------------------------------------------- G
lam_nc, lam_d = 2.0, 5.0
LE_full, LP_full, cacheL = forward_loss(xE, xP, L, eps_c, 5e-3, t_max, n_t,
                                        lam_nc, lam_d=lam_d)
ET = cacheL["ET"]; S_last = cacheL["S"][:, -1]; d_mean = cacheL["d"].mean(1)
OUT["G_loss_decomp"] = {
    "L_P": LP_full, "L_E": LE_full,
    "term_ET_mean": float(ET.mean()),
    "term_nc_mean": float(lam_nc * S_last.mean() * t_max),
    "term_shaping_mean": float(lam_d * d_mean.mean()),
    "no_capture_frac_soft": float(S_last.mean()),
}
def param_grads(is_pursuer, lam_d_use):
    _, _, cL = forward_loss(xE, xP, L, eps_c, 5e-3, t_max, n_t,
                            lam_nc, lam_d=lam_d_use)
    dxE, dxP = backward_loss(1.0, cL, n_h, lam_nc, t_max, is_pursuer)
    duE, duP = backward_rollout(dxE, dxP, L, v_E, v_P, t_max, n_t)
    if is_pursuer:
        _, cD = forward_dirs(pP, phig)
        return backward_dirs(pP, cD, duP)
    _, cD = forward_dirs(pE, phig)
    return backward_dirs(pE, cD, duE)
def gnorm(g):
    return float(np.sqrt(sum((v ** 2).sum() for v in g.values())))
for who, flag in (("P", True), ("E", False)):
    g_full = param_grads(flag, lam_d)
    g_nosh = param_grads(flag, 0.0)
    g_shap = {k: g_full[k] - g_nosh[k] for k in g_full}
    OUT["G_loss_decomp"][f"gradnorm_{who}_full"] = gnorm(g_full)
    OUT["G_loss_decomp"][f"gradnorm_{who}_ET_nc_only"] = gnorm(g_nosh)
    OUT["G_loss_decomp"][f"gradnorm_{who}_shaping_only"] = gnorm(g_shap)
    OUT["G_loss_decomp"][f"shaping_share_{who}"] = (
        gnorm(g_shap) / max(gnorm(g_full), 1e-300))

# ---------------------------------------------------------------- H
def load_critic(path):
    z = np.load(path)
    params = {k: z[k] for k in z.files if k not in ("iter0", "hist")}
    hid, D = params["W1"].shape
    cfg = v4.HonestVConfig()
    # infer n_freqs from encoded dim
    nf = None
    for cand in range(1, 16):
        if v4.encoded_dim(2, cand) == D:
            nf = cand; break
    cfg = dataclasses.replace(cfg, hidden=hid, n_freqs=nf)
    return params, cfg, int(z["iter0"]) if "iter0" in z.files else -1
try:
    pV4, cfg4, it4 = load_critic(ROOT + "v4_honestV.npz")
    pV5, cfg5, it5 = load_critic(ROOT + "v5_honestV.npz")
    OUT["H_bridge"] = {"v4_cfg": {"hidden": cfg4.hidden, "n_freqs": cfg4.n_freqs,
                                  "t_max": cfg4.t_max, "iters": it4},
                       "v5_cfg": {"hidden": cfg5.hidden, "n_freqs": cfg5.n_freqs,
                                  "t_max": cfg5.t_max, "iters": it5}}
    for tag, (pV, cfgV) in (("v4", (pV4, cfg4)), ("v5", (pV5, cfg5))):
        al = v4.alignment_actor_vs_honest_critic(
            pE, pP, x0, TrainConfig(n_h=n_h), pV, cfgV)
        OUT["H_bridge"][f"align_mean_{tag}"] = float(al.mean())
        # stack pursuer controls and critic grad dirs over (j,k)
        A, B = [], []
        Vtraj = np.zeros((n_h, n_t + 1))
        for j in range(n_h):
            for k in range(n_t + 1):
                delta = (xE[j, k] - xP[j, k] + L / 2) % L - L / 2
                g = v4.grad_V(pV, delta, cfgV.t_max, cfgV)
                B.append(g / max(np.linalg.norm(g), 1e-12))
                A.append(uPg[j, k])
                phiV = v4.encode_inputs(delta[None, :], np.array([cfgV.t_max]),
                                        cfgV.L, cfgV.n_freqs, cfgV.t_max)
                Vtraj[j, k] = v4.forward_VT(pV, phiV)[0][0]
        A = np.array(A); B = np.array(B)
        # CCA (2x2)
        Ac = A - A.mean(0); Bc = B - B.mean(0)
        Ca = Ac.T @ Ac / len(A); Cb = Bc.T @ Bc / len(B)
        Cab = Ac.T @ Bc / len(A)
        wa = np.linalg.inv(np.linalg.cholesky(Ca + 1e-9 * np.eye(2)))
        wb = np.linalg.inv(np.linalg.cholesky(Cb + 1e-9 * np.eye(2)))
        M = wa @ Cab @ wb.T
        cc = np.linalg.svd(M, compute_uv=False)
        OUT["H_bridge"][f"cca_{tag}"] = [float(c) for c in cc]
        # per-head linear R^2 of V(Delta_j(t), tau_max) on H_alpha(t) (65-pt grid)
        _, cg = forward_dirs(pE, phig); HEg = cg["H"]
        _, cg = forward_dirs(pP, phig); HPg = cg["H"]
        def r2_on(Hg):
            r2s = []
            X = np.hstack([np.ones((n_t + 1, 1)), Hg - Hg.mean(0)])
            for j in range(n_h):
                y = Vtraj[j]
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                r2s.append(1 - resid.var() / max(y.var(), 1e-300))
            return float(np.mean(r2s))
        OUT["H_bridge"][f"R2_V_on_HE_{tag}"] = r2_on(HEg)
        OUT["H_bridge"][f"R2_V_on_HP_{tag}"] = r2_on(HPg)
except Exception as e:
    OUT["H_bridge"] = {"error": repr(e)}

print(json.dumps(OUT, indent=1))
with open("./post_hoc_checks.json", "w") as f:
    json.dump(OUT, f, indent=1)
