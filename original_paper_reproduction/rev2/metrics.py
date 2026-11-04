"""Diagnostics for the revision, each defined by an explicit formula.

Notation.  For player a in {E, P} and head j = 1..n_h:
    H_a(t) in R^{n_b}                     latent curve (shared body output)
    z_a^j(t) = W_a^j H_a(t) in R^2        pre-normalisation head output
    u_a^j(t) = z_a^j / |z_a^j| in S^1     unit heading (the policy)
Curves are sampled on a dense grid t_1..t_T of [0, t_max] (T = 1024) and
time averages are written <.>_t.

Strategy-ensemble statistics (all gauge-invariant unless marked LEGACY)
----------------------------------------------------------------------
PR(C) = (tr C)^2 / tr(C^2) for a PSD matrix C (participation ratio).

gram_pr       PR of the uncentred Gram matrix G_jk = <u^j . u^k>_t of the
              n_h heading curves as elements of L^2([0,t_max]; R^2).
              Proposition 1 bounds THIS object: if every u^j is constant,
              rank G <= 2 and gram_pr <= 2, with equality iff the constant
              headings have an isotropic second-moment matrix.
gram_pr_hc    Same after subtracting the across-head mean curve
              (LEGACY: oracle_pr.head_PR, used for "rank two under stronger play").
constancy     eta = (1/n_h) sum_j |<u^j>_t|^2 in [0, 1]; eta = 1 iff every
              heading is constant in time.  1 - eta is the L^2 energy of the
              time-varying part of the ensemble.
tpr_U         PR of the time-centred covariance Cov_t[X(t)], X(t) = (u^j(t))_j
              in R^{2 n_h}.  Counts temporal modes of the *time-varying* part
              only (constant curves contribute nothing).
tpr_Z         Same with z^j instead of u^j.  LEGACY headline statistic
              ("PR(Z)").  NOT gauge-invariant: u^j is unchanged under
              z^j(t) -> c_j(t) z^j(t), c_j > 0, while tpr_Z changes.

Latent statistics
-----------------
pr_H          PR of Cov_t[H(t)] (time-centred; LEGACY "PR(H)").
pr_H_unc      PR of <H H^T>_t (uncentred; consistent with gram_pr, since the
              time-mean of H is what produces the constant part of u).
kpow_H(k)     integer-k power of the centred latent, sum_b |FFT_t[H_b - <H_b>](k)|^2 / T^2.
frac_le_K_H   sum_{1<=k<=K} kpow_H / sum_{k>=1} kpow_H  (fraction of latent
              temporal power inside the input Fourier passband).

Head-matrix statistics (Lemma 2)
--------------------------------
sv_raw        singular values of the stacked heads W_stack in R^{2 n_h x n_b}.
erank_gf      PR(s^2) of the gauge-fixed stack (each W^j scaled to unit
              Frobenius norm): the paper's "effective rank".

Game statistics (training ICs; pursuer at the origin)
-----------------------------------------------------
Delta0_j      min-image displacement of x_E0^j from the pursuer.
capturable_j  |Delta0_j| <= R = (v_P - v_E) t_max + eps.
Vstar_j       (|Delta0_j| - eps)/(v_P - v_E): feedback value (finite iff <= t_max).
Tc_j          first grid time with d_T(x_E, x_P) <= eps under the trained pair.
T_BR_j        best-response capture time of a pursuer that knows the trained
              evader's open-loop path: first t with d_T(x_E(t), 0) <= v_P t + eps
              (exact for straight-line interception, evaluated on a 16x refined grid).
              T_BR_j <= Tc_j always.
chart_opt_j   m* in Z^2 with x_E0 + m* L = Delta0 (nearest-image chart).
chart_cap_j   image m the trained pursuer actually captures: round(-D(T_c)/L),
              D = lifted (x_E - x_P).  chart agreement = (chart_cap == chart_opt).
chart_br_j    image intercepted by the best-response pursuer.
wind_E_j      LEGACY winding: lifted evader displacement over [0, t_max] / L.
"""
from __future__ import annotations

import numpy as np

from core import RunConfig, fourier_features, forward_dirs, rollout

T_DENSE = 1024
K_STORE = 129          # integer-k bins stored per spectrum


# ----------------------------------------------------------------- helpers
def PR(ev) -> float:
    ev = np.clip(np.asarray(ev, dtype=float), 0.0, None)
    s2 = (ev ** 2).sum()
    return float(ev.sum() ** 2 / s2) if s2 > 0 else float("nan")


def _eig_desc(C):
    return np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)


def dense_phi(cfg: RunConfig, n: int = T_DENSE):
    t = np.linspace(0.0, cfg.t_max, n)
    return t, fourier_features(t, cfg.t_max, cfg.K)


def curves(params: dict, phi: np.ndarray):
    """u (n_h,T,2), z (n_h,T,2), H (T,n_b)."""
    u, c = forward_dirs(params, phi)
    return u, c["z_head"], c["H"]


def kpow(X: np.ndarray) -> np.ndarray:
    """Integer-k power of the time-centred multichannel signal X (T, ...)."""
    Xc = X.reshape(X.shape[0], -1)
    Xc = Xc - Xc.mean(0, keepdims=True)
    return (np.abs(np.fft.rfft(Xc, axis=0)) ** 2).sum(1) / X.shape[0] ** 2


# ------------------------------------------------------ ensemble / latent
def ensemble_stats(u: np.ndarray, z: np.ndarray, idx=None) -> dict:
    if idx is not None:
        u, z = u[idx], z[idx]
    n_h, T, _ = u.shape
    out = {"n": int(n_h)}
    if n_h < 2:
        return out
    U = u.reshape(n_h, T * 2) / np.sqrt(T)                 # rows: unit L2 norm
    ev_g = _eig_desc(U @ U.T)
    Uc = U - U.mean(0, keepdims=True)
    ev_hc = _eig_desc(Uc @ Uc.T)
    ubar = u.mean(1)                                       # (n_h, 2)
    Xu = np.transpose(u, (1, 0, 2)).reshape(T, -1)
    Xz = np.transpose(z, (1, 0, 2)).reshape(T, -1)
    Xu_c = Xu - Xu.mean(0); Xz_c = Xz - Xz.mean(0)
    ev_tu = _eig_desc(Xu_c.T @ Xu_c / T)
    ev_tz = _eig_desc(Xz_c.T @ Xz_c / T)
    out.update({
        "gram_pr": PR(ev_g), "gram_pr_hc": PR(ev_hc),
        "constancy": float((ubar ** 2).sum(1).mean()),
        "tpr_U": PR(ev_tu), "tpr_Z": PR(ev_tz),
        "eig_gram": ev_g.tolist(),
    })
    return out


def latent_stats(H: np.ndarray, K: int) -> dict:
    T = H.shape[0]
    Hc = H - H.mean(0, keepdims=True)
    ev_c = _eig_desc(Hc.T @ Hc / T)
    ev_u = _eig_desc(H.T @ H / T)
    P = kpow(H)
    return {
        "pr_H": PR(ev_c), "pr_H_unc": PR(ev_u),
        "eig_H": ev_c.tolist(),
        "frac_le_K_H": float(P[1:K + 1].sum() / P[1:].sum()),
        "kpow_H": P[:K_STORE].tolist(),
    }


def unit_band_stats(u: np.ndarray, K: int) -> dict:
    X = np.transpose(u, (1, 0, 2))                         # (T, n_h, 2)
    P = kpow(X)
    tot = P[1:].sum()
    return {
        "band_11_50_U": float(P[11:51].sum() / tot),
        "band_1_10_U": float(P[1:11].sum() / tot),
        "pow_11_50_U": float(P[11:51].sum()),
        "kpow_U": P[:K_STORE].tolist(),
    }


def head_stats(W: np.ndarray) -> dict:
    n_h, _, n_b = W.shape
    s = np.linalg.svd(W.reshape(2 * n_h, n_b), compute_uv=False)
    Wg = W / np.linalg.norm(W.reshape(n_h, -1), axis=1)[:, None, None]
    sg = np.linalg.svd(Wg.reshape(2 * n_h, n_b), compute_uv=False)
    return {"sv_raw_max": float(s[0]), "erank_raw": PR(s ** 2),
            "erank_gf": PR(sg ** 2), "sv_gf": sg.tolist()}


# ------------------------------------------------------------------- game
def min_image(x, L):
    return (x + L / 2.0) % L - L / 2.0


def game_stats(pE: dict, pP: dict, x0: np.ndarray, cfg: RunConfig,
               refine: int = 16) -> dict:
    L, eps, dv = cfg.L, cfg.eps, cfg.dv
    tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phig = fourier_features(tg, cfg.t_max, cfg.K)
    uE, _ = forward_dirs(pE, phig)
    uP, _ = forward_dirs(pP, phig)
    xE, xP = rollout(uE, uP, x0, L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    dt = cfg.t_max / cfg.n_t
    # lifted (universal-cover) paths, identical increments to `rollout`
    incE = dt * cfg.v_E * 0.5 * (uE[:, :-1] + uE[:, 1:])
    incP = dt * cfg.v_P * 0.5 * (uP[:, :-1] + uP[:, 1:])
    zero = np.zeros((x0.shape[0], 1, 2))
    xE_l = x0[:, None, :] + np.concatenate([zero, np.cumsum(incE, 1)], 1)
    xP_l = np.concatenate([zero, np.cumsum(incP, 1)], 1)
    D = xE_l - xP_l                                          # lifted E - P
    d = np.linalg.norm(min_image(D, L), axis=-1)             # (n_h, n_t+1)
    assert np.allclose(d, np.linalg.norm(min_image(xE - xP, L), axis=-1), atol=1e-9)

    Delta0 = min_image(x0, L)
    d0 = np.linalg.norm(Delta0, axis=1)
    capturable = d0 <= cfg.reach
    Vstar = (d0 - eps) / dv
    chart_opt = np.rint((Delta0 - x0) / L).astype(int)

    captured = d.min(1) <= eps
    kc = np.where(captured, np.argmax(d <= eps, axis=1), -1)
    Tc = np.where(captured, kc * dt, np.nan)
    chart_cap = np.full((x0.shape[0], 2), 0, dtype=int)
    for j in np.where(captured)[0]:
        chart_cap[j] = np.rint(-D[j, kc[j]] / L).astype(int)
    kmin = d.argmin(1)
    chart_closest = np.stack([np.rint(-D[j, kmin[j]] / L).astype(int)
                              for j in range(x0.shape[0])])

    # best-response pursuer against the trained evader's open-loop path
    s = np.linspace(0.0, 1.0, refine + 1)[:-1]
    xs = (xE_l[:, :-1, None, :] + s[None, None, :, None] * incE[:, :, None, :])
    xs = np.concatenate([xs.reshape(x0.shape[0], -1, 2), xE_l[:, -1:, :]], 1)
    ts = np.concatenate([(tg[:-1, None] + s[None, :] * dt).reshape(-1), tg[-1:]])
    dT0 = np.linalg.norm(min_image(xs, L), axis=-1)          # torus dist to origin
    ok = dT0 <= cfg.v_P * ts[None, :] + eps
    br_cap = ok.any(1)
    kb = np.where(br_cap, np.argmax(ok, axis=1), -1)
    T_BR = np.where(br_cap, ts[np.maximum(kb, 0)], np.nan)
    chart_br = np.zeros((x0.shape[0], 2), dtype=int)
    for j in np.where(br_cap)[0]:
        chart_br[j] = np.rint(-xs[j, kb[j]] / L).astype(int)

    windE = (xE_l[:, -1] - x0) / L
    tc_cap = np.where(np.isnan(Tc), cfg.t_max, Tc)
    tb_cap = np.where(np.isnan(T_BR), cfg.t_max, T_BR)
    return {
        "Delta0": Delta0.tolist(), "d0": d0.tolist(),
        "capturable": capturable.tolist(), "n_capturable": int(capturable.sum()),
        "Vstar": Vstar.tolist(),
        "captured": captured.tolist(), "Tc": [None if np.isnan(v) else float(v) for v in Tc],
        "n_captured": int(captured.sum()),
        "n_captured_capturable": int((captured & capturable).sum()),
        "n_captured_uncapturable": int((captured & ~capturable).sum()),
        "T_BR": [None if np.isnan(v) else float(v) for v in T_BR],
        "n_br_captured": int(br_cap.sum()),
        "n_br_captured_uncapturable": int((br_cap & ~capturable).sum()),
        "pursuer_regret_mean": float((tc_cap - tb_cap).mean()),
        "chart_opt": chart_opt.tolist(), "chart_cap": chart_cap.tolist(),
        "chart_closest": chart_closest.tolist(), "chart_br": chart_br.tolist(),
        "wind_E": windE.tolist(),
        "dmin": d.min(1).tolist(),
    }


def static_alignment(u: np.ndarray, Delta0: np.ndarray) -> list:
    """cos angle between the time-mean heading of each head and the saddle
    direction Delta0/|Delta0| (both players' saddle headings are Delta0-hat)."""
    ubar = u.mean(1)
    dh = Delta0 / np.linalg.norm(Delta0, axis=1, keepdims=True)
    nb = np.linalg.norm(ubar, axis=1)
    return ((ubar * dh).sum(1) / np.maximum(nb, 1e-12)).tolist()


# ----------------------------------------------------------- aggregates
def full_metrics(pE: dict, pP: dict, x0: np.ndarray, cfg: RunConfig) -> dict:
    _, phid = dense_phi(cfg)
    g = game_stats(pE, pP, x0, cfg)
    cap = np.array(g["capturable"])
    out = {"game": g}
    for a, p in (("E", pE), ("P", pP)):
        u, z, H = curves(p, phid)
        out[a] = {
            "ens": ensemble_stats(u, z),
            "ens_cap": ensemble_stats(u, z, np.where(cap)[0]),
            "ens_uncap": ensemble_stats(u, z, np.where(~cap)[0]),
            "lat": latent_stats(H, cfg.K),
            "band": unit_band_stats(u, cfg.K),
            "heads": head_stats(p["W"]),
            "ubar": u.mean(1).tolist(),
            "static_align": static_alignment(u, np.array(g["Delta0"])),
        }
    return out


def light_metrics(pE: dict, pP: dict, x0: np.ndarray, cfg: RunConfig, it: int) -> dict:
    _, phid = dense_phi(cfg, 512)
    out = {"it": it}
    for a, p in (("E", pE), ("P", pP)):
        u, z, H = curves(p, phid)
        e = ensemble_stats(u, z)
        Hc = H - H.mean(0)
        out[a] = {"gram_pr": e["gram_pr"], "constancy": e["constancy"],
                  "tpr_U": e["tpr_U"], "tpr_Z": e["tpr_Z"],
                  "pr_H": PR(_eig_desc(Hc.T @ Hc))}
    tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phig = fourier_features(tg, cfg.t_max, cfg.K)
    uE, _ = forward_dirs(pE, phig); uP, _ = forward_dirs(pP, phig)
    xE, xP = rollout(uE, uP, x0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    d = np.linalg.norm(min_image(xE - xP, cfg.L), axis=-1)
    out["n_captured"] = int((d.min(1) <= cfg.eps).sum())
    return out


def saddle_reference(x0: np.ndarray, cfg: RunConfig) -> dict:
    """Gram PR of the analytic constant-heading saddle for this IC set."""
    Delta0 = min_image(x0, cfg.L)
    dh = Delta0 / np.linalg.norm(Delta0, axis=1, keepdims=True)
    cap = np.linalg.norm(Delta0, axis=1) <= cfg.reach
    res = {"gram_pr_saddle": PR(_eig_desc(dh.T @ dh))}
    res["gram_pr_saddle_cap"] = PR(_eig_desc(dh[cap].T @ dh[cap])) if cap.sum() >= 2 else None
    return res
