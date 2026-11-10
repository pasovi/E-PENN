"""Participation ratios with the time after capture removed.

Head j (one evader initial condition) is in play on A_j = {t <= T_c^j}, where T_c^j is the capture time of the
trained pair (first grid time with d <= eps, `game.Tc` of the records, as in checks/metrics.game_stats). A head that
is never captured is in play on the whole of [0, t_max]. After capture nothing a head does enters the game, nor the
capture-truncated loss, so every time average is restricted to the time in play. The curves are those of
checks/metrics (T = 1024 dense time points; u^j unit heading, z^j = W^j H head output, H latent).

  Gram PR   G_jk = <u^j . u^k> over A_j & A_k (the time both heads are in play); PR = (tr G)^2 / tr(G^2).
            The diagonal is 1, so each head counts once, as in the whole-window Gram. If every heading is constant
            while in play, G_jk = cos(theta_j - theta_k) exactly and Gram PR = 2 / (1 + r^2) <= 2 (Proposition 1).
            G need not be positive semidefinite, hence the trace form (`gram_min_eig` records its lowest eigenvalue).
  eta       (1/n_h) sum_j |<u^j>_{A_j}|^2 = 1 iff every heading is constant while in play.
  PR(Z)     PR of the covariance of the in-play head outputs: head j contributes z^j(t) - <z^j>_{A_j} for t in A_j and
            nothing after capture; C = sum_t X(t) X(t)^T, X = (x^1, ..., x^{n_h}) in R^{2 n_h}.
  PR(H)     PR of the latent covariance over the in-play samples (j, t), t in A_j, i.e. the time points weighted by the
            number of heads still in play, w(t) = #{j : t in A_j}: centred (pr_H) and uncentred (pr_H_unc).
All of them equal the whole-window statistics of checks/metrics when no head is captured.

Two descriptors of the latent's motion: latent_motion = tr C_H / (tr C_H + |<H>|^2) over the in-play samples (the share
of the latent's second moment that is time variation), and pr_H_t1, the centred PR(H) over t <= 1 (the first quarter
of the episode, whole window).

Other ways of removing the time after capture, kept to show how much the result depends on the choice:
  gram_pr_frozen   whole window, each captured heading held at its value at T_c (as in the dv_D1 READMEs)
  gram_pr_clock    each head's time in play rescaled to [0, 1] ("own clock"), then the whole-window Gram PR
  gram_pr_static   the Gram PR of the in-play mean headings, 2 / (1 + r^2) (the constant-heading value)
  tpr_Z_pair       PR(Z) from the covariances of z^j and z^k over A_j & A_k (trace form; not PSD in general)
  tpr_Z_clock      PR(Z) on the own clocks
  tpr_Z_lat        PR(W C_H W^T), C_H the in-play latent covariance of PR(H): the stacked outputs W H(t) over the same
                   samples as PR(H) (this keeps the outputs of heads already captured)

Writes <folder>/dv<dv>/precap.json, one entry per run. Usage: python3 torch_actor/dv_D1_precap/precap.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "checks"))
import metrics as M                                            # noqa: E402
from core import RunConfig, forward_dirs, load_ckpt, fourier_features  # noqa: E402


def pr_trace(C):
    """(tr C)^2 / tr(C^2), the participation ratio without an eigendecomposition."""
    return float(np.trace(C) ** 2 / (C * C).sum())


def pr_eig(C):
    return M.PR(M._eig_desc(C))


def in_play(t, Tc, t_max):
    """(n_h, T) mask of the time in play, t <= T_c^j (all of [0, t_max] if never captured)."""
    Tj = np.array([t_max if c is None else c for c in Tc])
    return t[None, :] <= Tj[:, None], Tj


def precap_stats(p, Tc, cfg, t, phi):
    u, z, H = M.curves(p, phi)
    n_h, T = u.shape[:2]
    m, Tj = in_play(t, Tc, cfg.t_max)
    mf = m.astype(float)
    nn = mf @ mf.T                                             # time both heads are in play (grid points)
    o = {}
    # strategy, gauge-invariant: Gram over the time both heads are in play
    um = u * mf[:, :, None]
    G = np.einsum("jtc,ktc->jk", um, um) / nn
    o["gram_pr"] = pr_trace(G)
    o["gram_min_eig"] = float(np.linalg.eigvalsh(G)[0])
    ubar = np.einsum("jt,jtc->jc", mf, u) / mf.sum(1)[:, None]
    o["constancy"] = float((ubar ** 2).sum(1).mean())
    dh = ubar / np.linalg.norm(ubar, axis=1, keepdims=True)
    o["gram_pr_static"] = pr_eig(dh @ dh.T)
    # strategy, the draft's PR(Z): covariance of the in-play head outputs
    zm = z * mf[:, :, None]
    mu = zm.sum(1) / mf.sum(1)[:, None]
    X = ((z - mu[:, None, :]) * mf[:, :, None]).transpose(1, 0, 2).reshape(T, -1)
    o["tpr_Z"] = pr_eig(X.T @ X / T)
    # latent: in-play samples (j, t), weights w(t) = number of heads in play
    w = mf.sum(0)
    Hc = H - (w[:, None] * H).sum(0) / w.sum()
    C_H = (w[:, None] * Hc).T @ Hc / w.sum()
    o["pr_H"] = pr_eig(C_H)
    o["pr_H_unc"] = pr_eig((w[:, None] * H).T @ H / w.sum())
    mu_H = (w[:, None] * H).sum(0) / w.sum()
    o["latent_motion"] = float(np.trace(C_H) / (np.trace(C_H) + mu_H @ mu_H))
    o["pr_H_t1"] = M.latent_stats(H[t <= 1.0], cfg.K)["pr_H"]
    # the alternatives
    uf = u.copy()
    for j, c in enumerate(Tc):
        if c is not None:
            k = int(m[j].sum()) - 1                            # last in-play grid point
            uf[j, k + 1:] = u[j, k]
    o["gram_pr_frozen"] = M.ensemble_stats(uf, z)["gram_pr"]
    tt = (np.linspace(0.0, 1.0, T)[None, :] * Tj[:, None]).reshape(-1)
    _, cache = forward_dirs(p, fourier_features(tt, cfg.t_max, cfg.K))
    zc = np.einsum("jab,jtb->jta", p["W"], cache["H"].reshape(n_h, T, -1))
    e = M.ensemble_stats(zc / np.linalg.norm(zc, axis=-1, keepdims=True), zc)
    o["gram_pr_clock"], o["tpr_Z_clock"] = e["gram_pr"], e["tpr_Z"]
    S = np.einsum("jtc,ktd->jkcd", zm, zm)
    s1 = np.einsum("jtc,kt->jkc", zm, mf)                      # sum of z^j over A_j & A_k
    Cp = S / nn[:, :, None, None] - np.einsum("jkc,kjd->jkcd", s1, s1) / (nn ** 2)[:, :, None, None]
    o["tpr_Z_pair"] = pr_trace(Cp.transpose(0, 2, 1, 3).reshape(2 * n_h, 2 * n_h))
    Ws = p["W"].reshape(2 * n_h, -1)
    o["tpr_Z_lat"] = pr_eig(Ws @ C_H @ Ws.T)
    return o


def grid(d):
    """Pre-capture statistics of every run of the grid in d, cached in d/precap.json."""
    out_path = os.path.join(d, "precap.json")
    files = sorted(glob.glob(os.path.join(d, "results", "grid_*.json")))
    if os.path.exists(out_path):
        out = json.load(open(out_path))
        if len(out) == len(files):
            return out
    out = {}
    for f in files:
        rec = json.load(open(f))
        c = rec["config"]
        cfg = RunConfig(K=c["K"], t_max=c["t_max"])
        t, phi = M.dense_phi(cfg)
        it = max(int(k) for k in rec["full"])
        Tc = rec["full"][str(it)]["game"]["Tc"]
        pE, pP, _ = load_ckpt(os.path.join(d, "ckpt", os.path.basename(f).replace(".json", ".npz")), it)
        m, _ = in_play(t, Tc, cfg.t_max)
        out[rec["name"]] = {"iteration": it, "free_share": float(1.0 - m.mean()),
                            "E": precap_stats(pE, Tc, cfg, t, phi), "P": precap_stats(pP, Tc, cfg, t, phi)}
    json.dump(out, open(out_path, "w"), indent=0)
    return out


def main():
    for d in sorted(glob.glob(os.path.join(HERE, "dv*"))):
        if os.path.isdir(os.path.join(d, "results")):
            out = grid(d)
            print(f"{os.path.basename(d)}: {len(out)} runs -> {os.path.relpath(os.path.join(d, 'precap.json'), HERE)}")


if __name__ == "__main__":
    main()
