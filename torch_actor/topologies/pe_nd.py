"""The baseline D1 pipeline on the flat torus T^d = [0, L)^d for any d (S^1 is d = 1, T^2 the paper, T^3).

Same actor (Fourier features of time, one tanh layer of 16 units, one linear bias-free head per IC),
same loss (soft capture time, one-sided no-capture penalty, capture-truncated distance shaping), same
protocol (pursuer warm-up, 3:1 Adam steps, Adam resets).  Headings are the head output normalised to
the unit sphere S^{d-1}, exactly as in pe_torch for d >= 2.  For d = 1 the unit sphere is {-1, +1} and
normalisation has no gradient, so the heading is u = tanh(z) in [-1, 1] (the player may also go slower
than full speed; the saddle uses full speed).  The game-theoretic quantities (geodesic distance,
reach R = (v_P - v_E) t_max + eps, V* = (d0 - eps) / (v_P - v_E), best-response time) are the same
formulas in every dimension.

Reuses pe_torch.rollout, pe_torch.game_loss, pe_torch.fourier_features and pe_torch.make_adam, which
are dimension-agnostic.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import pe_torch as pt  # noqa: E402

Config = pt.Config


# ------------------------------------------------------------------ init
def init_params(n_h: int, n_b: int, D: int, dim: int, rng: np.random.Generator) -> dict:
    return {"B": rng.standard_normal((n_b, D)) * np.sqrt(1.5 / D), "b": np.zeros(n_b),
            "W": rng.standard_normal((n_h, dim, n_b)) * np.sqrt(1.0 / n_b)}


def geodesic(x: np.ndarray, L: float) -> np.ndarray:
    """Torus distance of points x (..., d) to the origin."""
    m = np.minimum(x % L, L - x % L)
    return np.sqrt((m ** 2).sum(-1))


def sample_ICs(n_h: int, L: float, min_dist: float, seed: int, dim: int) -> np.ndarray:
    """Uniform on T^d, rejecting points closer than min_dist (geodesic) to the pursuer at the origin."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n_h:
        p = rng.uniform(0.0, L, size=dim)
        if geodesic(p, L) >= min_dist:
            pts.append(p)
    return np.array(pts)


def initial_state(cfg: Config, dim: int, init_seed: int, ic_seed: int):
    rng = np.random.default_rng(init_seed)
    pE = init_params(cfg.n_h, cfg.n_b, cfg.D, dim, rng)
    pP = init_params(cfg.n_h, cfg.n_b, cfg.D, dim, rng)
    return pE, pP, sample_ICs(cfg.n_h, cfg.L, cfg.ic_min_dist, ic_seed, dim)


def grid_ICs(n_per_axis: int, L: float, dim: int) -> np.ndarray:
    """Cell-centred uniform grid of n^d points on T^d (none on the origin or on the cut locus)."""
    c = (np.arange(n_per_axis) + 0.5) * L / n_per_axis
    return np.stack(np.meshgrid(*([c] * dim), indexing="ij"), -1).reshape(-1, dim)


# ------------------------------------------------------------------ model (torch)
def forward_dirs(p: dict, phi: torch.Tensor):
    """u: (R, n_h, T, d) headings; H: (R, T, n_b); z: (R, n_h, T, d)."""
    H = torch.tanh(torch.einsum("td,rbd->rtb", phi, p["B"]) + p["b"][:, None, :])
    z = torch.einsum("rjab,rtb->rjta", p["W"], H)
    if z.shape[-1] == 1:
        u = torch.tanh(z)
    else:
        u = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return u, H, z


def to_torch(p: dict, device, dtype, grad_keys=("B", "b", "W")):
    return {k: torch.tensor(np.asarray(v, np.float64)[None], device=device, dtype=dtype, requires_grad=(k in grad_keys))
            for k, v in p.items()}


def step(who: str, pE, pP, x0, phi, cfg: Config, optE, optP):
    uE, _, _ = forward_dirs(pE, phi)
    uP, _, _ = forward_dirs(pP, phi)
    if who == "P":
        uE = uE.detach()
    else:
        uP = uP.detach()
    xE, xP = pt.rollout(uE, uP, x0, cfg)
    L_E, L_P, _, _ = pt.game_loss(xE, xP, cfg)
    loss = L_P.sum() if who == "P" else L_E.sum()
    opt = optP if who == "P" else optE
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return float(L_E.item()), float(L_P.item())


def train(cfg: Config, pE_np: dict, pP_np: dict, x0_np: np.ndarray, device="cpu", dtype=torch.float64,
          frozen_body: bool = False, snapshot_every: int = 0, log_every: int = 0):
    """Self-play with the baseline protocol.  frozen_body: only the heads W carry gradients (TL).
    Returns final params (NumPy), the per-gradient-step loss history (L_E, L_P), the snapshots
    (list of (label, gradient step, pE, pP)) and the wall time."""
    gk = ("W",) if frozen_body else ("B", "b", "W")
    pE, pP = to_torch(pE_np, device, dtype, gk), to_torch(pP_np, device, dtype, gk)
    x0 = torch.tensor(x0_np[None], device=device, dtype=dtype)
    t = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=device, dtype=dtype)
    phi = pt.fourier_features(t, cfg.t_max, cfg.K)
    grads = lambda p: {k: v for k, v in p.items() if v.requires_grad}       # noqa: E731
    optE, optP = pt.make_adam(grads(pE), cfg.lr_E), pt.make_adam(grads(pP), cfg.lr_P)
    hist, snaps, gstep = [], [], 0
    npy = lambda p: {k: v.detach().cpu().numpy()[0].copy() for k, v in p.items()}   # noqa: E731
    t0 = time.time()
    if snapshot_every:
        snaps.append(("init", 0, npy(pE), npy(pP)))
    for k in range(1, cfg.warmup + 1):
        hist.append(step("P", pE, pP, x0, phi, cfg, optE, optP)); gstep += 1
        if snapshot_every and k % snapshot_every == 0:
            snaps.append((f"warm-up step {k} of {cfg.warmup}", gstep, npy(pE), npy(pP)))
    for it in range(1, cfg.n_iters + 1):
        if cfg.adam_reset_every and it > 1 and (it - 1) % cfg.adam_reset_every == 0:
            optE, optP = pt.make_adam(grads(pE), cfg.lr_E), pt.make_adam(grads(pP), cfg.lr_P)
        for _ in range(cfg.n_P_steps):
            hist.append(step("P", pE, pP, x0, phi, cfg, optE, optP)); gstep += 1
        for _ in range(cfg.n_E_steps):
            hist.append(step("E", pE, pP, x0, phi, cfg, optE, optP)); gstep += 1
        if snapshot_every and it % snapshot_every == 0:
            snaps.append((f"self-play iteration {it} of {cfg.n_iters}", gstep, npy(pE), npy(pP)))
        if log_every and it % log_every == 0:
            print(f"  it {it:6d}  L_E {hist[-1][0]:+.4f}  L_P {hist[-1][1]:+.4f}  [{time.time() - t0:.0f}s]", flush=True)
    return npy(pE), npy(pP), np.array(hist), snaps, time.time() - t0


# ------------------------------------------------------------------ metrics (NumPy)
def fourier_np(t, t_max, K):
    w = 2 * np.pi / t_max * np.arange(1, K + 1)
    return np.concatenate([np.sin(np.outer(t, w)), np.cos(np.outer(t, w)), (t / t_max)[:, None]], 1)


def curves_np(p: dict, phi: np.ndarray):
    """u (n_h, T, d), z (n_h, T, d), H (T, n_b)."""
    H = np.tanh(phi @ p["B"].T + p["b"])
    z = np.einsum("jab,tb->jta", p["W"], H)
    u = np.tanh(z) if z.shape[-1] == 1 else z / np.linalg.norm(z, axis=-1, keepdims=True).clip(min=1e-8)
    return u, z, H


def PR(ev):
    ev = np.clip(np.asarray(ev), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.sum() > 0 else float("nan")


def eig(C):
    return np.linalg.eigvalsh((C + C.T) / 2)[::-1]


def pr_stats(p: dict, cfg: Config, T: int = 512) -> dict:
    """Gram PR of the heading ensemble, temporal PR of u, constancy, PR(H) centred and uncentred."""
    t = np.linspace(0, cfg.t_max, T)
    u, z, H = curves_np(p, fourier_np(t, cfg.t_max, cfg.K))
    n_h = u.shape[0]
    U = u.reshape(n_h, -1) / np.sqrt(T)
    Xu = np.transpose(u, (1, 0, 2)).reshape(T, -1); Xu = Xu - Xu.mean(0)
    Hc = H - H.mean(0)
    return {"gram_pr": PR(eig(U @ U.T)), "tpr_U": PR(eig(Xu.T @ Xu / T)),
            "constancy": float((u.mean(1) ** 2).sum(1).mean()),
            "pr_H": PR(eig(Hc.T @ Hc / T)), "pr_H_unc": PR(eig(H.T @ H / T)),
            "eig_H": eig(Hc.T @ Hc / T).tolist(), "eig_gram": eig(U @ U.T).tolist()}


def game_stats(pE: dict, pP: dict, x0: np.ndarray, cfg: Config, refine: int = 16) -> dict:
    """Per-head game outcome on the training time grid: as checks/metrics.game_stats, any dimension."""
    L, eps, dv = cfg.L, cfg.eps, cfg.v_P - cfg.v_E
    reach = dv * cfg.t_max + eps
    tg = np.linspace(0, cfg.t_max, cfg.n_t + 1); dt = cfg.t_max / cfg.n_t
    phig = fourier_np(tg, cfg.t_max, cfg.K)
    uE, _, _ = curves_np(pE, phig); uP, _, _ = curves_np(pP, phig)
    incE = dt * cfg.v_E * 0.5 * (uE[:, :-1] + uE[:, 1:]); incP = dt * cfg.v_P * 0.5 * (uP[:, :-1] + uP[:, 1:])
    n_h, d = x0.shape
    zero = np.zeros((n_h, 1, d))
    xE = x0[:, None, :] + np.concatenate([zero, np.cumsum(incE, 1)], 1)
    xP = np.concatenate([zero, np.cumsum(incP, 1)], 1)
    D = xE - xP
    dist = np.linalg.norm(pt_min_image(D, L), axis=-1)                     # (n_h, n_t+1)
    Delta0 = pt_min_image(x0, L); d0 = np.linalg.norm(Delta0, axis=1)
    capturable = d0 <= reach
    captured = dist.min(1) <= eps
    kc = np.where(captured, np.argmax(dist <= eps, axis=1), -1)
    Tc = np.where(captured, kc * dt, np.nan)
    chart_opt = np.rint((Delta0 - x0) / L).astype(int)
    kmin = dist.argmin(1)
    chart_closest = np.stack([np.rint(-D[j, kmin[j]] / L).astype(int) for j in range(n_h)])
    # best response: earliest time the pursuer, at full speed from the origin, can reach the evader's path
    s = np.linspace(0, 1, refine + 1)[:-1]
    xs = xE[:, :-1, None, :] + s[None, None, :, None] * incE[:, :, None, :]
    xs = np.concatenate([xs.reshape(n_h, -1, d), xE[:, -1:, :]], 1)
    ts = np.concatenate([(tg[:-1, None] + s[None, :] * dt).reshape(-1), tg[-1:]])
    ok = np.linalg.norm(pt_min_image(xs, L), axis=-1) <= cfg.v_P * ts[None, :] + eps
    br = ok.any(1); kb = np.where(br, np.argmax(ok, axis=1), 0)
    T_BR = np.where(br, ts[kb], np.nan)
    return {"d0": d0.tolist(), "capturable": capturable.tolist(), "Vstar": ((d0 - eps) / dv).tolist(),
            "captured": captured.tolist(), "Tc": [None if np.isnan(v) else float(v) for v in Tc],
            "T_BR": [None if np.isnan(v) else float(v) for v in T_BR],
            "dmin": dist.min(1).tolist(), "chart_opt": chart_opt.tolist(), "chart_closest": chart_closest.tolist(),
            "reach": float(reach), "xE": xE.tolist() if n_h <= 64 else None, "xP": xP.tolist() if n_h <= 64 else None}


def pt_min_image(x, L):
    return (x + L / 2.0) % L - L / 2.0


def summarise(g: dict, prE: dict, prP: dict) -> dict:
    cap, capt = np.array(g["capturable"]), np.array(g["captured"])
    Tc, Vs = np.array([np.nan if t is None else t for t in g["Tc"]]), np.array(g["Vstar"])
    r = Tc[cap & capt] / Vs[cap & capt]
    return {"n_heads": int(len(cap)), "n_capturable": int(cap.sum()),
            "capture_rate_capturable": float(capt[cap].mean()) if cap.any() else None,
            "capture_rate_uncapturable": float(capt[~cap].mean()) if (~cap).any() else None,
            "n_capturable_escaped": int((cap & ~capt).sum()), "n_uncapturable_captured": int((~cap & capt).sum()),
            "Tc_over_Vstar_captured_median_q1_q3": [float(np.median(r)), float(np.percentile(r, 25)), float(np.percentile(r, 75))] if r.size else None,
            "nonnearest_chart_capturable": int(sum(1 for j, c in enumerate(cap) if c and g["chart_closest"][j] != g["chart_opt"][j])),
            "best_response_capture_rate_capturable": float(np.mean([t is not None for t, c in zip(g["T_BR"], cap) if c])) if cap.any() else None,
            **{f"{k}_E": v for k, v in prE.items() if not k.startswith("eig")},
            **{f"{k}_P": v for k, v in prP.items() if not k.startswith("eig")}}
