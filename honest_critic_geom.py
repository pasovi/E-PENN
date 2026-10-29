"""
Honest HJI critic training for bounded / non-orientable geometries.

V(x_E, x_P, tau) is a smooth neural network.  Gradients ∇_{x_E} V and
∇_{x_P} V are estimated by central finite differences in the appropriate
sub-coordinates.  The training loss is:

    L = HJI residual^2 + λ_term V(state, 0)^2 + λ_cap V(state, τ)^2

with the residual

    R = ∂_τ V - C(state) - v_E ||∇_{x_E} V|| + v_P ||∇_{x_P} V||,

C(state) = 1 outside the capture set, 0 inside.

NO ANALYTIC V VALUE IS USED ANYWHERE IN THE TRAINING LOSS.
"""
from __future__ import annotations
import math, os, json, time
from dataclasses import dataclass
import numpy as np

import pursuit_evasion_numpy as pen
from pursuit_evasion_numpy import Adam
import geometries as geo


# =============================================================================
# Two-hidden-layer Tanh MLP
# =============================================================================
def init_params(D: int, hidden: int, rng: np.random.Generator) -> dict:
    return {
        "B1": rng.standard_normal((hidden, D)) * np.sqrt(1.5 / D),
        "b1": np.zeros(hidden),
        "B2": rng.standard_normal((hidden, hidden)) * np.sqrt(1.5 / hidden),
        "b2": np.zeros(hidden),
        "w":  rng.standard_normal(hidden) * np.sqrt(1.5 / hidden),
        "c":  np.zeros(1),
    }


def forward(params: dict, phi: np.ndarray) -> tuple[np.ndarray, dict]:
    z1 = phi @ params["B1"].T + params["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ params["B2"].T + params["b2"]
    h2 = np.tanh(z2)
    V = h2 @ params["w"] + params["c"]
    cache = {"phi": phi, "h1": h1, "h2": h2,
             "B1": params["B1"], "B2": params["B2"], "w": params["w"]}
    return V, cache


def backward(params, cache, dV):
    h1, h2 = cache["h1"], cache["h2"]
    B1, B2, w = cache["B1"], cache["B2"], cache["w"]
    dw = h2.T @ dV
    dc = dV.sum(keepdims=True)
    dh2 = np.outer(dV, w)
    dz2 = (1 - h2 ** 2) * dh2
    dB2 = dz2.T @ h1
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ B2
    dz1 = (1 - h1 ** 2) * dh1
    dB1 = dz1.T @ cache["phi"]
    db1 = dz1.sum(axis=0)
    return {"B1": dB1, "b1": db1, "B2": dB2, "b2": db2, "w": dw, "c": dc}


# =============================================================================
# HJI residual via finite differences (in x_E and x_P subspaces)
# =============================================================================
def honest_loss(params: dict,
                 G: geo.Geometry,
                 xE_res: np.ndarray, xP_res: np.ndarray, tau_res: np.ndarray,
                 xE_bc: np.ndarray, xP_bc: np.ndarray,
                 xE_cap: np.ndarray, xP_cap: np.ndarray, tau_cap: np.ndarray,
                 v_E: float, v_P: float, eps: float,
                 h_d: float, h_t: float,
                 lam_term: float = 5.0, lam_cap: float = 5.0):
    N_res = xE_res.shape[0]
    N_bc = xE_bc.shape[0]
    N_cap = xE_cap.shape[0]
    d = G.d_pos

    # Build all evaluation points needed for FD derivatives.
    # Centre + ±e_i for x_E (2d perturbations), ±e_i for x_P (2d perturbations),
    # ± h_t for tau.
    eye = np.eye(d) * h_d
    chunks_xE, chunks_xP, chunks_tau = [], [], []

    def add(xE, xP, tau):
        chunks_xE.append(xE); chunks_xP.append(xP); chunks_tau.append(tau)

    # Centre
    add(xE_res, xP_res, tau_res)
    # x_E ± e_i
    for i in range(d):
        add(xE_res + eye[i], xP_res, tau_res)
        add(xE_res - eye[i], xP_res, tau_res)
    # x_P ± e_i
    for i in range(d):
        add(xE_res, xP_res + eye[i], tau_res)
        add(xE_res, xP_res - eye[i], tau_res)
    # tau ± h_t
    add(xE_res, xP_res, tau_res + h_t)
    add(xE_res, xP_res, tau_res - h_t)
    # BC points
    add(xE_bc, xP_bc, np.zeros(N_bc))
    add(xE_cap, xP_cap, tau_cap)

    xE_all = np.concatenate(chunks_xE, axis=0)
    xP_all = np.concatenate(chunks_xP, axis=0)
    tau_all = np.concatenate(chunks_tau, axis=0)
    phi = G.encode_state(xE_all, xP_all, tau_all)
    V_all, cache = forward(params, phi)

    # Slice
    cursor = 0
    def take(n):
        nonlocal cursor
        out = V_all[cursor: cursor + n]
        cursor += n
        return out
    V0 = take(N_res)
    V_E_plus = [take(N_res) for _ in range(d)]
    V_E_minus = []
    # rebuild interleaving: actually let me redo
    cursor = N_res
    V_E_plus, V_E_minus = [], []
    for i in range(d):
        V_E_plus.append(V_all[cursor: cursor + N_res]); cursor += N_res
        V_E_minus.append(V_all[cursor: cursor + N_res]); cursor += N_res
    V_P_plus, V_P_minus = [], []
    for i in range(d):
        V_P_plus.append(V_all[cursor: cursor + N_res]); cursor += N_res
        V_P_minus.append(V_all[cursor: cursor + N_res]); cursor += N_res
    V_t_plus = V_all[cursor: cursor + N_res]; cursor += N_res
    V_t_minus = V_all[cursor: cursor + N_res]; cursor += N_res
    V_bc = V_all[cursor: cursor + N_bc]; cursor += N_bc
    V_cap = V_all[cursor: cursor + N_cap]; cursor += N_cap

    # Derivatives
    dV_dt = (V_t_plus - V_t_minus) / (2.0 * h_t)
    grad_E = np.stack(
        [(V_E_plus[i] - V_E_minus[i]) / (2 * h_d) for i in range(d)], axis=1)
    grad_P = np.stack(
        [(V_P_plus[i] - V_P_minus[i]) / (2 * h_d) for i in range(d)], axis=1)
    gE_norm = np.sqrt((grad_E ** 2).sum(axis=1) + 1e-10)
    gP_norm = np.sqrt((grad_P ** 2).sum(axis=1) + 1e-10)

    # Residual = dt V - C - v_E |gE| + v_P |gP|; C = 1 outside capture set
    res = dV_dt - 1.0 - v_E * gE_norm + v_P * gP_norm
    loss_hji = (res ** 2).mean()
    loss_term = (V_bc ** 2).mean() if N_bc else 0.0
    loss_cap = (V_cap ** 2).mean() if N_cap else 0.0
    loss = loss_hji + lam_term * loss_term + lam_cap * loss_cap

    # Backward
    dV_total = np.zeros_like(V_all)
    d_res = 2.0 * res / N_res
    # dV_dt
    dV_total[N_res * (1 + 2 * 2 * d): N_res * (1 + 2 * 2 * d) + N_res] += d_res * 0.5 / h_t
    dV_total[N_res * (1 + 2 * 2 * d) + N_res: N_res * (1 + 2 * 2 * d) + 2 * N_res] -= d_res * 0.5 / h_t
    # -v_E |gE| contribution: ∂/∂grad_E[i] = -v_E grad_E[i]/|gE|
    for i in range(d):
        coef_p_E = d_res * (-v_E) * grad_E[:, i] / gE_norm * (0.5 / h_d)
        # indices for V_E_plus[i] and V_E_minus[i]
        idx_p = N_res * (1 + 2 * i)
        idx_m = N_res * (1 + 2 * i + 1)
        dV_total[idx_p: idx_p + N_res] += coef_p_E
        dV_total[idx_m: idx_m + N_res] -= coef_p_E
    # +v_P |gP|
    for i in range(d):
        coef_p_P = d_res * (+v_P) * grad_P[:, i] / gP_norm * (0.5 / h_d)
        idx_p = N_res * (1 + 2 * d + 2 * i)
        idx_m = N_res * (1 + 2 * d + 2 * i + 1)
        dV_total[idx_p: idx_p + N_res] += coef_p_P
        dV_total[idx_m: idx_m + N_res] -= coef_p_P
    # BC
    if N_bc:
        idx = N_res * (1 + 2 * d + 2 * d + 2)
        dV_total[idx: idx + N_bc] += lam_term * 2.0 * V_bc / N_bc
    if N_cap:
        idx = N_res * (1 + 2 * d + 2 * d + 2) + N_bc
        dV_total[idx: idx + N_cap] += lam_cap * 2.0 * V_cap / N_cap

    grads = backward(params, cache, dV_total)
    diag = {
        "loss": float(loss),
        "loss_hji": float(loss_hji),
        "loss_term": float(loss_term),
        "loss_cap": float(loss_cap),
        "mean_gE": float(gE_norm.mean()),
        "mean_gP": float(gP_norm.mean()),
        "mean_dV_dt": float(dV_dt.mean()),
    }
    return loss, grads, diag


# =============================================================================
# Sampler
# =============================================================================
def sample_batch(G: geo.Geometry, rng: np.random.Generator,
                  eps: float, t_max: float,
                  n_res: int, n_term: int, n_cap: int):
    # Residual sample: any two positions, with x_E - x_P far enough from
    # capture
    xE_r = G.sample_positions(rng, n_res * 2)
    xP_r = G.sample_positions(rng, n_res * 2)
    keep = G.distance(xE_r, xP_r) > 1.5 * eps
    xE_r = xE_r[keep][:n_res]; xP_r = xP_r[keep][:n_res]
    # top up if needed
    while xE_r.shape[0] < n_res:
        need = n_res - xE_r.shape[0]
        xe = G.sample_positions(rng, need)
        xp = G.sample_positions(rng, need)
        xE_r = np.concatenate([xE_r, xe], axis=0)
        xP_r = np.concatenate([xP_r, xp], axis=0)
    tau_r = rng.uniform(0.05, t_max, size=n_res)

    # Terminal
    xE_b = G.sample_positions(rng, n_term)
    xP_b = G.sample_positions(rng, n_term)

    # Capture set: x_E close to x_P
    xP_c = G.sample_positions(rng, n_cap, min_dist_from_origin=0.0)
    # perturb x_E by a small random direction
    dirn = rng.standard_normal(size=(n_cap, G.d_pos))
    dirn /= np.linalg.norm(dirn, axis=1, keepdims=True).clip(min=1e-12)
    radii = rng.uniform(0, eps * 0.9, size=n_cap)[:, None]
    xE_c = G.project(xP_c + radii * dirn)
    tau_c = rng.uniform(0.0, t_max, size=n_cap)

    return xE_r, xP_r, tau_r, xE_b, xP_b, xE_c, xP_c, tau_c


# =============================================================================
# Training driver
# =============================================================================
@dataclass
class GeomTrainConfig:
    v_E: float = 1.0
    v_P: float = 1.2
    eps: float = 0.1
    hidden: int = 96
    n_iters: int = 3000
    n_res: int = 256
    n_term: int = 64
    n_cap: int = 64
    lr_max: float = 3e-3
    lr_min: float = 1e-4
    schedule_total: int = 30000
    lam_term: float = 5.0
    lam_cap: float = 5.0
    log_every: int = 100
    seed: int = 0


def train_geom(G: geo.Geometry, cfg: GeomTrainConfig,
                ckpt_path: str, resume: bool = False):
    rng = np.random.default_rng(cfg.seed)
    D = G.encoded_dim()
    params = init_params(D, cfg.hidden, rng)
    opt = Adam(params, lr=cfg.lr_max)
    hist = {"iter": [], "loss": [], "loss_hji": [], "loss_term": [],
            "loss_cap": [], "mean_gE": [], "mean_gP": [], "lr": []}
    iter0 = 0
    if resume and os.path.exists(ckpt_path):
        npz = np.load(ckpt_path)
        for k in params:
            params[k] = npz[k]
        opt = Adam(params, lr=cfg.lr_max)
        iter0 = int(npz["iter0"])
        if "hist" in npz.files:
            hist = json.loads(str(npz["hist"]))
        rng = np.random.default_rng(int(cfg.seed + iter0))
        print(f"[{G.name}] resumed at iter {iter0}")

    h_d = G.fd_step(); h_t = G.fd_step()
    t_max_attr = getattr(G, "t_max", 4.0)
    t0 = time.time()
    for it in range(cfg.n_iters):
        global_it = iter0 + it
        progress = min(global_it / cfg.schedule_total, 1.0)
        lr = cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1 + math.cos(math.pi * progress))
        opt.lr = lr

        batches = sample_batch(G, rng, cfg.eps, t_max_attr,
                                cfg.n_res, cfg.n_term, cfg.n_cap)
        loss, grads, diag = honest_loss(
            params, G, *batches,
            cfg.v_E, cfg.v_P, cfg.eps,
            h_d=h_d, h_t=h_t,
            lam_term=cfg.lam_term, lam_cap=cfg.lam_cap)
        opt.step(params, grads, sign=-1.0)
        if it % cfg.log_every == 0 or it == cfg.n_iters - 1:
            hist["iter"].append(global_it)
            for k in ["loss", "loss_hji", "loss_term", "loss_cap",
                       "mean_gE", "mean_gP"]:
                hist[k].append(diag[k])
            hist["lr"].append(lr)
            print(f"[{G.name}] it {global_it:5d}  loss={diag['loss']:.4e}  "
                  f"hji={diag['loss_hji']:.4e}  "
                  f"|∇_E V|≈{diag['mean_gE']:.3f}  "
                  f"|∇_P V|≈{diag['mean_gP']:.3f}  "
                  f"lr={lr:.2e}  [{time.time()-t0:.0f}s]")

    npz_out = {k: v for k, v in params.items()}
    npz_out["iter0"] = iter0 + cfg.n_iters
    npz_out["hist"] = json.dumps(hist)
    np.savez(ckpt_path, **npz_out)
    return params, hist
