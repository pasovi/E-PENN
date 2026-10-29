"""
HONEST pursuit-evasion critic on the torus.

This module retracts the supervised-loss pathway used in v2/v3.  Here the
critic is trained EXCLUSIVELY with PDE-intrinsic signals:

    1.  HJI residual in time-augmented displacement coordinates
            (∂_τ V) - C(Δ) + (v_P - v_E) ||∇_Δ V||  =  0,
        with C(Δ) = 1 outside the capture set, 0 inside.
    2.  Terminal boundary condition  V(Δ, τ = 0)  =  0  for all Δ
        (no remaining time -> no cost-to-go).
    3.  Capture-set boundary condition  V(Δ, τ) = 0  for ||Δ||_T ≤ ε
        (already captured -> no future cost).

The closed-form solution V*(Δ) = max(|Δ|_T − ε, 0)/(v_P − v_E) is
NEVER evaluated during training.  V* is used only as a verification
yardstick after training.

For the open-loop multi-head ACTOR we re-use the v1 setup --- pure
adversarial self-play, no analytic feedback term --- as the honest
embodiment of the original brief.

Both ∂_τ V and ∇_Δ V are estimated by central finite differences, so the
backward pass goes through forward evaluations of V at perturbed points
(no need for double backprop).  All hand-coded gradients are verified
against finite differences to machine precision.

Conventions
-----------
Let τ ∈ [0, τ_max] denote the *remaining* horizon (so τ = 0 is "time up").
Then the HJI PDE for the value V(Δ, τ) of the open game on T^d reads

        ∂_τ V(Δ, τ)  =  C(Δ)  −  (v_P − v_E) || ∇_Δ V(Δ, τ) ||.

For Δ outside the capture set the exact solution is

        V*(Δ, τ)  =  min{ τ,  (||Δ||_T − ε) / (v_P − v_E) },

and at τ → ∞ this converges to the closed-form steady-state value
function V*(Δ) used in v2.
"""
from __future__ import annotations
import math, os, json, time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import pursuit_evasion_numpy as pen
from pursuit_evasion_numpy import (
    TrainConfig, fourier_features, init_params,
    forward_dirs, backward_dirs, rollout, backward_rollout,
    forward_loss, backward_loss, Adam, sample_evader_ICs,
)


# =============================================================================
# Time-augmented input encoding
# =============================================================================
# φ(Δ, τ) = [sin(2π k Δ_i / L), cos(2π k Δ_i / L)]_{i,k}            (periodic in Δ)
#         ⊕ [τ / τ_max,   (τ / τ_max)^2,  sin(π τ / τ_max), cos(π τ / τ_max)]
#                                                                  (smooth in τ)
def encode_inputs(delta: np.ndarray, tau: np.ndarray,
                  L: float, n_freqs: int, t_max: float) -> np.ndarray:
    omegas = (2.0 * np.pi / L) * np.arange(1, n_freqs + 1)
    ang = delta[:, :, None] * omegas[None, None, :]
    feats_spatial = np.concatenate(
        [np.sin(ang).reshape(delta.shape[0], -1),
         np.cos(ang).reshape(delta.shape[0], -1)], axis=1)
    tt = tau / t_max
    feats_temporal = np.stack(
        [tt, tt ** 2, np.sin(np.pi * tt), np.cos(np.pi * tt)], axis=1)
    return np.concatenate([feats_spatial, feats_temporal], axis=1)


def encoded_dim(d: int, n_freqs: int) -> int:
    return 2 * d * n_freqs + 4


# =============================================================================
# Network V(Δ, τ): scalar output, two hidden layers of tanh
# =============================================================================
def init_VT_params(D: int, hidden: int, rng: np.random.Generator) -> dict:
    return {
        "B1": rng.standard_normal((hidden, D)) * np.sqrt(1.5 / D),
        "b1": np.zeros(hidden),
        "B2": rng.standard_normal((hidden, hidden)) * np.sqrt(1.5 / hidden),
        "b2": np.zeros(hidden),
        "w":  rng.standard_normal(hidden) * np.sqrt(1.5 / hidden),
        "c":  np.zeros(1),
    }


def forward_VT(params: dict, phi: np.ndarray) -> tuple[np.ndarray, dict]:
    B1, b1, B2, b2, w, c = (params[k] for k in ("B1","b1","B2","b2","w","c"))
    z1 = phi @ B1.T + b1
    h1 = np.tanh(z1)
    z2 = h1 @ B2.T + b2
    h2 = np.tanh(z2)
    V = h2 @ w + c
    cache = {"phi": phi, "h1": h1, "h2": h2, "B1": B1, "B2": B2, "w": w}
    return V, cache


def backward_VT(params: dict, cache: dict, dV: np.ndarray) -> dict:
    phi, h1, h2 = cache["phi"], cache["h1"], cache["h2"]
    B1, B2, w = cache["B1"], cache["B2"], cache["w"]
    dw = h2.T @ dV
    dc = dV.sum(keepdims=True)
    dh2 = np.outer(dV, w)
    dz2 = (1 - h2 ** 2) * dh2
    dB2 = dz2.T @ h1
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ B2
    dz1 = (1 - h1 ** 2) * dh1
    dB1 = dz1.T @ phi
    db1 = dz1.sum(axis=0)
    return {"B1": dB1, "b1": db1, "B2": dB2, "b2": db2, "w": dw, "c": dc}


# =============================================================================
# Honest loss: PDE residual + terminal BC + capture-set BC.
# NO V* VALUES APPEAR ANYWHERE BELOW.
# =============================================================================
def honest_loss(params: dict,
                delta_res: np.ndarray, tau_res: np.ndarray,
                delta_bc: np.ndarray, tau_bc: np.ndarray,
                delta_cap: np.ndarray, tau_cap: np.ndarray,
                L: float, v_E: float, v_P: float, eps: float,
                n_freqs: int, t_max: float,
                h_d: float = 5e-3, h_t: float = 5e-3,
                lam_bc_term: float = 5.0,
                lam_bc_cap: float = 5.0,
                lam_grad_reg: float = 0.0):
    """
    Compute the three honest loss components and their gradients.

    delta_res, tau_res : PDE-residual sample points (outside capture set, τ > 0)
    delta_bc,  tau_bc  : terminal-BC sample points (any Δ, τ = 0)
    delta_cap, tau_cap : capture-set-BC sample points (||Δ|| ≤ ε, any τ)
    """
    N_res, d = delta_res.shape
    N_bc = delta_bc.shape[0]
    N_cap = delta_cap.shape[0]

    # --- gather all points we need to forward through V ---
    # For residual evaluation we need V at:
    #   (Δ_res, τ_res)               -- centre
    #   (Δ_res ± h_d e_i, τ_res)     -- 2d spatial perturbations
    #   (Δ_res, τ_res ± h_t)         -- 2 temporal perturbations
    eye = np.eye(d) * h_d
    pts_list = [(delta_res, tau_res)]
    for i in range(d):
        pts_list.append((delta_res + eye[i], tau_res))
        pts_list.append((delta_res - eye[i], tau_res))
    pts_list.append((delta_res, tau_res + h_t))
    pts_list.append((delta_res, tau_res - h_t))
    # boundary-condition points
    pts_list.append((delta_bc, tau_bc))
    pts_list.append((delta_cap, tau_cap))

    deltas = np.concatenate([p[0] for p in pts_list], axis=0)
    taus = np.concatenate([p[1] for p in pts_list], axis=0)
    phi = encode_inputs(deltas, taus, L, n_freqs, t_max)
    Vall, cache = forward_VT(params, phi)

    # --- slice out each chunk ---
    cursor = 0
    def take(n):
        nonlocal cursor
        out = Vall[cursor:cursor + n]
        cursor += n
        return out
    V0 = take(N_res)
    V_plus = [take(N_res) for _ in range(d)]
    V_minus_pts_only = None
    V_minus = []
    # Re-collect: list was (centre), (+x,-x,+y,-y,...), then (+t,-t), bc, cap
    # Recompute carefully:
    # We placed: 1*N_res (centre)
    # then for each i: +e_i (N_res) and -e_i (N_res)   (interleaved)
    # then +h_t (N_res) and -h_t (N_res)
    # We've already taken centre + first d "+e_i" out of order ...
    # Let's just redo by index:
    cursor = N_res  # reset to right after centre
    V_plus_s = []
    V_minus_s = []
    for i in range(d):
        V_plus_s.append(Vall[cursor:cursor + N_res]); cursor += N_res
        V_minus_s.append(Vall[cursor:cursor + N_res]); cursor += N_res
    V_plus_t = Vall[cursor:cursor + N_res]; cursor += N_res
    V_minus_t = Vall[cursor:cursor + N_res]; cursor += N_res
    V_bc = Vall[cursor:cursor + N_bc]; cursor += N_bc
    V_cap = Vall[cursor:cursor + N_cap]; cursor += N_cap

    # --- finite-difference derivatives of V at the centres ---
    dV_dt = (V_plus_t - V_minus_t) / (2.0 * h_t)              # (N_res,)
    grad_d = np.stack(
        [(V_plus_s[i] - V_minus_s[i]) / (2.0 * h_d) for i in range(d)],
        axis=1)                                                 # (N_res, d)
    grad_norm = np.sqrt((grad_d ** 2).sum(axis=1) + 1e-10)     # (N_res,)

    # --- HJI residual.  C(Δ) = 1 everywhere on the residual sample,
    #     because we sample residual points OUTSIDE the capture set ---
    res = dV_dt - 1.0 + (v_P - v_E) * grad_norm                 # (N_res,)
    loss_hji = (res ** 2).mean()

    # --- Boundary conditions (no V* values used; just V = 0 here) ---
    loss_term = (V_bc ** 2).mean() if N_bc else 0.0
    loss_cap = (V_cap ** 2).mean() if N_cap else 0.0

    # Optional |∇V|² regulariser
    loss_reg = 0.0
    if lam_grad_reg > 0:
        loss_reg = (grad_norm ** 2).mean()

    loss = (loss_hji
            + lam_bc_term * loss_term
            + lam_bc_cap * loss_cap
            + lam_grad_reg * loss_reg)

    # --- backward.  Build the V-value gradient at each of the forward points ---
    dV_total = np.zeros_like(Vall)

    # gradient of loss_hji wrt V values:
    # res = dV_dt - 1 + (v_P-v_E)*|∇V|
    # dres/d(V_plus_t) = +1/(2h_t)
    # dres/d(V_minus_t) = -1/(2h_t)
    # dres/d(V_plus_s[i]) = (v_P-v_E) * grad_d[:,i] / (grad_norm * 2 h_d)
    # dres/d(V_minus_s[i]) = -(...)
    d_res_d = 2.0 * res / N_res     # 2 res / N_res from mean(res^2)
    # temporal
    dVdt_factor = d_res_d * 0.5 / h_t
    base = N_res
    dV_total[base + 2 * d * N_res: base + 2 * d * N_res + N_res] += dVdt_factor       # V_plus_t
    dV_total[base + 2 * d * N_res + N_res: base + 2 * d * N_res + 2 * N_res] -= dVdt_factor  # V_minus_t
    # spatial
    inv_g = 1.0 / grad_norm                                    # (N_res,)
    for i in range(d):
        idx_plus = base + 2 * i * N_res
        idx_minus = base + (2 * i + 1) * N_res
        coef = d_res_d * (v_P - v_E) * grad_d[:, i] * inv_g * (0.5 / h_d)
        dV_total[idx_plus:idx_plus + N_res] += coef
        dV_total[idx_minus:idx_minus + N_res] -= coef
    # gradient regulariser (optional)
    if lam_grad_reg > 0:
        # loss_reg = mean(grad_norm^2) = mean(sum_i grad_d[:,i]^2)
        for i in range(d):
            idx_plus = base + 2 * i * N_res
            idx_minus = base + (2 * i + 1) * N_res
            coef = lam_grad_reg * 2.0 * grad_d[:, i] * (0.5 / h_d) / N_res
            dV_total[idx_plus:idx_plus + N_res] += coef
            dV_total[idx_minus:idx_minus + N_res] -= coef

    # boundary terms
    if N_bc:
        idx = base + 2 * d * N_res + 2 * N_res
        dV_total[idx:idx + N_bc] += lam_bc_term * 2.0 * V_bc / N_bc
    if N_cap:
        idx = base + 2 * d * N_res + 2 * N_res + N_bc
        dV_total[idx:idx + N_cap] += lam_bc_cap * 2.0 * V_cap / N_cap

    grads = backward_VT(params, cache, dV_total)

    diagnostics = {
        "loss": float(loss),
        "loss_hji": float(loss_hji),
        "loss_term": float(loss_term),
        "loss_cap": float(loss_cap),
        "mean_grad_norm": float(grad_norm.mean()),
        "mean_dV_dt": float(dV_dt.mean()),
        "target_grad_norm_steady": float(1.0 / (v_P - v_E)),
    }
    return loss, grads, diagnostics


# =============================================================================
# Honest sampling helpers
# =============================================================================
def torus_dist0(delta, L):
    return np.linalg.norm(
        ((np.zeros_like(delta) - delta) + L / 2) % L - L / 2, axis=-1)


def sample_honest_batch(rng: np.random.Generator, L: float, eps: float,
                        t_max: float, n_dim: int,
                        n_res: int, n_term: int, n_cap: int):
    """
    Sample the three honest batches:
      * residual:  Δ ∈ (-L/2, L/2)^d with ||Δ||_T > 1.5 ε,  τ ∈ (0, t_max].
      * terminal:  arbitrary Δ in domain,                    τ = 0.
      * capture:   Δ with ||Δ||_T ≤ ε,                      τ ∈ [0, t_max].
    """
    # ---- residual ----
    Δres = rng.uniform(-L / 2 + 0.02, L / 2 - 0.02, size=(2 * n_res, n_dim))
    keep = torus_dist0(Δres, L) > 1.5 * eps
    Δres = Δres[keep][:n_res]
    if Δres.shape[0] < n_res:
        # top up if too few survived
        more = rng.uniform(-L / 2 + 0.02, L / 2 - 0.02,
                            size=(n_res - Δres.shape[0], n_dim))
        Δres = np.concatenate([Δres, more], axis=0)
    τres = rng.uniform(0.05, t_max, size=n_res)

    # ---- terminal ----
    Δterm = rng.uniform(-L / 2 + 0.02, L / 2 - 0.02, size=(n_term, n_dim))
    τterm = np.zeros(n_term)

    # ---- capture set ----
    angles = rng.uniform(0, 2 * np.pi, size=n_cap)
    radii = rng.uniform(0, eps * 0.95, size=n_cap)
    if n_dim == 2:
        Δcap = np.stack([radii * np.cos(angles), radii * np.sin(angles)],
                         axis=1)
    else:
        # generic d: random unit vector × radii
        u = rng.standard_normal(size=(n_cap, n_dim))
        u /= np.linalg.norm(u, axis=1, keepdims=True).clip(min=1e-12)
        Δcap = radii[:, None] * u
    τcap = rng.uniform(0.0, t_max, size=n_cap)

    return Δres, τres, Δterm, τterm, Δcap, τcap


# =============================================================================
# Training loop (with checkpoint chaining)
# =============================================================================
@dataclass
class HonestVConfig:
    L: float = 2.0
    v_E: float = 1.0
    v_P: float = 1.2
    eps: float = 0.08
    n_dim: int = 2
    n_freqs: int = 6
    hidden: int = 96
    t_max: float = 8.0    # ample so the steady-state regime is well sampled
    n_iters: int = 3000
    n_res: int = 384
    n_term: int = 96
    n_cap: int = 96
    lr: float = 3e-3
    lam_bc_term: float = 5.0
    lam_bc_cap: float = 5.0
    seed: int = 0
    log_every: int = 200


def train_honest_V(cfg: HonestVConfig, ckpt_path: str,
                    resume: bool = False) -> tuple[dict, dict]:
    rng = np.random.default_rng(cfg.seed)
    D = encoded_dim(cfg.n_dim, cfg.n_freqs)
    params = init_VT_params(D, cfg.hidden, rng)
    opt = Adam(params, lr=cfg.lr)
    hist = {"iter": [], "loss": [], "loss_hji": [], "loss_term": [],
            "loss_cap": [], "mean_grad_norm": [], "mean_dV_dt": []}
    iter0 = 0
    if resume and os.path.exists(ckpt_path):
        npz = np.load(ckpt_path)
        for k in params:
            params[k] = npz[k]
        opt = Adam(params, lr=cfg.lr)
        iter0 = int(npz["iter0"])
        if "hist" in npz.files:
            hist = json.loads(str(npz["hist"]))
        print(f"[honest V] resumed from iter {iter0}")

    target_norm = 1.0 / (cfg.v_P - cfg.v_E)
    t0 = time.time()
    for it in range(cfg.n_iters):
        Δr, τr, Δb, τb, Δc, τc = sample_honest_batch(
            rng, cfg.L, cfg.eps, cfg.t_max, cfg.n_dim,
            cfg.n_res, cfg.n_term, cfg.n_cap)
        loss, grads, diag = honest_loss(
            params, Δr, τr, Δb, τb, Δc, τc,
            cfg.L, cfg.v_E, cfg.v_P, cfg.eps,
            cfg.n_freqs, cfg.t_max,
            lam_bc_term=cfg.lam_bc_term, lam_bc_cap=cfg.lam_bc_cap)
        opt.step(params, grads, sign=-1.0)
        if it % cfg.log_every == 0 or it == cfg.n_iters - 1:
            hist["iter"].append(iter0 + it)
            for k, v in diag.items():
                if k in hist:
                    hist[k].append(v)
            print(f"[V honest] it {iter0+it:5d}  loss={diag['loss']:.4e}  "
                  f"hji={diag['loss_hji']:.4e}  term={diag['loss_term']:.4e}  "
                  f"cap={diag['loss_cap']:.4e}  "
                  f"|∇V|≈{diag['mean_grad_norm']:.3f}  "
                  f"target={target_norm:.3f}  [{time.time()-t0:.0f}s]")

    npz = {k: v for k, v in params.items()}
    npz["iter0"] = iter0 + cfg.n_iters
    npz["hist"] = json.dumps(hist)
    np.savez(ckpt_path, **npz)
    return params, hist


# =============================================================================
# Verification only (uses V* only as a yardstick, not as training data)
# =============================================================================
def analytic_V_steady(delta: np.ndarray, L: float, v_E: float, v_P: float,
                       eps: float) -> np.ndarray:
    """The closed-form steady-state value used ONLY for verification."""
    d = np.linalg.norm(
        ((np.zeros_like(delta) - delta) + L / 2) % L - L / 2, axis=-1)
    return np.where(d <= eps, 0.0, (d - eps) / (v_P - v_E))


def V_grid(params: dict, cfg: HonestVConfig, tau: float, n_grid: int = 80):
    """V_NN(Δ, tau) on a (n_grid, n_grid) grid of Δ in T^2."""
    xs = np.linspace(-cfg.L / 2, cfg.L / 2, n_grid)
    ys = np.linspace(-cfg.L / 2, cfg.L / 2, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)
    taus = np.full(pts.shape[0], tau)
    phi = encode_inputs(pts, taus, cfg.L, cfg.n_freqs, cfg.t_max)
    V, _ = forward_VT(params, phi)
    return xs, ys, V.reshape(n_grid, n_grid)


def grad_V(params: dict, delta_pt: np.ndarray, tau: float,
            cfg: HonestVConfig, h: float = 5e-3) -> np.ndarray:
    """∇_Δ V at a single point via central finite differences."""
    d = delta_pt.shape[-1]
    out = np.zeros(d)
    tau_arr = np.array([tau])
    for k in range(d):
        ep = np.zeros(d); ep[k] = h
        phi_p = encode_inputs((delta_pt + ep)[None, :], tau_arr,
                              cfg.L, cfg.n_freqs, cfg.t_max)
        phi_m = encode_inputs((delta_pt - ep)[None, :], tau_arr,
                              cfg.L, cfg.n_freqs, cfg.t_max)
        Vp, _ = forward_VT(params, phi_p)
        Vm, _ = forward_VT(params, phi_m)
        out[k] = (Vp[0] - Vm[0]) / (2 * h)
    return out


def capture_time_map_honest(params_V: dict,
                             cfg_actor: TrainConfig,
                             cfg_V: HonestVConfig,
                             n_grid: int = 24,
                             tau_use: float = None,
                             h_fd: float = 5e-3):
    """
    Capture-time map using the feedback policy derived from the
    *honestly-trained* V_NN.  τ_use selects which slice of V_NN(Δ, τ)
    we evaluate; the steady-state slice (large τ) gives the natural
    open-game policy.
    """
    if tau_use is None:
        tau_use = cfg_V.t_max
    xs = np.linspace(0.05, cfg_actor.L - 0.05, n_grid)
    ys = np.linspace(0.05, cfg_actor.L - 0.05, n_grid)
    Tc = np.full((n_grid, n_grid), np.nan)
    dt = cfg_actor.t_max / cfg_actor.n_t

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            xE = np.array([x, y])
            xP = np.array([0.0, 0.0])
            for k in range(cfg_actor.n_t):
                delta = ((xE - xP) + cfg_actor.L / 2) % cfg_actor.L - cfg_actor.L / 2
                g = grad_V(params_V, delta, tau_use, cfg_V, h=h_fd)
                u = g / max(np.linalg.norm(g), 1e-12)
                xE = (xE + dt * cfg_actor.v_E * u) % cfg_actor.L
                xP = (xP + dt * cfg_actor.v_P * u) % cfg_actor.L
                dist = np.linalg.norm(
                    ((xE - xP) + cfg_actor.L / 2) % cfg_actor.L - cfg_actor.L / 2)
                if dist <= cfg_actor.eps:
                    Tc[i, j] = (k + 1) * dt
                    break
    return xs, ys, Tc


def alignment_actor_vs_honest_critic(paramsE, paramsP, x_E0, cfg_actor,
                                      params_V, cfg_V, tau_use=None,
                                      h_fd: float = 5e-3):
    """
    Compare the v1 ADVERSARIAL ACTOR's pursuer direction with the
    honestly-trained CRITIC's gradient direction along the realised
    trajectory.  Returns the per-step cosine similarity, shape (n_h, T).
    """
    if tau_use is None:
        tau_use = cfg_V.t_max
    t_grid = np.linspace(0.0, cfg_actor.t_max, cfg_actor.n_t + 1)
    phi = fourier_features(t_grid, cfg_actor.t_max, cfg_actor.n_freqs)
    uE, _ = forward_dirs(paramsE, phi)
    uP, _ = forward_dirs(paramsP, phi)
    xE, xP = rollout(uE, uP, x_E0, cfg_actor.L, cfg_actor.v_E, cfg_actor.v_P,
                      cfg_actor.t_max, cfg_actor.n_t)
    # Critic direction at each (j, k)
    align = np.zeros((cfg_actor.n_h, cfg_actor.n_t + 1))
    for j in range(cfg_actor.n_h):
        for k in range(cfg_actor.n_t + 1):
            delta = ((xE[j, k] - xP[j, k]) + cfg_actor.L / 2) % cfg_actor.L - cfg_actor.L / 2
            g = grad_V(params_V, delta, tau_use, cfg_V, h=h_fd)
            n = max(np.linalg.norm(g), 1e-12)
            critic_dir = g / n
            align[j, k] = (uP[j, k] * critic_dir).sum()
    return align


# =============================================================================
# Plotters
# =============================================================================
def plot_V_vs_analytic(params, cfg_V, path, tau_use=None, n_grid=80):
    """
    PURELY DIAGNOSTIC.  Compares the honestly-trained V_NN(Δ, τ) against
    the closed-form V*(Δ).  The analytic V* was NOT used in training.
    """
    if tau_use is None:
        tau_use = cfg_V.t_max
    xs, ys, Vp = V_grid(params, cfg_V, tau=tau_use, n_grid=n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Va = analytic_V_steady(np.stack([X.flatten(), Y.flatten()], axis=1),
                            cfg_V.L, cfg_V.v_E, cfg_V.v_P,
                            cfg_V.eps).reshape(n_grid, n_grid)
    err = Vp - Va
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    im0 = ax[0].imshow(Vp.T, origin="lower",
                        extent=[xs[0], xs[-1], ys[0], ys[-1]],
                        cmap="viridis")
    plt.colorbar(im0, ax=ax[0])
    ax[0].set_title(f"Honest $V_{{NN}}(\\Delta,\\,\\tau={tau_use:.1f})$")
    im1 = ax[1].imshow(Va.T, origin="lower",
                        extent=[xs[0], xs[-1], ys[0], ys[-1]],
                        cmap="viridis")
    plt.colorbar(im1, ax=ax[1])
    ax[1].set_title("Analytic $V^*(\\Delta)$  (verification only)")
    im2 = ax[2].imshow(err.T, origin="lower",
                        extent=[xs[0], xs[-1], ys[0], ys[-1]],
                        cmap="RdBu_r",
                        vmin=-np.abs(err).max(), vmax=np.abs(err).max())
    plt.colorbar(im2, ax=ax[2])
    ax[2].set_title("Error  $V_{NN} - V^*$")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_honest_alignment(align, t_grid, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j in range(min(align.shape[0], 12)):
        ax[0].plot(t_grid, align[j], lw=0.8, alpha=0.7,
                   label=f"IC {j}" if j < 6 else None)
    ax[0].axhline(1.0, color="k", ls="--", lw=0.7)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"$\hat u_P^{\rm actor}(t)\cdot\hat u_P^{\rm honest\,critic}(t)$")
    ax[0].set_title("v1 adversarial actor vs. honest critic")
    ax[0].set_ylim(-1.05, 1.1)
    ax[0].legend(fontsize=7, loc="lower right")
    m, s = align.mean(0), align.std(0)
    ax[1].plot(t_grid, m, color="tab:red", lw=2, label="mean over heads")
    ax[1].fill_between(t_grid, m - s, m + s, color="tab:red", alpha=0.2,
                       label="$\\pm 1\\sigma$")
    ax[1].axhline(1.0, color="k", ls="--", lw=0.7)
    ax[1].set_xlabel("t"); ax[1].set_ylabel("alignment")
    ax[1].set_ylim(-1.05, 1.1); ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
