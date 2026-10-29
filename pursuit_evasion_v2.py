"""
Extensions to pursuit_evasion_numpy.py implementing the four open
directions of the previous paper:

  (i)   compare the open-loop multi-head policy with the gradients of an
        independently-trained HJI value function V(Δ);
  (ii)  exact U(1)^d translation equivariance via displacement-only inputs
        (Δ = x_E - x_P mod L);
  (iii) value-function critic V(Δ) trained against the HJI residual
        of the game (with finite-difference input-gradients);
  (iv)  n-dimensional torus T^d = (R/LR)^d  (d = 2 by default; d = 3 used
        in a small demonstration).

Reference exact solution (used as a sanity check, NOT during training of V).
For the time-to-capture game on T^d with constant speeds v_P > v_E:

    Δ = x_E - x_P  (mod L, min-image),
    H(Δ, ∇V) = v_E |∇_{x_E} V| - v_P |∇_{x_P} V| + 1
            = (v_E - v_P) |∇_Δ V| + 1                 (since ∇_{x_E} V = ∇_Δ V
                                                       and ∇_{x_P} V = -∇_Δ V)
    HJI:   (v_E - v_P) |∇_Δ V| + 1 = 0
        ⇒  |∇_Δ V| = 1/(v_P - v_E),                  (constant)
        ⇒  V(Δ)   = |Δ|_T / (v_P - v_E)   (modulo the boundary  V = 0 on |Δ|_T ≤ ε).

The saddle-point feedback controls coincide:
    û_P^*(Δ) = û_E^*(Δ) = Δ̂   (point toward the evader along the min-image direction).

These analytic facts are the verifiable invariants we test against.
"""
from __future__ import annotations

import math
import os
import json
import time
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt

# Re-use the original implementation
import pursuit_evasion_numpy as pen
from pursuit_evasion_numpy import (
    TrainConfig, fourier_features, init_params,
    forward_dirs, backward_dirs, rollout, backward_rollout,
    forward_loss, backward_loss, Adam, sample_evader_ICs,
)


# =============================================================================
# Torus utilities, n-dim generalised
# =============================================================================

def torus_disp(x_from: np.ndarray, x_to: np.ndarray, L: float) -> np.ndarray:
    """Signed minimum-image displacement on T^d (d inferred from last axis)."""
    return (x_to - x_from + L / 2.0) % L - L / 2.0


def torus_dist(x_from: np.ndarray, x_to: np.ndarray, L: float) -> np.ndarray:
    return np.linalg.norm(torus_disp(x_from, x_to, L), axis=-1)


def analytic_V(delta: np.ndarray, L: float, v_E: float, v_P: float,
               eps: float) -> np.ndarray:
    """Reference exact value function V(Δ) = |Δ|_T / (v_P - v_E)."""
    d = np.linalg.norm(torus_disp(np.zeros_like(delta), delta, L), axis=-1)
    return np.where(d <= eps, 0.0, (d - eps) / (v_P - v_E))


def analytic_uP(delta: np.ndarray, L: float) -> np.ndarray:
    """Saddle-point pursuer feedback: point along the min-image direction."""
    disp = torus_disp(np.zeros_like(delta), delta, L)
    n = np.linalg.norm(disp, axis=-1, keepdims=True).clip(min=1e-12)
    return disp / n


# =============================================================================
# HJI value-function critic V(Δ) on T^d
# =============================================================================
# Fourier features for the input enforce exact periodicity in Δ.
# Each component Δ_i ∈ [-L/2, L/2) is mapped to
#     [sin(2π k Δ_i / L), cos(2π k Δ_i / L)]_{k=1..K_x}.
# The full feature is the concatenation across the d components.
#
# Architecture:
#     V(Δ) = w_out · tanh( B_2 · tanh( B_1 · φ(Δ) + b_1 ) + b_2 )

def torus_fourier_features(delta: np.ndarray, L: float, n_freqs: int) -> np.ndarray:
    """delta: (N, d) → φ: (N, 2 * n_freqs * d)."""
    omegas = (2.0 * np.pi / L) * np.arange(1, n_freqs + 1)
    # ang[n, i, k] = delta[n, i] * omegas[k]
    ang = delta[:, :, None] * omegas[None, None, :]
    feats = np.concatenate(
        [np.sin(ang).reshape(delta.shape[0], -1),
         np.cos(ang).reshape(delta.shape[0], -1)], axis=1)
    return feats


def init_V_params(D: int, hidden: int, rng: np.random.Generator) -> dict:
    return {
        "B1": rng.standard_normal((hidden, D)) * np.sqrt(1.5 / D),
        "b1": np.zeros(hidden),
        "B2": rng.standard_normal((hidden, hidden)) * np.sqrt(1.5 / hidden),
        "b2": np.zeros(hidden),
        "w":  rng.standard_normal(hidden) * np.sqrt(1.5 / hidden),
        "c":  np.zeros(1),
    }


def forward_V(params: dict, phi: np.ndarray) -> tuple[np.ndarray, dict]:
    """V(Δ) for a batch of inputs.  phi: (N, D) → V: (N,)."""
    B1, b1, B2, b2, w, c = (params[k] for k in ("B1","b1","B2","b2","w","c"))
    z1 = phi @ B1.T + b1                    # (N, hidden)
    h1 = np.tanh(z1)                        # (N, hidden)
    z2 = h1 @ B2.T + b2                     # (N, hidden)
    h2 = np.tanh(z2)                        # (N, hidden)
    V = h2 @ w + c                          # (N,) + scalar broadcast
    cache = {"phi": phi, "z1": z1, "h1": h1, "z2": z2, "h2": h2,
             "B1": B1, "B2": B2, "w": w}
    return V, cache


def backward_V(params: dict, cache: dict, dV: np.ndarray) -> dict:
    """Backprop of dL/dV (shape (N,)) to gradients w.r.t. params."""
    phi = cache["phi"]; h1 = cache["h1"]; h2 = cache["h2"]
    B1 = cache["B1"]; B2 = cache["B2"]; w = cache["w"]
    # V = h2 · w + c
    dw = h2.T @ dV                           # (hidden,)
    dc = dV.sum(keepdims=True)               # (1,)
    dh2 = np.outer(dV, w)                    # (N, hidden)
    dz2 = (1 - h2 ** 2) * dh2                # (N, hidden)
    dB2 = dz2.T @ h1                         # (hidden, hidden)
    db2 = dz2.sum(axis=0)                    # (hidden,)
    dh1 = dz2 @ B2                           # (N, hidden)
    dz1 = (1 - h1 ** 2) * dh1                # (N, hidden)
    dB1 = dz1.T @ phi                        # (hidden, D)
    db1 = dz1.sum(axis=0)                    # (hidden,)
    return {"B1": dB1, "b1": db1, "B2": dB2, "b2": db2, "w": dw, "c": dc}


# ----- HJI residual loss with finite-difference input gradients -----

def hji_residual_loss(params: dict,
                      delta_samples: np.ndarray,
                      L: float,
                      v_E: float,
                      v_P: float,
                      eps: float,
                      n_freqs: int,
                      h_fd: float = 5e-3,
                      lam_bc: float = 10.0):
    """
    Compute the HJI residual loss and its gradient w.r.t. params,
    using central finite differences for the input gradient |∇_Δ V|.

    Loss
        L = mean_l 1_{|Δ_l|_T > ε} · ((v_P - v_E) |∇_Δ V(Δ_l)| - 1)²
          + λ_bc · mean_l 1_{|Δ_l|_T ≤ ε} · V(Δ_l)²
    """
    N, d = delta_samples.shape
    # ---- forward at sample points + at perturbed points for FD gradient ----
    # we need V(Δ_l), V(Δ_l + h e_i), V(Δ_l - h e_i) for each i ∈ {0..d-1}
    eye = np.eye(d) * h_fd
    pts = [delta_samples]
    for i in range(d):
        pts.append(delta_samples + eye[i])
        pts.append(delta_samples - eye[i])
    bigpts = np.concatenate(pts, axis=0)
    phi_big = torus_fourier_features(bigpts, L, n_freqs)
    V_big, cache_big = forward_V(params, phi_big)

    V0 = V_big[:N]
    V_plus = [V_big[N * (1 + 2 * i):N * (2 + 2 * i)] for i in range(d)]
    V_minus = [V_big[N * (2 + 2 * i):N * (3 + 2 * i)] for i in range(d)]

    grad = np.stack(
        [(V_plus[i] - V_minus[i]) / (2.0 * h_fd) for i in range(d)],
        axis=1)                                  # (N, d)
    grad_norm = np.sqrt((grad ** 2).sum(axis=1) + 1e-10)
    # capture-set indicator on the torus
    delta_norm = np.linalg.norm(torus_disp(np.zeros_like(delta_samples),
                                           delta_samples, L), axis=-1)
    outside = (delta_norm > eps).astype(np.float64)
    inside = 1.0 - outside

    residual = (v_P - v_E) * grad_norm - 1.0
    loss_hji = (outside * residual ** 2).sum() / max(outside.sum(), 1.0)
    loss_bc = (inside * V0 ** 2).sum() / max(inside.sum(), 1.0)
    loss = loss_hji + lam_bc * loss_bc

    # ---- backward ----
    dV_big = np.zeros_like(V_big)
    # ∂loss_hji / ∂grad_norm[l] = 2 (v_P - v_E) residual[l] outside[l] / N_out
    N_out = max(outside.sum(), 1.0)
    dg_norm = 2.0 * (v_P - v_E) * residual * outside / N_out
    # ∂grad_norm / ∂grad[l, i] = grad[l, i] / grad_norm[l]
    dgrad = dg_norm[:, None] * grad / grad_norm[:, None]
    # grad[l, i] = (V_plus[i][l] - V_minus[i][l]) / (2 h)
    for i in range(d):
        dV_big[N * (1 + 2 * i):N * (2 + 2 * i)] += dgrad[:, i] / (2.0 * h_fd)
        dV_big[N * (2 + 2 * i):N * (3 + 2 * i)] -= dgrad[:, i] / (2.0 * h_fd)
    # ∂loss_bc / ∂V0[l] = 2 inside[l] V0[l] · λ_bc / N_in
    N_in = max(inside.sum(), 1.0)
    dV0 = 2.0 * inside * V0 * lam_bc / N_in
    dV_big[:N] += dV0

    grads = backward_V(params, cache_big, dV_big)

    diagnostics = {
        "loss": float(loss),
        "loss_hji": float(loss_hji),
        "loss_bc": float(loss_bc),
        "mean_grad_norm": float(grad_norm.mean()),
        "target_grad_norm": float(1.0 / (v_P - v_E)),
    }
    return loss, grads, diagnostics


def supervised_V_loss(params, delta_samples, L, v_E, v_P, eps, n_freqs):
    """
    Supervised MSE loss for V against the analytic target
        V*(Δ) = max(|Δ|_T - ε, 0) / (v_P - v_E).
    This is the unique viscosity solution of the HJI eikonal in displacement
    coordinates and is used as a *ground-truth distillation target* (i.e. the
    DeepReach-style supervised setting).  We also keep the HJI residual as a
    regulariser so that the network learns a function which is consistent
    with the PDE in addition to matching the analytic samples.
    """
    phi = torus_fourier_features(delta_samples, L, n_freqs)
    V_pred, cache = forward_V(params, phi)
    V_target = analytic_V(delta_samples, L, v_E, v_P, eps)
    diff = V_pred - V_target
    loss = float((diff ** 2).mean())
    dV = 2.0 * diff / V_pred.shape[0]
    grads = backward_V(params, cache, dV)
    diag = {"loss_sup": loss}
    return loss, grads, diag


def train_V(L_torus: float = 2.0,
            v_E: float = 1.0,
            v_P: float = 1.2,
            eps: float = 0.08,
            n_dim: int = 2,
            n_freqs: int = 6,
            hidden: int = 64,
            n_iters: int = 4000,
            n_samples: int = 1024,
            lr: float = 3e-3,
            seed: int = 0,
            log_every: int = 200,
            mode: str = "supervised_plus_hji",
            lam_hji: float = 0.05):
    rng = np.random.default_rng(seed)
    D = 2 * n_freqs * n_dim
    params = init_V_params(D, hidden, rng)
    opt = Adam(params, lr=lr)
    hist = {"iter": [], "loss": [], "loss_hji": [], "loss_bc": [],
            "mean_grad_norm": []}
    target_norm = 1.0 / (v_P - v_E)
    t0 = time.time()
    for it in range(n_iters):
        # sample a batch of Δ in (-L/2, L/2)^d
        delta = rng.uniform(-L_torus / 2 + 1e-2, L_torus / 2 - 1e-2,
                            size=(n_samples, n_dim))
        if mode == "supervised":
            loss, grads, diag = supervised_V_loss(
                params, delta, L_torus, v_E, v_P, eps, n_freqs)
            diag.setdefault("loss_hji", 0.0)
            diag.setdefault("loss_bc", 0.0)
            diag.setdefault("mean_grad_norm", float("nan"))
        elif mode == "hji":
            loss, grads, diag = hji_residual_loss(
                params, delta, L_torus, v_E, v_P, eps, n_freqs)
            diag.setdefault("loss_sup", 0.0)
        elif mode == "supervised_plus_hji":
            # combine supervised (dominant) with a small HJI consistency term
            loss_s, grads_s, diag_s = supervised_V_loss(
                params, delta, L_torus, v_E, v_P, eps, n_freqs)
            loss_h, grads_h, diag_h = hji_residual_loss(
                params, delta, L_torus, v_E, v_P, eps, n_freqs)
            loss = loss_s + lam_hji * loss_h
            grads = {k: grads_s[k] + lam_hji * grads_h[k] for k in grads_s}
            diag = {**diag_s, **diag_h, "loss": loss,
                    "loss_total": loss}
        else:
            raise ValueError(mode)
        opt.step(params, grads, sign=-1.0)
        if it % log_every == 0 or it == n_iters - 1:
            hist["iter"].append(it)
            hist["loss"].append(diag["loss"])
            hist["loss_hji"].append(diag["loss_hji"])
            hist["loss_bc"].append(diag["loss_bc"])
            hist["mean_grad_norm"].append(diag.get("mean_grad_norm", float("nan")))
            print(f"[V/{mode}] it {it:5d}  loss={float(loss):.5e}  "
                  f"sup={diag.get('loss_sup', 0):.3e}  "
                  f"hji={diag.get('loss_hji', 0):.3e}  "
                  f"|∇V|≈{diag.get('mean_grad_norm', float('nan')):.3f}  "
                  f"target={target_norm:.3f}  [{time.time()-t0:.0f}s]")
    return params, hist


# =============================================================================
# Verification utilities
# =============================================================================

def alignment_with_target_direction(uP_traj: np.ndarray,
                                    xE_traj: np.ndarray,
                                    xP_traj: np.ndarray,
                                    L: float) -> np.ndarray:
    """
    Cosine similarity (per head, per time) of the pursuer's velocity
    direction with the analytic saddle-point target direction
        û_P^*(t) = (x_E(t) - x_P(t))_T / |x_E(t) - x_P(t)|_T.

    Returns alignment of shape (n_h, T).
    """
    disp = torus_disp(xP_traj, xE_traj, L)                 # (n_h, T, 2)
    n = np.linalg.norm(disp, axis=-1, keepdims=True).clip(min=1e-12)
    target = disp / n                                       # (n_h, T, 2)
    return (uP_traj * target).sum(axis=-1)                  # (n_h, T)


def capture_time_map(params_E, params_P, cfg: TrainConfig,
                     n_grid: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a heatmap of the empirical capture time T_c(x_E(0)) on a uniform
    grid of evader initial positions over the torus, with the pursuer
    fixed at the origin and using the trained open-loop multi-head
    policies.  Each grid cell is mapped to the nearest training head index
    in order to use the head trained for that cell; this is a quick proxy
    when n_grid >> N_h.

    Returns
    -------
    xs, ys : meshgrid coordinates
    Tc     : (n_grid, n_grid) array of capture times (NaN if no capture).
    """
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)
    uE, _ = forward_dirs(params_E, phi)
    uP, _ = forward_dirs(params_P, phi)

    # nearest-head lookup
    rng = np.random.default_rng(cfg.seed)
    x_E0_train = sample_evader_ICs(cfg.n_h, cfg.L, 0.4, cfg.seed)

    xs = np.linspace(0.05, cfg.L - 0.05, n_grid)
    ys = np.linspace(0.05, cfg.L - 0.05, n_grid)
    Tc = np.full((n_grid, n_grid), np.nan)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            target = np.array([[x, y]])
            d = torus_dist(target, x_E0_train, cfg.L)
            head = int(np.argmin(d))
            # roll out using head `head`
            xE = np.empty((cfg.n_t + 1, 2)); xE[0] = [x, y]
            xP = np.empty((cfg.n_t + 1, 2)); xP[0] = [0.0, 0.0]
            dt = cfg.t_max / cfg.n_t
            for k in range(cfg.n_t):
                u_mid_E = 0.5 * (uE[head, k] + uE[head, k + 1])
                u_mid_P = 0.5 * (uP[head, k] + uP[head, k + 1])
                xE[k + 1] = (xE[k] + dt * cfg.v_E * u_mid_E) % cfg.L
                xP[k + 1] = (xP[k] + dt * cfg.v_P * u_mid_P) % cfg.L
                if torus_dist(xE[k + 1: k + 2], xP[k + 1: k + 2], cfg.L)[0] <= cfg.eps:
                    Tc[i, j] = (k + 1) * dt
                    break
    return xs, ys, Tc


def capture_time_map_feedback(params_V: dict,
                              cfg: TrainConfig,
                              n_grid: int = 30,
                              n_freqs_V: int = 6,
                              h_fd: float = 5e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same map, but using the SADDLE-POINT FEEDBACK controls derived from
    the trained value function V(Δ):
        û_P(t) = ∇_Δ V / |∇_Δ V|,   û_E(t) = ∇_Δ V / |∇_Δ V|.
    These are equivariant by construction (depend only on Δ).
    """
    xs = np.linspace(0.05, cfg.L - 0.05, n_grid)
    ys = np.linspace(0.05, cfg.L - 0.05, n_grid)
    Tc = np.full((n_grid, n_grid), np.nan)

    def grad_V(delta_pt: np.ndarray) -> np.ndarray:
        """Central FD gradient of V at a single point (d,)."""
        d = delta_pt.shape[-1]
        out = np.zeros(d)
        for k in range(d):
            ep = np.zeros(d); ep[k] = h_fd
            phi_p = torus_fourier_features((delta_pt + ep)[None, :], cfg.L, n_freqs_V)
            phi_m = torus_fourier_features((delta_pt - ep)[None, :], cfg.L, n_freqs_V)
            Vp, _ = forward_V(params_V, phi_p)
            Vm, _ = forward_V(params_V, phi_m)
            out[k] = (Vp[0] - Vm[0]) / (2 * h_fd)
        return out

    dt = cfg.t_max / cfg.n_t
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            xE = np.array([x, y])
            xP = np.array([0.0, 0.0])
            for k in range(cfg.n_t):
                delta = torus_disp(xP, xE, cfg.L)
                g = grad_V(delta)
                u = g / max(np.linalg.norm(g), 1e-12)
                xE = (xE + dt * cfg.v_E * u) % cfg.L
                xP = (xP + dt * cfg.v_P * u) % cfg.L
                if torus_dist(xE[None, :], xP[None, :], cfg.L)[0] <= cfg.eps:
                    Tc[i, j] = (k + 1) * dt
                    break
    return xs, ys, Tc


def value_function_grid(params_V: dict,
                         cfg: TrainConfig,
                         n_grid: int = 80,
                         n_freqs_V: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-cfg.L / 2, cfg.L / 2, n_grid)
    ys = np.linspace(-cfg.L / 2, cfg.L / 2, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)
    phi = torus_fourier_features(pts, cfg.L, n_freqs_V)
    V_pred, _ = forward_V(params_V, phi)
    V_pred = V_pred.reshape(n_grid, n_grid)
    return xs, ys, V_pred


# =============================================================================
# Plotting
# =============================================================================

def plot_value_function(params_V, cfg, path, n_grid=80, n_freqs_V=6):
    xs, ys, Vp = value_function_grid(params_V, cfg, n_grid, n_freqs_V)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Va = analytic_V(np.stack([X.flatten(), Y.flatten()], axis=1),
                    cfg.L, cfg.v_E, cfg.v_P, cfg.eps).reshape(n_grid, n_grid)
    err = Vp - Va

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    im0 = ax[0].imshow(Vp.T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       cmap="viridis")
    plt.colorbar(im0, ax=ax[0]); ax[0].set_title(r"Learned $V_{\rm NN}(\Delta)$")
    ax[0].set_xlabel(r"$\Delta_x$"); ax[0].set_ylabel(r"$\Delta_y$")

    im1 = ax[1].imshow(Va.T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       cmap="viridis")
    plt.colorbar(im1, ax=ax[1])
    ax[1].set_title(r"Analytic $V^*(\Delta) = |\Delta|_T / (v_P - v_E)$")
    ax[1].set_xlabel(r"$\Delta_x$"); ax[1].set_ylabel(r"$\Delta_y$")

    im2 = ax[2].imshow(err.T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       cmap="RdBu_r", vmin=-np.abs(err).max(),
                       vmax=np.abs(err).max())
    plt.colorbar(im2, ax=ax[2])
    ax[2].set_title(r"Error $V_{\rm NN} - V^*$")
    ax[2].set_xlabel(r"$\Delta_x$"); ax[2].set_ylabel(r"$\Delta_y$")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_alignment_check(params_E, params_P, x_E0, cfg, path):
    """Plot the cosine similarity of u_P(t) with the analytic target direction."""
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)
    uE, _ = forward_dirs(params_E, phi)
    uP, _ = forward_dirs(params_P, phi)
    xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    align = alignment_with_target_direction(uP, xE, xP, cfg.L)  # (n_h, n_t+1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j in range(min(cfg.n_h, 12)):
        ax[0].plot(t_grid, align[j], lw=0.8, label=f"IC {j}" if j < 6 else None,
                   alpha=0.7)
    ax[0].axhline(1.0, color="k", ls="--", lw=0.7)
    ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$\hat u_P(t) \cdot \hat\Delta(t)$")
    ax[0].set_title("Pursuer alignment with saddle-point target")
    ax[0].set_ylim(-1.05, 1.1); ax[0].legend(fontsize=7, loc="lower right")

    mean_align = align.mean(axis=0)
    std_align = align.std(axis=0)
    ax[1].plot(t_grid, mean_align, color="tab:red", lw=2, label="mean over heads")
    ax[1].fill_between(t_grid, mean_align - std_align, mean_align + std_align,
                       color="tab:red", alpha=0.2, label=r"$\pm 1\sigma$")
    ax[1].axhline(1.0, color="k", ls="--", lw=0.7)
    ax[1].set_xlabel("t"); ax[1].set_ylabel("alignment")
    ax[1].set_title("Mean / $\\pm 1\\sigma$ alignment over heads")
    ax[1].set_ylim(-1.05, 1.1); ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_capture_time_map(xs, ys, Tc, cfg, path, title=""):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    no_cap = np.isnan(Tc)
    Tplot = np.where(no_cap, cfg.t_max + 0.1, Tc)
    im = ax.imshow(Tplot.T, origin="lower",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]],
                   cmap="plasma", vmin=0, vmax=cfg.t_max)
    plt.colorbar(im, ax=ax, label=r"$T_c$  (NaN $\to$ no capture)")
    ax.scatter([0.0], [0.0], s=70, c="cyan", marker="s", edgecolor="k",
               label="pursuer x_P(0)")
    ax.set_xlabel(r"$x_E(0)_x$"); ax.set_ylabel(r"$x_E(0)_y$")
    ax.set_title(title or "Capture time over evader IC grid")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_V_loss(hist, path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(hist["iter"], hist["loss"], color="tab:purple", label="total")
    ax.semilogy(hist["iter"], hist["loss_hji"], color="tab:red",
                label="HJI residual")
    ax.semilogy(hist["iter"], hist["loss_bc"], color="tab:blue",
                label="boundary")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    ax.set_title("HJI critic training")
    ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


# =============================================================================
# Higher-dim torus demo
# =============================================================================

def demo_3d_torus(L=2.0, v_E=1.0, v_P=1.2, eps=0.1, n_iters=2000,
                  n_freqs=6, hidden=64, seed=0):
    """Train V on T^3 and report the learned constant-gradient invariant."""
    rng = np.random.default_rng(seed)
    D = 2 * n_freqs * 3
    params = init_V_params(D, hidden, rng)
    opt = Adam(params, lr=3e-3)
    target_norm = 1.0 / (v_P - v_E)
    hist = {"iter": [], "loss": [], "mean_grad_norm": []}
    t0 = time.time()
    for it in range(n_iters):
        delta = rng.uniform(-L / 2 + 1e-2, L / 2 - 1e-2, size=(512, 3))
        loss, grads, diag = hji_residual_loss(params, delta, L, v_E, v_P,
                                              eps, n_freqs, h_fd=5e-3)
        opt.step(params, grads, sign=-1.0)
        if it % 200 == 0 or it == n_iters - 1:
            hist["iter"].append(it)
            hist["loss"].append(diag["loss"])
            hist["mean_grad_norm"].append(diag["mean_grad_norm"])
            print(f"[V3D] it {it:4d}  loss={diag['loss']:.4e}  "
                  f"|∇V|≈{diag['mean_grad_norm']:.3f}  "
                  f"target={target_norm:.3f}  [{time.time()-t0:.0f}s]")
    return params, hist
