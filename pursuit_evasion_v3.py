"""
v3 extensions:
  (1) Actor-critic distillation: the open-loop multi-head actor is pulled
      toward the analytic saddle-point feedback u*(Delta) = Delta/|Delta|_T
      along its own realised trajectory.  This is exactly the "critic-as-teacher"
      step proposed in the v2 conclusion.
  (2) Obstacle field: a soft Gaussian obstacle on the torus breaks the
      analytic ground truth so we can test the framework when no closed-form
      V* is available.
  (4) Scaling with N_h: scripted utility to train identical actors at
      multiple head counts and measure the mean alignment with the
      analytic feedback.

All gradient calculations are hand-coded and validated against finite
differences.
"""
from __future__ import annotations
import math, os, json, time
from dataclasses import dataclass, asdict
import numpy as np

import pursuit_evasion_numpy as pen
from pursuit_evasion_numpy import (
    TrainConfig, fourier_features, init_params,
    forward_dirs, backward_dirs, rollout, backward_rollout,
    forward_loss, backward_loss, Adam, sample_evader_ICs,
)
import pursuit_evasion_v2 as v2


# =============================================================================
# Distillation loss (analytic saddle-point teacher)
# =============================================================================
# Target along the trajectory:
#     û*(t,j) = Δ^j(t) / ||Δ^j(t)||_T,
# where Δ^j(t) is the *realised* displacement on the actor's own rollout.
# We use stop-gradient on the target (standard DAgger-style behaviour
# cloning) so the gradient flows only through the actor's direction
# outputs and into the rollout.
#
# L_dist_P = mean_{j, t} ||u_P^j(t) - target^j(t)||^2
# L_dist_E = mean_{j, t} ||u_E^j(t) - target^j(t)||^2

def distillation_loss(uE: np.ndarray, uP: np.ndarray,
                      xE: np.ndarray, xP: np.ndarray,
                      L: float, eps: float):
    """
    Returns (L_dist_E, L_dist_P, du_E_dist, du_P_dist) where du_*_dist are
    the gradients of the distillation loss with respect to u_E, u_P.

    Both u_E and u_P are pulled toward the analytic saddle-point target
    û*(Δ) = Δ / ||Δ||_T  (the saddle-point coincides on the optimal
    trajectory: both players head along the relative displacement axis).
    """
    disp = ((xE - xP) + L / 2.0) % L - L / 2.0          # (n_h, T, 2)
    d = np.sqrt((disp ** 2).sum(axis=-1) + 1e-12)        # (n_h, T)
    # mask out the capture set (no well-defined target there)
    mask = (d > eps).astype(np.float64)
    target = disp / d[..., None]                          # (n_h, T, 2)
    n_total = max(mask.sum(), 1.0)

    # P loss
    diffP = (uP - target) * mask[..., None]
    L_dist_P = float((diffP ** 2).sum() / n_total)
    duP = 2.0 * diffP / n_total

    # E loss
    diffE = (uE - target) * mask[..., None]
    L_dist_E = float((diffE ** 2).sum() / n_total)
    duE = 2.0 * diffE / n_total

    return L_dist_E, L_dist_P, duE, duP


# =============================================================================
# Combined-objective training: game loss + distillation
# =============================================================================
@dataclass
class DistillConfig:
    n_h: int = 16
    n_b: int = 16
    n_freqs: int = 12
    t_max: float = 4.0
    n_t: int = 64
    v_E: float = 1.0
    v_P: float = 1.2
    eps: float = 0.08
    tau: float = 5e-3
    L: float = 2.0
    lr_P: float = 1e-2
    lr_E: float = 3e-3
    no_capture_penalty: float = 2.0
    lam_d: float = 5.0
    lam_dist_P: float = 5.0      # how strongly the pursuer follows the critic
    lam_dist_E: float = 0.5      # the evader is less constrained
    n_P_steps: int = 3
    n_E_steps: int = 1
    pretrain_P_iters: int = 1500
    seed: int = 0


def train_distill(cfg: DistillConfig,
                  n_iters: int,
                  ckpt_path: str,
                  log_every: int = 200,
                  resume: bool = False):
    """
    Train the multi-head actor with the game loss augmented by a
    distillation term toward the analytic saddle-point feedback.
    Supports checkpoint chaining for arbitrarily long training.
    """
    rng = np.random.default_rng(cfg.seed)
    D = 2 * cfg.n_freqs + 1
    paramsE = init_params(cfg.n_h, cfg.n_b, D, rng)
    paramsP = init_params(cfg.n_h, cfg.n_b, D, rng)
    optE = Adam(paramsE, lr=cfg.lr_E)
    optP = Adam(paramsP, lr=cfg.lr_P)
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, 0.4, cfg.seed)
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)

    history = {"iter": [], "L_P": [], "L_E": [], "L_dist_P": [],
               "L_dist_E": [], "ET": [], "final_d": [], "no_cap": [],
               "align_P_mean": []}
    iter0 = 0
    pretrain_done = False
    if resume and os.path.exists(ckpt_path):
        npz = np.load(ckpt_path)
        paramsE = {"B": npz["BE"], "b_body": npz["bE"], "W": npz["WE"]}
        paramsP = {"B": npz["BP"], "b_body": npz["bP"], "W": npz["WP"]}
        x_E0 = npz["x_E0"]
        pretrain_done = True
        iter0 = int(npz["iter0"]) if "iter0" in npz.files else 0
        if "history" in npz.files:
            history = json.loads(str(npz["history"]))
        optE = Adam(paramsE, lr=cfg.lr_E)
        optP = Adam(paramsP, lr=cfg.lr_P)
        print(f"[distill] resumed at iter {iter0}")

    if not pretrain_done:
        # warm up the pursuer with vanilla game loss only (no distillation)
        for it in range(cfg.pretrain_P_iters):
            uE, cE = forward_dirs(paramsE, phi)
            uP, cP = forward_dirs(paramsP, phi)
            xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max,
                              cfg.n_t)
            LE, LP, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau,
                                      cfg.t_max, cfg.n_t, cfg.no_capture_penalty,
                                      lam_d=cfg.lam_d)
            dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty,
                                     cfg.t_max, is_pursuer=True)
            duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                        cfg.t_max, cfg.n_t)
            gP = backward_dirs(paramsP, cP, duP)
            optP.step(paramsP, gP, sign=-1.0)
            if it % 400 == 0:
                print(f"[distill warm-P] it {it:4d}  L_P={LP:.3f}  "
                      f"no_cap={float((cL['S'][:, -1] > 0.5).mean()):.2f}")

    def step(who):
        uE, cE = forward_dirs(paramsE, phi)
        uP, cP = forward_dirs(paramsP, phi)
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max,
                          cfg.n_t)
        LE, LP, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                  cfg.n_t, cfg.no_capture_penalty,
                                  lam_d=cfg.lam_d)
        Ld_E, Ld_P, duE_d, duP_d = distillation_loss(uE, uP, xE, xP, cfg.L,
                                                       cfg.eps)
        is_p = (who == "P")
        # gradient of game loss part
        dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty,
                                 cfg.t_max, is_pursuer=is_p)
        duE_g, duP_g = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                        cfg.t_max, cfg.n_t)
        # combine game + distillation gradients on u (these flow into actor params)
        if is_p:
            duP_total = duP_g + cfg.lam_dist_P * duP_d
            gP = backward_dirs(paramsP, cP, duP_total)
            optP.step(paramsP, gP, sign=-1.0)
        else:
            duE_total = duE_g + cfg.lam_dist_E * duE_d
            gE = backward_dirs(paramsE, cE, duE_total)
            optE.step(paramsE, gE, sign=-1.0)
        # diagnostics
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = ((xE - xP) + cfg.L / 2) % cfg.L - cfg.L / 2
            dn = np.sqrt((disp ** 2).sum(-1) + 1e-12)
            mask = dn > cfg.eps
            target = disp / dn[..., None]
            align = (uP * target).sum(-1)
            align_mean = float(align[mask].mean()) if mask.any() else float("nan")
        diag = {"ET": float(cL["ET"].mean()),
                "final_d": float(np.sqrt(cL["d2"][:, -1]).mean()),
                "no_cap": float((cL["S"][:, -1] > 0.5).mean()),
                "L_dist_P": Ld_P, "L_dist_E": Ld_E,
                "align_P_mean": align_mean}
        return LE, LP, diag

    t0 = time.time()
    for it in range(n_iters):
        for _ in range(cfg.n_P_steps):
            LE, LP, diag = step("P")
        for _ in range(cfg.n_E_steps):
            LE, LP, diag = step("E")
        if it % log_every == 0 or it == n_iters - 1:
            history["iter"].append(iter0 + it)
            history["L_P"].append(float(LP))
            history["L_E"].append(float(LE))
            history["L_dist_P"].append(diag["L_dist_P"])
            history["L_dist_E"].append(diag["L_dist_E"])
            history["ET"].append(diag["ET"])
            history["final_d"].append(diag["final_d"])
            history["no_cap"].append(diag["no_cap"])
            history["align_P_mean"].append(diag["align_P_mean"])
            print(f"it {iter0+it:5d}  L_P={LP:.3f}  L_dist_P={diag['L_dist_P']:.4f}  "
                  f"align_P={diag['align_P_mean']:.3f}  "
                  f"no_cap={diag['no_cap']:.2f}  "
                  f"[{time.time()-t0:.0f}s]")

    np.savez(ckpt_path,
             BE=paramsE["B"], bE=paramsE["b_body"], WE=paramsE["W"],
             BP=paramsP["B"], bP=paramsP["b_body"], WP=paramsP["W"],
             x_E0=x_E0, iter0=iter0 + n_iters,
             history=json.dumps(history))
    return paramsE, paramsP, x_E0, history


# =============================================================================
# Soft obstacle field (Direction 2)
# =============================================================================
# Breaks translation invariance.  The "obstacle" is a smooth Gaussian
# repulsive region on the torus:
#     U(x) = K * exp(-||x - x_obs||_T^2 / (2 sigma^2))
# Both players pay an additive penalty
#     L_obs(alpha) = mean_{t,j} U(x_alpha^j(t))
# integrated into the game loss.  The optimal feedback is no longer
# analytic in closed form.

def forward_loss_with_obstacle(xE, xP, L, eps, tau, t_max, n_t,
                                no_cap_pen, lam_d,
                                obs_center: np.ndarray,
                                obs_sigma: float,
                                obs_strength: float):
    """Same as forward_loss but adds a soft obstacle penalty on both players."""
    L_E, L_P, cache = forward_loss(xE, xP, L, eps, tau, t_max, n_t,
                                   no_cap_pen, lam_d=lam_d)
    # add the obstacle penalty
    dispE = (xE - obs_center + L / 2) % L - L / 2
    dispP = (xP - obs_center + L / 2) % L - L / 2
    d2E = (dispE ** 2).sum(-1)
    d2P = (dispP ** 2).sum(-1)
    U_E = obs_strength * np.exp(-d2E / (2.0 * obs_sigma ** 2))
    U_P = obs_strength * np.exp(-d2P / (2.0 * obs_sigma ** 2))
    L_obs_E = float(U_E.mean())
    L_obs_P = float(U_P.mean())
    L_P_total = L_P + L_obs_P
    L_E_total = L_E + L_obs_E
    cache["obstacle"] = {"dispE": dispE, "dispP": dispP, "U_E": U_E,
                          "U_P": U_P, "obs_sigma": obs_sigma,
                          "obs_strength": obs_strength}
    return L_E_total, L_P_total, cache


def backward_loss_with_obstacle(d_loss, cache, n_h, no_cap_pen, t_max,
                                is_pursuer):
    """Backward through the loss INCLUDING the obstacle penalty."""
    dxE, dxP = backward_loss(d_loss, cache, n_h, no_cap_pen, t_max, is_pursuer)
    obs = cache["obstacle"]
    sigma = obs["obs_sigma"]; strength = obs["obs_strength"]
    if is_pursuer:
        U = obs["U_P"]; disp = obs["dispP"]
        # d L_obs_P / d xP = -(disp / sigma^2) * U / (n_h * T)
        N = U.size
        dxP_obs = -(disp / (sigma ** 2)) * U[..., None] * d_loss / N
        dxP = dxP + dxP_obs
    else:
        U = obs["U_E"]; disp = obs["dispE"]
        N = U.size
        dxE_obs = -(disp / (sigma ** 2)) * U[..., None] * d_loss / N
        dxE = dxE + dxE_obs
    return dxE, dxP


def train_obstacle(cfg: DistillConfig,
                   n_iters: int,
                   ckpt_path: str,
                   obs_center: np.ndarray,
                   obs_sigma: float = 0.25,
                   obs_strength: float = 8.0,
                   log_every: int = 200,
                   resume: bool = False):
    """
    Train the actor in the presence of a soft Gaussian obstacle.
    No closed-form V* is available in this case; the framework runs
    unchanged.
    """
    rng = np.random.default_rng(cfg.seed)
    D = 2 * cfg.n_freqs + 1
    paramsE = init_params(cfg.n_h, cfg.n_b, D, rng)
    paramsP = init_params(cfg.n_h, cfg.n_b, D, rng)
    optE = Adam(paramsE, lr=cfg.lr_E)
    optP = Adam(paramsP, lr=cfg.lr_P)
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, 0.4, cfg.seed)
    # push any IC that happens to start inside the obstacle outward
    for j in range(cfg.n_h):
        delta = (x_E0[j] - obs_center + cfg.L / 2) % cfg.L - cfg.L / 2
        if np.linalg.norm(delta) < 2 * obs_sigma:
            shift = delta / max(np.linalg.norm(delta), 1e-3) * (2 * obs_sigma + 0.05)
            x_E0[j] = (obs_center + shift) % cfg.L
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)

    history = {"iter": [], "L_P": [], "L_E": [], "ET": [], "final_d": [],
               "no_cap": [], "obs_pen_P": [], "obs_pen_E": []}
    iter0 = 0
    pretrain_done = False
    if resume and os.path.exists(ckpt_path):
        npz = np.load(ckpt_path)
        paramsE = {"B": npz["BE"], "b_body": npz["bE"], "W": npz["WE"]}
        paramsP = {"B": npz["BP"], "b_body": npz["bP"], "W": npz["WP"]}
        x_E0 = npz["x_E0"]
        pretrain_done = True
        iter0 = int(npz["iter0"]) if "iter0" in npz.files else 0
        if "history" in npz.files:
            history = json.loads(str(npz["history"]))
        optE = Adam(paramsE, lr=cfg.lr_E)
        optP = Adam(paramsP, lr=cfg.lr_P)
        print(f"[obs] resumed at iter {iter0}")

    if not pretrain_done:
        for it in range(cfg.pretrain_P_iters):
            uE, cE = forward_dirs(paramsE, phi)
            uP, cP = forward_dirs(paramsP, phi)
            xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max,
                              cfg.n_t)
            LE, LP, cL = forward_loss_with_obstacle(
                xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max, cfg.n_t,
                cfg.no_capture_penalty, cfg.lam_d,
                obs_center, obs_sigma, obs_strength)
            dxE, dxP = backward_loss_with_obstacle(1.0, cL, cfg.n_h,
                                                   cfg.no_capture_penalty,
                                                   cfg.t_max, is_pursuer=True)
            duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                        cfg.t_max, cfg.n_t)
            gP = backward_dirs(paramsP, cP, duP)
            optP.step(paramsP, gP, sign=-1.0)
            if it % 400 == 0:
                print(f"[obs warm-P] it {it:4d}  L_P={LP:.3f}  "
                      f"obs_P={float(cL['obstacle']['U_P'].mean()):.3f}")

    def step(who):
        uE, cE = forward_dirs(paramsE, phi)
        uP, cP = forward_dirs(paramsP, phi)
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max,
                          cfg.n_t)
        LE, LP, cL = forward_loss_with_obstacle(
            xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max, cfg.n_t,
            cfg.no_capture_penalty, cfg.lam_d,
            obs_center, obs_sigma, obs_strength)
        is_p = (who == "P")
        dxE, dxP = backward_loss_with_obstacle(1.0, cL, cfg.n_h,
                                               cfg.no_capture_penalty,
                                               cfg.t_max, is_pursuer=is_p)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                    cfg.t_max, cfg.n_t)
        if is_p:
            gP = backward_dirs(paramsP, cP, duP)
            optP.step(paramsP, gP, sign=-1.0)
        else:
            gE = backward_dirs(paramsE, cE, duE)
            optE.step(paramsE, gE, sign=-1.0)
        diag = {"ET": float(cL["ET"].mean()),
                "final_d": float(np.sqrt(cL["d2"][:, -1]).mean()),
                "no_cap": float((cL["S"][:, -1] > 0.5).mean()),
                "obs_P": float(cL["obstacle"]["U_P"].mean()),
                "obs_E": float(cL["obstacle"]["U_E"].mean())}
        return LE, LP, diag

    t0 = time.time()
    for it in range(n_iters):
        for _ in range(cfg.n_P_steps):
            LE, LP, diag = step("P")
        for _ in range(cfg.n_E_steps):
            LE, LP, diag = step("E")
        if it % log_every == 0 or it == n_iters - 1:
            history["iter"].append(iter0 + it)
            history["L_P"].append(float(LP))
            history["L_E"].append(float(LE))
            history["ET"].append(diag["ET"])
            history["final_d"].append(diag["final_d"])
            history["no_cap"].append(diag["no_cap"])
            history["obs_pen_P"].append(diag["obs_P"])
            history["obs_pen_E"].append(diag["obs_E"])
            print(f"it {iter0+it:5d}  L_P={LP:.3f}  obs_P={diag['obs_P']:.3f}  "
                  f"obs_E={diag['obs_E']:.3f}  no_cap={diag['no_cap']:.2f}  "
                  f"[{time.time()-t0:.0f}s]")

    np.savez(ckpt_path,
             BE=paramsE["B"], bE=paramsE["b_body"], WE=paramsE["W"],
             BP=paramsP["B"], bP=paramsP["b_body"], WP=paramsP["W"],
             x_E0=x_E0, iter0=iter0 + n_iters,
             obs_center=obs_center, obs_sigma=obs_sigma,
             obs_strength=obs_strength,
             history=json.dumps(history))
    return paramsE, paramsP, x_E0, history


# =============================================================================
# Diagnostics: alignment as a function of N_h
# =============================================================================

def mean_pursuer_alignment(paramsE, paramsP, x_E0, cfg) -> float:
    """Average pursuer-toward-evader cosine on the realised rollout."""
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)
    uE, _ = forward_dirs(paramsE, phi)
    uP, _ = forward_dirs(paramsP, phi)
    xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    align = v2.alignment_with_target_direction(uP, xE, xP, cfg.L)
    return float(align.mean())


def capture_fraction_on_grid(paramsE, paramsP, x_E0, cfg, n_grid=20) -> float:
    """Fraction of grid cells captured by the open-loop multi-head actor
    using nearest-head IC lookup."""
    xs, ys, Tc = v2.capture_time_map(paramsE, paramsP, cfg, n_grid=n_grid)
    return float((~np.isnan(Tc)).mean())
