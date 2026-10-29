"""PyTorch rewrite of the multi-head open-loop pursuit-evasion actor.

Same model, loss and training protocol as `pursuit_evasion_numpy.py` run under the
published protocol of `rev2/core.py` (warm-up, 3:1 steps, Adam resets), with three
changes that do not alter the mathematics:

  * gradients come from autograd instead of the hand-written backward pass;
  * the 64-step rollout loop is a cumulative sum of the velocity increments (the
    positions are kept on the universal cover; the loss takes the minimum-image
    displacement, so the per-step wrap of the original is unnecessary);
  * every tensor carries a leading "run" axis R, so R independent runs
    (init seed x IC seed) train in one tensor program.  R = 1 is one run, and the
    per-run gradients are identical to training the runs separately because the
    batched loss is the *sum* over runs and Adam is elementwise.

Parameter initialisation and the IC sampler reuse NumPy's `default_rng` streams
exactly as the original code does, so (init_seed, ic_seed) = (0, 0) starts from
the published baseline's initial state.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- config
@dataclass(frozen=True)
class Config:
    """Field for field the `RunConfig` of rev2/core.py (published baseline defaults)."""
    # game
    L: float = 2.0
    v_E: float = 1.0
    v_P: float = 1.2
    eps: float = 0.08
    t_max: float = 4.0
    n_t: int = 64
    tau: float = 5e-3
    # architecture
    n_h: int = 16
    n_b: int = 16
    K: int = 12
    # loss
    lam_nc: float = 2.0
    lam_d: float = 5.0
    # protocol
    warmup: int = 1500
    n_iters: int = 14000
    n_P_steps: int = 3
    n_E_steps: int = 1
    lr_P: float = 1e-2
    lr_E: float = 1e-3
    adam_reset_every: int = 3500      # 0 = never
    # ICs
    ic_min_dist: float = 0.4
    # bookkeeping
    snapshot_every: int = 1000
    ckpt_at: tuple = (3500, 7000, 10500, 14000)

    @property
    def D(self) -> int:
        return 2 * self.K + 1

    @property
    def reach(self) -> float:
        return (self.v_P - self.v_E) * self.t_max + self.eps


# ------------------------------------------------- init (NumPy streams, verbatim)
def init_params_np(n_h: int, n_b: int, D: int, rng: np.random.Generator) -> dict:
    """Identical to pursuit_evasion_numpy.init_params (same draws, same order)."""
    return {
        "B": rng.standard_normal((n_b, D)) * np.sqrt(1.5 / D),
        "b": np.zeros(n_b),
        "W": rng.standard_normal((n_h, 2, n_b)) * np.sqrt(1.0 / n_b),
    }


def sample_evader_ICs(n_h: int, L: float, min_dist: float, seed: int) -> np.ndarray:
    """Identical to pursuit_evasion_numpy.sample_evader_ICs."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n_h:
        p = rng.uniform(0.0, L, size=2)
        dx = min(p[0], L - p[0])
        dy = min(p[1], L - p[1])
        if math.sqrt(dx * dx + dy * dy) >= min_dist:
            pts.append(p)
    return np.array(pts)


def initial_state_np(cfg: Config, init_seed: int, ic_seed: int,
                     perturb_seed: int = -1, perturb_scale: float = 0.0):
    """(pE, pP, x0) as NumPy arrays for one run; evader drawn first, then pursuer,
    from one stream seeded by init_seed, exactly as rev2/core.initial_state."""
    rng = np.random.default_rng(init_seed)
    pE = init_params_np(cfg.n_h, cfg.n_b, cfg.D, rng)
    pP = init_params_np(cfg.n_h, cfg.n_b, cfg.D, rng)
    if perturb_scale > 0.0:
        prng = np.random.default_rng(perturb_seed)
        for p in (pE, pP):
            for k in ("B", "W", "b"):            # same sorted-key order as core.py
                p[k] = p[k] + perturb_scale * prng.standard_normal(p[k].shape)
    x0 = sample_evader_ICs(cfg.n_h, cfg.L, cfg.ic_min_dist, ic_seed)
    return pE, pP, x0


def stack_runs(states, device, dtype):
    """List of (pE, pP, x0) -> batched torch parameters with requires_grad, and x0."""
    def stack(key, which):
        arr = np.stack([s[which][key] for s in states])
        return torch.tensor(arr, device=device, dtype=dtype, requires_grad=True)
    pE = {k: stack(k, 0) for k in ("B", "b", "W")}
    pP = {k: stack(k, 1) for k in ("B", "b", "W")}
    x0 = torch.tensor(np.stack([s[2] for s in states]), device=device, dtype=dtype)
    return pE, pP, x0


# ------------------------------------------------------------------ model
def fourier_features(t: torch.Tensor, t_max: float, K: int) -> torch.Tensor:
    """phi(t): (T, 2K+1) = [sin(w_k t), cos(w_k t), t/t_max]."""
    omegas = (2.0 * math.pi / t_max) * torch.arange(1, K + 1, device=t.device, dtype=t.dtype)
    ang = t[:, None] * omegas[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang), (t / t_max)[:, None]], dim=1)


def forward_dirs(p: dict, phi: torch.Tensor):
    """u: (R, n_h, T, 2) unit headings; H: (R, T, n_b) latent curve; z: (R, n_h, T, 2)."""
    z_body = torch.einsum("td,rbd->rtb", phi, p["B"]) + p["b"][:, None, :]
    H = torch.tanh(z_body)
    z = torch.einsum("rjab,rtb->rjta", p["W"], H)
    u = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return u, H, z


def rollout(uE: torch.Tensor, uP: torch.Tensor, x0: torch.Tensor, cfg: Config):
    """Trapezoidal (midpoint-average) step, as the original, on the universal cover.
    Returns lifted positions xE, xP: (R, n_h, n_t+1, 2)."""
    dt = cfg.t_max / cfg.n_t
    incE = dt * cfg.v_E * 0.5 * (uE[:, :, :-1] + uE[:, :, 1:])
    incP = dt * cfg.v_P * 0.5 * (uP[:, :, :-1] + uP[:, :, 1:])
    zero = torch.zeros_like(incE[:, :, :1])
    xE = x0[:, :, None, :] + torch.cat([zero, torch.cumsum(incE, dim=2)], dim=2)
    xP = torch.cat([zero, torch.cumsum(incP, dim=2)], dim=2)
    return xE, xP


def game_loss(xE: torch.Tensor, xP: torch.Tensor, cfg: Config):
    """Per-run losses L_E, L_P: (R,), plus survival S and torus distance d for diagnostics.

    L_P = <T_soft + lam_nc S(t_max) t_max + lam_d <d>_t>_heads
    L_E = <-T_soft - lam_d <d>_t>_heads
    """
    dt = cfg.t_max / cfg.n_t
    L = cfg.L
    disp = torch.remainder(xE - xP + L / 2.0, L) - L / 2.0           # min-image
    d2 = (disp ** 2).sum(-1)                                          # (R, n_h, T)
    d = torch.sqrt(d2 + 1e-12)
    arg = (cfg.eps ** 2 - d2) / cfg.tau
    log_one_minus_p = -F.softplus(arg)                                # log(1 - sigmoid(arg))
    S = torch.exp(torch.cumsum(log_one_minus_p, dim=-1))
    ET = dt * (0.5 * S[..., 0] + S[..., 1:-1].sum(-1) + 0.5 * S[..., -1])
    d_mean = d.mean(-1)
    L_P = (ET + cfg.lam_nc * S[..., -1] * cfg.t_max + cfg.lam_d * d_mean).mean(-1)
    L_E = (-ET - cfg.lam_d * d_mean).mean(-1)
    return L_E, L_P, S, d


# --------------------------------------------------------------- training
def make_adam(p: dict, lr: float):
    return torch.optim.Adam([p["B"], p["b"], p["W"]], lr=lr, betas=(0.9, 0.999), eps=1e-8)


def step(who: str, pE: dict, pP: dict, x0: torch.Tensor, phi: torch.Tensor,
         cfg: Config, optE, optP):
    """One gradient step of player `who` in every run (the other player is frozen)."""
    uE, _, _ = forward_dirs(pE, phi)
    uP, _, _ = forward_dirs(pP, phi)
    if who == "P":
        uE = uE.detach()
    else:
        uP = uP.detach()
    xE, xP = rollout(uE, uP, x0, cfg)
    L_E, L_P, _, _ = game_loss(xE, xP, cfg)
    loss = L_P.sum() if who == "P" else L_E.sum()    # sum: per-run gradients untouched
    opt = optP if who == "P" else optE
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss


def to_numpy(p: dict) -> dict:
    return {k: v.detach().cpu().numpy().copy() for k, v in p.items()}


def train(cfg: Config, runs: list, device="cpu", dtype=torch.float64,
          snapshot_fn=None, log_every: int = 0):
    """Train len(runs) independent runs in one batched program.

    runs : list of (init_seed, ic_seed) or (init_seed, ic_seed, perturb_seed, perturb_scale)
    snapshot_fn : callable(pE, pP, x0, it, cfg) -> list of dicts (one per run),
                  called after warm-up (it = 0) and every cfg.snapshot_every iterations.
    Returns dict with final params (numpy, run axis first), x0, snapshots and
    ckpts {it: (pE, pP)} at cfg.ckpt_at.
    """
    states = [initial_state_np(cfg, *r) for r in runs]
    pE, pP, x0 = stack_runs(states, device, dtype)
    t = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=device, dtype=dtype)
    phi = fourier_features(t, cfg.t_max, cfg.K)

    optE, optP = make_adam(pE, cfg.lr_E), make_adam(pP, cfg.lr_P)
    t0 = time.time()
    for _ in range(cfg.warmup):
        step("P", pE, pP, x0, phi, cfg, optE, optP)

    snapshots, ckpts = [], {}
    if snapshot_fn is not None:
        snapshots.append(snapshot_fn(pE, pP, x0, 0, cfg))
    for it in range(1, cfg.n_iters + 1):
        if cfg.adam_reset_every and it > 1 and (it - 1) % cfg.adam_reset_every == 0:
            optE, optP = make_adam(pE, cfg.lr_E), make_adam(pP, cfg.lr_P)
        for _ in range(cfg.n_P_steps):
            step("P", pE, pP, x0, phi, cfg, optE, optP)
        for _ in range(cfg.n_E_steps):
            loss = step("E", pE, pP, x0, phi, cfg, optE, optP)
        if snapshot_fn is not None and it % cfg.snapshot_every == 0:
            snapshots.append(snapshot_fn(pE, pP, x0, it, cfg))
        if it in cfg.ckpt_at:
            ckpts[it] = (to_numpy(pE), to_numpy(pP))
        if log_every and (it % log_every == 0 or it == cfg.n_iters):
            print(f"  it {it:6d}  mean L_E {loss.item() / len(runs):+.4f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return {"pE": to_numpy(pE), "pP": to_numpy(pP), "x0": x0.cpu().numpy(),
            "snapshots": snapshots, "ckpts": ckpts, "runs": runs,
            "seconds": time.time() - t0}


# ------------------------------------------------------- quick diagnostics
def _pr(ev: torch.Tensor) -> torch.Tensor:
    ev = ev.clamp_min(0.0)
    return ev.sum(-1) ** 2 / (ev ** 2).sum(-1)


@torch.no_grad()
def light_metrics(pE: dict, pP: dict, x0: torch.Tensor, it: int, cfg: Config,
                  n_dense: int = 512) -> list:
    """Same formulas as rev2/metrics.light_metrics, for every run at once:
    Gram PR and constancy of the unit-heading ensemble, PR of the time-centred latent,
    temporal PR of u, number of captured heads.  Returns one dict per run."""
    dev, dt_ = x0.device, x0.dtype
    td = torch.linspace(0.0, cfg.t_max, n_dense, device=dev, dtype=dt_)
    phid = fourier_features(td, cfg.t_max, cfg.K)
    out = {"it": it}
    for a, p in (("E", pE), ("P", pP)):
        u, H, _ = forward_dirs(p, phid)                        # (R,n_h,T,2), (R,T,n_b)
        R, n_h, T, _ = u.shape
        U = u.reshape(R, n_h, T * 2) / math.sqrt(T)
        ubar = u.mean(2)
        Xu = u.permute(0, 2, 1, 3).reshape(R, T, -1)
        Xu = Xu - Xu.mean(1, keepdim=True)
        Hc = H - H.mean(1, keepdim=True)
        # eigvalsh is not implemented on MPS; the matrices are tiny, so do them on CPU
        gram = torch.linalg.eigvalsh((U @ U.transpose(1, 2)).cpu())
        tpr = torch.linalg.eigvalsh((Xu.transpose(1, 2) @ Xu / T).cpu())
        prH = torch.linalg.eigvalsh((Hc.transpose(1, 2) @ Hc).cpu())
        out[a] = {"gram_pr": _pr(gram), "constancy": (ubar ** 2).sum(-1).mean(-1),
                  "tpr_U": _pr(tpr), "pr_H": _pr(prH)}
    tg = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=dev, dtype=dt_)
    phig = fourier_features(tg, cfg.t_max, cfg.K)
    uE, _, _ = forward_dirs(pE, phig)
    uP, _, _ = forward_dirs(pP, phig)
    xE, xP = rollout(uE, uP, x0, cfg)
    _, _, _, d = game_loss(xE, xP, cfg)
    n_cap = (d.min(-1).values <= cfg.eps).sum(-1)
    recs = []
    for r in range(x0.shape[0]):
        rec = {"it": it, "n_captured": int(n_cap[r])}
        for a in ("E", "P"):
            rec[a] = {k: float(v[r]) for k, v in out[a].items()}
        recs.append(rec)
    return recs


# ------------------------------------------------------------ checkpoints
def save_run_ckpt(path: str, cfg: Config, out: dict, r: int) -> None:
    """Write run r of a batched result in the key layout of rev2/core.save_ckpts
    (BE_<it>, bE_<it>, WE_<it>, BP_<it>, ..., x_E0, config), so rev2/core.load_ckpt
    and rev2/metrics.py read it unchanged."""
    arrs = {"x_E0": out["x0"][r], "config": np.array(repr(asdict(cfg)))}
    for it, (pE, pP) in out["ckpts"].items():
        arrs.update({f"BE_{it}": pE["B"][r], f"bE_{it}": pE["b"][r], f"WE_{it}": pE["W"][r],
                     f"BP_{it}": pP["B"][r], f"bP_{it}": pP["b"][r], f"WP_{it}": pP["W"][r]})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrs)


def load_ckpt(path: str, it: int = 14000):
    z = np.load(path)
    pE = {"B": z[f"BE_{it}"], "b": z[f"bE_{it}"], "W": z[f"WE_{it}"]}
    pP = {"B": z[f"BP_{it}"], "b": z[f"bP_{it}"], "W": z[f"WP_{it}"]}
    return pE, pP, z["x_E0"]
