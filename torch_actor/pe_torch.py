"""PyTorch rewrite of the multi-head open-loop pursuit-evasion actor.

Same model, loss and training protocol as `pursuit_evasion_numpy.py` run under the
published protocol of `rev2/core.py` (warm-up, 3:1 steps, Adam resets), with three
changes that do not alter the mathematics:

  * gradients come from autograd instead of the hand-written backward pass;
  * the 64-step rollout loop is a cumulative sum of the velocity increments (the
    positions are kept on the universal cover; the loss takes the minimum-image
    displacement, so the per-step wrap of the original is unnecessary);
  * every tensor carries a leading "run" axis R, so R independent runs
    (init seed x IC seed, or different lam_d / v_P / IC sets) train in one tensor
    program.  R = 1 is one run, and the per-run gradients are identical to training
    the runs separately because the batched loss is the *sum* over runs and Adam is
    elementwise.

Parameter initialisation and the IC sampler reuse NumPy's `default_rng` streams
exactly as the original code does, so (init_seed, ic_seed) = (0, 0) starts from
the published baseline's initial state.

Per-run variations used by the paper's control arms are supported through
`train(...)`: `lam_d` and `v_P` may be arrays of length R, `symmetric=True`
mirrors the survival bonus into the evader loss (C2), `lam_d_schedule(it)` anneals
the shaping term (E7), `states` gives explicit (pE, pP, x0) per run (C4, E1, C7).
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
    depth: int = 1              # tanh layers in the body; layers 2..depth are n_b x n_b
    # loss
    lam_nc: float = 2.0
    lam_d: float = 5.0
    truncate: bool = False      # True: distance term weighted by survival, <S(t) d(t)>_t
    #                             (the game ends at capture); False: the draft's <d(t)>_t
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
    def dv(self) -> float:
        return self.v_P - self.v_E

    @property
    def reach(self) -> float:
        return (self.v_P - self.v_E) * self.t_max + self.eps


# ------------------------------------------------- init (NumPy streams, verbatim)
def init_params_np(n_h: int, n_b: int, D: int, rng: np.random.Generator, depth: int = 1) -> dict:
    """Identical to pursuit_evasion_numpy.init_params (same draws, same order) for depth 1;
    extra body layers B2, b2, ... are drawn afterwards, so depth-1 streams are unchanged."""
    p = {
        "B": rng.standard_normal((n_b, D)) * np.sqrt(1.5 / D),
        "b": np.zeros(n_b),
        "W": rng.standard_normal((n_h, 2, n_b)) * np.sqrt(1.0 / n_b),
    }
    for l in range(2, depth + 1):
        p[f"B{l}"] = rng.standard_normal((n_b, n_b)) * np.sqrt(1.5 / n_b)
        p[f"b{l}"] = np.zeros(n_b)
    return p


def extra_layers(p: dict) -> list:
    """Names of the body layers after the first, in order: [2, 3, ...]."""
    return sorted(int(k[1:]) for k in p if k.startswith("B") and k[1:].isdigit())


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
    pE = init_params_np(cfg.n_h, cfg.n_b, cfg.D, rng, cfg.depth)
    pP = init_params_np(cfg.n_h, cfg.n_b, cfg.D, rng, cfg.depth)
    if perturb_scale > 0.0:
        prng = np.random.default_rng(perturb_seed)
        for p in (pE, pP):
            for k in ["B", "W", "b"] + [f"{a}{l}" for l in extra_layers(p) for a in "Bb"]:
                p[k] = p[k] + perturb_scale * prng.standard_normal(p[k].shape)
    x0 = sample_evader_ICs(cfg.n_h, cfg.L, cfg.ic_min_dist, ic_seed)
    return pE, pP, x0


def stack_runs(states, device, dtype):
    """List of (pE, pP, x0) -> batched torch parameters with requires_grad, and x0."""
    def stack(key, which):
        arr = np.stack([np.asarray(s[which][key], dtype=np.float64) for s in states])
        return torch.tensor(arr, device=device, dtype=dtype, requires_grad=True)
    pE = {k: stack(k, 0) for k in states[0][0]}
    pP = {k: stack(k, 1) for k in states[0][1]}
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
    for l in extra_layers(p):                      # deeper bodies: H <- tanh(B_l H + b_l)
        H = torch.tanh(torch.einsum("rtb,rcb->rtc", H, p[f"B{l}"]) + p[f"b{l}"][:, None, :])
    z = torch.einsum("rjab,rtb->rjta", p["W"], H)
    u = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return u, H, z


def _per_run(x, ref: torch.Tensor):
    """float or (R,) array -> tensor broadcastable over (R, n_h, T, ...)."""
    if x is None:
        return None
    if np.isscalar(x):
        return float(x)
    return torch.as_tensor(np.asarray(x), device=ref.device, dtype=ref.dtype)[:, None, None]


def rollout(uE: torch.Tensor, uP: torch.Tensor, x0: torch.Tensor, cfg: Config, v_P=None):
    """Trapezoidal (midpoint-average) step, as the original, on the universal cover.
    Returns lifted positions xE, xP: (R, n_h, n_t+1, 2).  v_P: float or (R,) array."""
    dt = cfg.t_max / cfg.n_t
    vP = cfg.v_P if v_P is None else _per_run(v_P, uP)
    vP = vP[..., None] if torch.is_tensor(vP) else vP
    incE = dt * cfg.v_E * 0.5 * (uE[:, :, :-1] + uE[:, :, 1:])
    incP = dt * vP * 0.5 * (uP[:, :, :-1] + uP[:, :, 1:])
    zero = torch.zeros_like(incE[:, :, :1])
    xE = x0[:, :, None, :] + torch.cat([zero, torch.cumsum(incE, dim=2)], dim=2)
    xP = torch.cat([zero, torch.cumsum(incP, dim=2)], dim=2)
    return xE, xP


def game_loss(xE: torch.Tensor, xP: torch.Tensor, cfg: Config, lam_d=None,
              symmetric: bool = False):
    """Per-run losses L_E, L_P: (R,), plus survival S and torus distance d.

    L_P = <T_soft + lam_nc S(t_max) t_max + lam_d <d>_t>_heads
    L_E = <-T_soft - lam_d <d>_t>_heads                  (baseline, one-sided bonus)
    L_E = -L_P                                           (symmetric=True, C2)
    lam_d: float or (R,) array (per run).
    cfg.truncate: the distance term becomes <S(t) d(t)>_t, with S the survival
    probability, so time steps after capture do not contribute (rev2's R2).
    """
    dt = cfg.t_max / cfg.n_t
    L = cfg.L
    ld = cfg.lam_d if lam_d is None else _per_run(lam_d, xE)
    ld = ld[..., 0] if torch.is_tensor(ld) else ld                      # (R,1)
    disp = torch.remainder(xE - xP + L / 2.0, L) - L / 2.0           # min-image
    d2 = (disp ** 2).sum(-1)                                          # (R, n_h, T)
    d = torch.sqrt(d2 + 1e-12)
    arg = (cfg.eps ** 2 - d2) / cfg.tau
    log_one_minus_p = -F.softplus(arg)                                # log(1 - sigmoid(arg))
    S = torch.exp(torch.cumsum(log_one_minus_p, dim=-1))
    ET = dt * (0.5 * S[..., 0] + S[..., 1:-1].sum(-1) + 0.5 * S[..., -1])
    d_mean = (S * d).mean(-1) if cfg.truncate else d.mean(-1)
    nc = cfg.lam_nc * S[..., -1] * cfg.t_max
    L_P = (ET + nc + ld * d_mean).mean(-1)
    if symmetric:
        L_E = -L_P
    else:
        L_E = (-ET - ld * d_mean).mean(-1)
    return L_E, L_P, S, d


# --------------------------------------------------------------- training
def make_adam(p: dict, lr: float):
    return torch.optim.Adam(list(p.values()), lr=lr, betas=(0.9, 0.999), eps=1e-8)


def step(who: str, pE: dict, pP: dict, x0: torch.Tensor, phi: torch.Tensor,
         cfg: Config, optE, optP, lam_d=None, v_P=None, symmetric: bool = False,
         uE_fixed: torch.Tensor | None = None):
    """One gradient step of player `who` in every run (the other player is frozen).
    uE_fixed: (R, n_h, T, 2) replaces the evader network (oracle arm)."""
    if uE_fixed is None:
        uE, _, _ = forward_dirs(pE, phi)
    else:
        uE = uE_fixed
    uP, _, _ = forward_dirs(pP, phi)
    if who == "P":
        uE = uE.detach()
    else:
        uP = uP.detach()
    xE, xP = rollout(uE, uP, x0, cfg, v_P=v_P)
    L_E, L_P, _, _ = game_loss(xE, xP, cfg, lam_d=lam_d, symmetric=symmetric)
    loss = L_P.sum() if who == "P" else L_E.sum()    # sum: per-run gradients untouched
    opt = optP if who == "P" else optE
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss


def to_numpy(p: dict) -> dict:
    return {k: v.detach().cpu().numpy().copy() for k, v in p.items()}


def train(cfg: Config, runs: list | None = None, device="cpu", dtype=torch.float64,
          snapshot_fn=None, log_every: int = 0, states: list | None = None,
          lam_d=None, v_P=None, symmetric: bool = False, lam_d_schedule=None,
          ckpt_at=None, snapshot_every=None):
    """Train R independent runs in one batched program.

    runs    : list of (init_seed, ic_seed[, perturb_seed, perturb_scale]); ignored if
              `states` is given.
    states  : list of (pE, pP, x0) NumPy triples (explicit ICs or warm starts).
    snapshot_fn : callable(pE, pP, x0, it, cfg) -> list of dicts (one per run),
              called after warm-up (it = 0) and every `snapshot_every` iterations.
    lam_d, v_P : float or (R,) array; lam_d_schedule(it) overrides lam_d for the
              whole batch at self-play iteration it (warm-up uses lam_d).
    Returns dict with final params (NumPy, run axis first), x0, snapshots and
    ckpts {it: (pE, pP)} at `ckpt_at` (default cfg.ckpt_at).
    """
    if states is None:
        states = [initial_state_np(cfg, *r) for r in runs]
    R = len(states)
    ckpt_at = tuple(cfg.ckpt_at if ckpt_at is None else ckpt_at)
    snapshot_every = cfg.snapshot_every if snapshot_every is None else snapshot_every
    pE, pP, x0 = stack_runs(states, device, dtype)
    t = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=device, dtype=dtype)
    phi = fourier_features(t, cfg.t_max, cfg.K)
    kw = dict(lam_d=lam_d, v_P=v_P, symmetric=symmetric)

    optE, optP = make_adam(pE, cfg.lr_E), make_adam(pP, cfg.lr_P)
    t0 = time.time()
    for _ in range(cfg.warmup):
        step("P", pE, pP, x0, phi, cfg, optE, optP, **kw)

    snapshots, ckpts = [], {}
    if snapshot_fn is not None:
        snapshots.append(snapshot_fn(pE, pP, x0, 0, cfg))
    if 0 in ckpt_at:
        ckpts[0] = (to_numpy(pE), to_numpy(pP))
    loss = torch.zeros(())
    for it in range(1, cfg.n_iters + 1):
        if cfg.adam_reset_every and it > 1 and (it - 1) % cfg.adam_reset_every == 0:
            optE, optP = make_adam(pE, cfg.lr_E), make_adam(pP, cfg.lr_P)
        if lam_d_schedule is not None:
            kw["lam_d"] = lam_d_schedule(it)
        for _ in range(cfg.n_P_steps):
            step("P", pE, pP, x0, phi, cfg, optE, optP, **kw)
        for _ in range(cfg.n_E_steps):
            loss = step("E", pE, pP, x0, phi, cfg, optE, optP, **kw)
        if snapshot_fn is not None and snapshot_every and it % snapshot_every == 0:
            snapshots.append(snapshot_fn(pE, pP, x0, it, cfg))
        if it in ckpt_at:
            ckpts[it] = (to_numpy(pE), to_numpy(pP))
        if log_every and (it % log_every == 0 or it == cfg.n_iters):
            print(f"  it {it:6d}  mean L_E {loss.item() / R:+.4f}  "
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
                  n_dense: int = 512, v_P=None) -> list:
    """Same formulas as rev2/metrics.light_metrics, for every run at once:
    Gram PR and constancy of the unit-heading ensemble, PR of the time-centred
    latent, temporal PR of u and of z, number of captured heads.  One dict per run."""
    dev, dt_ = x0.device, x0.dtype
    td = torch.linspace(0.0, cfg.t_max, n_dense, device=dev, dtype=dt_)
    phid = fourier_features(td, cfg.t_max, cfg.K)
    out = {"it": it}
    for a, p in (("E", pE), ("P", pP)):
        u, H, z = forward_dirs(p, phid)                        # (R,n_h,T,2), (R,T,n_b)
        R, n_h, T, _ = u.shape
        U = u.reshape(R, n_h, T * 2) / math.sqrt(T)
        ubar = u.mean(2)
        Xu = u.permute(0, 2, 1, 3).reshape(R, T, -1)
        Xu = Xu - Xu.mean(1, keepdim=True)
        Xz = z.permute(0, 2, 1, 3).reshape(R, T, -1)
        Xz = Xz - Xz.mean(1, keepdim=True)
        Hc = H - H.mean(1, keepdim=True)
        # eigvalsh is not implemented on MPS; the matrices are tiny, so do them on CPU
        gram = torch.linalg.eigvalsh((U @ U.transpose(1, 2)).cpu())
        tpr = torch.linalg.eigvalsh((Xu.transpose(1, 2) @ Xu / T).cpu())
        tprz = torch.linalg.eigvalsh((Xz.transpose(1, 2) @ Xz / T).cpu())
        prH = torch.linalg.eigvalsh((Hc.transpose(1, 2) @ Hc).cpu())
        out[a] = {"gram_pr": _pr(gram), "constancy": (ubar ** 2).sum(-1).mean(-1),
                  "tpr_U": _pr(tpr), "tpr_Z": _pr(tprz), "pr_H": _pr(prH)}
    tg = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=dev, dtype=dt_)
    phig = fourier_features(tg, cfg.t_max, cfg.K)
    uE, _, _ = forward_dirs(pE, phig)
    uP, _, _ = forward_dirs(pP, phig)
    xE, xP = rollout(uE, uP, x0, cfg, v_P=v_P)
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
def save_run_ckpt(path: str, cfg: Config, out: dict, r: int, extra: dict | None = None) -> None:
    """Write run r of a batched result in the key layout of rev2/core.save_ckpts
    (BE_<it>, bE_<it>, WE_<it>, BP_<it>, ..., x_E0, config), so rev2/core.load_ckpt
    and rev2/metrics.py read it unchanged.  Also writes the legacy keys (BE, bE, WE,
    BP, bP, WP, iter0) for the final parameters, so the original analysis scripts
    and viz/visualize_actor.py read it too."""
    arrs = {"x_E0": out["x0"][r], "config": np.array(repr(asdict(cfg)))}
    key = lambda k, a: k[0] + a + k[1:]           # "B2" -> "BE2", "b" -> "bE", "W" -> "WE"  # noqa: E731
    for it, (pE, pP) in out["ckpts"].items():
        arrs.update({f"{key(k, 'E')}_{it}": pE[k][r] for k in pE})
        arrs.update({f"{key(k, 'P')}_{it}": pP[k][r] for k in pP})
    arrs.update({key(k, "E"): out["pE"][k][r] for k in out["pE"]})
    arrs.update({key(k, "P"): out["pP"][k][r] for k in out["pP"]})
    arrs["iter0"] = np.array(cfg.n_iters)
    if extra:
        arrs.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrs)


def load_ckpt(path: str, it: int | None = None):
    """(pE, pP, x0) with keys B, b, W.  it=None -> final (legacy keys)."""
    z = np.load(path)
    suf = "" if it is None else f"_{it}"
    out = []
    for a in "EP":
        p = {"B": z[f"B{a}{suf}"], "b": z[f"b{a}{suf}"], "W": z[f"W{a}{suf}"]}
        for l in range(2, 50):
            if f"B{a}{l}{suf}" not in z.files:
                break
            p[f"B{l}"] = z[f"B{a}{l}{suf}"]; p[f"b{l}"] = z[f"b{a}{l}{suf}"]
        out.append(p)
    return out[0], out[1], z["x_E0"]
