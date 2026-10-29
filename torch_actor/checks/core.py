"""Canonical, deterministic training harness for the revision (rev2).

Every run in the revised paper goes through `train_actor`. It reuses the
verified forward/backward code of `pursuit_evasion_numpy.py` unchanged and
only fixes the *protocol*, which in the original bundle differed between
arms (3,500-iteration Adam chunks for the baseline, 2,500 for C3/C5/dv,
33-second wall-clock chunks for C1/C2/C7).  Here the chunking is an explicit
config field, so a run is a pure function of its `RunConfig` on a given
machine.

Nothing analytic (V*, u*) enters training.  The pursuer starts at the origin;
evader initial conditions come from `sample_evader_ICs(ic_seed)`; the network
initialisation comes from `default_rng(init_seed)` (evader first, then
pursuer), exactly as in the original scripts, so (init_seed, ic_seed) =
(0, 0) is the published baseline configuration.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
from dataclasses import dataclass, asdict, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from numpy_reference import (  # noqa: E402  (verbatim copy of pursuit_evasion_numpy.py)
    fourier_features, forward_dirs, backward_dirs, rollout, backward_rollout,
    forward_loss, backward_loss, init_params, sample_evader_ICs, Adam,
)


@dataclass(frozen=True)
class RunConfig:
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
    # protocol (defaults = the published baseline, run_v2.py phases 1,1b,1b,1b)
    warmup: int = 1500            # pursuer-only steps before self-play
    n_iters: int = 14000          # self-play iterations (each = P steps + E steps)
    n_P_steps: int = 3
    n_E_steps: int = 1
    lr_P: float = 1e-2
    lr_E: float = 1e-3
    adam_reset_every: int = 3500  # fresh Adam every N self-play iters; 0 = never
    # randomness
    init_seed: int = 0
    ic_seed: int = 0
    ic_min_dist: float = 0.4
    perturb_seed: int = -1        # >= 0: add perturb_scale * N(0,1) to every init param
    perturb_scale: float = 0.0
    # bookkeeping
    snapshot_every: int = 1000            # light metrics every N self-play iters
    ckpt_at: tuple = (3500, 7000, 10500, 14000)

    @property
    def dv(self) -> float:
        return self.v_P - self.v_E

    @property
    def reach(self) -> float:
        """Forced-capture radius R = (v_P - v_E) t_max + eps."""
        return self.dv * self.t_max + self.eps

    def tag(self) -> str:
        h = hashlib.sha1(repr(sorted(asdict(self).items())).encode()).hexdigest()
        return h[:10]


def initial_state(cfg: RunConfig):
    """Network parameters and evader ICs for a config (no training)."""
    rng = np.random.default_rng(cfg.init_seed)
    D = 2 * cfg.K + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)
    if cfg.perturb_scale > 0.0:
        prng = np.random.default_rng(cfg.perturb_seed)
        for p in (pE, pP):
            for k in sorted(p):
                p[k] = p[k] + cfg.perturb_scale * prng.standard_normal(p[k].shape)
    x0 = sample_evader_ICs(cfg.n_h, cfg.L, cfg.ic_min_dist, cfg.ic_seed)
    return pE, pP, x0


def train_grid(cfg: RunConfig) -> np.ndarray:
    return fourier_features(np.linspace(0.0, cfg.t_max, cfg.n_t + 1),
                            cfg.t_max, cfg.K)


def train_actor(cfg: RunConfig, snapshot_fn=None):
    """Self-play training under `cfg`.

    Parameters
    ----------
    snapshot_fn : callable(pE, pP, x0, it) -> dict, optional
        Called after the warm-up (it = 0) and every `cfg.snapshot_every`
        self-play iterations; its return values are collected.

    Returns
    -------
    dict with final params, the IC set, `snapshots` (list of dicts) and
    `ckpts` ({iteration: (pE, pP)} copies at `cfg.ckpt_at`).
    """
    pE, pP, x0 = initial_state(cfg)
    phi = train_grid(cfg)

    def step(who, optE, optP):
        uE, cE = forward_dirs(pE, phi)
        uP, cP = forward_dirs(pP, phi)
        xE, xP = rollout(uE, uP, x0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        _, _, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                cfg.n_t, cfg.lam_nc, lam_d=cfg.lam_d)
        is_p = who == "P"
        dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.lam_nc, cfg.t_max,
                                 is_pursuer=is_p)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                    cfg.t_max, cfg.n_t)
        if is_p:
            optP.step(pP, backward_dirs(pP, cP, duP), sign=-1.0)
        else:
            optE.step(pE, backward_dirs(pE, cE, duE), sign=-1.0)

    optE, optP = Adam(pE, lr=cfg.lr_E), Adam(pP, lr=cfg.lr_P)
    for _ in range(cfg.warmup):
        step("P", optE, optP)

    snapshots, ckpts = [], {}
    if snapshot_fn is not None:
        snapshots.append(snapshot_fn(pE, pP, x0, 0))
    for it in range(1, cfg.n_iters + 1):
        if cfg.adam_reset_every and it > 1 and (it - 1) % cfg.adam_reset_every == 0:
            optE, optP = Adam(pE, lr=cfg.lr_E), Adam(pP, lr=cfg.lr_P)
        for _ in range(cfg.n_P_steps):
            step("P", optE, optP)
        for _ in range(cfg.n_E_steps):
            step("E", optE, optP)
        if snapshot_fn is not None and it % cfg.snapshot_every == 0:
            snapshots.append(snapshot_fn(pE, pP, x0, it))
        if it in cfg.ckpt_at:
            ckpts[it] = ({k: v.copy() for k, v in pE.items()},
                         {k: v.copy() for k, v in pP.items()})
    return {"pE": pE, "pP": pP, "x0": x0, "snapshots": snapshots, "ckpts": ckpts}


def params_digest(pE: dict, pP: dict) -> str:
    h = hashlib.sha1()
    for p in (pE, pP):
        for k in sorted(p):
            h.update(np.ascontiguousarray(p[k]).tobytes())
    return h.hexdigest()[:16]


def save_ckpts(path: str, cfg: RunConfig, out: dict) -> None:
    arrs = {"x_E0": out["x0"], "config": np.array(repr(asdict(cfg)))}
    for it, (pE, pP) in out["ckpts"].items():
        arrs.update({f"BE_{it}": pE["B"], "bE_%d" % it: pE["b_body"], f"WE_{it}": pE["W"],
                     f"BP_{it}": pP["B"], "bP_%d" % it: pP["b_body"], f"WP_{it}": pP["W"]})
    np.savez_compressed(path, **arrs)


def load_ckpt(path: str, it: int = 14000):
    z = np.load(path)
    pE = {"B": z[f"BE_{it}"], "b_body": z[f"bE_{it}"], "W": z[f"WE_{it}"]}
    pP = {"B": z[f"BP_{it}"], "b_body": z[f"bP_{it}"], "W": z[f"WP_{it}"]}
    return pE, pP, z["x_E0"]
