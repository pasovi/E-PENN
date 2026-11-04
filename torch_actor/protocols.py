"""Section 8 of the draft: training-protocol study, honest corrections, Shapley.

Ports of run_game_protocols.py, run_honest_protocols.py and shap_attribution.py,
batched over seeds (one tensor program per protocol).  Result dicts use the same
keys as the original JSON lines, so the figure code is shared with the draft's.

  game protocols (Fig. 6, 14,000 gradient steps):
    selfplay  800 pursuer-only warm-up, then 3:1 P:E alternation
    oracle    evader fixed at the analytic constant-heading saddle; pursuer only
    simul     strict 1:1 simultaneous GDA (both gradients from one forward pass)
    abr       alternating best-response blocks of 250 pursuer, 250 evader steps
  honest protocols (Fig. 7, 6,000 iterations, Adam 1e-2 / 1e-3):
    gda, eg (extragradient), ogda (optimistic), avg (Cesaro tail mean of the
    parameters, evaluated on the mean), ogda_ema (optimistic + EMA 0.999), oracle
  shapley (Fig. 8): exact grouped Shapley of capture over the four input bands
    for an oracle-trained and a 1:1-self-play-trained pursuer (5,000 steps each)
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

import pe_torch as pt


# ------------------------------------------------------------- utilities
def saddle_evader(x0: torch.Tensor, cfg: pt.Config) -> torch.Tensor:
    """Constant heading along the min-image displacement: (R, n_h, n_t+1, 2)."""
    d = torch.remainder(x0 + cfg.L / 2, cfg.L) - cfg.L / 2
    dh = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return dh[:, :, None, :].expand(-1, -1, cfg.n_t + 1, -1).contiguous()


def capturable_mask(x0: torch.Tensor, cfg: pt.Config) -> torch.Tensor:
    d = torch.remainder(x0 + cfg.L / 2, cfg.L) - cfg.L / 2
    return d.norm(dim=-1) <= cfg.reach


def setup(cfg, seeds, device, dtype, ic_seed=0):
    states = [pt.initial_state_np(cfg, s, ic_seed) for s in seeds]
    pE, pP, x0 = pt.stack_runs(states, device, dtype)
    t = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=device, dtype=dtype)
    phi = pt.fourier_features(t, cfg.t_max, cfg.K)
    return pE, pP, x0, phi, saddle_evader(x0, cfg), capturable_mask(x0, cfg)


@torch.no_grad()
def measure(pP, pE, x0, phi, cfg, cap, uE_fixed=None):
    """Per run: mean pursuit gap and capture fraction on the capturable heads."""
    uP, _, _ = pt.forward_dirs(pP, phi)
    uE = uE_fixed if uE_fixed is not None else pt.forward_dirs(pE, phi)[0]
    xE, xP = pt.rollout(uE, uP, x0, cfg)
    _, _, _, d = pt.game_loss(xE, xP, cfg)
    dmin = d.min(-1).values                                        # (R, n_h)
    n = cap.sum(-1).clamp_min(1)
    gap = (dmin * cap).sum(-1) / n
    capf = ((dmin <= cfg.eps) & cap).sum(-1) / n
    return gap.cpu().numpy(), capf.cpu().numpy()


def grads(pE, pP, x0, phi, cfg, uE_fixed=None):
    """Per-player gradient dicts from one shared forward pass (simultaneous play)."""
    uP, _, _ = pt.forward_dirs(pP, phi)
    uE = uE_fixed if uE_fixed is not None else pt.forward_dirs(pE, phi)[0]
    xE, xP = pt.rollout(uE, uP, x0, cfg)
    L_E, L_P, _, _ = pt.game_loss(xE, xP, cfg)
    gP = dict(zip(("B", "b", "W"), torch.autograd.grad(L_P.sum(), [pP["B"], pP["b"], pP["W"]],
                                                       retain_graph=uE_fixed is None)))
    gE = None
    if uE_fixed is None:
        gE = dict(zip(("B", "b", "W"), torch.autograd.grad(L_E.sum(), [pE["B"], pE["b"], pE["W"]])))
    return gP, gE


def apply(opt, p, g):
    """Adam step with an explicit gradient dict."""
    for k in p:
        p[k].grad = g[k].detach().clone()
    opt.step()
    opt.zero_grad(set_to_none=True)


def copy_params(p):
    return {k: v.detach().clone() for k, v in p.items()}


def set_params(p, s):
    with torch.no_grad():
        for k in p:
            p[k].copy_(s[k])


def results(protocol, seeds, total, cap, gap, capf, traces):
    out = []
    for r, s in enumerate(seeds):
        tr = traces[r]
        tail = [t[1] for t in tr[len(tr) // 2:]]
        out.append({"protocol": protocol, "seed": int(s), "iters": int(total),
                    "n_capturable": int(cap[r].sum()),
                    "final_cap_of_capturable": round(float(capf[r]), 3),
                    "final_gap_capturable": round(float(gap[r]), 4),
                    "osc_sd_2ndhalf": round(float(np.std(tail)), 4),
                    "trace": tr})
    return out


# ------------------------------------------------------- Fig. 6 protocols
def run_game_protocol(protocol: str, seeds, total: int = 14000, device="cpu",
                      dtype=torch.float64, cfg: pt.Config | None = None):
    cfg = cfg or pt.Config()
    pE, pP, x0, phi, uE_opt, cap = setup(cfg, seeds, device, dtype)
    optE, optP = pt.make_adam(pE, 1e-3), pt.make_adam(pP, 1e-2)
    R = len(seeds)
    traces = [[] for _ in range(R)]
    ck = max(total // 40, 1)
    gstep = 0
    uEf = uE_opt if protocol == "oracle" else None

    def maybe_check():
        nonlocal gstep
        if gstep % ck == 0:
            g, c = measure(pP, pE, x0, phi, cfg, cap, uE_fixed=uEf)
            for r in range(R):
                traces[r].append((round(gstep / total, 3), round(float(g[r]), 4), round(float(c[r]), 3)))
        gstep += 1

    def st(who):
        pt.step(who, pE, pP, x0, phi, cfg, optE, optP, uE_fixed=uEf)

    if protocol == "selfplay":
        for _ in range(800):
            st("P"); maybe_check()
        while gstep < total:
            for _ in range(3):
                st("P"); maybe_check()
            st("E"); maybe_check()
    elif protocol == "oracle":
        while gstep < total:
            st("P"); maybe_check()
    elif protocol == "simul":
        while gstep < total:
            gP, gE = grads(pE, pP, x0, phi, cfg)
            apply(optP, pP, gP); apply(optE, pE, gE); maybe_check()
    elif protocol == "abr":
        M = 250
        while gstep < total:
            for _ in range(M):
                if gstep >= total:
                    break
                st("P"); maybe_check()
            for _ in range(M):
                if gstep >= total:
                    break
                st("E"); maybe_check()
    else:
        raise ValueError(protocol)
    g, c = measure(pP, pE, x0, phi, cfg, cap, uE_fixed=uEf)
    return results(protocol, seeds, total, cap.cpu().numpy(), g, c, traces)


# ------------------------------------------------- Fig. 7 honest protocols
def run_honest_protocol(protocol: str, seeds, total: int = 6000, device="cpu",
                        dtype=torch.float64, cfg: pt.Config | None = None, ema: float = 0.999):
    cfg = cfg or pt.Config()
    pE, pP, x0, phi, uE_opt, cap = setup(cfg, seeds, device, dtype)
    lrP, lrE = 1e-2, 1e-3
    optE, optP = pt.make_adam(pE, lrE), pt.make_adam(pP, lrP)
    R = len(seeds)
    oracle = protocol == "oracle"
    uEf = uE_opt if oracle else None
    start_avg = total // 2
    accP = {k: torch.zeros_like(v) for k, v in pP.items()}
    accE = {k: torch.zeros_like(v) for k, v in pE.items()}
    nacc = 0
    emaP, emaE = copy_params(pP), copy_params(pE)
    gP_prev = gE_prev = None
    traces = [[] for _ in range(R)]
    ck = max(total // 40, 1)

    def eval_params():
        if protocol == "avg" and nacc > 0:
            return ({k: v / nacc for k, v in accP.items()}, {k: v / nacc for k, v in accE.items()})
        if protocol in ("ema", "ogda_ema"):
            return emaP, emaE
        return pP, pE

    for it in range(total):
        gP, gE = grads(pE, pP, x0, phi, cfg, uE_fixed=uEf)
        if protocol in ("gda", "avg", "ema", "oracle"):
            apply(optP, pP, gP)
            if gE is not None:
                apply(optE, pE, gE)
        elif protocol == "eg":
            sP, sE = copy_params(pP), copy_params(pE)
            with torch.no_grad():                       # plain lookahead half-step
                for k in pP:
                    pP[k] -= lrP * gP[k]
                for k in pE:
                    pE[k] -= lrE * gE[k]
            gP2, gE2 = grads(pE, pP, x0, phi, cfg)      # gradient at the lookahead
            set_params(pP, sP); set_params(pE, sE)
            apply(optP, pP, gP2); apply(optE, pE, gE2)
        elif protocol in ("ogda", "ogda_ema"):
            eP = gP if gP_prev is None else {k: 2 * gP[k] - gP_prev[k] for k in gP}
            eE = gE if gE_prev is None else {k: 2 * gE[k] - gE_prev[k] for k in gE}
            apply(optP, pP, eP); apply(optE, pE, eE)
            gP_prev, gE_prev = {k: v.detach() for k, v in gP.items()}, {k: v.detach() for k, v in gE.items()}
        else:
            raise ValueError(protocol)
        if protocol == "avg" and it >= start_avg:
            with torch.no_grad():
                for k in accP:
                    accP[k] += pP[k]
                for k in accE:
                    accE[k] += pE[k]
            nacc += 1
        if protocol in ("ema", "ogda_ema"):
            with torch.no_grad():
                if it < start_avg // 2:
                    for k in emaP:
                        emaP[k].copy_(pP[k]); emaE[k].copy_(pE[k])
                else:
                    for k in emaP:
                        emaP[k].mul_(ema).add_((1 - ema) * pP[k])
                        emaE[k].mul_(ema).add_((1 - ema) * pE[k])
        if it % ck == 0:
            qP, qE = eval_params()
            g, c = measure(qP, qE, x0, phi, cfg, cap, uE_fixed=uEf)
            for r in range(R):
                traces[r].append((round(it / total, 3), round(float(g[r]), 4), round(float(c[r]), 3)))
    qP, qE = eval_params()
    g, c = measure(qP, qE, x0, phi, cfg, cap, uE_fixed=uEf)
    return results(protocol, seeds, total, cap.cpu().numpy(), g, c, traces)


# ------------------------------------------------------------ Fig. 8 Shapley
BANDS = {"linear": [24], "low": [0, 1, 2, 3, 12, 13, 14, 15],
         "mid": [4, 5, 6, 7, 16, 17, 18, 19], "high": [8, 9, 10, 11, 20, 21, 22, 23]}
GROUPS = ["linear", "low", "mid", "high"]


def run_shapley(seeds, n_iters: int = 5000, device="cpu", dtype=torch.float64,
                cfg: pt.Config | None = None):
    cfg = cfg or pt.Config()
    assert cfg.K == 12, "band definitions assume K = 12"
    pE, pP, x0, phi, uE_opt, cap = setup(cfg, seeds, device, dtype)
    R = len(seeds)
    # oracle pursuer: against the fixed saddle evader
    pP_o = copy_params(pP); pP_o = {k: v.requires_grad_(True) for k, v in pP_o.items()}
    opt = pt.make_adam(pP_o, 1e-2)
    for _ in range(n_iters):
        pt.step("P", pE, pP_o, x0, phi, cfg, None, opt, uE_fixed=uE_opt)
    # self-play pursuer: honest 1:1 simultaneous GDA
    optE, optP = pt.make_adam(pE, 1e-3), pt.make_adam(pP, 1e-2)
    for _ in range(n_iters):
        gP, gE = grads(pE, pP, x0, phi, cfg)
        apply(optP, pP, gP); apply(optE, pE, gE)
    phi_mean = phi.mean(0, keepdim=True)

    def masked_phi(active):
        keep = set()
        for g in active:
            keep |= set(BANDS[g])
        m = phi.clone()
        for c in range(phi.shape[1]):
            if c not in keep:
                m[:, c] = phi_mean[0, c]
        return m

    @torch.no_grad()
    def perf(p, active):
        uP, _, _ = pt.forward_dirs(p, masked_phi(active))
        xE, xP = pt.rollout(uE_opt, uP, x0, cfg)
        _, _, _, d = pt.game_loss(xE, xP, cfg)
        dmin = d.min(-1).values
        n = cap.sum(-1).clamp_min(1)
        capf = ((dmin <= cfg.eps) & cap).sum(-1) / n
        gap = (dmin * cap).sum(-1) / n
        return capf.cpu().numpy(), gap.cpu().numpy()

    def shapley(p, which):
        val = {}
        for r_ in range(len(GROUPS) + 1):
            for S in itertools.combinations(GROUPS, r_):
                c, g = perf(p, set(S))
                val[frozenset(S)] = c if which == 0 else -g
        n = len(GROUPS)
        phe = {}
        for gi in GROUPS:
            rest = [x for x in GROUPS if x != gi]
            tot = np.zeros(R)
            for r_ in range(len(rest) + 1):
                for S in itertools.combinations(rest, r_):
                    S = frozenset(S)
                    w = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                    tot = tot + w * (val[S | {gi}] - val[S])
            phe[gi] = tot
        return phe

    outs = []
    per = {}
    for name, p in (("oracle", pP_o), ("selfplay", pP)):
        capF, gapF = perf(p, set(GROUPS))
        capE, gapE = perf(p, set())
        per[name] = (capF, gapF, capE, gapE, shapley(p, 0), shapley(p, 1))
    for r, s in enumerate(seeds):
        out = {"seed": int(s), "iters": int(n_iters)}
        for name in ("oracle", "selfplay"):
            capF, gapF, capE, gapE, sc, sg = per[name]
            out[name] = {"cap_full": round(float(capF[r]), 3), "gap_full": round(float(gapF[r]), 4),
                         "cap_baseline(DC)": round(float(capE[r]), 3),
                         "gap_baseline(DC)": round(float(gapE[r]), 4),
                         "shap_cap": {k: round(float(v[r]), 3) for k, v in sc.items()},
                         "shap_gapred": {k: round(float(v[r]), 4) for k, v in sg.items()}}
        outs.append(out)
    return outs
