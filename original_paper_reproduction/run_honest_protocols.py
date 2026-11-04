"""Honest convergent training protocols for the adversarial actor.

Open question (Ignatenko): is there an HONEST protocol (both players learning
from the game only, no analytic oracle) that tames the co-adaptation cycling
of simultaneous / best-response play?  We test three literature-standard
game-optimisation corrections, all self-play, on the SAME strict 1:1
simultaneous setting, isolating the update rule (same ICs, seeds, budget,
Adam optimisers lr_P=1e-2 / lr_E=1e-3):

  gda   simultaneous gradient descent-ascent (the unstable reference)
  eg    extragradient (Korpelevich; a lookahead half-step, then update with
        the gradient AT the lookahead) -- 2 grad evals / update
  ogda  optimistic GDA / extrapolation-from-the-past (Daskalakis et al.):
        step with 2 g_t - g_{t-1} through Adam ("optimistic Adam")
  avg   plain GDA but evaluate the iterate AVERAGE (Cesaro tail mean of the
        parameters) -- the classical zero-sum time-average that converges

References for the comparison: selfplay (3:1 baseline) and oracle (fixed
optimal evader; the non-honest stability ceiling).

Metric: mean pursuit gap on the analytically capturable heads, its late
oscillation (s.d. over the 2nd half), and final capture.  For 'avg' all three
are computed on the AVERAGED model.

Usage: python3 run_honest_protocols.py <protocol> <seed>
"""
import os, sys, json, copy, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
    backward_dirs, rollout, backward_rollout, forward_loss, backward_loss,
    init_params, sample_evader_ICs, Adam)

TOTAL = int(os.environ.get("HP_ITERS", 8000))
OUT   = os.environ.get("HP_OUT", "/tmp/hp_ckpts")
LR_P, LR_E = 1e-2, 1e-3

def cfg_(): return TrainConfig(n_h=16, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
    L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3, no_capture_penalty=2.0,
    lam_d=5.0, seed=0)

def opt_evader(x_E0, L, n_t):
    d = (x_E0 + L/2) % L - L/2
    dhat = d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9, None)
    return np.repeat(dhat[:, None, :], n_t + 1, axis=1)

def cp(p): return {k: v.copy() for k, v in p.items()}
def setp(p, s):
    for k in p: p[k][...] = s[k]
def plain(p, g, lr):
    for k in p: p[k] -= lr * g[k]
def comb(a, b, s):                       # a + s*b, elementwise per key
    return {k: a[k] + s * b[k] for k in a}

def main(protocol, seed):
    cfg = cfg_()
    tg = np.linspace(0, cfg.t_max, cfg.n_t + 1)
    phig = fourier_features(tg, cfg.t_max, cfg.n_freqs)
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, 0.4, seed=0)     # ICs fixed
    rng = np.random.default_rng(seed); D = 2 * cfg.n_freqs + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)
    uE_opt = opt_evader(x_E0, cfg.L, cfg.n_t)
    R = (cfg.v_P - cfg.v_E) * cfg.t_max + cfg.eps
    d0 = np.sqrt((((x_E0 + cfg.L/2) % cfg.L - cfg.L/2)**2).sum(-1))
    capturable = d0 <= R
    if protocol == "ogda_ema" and "HP_LR" in os.environ:
        lrP = lrE = float(os.environ["HP_LR"])          # symmetric override
    else:
        lrP, lrE = LR_P, LR_E   # asymmetric default: pursuer needs the faster step
    optP, optE = Adam(pP, lrP), Adam(pE, lrE)

    def grads(uE_fixed=None):
        """Per-player gradient dicts from one shared forward pass."""
        uP, cP = forward_dirs(pP, phig)
        if uE_fixed is None: uE, cE = forward_dirs(pE, phig)
        else: uE, cE = uE_fixed, None
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        _, _, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                cfg.n_t, cfg.no_capture_penalty, lam_d=cfg.lam_d)
        dxEp, dxPp = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty, cfg.t_max, True)
        _, duP = backward_rollout(dxEp, dxPp, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        gP = backward_dirs(pP, cP, duP)
        gE = None
        if cE is not None:
            dxEe, dxPe = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty, cfg.t_max, False)
            duE, _ = backward_rollout(dxEe, dxPe, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
            gE = backward_dirs(pE, cE, duE)
        return gP, gE

    # running (Cesaro tail) average of params for 'avg'
    start_avg = TOTAL // 2
    accP = cp(pP); accE = cp(pE); nacc = 0
    for k in accP: accP[k][...] = 0.0
    for k in accE: accE[k][...] = 0.0
    emaP = cp(pP); emaE = cp(pE); EMA=float(os.environ.get("HP_EMA", 0.999))
    gP_prev = gE_prev = None

    def meas(useP, useE, uEf=None):
        uP, _ = forward_dirs(useP, phig)
        uE = uEf if uEf is not None else forward_dirs(useE, phig)[0]
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        disp = (xE - xP + cfg.L/2) % cfg.L - cfg.L/2
        dmin = np.sqrt((disp**2).sum(-1)).min(1)
        return float(dmin[capturable].mean()), float((dmin[capturable] <= cfg.eps).mean())

    def avg_params():
        aP = {k: accP[k]/max(nacc,1) for k in accP}
        aE = {k: accE[k]/max(nacc,1) for k in accE}
        return aP, aE

    trace = []; ck = max(TOTAL//40, 1)
    for it in range(TOTAL):
        oracle = (protocol == "oracle")
        gP, gE = grads(uE_fixed=(uE_opt if oracle else None))
        if protocol in ("gda", "avg", "ema", "oracle"):
            optP.step(pP, gP, sign=-1.0)
            if gE is not None: optE.step(pE, gE, sign=-1.0)
        elif protocol == "selfplay":
            # 3:1 -- approximate by stepping P thrice (recompute) then E once
            optP.step(pP, gP, sign=-1.0)
            for _ in range(2):
                gP2, _ = grads(); optP.step(pP, gP2, sign=-1.0)
            if gE is not None: optE.step(pE, gE, sign=-1.0)
        elif protocol == "eg":
            sP, sE = cp(pP), cp(pE)                  # save
            plain(pP, gP, LR_P); plain(pE, gE, LR_E) # lookahead half-step
            gP2, gE2 = grads()                       # gradient at lookahead
            setp(pP, sP); setp(pE, sE)               # restore
            optP.step(pP, gP2, sign=-1.0); optE.step(pE, gE2, sign=-1.0)
        elif protocol in ("ogda", "ogda_ema"):
            eP = gP if gP_prev is None else comb(gP, comb(gP, gP_prev, -1.0), 1.0)  # 2gP-gP_prev
            eE = gE if gE_prev is None else comb(gE, comb(gE, gE_prev, -1.0), 1.0)
            optP.step(pP, eP, sign=-1.0); optE.step(pE, eE, sign=-1.0)
            gP_prev, gE_prev = gP, gE
        if protocol == "avg" and it >= start_avg:
            for k in accP: accP[k] += pP[k]
            for k in accE: accE[k] += pE[k]
            nacc += 1
        if protocol in ("ema", "ogda_ema"):
            if it < start_avg//2:
                for k in emaP: emaP[k][...] = pP[k]
                for k in emaE: emaE[k][...] = pE[k]
            else:
                for k in emaP: emaP[k][...] = EMA*emaP[k] + (1-EMA)*pP[k]
                for k in emaE: emaE[k][...] = EMA*emaE[k] + (1-EMA)*pE[k]
        if it % ck == 0:
            if protocol == "avg" and nacc > 0:
                aP, aE = avg_params(); g, c = meas(aP, aE)
            elif protocol in ("ema", "ogda_ema"): g, c = meas(emaP, emaE)
            elif oracle: g, c = meas(pP, None, uEf=uE_opt)
            else: g, c = meas(pP, pE)
            trace.append((round(it/TOTAL,3), round(g,4), round(c,3)))

    if protocol == "avg":
        aP, aE = avg_params(); gap, cap = meas(aP, aE)
    elif protocol in ("ema", "ogda_ema"): gap, cap = meas(emaP, emaE)
    elif protocol == "oracle": gap, cap = meas(pP, None, uEf=uE_opt)
    else: gap, cap = meas(pP, pE)
    tail = [t[1] for t in trace[len(trace)//2:]]
    res = {"protocol": protocol, "seed": seed, "iters": TOTAL,
           "n_capturable": int(capturable.sum()),
           "final_cap_of_capturable": round(cap,3),
           "final_gap_capturable": round(gap,4),
           "osc_sd_2ndhalf": round(float(np.std(tail)),4),
           "trace": trace}
    os.makedirs(OUT, exist_ok=True)
    print(json.dumps(res), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
