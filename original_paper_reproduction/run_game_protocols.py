"""Actor training-protocol comparison (referee: Ignatenko).

Tests whether the actor's mediocre capture rate is a self-play/co-adaptation
artifact by comparing three protocols at matched budget, seeds, ICs and
architecture (N_h=16, K=12, t_max=4, L=2, v_P=1.2, v_E=1, eps=0.08):

  selfplay : paper baseline -- 800 pursuer-only warmup, then 3:1 P:E steps.
  oracle   : evader FIXED to the analytic saddle (constant heading = min-image
             displacement, the open-loop optimum, Prop.1 of the companion paper);
             train the pursuer only.  Removes non-stationarity entirely.
  abr      : alternating best-response -- coarse Gauss-Seidel blocks
             (M pursuer-only iters, then M evader-only iters), iterated.

Reports capture among the analytically capturable ICs (d0 <= R = Dv*t_max+eps)
and a late-training oscillation metric (s.d. of capture fraction over the last
checkpoints) to probe cycling.

Usage: python3 run_game_protocols.py <protocol> <seed>
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
    backward_dirs, rollout, backward_rollout, forward_loss, backward_loss,
    init_params, sample_evader_ICs, Adam)

TOTAL = int(os.environ.get("GP_ITERS", 5000))
OUT = os.environ.get("GP_OUT", "/tmp/gp_ckpts")

def make_cfg(): return TrainConfig(n_h=16, n_b=16, n_freqs=12, t_max=4.0,
    n_t=64, L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
    no_capture_penalty=2.0, lam_d=5.0, seed=0)

def optimal_evader_dirs(x_E0, L, n_t):
    """Constant-heading saddle evader: flee along the min-image displacement."""
    d = (x_E0 + L/2) % L - L/2                 # min-image of (x_E0 - origin)
    dhat = d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9, None)
    return np.repeat(dhat[:, None, :], n_t + 1, axis=1)   # (n_h, n_t+1, 2)

def capture_stats(pE, pP, x_E0, cfg, phig, uE_fixed=None):
    uP, _ = forward_dirs(pP, phig)
    uE = uE_fixed if uE_fixed is not None else forward_dirs(pE, phig)[0]
    xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    disp = (xE - xP + cfg.L/2) % cfg.L - cfg.L/2
    dmin = np.sqrt((disp**2).sum(-1)).min(1)          # per head
    R = (cfg.v_P - cfg.v_E) * cfg.t_max + cfg.eps
    d0 = np.sqrt((((x_E0 + cfg.L/2) % cfg.L - cfg.L/2)**2).sum(-1))
    cap = dmin <= cfg.eps
    capturable = d0 <= R
    return cap, capturable, dmin

def run(protocol, seed):
    cfg = make_cfg()
    tg = np.linspace(0, cfg.t_max, cfg.n_t + 1)
    phig = fourier_features(tg, cfg.t_max, cfg.n_freqs)
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, 0.4, seed=0)      # ICs fixed
    rng = np.random.default_rng(seed)
    D = 2 * cfg.n_freqs + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)
    uE_opt = optimal_evader_dirs(x_E0, cfg.L, cfg.n_t)
    R = (cfg.v_P - cfg.v_E) * cfg.t_max + cfg.eps
    d0 = np.sqrt((((x_E0 + cfg.L/2) % cfg.L - cfg.L/2)**2).sum(-1))
    capturable = d0 <= R
    optE, optP = Adam(pE, 1e-3), Adam(pP, 1e-2)

    def grads_from_shared():
        """One shared forward pass -> gradients for BOTH players (simultaneous)."""
        uE, cE = forward_dirs(pE, phig); uP, cP = forward_dirs(pP, phig)
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        _, _, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                cfg.n_t, cfg.no_capture_penalty, lam_d=cfg.lam_d)
        dxE_p, dxP_p = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty, cfg.t_max, True)
        dxE_e, dxP_e = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty, cfg.t_max, False)
        _, duP = backward_rollout(dxE_p, dxP_p, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        duE, _ = backward_rollout(dxE_e, dxP_e, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        return backward_dirs(pP, cP, duP), backward_dirs(pE, cE, duE)

    def step(who, uE_fixed=None):
        uP, cP = forward_dirs(pP, phig)
        if uE_fixed is None: uE, cE = forward_dirs(pE, phig)
        else: uE, cE = uE_fixed, None
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        _, _, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                cfg.n_t, cfg.no_capture_penalty, lam_d=cfg.lam_d)
        ip = (who == "P")
        dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty, cfg.t_max, ip)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        if ip: optP.step(pP, backward_dirs(pP, cP, duP), sign=-1.0)
        elif cE is not None: optE.step(pE, backward_dirs(pE, cE, duE), sign=-1.0)

    trace = []                      # (frac_done, mean_gap_capturable)
    ncheck = 40
    def measure():
        uEf = uE_opt if protocol == "oracle" else None
        uP, _ = forward_dirs(pP, phig)
        uE = uEf if uEf is not None else forward_dirs(pE, phig)[0]
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        disp = (xE - xP + cfg.L/2) % cfg.L - cfg.L/2
        dmin = np.sqrt((disp**2).sum(-1)).min(1)
        return float(dmin[capturable].mean()), float((dmin[capturable] <= cfg.eps).mean()), dmin

    gstep = 0; ck = max(TOTAL // ncheck, 1)
    def maybe_check():
        nonlocal gstep
        if gstep % ck == 0:
            g, c, _ = measure(); trace.append((round(gstep/TOTAL,3), round(g,4), round(c,3)))
        gstep += 1

    if protocol == "selfplay":              # paper baseline: warmup + 3:1
        for _ in range(800): step("P"); maybe_check()
        while gstep < TOTAL:
            for _ in range(3): step("P"); maybe_check()
            step("E"); maybe_check()
    elif protocol == "oracle":              # fixed optimal (committed) evader
        while gstep < TOTAL: step("P", uE_fixed=uE_opt); maybe_check()
    elif protocol == "simul":               # strict 1:1 simultaneous GDA
        while gstep < TOTAL:
            gP, gE = grads_from_shared()
            optP.step(pP, gP, sign=-1.0); optE.step(pE, gE, sign=-1.0); maybe_check()
    elif protocol == "abr":                 # alternating best-response blocks
        M = 250
        while gstep < TOTAL:
            for _ in range(M):
                if gstep >= TOTAL: break
                step("P"); maybe_check()
            for _ in range(M):
                if gstep >= TOTAL: break
                step("E"); maybe_check()

    gap, cap, dmin = measure()
    tail = [t[1] for t in trace[len(trace)//2:]]          # 2nd-half gap trace
    res = {"protocol": protocol, "seed": seed, "iters": TOTAL,
           "n_capturable": int(capturable.sum()),
           "final_cap_of_capturable": round(cap,3),
           "final_gap_capturable": round(gap,4),
           "osc_sd_2ndhalf": round(float(np.std(tail)),4),
           "trace": trace}
    os.makedirs(OUT, exist_ok=True)
    print(json.dumps(res), flush=True)

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
