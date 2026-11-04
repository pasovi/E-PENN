"""
C4 and C5: the two outstanding committed runs (referee report M4, M1-residual).

C4  Capturability-matched sector restriction (corrected design).
    The baseline displacement multiset (pursuer at origin, evader ICs =
    sample_evader_ICs(16, L=2, min_dist=0.4, seed=0)) is ROTATED into a
    +-15 deg sector about the DIAGONAL: radii are the exact baseline
    radius multiset (so the 4/16 capturable split at reach 0.88 is
    preserved by construction); angles are 45 deg + U(-15, 15) deg
    (seed 0).  Along the diagonal the min-image chart supports distances
    up to sqrt(2); max baseline radius 1.134 -> max component
    1.134*cos(30 deg) = 0.982 < L/2, so every IC stays inside the chart.
    Monitored side effect (per the committed spec): proximity of the far
    ICs to the antipodal ridge, reported as margin = L/2 - max|Delta_i|
    per IC in the analysis JSON.

C5  lambda_d in {0, 1, 5} from-scratch sweep (S 'shaping_dep').  Full
    protocol at the arm's lambda_d from initialisation on (including the
    pursuer-only warm-up); lambda_d = 5 reproduces the baseline protocol
    and doubles as a from-scratch replication.

Protocol otherwise identical to the baseline / C1 (warm-up 1500
pursuer-only steps, 3:1 P:E adversarial steps, Adam lr 1e-2 / 1e-3,
lam_nc = 2, lam_orth absent, seed 0 init, 14,000 adversarial iters to
match the E1/E7 comparison points).  Config-only: no loss-code changes,
so the grad_check gate does not trigger.  No V*/u* anywhere.

Usage:
  python3 run_c4c5.py train c4
  python3 run_c4c5.py train c5_lam0 | c5_lam1 | c5_lam5
  python3 run_c4c5.py analyze          (writes revision/c4c5_analysis.json)
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
                                   backward_dirs, rollout, backward_rollout,
                                   forward_loss, backward_loss, init_params,
                                   sample_evader_ICs, Adam)

MAX_ITERS = 14000
cfg = TrainConfig(n_h=16, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
                  L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
                  no_capture_penalty=2.0, lam_d=5.0, seed=0)
tg = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
phig = fourier_features(tg, cfg.t_max, cfg.n_freqs)

BASELINE_ICS = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.4, seed=0)


def c4_ICs() -> np.ndarray:
    """Baseline radius multiset, angles in the +-15 deg diagonal sector."""
    d = (BASELINE_ICS + cfg.L / 2) % cfg.L - cfg.L / 2   # min-image displacement
    r = np.linalg.norm(d, axis=1)                        # exact baseline radii
    rng = np.random.default_rng(0)
    ang = np.pi / 4 + rng.uniform(-np.pi / 12, np.pi / 12, size=cfg.n_h)
    pts = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    assert np.abs(pts).max() < cfg.L / 2, "IC left the min-image chart"
    assert np.allclose(np.sort(np.linalg.norm(pts, axis=1)), np.sort(r))
    return pts % cfg.L


ARMS = {
    "c4":      {"ics": c4_ICs,               "lam_d": 5.0},
    "c5_lam0": {"ics": lambda: BASELINE_ICS, "lam_d": 0.0},
    "c5_lam1": {"ics": lambda: BASELINE_ICS, "lam_d": 1.0},
    "c5_lam5": {"ics": lambda: BASELINE_ICS, "lam_d": 5.0},
}


def train(arm: str):
    spec = ARMS[arm]
    lam_d = spec["lam_d"]
    x_E0 = spec["ics"]()
    ckpt = os.path.join(HERE, f"{arm}_actor.npz")
    rng = np.random.default_rng(cfg.seed)
    D = 2 * cfg.n_freqs + 1
    pE = init_params(cfg.n_h, cfg.n_b, D, rng)
    pP = init_params(cfg.n_h, cfg.n_b, D, rng)
    it0, warm = 0, False
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
        pP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
        x_E0 = z["x_E0"]; it0 = int(z["iter0"]); warm = True
        print(f"[{arm}] resumed at {it0}")
    if it0 >= MAX_ITERS:
        print(f"[{arm}] DONE at {it0}"); return
    optE, optP = Adam(pE, lr=1e-3), Adam(pP, lr=1e-2)

    def step(who):
        uE, cE = forward_dirs(pE, phig); uP, cP = forward_dirs(pP, phig)
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P,
                         cfg.t_max, cfg.n_t)
        L_E, L_P, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau,
                                    cfg.t_max, cfg.n_t,
                                    cfg.no_capture_penalty, lam_d=lam_d)
        is_p = who == "P"
        dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty,
                                 cfg.t_max, is_pursuer=is_p)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                    cfg.t_max, cfg.n_t)
        if is_p:
            optP.step(pP, backward_dirs(pP, cP, duP), sign=-1.0)
        else:
            optE.step(pE, backward_dirs(pE, cE, duE), sign=-1.0)
        return L_E, L_P, float((cL["S"][:, -1] > 0.5).mean())

    t0 = time.time()
    if not warm:
        for i in range(1500):
            _, LP, _ = step("P")
            if i % 500 == 0:
                print(f"[{arm} warmup] {i} L_P={LP:.3f}")
    it = it0
    while it < MAX_ITERS:
        for _ in range(3):
            step("P")
        L_E, L_P, nc = step("E")
        it += 1
        if it % 2000 == 0:
            print(f"[{arm}] it {it} lam_d={lam_d} L_P={L_P:.3f} "
                  f"L_E={L_E:.3f} no_cap={nc:.2f} [{time.time()-t0:.0f}s]")
            np.savez(ckpt, BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
                     BP=pP["B"], bP=pP["b_body"], WP=pP["W"],
                     x_E0=x_E0, iter0=it)
    np.savez(ckpt, BE=pE["B"], bE=pE["b_body"], WE=pE["W"],
             BP=pP["B"], bP=pP["b_body"], WP=pP["W"], x_E0=x_E0, iter0=it)
    print(f"[{arm}] saved at {it} ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------- analysis
# Conventions identical to revision/analyze_e1e7.py (dense grid T = 1024).
T = 1024
td = np.linspace(0, cfg.t_max, T)
phid = fourier_features(td, cfg.t_max, cfg.n_freqs)


def PR(ev):
    ev = np.maximum(ev, 0)
    return float(ev.sum()**2 / max((ev**2).sum(), 1e-300))
def cov_eigs(X):
    Xc = X - X.mean(0, keepdims=True)
    return np.linalg.eigvalsh(Xc.T @ Xc / X.shape[0])[::-1]
def pcs_for(ev, thr):
    return int(np.searchsorted(np.cumsum(ev)/ev.sum(), thr) + 1)
def unit(z):
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


def full_suite(pE, pP, x0):
    n_h = pE["W"].shape[0]
    _, cE = forward_dirs(pE, phid); HE = cE["H"]
    _, cP = forward_dirs(pP, phid); HP = cP["H"]
    zE = np.einsum("hab,tb->tha", pE["W"], HE)
    zP = np.einsum("hab,tb->tha", pP["W"], HP)
    evHE, evHP = cov_eigs(HE), cov_eigs(HP)
    evZE = cov_eigs(zE.reshape(T, -1)); evZP = cov_eigs(zP.reshape(T, -1))
    uEg, _ = forward_dirs(pE, phig); uPg, _ = forward_dirs(pP, phig)
    xE, xP = rollout(uEg, uPg, x0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    dt = cfg.t_max / cfg.n_t
    dispE = dt * cfg.v_E * (0.5 * (uEg[:, :-1] + uEg[:, 1:])).sum(1)
    sec = [tuple(s) for s in np.round(dispE / cfg.L).astype(int)]
    d = np.linalg.norm((xE - xP + cfg.L/2) % cfg.L - cfg.L/2, axis=-1)
    def bandfrac(z):
        X = unit(z).reshape(T, -1); X = X - X.mean(0, keepdims=True)
        P = (np.abs(np.fft.rfft(X, axis=0))**2).sum(1)
        return float(P[11:51].sum() / P[1:].sum())
    def gf_svpr(p):
        Wg = np.stack([w/np.linalg.norm(w) for w in p["W"]]).reshape(-1, p["W"].shape[2])
        s2 = np.linalg.svd(Wg, compute_uv=False)**2
        return PR(s2)
    return {
        "PR_HE": PR(evHE), "PR_HP": PR(evHP),
        "pc95_HE": pcs_for(evHE, .95), "pc95_HP": pcs_for(evHP, .95),
        "PR_ZE": PR(evZE), "PR_ZP": PR(evZP),
        "pc95_ZE": pcs_for(evZE, .95), "pc95_ZP": pcs_for(evZP, .95),
        "n_sectors": len(set(sec)), "sectors": [str(s) for s in sec],
        "n_captured": int((d.min(1) <= cfg.eps).sum()), "n_h": n_h,
        "midband_frac_uE": bandfrac(zE), "midband_frac_uP": bandfrac(zP),
        "gf_svPR_E": gf_svpr(pE), "gf_svPR_P": gf_svpr(pP),
    }


def analyze():
    OUT = {}
    for arm in ARMS:
        f = os.path.join(HERE, f"{arm}_actor.npz")
        if not os.path.exists(f):
            print(f"[analyze] missing {f}, skipping"); continue
        z = np.load(f)
        pE = {"B": z["BE"], "b_body": z["bE"], "W": z["WE"]}
        pP = {"B": z["BP"], "b_body": z["bP"], "W": z["WP"]}
        OUT[arm] = full_suite(pE, pP, z["x_E0"])
        OUT[arm]["iters"] = int(z["iter0"])
    # C4 design record + ridge-proximity monitor (committed side effect)
    if "c4" in OUT:
        z = np.load(os.path.join(HERE, "c4_actor.npz"))
        d = (z["x_E0"] + cfg.L/2) % cfg.L - cfg.L/2
        r = np.linalg.norm(d, axis=1)
        OUT["c4_design"] = {
            "radii": [round(float(v), 4) for v in np.sort(r)],
            "n_capturable_r_le_0.88": int((r <= 0.88).sum()),
            "angles_deg": [round(float(a), 2) for a in
                           np.degrees(np.arctan2(d[:, 1], d[:, 0]))],
            "ridge_margin_per_IC": [round(float(v), 4) for v in
                                    np.sort(cfg.L/2 - np.abs(d).max(1))],
        }
    out_path = os.path.join(HERE, "revision", "c4c5_analysis.json")
    json.dump(OUT, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in OUT.items() if k != "c4_design"
                      and isinstance(v, dict)
                      and not k.endswith("sectors")}, indent=1, default=str)[:2000])
    print("written:", out_path)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2])
    elif sys.argv[1] == "analyze":
        analyze()
