"""Parity of the torch port against the hand-written NumPy implementation.

Checks, in float64 on CPU, starting from the published baseline's initial state
(init seed 0, IC seed 0):
  1. loss values L_E, L_P agree;
  2. autograd parameter gradients agree with the NumPy backward pass, for a
     pursuer step and for an evader step;
  3. after N identical self-play iterations (warm-up included) the parameters
     of both implementations agree, and how the gap grows with N
     (adversarial training is chaotic, so bit-for-bit agreement over a full
     run is not expected; see the README).
Usage: python3 torch_actor/test_parity.py [n_iters_to_follow]
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy_reference as ref                      # noqa: E402  (verbatim original)
import pe_torch as pt                              # noqa: E402

torch.set_default_dtype(torch.float64)
cfg = pt.Config()
N_FOLLOW = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ---------------------------------------------------------------- state
pE_np, pP_np, x0_np = pt.initial_state_np(cfg, 0, 0)
# NumPy reference keeps its own key names
pE_r = {"B": pE_np["B"].copy(), "b_body": pE_np["b"].copy(), "W": pE_np["W"].copy()}
pP_r = {"B": pP_np["B"].copy(), "b_body": pP_np["b"].copy(), "W": pP_np["W"].copy()}
t_np = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
phi_np = ref.fourier_features(t_np, cfg.t_max, cfg.K)

pE_t, pP_t, x0_t = pt.stack_runs([(pE_np, pP_np, x0_np)], "cpu", torch.float64)
phi_t = pt.fourier_features(torch.tensor(t_np), cfg.t_max, cfg.K)
print(f"fourier features      rel err {rel(phi_t.numpy(), phi_np):.1e}")


def numpy_forward_backward(who):
    uE, cE = ref.forward_dirs(pE_r, phi_np)
    uP, cP = ref.forward_dirs(pP_r, phi_np)
    xE, xP = ref.rollout(uE, uP, x0_np, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    LE, LP, cL = ref.forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max, cfg.n_t,
                                  cfg.lam_nc, lam_d=cfg.lam_d)
    is_p = who == "P"
    dxE, dxP = ref.backward_loss(1.0, cL, cfg.n_h, cfg.lam_nc, cfg.t_max, is_pursuer=is_p)
    duE, duP = ref.backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    g = ref.backward_dirs(pP_r, cP, duP) if is_p else ref.backward_dirs(pE_r, cE, duE)
    return LE, LP, {"B": g["B"], "b": g["b_body"], "W": g["W"]}


def torch_forward_backward(who):
    for p in (pE_t, pP_t):
        for v in p.values():
            v.grad = None
    uE, _, _ = pt.forward_dirs(pE_t, phi_t)
    uP, _, _ = pt.forward_dirs(pP_t, phi_t)
    xE, xP = pt.rollout(uE, uP, x0_t, cfg)
    LE, LP, _, _ = pt.game_loss(xE, xP, cfg)
    (LP if who == "P" else LE).sum().backward()
    p = pP_t if who == "P" else pE_t
    return LE.item(), LP.item(), {k: v.grad[0].numpy() for k, v in p.items()}


# ------------------------------------------------- 1 + 2: loss and gradients
ok = True
for who in ("P", "E"):
    LE_n, LP_n, g_n = numpy_forward_backward(who)
    LE_t, LP_t, g_t = torch_forward_backward(who)
    print(f"[{who} step] L_E {LE_n:+.10f} vs {LE_t:+.10f} (rel {rel(LE_t, LE_n):.1e});  "
          f"L_P {LP_n:+.10f} vs {LP_t:+.10f} (rel {rel(LP_t, LP_n):.1e})")
    for k in ("B", "b", "W"):
        e = rel(g_t[k], g_n[k])
        ok &= e < 1e-9
        print(f"          grad {k:2s} rel err {e:.1e}")

# ---------------------------------------- 3: follow the training trajectory
optE_r, optP_r = ref.Adam(pE_r, lr=cfg.lr_E), ref.Adam(pP_r, lr=cfg.lr_P)
optE_t, optP_t = pt.make_adam(pE_t, cfg.lr_E), pt.make_adam(pP_t, cfg.lr_P)


def ref_step(who):
    uE, cE = ref.forward_dirs(pE_r, phi_np)
    uP, cP = ref.forward_dirs(pP_r, phi_np)
    xE, xP = ref.rollout(uE, uP, x0_np, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    _, _, cL = ref.forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max, cfg.n_t,
                                cfg.lam_nc, lam_d=cfg.lam_d)
    is_p = who == "P"
    dxE, dxP = ref.backward_loss(1.0, cL, cfg.n_h, cfg.lam_nc, cfg.t_max, is_pursuer=is_p)
    duE, duP = ref.backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    if is_p:
        optP_r.step(pP_r, ref.backward_dirs(pP_r, cP, duP), sign=-1.0)
    else:
        optE_r.step(pE_r, ref.backward_dirs(pE_r, cE, duE), sign=-1.0)


def gap():
    return max(rel(pE_t["B"].detach()[0].numpy(), pE_r["B"]),
               rel(pE_t["W"].detach()[0].numpy(), pE_r["W"]),
               rel(pP_t["B"].detach()[0].numpy(), pP_r["B"]),
               rel(pP_t["W"].detach()[0].numpy(), pP_r["W"]))


warm = min(cfg.warmup, 200)
for _ in range(warm):
    ref_step("P")
    pt.step("P", pE_t, pP_t, x0_t, phi_t, cfg, optE_t, optP_t)
print(f"after {warm} warm-up steps          max rel param gap {gap():.1e}")
report_at = sorted({10, 50, 100, 200, 500, 1000, 2000, N_FOLLOW} & set(range(1, N_FOLLOW + 1)))
for it in range(1, N_FOLLOW + 1):
    for _ in range(cfg.n_P_steps):
        ref_step("P")
        pt.step("P", pE_t, pP_t, x0_t, phi_t, cfg, optE_t, optP_t)
    for _ in range(cfg.n_E_steps):
        ref_step("E")
        pt.step("E", pE_t, pP_t, x0_t, phi_t, cfg, optE_t, optP_t)
    if it in report_at:
        print(f"after {it:5d} self-play iterations  max rel param gap {gap():.1e}")

# ---------------- reference for the drift: NumPy vs NumPy with a 1e-15 init perturbation
pE_a, pP_a, _ = pt.initial_state_np(cfg, 0, 0)
pE_b, pP_b, _ = pt.initial_state_np(cfg, 0, 0, perturb_seed=1, perturb_scale=1e-15)
pairs = []
for pE_, pP_ in ((pE_a, pP_a), (pE_b, pP_b)):
    pE_ = {"B": pE_["B"], "b_body": pE_["b"], "W": pE_["W"]}
    pP_ = {"B": pP_["B"], "b_body": pP_["b"], "W": pP_["W"]}
    pairs.append((pE_, pP_, ref.Adam(pE_, lr=cfg.lr_E), ref.Adam(pP_, lr=cfg.lr_P)))


def ref_step_on(pE_, pP_, oE, oP, who):
    uE, cE = ref.forward_dirs(pE_, phi_np)
    uP, cP = ref.forward_dirs(pP_, phi_np)
    xE, xP = ref.rollout(uE, uP, x0_np, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    _, _, cL = ref.forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max, cfg.n_t,
                                cfg.lam_nc, lam_d=cfg.lam_d)
    is_p = who == "P"
    dxE, dxP = ref.backward_loss(1.0, cL, cfg.n_h, cfg.lam_nc, cfg.t_max, is_pursuer=is_p)
    duE, duP = ref.backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    if is_p:
        oP.step(pP_, ref.backward_dirs(pP_, cP, duP), sign=-1.0)
    else:
        oE.step(pE_, ref.backward_dirs(pE_, cE, duE), sign=-1.0)


for _ in range(warm):
    for pr in pairs:
        ref_step_on(*pr, "P")
for it in range(1, N_FOLLOW + 1):
    for _ in range(cfg.n_P_steps):
        for pr in pairs:
            ref_step_on(*pr, "P")
    for pr in pairs:
        ref_step_on(*pr, "E")
    if it in report_at:
        g = max(rel(pairs[0][0]["B"], pairs[1][0]["B"]), rel(pairs[0][1]["B"], pairs[1][1]["B"]))
        print(f"  (reference: numpy vs numpy+1e-15 init perturbation, {it:5d} iters: gap {g:.1e})")

print("PARITY OK" if ok else "PARITY FAILED (gradient mismatch > 1e-9)")
sys.exit(0 if ok else 1)
