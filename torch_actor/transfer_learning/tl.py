"""Transfer learning (TL) of new heads on a frozen multi-head actor.

The trained actor of a player is a body H(t) = tanh(B phi(t) + b) (one or more tanh layers) shared
by all heads, and one *linear, bias-free* head per initial condition: z^j(t) = W^j H(t),
u^j = z^j / |z^j|.  TL = load the trained actor of both players, freeze both bodies (no
gradient reaches B, b, B2, ...), and train one new head W^j (2 x n_b) per new IC with the same
loss (capture-truncated by default) and the same self-play protocol (pursuer warm-up, 3:1 Adam
steps, Adam resets), gradients taken with respect to the new heads only.  The 100 new ICs form a
uniform grid on the flat torus, the pursuer starts at the origin as in training.

Usage:  python3 torch_actor/transfer_learning/tl.py [--ckpt paper/ckpt/baseline_trunc/grid_i0_c0.npz]
            [--grid 10] [--seed 0] [--tag D1_grid100_seed0] [--no-warmup] [--time-only]
Writes transfer_learning/results/<tag>.json (game statistics of every new head, ensemble statistics
of the new heads, timing), transfer_learning/results/<tag>.npz (the new heads and the frozen bodies),
transfer_learning/figs/<tag>_game.pdf and <tag>_torus_map.pdf.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "checks"))
import pe_torch as pt  # noqa: E402
from core import RunConfig  # noqa: E402
import metrics as M  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_E, C_P = "#2a78d6", "#eb6834"
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figs")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def ic_grid(n: int, L: float) -> np.ndarray:
    """n x n uniform grid on [0, L)^2, cell-centred so no point sits on the pursuer (origin) or the cut locus."""
    c = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def frozen_actor(p_np: dict, n_new: int, rng: np.random.Generator, device, dtype):
    """Batched (R = 1) parameters: the body as constants, n_new fresh heads with gradients."""
    p = {k: torch.tensor(np.asarray(v, np.float64)[None], device=device, dtype=dtype) for k, v in p_np.items()}
    n_b = p_np["B"].shape[0]
    W = rng.standard_normal((n_new, 2, n_b)) * np.sqrt(1.0 / n_b)        # the draw of init_params_np
    p["W"] = torch.tensor(W[None], device=device, dtype=dtype, requires_grad=True)
    return p


def transfer(pE_np, pP_np, x0_new, cfg: pt.Config, seed: int, device="cpu", dtype=torch.float64, log_every=0):
    """Train new heads of both players on frozen bodies; returns the new heads (NumPy) and the wall time."""
    rng = np.random.default_rng(seed)
    pE = frozen_actor(pE_np, len(x0_new), rng, device, dtype)
    pP = frozen_actor(pP_np, len(x0_new), rng, device, dtype)
    x0 = torch.tensor(x0_new[None], device=device, dtype=dtype)
    t = torch.linspace(0.0, cfg.t_max, cfg.n_t + 1, device=device, dtype=dtype)
    phi = pt.fourier_features(t, cfg.t_max, cfg.K)
    heads = lambda p: {"W": p["W"]}                                          # noqa: E731  Adam over the heads only
    optE, optP = pt.make_adam(heads(pE), cfg.lr_E), pt.make_adam(heads(pP), cfg.lr_P)
    t0 = time.time()
    for _ in range(cfg.warmup):
        pt.step("P", pE, pP, x0, phi, cfg, optE, optP)
    for it in range(1, cfg.n_iters + 1):
        if cfg.adam_reset_every and it > 1 and (it - 1) % cfg.adam_reset_every == 0:
            optE, optP = pt.make_adam(heads(pE), cfg.lr_E), pt.make_adam(heads(pP), cfg.lr_P)
        for _ in range(cfg.n_P_steps):
            pt.step("P", pE, pP, x0, phi, cfg, optE, optP)
        for _ in range(cfg.n_E_steps):
            loss = pt.step("E", pE, pP, x0, phi, cfg, optE, optP)
        if log_every and it % log_every == 0:
            print(f"  it {it:6d}  L_E {loss.item():+.4f}  [{time.time() - t0:.0f}s]", flush=True)
    for p in (pE, pP):
        assert all(not v.requires_grad for k, v in p.items() if k != "W"), "a body parameter carries gradients"
    return pt.to_numpy(pE)["W"][0], pt.to_numpy(pP)["W"][0], time.time() - t0


def legacy(p_np: dict, W: np.ndarray) -> dict:
    d = {"B": p_np["B"], "b_body": p_np["b"], "W": W}
    d.update({k: p_np[k] for k in p_np if k not in ("B", "b", "W")})
    return d


def figure_game(g: dict, cfg: RunConfig, path: str, label: str):
    d0, Tc, Tbr = np.array(g["d0"]), np.array([np.nan if t is None else t for t in g["Tc"]]), \
        np.array([np.nan if t is None else t for t in g["T_BR"]])
    cap, capt = np.array(g["capturable"]), np.array(g["captured"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    s_pt = 14 if len(d0) <= 200 else 5
    ax.scatter(d0, Tc, s=s_pt, color=C_P, alpha=0.7, label="TL heads: $T_c$", linewidths=0)
    ax.scatter(d0, Tbr, s=s_pt, color=C_E, alpha=0.7, label="best response $T_{BR}$", linewidths=0)
    xx = np.linspace(cfg.eps, cfg.reach, 50); ax.plot(xx, (xx - cfg.eps) / cfg.dv, "k-", lw=1.2, label=r"$V^*(\Delta_0)$")
    ax.axvline(cfg.reach, color="0.5", lw=1); ax.text(cfg.reach + 0.02, 1.7, "reach R", fontsize=8, color="0.4")
    ax.set_ylim(0, cfg.t_max + 0.1); ax.set_xlim(0.0, 1.45)
    ax.set_xlabel(r"initial distance $|\Delta_0|$"); ax.set_ylabel("capture time")
    ax.set_title(f"{label}: capture times of the new heads", fontsize=9.5); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    fr = [capt[cap].mean(), (~np.isnan(Tbr[cap])).mean(), capt[~cap].mean() if (~cap).any() else 0, (~np.isnan(Tbr[~cap])).mean() if (~cap).any() else 0]
    x = np.array([0, 0.38, 1.1, 1.48])
    bars = ax.bar(x, fr, width=0.36, color=[C_P, C_E, C_P, C_E])
    for b, v in zip(bars, fr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{100 * v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks([0.19, 1.29]); ax.set_xticklabels([f"capturable\n(n={int(cap.sum())})", f"uncapturable\n(n={int((~cap).sum())})"])
    ax.set_ylim(0, 1.12); ax.set_ylabel("fraction of heads captured"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("TL pursuer heads (orange) vs best response (blue)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)


CORNER = {(0, 0): "#1f77b4", (1, 0): "#ff7f0e", (0, 1): "#9467bd", (1, 1): "#4b2e1e"}   # image of the pursuer -> colour
CORNER_NAME = {(0, 0): "(0, 0)", (1, 0): "(L, 0)", (0, 1): "(0, L)", (1, 1): "(L, L)"}


def chased_corner(chart) -> tuple:
    """metrics.py charts are integer images c with min-image displacement E - (P - cL); the pursuer image chased
    is therefore at -cL, i.e. the corner (0 or L) per axis."""
    return tuple(int(-c) for c in chart)


def figure_torus_map(x0: np.ndarray, g: dict, cfg: RunConfig, path: str, label: str, s_pt: float = 70):
    """The flat torus with one point per new IC: red = captured by its TL pursuer head, green = escaped.
    Dotted circles: the capturable region (geodesic distance <= R from the pursuer's images);
    dashed lines: the cut locus of the origin (x = L/2, y = L/2).  Capturable heads that escaped get
    a ring coloured by the periodic image of the pursuer that its head chased (the closest image at the
    moment of closest approach), one colour per corner; the corner markers carry the same colours."""
    L, R = cfg.L, cfg.reach
    cap, capt = np.array(g["capturable"]), np.array(g["captured"])
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    th = np.linspace(0, 2 * np.pi, 200)
    for cx in (0, L):
        for cy in (0, L):
            ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), color="0.55", lw=1.0, ls=":")
    ax.axvline(L / 2, color="0.75", lw=0.9, ls="--"); ax.axhline(L / 2, color="0.75", lw=0.9, ls="--")
    ax.scatter(x0[capt, 0], x0[capt, 1], s=s_pt, color="#d62728", label=f"captured ({int(capt.sum())})", zorder=3, linewidths=0)
    ax.scatter(x0[~capt, 0], x0[~capt, 1], s=s_pt, color="#2ca02c", label=f"escaped ({int((~capt).sum())})", zorder=3, linewidths=0)
    wrong = np.where(cap & ~capt)[0]
    corners = [chased_corner(g["chart_closest"][j]) for j in wrong]
    for c in CORNER:
        idx = [j for j, cj in zip(wrong, corners) if cj == c]
        if idx:
            ax.scatter(x0[idx, 0], x0[idx, 1], s=2.4 * s_pt, facecolor="none", edgecolor=CORNER[c], lw=1.4, zorder=4,
                       label=f"capturable, escaped: chased the image at {CORNER_NAME[c]} ({len(idx)})")
    wrong2 = ~cap & capt
    if wrong2.any():
        ax.scatter(x0[wrong2, 0], x0[wrong2, 1], s=2.4 * s_pt, facecolor="none", edgecolor="k", lw=1.3, ls="--",
                   label=f"uncapturable but captured ({int(wrong2.sum())})", zorder=4)
    for (cx, cy), col in CORNER.items():
        ax.plot(cx * L, cy * L, "s", color=col, ms=9, zorder=5, clip_on=False)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    ax.set_xlabel("x  (x = 0 and x = L identified)"); ax.set_ylabel("y  (y = 0 and y = L identified)")
    ax.set_title(f"{label}: outcome per new IC on the flat torus ({len(x0)} heads, {int(cap.sum())} capturable)\n"
                 f"dotted = capturable region (geodesic distance <= R = {R:.2f}); dashed = cut locus; "
                 f"squares = pursuer start and its images", fontsize=9)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "paper", "ckpt", "baseline_trunc", "grid_i0_c0.npz"),
                    help="trained actor whose bodies are frozen (default: the 1-layer truncated baseline, init 0 / IC 0)")
    ap.add_argument("--grid", type=int, default=10, help="n x n uniform grid of new ICs on the torus")
    ap.add_argument("--seed", type=int, default=0, help="seed of the new heads' initialisation")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-truncate", action="store_true", help="use the draft's untruncated distance term")
    ap.add_argument("--no-warmup", action="store_true", help="skip the 1,500-step pursuer-head warm-up")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--dtype", default="float64")
    ap.add_argument("--time-only", action="store_true", help="only time the TL of the grid, write nothing")
    ap.add_argument("--log-every", type=int, default=0)
    a = ap.parse_args()
    tag = a.tag or f"D1_grid{a.grid * a.grid}_seed{a.seed}"
    dtype = {"float64": torch.float64, "float32": torch.float32}[a.dtype]

    pE_np, pP_np, x0_train = pt.load_ckpt(a.ckpt)                 # final state (iteration 14,000)
    depth = 1 + len(pt.extra_layers(pE_np))
    cfg = pt.Config(truncate=not a.no_truncate, depth=depth)
    if a.no_warmup:
        cfg = replace(cfg, warmup=0)
    x0_new = ic_grid(a.grid, cfg.L)
    print(f"frozen actor: {a.ckpt} ({depth}-layer body, {x0_train.shape[0]} training heads); "
          f"{len(x0_new)} new heads on a {a.grid}x{a.grid} grid; loss {'truncated' if cfg.truncate else 'draft'}; "
          f"protocol warm-up {cfg.warmup}, {cfg.n_iters} x ({cfg.n_P_steps} P + {cfg.n_E_steps} E), lr {cfg.lr_P} / {cfg.lr_E}; "
          f"{a.device} {a.dtype}", flush=True)
    WE, WP, secs = transfer(pE_np, pP_np, x0_new, cfg, a.seed, a.device, dtype, a.log_every)
    n_steps = cfg.warmup + cfg.n_iters * (cfg.n_P_steps + cfg.n_E_steps)
    print(f"TL of {len(x0_new)} heads: {secs:.1f} s wall ({n_steps} gradient steps, {1e3 * secs / n_steps:.2f} ms per step)", flush=True)
    if a.time_only:
        return

    rcfg = RunConfig(truncate=cfg.truncate) if "truncate" in RunConfig.__dataclass_fields__ else RunConfig()
    pE_l, pP_l = legacy(pE_np, WE), legacy(pP_np, WP)
    full = M.full_metrics(pE_l, pP_l, x0_new, rcfg)
    g = full["game"]
    cap, capt = np.array(g["capturable"]), np.array(g["captured"])
    Tc, Vs = np.array([np.nan if t is None else t for t in g["Tc"]]), np.array(g["Vstar"])
    wrong_chart = int(sum(1 for j, c in enumerate(cap) if c and g["chart_closest"][j] != g["chart_opt"][j]))
    summary = {
        "n_new_heads": int(len(x0_new)), "n_capturable": int(cap.sum()),
        "capture_rate_capturable": float(capt[cap].mean()),
        "capture_rate_uncapturable": float(capt[~cap].mean()) if (~cap).any() else None,
        "n_capturable_escaped": int((cap & ~capt).sum()), "n_uncapturable_captured": int((~cap & capt).sum()),
        "Tc_over_Vstar_captured_median_q1_q3": [float(np.median(Tc[cap & capt] / Vs[cap & capt])),
                                                float(np.percentile(Tc[cap & capt] / Vs[cap & capt], 25)),
                                                float(np.percentile(Tc[cap & capt] / Vs[cap & capt], 75))],
        "nonnearest_chart_capturable": wrong_chart,
        "best_response_capture_rate_capturable": float(np.mean([t is not None for t, c in zip(g["T_BR"], cap) if c])),
        "gram_pr_E": full["E"]["ens"]["gram_pr"], "gram_pr_P": full["P"]["ens"]["gram_pr"],
        "constancy_E": full["E"]["ens"]["constancy"], "constancy_P": full["P"]["ens"]["constancy"],
        "tpr_U_E": full["E"]["ens"]["tpr_U"], "tpr_U_P": full["P"]["ens"]["tpr_U"],
        "pr_H_E": full["E"]["lat"]["pr_H"], "pr_H_P": full["P"]["lat"]["pr_H"],
        "pr_H_unc_E": full["E"]["lat"]["pr_H_unc"], "pr_H_unc_P": full["P"]["lat"]["pr_H_unc"],
        "seconds": secs, "ms_per_gradient_step": 1e3 * secs / n_steps,
    }
    rec = {"tag": tag, "ckpt": os.path.relpath(a.ckpt, ROOT), "seed": a.seed, "grid": a.grid,
           "config": asdict(cfg), "x0_new": x0_new.tolist(), "x0_train": x0_train.tolist(),
           "summary": summary, "full": full, "device": f"{a.device} {a.dtype}"}
    with open(os.path.join(RES, tag + ".json"), "w") as f:
        json.dump(rec, f, indent=1)
    np.savez_compressed(os.path.join(RES, tag + ".npz"), x_E0=x0_new, WE=WE, WP=WP,
                        **{f"{k}E": v for k, v in pE_np.items() if k != "W"}, **{f"{k}P": v for k, v in pP_np.items() if k != "W"})
    figure_game(g, rcfg, os.path.join(FIG, tag + "_game"), tag)
    figure_torus_map(x0_new, g, rcfg, os.path.join(FIG, tag + "_torus_map"), tag, s_pt=70 if len(x0_new) <= 200 else 11)
    print(json.dumps(summary, indent=1))
    print("wrote", os.path.join(RES, tag + ".json/.npz"), "and", os.path.join(FIG, tag + "_{game,torus_map}.pdf"))


if __name__ == "__main__":
    main()
