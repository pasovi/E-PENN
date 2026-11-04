"""
Embedding analysis of the trained adversarial actor on the torus
pursuit--evasion game.

This is the analogue, for differential games, of the PDE-embedding
analysis of Jimenez et al.  We extract the body latent representations
H_E(t) and H_P(t) from the trained multi-head actor, study the PCA of
their centred covariance matrices, perform a Fourier-shell
decomposition of the spectral content of each principal component, and
visualise the dominant modes in both the latent and physical (heading-
angle) spaces.

The actor checkpoint that we analyse was trained by pure adversarial
self-play (no V*-supervision), so the embedding interpretation is
clean: whatever structure we find in H_E(t), H_P(t) is what the
two-network minimax game put there.
"""
from __future__ import annotations
import os, json
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import pursuit_evasion_numpy as pen
from pursuit_evasion_numpy import (
    TrainConfig, fourier_features, forward_dirs, rollout,
)


# =============================================================================
# Loading the v1 adversarial actor checkpoint (no V*-supervision in its training)
# =============================================================================
def load_actor(ckpt_path: str):
    npz = np.load(ckpt_path)
    paramsE = {"B": npz["BE"], "b_body": npz["bE"], "W": npz["WE"]}
    paramsP = {"B": npz["BP"], "b_body": npz["bP"], "W": npz["WP"]}
    x_E0 = npz["x_E0"]
    iters = int(npz["iter0"])
    return paramsE, paramsP, x_E0, iters


# =============================================================================
# Latent extraction
# =============================================================================
def extract_latents(params, cfg: TrainConfig, n_grid: int = 1024):
    """Forward H_alpha(t) on a dense uniform grid of t."""
    t = np.linspace(0.0, cfg.t_max, n_grid)
    phi = fourier_features(t, cfg.t_max, cfg.n_freqs)
    _, cache = forward_dirs(params, phi)
    return t, cache["H"]   # (T, n_b)


# =============================================================================
# Centred covariance + PCA
# =============================================================================
def pca_of_latents(H: np.ndarray):
    """
    Centre H, compute covariance, return (eigenvalues, eigenvectors, mu).
    Sorted in decreasing order.
    """
    mu = H.mean(axis=0, keepdims=True)
    Hc = H - mu
    C = Hc.T @ Hc / Hc.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)        # ascending
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx], mu


def effective_dimension(eigvals: np.ndarray, threshold: float) -> int:
    """Smallest k such that the leading k eigenvalues sum to >= threshold."""
    total = eigvals.sum()
    cum = np.cumsum(eigvals) / total
    return int(np.searchsorted(cum, threshold) + 1)


def participation_ratio(eigvals: np.ndarray) -> float:
    """PR = (sum lambda_i)^2 / sum lambda_i^2 -- effective number of modes."""
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    return float(s1 ** 2 / max(s2, 1e-12))


# =============================================================================
# Project H onto the leading-k principal modes and reconstruct
# =============================================================================
def reconstruct_with_k(H: np.ndarray, eigvecs: np.ndarray, mu: np.ndarray,
                       k: int) -> np.ndarray:
    Hc = H - mu
    V_k = eigvecs[:, :k]
    coords = Hc @ V_k                  # (T, k)
    return mu + coords @ V_k.T         # (T, n_b)


def reconstruction_error(H: np.ndarray, eigvecs: np.ndarray, mu: np.ndarray,
                          k_max: int) -> np.ndarray:
    err = np.zeros(k_max + 1)
    err[0] = float(((H - mu) ** 2).sum())
    for k in range(1, k_max + 1):
        Hk = reconstruct_with_k(H, eigvecs, mu, k)
        err[k] = float(((H - Hk) ** 2).sum())
    return err / err[0]


# =============================================================================
# Fourier-shell decomposition of the latent covariance (Jimenez-style)
# =============================================================================
def fourier_shell_decomposition(H: np.ndarray, n_shells: int = 8):
    """
    1D DFT of H along the time axis, then split frequencies into n_shells
    log-spaced bands.  For each shell, compute the per-mode power
    P_i(Q) and the full Q-shelled covariance trace tr(C^Q_alpha).
    Returns shell-band edges and the n_shells x n_b matrix
    P_iQ[shell, i] = sum over k in shell of |hat H_i(k)|^2 / T.
    """
    T, n_b = H.shape
    # subtract mean
    Hc = H - H.mean(0, keepdims=True)
    # 1D real DFT along time axis
    Htilde = np.fft.rfft(Hc, axis=0)              # (T//2 + 1, n_b)
    K = Htilde.shape[0]
    # frequency indices (skip the dc bin since centred)
    log_edges = np.linspace(0, np.log(K - 1), n_shells + 1)
    edges = np.exp(log_edges).astype(int)
    edges[0] = 1   # skip the dc
    edges = np.unique(edges)
    n_real = len(edges) - 1
    P_iQ = np.zeros((n_real, n_b))
    centres = np.zeros(n_real)
    for s in range(n_real):
        lo, hi = edges[s], edges[s + 1]
        if lo >= hi:
            continue
        # Parseval-normalised power
        P_iQ[s] = (np.abs(Htilde[lo:hi]) ** 2).sum(axis=0) / T ** 2
        centres[s] = 0.5 * (lo + hi)
    return centres, edges, P_iQ


def shell_weights_for_pc(P_iQ: np.ndarray, eigvecs: np.ndarray,
                          k: int) -> np.ndarray:
    """
    rho_n(Q) = v_n^T C^Q v_n where v_n is the n-th PC.  We approximate
    by projecting the per-mode shell power onto the PC:
        rho_n(Q) ≈ sum_i (v_n^i)^2 P_iQ[Q, i].
    (Strict equality holds when the latent components are orthogonal
    in the spatial direction; here we use this as a proxy.)
    """
    out = P_iQ @ (eigvecs[:, k] ** 2)
    return out


# =============================================================================
# Stability under input-noise perturbation
# =============================================================================
def latent_stability(params: dict, cfg: TrainConfig, n_grid: int,
                      seeds=(0, 1, 2, 3, 4), noise: float = 0.01):
    """
    Perturb the input phi by additive Gaussian noise and measure how
    the principal-direction eigenvalues change.  This is a quick proxy
    for cross-seed PCA-spectrum stability used in Jimenez et al.; a
    cleaner check would require re-training under different random
    initialisations, which is too expensive in this sandbox.
    """
    t = np.linspace(0.0, cfg.t_max, n_grid)
    phi = fourier_features(t, cfg.t_max, cfg.n_freqs)
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        phi_noisy = phi + noise * rng.standard_normal(phi.shape)
        _, cache = forward_dirs(params, phi_noisy)
        eigvals, _, _ = pca_of_latents(cache["H"])
        rows.append(eigvals[:6] / eigvals.sum())
    return np.array(rows)


# =============================================================================
# Connection to the honest critic (state-conditioned)
# =============================================================================
def head_weight_singular_values(params: dict) -> np.ndarray:
    """
    Stack the per-head matrices W^j in R^{2 x n_b} into a single
    (2 N_h, n_b) matrix and return its singular values.  Near-orthogonal
    heads have all singular values clustered near a common value.
    """
    W = params["W"]                       # (n_h, 2, n_b)
    n_h, _, n_b = W.shape
    flat = W.reshape(2 * n_h, n_b)
    return np.linalg.svd(flat, compute_uv=False)


# =============================================================================
# Plotters
# =============================================================================
def plot_pca_spectrum(eigvals_E, eigvals_P, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, (ev, lab, col) in enumerate(((eigvals_E, "Evader", "tab:blue"),
                                          (eigvals_P, "Pursuer", "tab:red"))):
        norm = ev / ev.sum()
        cum = np.cumsum(norm)
        ax[0].bar(np.arange(1, len(norm) + 1) + 0.2 * (2 * j - 1), norm,
                  width=0.4, alpha=0.7, color=col, label=lab)
        ax[1].plot(np.arange(1, len(cum) + 1), cum, "o-", color=col,
                    label=lab)
    ax[0].set_xlabel("PC index"); ax[0].set_ylabel("normalised eigenvalue")
    ax[0].set_title("Variance explained per principal mode")
    ax[0].legend()
    ax[1].set_xlabel("PC index"); ax[1].set_ylabel("cumulative variance")
    ax[1].axhline(0.95, color="k", ls="--", lw=0.7,
                  label="95% threshold")
    ax[1].set_ylim(0, 1.05); ax[1].set_title("Cumulative explained variance")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_pc_time_profiles(t, H, eigvecs, mu, path, k_show: int = 4,
                           title: str = ""):
    """Plot the first k principal directions as functions of time."""
    Hc = H - mu
    coords = Hc @ eigvecs[:, :k_show]           # (T, k)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for k in range(k_show):
        ax.plot(t, coords[:, k], label=f"PC {k+1}", lw=1.5)
    ax.set_xlabel("t"); ax.set_ylabel("PC projection")
    ax.set_title(title or "Principal-mode time profiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_reconstruction(err_E, err_P, path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(np.arange(len(err_E)), err_E, "o-", color="tab:blue",
                label="Evader")
    ax.semilogy(np.arange(len(err_P)), err_P, "s-", color="tab:red",
                label="Pursuer")
    ax.axhline(0.05, color="k", ls="--", lw=0.7,
               label="5% relative residual")
    ax.set_xlabel("number of PCs"); ax.set_ylabel("residual energy ratio")
    ax.set_title("Latent reconstruction by leading-$k$ PCs")
    ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_fourier_shells(P_iQ_E, P_iQ_P, edges, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for jj, (P_iQ, lab) in enumerate(((P_iQ_E, "Evader"),
                                       (P_iQ_P, "Pursuer"))):
        # Plot the total shell power summed over latent components
        shell_power = P_iQ.sum(axis=1)
        shell_centres = 0.5 * (edges[:-1] + edges[1:])
        ax[jj].loglog(shell_centres, shell_power, "o-",
                       color="tab:blue" if jj == 0 else "tab:red")
        ax[jj].set_xlabel("Fourier index (time)")
        ax[jj].set_ylabel("shell power")
        ax[jj].set_title(f"{lab}: latent-space spectral profile")
        ax[jj].grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_heading_modes(t, params, cfg, eigvecs, mu, path, k_show=3,
                        head_idx=0):
    """
    Show how the leading PCs of H_E translate into heading directions for
    one IC.  At the leading order each PC is a linear modulation of the
    raw output W^j H(t), so the heading-angle perturbation for PC k is
    approximately atan2(W^j v_k).
    """
    phi = fourier_features(t, cfg.t_max, cfg.n_freqs)
    _, cache = forward_dirs(params, phi)
    H = cache["H"]
    Hc = H - mu
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    raw_zero = params["W"][head_idx] @ mu[0]            # baseline z
    base_angle = np.arctan2(raw_zero[1], raw_zero[0])
    # baseline trajectory of heading
    z_full = np.einsum("ab,tb->ta", params["W"][head_idx], H)
    angle_full = np.unwrap(np.arctan2(z_full[:, 1], z_full[:, 0]))
    ax.plot(t, angle_full, label="full", color="black", lw=2)
    for k in range(k_show):
        z_k = np.einsum("ab,tb->ta",
                        params["W"][head_idx],
                        mu + (Hc @ eigvecs[:, k: k + 1]) @ eigvecs[:, k: k + 1].T)
        a_k = np.unwrap(np.arctan2(z_k[:, 1], z_k[:, 0]))
        ax.plot(t, a_k, label=f"PC {k+1} only", lw=1.3, alpha=0.8)
    ax.set_xlabel("t"); ax.set_ylabel(r"$\theta(t)$ [rad]")
    ax.set_title(f"Heading reconstructed from individual PCs (IC {head_idx})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_stability(stab_E, stab_P, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, (stab, lab, col) in enumerate(((stab_E, "Evader", "tab:blue"),
                                            (stab_P, "Pursuer", "tab:red"))):
        m = stab.mean(0); s = stab.std(0)
        ax[j].bar(np.arange(1, m.size + 1), m,
                   yerr=s, capsize=4, color=col, alpha=0.7,
                   label="mean $\\pm$ std over 5 noise seeds")
        ax[j].set_xlabel("PC index"); ax[j].set_ylabel("normalised eigenvalue")
        ax[j].set_title(f"{lab}: PCA stability under 1% input noise")
        ax[j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_head_orthogonality(params_E, params_P, path):
    sigE = head_weight_singular_values(params_E)
    sigP = head_weight_singular_values(params_P)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(np.arange(1, sigE.size + 1), sigE, "o-", color="tab:blue",
            label="Evader heads")
    ax.plot(np.arange(1, sigP.size + 1), sigP, "s-", color="tab:red",
            label="Pursuer heads")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("singular value of $W$")
    ax.set_title("Head-matrix singular values (orthogonality diagnostic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


# =============================================================================
# Driver
# =============================================================================
def run_full_analysis(actor_ckpt: str, out_dir: str):
    pE, pP, x_E0, iters = load_actor(actor_ckpt)
    cfg = TrainConfig(n_h=12, n_b=16, n_freqs=12, t_max=4.0, n_t=64,
                       L=2.0, v_E=1.0, v_P=1.2, eps=0.08, tau=5e-3,
                       no_capture_penalty=2.0, lam_d=5.0, seed=0)
    t, H_E = extract_latents(pE, cfg, n_grid=1024)
    _, H_P = extract_latents(pP, cfg, n_grid=1024)

    ev_E, vec_E, mu_E = pca_of_latents(H_E)
    ev_P, vec_P, mu_P = pca_of_latents(H_P)

    summary = {
        "checkpoint_iters": iters,
        "evader_pc_95": effective_dimension(ev_E, 0.95),
        "evader_pc_99": effective_dimension(ev_E, 0.99),
        "evader_participation_ratio": participation_ratio(ev_E),
        "pursuer_pc_95": effective_dimension(ev_P, 0.95),
        "pursuer_pc_99": effective_dimension(ev_P, 0.99),
        "pursuer_participation_ratio": participation_ratio(ev_P),
        "n_b_nominal": int(cfg.n_b),
    }
    print(json.dumps(summary, indent=2))

    # plots
    plot_pca_spectrum(ev_E, ev_P, os.path.join(out_dir,
                       "fig_embed_pca_spectrum.png"))
    plot_pc_time_profiles(t, H_E, vec_E, mu_E,
                          os.path.join(out_dir, "fig_embed_pc_time_E.png"),
                          k_show=4, title="Evader PC time profiles")
    plot_pc_time_profiles(t, H_P, vec_P, mu_P,
                          os.path.join(out_dir, "fig_embed_pc_time_P.png"),
                          k_show=4, title="Pursuer PC time profiles")
    err_E = reconstruction_error(H_E, vec_E, mu_E, k_max=cfg.n_b)
    err_P = reconstruction_error(H_P, vec_P, mu_P, k_max=cfg.n_b)
    plot_reconstruction(err_E, err_P,
                         os.path.join(out_dir, "fig_embed_reconstruction.png"))

    _, edges_E, P_iQ_E = fourier_shell_decomposition(H_E, n_shells=8)
    _, edges_P, P_iQ_P = fourier_shell_decomposition(H_P, n_shells=8)
    plot_fourier_shells(P_iQ_E, P_iQ_P, edges_E,
                         os.path.join(out_dir, "fig_embed_fourier_shells.png"))

    plot_heading_modes(t, pE, cfg, vec_E, mu_E,
                       os.path.join(out_dir, "fig_embed_headings_E.png"),
                       k_show=3, head_idx=0)
    plot_heading_modes(t, pP, cfg, vec_P, mu_P,
                       os.path.join(out_dir, "fig_embed_headings_P.png"),
                       k_show=3, head_idx=0)

    stab_E = latent_stability(pE, cfg, n_grid=512)
    stab_P = latent_stability(pP, cfg, n_grid=512)
    plot_stability(stab_E, stab_P,
                    os.path.join(out_dir, "fig_embed_stability.png"))

    plot_head_orthogonality(pE, pP,
                             os.path.join(out_dir, "fig_embed_head_ortho.png"))

    with open(os.path.join(out_dir, "embedding_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    run_full_analysis(os.path.join(here, "v2_actor.npz"), here)
