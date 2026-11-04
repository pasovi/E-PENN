"""
Geometries for pursuit-evasion: disk, 3D ball, Möbius strip.

Each Geometry exposes the operations needed by the honest critic
training: sampling, displacement, distance, boundary handling, and
input encoding.  The HJI equation in each case is

    ∂_τ V(x, τ) = C(x) + v_E ||∇_{x_E} V|| - v_P ||∇_{x_P} V||

with the boundary conditions
    V(x, 0) = 0   (terminal),
    V(x, τ) = 0   for ||x_E - x_P||_geom ≤ ε   (capture).

The bounded domains (disk, ball) break the translation invariance of
the torus problem: V depends on the absolute positions of both players,
not just their displacement.  The Möbius strip has the extra subtlety
that the minimum-image distance involves an orientation flip.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


# =============================================================================
# Base interface
# =============================================================================
class Geometry:
    """Common interface."""
    d_pos: int      # spatial dimensionality of a single player's position
    name: str

    def state_dim(self) -> int:
        """Dimensionality of the joint state (x_E, x_P)."""
        return 2 * self.d_pos

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project a position to a valid domain point (after a tentative step)."""
        raise NotImplementedError

    def sample_positions(self, rng: np.random.Generator, n: int,
                          min_dist_from_origin: float = 0.0) -> np.ndarray:
        """Sample n positions inside the domain."""
        raise NotImplementedError

    def distance(self, x_E: np.ndarray, x_P: np.ndarray) -> np.ndarray:
        """Geodesic distance(s) used for capture detection."""
        raise NotImplementedError

    def encode_state(self, x_E: np.ndarray, x_P: np.ndarray,
                      tau: np.ndarray) -> np.ndarray:
        """Combined feature vector (N, D_features) for the critic."""
        raise NotImplementedError

    def encoded_dim(self) -> int:
        raise NotImplementedError

    def fd_step(self) -> float:
        """Finite-difference step for ∇V."""
        return 5e-3


# =============================================================================
# Disk:  D = { x ∈ R^2 : ||x|| < R }
# =============================================================================
class Disk(Geometry):
    name = "disk"
    d_pos = 2

    def __init__(self, R: float = 1.0, n_freqs: int = 4, t_max: float = 4.0):
        self.R = R
        self.n_freqs = n_freqs   # polynomial feature degree
        self.t_max = t_max

    def project(self, x: np.ndarray) -> np.ndarray:
        # project to disk by radial clipping
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        scale = np.minimum(1.0, (self.R - 1e-3) / np.clip(r, 1e-12, None))
        return x * scale

    def sample_positions(self, rng, n: int,
                          min_dist_from_origin: float = 0.0) -> np.ndarray:
        # rejection sample (rare rejection if min_dist is small)
        pts = np.empty((n, 2))
        cnt = 0
        while cnt < n:
            x = rng.uniform(-self.R, self.R, size=(n, 2))
            keep = (np.linalg.norm(x, axis=-1) <= self.R - 1e-3) & \
                   (np.linalg.norm(x, axis=-1) >= min_dist_from_origin)
            x = x[keep]
            take = min(len(x), n - cnt)
            pts[cnt:cnt + take] = x[:take]
            cnt += take
        return pts

    def distance(self, x_E, x_P):
        return np.linalg.norm(x_E - x_P, axis=-1)

    def encode_state(self, x_E, x_P, tau):
        # raw positions (scaled to [-1, 1]) + radial features + polynomial of tau
        xe = x_E / self.R
        xp = x_P / self.R
        re = np.linalg.norm(xe, axis=-1, keepdims=True)
        rp = np.linalg.norm(xp, axis=-1, keepdims=True)
        diff = xe - xp
        diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True)
        tt = tau[:, None] / self.t_max
        feats = [
            xe, xp,                  # 4
            xe * xe, xp * xp,        # 4 (radial)
            diff, diff_norm,         # 3
            re, rp,                  # 2
            tt, tt ** 2,             # 2
            np.sin(np.pi * tt),      # 1
            np.cos(np.pi * tt),      # 1
        ]
        return np.concatenate(feats, axis=1)

    def encoded_dim(self):
        return 4 + 4 + 3 + 2 + 4

    def fd_step(self):
        return 5e-3


# =============================================================================
# Ball:  B = { x ∈ R^3 : ||x|| < R }
# =============================================================================
class Ball3D(Geometry):
    name = "ball"
    d_pos = 3

    def __init__(self, R: float = 1.0, t_max: float = 4.0):
        self.R = R
        self.t_max = t_max

    def project(self, x):
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        scale = np.minimum(1.0, (self.R - 1e-3) / np.clip(r, 1e-12, None))
        return x * scale

    def sample_positions(self, rng, n, min_dist_from_origin=0.0):
        pts = np.empty((n, 3)); cnt = 0
        while cnt < n:
            x = rng.uniform(-self.R, self.R, size=(n, 3))
            r = np.linalg.norm(x, axis=-1)
            keep = (r <= self.R - 1e-3) & (r >= min_dist_from_origin)
            x = x[keep]
            take = min(len(x), n - cnt)
            pts[cnt:cnt + take] = x[:take]; cnt += take
        return pts

    def distance(self, x_E, x_P):
        return np.linalg.norm(x_E - x_P, axis=-1)

    def encode_state(self, x_E, x_P, tau):
        xe = x_E / self.R; xp = x_P / self.R
        re = np.linalg.norm(xe, axis=-1, keepdims=True)
        rp = np.linalg.norm(xp, axis=-1, keepdims=True)
        diff = xe - xp
        diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True)
        tt = tau[:, None] / self.t_max
        feats = [
            xe, xp,                # 6
            xe * xe, xp * xp,      # 6
            diff, diff_norm,       # 4
            re, rp,                # 2
            tt, tt ** 2,           # 2
            np.sin(np.pi * tt),    # 1
            np.cos(np.pi * tt),    # 1
        ]
        return np.concatenate(feats, axis=1)

    def encoded_dim(self):
        return 6 + 6 + 4 + 2 + 4

    def fd_step(self):
        return 5e-3


# =============================================================================
# Möbius strip:  parameterised by (u, v) ∈ [-L/2, L/2) × [-w, w]
#   with identification (u, v) ~ (u + L, -v).
# =============================================================================
class Mobius(Geometry):
    name = "mobius"
    d_pos = 2

    def __init__(self, L: float = 2.0, w: float = 0.5,
                  n_freqs: int = 4, t_max: float = 4.0):
        self.L = L
        self.w = w
        self.n_freqs = n_freqs
        self.t_max = t_max

    def project(self, x: np.ndarray) -> np.ndarray:
        """Normalise (u, v) to the strip, applying the orientation flip."""
        u, v = x[..., 0], x[..., 1]
        # wrap u by L using floor division; track flips by parity
        n_wraps = np.floor((u + self.L / 2) / self.L)
        u_new = u - n_wraps * self.L
        # each full wrap flips v
        v_new = v * np.where(np.mod(n_wraps, 2) == 0, 1.0, -1.0)
        # clip v to strip width (reflective walls at v = ±w)
        v_new = np.clip(v_new, -self.w + 1e-3, self.w - 1e-3)
        return np.stack([u_new, v_new], axis=-1)

    def sample_positions(self, rng, n, min_dist_from_origin=0.0):
        pts = np.empty((n, 2)); cnt = 0
        while cnt < n:
            u = rng.uniform(-self.L / 2 + 0.02, self.L / 2 - 0.02, n)
            v = rng.uniform(-self.w + 0.02, self.w - 0.02, n)
            x = np.stack([u, v], axis=1)
            r = np.linalg.norm(x, axis=-1)
            keep = r >= min_dist_from_origin
            x = x[keep]
            take = min(len(x), n - cnt)
            pts[cnt:cnt + take] = x[:take]; cnt += take
        return pts

    def distance(self, x_E: np.ndarray, x_P: np.ndarray) -> np.ndarray:
        """
        Geodesic distance.  Compute both the straight displacement and
        the once-around-with-flip displacement; return the smaller norm.
        """
        # straight
        du = x_E[..., 0] - x_P[..., 0]
        dv = x_E[..., 1] - x_P[..., 1]
        du = (du + self.L / 2) % self.L - self.L / 2
        d_straight = np.sqrt(du ** 2 + dv ** 2)
        # flipped (go once around): u shifts by ±L, v flips sign
        dv_flip = x_E[..., 1] + x_P[..., 1]   # because v_E becomes -v_E after wrap
        d_flip = np.sqrt(du ** 2 + dv_flip ** 2)
        return np.minimum(d_straight, d_flip)

    def encode_state(self, x_E, x_P, tau):
        """Encode (u_E, v_E, u_P, v_P) with periodicity in u and the
        Möbius identification.

        For each player encode:
            sin(2π k u / L), cos(2π k u / L)   (k = 1..n_freqs/2)  --  periodic in u
            sin(π k v / w),  cos(π k v / w)    (k = 1..n_freqs/2)  --  oscillation in v
            v / w, (v/w)^2                                          --  smooth in v
        Plus tau features.
        """
        K = self.n_freqs
        outs = []
        for x in (x_E, x_P):
            u, v = x[..., 0:1], x[..., 1:2]
            for k in range(1, K + 1):
                outs.append(np.sin(2 * np.pi * k * u / self.L))
                outs.append(np.cos(2 * np.pi * k * u / self.L))
            for k in range(1, K // 2 + 1):
                outs.append(np.sin(np.pi * k * v / self.w))
                outs.append(np.cos(np.pi * k * v / self.w))
            outs.append(v / self.w)
            outs.append((v / self.w) ** 2)
        # cross feature: sin/cos of (u_E - u_P) -- emphasises the relative
        du = (x_E[..., 0:1] - x_P[..., 0:1])
        for k in range(1, K + 1):
            outs.append(np.sin(2 * np.pi * k * du / self.L))
            outs.append(np.cos(2 * np.pi * k * du / self.L))
        # tau
        tt = tau[:, None] / self.t_max
        outs += [tt, tt ** 2, np.sin(np.pi * tt), np.cos(np.pi * tt)]
        return np.concatenate(outs, axis=1)

    def encoded_dim(self):
        K = self.n_freqs
        per_player = 2 * K + 2 * (K // 2) + 2
        return 2 * per_player + 2 * K + 4

    def fd_step(self):
        return 5e-3
