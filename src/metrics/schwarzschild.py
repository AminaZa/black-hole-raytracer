"""Schwarzschild metric in (t, r, theta, phi) spherical coordinates.

Geometrized units throughout: G = c = 1. The Schwarzschild radius is rs = 2M.

    ds² = -(1 - rs/r) dt² + (1 - rs/r)^(-1) dr² + r² dθ² + r² sin²θ dφ²
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from utils.cuda_loader import cupy as _cupy, get_xp


if _cupy is not None:

    @_cupy.fuse(kernel_name="schwarzschild_accel_fused")
    def _schwarzschild_accel_fused(  # type: ignore[no-untyped-def]
        r, theta, pt, pr, pth, pph, rs, M
    ):
        """Element-wise Schwarzschild acceleration, JIT-fused into one CUDA kernel."""
        sin_t = _cupy.sin(theta)
        cos_t = _cupy.cos(theta)
        f = 1.0 - rs / r
        inv_r = 1.0 / r
        r_minus_rs = r - rs

        gamma_t_tr = M / (r * r_minus_rs)
        gamma_r_tt = (M / (r * r)) * f

        dpt = -2.0 * gamma_t_tr * pt * pr
        dpr = -(
            gamma_r_tt * pt * pt
            - gamma_t_tr * pr * pr
            - r_minus_rs * pth * pth
            - r_minus_rs * sin_t * sin_t * pph * pph
        )
        dpth = -(2.0 * inv_r * pr * pth - sin_t * cos_t * pph * pph)
        cot_t = cos_t / sin_t
        dpph = -2.0 * (inv_r * pr * pph + cot_t * pth * pph)
        return dpt, dpr, dpth, dpph

else:
    _schwarzschild_accel_fused = None  # type: ignore[assignment]


class SchwarzschildMetric:
    """Static, spherically symmetric vacuum spacetime around a point mass.

    Parameters
    ----------
    mass:
        Black hole mass M in geometrized units. Must be positive.
    """

    def __init__(self, mass: float = 1.0) -> None:
        if mass <= 0.0:
            raise ValueError(f"mass must be positive, got {mass}.")
        self.mass: float = float(mass)
        self.rs: float = 2.0 * self.mass

    def metric_tensor(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Covariant metric components g_{μν} at the given event.

        Parameters
        ----------
        position:
            Shape (4,), spacetime coordinates (t, r, θ, φ).

        Returns
        -------
        g:
            Shape (4, 4), diagonal in these coordinates.
        """
        r: float = float(position[1])
        theta: float = float(position[2])
        rs: float = self.rs

        f: float = 1.0 - rs / r
        sin_theta: float = float(np.sin(theta))

        g: NDArray[np.float64] = np.zeros((4, 4), dtype=np.float64)
        g[0, 0] = -f
        g[1, 1] = 1.0 / f
        g[2, 2] = r * r
        g[3, 3] = r * r * sin_theta * sin_theta
        return g

    def christoffel_symbols(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Connection coefficients Γ^μ_{αβ} at the given event.

        Indexing: ``gamma[mu, alpha, beta] = Γ^μ_{αβ}``. The result is symmetric
        in the lower pair of indices.

        Parameters
        ----------
        position:
            Shape (4,), coordinates (t, r, θ, φ). The t coordinate is unused
            (the metric is static).

        Returns
        -------
        gamma:
            Shape (4, 4, 4).
        """
        r: float = float(position[1])
        theta: float = float(position[2])
        rs: float = self.rs
        M: float = self.mass

        sin_theta: float = float(np.sin(theta))
        cos_theta: float = float(np.cos(theta))

        gamma: NDArray[np.float64] = np.zeros((4, 4, 4), dtype=np.float64)

        f: float = 1.0 - rs / r
        r_minus_rs: float = r - rs

        # Γ^t_{tr} = Γ^t_{rt} = M / (r (r - rs))
        gamma_t_tr: float = M / (r * r_minus_rs)
        gamma[0, 0, 1] = gamma_t_tr
        gamma[0, 1, 0] = gamma_t_tr

        # Γ^r_{tt} = (M / r²)(1 - rs/r)
        gamma[1, 0, 0] = (M / (r * r)) * f
        # Γ^r_{rr} = -M / (r (r - rs))
        gamma[1, 1, 1] = -gamma_t_tr
        # Γ^r_{θθ} = -(r - rs)
        gamma[1, 2, 2] = -r_minus_rs
        # Γ^r_{φφ} = -(r - rs) sin²θ
        gamma[1, 3, 3] = -r_minus_rs * sin_theta * sin_theta

        # Γ^θ_{rθ} = Γ^θ_{θr} = 1/r
        inv_r: float = 1.0 / r
        gamma[2, 1, 2] = inv_r
        gamma[2, 2, 1] = inv_r
        # Γ^θ_{φφ} = -sinθ cosθ
        gamma[2, 3, 3] = -sin_theta * cos_theta

        # Γ^φ_{rφ} = Γ^φ_{φr} = 1/r
        gamma[3, 1, 3] = inv_r
        gamma[3, 3, 1] = inv_r
        # Γ^φ_{θφ} = Γ^φ_{φθ} = cot θ
        cot_theta: float = cos_theta / sin_theta
        gamma[3, 2, 3] = cot_theta
        gamma[3, 3, 2] = cot_theta

        return gamma

    def acceleration(
        self,
        position: NDArray[np.float64],
        momentum: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Closed-form ``dp^μ/dλ = -Γ^μ_{αβ} p^α p^β`` for Schwarzschild.

        Equivalent to ``-einsum('mab,a,b->m', christoffel_symbols(x), p, p)`` but
        skips the (4, 4, 4) tensor allocation and the einsum dispatch by inlining
        the eleven non-zero connection coefficients directly. This is the
        per-step hot path in the integrator.
        """
        r: float = float(position[1])
        theta: float = float(position[2])
        rs: float = self.rs
        M: float = self.mass

        pt: float = float(momentum[0])
        pr: float = float(momentum[1])
        pth: float = float(momentum[2])
        pph: float = float(momentum[3])

        sin_t: float = float(np.sin(theta))
        cos_t: float = float(np.cos(theta))
        f: float = 1.0 - rs / r
        inv_r: float = 1.0 / r
        r_minus_rs: float = r - rs

        gamma_t_tr: float = M / (r * r_minus_rs)
        gamma_r_tt: float = (M / (r * r)) * f

        dpt: float = -2.0 * gamma_t_tr * pt * pr
        dpr: float = -(
            gamma_r_tt * pt * pt
            - gamma_t_tr * pr * pr
            - r_minus_rs * pth * pth
            - r_minus_rs * sin_t * sin_t * pph * pph
        )
        dpth: float = -(
            2.0 * inv_r * pr * pth
            - sin_t * cos_t * pph * pph
        )
        cot_t: float = cos_t / sin_t
        dpph: float = -2.0 * (inv_r * pr * pph + cot_t * pth * pph)

        return np.array([dpt, dpr, dpth, dpph], dtype=np.float64)

    def acceleration_batch(
        self,
        positions: NDArray[np.float64],
        momenta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Vectorised acceleration for an array of rays, shape (N, 4).

        On cupy inputs the inner math is dispatched to a JIT-fused kernel
        (one CUDA launch instead of ~15). On numpy the same arithmetic
        runs directly. Output array module always matches the input.
        """
        xp = get_xp(positions)
        r = positions[:, 1]
        theta = positions[:, 2]
        pt = momenta[:, 0]
        pr = momenta[:, 1]
        pth = momenta[:, 2]
        pph = momenta[:, 3]

        if _cupy is not None and xp is _cupy:
            dpt, dpr, dpth, dpph = _schwarzschild_accel_fused(
                r, theta, pt, pr, pth, pph, self.rs, self.mass
            )
        else:
            rs = self.rs
            M = self.mass
            sin_t = xp.sin(theta)
            cos_t = xp.cos(theta)
            f = 1.0 - rs / r
            inv_r = 1.0 / r
            r_minus_rs = r - rs
            gamma_t_tr = M / (r * r_minus_rs)
            gamma_r_tt = (M / (r * r)) * f
            dpt = -2.0 * gamma_t_tr * pt * pr
            dpr = -(
                gamma_r_tt * pt * pt
                - gamma_t_tr * pr * pr
                - r_minus_rs * pth * pth
                - r_minus_rs * sin_t * sin_t * pph * pph
            )
            dpth = -(2.0 * inv_r * pr * pth - sin_t * cos_t * pph * pph)
            cot_t = cos_t / sin_t
            dpph = -2.0 * (inv_r * pr * pph + cot_t * pth * pph)

        return xp.stack([dpt, dpr, dpth, dpph], axis=1)
