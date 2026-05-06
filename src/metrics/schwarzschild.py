"""Schwarzschild metric in (t, r, theta, phi) spherical coordinates.

Geometrized units throughout: G = c = 1. The Schwarzschild radius is rs = 2M.

    ds² = -(1 - rs/r) dt² + (1 - rs/r)^(-1) dr² + r² dθ² + r² sin²θ dφ²
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
