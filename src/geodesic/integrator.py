"""Null geodesic integrator using classical RK4.

Solves the geodesic equation rewritten as a first-order system:

    dx^μ/dλ = p^μ
    dp^μ/dλ = -Γ^μ_{αβ} p^α p^β

The integrator is metric-agnostic: it accepts any object exposing ``rs`` and
``christoffel_symbols(position)``, so swapping a different metric (e.g. Kerr)
requires no changes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class MetricProtocol(Protocol):
    """Interface required by the integrator."""

    rs: float

    def christoffel_symbols(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the (4, 4, 4) array Γ^μ_{αβ} at ``position``."""
        ...


@dataclass
class GeodesicResult:
    """Outcome of a single geodesic integration.

    Attributes
    ----------
    termination:
        One of ``"horizon"``, ``"escape"``, ``"disk"``, ``"max_steps"``.
    final_position:
        Shape (4,). Spacetime coordinates at termination.
    final_momentum:
        Shape (4,). 4-momentum at termination.
    affine_lambda:
        Total affine parameter elapsed.
    n_steps:
        Number of RK4 steps executed.
    trajectory:
        Shape (n_steps + 1, 4) when storage was requested, else ``None``.
    """

    termination: str
    final_position: NDArray[np.float64]
    final_momentum: NDArray[np.float64]
    affine_lambda: float
    n_steps: int
    trajectory: NDArray[np.float64] | None = None


class GeodesicIntegrator:
    """RK4 integrator for null (or general) geodesics in any metric.

    Parameters
    ----------
    metric:
        Spacetime metric satisfying ``MetricProtocol``.
    r_max:
        Radius beyond which a ray is treated as having reached infinity.
    horizon_eps:
        Radial buffer above ``rs``; integration stops at ``r <= rs + horizon_eps``
        to avoid the coordinate singularity.
    base_step:
        Affine-parameter step used in the far field.
    near_field_factor:
        Step is multiplied by this factor inside ``r < 5 rs`` to resolve the
        rapidly varying connection there.
    """

    def __init__(
        self,
        metric: MetricProtocol,
        r_max: float = 100.0,
        horizon_eps: float = 1e-3,
        base_step: float = 0.2,
        near_field_factor: float = 0.1,
    ) -> None:
        self.metric: MetricProtocol = metric
        self.r_max: float = float(r_max)
        self.horizon_eps: float = float(horizon_eps)
        self.base_step: float = float(base_step)
        self.near_field_factor: float = float(near_field_factor)

    def _step_size(self, r: float) -> float:
        """Adaptive affine-parameter step size based on radial distance."""
        if r < 5.0 * self.metric.rs:
            return self.base_step * self.near_field_factor
        return self.base_step

    def _accel(
        self, position: NDArray[np.float64], momentum: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute dp^μ/dλ = -Γ^μ_{αβ} p^α p^β at the given state."""
        gamma: NDArray[np.float64] = self.metric.christoffel_symbols(position)
        return -np.einsum("mab,a,b->m", gamma, momentum, momentum)

    def integrate(
        self,
        position: NDArray[np.float64],
        momentum: NDArray[np.float64],
        max_steps: int = 5000,
        disk_inner: float | None = None,
        disk_outer: float | None = None,
        store_trajectory: bool = False,
    ) -> GeodesicResult:
        """Trace a single geodesic from the given initial state.

        Parameters
        ----------
        position:
            Shape (4,). Initial spacetime coordinates (t, r, θ, φ).
        momentum:
            Shape (4,). Initial 4-momentum.
        max_steps:
            Hard upper bound on RK4 iterations.
        disk_inner, disk_outer:
            If both are provided, integration terminates when the trajectory
            crosses θ = π/2 with ``disk_inner <= r <= disk_outer``. Equatorial
            crossings outside this annulus do not terminate.
        store_trajectory:
            When True, every position is recorded. Off by default to keep
            per-ray memory low during rendering.
        """
        rs: float = self.metric.rs
        check_disk: bool = disk_inner is not None and disk_outer is not None

        x: NDArray[np.float64] = position.astype(np.float64, copy=True)
        p: NDArray[np.float64] = momentum.astype(np.float64, copy=True)
        lam: float = 0.0
        termination: str = "max_steps"

        traj: list[NDArray[np.float64]] | None = [x.copy()] if store_trajectory else None

        steps_taken: int = 0
        for _ in range(max_steps):
            r: float = float(x[1])

            if r <= rs + self.horizon_eps:
                termination = "horizon"
                break
            if r >= self.r_max:
                termination = "escape"
                break

            h: float = self._step_size(r)

            k1x: NDArray[np.float64] = p
            k1p: NDArray[np.float64] = self._accel(x, p)

            x2: NDArray[np.float64] = x + 0.5 * h * k1x
            p2: NDArray[np.float64] = p + 0.5 * h * k1p
            k2x: NDArray[np.float64] = p2
            k2p: NDArray[np.float64] = self._accel(x2, p2)

            x3: NDArray[np.float64] = x + 0.5 * h * k2x
            p3: NDArray[np.float64] = p + 0.5 * h * k2p
            k3x: NDArray[np.float64] = p3
            k3p: NDArray[np.float64] = self._accel(x3, p3)

            x4: NDArray[np.float64] = x + h * k3x
            p4: NDArray[np.float64] = p + h * k3p
            k4x: NDArray[np.float64] = p4
            k4p: NDArray[np.float64] = self._accel(x4, p4)

            x_next: NDArray[np.float64] = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            p_next: NDArray[np.float64] = p + (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)

            theta_prev: float = float(x[2])
            theta_next: float = float(x_next[2])
            half_pi: float = 0.5 * np.pi

            crossed: bool = (theta_prev - half_pi) * (theta_next - half_pi) < 0.0
            if check_disk and crossed:
                alpha: float = (half_pi - theta_prev) / (theta_next - theta_prev)
                x_cross: NDArray[np.float64] = x + alpha * (x_next - x)
                p_cross: NDArray[np.float64] = p + alpha * (p_next - p)
                r_cross: float = float(x_cross[1])
                # disk_inner/disk_outer are non-None when check_disk is True.
                assert disk_inner is not None and disk_outer is not None
                if disk_inner <= r_cross <= disk_outer:
                    lam += alpha * h
                    steps_taken += 1
                    if traj is not None:
                        traj.append(x_cross.copy())
                    return GeodesicResult(
                        termination="disk",
                        final_position=x_cross,
                        final_momentum=p_cross,
                        affine_lambda=lam,
                        n_steps=steps_taken,
                        trajectory=(np.array(traj) if traj is not None else None),
                    )

            x = x_next
            p = p_next
            lam += h
            steps_taken += 1
            if traj is not None:
                traj.append(x.copy())

        return GeodesicResult(
            termination=termination,
            final_position=x,
            final_momentum=p,
            affine_lambda=lam,
            n_steps=steps_taken,
            trajectory=(np.array(traj) if traj is not None else None),
        )
