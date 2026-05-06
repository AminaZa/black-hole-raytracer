"""Null geodesic integrator using classical RK4.

Solves the geodesic equation rewritten as a first-order system:

    dx^μ/dλ = p^μ
    dp^μ/dλ = -Γ^μ_{αβ} p^α p^β

The integrator is metric-agnostic: it accepts any object exposing ``rs`` and
``christoffel_symbols(position)``. If the metric also exposes
``acceleration(position, momentum)``, the integrator uses that closed-form
fast path instead of building the full Christoffel tensor each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

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


AccelFn = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


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
        Affine-parameter step used in the medium band ``3 rs <= r <= 10 rs``.
    near_field_factor:
        Step is multiplied by this factor when ``r < 3 rs`` to resolve the
        rapidly varying connection near the photon sphere.
    far_field_factor:
        Step is multiplied by this factor when ``r > 10 rs`` to skip through
        nearly-flat regions where photon paths are almost straight.
    """

    def __init__(
        self,
        metric: MetricProtocol,
        r_max: float = 100.0,
        horizon_eps: float = 1e-3,
        base_step: float = 0.2,
        near_field_factor: float = 0.1,
        far_field_factor: float = 2.0,
    ) -> None:
        self.metric: MetricProtocol = metric
        self.r_max: float = float(r_max)
        self.horizon_eps: float = float(horizon_eps)
        self.base_step: float = float(base_step)
        self.near_field_factor: float = float(near_field_factor)
        self.far_field_factor: float = float(far_field_factor)
        self._has_fast_accel: bool = hasattr(metric, "acceleration")

    def _step_size(self, r: float) -> float:
        """Adaptive affine-parameter step size based on radial distance."""
        rs = self.metric.rs
        if r < 3.0 * rs:
            return self.base_step * self.near_field_factor
        if r > 10.0 * rs:
            return self.base_step * self.far_field_factor
        return self.base_step

    def _accel(
        self, position: NDArray[np.float64], momentum: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute dp^μ/dλ = -Γ^μ_{αβ} p^α p^β at the given state."""
        if self._has_fast_accel:
            return self.metric.acceleration(position, momentum)  # type: ignore[attr-defined]
        gamma: NDArray[np.float64] = self.metric.christoffel_symbols(position)
        return -np.einsum("mab,a,b->m", gamma, momentum, momentum)

    def _build_accel_fn(self) -> AccelFn:
        """Return a tight closure over the metric's acceleration routine."""
        metric = self.metric
        if self._has_fast_accel:
            return metric.acceleration  # type: ignore[attr-defined,return-value]

        def accel(
            x: NDArray[np.float64], p: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            gamma = metric.christoffel_symbols(x)
            return -np.einsum("mab,a,b->m", gamma, p, p)

        return accel

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
        r_max: float = self.r_max
        horizon_eps: float = self.horizon_eps
        base_step: float = self.base_step
        step_near: float = base_step * self.near_field_factor
        step_far: float = base_step * self.far_field_factor
        rs_3: float = 3.0 * rs
        rs_10: float = 10.0 * rs
        half_pi: float = 0.5 * np.pi
        accel: AccelFn = self._build_accel_fn()
        check_disk: bool = disk_inner is not None and disk_outer is not None

        x: NDArray[np.float64] = position.astype(np.float64, copy=True)
        p: NDArray[np.float64] = momentum.astype(np.float64, copy=True)
        lam: float = 0.0
        termination: str = "max_steps"

        traj: list[NDArray[np.float64]] | None = [x.copy()] if store_trajectory else None

        steps_taken: int = 0
        for _ in range(max_steps):
            r: float = float(x[1])

            if r <= rs + horizon_eps:
                termination = "horizon"
                break
            if r >= r_max:
                termination = "escape"
                break

            if r < rs_3:
                h = step_near
            elif r > rs_10:
                h = step_far
            else:
                h = base_step

            half_h: float = 0.5 * h
            sixth_h: float = h / 6.0

            k1p = accel(x, p)
            x2 = x + half_h * p
            p2 = p + half_h * k1p

            k2p = accel(x2, p2)
            x3 = x + half_h * p2
            p3 = p + half_h * k2p

            k3p = accel(x3, p3)
            x4 = x + h * p3
            p4 = p + h * k3p

            k4p = accel(x4, p4)

            x_next = x + sixth_h * (p + 2.0 * p2 + 2.0 * p3 + p4)
            p_next = p + sixth_h * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)

            theta_prev: float = float(x[2])
            theta_next: float = float(x_next[2])

            crossed: bool = (theta_prev - half_pi) * (theta_next - half_pi) < 0.0
            if check_disk and crossed:
                alpha: float = (half_pi - theta_prev) / (theta_next - theta_prev)
                x_cross = x + alpha * (x_next - x)
                p_cross = p + alpha * (p_next - p)
                r_cross: float = float(x_cross[1])
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
