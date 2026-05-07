"""GPU-batched null-geodesic integrator.

All N rays advance together: a single RK4 step touches the entire (N, 4)
position and momentum arrays at once via the metric's
``acceleration_batch``. A boolean ``live_mask`` tracks which rays are still
being integrated; horizon, escape, and equatorial-disk crossings flip the
mask off and freeze that ray's state. The loop exits early once every ray
has terminated.

The metric must implement ``acceleration_batch(positions, momenta)``
returning an (N, 4) array of the same array module (numpy or cupy) as
its inputs. The integrator itself is array-module agnostic — pass numpy
arrays for a CPU dry-run or cupy arrays for the GPU path. The companion
``GpuRenderer`` chooses the array module via ``utils.cuda_loader``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from utils.cuda_loader import get_xp


@runtime_checkable
class BatchedMetricProtocol(Protocol):
    """Metric interface required by the batched integrator."""

    rs: float
    mass: float

    def acceleration_batch(self, positions, momenta):  # type: ignore[no-untyped-def]
        """Return (N, 4) accelerations for (N, 4) positions and momenta."""
        ...


# Numeric termination codes shared with consumers (keeps the result format
# compact and array-module-friendly).
TERM_RUNNING: int = 0
TERM_HORIZON: int = 1
TERM_ESCAPE: int = 2
TERM_DISK: int = 3
TERM_MAX_STEPS: int = 4


@dataclass
class BatchedGeodesicResult:
    """Outcome of integrating a batch of rays.

    Attributes
    ----------
    termination:
        Shape (N,) integer array. ``0`` running, ``1`` horizon, ``2`` escape,
        ``3`` disk, ``4`` max_steps.
    final_position:
        Shape (N, 4). Final spacetime coordinates per ray.
    final_momentum:
        Shape (N, 4). Final 4-momentum per ray.
    n_steps:
        Number of integration steps the loop executed (a single integer).
    """

    termination: Any
    final_position: Any
    final_momentum: Any
    n_steps: int


class GpuGeodesicIntegrator:
    """Batched RK4 null-geodesic integrator.

    Parameters mirror :class:`geodesic.integrator.GeodesicIntegrator` so the
    same defaults work for both CPU and GPU paths. The integrator does not
    copy ``positions`` / ``momenta`` to GPU — the caller is responsible for
    placing them on the desired device first.
    """

    def __init__(
        self,
        metric: BatchedMetricProtocol,
        r_max: float = 100.0,
        horizon_eps: float = 1e-3,
        base_step: float = 0.2,
        near_field_factor: float = 0.1,
        far_field_factor: float = 2.0,
    ) -> None:
        self.metric: BatchedMetricProtocol = metric
        self.r_max: float = float(r_max)
        self.horizon_eps: float = float(horizon_eps)
        self.base_step: float = float(base_step)
        self.near_field_factor: float = float(near_field_factor)
        self.far_field_factor: float = float(far_field_factor)

    def integrate_batch(
        self,
        positions,  # type: ignore[no-untyped-def]
        momenta,  # type: ignore[no-untyped-def]
        max_steps: int = 5000,
        disk_inner: float | None = None,
        disk_outer: float | None = None,
    ) -> BatchedGeodesicResult:
        """Trace ``N`` rays simultaneously.

        ``positions`` and ``momenta`` are (N, 4) arrays — numpy for a CPU
        run, cupy for GPU. Returns a :class:`BatchedGeodesicResult` whose
        arrays live on the same device as the inputs.
        """
        xp = get_xp(positions)
        rs: float = self.metric.rs
        r_max: float = self.r_max
        horizon_eps: float = self.horizon_eps
        base_step: float = self.base_step
        step_near: float = base_step * self.near_field_factor
        step_far: float = base_step * self.far_field_factor
        rs_3: float = 3.0 * rs
        rs_10: float = 10.0 * rs
        half_pi: float = 0.5 * float(np.pi)
        check_disk: bool = disk_inner is not None and disk_outer is not None

        pos = positions.astype(xp.float64, copy=True)
        mom = momenta.astype(xp.float64, copy=True)

        n_rays: int = int(pos.shape[0])
        termination = xp.zeros(n_rays, dtype=xp.int8)
        live_mask = xp.ones(n_rays, dtype=xp.bool_)

        accel = self.metric.acceleration_batch
        steps_taken: int = 0

        # Periodically poll for "all done" — sync is expensive, so do it
        # every ``poll_every`` steps rather than each iteration. This keeps
        # the inner loop almost-pure GPU work.
        poll_every: int = 32

        for step_idx in range(max_steps):
            r = pos[:, 1]

            horizon_hit = (r <= rs + horizon_eps) & live_mask
            escape_hit = (r >= r_max) & live_mask
            termination = xp.where(horizon_hit, TERM_HORIZON, termination)
            termination = xp.where(escape_hit, TERM_ESCAPE, termination)
            live_mask = live_mask & ~(horizon_hit | escape_hit)

            # Per-ray step size based on radial distance.
            h_arr = xp.where(
                r < rs_3,
                step_near,
                xp.where(r > rs_10, step_far, base_step),
            )
            h_col = h_arr[:, None]

            k1 = accel(pos, mom)
            x2 = pos + 0.5 * h_col * mom
            p2 = mom + 0.5 * h_col * k1
            k2 = accel(x2, p2)
            x3 = pos + 0.5 * h_col * p2
            p3 = mom + 0.5 * h_col * k2
            k3 = accel(x3, p3)
            x4 = pos + h_col * p3
            p4 = mom + h_col * k3
            k4 = accel(x4, p4)

            sixth = h_col / 6.0
            new_pos = pos + sixth * (mom + 2.0 * p2 + 2.0 * p3 + p4)
            new_mom = mom + sixth * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if check_disk:
                theta_prev = pos[:, 2]
                theta_next = new_pos[:, 2]
                crossed = ((theta_prev - half_pi) * (theta_next - half_pi) < 0.0) & live_mask
                denom = theta_next - theta_prev
                safe_denom = xp.where(denom == 0.0, 1.0, denom)
                alpha = (half_pi - theta_prev) / safe_denom
                alpha_col = alpha[:, None]
                x_cross = pos + alpha_col * (new_pos - pos)
                p_cross = mom + alpha_col * (new_mom - mom)
                r_cross = x_cross[:, 1]
                in_annulus = (r_cross >= float(disk_inner)) & (
                    r_cross <= float(disk_outer)
                )
                disk_hit = crossed & in_annulus
                # Apply disk-hit rays first so their final state is the
                # interpolated crossing, not the post-step state.
                mask_disk = disk_hit[:, None]
                pos = xp.where(mask_disk, x_cross, pos)
                mom = xp.where(mask_disk, p_cross, mom)
                termination = xp.where(disk_hit, TERM_DISK, termination)
                live_mask = live_mask & ~disk_hit

            # Apply the RK4 update only to rays still alive after this step.
            mask_live = live_mask[:, None]
            pos = xp.where(mask_live, new_pos, pos)
            mom = xp.where(mask_live, new_mom, mom)
            steps_taken += 1

            # Cheap-ish sync: only poll every poll_every iterations.
            if (step_idx + 1) % poll_every == 0:
                if not bool(live_mask.any()):
                    break

        # Any rays still live at the end ran out of steps.
        termination = xp.where(live_mask, TERM_MAX_STEPS, termination)

        return BatchedGeodesicResult(
            termination=termination,
            final_position=pos,
            final_momentum=mom,
            n_steps=steps_taken,
        )
