"""Geometrically thin accretion disk in the equatorial plane (y = 0).

The disk is an annulus: inner_radius <= sqrt(x^2 + z^2) <= outer_radius.
Polar coordinates: r = sqrt(x^2 + z^2), phi = atan2(z, x).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class AccretionDisk:
    """Thin opaque disk in the y = 0 plane.

    Parameters
    ----------
    inner_radius:
        Inner edge radius. Must be positive.
    outer_radius:
        Outer edge radius. Must exceed inner_radius.
    """

    def __init__(self, inner_radius: float, outer_radius: float) -> None:
        if inner_radius <= 0.0:
            raise ValueError(f"inner_radius must be positive, got {inner_radius}.")
        if outer_radius <= inner_radius:
            raise ValueError(
                f"outer_radius ({outer_radius}) must exceed inner_radius ({inner_radius})."
            )
        self.inner_radius: float = float(inner_radius)
        self.outer_radius: float = float(outer_radius)

    def intersect(
        self,
        ray_origins: NDArray[np.float64],
        ray_directions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Intersect rays with the disk plane.

        Solves t = -origin.y / direction.y for the y = 0 crossing, then
        checks the hit point falls within the annulus.

        Parameters
        ----------
        ray_origins:
            Shape (N, 3).
        ray_directions:
            Shape (N, 3), unit vectors.

        Returns
        -------
        t_values:
            Shape (N,). Ray parameter at the disk plane. Undefined where mask is False.
        hit_points:
            Shape (N, 3). World-space hit positions. Undefined where mask is False.
        mask:
            Shape (N,) bool. True where the ray hits the disk.
        """
        oy: NDArray[np.float64] = ray_origins[:, 1]
        dy: NDArray[np.float64] = ray_directions[:, 1]

        valid_dir: NDArray[np.bool_] = np.abs(dy) > 1e-10
        safe_dy: NDArray[np.float64] = np.where(valid_dir, dy, 1.0)
        t: NDArray[np.float64] = np.where(valid_dir, -oy / safe_dy, -1.0)

        hit_points: NDArray[np.float64] = ray_origins + t[:, np.newaxis] * ray_directions
        r: NDArray[np.float64] = np.sqrt(hit_points[:, 0] ** 2 + hit_points[:, 2] ** 2)

        mask: NDArray[np.bool_] = (
            valid_dir
            & (t > 1e-6)
            & (r >= self.inner_radius)
            & (r <= self.outer_radius)
        )

        return t, hit_points, mask

    def color(
        self,
        hit_points: NDArray[np.float64],
        photon_momenta: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Checkerboard pattern in polar (r, phi) coordinates.

        10 radial bands x 16 angular sectors, alternating amber and mahogany.
        Useful for verifying perspective geometry.

        Parameters
        ----------
        hit_points:
            Shape (N, 3).
        photon_momenta:
            Ignored; accepted to satisfy the ``EquatorialDisk`` protocol so
            the same renderer call site works for any disk implementation.

        Returns
        -------
        colors:
            Shape (N, 3), linear RGB in [0, 1].
        """
        del photon_momenta
        x: NDArray[np.float64] = hit_points[:, 0]
        z: NDArray[np.float64] = hit_points[:, 2]

        r: NDArray[np.float64] = np.sqrt(x**2 + z**2)
        phi: NDArray[np.float64] = np.arctan2(z, x)  # (-pi, pi]

        r_norm: NDArray[np.float64] = (r - self.inner_radius) / (
            self.outer_radius - self.inner_radius
        )

        radial_idx: NDArray[np.int32] = np.floor(r_norm * 10).astype(np.int32)
        phi_norm: NDArray[np.float64] = (phi + np.pi) / (2.0 * np.pi)
        angular_idx: NDArray[np.int32] = np.floor(phi_norm * 16).astype(np.int32)

        checker: NDArray[np.int32] = (radial_idx + angular_idx) % 2

        color_a: NDArray[np.float64] = np.array([1.00, 0.65, 0.10])  # amber
        color_b: NDArray[np.float64] = np.array([0.25, 0.08, 0.02])  # mahogany

        return np.where(checker[:, np.newaxis] == 0, color_a, color_b)
