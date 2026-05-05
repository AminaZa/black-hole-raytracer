"""Geometrically thin accretion disk in the equatorial plane.

Coordinate convention
---------------------
The equatorial plane is **y = 0** (the xz-plane).  The black hole sits at
the origin.  The disk occupies the annular region

    inner_radius ≤ √(x² + z²) ≤ outer_radius,   y = 0.

Polar coordinates used for colouring are

    r   = √(x² + z²)
    φ   = atan2(z, x)  ∈ (−π, π]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class AccretionDisk:
    """Geometrically thin, optically opaque accretion disk.

    The disk lies in the **equatorial plane y = 0** as an annulus between
    *inner_radius* and *outer_radius*.

    Parameters
    ----------
    inner_radius:
        Inner edge radius (same length units as the scene).
        Must be positive.
    outer_radius:
        Outer edge radius.  Must be strictly greater than *inner_radius*.

    Raises
    ------
    ValueError
        If the radii are non-positive or incorrectly ordered.
    """

    def __init__(self, inner_radius: float, outer_radius: float) -> None:
        if inner_radius <= 0.0:
            raise ValueError(f"inner_radius must be positive, got {inner_radius}.")
        if outer_radius <= inner_radius:
            raise ValueError(
                f"outer_radius ({outer_radius}) must be greater than "
                f"inner_radius ({inner_radius})."
            )
        self.inner_radius: float = float(inner_radius)
        self.outer_radius: float = float(outer_radius)

    # ------------------------------------------------------------------
    # Intersection
    # ------------------------------------------------------------------

    def intersect(
        self,
        ray_origins: NDArray[np.float64],
        ray_directions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Intersect a batch of rays with the disk.

        Solves for the ray parameter *t* at which each ray crosses y = 0:

            origin.y + t · direction.y = 0   →   t = −origin.y / direction.y

        Then checks that the crossing point lies within the annulus and that
        *t* is positive (i.e. the hit is in front of the camera).

        Parameters
        ----------
        ray_origins:
            Shape ``(N, 3)``.
        ray_directions:
            Shape ``(N, 3)``, unit vectors.

        Returns
        -------
        t_values:
            Shape ``(N,)``.  Ray parameter at the disk plane.  **Undefined**
            (set to −1) where *mask* is ``False``.
        hit_points:
            Shape ``(N, 3)``.  World-space positions on the disk plane.
            **Undefined** where *mask* is ``False``.
        mask:
            Shape ``(N,)``, dtype bool.  ``True`` where the ray hits the disk.
        """
        oy: NDArray[np.float64] = ray_origins[:, 1]
        dy: NDArray[np.float64] = ray_directions[:, 1]

        # Rays nearly parallel to y = 0 produce no reliable intersection
        valid_dir: NDArray[np.bool_] = np.abs(dy) > 1e-10

        # Use safe denominator (1.0 where invalid) to avoid NaN/Inf
        safe_dy: NDArray[np.float64] = np.where(valid_dir, dy, 1.0)
        t: NDArray[np.float64] = np.where(valid_dir, -oy / safe_dy, -1.0)

        hit_points: NDArray[np.float64] = ray_origins + t[:, np.newaxis] * ray_directions

        r: NDArray[np.float64] = np.sqrt(
            hit_points[:, 0] ** 2 + hit_points[:, 2] ** 2
        )

        mask: NDArray[np.bool_] = (
            valid_dir
            & (t > 1e-6)
            & (r >= self.inner_radius)
            & (r <= self.outer_radius)
        )

        return t, hit_points, mask

    # ------------------------------------------------------------------
    # Shading
    # ------------------------------------------------------------------

    def color(self, hit_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return a checkerboard colour for each intersection point.

        The pattern uses alternating **warm amber** and **dark mahogany**
        tiles in both the radial and angular directions.  The tile boundaries
        are evenly spaced in normalised radius and in angle, so the tiles
        appear as curved trapezoids in perspective — an effective visual
        test that the projection geometry is correct.

        Parameters
        ----------
        hit_points:
            Shape ``(N, 3)``.  Points on the disk (y ≈ 0).

        Returns
        -------
        colors:
            Shape ``(N, 3)``, dtype float64, values in [0, 1].
            Linear (non-gamma-corrected) RGB.
        """
        x: NDArray[np.float64] = hit_points[:, 0]
        z: NDArray[np.float64] = hit_points[:, 2]

        r: NDArray[np.float64] = np.sqrt(x**2 + z**2)
        phi: NDArray[np.float64] = np.arctan2(z, x)  # (−π, π]

        # Normalise radius to [0, 1] across the disk width
        r_norm: NDArray[np.float64] = (r - self.inner_radius) / (
            self.outer_radius - self.inner_radius
        )

        n_radial: int = 10
        n_angular: int = 16

        radial_idx: NDArray[np.int32] = np.floor(r_norm * n_radial).astype(np.int32)

        # Map φ from (−π, π] → [0, 1) then to sector index
        phi_norm: NDArray[np.float64] = (phi + np.pi) / (2.0 * np.pi)
        angular_idx: NDArray[np.int32] = np.floor(phi_norm * n_angular).astype(np.int32)

        # Parity determines which of the two colours to use
        checker: NDArray[np.int32] = (radial_idx + angular_idx) % 2

        color_a: NDArray[np.float64] = np.array([1.00, 0.65, 0.10])  # warm amber
        color_b: NDArray[np.float64] = np.array([0.25, 0.08, 0.02])  # dark mahogany

        # checker[:, np.newaxis]: (N, 1) broadcasts against (3,) colours
        colors: NDArray[np.float64] = np.where(
            checker[:, np.newaxis] == 0, color_a, color_b
        )

        return colors
