"""Procedural starfield used as the background for escaped rays.

Stars are point-source samples uniformly distributed on the unit sphere with
a random colour tint biased toward white-blue and a power-law brightness
distribution. Sampling is brute-force: each query direction's cosine against
every star is computed and the closest star within a user-defined angular
radius lights that pixel.

The (N, S) cosine matrix is materialised in one shot, so callers should batch
escape rays by render chunk to keep memory bounded.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Starfield:
    """Static random starfield with deterministic seeding.

    Parameters
    ----------
    n_stars:
        Number of stars to draw on the unit sphere.
    seed:
        RNG seed for reproducibility.
    background:
        Linear RGB used where no star is in range.
    star_radius_deg:
        Angular radius of each star, in degrees. A direction within this angle
        of a star direction sees the star colour with a smooth falloff toward
        the radius edge.
    """

    def __init__(
        self,
        n_stars: int = 2000,
        seed: int = 42,
        background: tuple[float, float, float] | NDArray[np.float64] = (0.005, 0.005, 0.02),
        star_radius_deg: float = 0.35,
    ) -> None:
        if n_stars <= 0:
            raise ValueError(f"n_stars must be positive, got {n_stars}.")
        if star_radius_deg <= 0.0:
            raise ValueError(f"star_radius_deg must be positive, got {star_radius_deg}.")

        rng = np.random.default_rng(int(seed))
        u = rng.uniform(0.0, 1.0, n_stars)
        v = rng.uniform(0.0, 1.0, n_stars)
        theta = np.arccos(2.0 * u - 1.0)
        phi = 2.0 * np.pi * v
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        self.directions: NDArray[np.float64] = np.stack(
            [sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)],
            axis=1,
        ).astype(np.float64)

        # Power-law brightnesses: lots of dim stars, a few bright ones.
        intensity = rng.uniform(0.0, 1.0, n_stars) ** 4.0 * 1.5 + 0.05

        # Colour temperature draw: most stars white-blue, a sprinkle warm.
        h = rng.uniform(0.0, 1.0, n_stars)
        r_tint = 1.0 - 0.35 * h
        g_tint = 0.85 + 0.15 * np.sin(np.pi * h)
        b_tint = 0.55 + 0.45 * h
        tints = np.stack([r_tint, g_tint, b_tint], axis=1)
        self.colors: NDArray[np.float64] = tints * intensity[:, np.newaxis]

        self.background: NDArray[np.float64] = np.asarray(background, dtype=np.float64)
        self.cos_threshold: float = float(np.cos(np.radians(star_radius_deg)))

    def sample(self, directions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return RGB for an array of escape directions.

        Parameters
        ----------
        directions:
            Shape (N, 3), unit vectors. Need not be exactly normalised; tiny
            deviations only widen the effective star radius slightly.

        Returns
        -------
        rgb:
            Shape (N, 3), linear RGB. Background where no star is in range.
        """
        if directions.size == 0:
            return np.zeros((0, 3), dtype=np.float64)

        cos_angles = directions @ self.directions.T
        max_cos = cos_angles.max(axis=1)
        max_idx = cos_angles.argmax(axis=1)

        out = np.broadcast_to(self.background, directions.shape).copy()

        hit = max_cos > self.cos_threshold
        if hit.any():
            t = (max_cos[hit] - self.cos_threshold) / (1.0 - self.cos_threshold)
            t = np.clip(t, 0.0, 1.0) ** 0.5
            out[hit] = self.background + self.colors[max_idx[hit]] * t[:, np.newaxis]
        return out
