"""Pinhole camera model for perspective ray generation.

Coordinate convention used throughout this project
---------------------------------------------------
* Right-handed coordinate system, **y-up**.
* The equatorial plane (where the accretion disk lives) is **y = 0**.
* Positive z points *out of* the screen toward the viewer in the default
  orientation (i.e. the camera looks in the –z direction when placed on
  the positive z-axis looking at the origin).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Camera:
    """Pinhole camera with perspective projection.

    All rays share the camera position as their origin.  Directions are
    computed analytically for every pixel in a fully vectorised manner —
    no Python-level loops.

    Parameters
    ----------
    position:
        Camera position in world space, shape (3,).
    target:
        World-space point the camera is aimed at.
    up:
        World-space hint vector defining the "up" direction.
        Defaults to (0, 1, 0).  Must not be parallel to the look direction.
    fov_deg:
        Vertical field of view in **degrees**.
    width:
        Image width in pixels.
    height:
        Image height in pixels.

    Raises
    ------
    ValueError
        If *position* equals *target*, or if *up* is parallel to the
        look direction.
    """

    def __init__(
        self,
        position: NDArray[np.float64],
        target: NDArray[np.float64],
        up: NDArray[np.float64] | None = None,
        fov_deg: float = 45.0,
        width: int = 800,
        height: int = 600,
    ) -> None:
        self.position: NDArray[np.float64] = np.asarray(position, dtype=np.float64)
        self.target: NDArray[np.float64] = np.asarray(target, dtype=np.float64)
        self.up: NDArray[np.float64] = np.asarray(
            up if up is not None else [0.0, 1.0, 0.0], dtype=np.float64
        )
        self.fov_deg: float = float(fov_deg)
        self.width: int = int(width)
        self.height: int = int(height)

        self._forward: NDArray[np.float64]
        self._right: NDArray[np.float64]
        self._up_cam: NDArray[np.float64]
        self._build_basis()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_basis(self) -> None:
        """Compute and cache the orthonormal camera basis vectors.

        Sets
        ----
        _forward : unit vector pointing from *position* toward *target*.
        _right   : unit vector pointing to the right of the camera.
        _up_cam  : recomputed up vector, perpendicular to _forward/_right.
        """
        forward = self.target - self.position
        norm = np.linalg.norm(forward)
        if norm < 1e-12:
            raise ValueError("Camera position and target must be distinct points.")
        self._forward = forward / norm

        right = np.cross(self._forward, self.up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-12:
            raise ValueError(
                "The look direction and the up vector are parallel; "
                "choose a different up vector."
            )
        self._right = right / right_norm

        # Reorthogonalise up so the basis is exactly orthonormal
        self._up_cam = np.cross(self._right, self._forward)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_rays(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate one ray per pixel using the pinhole camera model.

        Pixels are enumerated in **row-major order**: pixel (row=0, col=0)
        maps to flat index 0, pixel (row=0, col=1) to index 1, and so on.
        This matches NumPy's default C-order reshape and the way Pillow
        interprets image arrays.

        The projection follows standard OpenGL-style NDC conventions:
        * Pixel (col=0, row=0) is the **top-left** corner of the image.
        * ``x_ndc ∈ [−aspect·tan(fov/2), +aspect·tan(fov/2)]``
        * ``y_ndc ∈ [−tan(fov/2), +tan(fov/2)]`` (positive = up)

        Returns
        -------
        origins:
            Shape ``(H*W, 3)``.  Every origin is the camera position.
        directions:
            Shape ``(H*W, 3)``.  Unit-length ray direction for each pixel.
        """
        W, H = self.width, self.height
        aspect: float = W / H
        tan_half_fov: float = float(np.tan(np.radians(self.fov_deg) / 2.0))

        cols = np.arange(W, dtype=np.float64)
        rows = np.arange(H, dtype=np.float64)

        # x_ndc shape (1, W) — horizontal offset scaled by aspect and fov
        x_ndc = ((cols[np.newaxis, :] + 0.5) / W * 2.0 - 1.0) * aspect * tan_half_fov

        # y_ndc shape (H, 1) — vertical offset; row 0 → top → positive y
        y_ndc = (1.0 - (rows[:, np.newaxis] + 0.5) / H * 2.0) * tan_half_fov

        # Build direction array with broadcasting; result shape (H, W, 3).
        # x_ndc[..., np.newaxis]: (1, W, 1)  ×  _right (3,)  → (1, W, 3)
        # y_ndc[..., np.newaxis]: (H, 1, 1)  ×  _up_cam (3,) → (H, 1, 3)
        # _forward (3,) broadcasts to (H, W, 3)
        dirs: NDArray[np.float64] = (
            self._forward
            + x_ndc[..., np.newaxis] * self._right
            + y_ndc[..., np.newaxis] * self._up_cam
        )

        # Normalise in one vectorised pass
        norms: NDArray[np.float64] = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs = dirs / norms

        # Flatten spatial dimensions → (H*W, 3)
        directions: NDArray[np.float64] = dirs.reshape(-1, 3)

        # All rays originate from the same point; copy for a writable array
        origins: NDArray[np.float64] = np.broadcast_to(
            self.position, (H * W, 3)
        ).copy()

        return origins, directions
