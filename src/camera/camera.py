"""Pinhole camera model.

Coordinate convention: right-handed, y-up. The equatorial plane is y = 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Camera:
    """Pinhole camera with perspective projection.

    Parameters
    ----------
    position:
        Camera position in world space, shape (3,).
    target:
        Point the camera looks toward.
    up:
        World-space up hint. Defaults to (0, 1, 0).
    fov_deg:
        Vertical field of view in degrees.
    width:
        Image width in pixels.
    height:
        Image height in pixels.
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

    def _build_basis(self) -> None:
        """Build the orthonormal camera basis from position/target/up."""
        forward = self.target - self.position
        norm = np.linalg.norm(forward)
        if norm < 1e-12:
            raise ValueError("Camera position and target must be distinct points.")
        self._forward = forward / norm

        right = np.cross(self._forward, self.up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-12:
            raise ValueError(
                "Look direction and up vector are parallel; choose a different up vector."
            )
        self._right = right / right_norm

        # Reorthogonalise so the basis is exactly orthonormal
        self._up_cam = np.cross(self._right, self._forward)

    def generate_rays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return one ray per pixel in row-major order.

        NDC convention: pixel (row=0, col=0) is top-left, x increases right,
        y increases up.

        Returns
        -------
        origins:
            Shape (H*W, 3). All rays share the camera position.
        directions:
            Shape (H*W, 3). Unit-length per-pixel ray directions.
        """
        W, H = self.width, self.height
        aspect: float = W / H
        tan_half_fov: float = float(np.tan(np.radians(self.fov_deg) / 2.0))

        cols = np.arange(W, dtype=np.float64)
        rows = np.arange(H, dtype=np.float64)

        # x_ndc: (1, W) — horizontal offset; y_ndc: (H, 1) — vertical offset
        x_ndc = ((cols[np.newaxis, :] + 0.5) / W * 2.0 - 1.0) * aspect * tan_half_fov
        y_ndc = (1.0 - (rows[:, np.newaxis] + 0.5) / H * 2.0) * tan_half_fov

        # Broadcasting: (1,W,1)*(3,) + (H,1,1)*(3,) + (3,) -> (H, W, 3)
        dirs: NDArray[np.float64] = (
            self._forward
            + x_ndc[..., np.newaxis] * self._right
            + y_ndc[..., np.newaxis] * self._up_cam
        )

        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

        directions: NDArray[np.float64] = dirs.reshape(-1, 3)
        origins: NDArray[np.float64] = np.broadcast_to(self.position, (H * W, 3)).copy()

        return origins, directions
