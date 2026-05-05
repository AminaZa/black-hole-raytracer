"""Flat-spacetime renderer.

Shoots rays from a camera, tests intersections against scene objects using
closest-hit ordering, and returns an (H, W, 3) uint8 image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@runtime_checkable
class SceneObject(Protocol):
    """Interface for renderable scene objects."""

    def intersect(
        self,
        ray_origins: NDArray[np.float64],
        ray_directions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Return (t_values, hit_points, mask), each shape (N,), (N,3), (N,)."""
        ...

    def color(self, hit_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return linear RGB colors, shape (N, 3), values in [0, 1]."""
        ...


@runtime_checkable
class CameraProtocol(Protocol):
    """Interface the renderer expects from a camera."""

    width: int
    height: int

    def generate_rays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (origins, directions), each shape (H*W, 3)."""
        ...


class Renderer:
    """Ray renderer using closest-hit compositing.

    For each ray the scene object with the smallest positive t wins.

    Parameters
    ----------
    camera:
        Any object satisfying CameraProtocol.
    scene_objects:
        Scene objects to test. All are intersected; only the closest hit is kept.
    background_color:
        Linear RGB for rays that hit nothing. Defaults to deep navy (0.02, 0.02, 0.08).
    """

    def __init__(
        self,
        camera: CameraProtocol,
        scene_objects: list[SceneObject],
        background_color: NDArray[np.float64] | None = None,
    ) -> None:
        self.camera = camera
        self.scene_objects = scene_objects
        self.background_color: NDArray[np.float64] = (
            np.asarray(background_color, dtype=np.float64)
            if background_color is not None
            else np.array([0.02, 0.02, 0.08], dtype=np.float64)
        )

    def render(self) -> NDArray[np.uint8]:
        """Trace all camera rays and return an (H, W, 3) uint8 image."""
        W: int = self.camera.width
        H: int = self.camera.height
        N: int = H * W

        origins, directions = self.camera.generate_rays()

        image_linear: NDArray[np.float64] = np.tile(self.background_color, (N, 1))
        best_t: NDArray[np.float64] = np.full(N, np.inf, dtype=np.float64)

        for obj in self.scene_objects:
            t_values, hit_points, mask = obj.intersect(origins, directions)

            closer: NDArray[np.bool_] = mask & (t_values < best_t)
            if not np.any(closer):
                continue

            colors: NDArray[np.float64] = obj.color(hit_points)
            image_linear[closer] = colors[closer]
            best_t[closer] = t_values[closer]

        image_uint8: NDArray[np.uint8] = (
            np.clip(image_linear, 0.0, 1.0) * 255.0 + 0.5
        ).astype(np.uint8)

        return image_uint8.reshape(H, W, 3)

    def save_png(self, image: NDArray[np.uint8], path: str | Path) -> None:
        """Save an (H, W, 3) uint8 image array to a PNG file.

        Creates parent directories if needed.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(dest)
        print(f"Saved {image.shape[1]}x{image.shape[0]} image to {dest}")
