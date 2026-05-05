"""Flat-spacetime renderer.

Shoots rays from a camera, intersects them against all scene objects,
composites the result using closest-hit ordering (smallest positive *t*),
and returns an 8-bit RGB image array.

The renderer is intentionally simple and free of GR physics.  It exists
as a baseline to verify the camera model and scene geometry before the
geodesic integrator is added in later phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@runtime_checkable
class SceneObject(Protocol):
    """Protocol that every renderable scene object must satisfy."""

    def intersect(
        self,
        ray_origins: NDArray[np.float64],
        ray_directions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Test ray–object intersections for a batch of rays.

        Parameters
        ----------
        ray_origins:
            Shape ``(N, 3)``.
        ray_directions:
            Shape ``(N, 3)``, unit vectors.

        Returns
        -------
        t_values:
            Shape ``(N,)``.  Ray parameter at the nearest intersection.
            Undefined (but finite) where *mask* is ``False``.
        hit_points:
            Shape ``(N, 3)``.  World-space hit positions.
            Undefined where *mask* is ``False``.
        mask:
            Shape ``(N,)``, dtype bool.  ``True`` where the ray hits.
        """
        ...

    def color(self, hit_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return linear RGB colours for a batch of hit points.

        Parameters
        ----------
        hit_points:
            Shape ``(N, 3)``.

        Returns
        -------
        colors:
            Shape ``(N, 3)``, values in ``[0, 1]``.
        """
        ...


@runtime_checkable
class CameraProtocol(Protocol):
    """Minimal interface the renderer requires from a camera."""

    width: int
    height: int

    def generate_rays(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(origins, directions)``, each shape ``(H*W, 3)``."""
        ...


class Renderer:
    """Flat-spacetime ray renderer.

    Composites contributions from multiple scene objects using
    **closest-hit** ordering: for each ray the object with the smallest
    positive *t* value wins.

    Parameters
    ----------
    camera:
        Any object satisfying :class:`CameraProtocol`.
    scene_objects:
        Ordered list of scene objects satisfying :class:`SceneObject`.
        All objects are tested; only the closest visible hit is kept.
    background_color:
        Linear RGB fallback colour for rays that hit nothing.
        Defaults to a deep navy ``(0.02, 0.02, 0.08)``.
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

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> NDArray[np.uint8]:
        """Trace all camera rays and composite scene objects.

        Algorithm
        ---------
        1. Generate ``H*W`` rays from the camera.
        2. For each scene object, call ``intersect`` to get per-ray
           *t* values, hit points, and a boolean hit mask.
        3. Keep the **closest** hit (smallest positive *t*) across all
           objects; write its colour into the output buffer.
        4. Rays that hit nothing retain the background colour.
        5. Clamp linear values to [0, 1] and quantise to uint8.

        Returns
        -------
        image:
            Shape ``(H, W, 3)``, dtype ``uint8``.  Ready for display or
            saving with Pillow.
        """
        W: int = self.camera.width
        H: int = self.camera.height
        N: int = H * W

        origins, directions = self.camera.generate_rays()

        # Pixel buffer initialised to the background colour
        image_linear: NDArray[np.float64] = np.tile(
            self.background_color, (N, 1)
        )  # (N, 3)

        # Track the smallest *t* seen so far per ray
        best_t: NDArray[np.float64] = np.full(N, np.inf, dtype=np.float64)

        for obj in self.scene_objects:
            t_values, hit_points, mask = obj.intersect(origins, directions)

            # This object wins only where it hits AND is closer than anything seen so far
            closer: NDArray[np.bool_] = mask & (t_values < best_t)
            if not np.any(closer):
                continue

            colors: NDArray[np.float64] = obj.color(hit_points)
            image_linear[closer] = colors[closer]
            best_t[closer] = t_values[closer]

        # Clamp, scale to 8-bit, and reshape to image dimensions
        image_uint8: NDArray[np.uint8] = (
            np.clip(image_linear, 0.0, 1.0) * 255.0 + 0.5
        ).astype(np.uint8)

        return image_uint8.reshape(H, W, 3)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save_png(
        self,
        image: NDArray[np.uint8],
        path: str | Path,
    ) -> None:
        """Save an image array to a PNG file.

        Parent directories are created automatically if they do not exist.

        Parameters
        ----------
        image:
            Shape ``(H, W, 3)``, dtype ``uint8``.
        path:
            Destination file path (``*.png`` recommended).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(dest)
        print(f"Saved {image.shape[1]}x{image.shape[0]} image to {dest}")
