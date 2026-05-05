"""Flat-spacetime render: camera + accretion disk baseline.

Scene
-----
* Camera at (0, 5, −15) aimed at the origin, 45° vertical FOV.
* Accretion disk in the equatorial plane (y = 0),
  inner_radius = 3, outer_radius = 12.
* 800 × 600 pixels, saved to gallery/flat_test.png.

This render contains no GR physics — rays travel in straight lines.
Its purpose is to verify the pinhole camera model and disk geometry
before the geodesic integrator is introduced.

Usage
-----
Run from the repository root::

    python examples/flat_render.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without ``pip install`` by adding src/ to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from camera.camera import Camera
from render.renderer import Renderer
from scene.accretion_disk import AccretionDisk


def main() -> None:
    """Build the scene, render, and save the output image."""

    # ------------------------------------------------------------------
    # Camera
    # Camera sits above (y = 5) and behind (z = −15) the disk, looking
    # toward the origin.  This gives a moderate downward viewing angle
    # (~18°) that shows the disk in perspective without flattening it.
    # ------------------------------------------------------------------
    camera = Camera(
        position=np.array([0.0, 5.0, -15.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=45.0,
        width=800,
        height=600,
    )

    # ------------------------------------------------------------------
    # Scene objects
    # inner_radius = 3  matches the Schwarzschild ISCO at r = 6M when
    # M = 0.5, giving a plausible placeholder gap around the black hole.
    # ------------------------------------------------------------------
    disk = AccretionDisk(inner_radius=3.0, outer_radius=12.0)

    # ------------------------------------------------------------------
    # Renderer
    # ------------------------------------------------------------------
    renderer = Renderer(
        camera=camera,
        scene_objects=[disk],
        background_color=np.array([0.02, 0.02, 0.08]),  # deep navy
    )

    print("Rendering flat baseline (no GR)…")
    image = renderer.render()

    output_path = Path(__file__).resolve().parent.parent / "gallery" / "flat_test.png"
    renderer.save_png(image, output_path)


if __name__ == "__main__":
    main()
