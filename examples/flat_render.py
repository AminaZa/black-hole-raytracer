"""Flat-spacetime baseline render.

Camera at (0, 5, -15) looking at the origin, 45 deg FOV.
Accretion disk in y=0 plane, inner_radius=3, outer_radius=12.
Output: gallery/flat_test.png (800x600).

Usage:
    python examples/flat_render.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from camera.camera import Camera
from render.renderer import Renderer
from scene.accretion_disk import AccretionDisk


def main() -> None:
    camera = Camera(
        position=np.array([0.0, 5.0, -15.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=45.0,
        width=800,
        height=600,
    )

    # inner_radius=3 approximates the Schwarzschild ISCO gap at r=6M for M=0.5
    disk = AccretionDisk(inner_radius=3.0, outer_radius=12.0)

    renderer = Renderer(
        camera=camera,
        scene_objects=[disk],
        background_color=np.array([0.02, 0.02, 0.08]),
    )

    print("Rendering...")
    image = renderer.render()

    output_path = Path(__file__).resolve().parent.parent / "gallery" / "flat_test.png"
    renderer.save_png(image, output_path)


if __name__ == "__main__":
    main()
