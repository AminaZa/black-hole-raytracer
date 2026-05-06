"""Schwarzschild ray-traced render.

Camera at (0, 5, -30) looking at the origin. M = 1, so rs = 2. The accretion
disk extends from the ISCO at 3 rs = 6 out to 12 rs = 24, in the y = 0 plane.

Output: ``gallery/schwarzschild_test.png`` (400x300).

Usage:
    python examples/schwarzschild_render.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from camera.camera import Camera
from geodesic.integrator import GeodesicIntegrator
from metrics.schwarzschild import SchwarzschildMetric
from render.curved_renderer import CurvedRenderer
from scene.accretion_disk import AccretionDisk


def main() -> None:
    metric = SchwarzschildMetric(mass=1.0)
    rs: float = metric.rs

    camera = Camera(
        position=np.array([0.0, 5.0, -30.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=60.0,
        width=400,
        height=300,
    )

    disk = AccretionDisk(inner_radius=3.0 * rs, outer_radius=12.0 * rs)

    integrator = GeodesicIntegrator(
        metric=metric,
        r_max=100.0,
        horizon_eps=1e-3,
        base_step=0.25,
        near_field_factor=0.1,
    )

    renderer = CurvedRenderer(
        camera=camera,
        metric=metric,
        integrator=integrator,
        scene_objects=[disk],
        background_color=np.array([0.02, 0.02, 0.08]),
        progress_every=5000,
    )

    print(f"Rendering Schwarzschild scene (M={metric.mass}, rs={rs})...")
    image = renderer.render()

    output_path = Path(__file__).resolve().parent.parent / "gallery" / "schwarzschild_test.png"
    renderer.save_png(image, output_path)


if __name__ == "__main__":
    main()
