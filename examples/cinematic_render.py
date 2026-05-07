"""Cinematic Schwarzschild render with a physical disk and starfield.

Camera at radius r = 30 (geometric units), inclination 75° from the polar
axis — looking slightly down at the equatorial disk. The disk runs from the
ISCO at 6 M out to 20 M and is shaded by the simplified Novikov-Thorne /
Doppler / gravitational-redshift model in :mod:`scene.physical_disk`. Rays
that escape sample a seeded :mod:`scene.starfield` background, so the
gravitational lensing visibly distorts the stars near the shadow.

Output: ``gallery/cinematic_render.png`` (800×600).

Usage:
    python examples/cinematic_render.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from camera.camera import Camera
from geodesic.integrator import GeodesicIntegrator
from metrics.schwarzschild import SchwarzschildMetric
from render.curved_renderer import CurvedRenderer
from scene.physical_disk import PhysicalAccretionDisk
from scene.starfield import Starfield


def main() -> None:
    metric = SchwarzschildMetric(mass=1.0)
    rs: float = metric.rs

    inclination_deg: float = 75.0
    r_cam: float = 30.0
    theta_cam: float = np.radians(inclination_deg)
    cam_pos = np.array(
        [0.0, r_cam * np.cos(theta_cam), -r_cam * np.sin(theta_cam)],
        dtype=np.float64,
    )

    camera = Camera(
        position=cam_pos,
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=45.0,
        width=800,
        height=600,
    )

    disk = PhysicalAccretionDisk(
        inner_radius=6.0,
        outer_radius=20.0,
        mass=metric.mass,
        t_peak=12000.0,
        beaming_exponent=3.0,
    )

    starfield = Starfield(n_stars=2500, seed=42, star_radius_deg=0.3)

    integrator = GeodesicIntegrator(
        metric=metric,
        r_max=120.0,
        horizon_eps=1e-3,
        base_step=0.25,
        near_field_factor=0.1,
        far_field_factor=2.0,
    )

    renderer = CurvedRenderer(
        camera=camera,
        metric=metric,
        integrator=integrator,
        scene_objects=[disk],
        background_sampler=starfield,
    )

    print(
        f"Rendering cinematic scene "
        f"(M={metric.mass}, rs={rs}, inclination={inclination_deg}°, r_cam={r_cam})..."
    )
    wall_start = time.perf_counter()
    image = renderer.render()
    wall_elapsed = time.perf_counter() - wall_start

    output_path = Path(__file__).resolve().parent.parent / "gallery" / "cinematic_render.png"
    renderer.save_png(image, output_path)

    print(f"Total render time: {wall_elapsed:.2f}s ({wall_elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
