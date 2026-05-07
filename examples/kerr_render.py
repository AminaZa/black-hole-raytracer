"""Near-extremal (a = 0.99 M) Kerr black hole render.

Camera at r = 30 M, inclination 75° from the polar axis — same view as the
cinematic Schwarzschild scene so the spin asymmetry is directly comparable.
The disk runs from the prograde Kerr ISCO (≈ 1.45 M at this spin) to 20 M;
its rest-frame temperature follows a simplified Novikov-Thorne profile and
the per-pixel colour picks up the combined gravitational, Doppler, and
frame-dragging shift via the conserved ``E`` and ``L`` of each traced ray.

Output: ``gallery/kerr_render.png`` (800 × 600).

Usage:
    python examples/kerr_render.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from camera.camera import Camera
from geodesic.integrator import GeodesicIntegrator
from metrics.kerr import KerrMetric
from render.curved_renderer import CurvedRenderer
from scene.kerr_accretion_disk import KerrAccretionDisk
from scene.starfield import Starfield


def main() -> None:
    spin: float = 0.99
    metric = KerrMetric(mass=1.0, spin=spin)

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

    disk = KerrAccretionDisk(
        outer_radius=20.0,
        mass=metric.mass,
        spin=metric.spin,
        t_peak=12000.0,
        beaming_exponent=3.0,
        prograde=True,
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
        f"Rendering Kerr (M={metric.mass}, a={metric.spin}, r+={metric.r_plus:.4f}, "
        f"ISCO={disk.isco:.4f}, inclination={inclination_deg}°, r_cam={r_cam})..."
    )
    wall_start = time.perf_counter()
    image = renderer.render()
    wall_elapsed = time.perf_counter() - wall_start

    output_path = (
        Path(__file__).resolve().parent.parent / "gallery" / "kerr_render.png"
    )
    renderer.save_png(image, output_path)

    print(f"Total render time: {wall_elapsed:.2f}s ({wall_elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
