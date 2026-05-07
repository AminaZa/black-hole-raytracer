"""Camera orbit around a Kerr black hole.

The metric, disk and starfield are built once and reused; only the camera
position changes between frames. The camera sits at radius r = 30 M with a
75° inclination from the spin axis and orbits 360° in azimuth over 120
frames.

Output: ``gallery/orbit_animation.gif`` (800 × 600 @ 24 fps).

Usage:
    python examples/orbit_animation.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from camera.camera import Camera
from metrics.kerr import KerrMetric
from render.animator import Animator
from scene.kerr_accretion_disk import KerrAccretionDisk
from scene.starfield import Starfield
from utils.cuda_loader import device_summary, gpu_available


SPIN: float = 0.9
WIDTH: int = 800
HEIGHT: int = 600
FRAMES: int = 120
FPS: int = 24
R_CAM: float = 30.0
INCLINATION_DEG: float = 75.0


def main() -> None:
    print(f"Compute backend: {device_summary()}")
    if not gpu_available:
        print(
            "Warning: GPU not available; falling back to numpy path "
            "(this will take much longer per frame)."
        )

    metric = KerrMetric(mass=1.0, spin=SPIN)
    disk = KerrAccretionDisk(
        outer_radius=20.0,
        mass=metric.mass,
        spin=metric.spin,
        t_peak=12000.0,
        beaming_exponent=3.0,
        prograde=True,
    )
    starfield = Starfield(n_stars=2500, seed=42, star_radius_deg=0.3)

    if gpu_available:
        from geodesic.gpu_integrator import GpuGeodesicIntegrator
        from render.gpu_renderer import GpuRenderer

        integrator = GpuGeodesicIntegrator(
            metric=metric,
            r_max=120.0,
            horizon_eps=1e-3,
            base_step=0.25,
            near_field_factor=0.1,
            far_field_factor=2.0,
        )
        renderer = GpuRenderer(
            camera=None,  # will be set per frame
            metric=metric,
            integrator=integrator,
            scene_objects=[disk],
            background_sampler=starfield,
        )
    else:
        from geodesic.integrator import GeodesicIntegrator
        from render.curved_renderer import CurvedRenderer

        integrator = GeodesicIntegrator(
            metric=metric,
            r_max=120.0,
            horizon_eps=1e-3,
            base_step=0.25,
            near_field_factor=0.1,
            far_field_factor=2.0,
        )
        renderer = CurvedRenderer(
            camera=None,
            metric=metric,
            integrator=integrator,
            scene_objects=[disk],
            background_sampler=starfield,
        )

    theta_cam = np.radians(INCLINATION_DEG)
    cos_t = np.cos(theta_cam)
    sin_t = np.sin(theta_cam)
    cam_y = R_CAM * cos_t

    def render_frame(idx: int, total: int) -> np.ndarray:
        # Azimuth wraps once over the loop; the last frame is one step short
        # of a full rotation so the GIF loops cleanly.
        phi = 2.0 * np.pi * idx / total
        cam_x = R_CAM * sin_t * np.cos(phi)
        cam_z = R_CAM * sin_t * np.sin(phi)
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
        renderer.camera = Camera(
            position=cam_pos,
            target=np.array([0.0, 0.0, 0.0]),
            up=np.array([0.0, 1.0, 0.0]),
            fov_deg=45.0,
            width=WIDTH,
            height=HEIGHT,
        )
        return renderer.render()

    out = Path(__file__).resolve().parent.parent / "gallery" / "orbit_animation.gif"
    max_per_run_env = os.environ.get("ANIM_MAX_FRAMES_PER_RUN")
    max_per_run = int(max_per_run_env) if max_per_run_env else None
    animator = Animator(
        render_frame=render_frame,
        frame_count=FRAMES,
        output_path=out,
        fps=FPS,
        max_frames_per_run=max_per_run,
    )
    wall = time.perf_counter()
    animator.run()
    print(f"Orbit animation done in {time.perf_counter() - wall:.1f}s")


if __name__ == "__main__":
    main()
