"""Spin sweep animation: a = 0 → 0.99 with the camera held fixed.

Each frame rebuilds the Kerr metric and disk for the current spin, so the
ISCO (and therefore the disk inner edge) shrinks frame by frame, the
shadow flattens, and the prograde Doppler asymmetry intensifies — the
canonical Kerr signature unfolding live.

Output: ``gallery/spin_evolution.gif`` (800 × 600 @ 24 fps).

Usage:
    python examples/spin_animation.py
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


WIDTH: int = 800
HEIGHT: int = 600
FRAMES: int = 90
FPS: int = 24
SPIN_END: float = 0.99
R_CAM: float = 30.0
INCLINATION_DEG: float = 75.0


def main() -> None:
    print(f"Compute backend: {device_summary()}")
    if not gpu_available:
        print(
            "Warning: GPU not available; falling back to numpy path "
            "(this will take much longer per frame)."
        )

    starfield = Starfield(n_stars=2500, seed=42, star_radius_deg=0.3)

    theta_cam = np.radians(INCLINATION_DEG)
    cam_pos = np.array(
        [0.0, R_CAM * np.cos(theta_cam), -R_CAM * np.sin(theta_cam)],
        dtype=np.float64,
    )
    camera = Camera(
        position=cam_pos,
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=45.0,
        width=WIDTH,
        height=HEIGHT,
    )

    if gpu_available:
        from geodesic.gpu_integrator import GpuGeodesicIntegrator
        from render.gpu_renderer import GpuRenderer

        IntegratorCls = GpuGeodesicIntegrator
        RendererCls = GpuRenderer
    else:
        from geodesic.integrator import GeodesicIntegrator
        from render.curved_renderer import CurvedRenderer

        IntegratorCls = GeodesicIntegrator  # type: ignore[assignment]
        RendererCls = CurvedRenderer  # type: ignore[assignment]

    def render_frame(idx: int, total: int) -> np.ndarray:
        # Linear sweep so the timing of the disk edge moving in is predictable.
        a = SPIN_END * idx / max(total - 1, 1)
        # Avoid the exact extremal pole; the metric requires |a| < M.
        a = float(min(a, SPIN_END))

        metric = KerrMetric(mass=1.0, spin=a)
        disk = KerrAccretionDisk(
            outer_radius=20.0,
            mass=metric.mass,
            spin=metric.spin,
            t_peak=12000.0,
            beaming_exponent=3.0,
            prograde=True,
        )
        integrator = IntegratorCls(
            metric=metric,
            r_max=120.0,
            horizon_eps=1e-3,
            base_step=0.25,
            near_field_factor=0.1,
            far_field_factor=2.0,
        )
        renderer = RendererCls(
            camera=camera,
            metric=metric,
            integrator=integrator,
            scene_objects=[disk],
            background_sampler=starfield,
        )
        return renderer.render()

    def label(idx: int, total: int) -> str:
        a = SPIN_END * idx / max(total - 1, 1)
        return f"a = {a:.3f}"

    out = Path(__file__).resolve().parent.parent / "gallery" / "spin_evolution.gif"
    max_per_run_env = os.environ.get("ANIM_MAX_FRAMES_PER_RUN")
    max_per_run = int(max_per_run_env) if max_per_run_env else None
    animator = Animator(
        render_frame=render_frame,
        frame_count=FRAMES,
        output_path=out,
        fps=FPS,
        label=label,
        max_frames_per_run=max_per_run,
    )
    wall = time.perf_counter()
    animator.run()
    print(f"Spin animation done in {time.perf_counter() - wall:.1f}s")


if __name__ == "__main__":
    main()
