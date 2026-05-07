"""Side-by-side render of four spins: a = 0, 0.5, 0.9, 0.99.

Camera, disk outer radius, FOV and exposure are held fixed; only the spin
parameter (and therefore the ISCO and frame-dragging strength) changes
between panels. The output is a single tiled PNG with each panel labelled,
intended as the portfolio showpiece for the Kerr work.

Output: ``gallery/spin_comparison.png``.

Usage:
    python examples/spin_comparison.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera.camera import Camera
from metrics.kerr import KerrMetric
from scene.kerr_accretion_disk import KerrAccretionDisk
from scene.starfield import Starfield
from utils.cuda_loader import device_summary, gpu_available


SPINS: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99)
PANEL_WIDTH: int = 400
PANEL_HEIGHT: int = 300


def render_panel(spin: float, starfield: Starfield) -> np.ndarray:
    """Render one panel at the given spin with all other settings fixed."""
    metric = KerrMetric(mass=1.0, spin=spin)

    inclination_deg = 75.0
    r_cam = 30.0
    theta_cam = np.radians(inclination_deg)
    cam_pos = np.array(
        [0.0, r_cam * np.cos(theta_cam), -r_cam * np.sin(theta_cam)],
        dtype=np.float64,
    )
    camera = Camera(
        position=cam_pos,
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=45.0,
        width=PANEL_WIDTH,
        height=PANEL_HEIGHT,
    )

    disk = KerrAccretionDisk(
        outer_radius=20.0,
        mass=metric.mass,
        spin=metric.spin,
        t_peak=12000.0,
        beaming_exponent=3.0,
        prograde=True,
    )

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
            camera=camera,
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
            camera=camera,
            metric=metric,
            integrator=integrator,
            scene_objects=[disk],
            background_sampler=starfield,
        )

    print(
        f"\n--- a = {spin}  (r+ = {metric.r_plus:.4f}, ISCO = {disk.isco:.4f}) ---"
    )
    return renderer.render()


def _load_font(size: int) -> ImageFont.ImageFont:
    """Try a few common system fonts before falling back to PIL's default."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    starfield = Starfield(n_stars=2500, seed=42, star_radius_deg=0.3)
    print(f"Compute backend: {device_summary()}")

    panels: list[np.ndarray] = []
    wall_start = time.perf_counter()
    for spin in SPINS:
        panels.append(render_panel(spin, starfield))
    wall_elapsed = time.perf_counter() - wall_start

    label_height = 30
    pad = 6
    cell_w = PANEL_WIDTH
    cell_h = PANEL_HEIGHT + label_height
    big_w = 2 * cell_w + 3 * pad
    big_h = 2 * cell_h + 3 * pad

    big = Image.new("RGB", (big_w, big_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(big)
    font = _load_font(18)

    for i, (spin, panel) in enumerate(zip(SPINS, panels)):
        row, col = divmod(i, 2)
        x0 = pad + col * (cell_w + pad)
        y0 = pad + row * (cell_h + pad)
        big.paste(Image.fromarray(panel), (x0, y0 + label_height))

        if spin == 0.0:
            label = "a = 0  (Schwarzschild)"
        else:
            label = f"a = {spin:g}  (Kerr)"
        draw.text((x0 + 8, y0 + 6), label, fill=(230, 230, 230), font=font)

    out = (
        Path(__file__).resolve().parent.parent / "gallery" / "spin_comparison.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    big.save(out)
    print(f"\nSaved {big_w}x{big_h} comparison to {out}")
    print(f"Total render time: {wall_elapsed:.2f}s ({wall_elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
