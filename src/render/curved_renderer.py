"""Curved-spacetime renderer.

For each pixel: convert the Cartesian ray to Schwarzschild-like spherical
coordinates, build a null 4-momentum in the static observer's local frame,
hand it to the geodesic integrator, and colour the pixel based on what the
ray hit (event horizon → black, equatorial disk → disk colour, escape →
background).

The render loop is per-ray Python and is intentionally slow; it is the
correctness baseline for the vectorised renderer that comes in Phase 3.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from geodesic.integrator import GeodesicIntegrator, MetricProtocol


@runtime_checkable
class EquatorialDisk(Protocol):
    """Object embedded in the θ = π/2 plane, identified by a radial range."""

    inner_radius: float
    outer_radius: float

    def color(self, hit_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return linear RGB at Cartesian hit points, shape (N, 3)."""
        ...


@runtime_checkable
class CameraProtocol(Protocol):
    """Interface the renderer expects from a camera."""

    width: int
    height: int
    position: NDArray[np.float64]

    def generate_rays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ...


def cartesian_to_spherical(point: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert (x, y, z) to (r, θ, φ) with the +y axis as the polar axis."""
    x: float = float(point[0])
    y: float = float(point[1])
    z: float = float(point[2])
    r: float = float(np.sqrt(x * x + y * y + z * z))
    theta: float = float(np.arccos(np.clip(y / r, -1.0, 1.0)))
    phi: float = float(np.arctan2(z, x))
    return r, theta, phi


def spherical_to_cartesian(r: float, theta: float, phi: float) -> NDArray[np.float64]:
    """Inverse of :func:`cartesian_to_spherical`."""
    sin_t: float = float(np.sin(theta))
    return np.array(
        [r * sin_t * np.cos(phi), r * np.cos(theta), r * sin_t * np.sin(phi)],
        dtype=np.float64,
    )


def cartesian_to_spherical_basis(
    direction: NDArray[np.float64], theta: float, phi: float
) -> tuple[float, float, float]:
    """Project a Cartesian direction onto the local orthonormal spherical frame.

    With the +y axis as the polar axis, the orthonormal basis at (r, θ, φ) is

        ê_r = (sinθ cosφ, cosθ, sinθ sinφ)
        ê_θ = (cosθ cosφ, -sinθ, cosθ sinφ)
        ê_φ = (-sinφ, 0, cosφ)
    """
    sin_t: float = float(np.sin(theta))
    cos_t: float = float(np.cos(theta))
    sin_p: float = float(np.sin(phi))
    cos_p: float = float(np.cos(phi))

    dx: float = float(direction[0])
    dy: float = float(direction[1])
    dz: float = float(direction[2])

    n_r: float = dx * sin_t * cos_p + dy * cos_t + dz * sin_t * sin_p
    n_theta: float = dx * cos_t * cos_p - dy * sin_t + dz * cos_t * sin_p
    n_phi: float = -dx * sin_p + dz * cos_p
    return n_r, n_theta, n_phi


class CurvedRenderer:
    """Geodesic ray tracer in a static, asymptotically-flat spacetime.

    Parameters
    ----------
    camera:
        Object satisfying ``CameraProtocol``.
    metric:
        Spacetime metric satisfying ``MetricProtocol``.
    integrator:
        Pre-configured ``GeodesicIntegrator`` wired to ``metric``.
    scene_objects:
        Equatorial disks tested when a geodesic crosses the equatorial plane.
    background_color:
        Linear RGB for rays that escape to infinity.
    progress_every:
        Print a progress line every N pixels. Set to 0 to disable.
    """

    def __init__(
        self,
        camera: CameraProtocol,
        metric: MetricProtocol,
        integrator: GeodesicIntegrator,
        scene_objects: list[EquatorialDisk],
        background_color: NDArray[np.float64] | None = None,
        progress_every: int = 5000,
    ) -> None:
        self.camera = camera
        self.metric = metric
        self.integrator = integrator
        self.scene_objects = list(scene_objects)
        self.background_color: NDArray[np.float64] = (
            np.asarray(background_color, dtype=np.float64)
            if background_color is not None
            else np.array([0.02, 0.02, 0.08], dtype=np.float64)
        )
        self.progress_every: int = int(progress_every)

    def _disk_radii(self) -> tuple[float, float] | tuple[None, None]:
        """Return the union of all disk radii, or (None, None) if no disks."""
        if not self.scene_objects:
            return None, None
        inner: float = min(o.inner_radius for o in self.scene_objects)
        outer: float = max(o.outer_radius for o in self.scene_objects)
        return inner, outer

    def _initial_momentum(
        self, r: float, theta: float, n_r: float, n_theta: float, n_phi: float
    ) -> NDArray[np.float64]:
        """Build a null 4-momentum from a unit direction in the static frame.

        Static observer at radius r with energy E = 1 sees a photon moving in
        direction (n_r, n_θ, n_φ). The corresponding coordinate-basis components
        satisfy g_{μν} p^μ p^ν = 0 by construction.
        """
        f: float = 1.0 - self.metric.rs / r
        sqrt_f: float = float(np.sqrt(f))
        sin_theta: float = float(np.sin(theta))

        p_t: float = 1.0 / sqrt_f
        p_r: float = n_r * sqrt_f
        p_theta: float = n_theta / r
        p_phi: float = n_phi / (r * sin_theta)
        return np.array([p_t, p_r, p_theta, p_phi], dtype=np.float64)

    def _shade_disk_hit(self, hit_position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the linear-RGB colour at a (4,) Schwarzschild disk-crossing point."""
        r_hit: float = float(hit_position[1])
        theta_hit: float = float(hit_position[2])
        phi_hit: float = float(hit_position[3])
        cart: NDArray[np.float64] = spherical_to_cartesian(r_hit, theta_hit, phi_hit)

        for obj in self.scene_objects:
            if obj.inner_radius <= r_hit <= obj.outer_radius:
                return obj.color(cart[np.newaxis, :])[0]
        return self.background_color

    def render(self) -> NDArray[np.uint8]:
        """Trace every camera ray and return an (H, W, 3) uint8 image."""
        W: int = self.camera.width
        H: int = self.camera.height
        N: int = H * W

        origins, directions = self.camera.generate_rays()
        cam_pos: NDArray[np.float64] = np.asarray(self.camera.position, dtype=np.float64)
        r_cam, theta_cam, phi_cam = cartesian_to_spherical(cam_pos)

        disk_inner, disk_outer = self._disk_radii()

        image_linear: NDArray[np.float64] = np.tile(self.background_color, (N, 1))
        black: NDArray[np.float64] = np.zeros(3, dtype=np.float64)

        start_time: float = time.perf_counter()
        for i in range(N):
            n_r, n_theta, n_phi = cartesian_to_spherical_basis(
                directions[i], theta_cam, phi_cam
            )
            position: NDArray[np.float64] = np.array(
                [0.0, r_cam, theta_cam, phi_cam], dtype=np.float64
            )
            momentum: NDArray[np.float64] = self._initial_momentum(
                r_cam, theta_cam, n_r, n_theta, n_phi
            )

            result = self.integrator.integrate(
                position,
                momentum,
                disk_inner=disk_inner,
                disk_outer=disk_outer,
            )

            if result.termination == "horizon":
                image_linear[i] = black
            elif result.termination == "disk":
                image_linear[i] = self._shade_disk_hit(result.final_position)
            # "escape" and "max_steps" both fall through to background.

            if self.progress_every and (i + 1) % self.progress_every == 0:
                elapsed: float = time.perf_counter() - start_time
                rate: float = (i + 1) / elapsed
                eta: float = (N - (i + 1)) / rate if rate > 0 else float("inf")
                pct: float = 100.0 * (i + 1) / N
                print(
                    f"  {i + 1:>7}/{N} ({pct:5.1f}%)  "
                    f"{rate:.0f} rays/s  ETA {eta:5.1f}s"
                )

        total: float = time.perf_counter() - start_time
        print(f"Render finished in {total:.2f}s ({N / total:.0f} rays/s).")

        image_uint8: NDArray[np.uint8] = (
            np.clip(image_linear, 0.0, 1.0) * 255.0 + 0.5
        ).astype(np.uint8)
        return image_uint8.reshape(H, W, 3)

    def save_png(self, image: NDArray[np.uint8], path: str | Path) -> None:
        """Save an (H, W, 3) uint8 image to a PNG file, creating parents."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(dest)
        print(f"Saved {image.shape[1]}x{image.shape[0]} image to {dest}")
