"""Curved-spacetime renderer.

For each pixel: convert the Cartesian ray to Schwarzschild-like spherical
coordinates, build a null 4-momentum in the static observer's local frame,
hand it to the geodesic integrator, and colour the pixel based on what the
ray hit (event horizon → black, equatorial disk → disk colour, escape →
background).

The render loop fans pixels out across a ``multiprocessing.Pool`` so each
worker advances its own batch of rays in parallel. Initial conditions
(spherical-frame projection and null 4-momentum) are computed vectorised in
the master before chunks are dispatched, so workers do nothing but RK4.
"""

from __future__ import annotations

import multiprocessing as mp
import os
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

    def color(
        self,
        hit_points: NDArray[np.float64],
        photon_momenta: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return linear RGB at Cartesian hit points, shape (N, 3).

        ``photon_momenta`` (shape (N, 4), coordinate-basis 4-momentum at the
        hit) is optional; simple disks may ignore it. Disks that want Doppler
        / redshift effects use it.
        """
        ...


@runtime_checkable
class BackgroundSampler(Protocol):
    """Sky source queried for rays that escape to infinity."""

    def sample(self, directions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map (N, 3) Cartesian unit directions to (N, 3) linear RGB."""
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


def _project_basis_batch(
    directions: NDArray[np.float64], theta: float, phi: float
) -> NDArray[np.float64]:
    """Vectorised :func:`cartesian_to_spherical_basis` over (N, 3) directions."""
    sin_t: float = float(np.sin(theta))
    cos_t: float = float(np.cos(theta))
    sin_p: float = float(np.sin(phi))
    cos_p: float = float(np.cos(phi))

    dx = directions[:, 0]
    dy = directions[:, 1]
    dz = directions[:, 2]

    n_r = dx * sin_t * cos_p + dy * cos_t + dz * sin_t * sin_p
    n_theta = dx * cos_t * cos_p - dy * sin_t + dz * cos_t * sin_p
    n_phi = -dx * sin_p + dz * cos_p
    return np.stack([n_r, n_theta, n_phi], axis=1)


def _build_initial_momenta(
    n_components: NDArray[np.float64], r: float, theta: float, rs: float
) -> NDArray[np.float64]:
    """Convert local-frame unit directions to coordinate-basis null 4-momenta."""
    f: float = 1.0 - rs / r
    sqrt_f: float = float(np.sqrt(f))
    sin_theta: float = float(np.sin(theta))
    n: int = n_components.shape[0]

    momenta = np.empty((n, 4), dtype=np.float64)
    momenta[:, 0] = 1.0 / sqrt_f
    momenta[:, 1] = n_components[:, 0] * sqrt_f
    momenta[:, 2] = n_components[:, 1] / r
    momenta[:, 3] = n_components[:, 2] / (r * sin_theta)
    return momenta


def _render_chunk(args: tuple) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Worker: trace a batch of rays and return (pixel_indices, linear-RGB).

    Outcomes are bucketed (disk-hit grouped by which disk, escape, horizon)
    and each bucket is shaded in one batched call. This keeps Doppler /
    blackbody / starfield maths off the per-pixel hot path.
    """
    (
        indices,
        positions,
        momenta,
        integrator,
        scene_objects,
        disk_inner,
        disk_outer,
        background,
        background_sampler,
    ) = args

    n: int = len(indices)
    colors: NDArray[np.float64] = np.tile(background, (n, 1))
    black: NDArray[np.float64] = np.zeros(3, dtype=np.float64)

    disk_locals: list[int] = []
    disk_obj_idx: list[int] = []
    disk_carts: list[NDArray[np.float64]] = []
    disk_moms: list[NDArray[np.float64]] = []
    escape_locals: list[int] = []
    escape_dirs: list[NDArray[np.float64]] = []

    for i in range(n):
        result = integrator.integrate(
            positions[i],
            momenta[i],
            disk_inner=disk_inner,
            disk_outer=disk_outer,
        )

        if result.termination == "horizon":
            colors[i] = black
        elif result.termination == "disk":
            r_hit = float(result.final_position[1])
            for j, obj in enumerate(scene_objects):
                if obj.inner_radius <= r_hit <= obj.outer_radius:
                    theta_hit = float(result.final_position[2])
                    phi_hit = float(result.final_position[3])
                    cart = spherical_to_cartesian(r_hit, theta_hit, phi_hit)
                    disk_locals.append(i)
                    disk_obj_idx.append(j)
                    disk_carts.append(cart)
                    disk_moms.append(result.final_momentum)
                    break
        elif result.termination == "escape" and background_sampler is not None:
            r_f = float(result.final_position[1])
            th_f = float(result.final_position[2])
            ph_f = float(result.final_position[3])
            cart = spherical_to_cartesian(r_f, th_f, ph_f)
            norm = float(np.linalg.norm(cart))
            if norm > 0.0:
                escape_locals.append(i)
                escape_dirs.append(cart / norm)

    if disk_locals:
        disk_locals_arr = np.asarray(disk_locals, dtype=np.int64)
        disk_obj_arr = np.asarray(disk_obj_idx, dtype=np.int64)
        carts_arr = np.stack(disk_carts)
        moms_arr = np.stack(disk_moms)
        for j, obj in enumerate(scene_objects):
            mask = disk_obj_arr == j
            if mask.any():
                colors[disk_locals_arr[mask]] = obj.color(
                    carts_arr[mask], photon_momenta=moms_arr[mask]
                )

    if escape_locals:
        escape_locals_arr = np.asarray(escape_locals, dtype=np.int64)
        dirs_arr = np.stack(escape_dirs)
        # background_sampler is non-None inside this branch by construction.
        assert background_sampler is not None
        colors[escape_locals_arr] = background_sampler.sample(dirs_arr)

    return indices, colors


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
        Linear RGB for rays that escape to infinity when no
        ``background_sampler`` is provided.
    background_sampler:
        Optional sky source (e.g. a ``Starfield``). When set, escape rays are
        coloured by ``sampler.sample(direction)`` instead of the constant
        ``background_color``.
    n_workers:
        Worker processes for parallel rendering. ``None`` (default) uses every
        CPU. ``1`` runs serially in the calling process and skips
        multiprocessing entirely (handy for debugging).
    chunks_per_worker:
        Tasks each worker receives (smaller chunks → smoother progress bar
        and better load balance, larger chunks → less pickling overhead).
    """

    def __init__(
        self,
        camera: CameraProtocol,
        metric: MetricProtocol,
        integrator: GeodesicIntegrator,
        scene_objects: list[EquatorialDisk],
        background_color: NDArray[np.float64] | None = None,
        background_sampler: BackgroundSampler | None = None,
        n_workers: int | None = None,
        chunks_per_worker: int = 4,
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
        self.background_sampler: BackgroundSampler | None = background_sampler
        cpu = os.cpu_count() or 1
        self.n_workers: int = int(n_workers) if n_workers is not None else cpu
        self.chunks_per_worker: int = max(1, int(chunks_per_worker))

    def _disk_radii(self) -> tuple[float, float] | tuple[None, None]:
        """Return the union of all disk radii, or (None, None) if no disks."""
        if not self.scene_objects:
            return None, None
        inner: float = min(o.inner_radius for o in self.scene_objects)
        outer: float = max(o.outer_radius for o in self.scene_objects)
        return inner, outer

    def render(self) -> NDArray[np.uint8]:
        """Trace every camera ray and return an (H, W, 3) uint8 image."""
        W: int = self.camera.width
        H: int = self.camera.height
        N: int = H * W

        _, directions = self.camera.generate_rays()
        cam_pos: NDArray[np.float64] = np.asarray(self.camera.position, dtype=np.float64)
        r_cam, theta_cam, phi_cam = cartesian_to_spherical(cam_pos)

        n_local = _project_basis_batch(directions, theta_cam, phi_cam)
        momenta = _build_initial_momenta(n_local, r_cam, theta_cam, self.metric.rs)
        positions = np.empty((N, 4), dtype=np.float64)
        positions[:, 0] = 0.0
        positions[:, 1] = r_cam
        positions[:, 2] = theta_cam
        positions[:, 3] = phi_cam

        disk_inner, disk_outer = self._disk_radii()
        image_linear: NDArray[np.float64] = np.tile(self.background_color, (N, 1))

        n_chunks = max(1, self.n_workers * self.chunks_per_worker)
        index_chunks = np.array_split(np.arange(N, dtype=np.int64), n_chunks)
        tasks = [
            (
                idx,
                positions[idx],
                momenta[idx],
                self.integrator,
                self.scene_objects,
                disk_inner,
                disk_outer,
                self.background_color,
                self.background_sampler,
            )
            for idx in index_chunks
        ]

        print(
            f"Rendering {N} rays across {len(tasks)} chunks "
            f"on {self.n_workers} worker(s)..."
        )
        start_time: float = time.perf_counter()

        if self.n_workers <= 1:
            self._render_serial(tasks, image_linear, start_time)
        else:
            self._render_parallel(tasks, image_linear, start_time)

        total: float = time.perf_counter() - start_time
        rate: float = N / total if total > 0 else float("inf")
        print(f"Render finished in {total:.2f}s ({rate:.0f} rays/s).")

        image_uint8: NDArray[np.uint8] = (
            np.clip(image_linear, 0.0, 1.0) * 255.0 + 0.5
        ).astype(np.uint8)
        return image_uint8.reshape(H, W, 3)

    def _render_serial(
        self,
        tasks: list[tuple],
        image_linear: NDArray[np.float64],
        start_time: float,
    ) -> None:
        """In-process fallback used when ``n_workers <= 1``."""
        n_chunks = len(tasks)
        for i, task in enumerate(tasks, start=1):
            indices, colors = _render_chunk(task)
            image_linear[indices] = colors
            self._print_progress(i, n_chunks, start_time)

    def _render_parallel(
        self,
        tasks: list[tuple],
        image_linear: NDArray[np.float64],
        start_time: float,
    ) -> None:
        """Run chunks across a process pool, splicing results back as they land."""
        n_chunks = len(tasks)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=self.n_workers) as pool:
            completed = 0
            for indices, colors in pool.imap_unordered(_render_chunk, tasks):
                image_linear[indices] = colors
                completed += 1
                self._print_progress(completed, n_chunks, start_time)

    @staticmethod
    def _print_progress(done: int, total: int, start_time: float) -> None:
        """Single-line percentage progress with rate and ETA."""
        elapsed = time.perf_counter() - start_time
        pct = 100.0 * done / total
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("inf")
        print(
            f"  [{done:>3}/{total}] {pct:5.1f}%  "
            f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s",
            flush=True,
        )

    def save_png(self, image: NDArray[np.uint8], path: str | Path) -> None:
        """Save an (H, W, 3) uint8 image to a PNG file, creating parents."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(dest)
        print(f"Saved {image.shape[1]}x{image.shape[0]} image to {dest}")
