"""GPU-batched curved-spacetime renderer.

The whole image is one CUDA-side integration: every camera ray is uploaded
to device memory in a single (N, 4) array, the
:class:`GpuGeodesicIntegrator` advances them in lockstep, and the per-ray
outcomes are pulled back to host for shading. Disk colour and starfield
sampling stay on the CPU for now — they are O(N) but cheap compared to the
RK4 loop, and they reuse the same numpy code paths the CPU renderer
exercises.

Falls back gracefully: if CuPy is unavailable, the renderer keeps working
on numpy arrays (the integrator is array-module agnostic), so the GPU
renderer doubles as a single-process batched CPU renderer when needed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from geodesic.gpu_integrator import (
    GpuGeodesicIntegrator,
    TERM_DISK,
    TERM_ESCAPE,
    TERM_HORIZON,
)
from render.curved_renderer import (
    BackgroundSampler,
    CameraProtocol,
    EquatorialDisk,
    _build_initial_momenta,
    _project_basis_batch,
    cartesian_to_spherical,
)
from utils.cuda_loader import asnumpy, gpu_available, xp_module


def _spherical_to_cartesian_batch(
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorised inverse of ``cartesian_to_spherical``; +y is the polar axis."""
    sin_t = np.sin(theta)
    return np.stack(
        [r * sin_t * np.cos(phi), r * np.cos(theta), r * sin_t * np.sin(phi)],
        axis=1,
    )


class GpuRenderer:
    """Curved-spacetime renderer that pushes the entire ray batch to one
    GPU integrator call.

    Parameters
    ----------
    camera, metric, integrator, scene_objects, background_color,
    background_sampler:
        Same role as in :class:`render.curved_renderer.CurvedRenderer`. The
        integrator must be a :class:`GpuGeodesicIntegrator` (which itself is
        array-module agnostic; ``utils.cuda_loader.xp_module`` decides
        whether the rays live on GPU or CPU).
    """

    def __init__(
        self,
        camera: CameraProtocol,
        metric,  # type: ignore[no-untyped-def]
        integrator: GpuGeodesicIntegrator,
        scene_objects: list[EquatorialDisk],
        background_color: NDArray[np.float64] | None = None,
        background_sampler: BackgroundSampler | None = None,
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

    def _disk_radii(self) -> tuple[float, float] | tuple[None, None]:
        if not self.scene_objects:
            return None, None
        inner = float(min(o.inner_radius for o in self.scene_objects))
        outer = float(max(o.outer_radius for o in self.scene_objects))
        return inner, outer

    def render(self, max_steps: int = 5000) -> NDArray[np.uint8]:
        """Trace every camera ray and return an (H, W, 3) uint8 image."""
        W: int = self.camera.width
        H: int = self.camera.height
        N: int = H * W

        _, directions = self.camera.generate_rays()
        cam_pos: NDArray[np.float64] = np.asarray(
            self.camera.position, dtype=np.float64
        )
        r_cam, theta_cam, phi_cam = cartesian_to_spherical(cam_pos)

        n_local = _project_basis_batch(directions, theta_cam, phi_cam)
        momenta_cpu = _build_initial_momenta(
            n_local, r_cam, theta_cam, 2.0 * self.metric.mass
        )
        positions_cpu = np.empty((N, 4), dtype=np.float64)
        positions_cpu[:, 0] = 0.0
        positions_cpu[:, 1] = r_cam
        positions_cpu[:, 2] = theta_cam
        positions_cpu[:, 3] = phi_cam

        # Push to whatever array module the loader picked (cupy on GPU,
        # numpy if CuPy isn't available — the integrator handles either).
        xp = xp_module
        positions = xp.asarray(positions_cpu)
        momenta = xp.asarray(momenta_cpu)

        disk_inner, disk_outer = self._disk_radii()

        backend = "GPU" if gpu_available else "CPU (numpy fallback)"
        print(
            f"GpuRenderer: tracing {N} rays in one batch on {backend}, "
            f"max_steps={max_steps}..."
        )
        start = time.perf_counter()
        result = self.integrator.integrate_batch(
            positions,
            momenta,
            max_steps=max_steps,
            disk_inner=disk_inner,
            disk_outer=disk_outer,
        )
        if gpu_available:
            xp.cuda.Stream.null.synchronize()
        integ_elapsed = time.perf_counter() - start
        rate = N / integ_elapsed if integ_elapsed > 0 else float("inf")
        print(
            f"  integrator: {integ_elapsed:.2f}s for {result.n_steps} steps "
            f"({rate:.0f} rays/s)"
        )

        term = asnumpy(result.termination)
        pos = asnumpy(result.final_position)
        mom = asnumpy(result.final_momentum)

        image_linear: NDArray[np.float64] = np.tile(self.background_color, (N, 1))
        image_linear[term == TERM_HORIZON] = 0.0

        disk_mask = term == TERM_DISK
        if disk_mask.any():
            disk_idx = np.where(disk_mask)[0]
            r_hits = pos[disk_idx, 1]
            theta_hits = pos[disk_idx, 2]
            phi_hits = pos[disk_idx, 3]
            cart = _spherical_to_cartesian_batch(r_hits, theta_hits, phi_hits)
            mom_hits = mom[disk_idx]
            for obj in self.scene_objects:
                m = (r_hits >= obj.inner_radius) & (r_hits <= obj.outer_radius)
                if not m.any():
                    continue
                local = disk_idx[m]
                image_linear[local] = obj.color(
                    cart[m], photon_momenta=mom_hits[m]
                )

        escape_mask = term == TERM_ESCAPE
        if self.background_sampler is not None and escape_mask.any():
            esc_idx = np.where(escape_mask)[0]
            r_e = pos[esc_idx, 1]
            th_e = pos[esc_idx, 2]
            ph_e = pos[esc_idx, 3]
            cart = _spherical_to_cartesian_batch(r_e, th_e, ph_e)
            norms = np.linalg.norm(cart, axis=1)
            valid = norms > 0.0
            dirs = cart[valid] / norms[valid, None]
            image_linear[esc_idx[valid]] = self.background_sampler.sample(dirs)

        total = time.perf_counter() - start
        print(
            f"  shading + transfers: {total - integ_elapsed:.2f}s; "
            f"total render {total:.2f}s"
        )

        image_uint8 = (np.clip(image_linear, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return image_uint8.reshape(H, W, 3)

    def save_png(self, image: NDArray[np.uint8], path: str | Path) -> None:
        """Save an (H, W, 3) uint8 image to a PNG file, creating parents."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(dest)
        print(f"Saved {image.shape[1]}x{image.shape[0]} image to {dest}")
