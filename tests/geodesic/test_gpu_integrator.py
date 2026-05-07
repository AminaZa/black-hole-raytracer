"""Tests for the batched GPU geodesic integrator.

The integrator is array-module agnostic — these tests run it on numpy
arrays so they pass without a GPU. A separate cupy-on-GPU smoke test
runs only when CuPy and a working CUDA device are available.
"""

from __future__ import annotations

import numpy as np
import pytest

from geodesic.gpu_integrator import (
    GpuGeodesicIntegrator,
    TERM_DISK,
    TERM_ESCAPE,
    TERM_HORIZON,
)
from geodesic.integrator import GeodesicIntegrator
from metrics.kerr import KerrMetric
from metrics.schwarzschild import SchwarzschildMetric
from utils.cuda_loader import gpu_available


def _null_p(metric, r: float, theta: float, n_r: float, n_th: float, n_ph: float):
    f = 1.0 - 2.0 * metric.mass / r
    sf = float(np.sqrt(f))
    return np.array(
        [1.0 / sf, n_r * sf, n_th / r, n_ph / (r * np.sin(theta))],
        dtype=np.float64,
    )


def test_batch_matches_per_ray_on_schwarzschild() -> None:
    """Three canonical rays — horizon, escape, disk — match the CPU integrator."""
    metric = SchwarzschildMetric(mass=1.0)
    cpu = GeodesicIntegrator(metric, r_max=50.0)
    gpu = GpuGeodesicIntegrator(metric, r_max=50.0)

    positions = np.array(
        [
            [0.0, 10.0, 0.5 * np.pi, 0.0],
            [0.0, 10.0, 0.5 * np.pi, 0.0],
            [0.0, 15.0, 0.4 * np.pi, 0.0],
        ]
    )
    momenta = np.stack(
        [
            _null_p(metric, 10.0, 0.5 * np.pi, -1.0, 0.0, 0.0),
            _null_p(metric, 10.0, 0.5 * np.pi, +1.0, 0.0, 0.0),
            _null_p(metric, 15.0, 0.4 * np.pi, 0.0, 1.0, 0.0),
        ]
    )

    cpu_results = [
        cpu.integrate(
            positions[i], momenta[i], max_steps=10000, disk_inner=6.0, disk_outer=24.0
        )
        for i in range(3)
    ]
    gpu_res = gpu.integrate_batch(
        positions, momenta, max_steps=10000, disk_inner=6.0, disk_outer=24.0
    )

    expected_codes = [TERM_HORIZON, TERM_ESCAPE, TERM_DISK]
    assert list(gpu_res.termination) == expected_codes
    for i, name in enumerate(("horizon", "escape", "disk")):
        assert cpu_results[i].termination == name
    # Disk-hit r and 4-momentum at the crossing should agree to many digits.
    np.testing.assert_allclose(
        gpu_res.final_position[2], cpu_results[2].final_position, atol=1e-10
    )
    np.testing.assert_allclose(
        gpu_res.final_momentum[2], cpu_results[2].final_momentum, atol=1e-10
    )


def test_batch_handles_kerr_metric() -> None:
    """Kerr radial inward photon hits the outer horizon r_+."""
    metric = KerrMetric(mass=1.0, spin=0.5)
    gpu = GpuGeodesicIntegrator(metric, r_max=50.0)

    r0 = 10.0
    theta0 = 0.5 * np.pi
    positions = np.array([[0.0, r0, theta0, 0.0]])
    momenta = _null_p(metric, r0, theta0, -1.0, 0.0, 0.0)[None, :]

    res = gpu.integrate_batch(positions, momenta, max_steps=10000)
    assert int(res.termination[0]) == TERM_HORIZON
    assert res.final_position[0, 1] <= metric.rs + gpu.horizon_eps + 1e-6


@pytest.mark.skipif(not gpu_available, reason="CuPy / CUDA not available")
def test_gpu_path_matches_numpy_path() -> None:
    """The same integrator on cupy arrays should agree with numpy arrays."""
    import cupy as cp  # type: ignore

    metric = KerrMetric(mass=1.0, spin=0.7)
    gpu = GpuGeodesicIntegrator(
        metric, r_max=80.0, base_step=0.25, near_field_factor=0.1, far_field_factor=2.0
    )

    rng = np.random.default_rng(0)
    n = 256
    r0 = 25.0
    theta0 = np.radians(75.0)
    sin_t = np.sin(theta0)
    f = 1.0 - 2.0 * metric.mass / r0
    sqrt_f = float(np.sqrt(f))
    n_dir = rng.standard_normal((n, 3))
    n_dir /= np.linalg.norm(n_dir, axis=1, keepdims=True)
    momenta = np.empty((n, 4))
    momenta[:, 0] = 1.0 / sqrt_f
    momenta[:, 1] = n_dir[:, 0] * sqrt_f
    momenta[:, 2] = n_dir[:, 1] / r0
    momenta[:, 3] = n_dir[:, 2] / (r0 * sin_t)
    positions = np.tile(np.array([0.0, r0, theta0, 0.0]), (n, 1))

    cpu_res = gpu.integrate_batch(positions, momenta, max_steps=2000)
    gpu_res = gpu.integrate_batch(
        cp.asarray(positions), cp.asarray(momenta), max_steps=2000
    )
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(cpu_res.termination, cp.asnumpy(gpu_res.termination))
    np.testing.assert_allclose(
        cpu_res.final_position,
        cp.asnumpy(gpu_res.final_position),
        atol=1e-9,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        cpu_res.final_momentum,
        cp.asnumpy(gpu_res.final_momentum),
        atol=1e-9,
        rtol=1e-9,
    )
