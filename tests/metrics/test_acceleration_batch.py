"""Tests for the batched acceleration on Schwarzschild and Kerr.

The batched form must match the scalar :meth:`acceleration` element-wise.
The numpy path is tested unconditionally; the cupy path is run only when
CuPy and a working CUDA device are available.
"""

from __future__ import annotations

import numpy as np
import pytest

from metrics.kerr import KerrMetric
from metrics.schwarzschild import SchwarzschildMetric
from utils.cuda_loader import gpu_available


def _random_states(n: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    positions = np.column_stack(
        [
            np.zeros(n),
            5.0 + 10.0 * rng.random(n),
            0.4 + 0.5 * np.pi * rng.random(n),
            2.0 * np.pi * rng.random(n),
        ]
    )
    momenta = rng.standard_normal((n, 4))
    return positions, momenta


@pytest.mark.parametrize(
    "metric",
    [
        SchwarzschildMetric(mass=1.0),
        KerrMetric(mass=1.0, spin=0.0),
        KerrMetric(mass=1.0, spin=0.5),
        KerrMetric(mass=1.0, spin=0.99),
    ],
    ids=["schwarzschild", "kerr-a0", "kerr-a05", "kerr-a099"],
)
def test_batch_matches_scalar_on_numpy(metric) -> None:
    positions, momenta = _random_states(64)
    batch = metric.acceleration_batch(positions, momenta)
    scalar = np.stack(
        [metric.acceleration(positions[i], momenta[i]) for i in range(len(positions))]
    )
    assert batch.shape == (64, 4)
    assert np.allclose(batch, scalar, atol=1e-13)


@pytest.mark.skipif(not gpu_available, reason="CuPy / CUDA not available")
@pytest.mark.parametrize(
    "metric",
    [
        SchwarzschildMetric(mass=1.0),
        KerrMetric(mass=1.0, spin=0.99),
    ],
    ids=["schwarzschild", "kerr"],
)
def test_batch_cupy_matches_numpy(metric) -> None:
    import cupy as cp  # type: ignore

    positions, momenta = _random_states(8192)
    cpu = metric.acceleration_batch(positions, momenta)
    gpu = metric.acceleration_batch(cp.asarray(positions), cp.asarray(momenta))
    cp.cuda.Stream.null.synchronize()
    diff = float(np.max(np.abs(cpu - cp.asnumpy(gpu))))
    # Allow float64 round-off (different operation order on the device).
    assert diff < 1e-12
