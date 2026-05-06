"""Unit tests for ``metrics.schwarzschild.SchwarzschildMetric``."""

from __future__ import annotations

import numpy as np
import pytest

from metrics.schwarzschild import SchwarzschildMetric


def test_schwarzschild_radius_is_twice_mass() -> None:
    metric = SchwarzschildMetric(mass=1.5)
    assert metric.rs == pytest.approx(3.0)


def test_invalid_mass_raises() -> None:
    with pytest.raises(ValueError):
        SchwarzschildMetric(mass=0.0)
    with pytest.raises(ValueError):
        SchwarzschildMetric(mass=-1.0)


def test_metric_components_at_known_point() -> None:
    metric = SchwarzschildMetric(mass=1.0)
    position = np.array([0.0, 10.0, 0.5 * np.pi, 0.0])
    g = metric.metric_tensor(position)

    f = 1.0 - 0.2  # 1 - rs/r with rs=2, r=10
    assert g[0, 0] == pytest.approx(-f)
    assert g[1, 1] == pytest.approx(1.0 / f)
    assert g[2, 2] == pytest.approx(100.0)
    assert g[3, 3] == pytest.approx(100.0)  # sin²(π/2) = 1

    off_diagonal = g - np.diag(np.diag(g))
    assert np.allclose(off_diagonal, 0.0)


def test_metric_tensor_shape() -> None:
    metric = SchwarzschildMetric()
    position = np.array([0.0, 5.0, 1.0, 0.0])
    g = metric.metric_tensor(position)
    assert g.shape == (4, 4)


def test_christoffel_shape_and_symmetry() -> None:
    metric = SchwarzschildMetric()
    position = np.array([0.0, 7.5, 1.2, 0.4])
    gamma = metric.christoffel_symbols(position)

    assert gamma.shape == (4, 4, 4)
    # Γ^μ_{αβ} must be symmetric in the lower indices.
    assert np.allclose(gamma, np.transpose(gamma, (0, 2, 1)))


def test_christoffel_known_values() -> None:
    """Spot-check non-zero connection coefficients at r = 10, θ = π/2, M = 1."""
    metric = SchwarzschildMetric(mass=1.0)
    position = np.array([0.0, 10.0, 0.5 * np.pi, 0.0])
    gamma = metric.christoffel_symbols(position)

    expected_t_tr = 1.0 / (10.0 * 8.0)  # M / (r (r - rs))
    assert gamma[0, 0, 1] == pytest.approx(expected_t_tr)
    assert gamma[0, 1, 0] == pytest.approx(expected_t_tr)

    assert gamma[1, 0, 0] == pytest.approx(0.008)  # (M/r²)(1 - rs/r)
    assert gamma[1, 1, 1] == pytest.approx(-expected_t_tr)
    assert gamma[1, 2, 2] == pytest.approx(-8.0)
    assert gamma[1, 3, 3] == pytest.approx(-8.0)  # sin²(π/2) = 1

    assert gamma[2, 1, 2] == pytest.approx(0.1)
    assert gamma[2, 3, 3] == pytest.approx(0.0)  # cos(π/2) = 0

    assert gamma[3, 1, 3] == pytest.approx(0.1)
    assert gamma[3, 2, 3] == pytest.approx(0.0)  # cot(π/2) = 0


def test_christoffel_off_equator_uses_cot_theta() -> None:
    metric = SchwarzschildMetric(mass=1.0)
    theta = np.pi / 3.0
    position = np.array([0.0, 10.0, theta, 0.0])
    gamma = metric.christoffel_symbols(position)

    expected_phi_theta_phi = float(np.cos(theta) / np.sin(theta))
    assert gamma[3, 2, 3] == pytest.approx(expected_phi_theta_phi)
    assert gamma[3, 3, 2] == pytest.approx(expected_phi_theta_phi)

    expected_theta_phi_phi = float(-np.sin(theta) * np.cos(theta))
    assert gamma[2, 3, 3] == pytest.approx(expected_theta_phi_phi)
