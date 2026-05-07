"""Unit tests for ``metrics.kerr.KerrMetric``."""

from __future__ import annotations

import numpy as np
import pytest

from geodesic.integrator import GeodesicIntegrator
from metrics.kerr import KerrMetric
from metrics.schwarzschild import SchwarzschildMetric


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        KerrMetric(mass=0.0, spin=0.0)
    with pytest.raises(ValueError):
        KerrMetric(mass=-1.0, spin=0.0)
    with pytest.raises(ValueError):
        KerrMetric(mass=1.0, spin=1.0)
    with pytest.raises(ValueError):
        KerrMetric(mass=1.0, spin=-1.0)
    with pytest.raises(ValueError):
        KerrMetric(mass=1.0, spin=2.0)


def test_outer_horizon_at_zero_spin_is_2M() -> None:
    metric = KerrMetric(mass=1.5, spin=0.0)
    assert metric.r_plus == pytest.approx(3.0)
    assert metric.rs == pytest.approx(3.0)


def test_outer_horizon_shrinks_with_spin() -> None:
    for a in [0.0, 0.5, 0.9, 0.99]:
        metric = KerrMetric(mass=1.0, spin=a)
        expected = 1.0 + np.sqrt(1.0 - a * a)
        assert metric.r_plus == pytest.approx(expected)
        assert metric.r_minus == pytest.approx(1.0 - np.sqrt(1.0 - a * a))


def test_metric_at_zero_spin_matches_schwarzschild() -> None:
    """g_{μν}(a=0) must equal Schwarzschild g_{μν} at every event."""
    kerr = KerrMetric(mass=1.0, spin=0.0)
    schw = SchwarzschildMetric(mass=1.0)
    for r in (3.0, 5.0, 12.5):
        for theta in (0.4, 1.0, 1.4):
            position = np.array([0.0, r, theta, 0.7])
            assert np.allclose(
                kerr.metric_tensor(position),
                schw.metric_tensor(position),
                atol=1e-12,
            )


def test_christoffel_at_zero_spin_matches_schwarzschild() -> None:
    """Γ(a=0) must equal Schwarzschild Γ at every event."""
    kerr = KerrMetric(mass=1.0, spin=0.0)
    schw = SchwarzschildMetric(mass=1.0)
    for r in (4.0, 7.0, 14.0):
        for theta in (0.3, 0.8, 1.5):
            position = np.array([0.0, r, theta, 0.0])
            gamma_kerr = kerr.christoffel_symbols(position)
            gamma_schw = schw.christoffel_symbols(position)
            assert np.allclose(gamma_kerr, gamma_schw, atol=1e-10)


def test_off_diagonal_at_equator_for_nonzero_spin() -> None:
    """g_{tφ} = -2Mar sin²θ / Σ; nonzero exactly when a ≠ 0."""
    metric = KerrMetric(mass=1.0, spin=0.5)
    position = np.array([0.0, 5.0, 0.5 * np.pi, 0.0])
    g = metric.metric_tensor(position)
    # Σ = r² + a²cos²θ = 25; sin²θ = 1 → g_tφ = -2·1·0.5·5·1/25 = -0.2
    expected = -2.0 * 1.0 * 0.5 * 5.0 * 1.0 / 25.0
    assert g[0, 3] == pytest.approx(expected)
    assert g[3, 0] == pytest.approx(expected)


def test_inverse_metric_consistency() -> None:
    metric = KerrMetric(mass=1.0, spin=0.7)
    position = np.array([0.0, 8.0, 0.6, 0.3])
    g = metric.metric_tensor(position)
    g_inv = metric.inverse_metric(position)
    assert np.allclose(g @ g_inv, np.eye(4), atol=1e-10)


def test_christoffel_shape_and_lower_index_symmetry() -> None:
    metric = KerrMetric(mass=1.0, spin=0.6)
    position = np.array([0.0, 8.0, 1.1, 0.2])
    gamma = metric.christoffel_symbols(position)
    assert gamma.shape == (4, 4, 4)
    assert np.allclose(gamma, np.transpose(gamma, (0, 2, 1)), atol=1e-12)


def test_acceleration_matches_christoffel_einsum() -> None:
    metric = KerrMetric(mass=1.0, spin=0.7)
    position = np.array([0.0, 9.0, 1.2, 0.4])
    momentum = np.array([1.5, -0.3, 0.1, 0.05])

    a_fast = metric.acceleration(position, momentum)
    gamma = metric.christoffel_symbols(position)
    a_slow = -np.einsum("mab,a,b->m", gamma, momentum, momentum)
    assert np.allclose(a_fast, a_slow, atol=1e-12)


def test_integrator_works_unmodified_with_kerr() -> None:
    """A radially infalling photon in Kerr terminates on the outer horizon."""
    metric = KerrMetric(mass=1.0, spin=0.5)
    integrator = GeodesicIntegrator(metric, r_max=50.0)
    r0 = 10.0
    theta0 = 0.5 * np.pi
    f = 1.0 - metric.rs / r0
    sqrt_f = float(np.sqrt(f))
    momentum = np.array([1.0 / sqrt_f, -1.0 * sqrt_f, 0.0, 0.0])
    position = np.array([0.0, r0, theta0, 0.0])

    result = integrator.integrate(position, momentum, max_steps=10_000)

    assert result.termination == "horizon"
    assert result.final_position[1] <= metric.rs + integrator.horizon_eps + 1e-6


def test_axial_angular_momentum_conserved_in_kerr() -> None:
    """L_z = p_φ should drift only at integration error level (axisymmetry)."""
    metric = KerrMetric(mass=1.0, spin=0.6)
    integrator = GeodesicIntegrator(
        metric, r_max=80.0, base_step=0.1, near_field_factor=0.05
    )
    r0 = 20.0
    theta0 = 0.5 * np.pi

    g0 = metric.metric_tensor(np.array([0.0, r0, theta0, 0.0]))
    f = 1.0 - 2.0 * metric.mass / r0
    sqrt_f = float(np.sqrt(f))
    p0 = np.array(
        [1.0 / sqrt_f, -0.3 * sqrt_f, 0.0, 0.9539 / r0],
        dtype=np.float64,
    )
    L_initial = g0[3, 0] * p0[0] + g0[3, 3] * p0[3]

    position = np.array([0.0, r0, theta0, 0.0])
    result = integrator.integrate(position, p0, max_steps=20_000)

    g_f = metric.metric_tensor(result.final_position)
    L_final = g_f[3, 0] * result.final_momentum[0] + g_f[3, 3] * result.final_momentum[3]

    rel_err = abs(L_final - L_initial) / max(abs(L_initial), 1e-12)
    assert rel_err < 1e-2, f"angular-momentum drift {rel_err:.2e} too large"
