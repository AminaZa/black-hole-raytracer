"""Unit tests for ``geodesic.integrator.GeodesicIntegrator``."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from geodesic.integrator import GeodesicIntegrator
from metrics.schwarzschild import SchwarzschildMetric


def _null_momentum(
    metric: SchwarzschildMetric,
    r: float,
    theta: float,
    n_r: float,
    n_theta: float,
    n_phi: float,
) -> NDArray[np.float64]:
    """Build a null 4-momentum for a static observer at (r, θ)."""
    f = 1.0 - metric.rs / r
    sqrt_f = np.sqrt(f)
    return np.array(
        [
            1.0 / sqrt_f,
            n_r * sqrt_f,
            n_theta / r,
            n_phi / (r * np.sin(theta)),
        ],
        dtype=np.float64,
    )


def test_radial_inward_photon_falls_into_horizon() -> None:
    metric = SchwarzschildMetric(mass=1.0)
    integrator = GeodesicIntegrator(metric, r_max=50.0)
    position = np.array([0.0, 10.0, 0.5 * np.pi, 0.0])
    momentum = _null_momentum(metric, 10.0, 0.5 * np.pi, n_r=-1.0, n_theta=0.0, n_phi=0.0)

    result = integrator.integrate(position, momentum, max_steps=10_000)

    assert result.termination == "horizon"
    assert result.final_position[1] <= metric.rs + integrator.horizon_eps + 1e-6


def test_radial_outward_photon_escapes() -> None:
    metric = SchwarzschildMetric(mass=1.0)
    integrator = GeodesicIntegrator(metric, r_max=50.0)
    position = np.array([0.0, 10.0, 0.5 * np.pi, 0.0])
    momentum = _null_momentum(metric, 10.0, 0.5 * np.pi, n_r=1.0, n_theta=0.0, n_phi=0.0)

    result = integrator.integrate(position, momentum, max_steps=10_000)

    assert result.termination == "escape"
    assert result.final_position[1] >= integrator.r_max


def test_energy_is_conserved_along_geodesic() -> None:
    """E = (1 - rs/r) p^t should be conserved by the geodesic flow."""
    metric = SchwarzschildMetric(mass=1.0)
    integrator = GeodesicIntegrator(
        metric, r_max=80.0, base_step=0.1, near_field_factor=0.05
    )
    r0 = 20.0
    theta0 = 0.5 * np.pi
    momentum = _null_momentum(metric, r0, theta0, n_r=-0.4, n_theta=0.0, n_phi=0.9165)
    position = np.array([0.0, r0, theta0, 0.0])

    f0 = 1.0 - metric.rs / r0
    energy_initial = f0 * momentum[0]

    result = integrator.integrate(position, momentum, max_steps=20_000)

    r_f = float(result.final_position[1])
    f_f = 1.0 - metric.rs / r_f
    energy_final = f_f * float(result.final_momentum[0])

    rel_err = abs(energy_final - energy_initial) / abs(energy_initial)
    assert rel_err < 1e-3, f"energy drift {rel_err:.2e} too large"


def test_axial_angular_momentum_is_conserved() -> None:
    """L_z = r² sin²θ p^φ should be conserved (axisymmetry)."""
    metric = SchwarzschildMetric(mass=1.0)
    integrator = GeodesicIntegrator(
        metric, r_max=80.0, base_step=0.1, near_field_factor=0.05
    )
    r0 = 20.0
    theta0 = 0.5 * np.pi
    momentum = _null_momentum(metric, r0, theta0, n_r=-0.3, n_theta=0.0, n_phi=0.9539)
    position = np.array([0.0, r0, theta0, 0.0])

    sin0 = float(np.sin(theta0))
    L_initial = r0 * r0 * sin0 * sin0 * momentum[3]

    result = integrator.integrate(position, momentum, max_steps=20_000)

    r_f = float(result.final_position[1])
    th_f = float(result.final_position[2])
    sin_f = float(np.sin(th_f))
    L_final = r_f * r_f * sin_f * sin_f * float(result.final_momentum[3])

    rel_err = abs(L_final - L_initial) / abs(L_initial)
    assert rel_err < 1e-3, f"angular-momentum drift {rel_err:.2e} too large"


def test_disk_termination_within_annulus() -> None:
    """A photon aimed downward through an in-range r terminates as 'disk'."""
    metric = SchwarzschildMetric(mass=1.0)
    integrator = GeodesicIntegrator(metric, r_max=50.0)

    # Start above the equatorial plane at moderate r and aim straight down (-θ̂).
    r0 = 15.0
    theta0 = 0.4 * np.pi  # above equator (θ < π/2)
    momentum = _null_momentum(metric, r0, theta0, n_r=0.0, n_theta=1.0, n_phi=0.0)
    position = np.array([0.0, r0, theta0, 0.0])

    result = integrator.integrate(
        position, momentum, max_steps=10_000, disk_inner=6.0, disk_outer=24.0
    )

    assert result.termination == "disk"
    assert result.final_position[2] == np.float64(np.pi / 2) or abs(
        float(result.final_position[2]) - 0.5 * np.pi
    ) < 1e-6
    r_hit = float(result.final_position[1])
    assert 6.0 <= r_hit <= 24.0


def test_disk_crossing_outside_annulus_does_not_terminate() -> None:
    """An equatorial crossing outside [inner, outer] should not stop the ray."""
    metric = SchwarzschildMetric(mass=1.0)
    integrator = GeodesicIntegrator(metric, r_max=50.0)

    r0 = 30.0
    theta0 = 0.4 * np.pi
    momentum = _null_momentum(metric, r0, theta0, n_r=1.0, n_theta=0.5, n_phi=0.0)
    momentum_norm = momentum.copy()  # photon moving outward and down

    position = np.array([0.0, r0, theta0, 0.0])
    result = integrator.integrate(
        position,
        momentum_norm,
        max_steps=10_000,
        disk_inner=6.0,
        disk_outer=24.0,
    )

    # The photon is heading outward and tilts down; it will cross θ = π/2 at
    # r > 24 and then escape, so disk termination must not fire.
    assert result.termination != "disk"


class _FlatStubMetric:
    """A spherical stub metric with zero connection, for protocol testing."""

    rs: float = 0.0

    def christoffel_symbols(
        self, position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.zeros((4, 4, 4), dtype=np.float64)


def test_integrator_is_metric_agnostic() -> None:
    """With Γ = 0, momentum is conserved exactly and position evolves linearly."""
    integrator = GeodesicIntegrator(
        _FlatStubMetric(), r_max=50.0, base_step=0.5, near_field_factor=1.0
    )
    position = np.array([0.0, 10.0, 1.0, 0.5])
    momentum = np.array([1.0, 0.7, 0.2, 0.1])

    result = integrator.integrate(position, momentum, max_steps=2_000)

    assert np.allclose(result.final_momentum, momentum, atol=1e-12)
    expected_position = position + result.affine_lambda * momentum
    assert np.allclose(result.final_position, expected_position, atol=1e-9)
