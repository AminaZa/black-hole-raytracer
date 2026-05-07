"""Unit tests for :mod:`scene.physical_disk`."""

from __future__ import annotations

import numpy as np
import pytest

from scene.physical_disk import PhysicalAccretionDisk


def _hit_at(r: float, phi: float = 0.0) -> np.ndarray:
    """Cartesian equatorial-plane hit point at radius r."""
    return np.array([[r * np.cos(phi), 0.0, r * np.sin(phi)]], dtype=np.float64)


def test_constructor_validates_radii() -> None:
    with pytest.raises(ValueError):
        PhysicalAccretionDisk(inner_radius=0.0, outer_radius=10.0)
    with pytest.raises(ValueError):
        PhysicalAccretionDisk(inner_radius=5.0, outer_radius=4.0)
    with pytest.raises(ValueError):
        PhysicalAccretionDisk(inner_radius=5.0, outer_radius=10.0, mass=0.0)


def test_color_returns_correct_shape() -> None:
    disk = PhysicalAccretionDisk(inner_radius=6.0, outer_radius=20.0, mass=1.0)
    points = np.stack([_hit_at(6.0)[0], _hit_at(12.0)[0], _hit_at(18.0)[0]])
    rgb = disk.color(points)
    assert rgb.shape == (3, 3)
    assert np.all(rgb >= 0.0)


def test_outside_annulus_is_zero() -> None:
    disk = PhysicalAccretionDisk(inner_radius=6.0, outer_radius=20.0, mass=1.0)
    points = np.stack([_hit_at(2.5)[0], _hit_at(50.0)[0]])
    rgb = disk.color(points)
    assert np.allclose(rgb, 0.0)


def test_inner_pixel_is_bluer_than_outer() -> None:
    """T_obs = √(1 - rs/r) · T_peak (r_isco/r)^(3/4) is hotter at small r,
    so the blackbody-RGB inner pixel sits bluer (higher blue/red ratio) than
    the outer one despite the inner pixel being dimmed by Doppler beaming."""
    disk = PhysicalAccretionDisk(
        inner_radius=6.0, outer_radius=20.0, mass=1.0, t_peak=12000.0
    )
    near = disk.color(_hit_at(6.5))[0]
    far = disk.color(_hit_at(18.0))[0]

    assert near.sum() > 0.0
    assert far.sum() > 0.0
    near_blue_red = near[2] / max(near[0], 1e-9)
    far_blue_red = far[2] / max(far[0], 1e-9)
    assert near_blue_red > far_blue_red


def test_doppler_brightens_approaching_side() -> None:
    """Photon arriving from the prograde side is blueshifted/beamed."""
    disk = PhysicalAccretionDisk(
        inner_radius=6.0, outer_radius=20.0, mass=1.0, t_peak=12000.0
    )
    r = 10.0
    point = _hit_at(r)

    # Build a coordinate-basis null 4-momentum at the hit so that the photon's
    # local-frame direction (after the n_phi_phys = -p_phi_hat / p_t_hat
    # convention used in the disk) is +φ̂. That corresponds to the disk
    # element rotating *toward* the camera.
    f = 1.0 - 2.0 / r
    sqrt_f = np.sqrt(f)
    # Backward-traced p^φ has the opposite sign to the physical photon's
    # local +φ̂ direction, so set p^φ_traced < 0 to get n_phi_phys > 0.
    p_t = 1.0 / sqrt_f
    p_phi_traced = -1.0 / r
    p_approach = np.array([[p_t, 0.0, 0.0, p_phi_traced]])
    p_recede = np.array([[p_t, 0.0, 0.0, -p_phi_traced]])

    rgb_approach = disk.color(point, photon_momenta=p_approach)[0]
    rgb_recede = disk.color(point, photon_momenta=p_recede)[0]

    # Approaching side should be brighter than receding side at the same r.
    assert rgb_approach.sum() > rgb_recede.sum()


def test_radial_photon_only_picks_up_time_dilation() -> None:
    """A radial photon (β·n = 0) has Doppler factor 1/γ, not 1: the orbiting
    fluid still time-dilates an axial photon. The intensity ratio between
    the with-momentum and gravitational-only paths must therefore equal
    (1/γ)^beaming_exponent, not 1."""
    disk = PhysicalAccretionDisk(
        inner_radius=6.0, outer_radius=20.0, mass=1.0, t_peak=12000.0,
        beaming_exponent=3.0,
    )
    r = 10.0
    point = _hit_at(r)
    f = 1.0 - 2.0 / r
    sqrt_f = np.sqrt(f)
    p_radial = np.array([[1.0 / sqrt_f, sqrt_f, 0.0, 0.0]])

    rgb_with = disk.color(point, photon_momenta=p_radial)[0]
    rgb_without = disk.color(point)[0]

    # Radial photon → β·n = 0, but g_dop = 1/γ ≠ 1 (time dilation only).
    # The exact ratio of total RGB also picks up a small spectral-shift
    # correction from the blackbody curve at the redshifted temperature, so
    # the comparison is approximate.
    v_orbit = np.sqrt(1.0 / r)
    inv_gamma = float(np.sqrt(1.0 - v_orbit * v_orbit))
    expected_ratio = inv_gamma**3
    actual_ratio = rgb_with.sum() / rgb_without.sum()
    assert actual_ratio < 1.0
    assert actual_ratio == pytest.approx(expected_ratio, rel=5e-2)
