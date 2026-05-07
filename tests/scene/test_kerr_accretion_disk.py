"""Unit tests for ``scene.kerr_accretion_disk``."""

from __future__ import annotations

import numpy as np
import pytest

from scene.kerr_accretion_disk import KerrAccretionDisk, kerr_isco_radius


def test_isco_at_zero_spin_is_6M() -> None:
    assert kerr_isco_radius(1.0, 0.0) == pytest.approx(6.0)


def test_isco_prograde_shrinks_with_spin() -> None:
    radii = [kerr_isco_radius(1.0, a, prograde=True) for a in (0.0, 0.5, 0.9, 0.999)]
    assert radii == sorted(radii, reverse=True)
    assert radii[-1] < 1.5  # near-extremal pushes ISCO toward M


def test_isco_retrograde_larger_than_prograde() -> None:
    pro = kerr_isco_radius(1.0, 0.5, prograde=True)
    retro = kerr_isco_radius(1.0, 0.5, prograde=False)
    assert retro > pro


def test_isco_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        kerr_isco_radius(0.0, 0.0)
    with pytest.raises(ValueError):
        kerr_isco_radius(1.0, 1.0)


def test_disk_defaults_to_isco() -> None:
    disk = KerrAccretionDisk(outer_radius=20.0, mass=1.0, spin=0.5)
    assert disk.inner_radius == pytest.approx(disk.isco)


def test_disk_invalid_radii_raise() -> None:
    with pytest.raises(ValueError):
        KerrAccretionDisk(inner_radius=5.0, outer_radius=4.0, mass=1.0, spin=0.5)
    with pytest.raises(ValueError):
        KerrAccretionDisk(inner_radius=-1.0, outer_radius=4.0, mass=1.0, spin=0.5)


def test_color_zeros_outside_annulus() -> None:
    disk = KerrAccretionDisk(
        inner_radius=4.0, outer_radius=12.0, mass=1.0, spin=0.5
    )
    hits = np.array(
        [
            [2.0, 0.0, 0.0],   # inside ISCO
            [8.0, 0.0, 0.0],   # in annulus
            [20.0, 0.0, 0.0],  # outside outer
        ],
        dtype=np.float64,
    )
    rgb = disk.color(hits)
    assert rgb.shape == (3, 3)
    assert np.all(rgb[0] == 0.0)
    assert np.any(rgb[1] > 0.0)
    assert np.all(rgb[2] == 0.0)


def test_color_no_momentum_path_uses_static_redshift() -> None:
    """T_obs = √(1 - 2M/r) · T_peak (r_isco/r)^(3/4) is hotter at small r;
    the blackbody-RGB inner pixel sits bluer than the outer one."""
    disk = KerrAccretionDisk(
        inner_radius=6.0, outer_radius=20.0, mass=1.0, spin=0.0,
    )
    hits = np.array([[7.0, 0.0, 0.0], [15.0, 0.0, 0.0]], dtype=np.float64)
    rgb = disk.color(hits)
    inner_blue_red = rgb[0, 2] / max(rgb[0, 0], 1e-9)
    outer_blue_red = rgb[1, 2] / max(rgb[1, 0], 1e-9)
    assert inner_blue_red > outer_blue_red


def test_doppler_breaks_left_right_symmetry_for_spinning_disk() -> None:
    """For a prograde disk, photons from opposite sides have opposite Doppler signs."""
    disk = KerrAccretionDisk(
        inner_radius=6.0, outer_radius=20.0, mass=1.0, spin=0.5, prograde=True
    )
    # Two equatorial hits at the same radius, one on each side of the BH (φ=0 and φ=π).
    r = 8.0
    hit_a = np.array([[r, 0.0, 0.0]])
    hit_b = np.array([[-r, 0.0, 0.0]])
    # Toy backward-traced 4-momentum aimed inward in r and with opposite p^φ.
    p_in = 1.0 / np.sqrt(1.0 - 2.0 / r)
    p_a = np.array([[p_in, -0.4, 0.0, +0.05]])
    p_b = np.array([[p_in, -0.4, 0.0, -0.05]])
    rgb_a = disk.color(hit_a, photon_momenta=p_a)
    rgb_b = disk.color(hit_b, photon_momenta=p_b)
    assert not np.allclose(rgb_a, rgb_b, atol=1e-3)
