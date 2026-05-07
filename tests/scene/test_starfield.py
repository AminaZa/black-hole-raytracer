"""Unit tests for :mod:`scene.starfield`."""

from __future__ import annotations

import numpy as np
import pytest

from scene.starfield import Starfield


def test_constructor_validates_inputs() -> None:
    with pytest.raises(ValueError):
        Starfield(n_stars=0)
    with pytest.raises(ValueError):
        Starfield(n_stars=10, star_radius_deg=0.0)


def test_same_seed_is_deterministic() -> None:
    a = Starfield(n_stars=200, seed=7)
    b = Starfield(n_stars=200, seed=7)
    np.testing.assert_array_equal(a.directions, b.directions)
    np.testing.assert_array_equal(a.colors, b.colors)


def test_different_seeds_differ() -> None:
    a = Starfield(n_stars=200, seed=1)
    b = Starfield(n_stars=200, seed=2)
    assert not np.allclose(a.directions, b.directions)


def test_directions_are_unit_vectors() -> None:
    s = Starfield(n_stars=500, seed=0)
    norms = np.linalg.norm(s.directions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_sample_shape_and_background() -> None:
    s = Starfield(n_stars=10, seed=0, star_radius_deg=0.001)
    # A handful of random query directions are vanishingly unlikely to land
    # on any of 10 stars at a 0.001° radius, so they should all read background.
    rng = np.random.default_rng(99)
    q = rng.normal(size=(20, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rgb = s.sample(q)
    assert rgb.shape == (20, 3)
    np.testing.assert_allclose(rgb, np.broadcast_to(s.background, rgb.shape))


def test_query_at_star_direction_lights_pixel() -> None:
    s = Starfield(n_stars=50, seed=3, star_radius_deg=1.0)
    target = s.directions[7].copy()
    rgb = s.sample(target[np.newaxis, :])[0]
    # A direct hit at peak intensity is background + star colour.
    expected = s.background + s.colors[7]
    np.testing.assert_allclose(rgb, expected, atol=1e-12)


def test_empty_input_returns_empty_output() -> None:
    s = Starfield(n_stars=10, seed=0)
    out = s.sample(np.zeros((0, 3), dtype=np.float64))
    assert out.shape == (0, 3)
