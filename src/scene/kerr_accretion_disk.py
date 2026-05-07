"""Equatorial Kerr Keplerian disk with frame-dragging Doppler and redshift.

The fluid orbits in the equatorial plane at the prograde (or retrograde)
Kepler angular velocity

    Ω(r) = √M / (r^(3/2) + a √M)

and the disk is cut off below the spin-dependent ISCO. For the marginally
stable circular orbit (Bardeen, Press & Teukolsky 1972),

    Z₁ = 1 + (1 − a²/M²)^(1/3) [(1 + a/M)^(1/3) + (1 − a/M)^(1/3)]
    Z₂ = √(3 a²/M² + Z₁²)
    r_isco = M [3 + Z₂ ∓ √((3 − Z₁)(3 + Z₁ + 2 Z₂))]

with the upper sign for prograde orbits (ISCO 6M → M as a → M) and the
lower sign for retrograde orbits (6M → 9M).

Each photon hit converts the integrator's contravariant 4-momentum at the
disk plane to the conserved scalars E = -p_t and L = p_φ, computes the
fluid's u^t from the on-axis (θ = π/2) Kerr line element and returns the
combined gravitational + Doppler + frame-dragging factor

    g = E / [u^t (E − Ω L)]

which is applied to both the rest-frame temperature and the Doppler-beamed
specific intensity ``∝ g^beaming_exponent``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scene.blackbody import blackbody_rgb


def kerr_isco_radius(mass: float, spin: float, prograde: bool = True) -> float:
    """Marginally stable circular orbit radius in the equatorial plane.

    Parameters
    ----------
    mass:
        Black hole mass M.
    spin:
        Spin parameter a (|a| < M).
    prograde:
        ``True`` for orbits aligned with the BH spin (visually striking
        Doppler-asymmetric ring); ``False`` for counter-rotating orbits.
    """
    if mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}.")
    if abs(spin) >= mass:
        raise ValueError(
            f"|spin| must be strictly less than mass; got spin={spin}, mass={mass}."
        )

    a_over_m: float = spin / mass
    one_minus_a2: float = 1.0 - a_over_m * a_over_m
    cube_root: float = one_minus_a2 ** (1.0 / 3.0)
    z1: float = 1.0 + cube_root * (
        (1.0 + a_over_m) ** (1.0 / 3.0) + (1.0 - a_over_m) ** (1.0 / 3.0)
    )
    z2: float = float(np.sqrt(3.0 * a_over_m * a_over_m + z1 * z1))
    sign: float = -1.0 if prograde else 1.0
    radical: float = float(np.sqrt(max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2))))
    return mass * (3.0 + z2 + sign * radical)


class KerrAccretionDisk:
    """Geometrically thin Kerr disk with relativistic colour shifts.

    Parameters
    ----------
    inner_radius:
        Inner cut-off in geometric units. ``None`` (default) sets it to the
        prograde Kerr ISCO for the configured spin.
    outer_radius:
        Outer cut-off in geometric units.
    mass:
        Black hole mass M.
    spin:
        Spin parameter a (|a| < M).
    t_peak:
        Rest-frame temperature at ``inner_radius`` in Kelvin.
    beaming_exponent:
        Exponent on the redshift factor for relativistic beaming. 3 in the
        narrow-band limit (Cunningham 1975), 4 bolometric.
    prograde:
        Disk rotation sense relative to the BH spin.
    """

    def __init__(
        self,
        inner_radius: float | None = None,
        outer_radius: float = 20.0,
        *,
        mass: float = 1.0,
        spin: float = 0.0,
        t_peak: float = 12000.0,
        beaming_exponent: float = 3.0,
        prograde: bool = True,
    ) -> None:
        if mass <= 0.0:
            raise ValueError(f"mass must be positive, got {mass}.")
        if abs(spin) >= mass:
            raise ValueError(
                f"|spin| must be strictly less than mass; got spin={spin}, mass={mass}."
            )

        self.mass: float = float(mass)
        self.spin: float = float(spin)
        self.prograde: bool = bool(prograde)
        self.t_peak: float = float(t_peak)
        self.beaming_exponent: float = float(beaming_exponent)

        isco: float = kerr_isco_radius(self.mass, self.spin, prograde=self.prograde)
        chosen_inner: float = isco if inner_radius is None else float(inner_radius)
        if chosen_inner <= 0.0:
            raise ValueError(f"inner_radius must be positive, got {chosen_inner}.")
        if outer_radius <= chosen_inner:
            raise ValueError(
                f"outer_radius ({outer_radius}) must exceed inner_radius ({chosen_inner})."
            )

        self.isco: float = isco
        self.inner_radius: float = chosen_inner
        self.outer_radius: float = float(outer_radius)

    def color(
        self,
        hit_positions: NDArray[np.float64],
        photon_momenta: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return linear RGB at equatorial hits, shape (N, 3).

        Parameters
        ----------
        hit_positions:
            Cartesian world coordinates, shape (N, 3); the disk radius is
            ``√(x² + y² + z²)``.
        photon_momenta:
            Coordinate-basis 4-momentum ``(p^t, p^r, p^θ, p^φ)`` at each hit
            in the integrator's BL frame, shape (N, 4). When ``None`` the
            Doppler factor is omitted; only gravitational redshift is kept.
        """
        x = hit_positions[:, 0]
        y = hit_positions[:, 1]
        z = hit_positions[:, 2]
        r = np.sqrt(x * x + y * y + z * z)
        r_safe = np.clip(r, self.inner_radius, self.outer_radius)

        t_emit = self.t_peak * (self.inner_radius / r_safe) ** 0.75

        if photon_momenta is None:
            # No photon info: use the static-observer redshift only. At the
            # equator Σ = r² so -g_tt = 1 - 2M/r, the same as Schwarzschild.
            f_static = np.clip(1.0 - 2.0 * self.mass / r_safe, 0.0, 1.0)
            g_total = np.sqrt(f_static)
            t_obs = t_emit * g_total
            rgb = blackbody_rgb(t_obs) * (g_total**self.beaming_exponent)[:, np.newaxis]
        else:
            M: float = self.mass
            a_eff: float = self.spin if self.prograde else -self.spin

            r2 = r_safe * r_safe
            # Equatorial Kerr metric components (θ = π/2 → Σ = r², sin²θ = 1).
            g_tt = -(1.0 - 2.0 * M / r_safe)
            g_tphi = -2.0 * M * a_eff / r_safe
            g_phiphi = r2 + a_eff * a_eff + 2.0 * M * a_eff * a_eff / r_safe

            p_t_up = photon_momenta[:, 0]
            p_phi_up = photon_momenta[:, 3]
            p_t_low = g_tt * p_t_up + g_tphi * p_phi_up
            p_phi_low = g_tphi * p_t_up + g_phiphi * p_phi_up

            energy = -p_t_low
            ang_mom = p_phi_low

            sqrt_M = np.sqrt(M)
            omega = sqrt_M / (r_safe**1.5 + a_eff * sqrt_M)

            # u^t² · (g_tt + 2 g_tφ Ω + g_φφ Ω²) = -1 → u^t = 1/√(-(...)).
            denom = -(g_tt + 2.0 * g_tphi * omega + g_phiphi * omega * omega)
            denom_safe = np.clip(denom, 1e-12, None)
            u_t = 1.0 / np.sqrt(denom_safe)

            energy_in_fluid = u_t * (energy - omega * ang_mom)
            energy_in_fluid = np.where(
                np.abs(energy_in_fluid) < 1e-12, 1e-12, energy_in_fluid
            )
            # Camera assumed asymptotic; at the camera frequency = E (conserved).
            g_factor = energy / energy_in_fluid
            g_factor = np.clip(g_factor, 1e-6, 50.0)

            t_obs = t_emit * g_factor
            beaming = g_factor**self.beaming_exponent
            rgb = blackbody_rgb(t_obs) * beaming[:, np.newaxis]

        in_annulus = (r >= self.inner_radius) & (r <= self.outer_radius)
        return np.where(in_annulus[:, np.newaxis], rgb, 0.0)
