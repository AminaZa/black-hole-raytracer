"""Physically motivated accretion disk in the equatorial plane.

A geometrically thin Keplerian disk with a simplified Novikov-Thorne radial
profile, gravitational redshift from the static metric, and a relativistic
Doppler factor for the orbiting fluid.

Per-ray pipeline:

1.  ``T_emit(r) = T_peak * (r_isco / r)^(3/4)`` — temperature in the rest frame
    of an orbiting fluid element.
2.  Gravitational redshift factor ``g_grav = sqrt(1 - rs/r)`` applied to the
    frequency the static observer at ``r`` would see.
3.  Doppler factor ``g_dop = 1 / (γ (1 - β · n̂_phys))`` where
    ``β = sqrt(M/r) φ̂`` is the locally measured orbital velocity and
    ``n̂_phys`` is the photon direction in the static observer's local
    orthonormal frame, pointing from the disk towards the camera (opposite to
    the backward-traced ray).
4.  Combined frequency factor ``g = g_grav * g_dop`` shifts the temperature:
    ``T_obs = g * T_emit``.
5.  Beaming: observed specific intensity scales as ``g^beaming_exponent``
    (3 in the narrow-band limit, 4 bolometric).
6.  ``T_obs`` is mapped to linear RGB via :func:`blackbody_rgb` and multiplied
    by the beaming factor.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scene.blackbody import blackbody_rgb


class PhysicalAccretionDisk:
    """Keplerian accretion disk with Doppler + gravitational shifts.

    Parameters
    ----------
    inner_radius:
        Inner cut-off (typically the ISCO at ``3 rs = 6M`` for Schwarzschild).
    outer_radius:
        Outer cut-off in geometric units.
    mass:
        Black hole mass M (geometric units; ``rs = 2M``).
    t_peak:
        Rest-frame temperature at ``inner_radius`` in Kelvin. Lower values
        push the colour palette redder; higher values bluer.
    beaming_exponent:
        Exponent on the Doppler factor for relativistic beaming. Use 3 for a
        narrow band (Cunningham 1975) or 4 for bolometric.
    """

    def __init__(
        self,
        inner_radius: float,
        outer_radius: float,
        *,
        mass: float = 1.0,
        t_peak: float = 12000.0,
        beaming_exponent: float = 3.0,
    ) -> None:
        if inner_radius <= 0.0:
            raise ValueError(f"inner_radius must be positive, got {inner_radius}.")
        if outer_radius <= inner_radius:
            raise ValueError(
                f"outer_radius ({outer_radius}) must exceed inner_radius ({inner_radius})."
            )
        if mass <= 0.0:
            raise ValueError(f"mass must be positive, got {mass}.")

        self.inner_radius: float = float(inner_radius)
        self.outer_radius: float = float(outer_radius)
        self.mass: float = float(mass)
        self.rs: float = 2.0 * self.mass
        self.t_peak: float = float(t_peak)
        self.beaming_exponent: float = float(beaming_exponent)

    def color(
        self,
        hit_positions: NDArray[np.float64],
        photon_momenta: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return linear RGB for hits in the disk annulus.

        Parameters
        ----------
        hit_positions:
            Shape (N, 3), Cartesian world coordinates of the equatorial
            crossings. ``r = sqrt(x² + y² + z²)`` is the disk radius.
        photon_momenta:
            Shape (N, 4), coordinate-basis 4-momentum ``(p^t, p^r, p^θ, p^φ)``
            at the hit, expressed in the disk's spherical coordinates. When
            ``None`` the Doppler effect is omitted (gravitational redshift
            still applies).

        Returns
        -------
        rgb:
            Shape (N, 3), linear RGB. Pre-clipped to the annulus: pixels whose
            radius lies outside ``[inner_radius, outer_radius]`` are zeroed.
        """
        x = hit_positions[:, 0]
        y = hit_positions[:, 1]
        z = hit_positions[:, 2]
        r = np.sqrt(x * x + y * y + z * z)
        r_safe = np.clip(r, self.inner_radius, self.outer_radius)

        t_emit = self.t_peak * (self.inner_radius / r_safe) ** 0.75

        f = 1.0 - self.rs / r_safe
        g_grav = np.sqrt(np.clip(f, 0.0, 1.0))

        if photon_momenta is None:
            t_obs = t_emit * g_grav
            rgb = blackbody_rgb(t_obs)
            beaming = g_grav**self.beaming_exponent
            rgb = rgb * beaming[:, np.newaxis]
        else:
            v_orbit = np.sqrt(self.mass / r_safe)
            sqrt_f = g_grav

            p_t = photon_momenta[:, 0]
            p_phi = photon_momenta[:, 3]

            # Only the φ component of the local orthonormal direction matters
            # for the Doppler dot product against β = v_orbit φ̂.
            p_t_hat = sqrt_f * p_t
            p_phi_hat = r_safe * p_phi
            safe_pt = np.where(np.abs(p_t_hat) < 1e-12, 1e-12, p_t_hat)
            # Physical photon goes from disk to camera: opposite to the
            # backward-traced direction, hence the leading minus.
            n_phi_phys = -p_phi_hat / safe_pt

            beta_dot_n = v_orbit * n_phi_phys
            gamma = 1.0 / np.sqrt(np.clip(1.0 - v_orbit * v_orbit, 1e-9, 1.0))
            g_dop = 1.0 / (gamma * (1.0 - beta_dot_n))

            g_total = g_grav * g_dop
            t_obs = t_emit * g_total
            beaming = g_total**self.beaming_exponent
            rgb = blackbody_rgb(t_obs) * beaming[:, np.newaxis]

        in_annulus = (r >= self.inner_radius) & (r <= self.outer_radius)
        rgb = np.where(in_annulus[:, np.newaxis], rgb, 0.0)
        return rgb
