"""Kerr metric in Boyer-Lindquist coordinates (t, r, theta, phi).

Geometrized units (G = c = 1) throughout. The line element is

    ds² = -(1 - 2Mr/Σ) dt² - (4Mar sin²θ/Σ) dt dφ
        + Σ/Δ dr² + Σ dθ² + ((r² + a²) + 2Ma²r sin²θ/Σ) sin²θ dφ²

with Σ = r² + a² cos²θ and Δ = r² - 2Mr + a². At a = 0 the metric reduces
exactly to Schwarzschild. The outer event horizon sits at r₊ = M + √(M² − a²).

The Christoffel symbols are derived from the analytical g_{μν}, g^{μν} and
∂_λ g_{μν} via the Levi-Civita formula

    Γ^μ_{αβ} = (1/2) g^{μν} (∂_α g_{νβ} + ∂_β g_{να} - ∂_ν g_{αβ})

contracted with einsum. No numerical differentiation is involved.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from utils.cuda_loader import cupy as _cupy, get_xp


if _cupy is not None:

    @_cupy.fuse(kernel_name="kerr_accel_fused")
    def _kerr_accel_fused(  # type: ignore[no-untyped-def]
        r, theta, pt, pr, pth, pph, M, a
    ):
        """Element-wise Kerr acceleration, JIT-fused into one CUDA kernel.

        Mirrors :meth:`KerrMetric.acceleration` but operates on (N,) inputs.
        Returns the four acceleration components as a tuple of (N,) arrays.
        """
        a2 = a * a
        sin_t = _cupy.sin(theta)
        cos_t = _cupy.cos(theta)
        sin2 = sin_t * sin_t
        cos2 = cos_t * cos_t
        sin_cos = sin_t * cos_t

        Sigma = r * r + a2 * cos2
        Delta = r * r - 2.0 * M * r + a2
        Sigma_sq = Sigma * Sigma
        dS_dr = 2.0 * r
        dS_dt = -2.0 * a2 * sin_cos
        dD_dr = 2.0 * (r - M)

        g_tt = -(1.0 - 2.0 * M * r / Sigma)
        g_tphi = -2.0 * M * a * r * sin2 / Sigma
        g_pp = (r * r + a2 + 2.0 * M * a2 * r * sin2 / Sigma) * sin2

        dg_dr_tt = 2.0 * M * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr_tp = -2.0 * M * a * sin2 * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr_rr = (dS_dr * Delta - Sigma * dD_dr) / (Delta * Delta)
        dg_dr_th = dS_dr
        dA2_dr = 2.0 * M * a2 * sin2 * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr_pp = (2.0 * r + dA2_dr) * sin2

        dg_dt_tt = -2.0 * M * r * dS_dt / Sigma_sq
        dg_dt_tp = (
            -2.0 * M * a * r * (2.0 * sin_cos * Sigma - sin2 * dS_dt) / Sigma_sq
        )
        dg_dt_rr = dS_dt / Delta
        dg_dt_th = dS_dt
        dA2_dt = (
            2.0 * M * a2 * r * (2.0 * sin_cos * Sigma - sin2 * dS_dt) / Sigma_sq
        )
        bracket = r * r + a2 + 2.0 * M * a2 * r * sin2 / Sigma
        dg_dt_pp = dA2_dt * sin2 + bracket * 2.0 * sin_cos

        det_tp = -Delta * sin2
        ginv_tt = g_pp / det_tp
        ginv_tp = -g_tphi / det_tp
        ginv_pp = g_tt / det_tp
        ginv_rr = Delta / Sigma
        ginv_th = 1.0 / Sigma

        a0 = dg_dr_tt * pt + dg_dr_tp * pph
        a1 = dg_dr_rr * pr
        a2_ = dg_dr_th * pth
        a3 = dg_dr_tp * pt + dg_dr_pp * pph

        b0 = dg_dt_tt * pt + dg_dt_tp * pph
        b1 = dg_dt_rr * pr
        b2 = dg_dt_th * pth
        b3 = dg_dt_tp * pt + dg_dt_pp * pph

        q1 = pt * a0 + pr * a1 + pth * a2_ + pph * a3
        q2 = pt * b0 + pr * b1 + pth * b2 + pph * b3

        r0 = pr * a0 + pth * b0
        r1 = pr * a1 + pth * b1
        r2 = pr * a2_ + pth * b2
        r3 = pr * a3 + pth * b3

        h0 = -r0
        h1 = 0.5 * q1 - r1
        h2 = 0.5 * q2 - r2
        h3 = -r3

        acc0 = ginv_tt * h0 + ginv_tp * h3
        acc1 = ginv_rr * h1
        acc2 = ginv_th * h2
        acc3 = ginv_tp * h0 + ginv_pp * h3
        return acc0, acc1, acc2, acc3

else:
    _kerr_accel_fused = None  # type: ignore[assignment]


class KerrMetric:
    """Stationary axisymmetric Kerr black hole in Boyer-Lindquist coordinates.

    Parameters
    ----------
    mass:
        Black hole mass M in geometrized units. Must be positive.
    spin:
        Spin parameter a. Must satisfy ``|a| < mass`` (subextremal). a = 0
        is Schwarzschild; ``|a| → mass`` is the extremal limit.
    """

    def __init__(self, mass: float = 1.0, spin: float = 0.0) -> None:
        if mass <= 0.0:
            raise ValueError(f"mass must be positive, got {mass}.")
        if abs(spin) >= mass:
            raise ValueError(
                f"|spin| must be strictly less than mass; got spin={spin}, mass={mass}."
            )

        self.mass: float = float(mass)
        self.spin: float = float(spin)
        disc: float = float(np.sqrt(mass * mass - spin * spin))
        self.r_plus: float = self.mass + disc
        self.r_minus: float = self.mass - disc
        # Outer horizon doubles as the integrator's ``rs`` (termination radius
        # and step-band reference scale). At a = 0 this gives 2M, matching
        # the Schwarzschild convention.
        self.rs: float = self.r_plus

    def _aux(
        self, position: NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        r: float = float(position[1])
        theta: float = float(position[2])
        a: float = self.spin
        M: float = self.mass
        sin_t: float = float(np.sin(theta))
        cos_t: float = float(np.cos(theta))
        sin2: float = sin_t * sin_t
        cos2: float = cos_t * cos_t
        Sigma: float = r * r + a * a * cos2
        Delta: float = r * r - 2.0 * M * r + a * a
        return r, theta, a, M, sin_t, cos_t, sin2, cos2, Sigma, Delta

    def metric_tensor(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Covariant metric components g_{μν} at the given event."""
        r, _, a, M, _, _, sin2, _, Sigma, _ = self._aux(position)

        g: NDArray[np.float64] = np.zeros((4, 4), dtype=np.float64)
        g[0, 0] = -(1.0 - 2.0 * M * r / Sigma)
        g_tphi: float = -2.0 * M * a * r * sin2 / Sigma
        g[0, 3] = g_tphi
        g[3, 0] = g_tphi
        g[1, 1] = Sigma / (r * r - 2.0 * M * r + a * a)
        g[2, 2] = Sigma
        g[3, 3] = (r * r + a * a + 2.0 * M * a * a * r * sin2 / Sigma) * sin2
        return g

    def inverse_metric(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Contravariant metric g^{μν}, exploiting det of the (t, φ) block."""
        _, _, _, _, _, _, sin2, _, Sigma, Delta = self._aux(position)
        g: NDArray[np.float64] = self.metric_tensor(position)

        g_inv: NDArray[np.float64] = np.zeros((4, 4), dtype=np.float64)
        # The (t, φ) block has determinant -Δ sin²θ exactly.
        det_tp: float = -Delta * sin2
        g_inv[0, 0] = g[3, 3] / det_tp
        cross: float = -g[0, 3] / det_tp
        g_inv[0, 3] = cross
        g_inv[3, 0] = cross
        g_inv[3, 3] = g[0, 0] / det_tp
        g_inv[1, 1] = Delta / Sigma
        g_inv[2, 2] = 1.0 / Sigma
        return g_inv

    def metric_derivatives(
        self, position: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (∂_r g, ∂_θ g) as two (4, 4) arrays (∂_t, ∂_φ vanish)."""
        r, _, a, M, sin_t, cos_t, sin2, _, Sigma, Delta = self._aux(position)

        sin_cos: float = sin_t * cos_t
        dS_dr: float = 2.0 * r
        dS_dt: float = -2.0 * a * a * sin_cos
        dD_dr: float = 2.0 * (r - M)
        Sigma_sq: float = Sigma * Sigma
        a2: float = a * a

        dg_dr: NDArray[np.float64] = np.zeros((4, 4), dtype=np.float64)
        dg_dt: NDArray[np.float64] = np.zeros((4, 4), dtype=np.float64)

        # g_tt = -1 + 2Mr/Σ
        dg_dr[0, 0] = 2.0 * M * (Sigma - r * dS_dr) / Sigma_sq
        dg_dt[0, 0] = -2.0 * M * r * dS_dt / Sigma_sq

        # g_tφ = -2Mar sin²θ / Σ
        dg_tphi_dr: float = -2.0 * M * a * sin2 * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr[0, 3] = dg_tphi_dr
        dg_dr[3, 0] = dg_tphi_dr
        dg_tphi_dt: float = (
            -2.0 * M * a * r * (2.0 * sin_cos * Sigma - sin2 * dS_dt) / Sigma_sq
        )
        dg_dt[0, 3] = dg_tphi_dt
        dg_dt[3, 0] = dg_tphi_dt

        # g_rr = Σ / Δ
        dg_dr[1, 1] = (dS_dr * Delta - Sigma * dD_dr) / (Delta * Delta)
        dg_dt[1, 1] = dS_dt / Delta

        # g_θθ = Σ
        dg_dr[2, 2] = dS_dr
        dg_dt[2, 2] = dS_dt

        # g_φφ = (r² + a² + 2 M a² r sin²θ / Σ) sin²θ
        dA2_dr: float = 2.0 * M * a2 * sin2 * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr[3, 3] = (2.0 * r + dA2_dr) * sin2

        dA2_dt: float = (
            2.0 * M * a2 * r * (2.0 * sin_cos * Sigma - sin2 * dS_dt) / Sigma_sq
        )
        bracket: float = r * r + a2 + 2.0 * M * a2 * r * sin2 / Sigma
        dg_dt[3, 3] = dA2_dt * sin2 + bracket * 2.0 * sin_cos

        return dg_dr, dg_dt

    def christoffel_symbols(
        self, position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Connection coefficients Γ^μ_{αβ} at ``position``, shape (4, 4, 4).

        Computed from the analytical metric, inverse metric and partial
        derivatives. Symmetric in the lower index pair by construction.
        """
        g_inv: NDArray[np.float64] = self.inverse_metric(position)
        dg_dr, dg_dt = self.metric_derivatives(position)
        dg: NDArray[np.float64] = np.zeros((4, 4, 4), dtype=np.float64)
        dg[1] = dg_dr
        dg[2] = dg_dt

        # Γ^μ_{αβ} = (1/2) g^{μν} (∂_α g_{νβ} + ∂_β g_{να} - ∂_ν g_{αβ})
        term_a: NDArray[np.float64] = np.einsum("mn,anb->mab", g_inv, dg)
        term_b: NDArray[np.float64] = np.einsum("mn,bna->mab", g_inv, dg)
        term_c: NDArray[np.float64] = np.einsum("mn,nab->mab", g_inv, dg)
        return 0.5 * (term_a + term_b - term_c)

    def acceleration(
        self,
        position: NDArray[np.float64],
        momentum: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Geodesic right-hand side ``dp^μ/dλ = -Γ^μ_{αβ} p^α p^β``.

        Inlined hot path: one ``_aux`` call, no full Christoffel tensor, and
        no intermediate 4×4 ``np.zeros`` allocations beyond what ``np.array``
        already does for the dg matrices. Using ``p^α p^β`` symmetry and
        ``∂_t g = ∂_φ g = 0``,

            -Γ^μ_{αβ} p^α p^β = g^{μν} [(1/2) (∂_ν g_{αβ}) p^α p^β
                                       - (∂_α g_{νβ}) p^α p^β]

        Verified against the Christoffel route in
        ``test_acceleration_matches_christoffel_einsum``.
        """
        r, _, a, M, sin_t, cos_t, sin2, _, Sigma, Delta = self._aux(position)
        a2: float = a * a
        Sigma_sq: float = Sigma * Sigma
        sin_cos: float = sin_t * cos_t
        dS_dr: float = 2.0 * r
        dS_dt: float = -2.0 * a2 * sin_cos
        dD_dr: float = 2.0 * (r - M)

        # Covariant metric components.
        g_tt: float = -(1.0 - 2.0 * M * r / Sigma)
        g_tphi: float = -2.0 * M * a * r * sin2 / Sigma
        g_pp: float = (r * r + a2 + 2.0 * M * a2 * r * sin2 / Sigma) * sin2

        # ∂_r g and ∂_θ g (only the symmetric upper-triangle entries needed).
        dg_dr_tt: float = 2.0 * M * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr_tp: float = -2.0 * M * a * sin2 * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr_rr: float = (dS_dr * Delta - Sigma * dD_dr) / (Delta * Delta)
        dg_dr_th: float = dS_dr
        dA2_dr: float = 2.0 * M * a2 * sin2 * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr_pp: float = (2.0 * r + dA2_dr) * sin2

        dg_dt_tt: float = -2.0 * M * r * dS_dt / Sigma_sq
        dg_dt_tp: float = (
            -2.0 * M * a * r * (2.0 * sin_cos * Sigma - sin2 * dS_dt) / Sigma_sq
        )
        dg_dt_rr: float = dS_dt / Delta
        dg_dt_th: float = dS_dt
        dA2_dt: float = (
            2.0 * M * a2 * r * (2.0 * sin_cos * Sigma - sin2 * dS_dt) / Sigma_sq
        )
        bracket: float = r * r + a2 + 2.0 * M * a2 * r * sin2 / Sigma
        dg_dt_pp: float = dA2_dt * sin2 + bracket * 2.0 * sin_cos

        # Inverse metric. det of (t, φ) block is -Δ sin²θ exactly.
        det_tp: float = -Delta * sin2
        ginv_tt: float = g_pp / det_tp
        ginv_tp: float = -g_tphi / det_tp
        ginv_pp: float = g_tt / det_tp
        ginv_rr: float = Delta / Sigma
        ginv_th: float = 1.0 / Sigma

        pt: float = float(momentum[0])
        pr: float = float(momentum[1])
        pth: float = float(momentum[2])
        pph: float = float(momentum[3])

        # a_vec[n] = (∂_r g)_{nb} p^b ; b_vec[n] = (∂_θ g)_{nb} p^b.
        a0: float = dg_dr_tt * pt + dg_dr_tp * pph
        a1: float = dg_dr_rr * pr
        a2_: float = dg_dr_th * pth
        a3: float = dg_dr_tp * pt + dg_dr_pp * pph

        b0: float = dg_dt_tt * pt + dg_dt_tp * pph
        b1: float = dg_dt_rr * pr
        b2: float = dg_dt_th * pth
        b3: float = dg_dt_tp * pt + dg_dt_pp * pph

        # q[ν] = (∂_ν g_{αβ}) p^α p^β, nonzero only for ν = r, θ.
        q1: float = pt * a0 + pr * a1 + pth * a2_ + pph * a3
        q2: float = pt * b0 + pr * b1 + pth * b2 + pph * b3

        # r[ν] = (∂_α g_{νβ}) p^α p^β = p^r a_vec[ν] + p^θ b_vec[ν].
        r0: float = pr * a0 + pth * b0
        r1: float = pr * a1 + pth * b1
        r2: float = pr * a2_ + pth * b2
        r3: float = pr * a3 + pth * b3

        # acc = g^{μν} (0.5 q[ν] - r[ν]).
        h0: float = -r0
        h1: float = 0.5 * q1 - r1
        h2: float = 0.5 * q2 - r2
        h3: float = -r3

        acc0: float = ginv_tt * h0 + ginv_tp * h3
        acc1: float = ginv_rr * h1
        acc2: float = ginv_th * h2
        acc3: float = ginv_tp * h0 + ginv_pp * h3

        return np.array([acc0, acc1, acc2, acc3], dtype=np.float64)

    def acceleration_batch(
        self,
        positions: NDArray[np.float64],
        momenta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Vectorised acceleration for an array of rays, shape (N, 4).

        On cupy inputs the inner math goes through a single fused CUDA
        kernel (one launch instead of ~80 small ones). On numpy the same
        arithmetic runs in NumPy directly. Output array module always
        matches the input.
        """
        xp = get_xp(positions)
        r = positions[:, 1]
        theta = positions[:, 2]
        pt = momenta[:, 0]
        pr = momenta[:, 1]
        pth = momenta[:, 2]
        pph = momenta[:, 3]

        if _cupy is not None and xp is _cupy:
            acc0, acc1, acc2, acc3 = _kerr_accel_fused(
                r, theta, pt, pr, pth, pph, self.mass, self.spin
            )
            return xp.stack([acc0, acc1, acc2, acc3], axis=1)

        a: float = self.spin
        M: float = self.mass
        a2: float = a * a

        sin_t = xp.sin(theta)
        cos_t = xp.cos(theta)
        sin2 = sin_t * sin_t
        cos2 = cos_t * cos_t
        sin_cos = sin_t * cos_t
        Sigma = r * r + a2 * cos2
        Delta = r * r - 2.0 * M * r + a2
        Sigma_sq = Sigma * Sigma
        dS_dr = 2.0 * r
        dS_dt = -2.0 * a2 * sin_cos
        dD_dr = 2.0 * (r - M)

        # Covariant metric components.
        g_tt = -(1.0 - 2.0 * M * r / Sigma)
        g_tphi = -2.0 * M * a * r * sin2 / Sigma
        g_pp = (r * r + a2 + 2.0 * M * a2 * r * sin2 / Sigma) * sin2

        # ∂_r g and ∂_θ g (only the symmetric upper-triangle entries needed).
        dg_dr_tt = 2.0 * M * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr_tp = -2.0 * M * a * sin2 * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr_rr = (dS_dr * Delta - Sigma * dD_dr) / (Delta * Delta)
        dg_dr_th = dS_dr
        dA2_dr = 2.0 * M * a2 * sin2 * (Sigma - r * dS_dr) / Sigma_sq
        dg_dr_pp = (2.0 * r + dA2_dr) * sin2

        dg_dt_tt = -2.0 * M * r * dS_dt / Sigma_sq
        dg_dt_tp = (
            -2.0 * M * a * r * (2.0 * sin_cos * Sigma - sin2 * dS_dt) / Sigma_sq
        )
        dg_dt_rr = dS_dt / Delta
        dg_dt_th = dS_dt
        dA2_dt = (
            2.0 * M * a2 * r * (2.0 * sin_cos * Sigma - sin2 * dS_dt) / Sigma_sq
        )
        bracket = r * r + a2 + 2.0 * M * a2 * r * sin2 / Sigma
        dg_dt_pp = dA2_dt * sin2 + bracket * 2.0 * sin_cos

        # Inverse metric. det of (t, φ) block is -Δ sin²θ exactly.
        det_tp = -Delta * sin2
        ginv_tt = g_pp / det_tp
        ginv_tp = -g_tphi / det_tp
        ginv_pp = g_tt / det_tp
        ginv_rr = Delta / Sigma
        ginv_th = 1.0 / Sigma

        # a_vec[n] = (∂_r g)_{nb} p^b ; b_vec[n] = (∂_θ g)_{nb} p^b.
        a0 = dg_dr_tt * pt + dg_dr_tp * pph
        a1 = dg_dr_rr * pr
        a2_ = dg_dr_th * pth
        a3 = dg_dr_tp * pt + dg_dr_pp * pph

        b0 = dg_dt_tt * pt + dg_dt_tp * pph
        b1 = dg_dt_rr * pr
        b2 = dg_dt_th * pth
        b3 = dg_dt_tp * pt + dg_dt_pp * pph

        # q[ν] = (∂_ν g_{αβ}) p^α p^β, nonzero only for ν = r, θ.
        q1 = pt * a0 + pr * a1 + pth * a2_ + pph * a3
        q2 = pt * b0 + pr * b1 + pth * b2 + pph * b3

        # r[ν] = (∂_α g_{νβ}) p^α p^β = p^r a_vec[ν] + p^θ b_vec[ν].
        r0 = pr * a0 + pth * b0
        r1 = pr * a1 + pth * b1
        r2 = pr * a2_ + pth * b2
        r3 = pr * a3 + pth * b3

        # acc = g^{μν} (0.5 q[ν] - r[ν]).
        h0 = -r0
        h1 = 0.5 * q1 - r1
        h2 = 0.5 * q2 - r2
        h3 = -r3

        acc0 = ginv_tt * h0 + ginv_tp * h3
        acc1 = ginv_rr * h1
        acc2 = ginv_th * h2
        acc3 = ginv_tp * h0 + ginv_pp * h3

        return xp.stack([acc0, acc1, acc2, acc3], axis=1)
