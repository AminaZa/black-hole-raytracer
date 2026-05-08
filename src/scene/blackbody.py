"""Blackbody temperature → linear sRGB.

Analytic pipeline: integrate Planck's spectral radiance against the CIE 1931
2° standard observer colour-matching functions to obtain XYZ tristimulus
values, convert to linear sRGB (D65), then per-temperature normalise so the
brightest channel sits at 1. The whole computation runs in ``float64`` and is
``C∞``-smooth in ``T``, so a continuous radial temperature profile produces a
continuous colour gradient — no piecewise step discontinuities.

The CIE 1931 2° observer is sampled at 10 nm intervals from 380 to 780 nm
(41 wavelengths). At 10 nm spacing the rectangular sum is dense enough that
the three integrals are visually indistinguishable from finer quadratures.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


_T_MIN: float = 1000.0
_T_MAX: float = 40000.0

_PLANCK_H: float = 6.62607015e-34
_LIGHT_C: float = 299792458.0
_BOLTZMANN_K: float = 1.380649e-23

# CIE 1931 2° standard observer, 10 nm steps, 380 → 780 nm.
_CIE_LAMBDA_NM: NDArray[np.float64] = np.arange(380.0, 781.0, 10.0, dtype=np.float64)

_CIE_X: NDArray[np.float64] = np.array(
    [
        0.001368, 0.004243, 0.014310, 0.043510, 0.134380, 0.283900, 0.348280,
        0.336200, 0.290800, 0.195360, 0.095640, 0.032010, 0.004900, 0.009300,
        0.063270, 0.165500, 0.290400, 0.433450, 0.594500, 0.762100, 0.916300,
        1.026300, 1.062200, 1.002600, 0.854450, 0.642400, 0.447900, 0.283500,
        0.164900, 0.087400, 0.046770, 0.022700, 0.011359, 0.005790, 0.002899,
        0.001440, 0.000690, 0.000332, 0.000166, 0.000083, 0.000042,
    ],
    dtype=np.float64,
)

_CIE_Y: NDArray[np.float64] = np.array(
    [
        0.000039, 0.000120, 0.000396, 0.001210, 0.004000, 0.011600, 0.023000,
        0.038000, 0.060000, 0.090980, 0.139020, 0.208020, 0.323000, 0.503000,
        0.710000, 0.862000, 0.954000, 0.994950, 0.995000, 0.952000, 0.870000,
        0.757000, 0.631000, 0.503000, 0.381000, 0.265000, 0.175000, 0.107000,
        0.061000, 0.032000, 0.017000, 0.008210, 0.004102, 0.002091, 0.001047,
        0.000520, 0.000249, 0.000120, 0.000060, 0.000030, 0.000015,
    ],
    dtype=np.float64,
)

_CIE_Z: NDArray[np.float64] = np.array(
    [
        0.006450, 0.020050, 0.067850, 0.207400, 0.645600, 1.385600, 1.747060,
        1.772110, 1.669200, 1.287640, 0.812950, 0.465180, 0.272000, 0.158200,
        0.078250, 0.042160, 0.020300, 0.008750, 0.003900, 0.002100, 0.001650,
        0.001100, 0.000800, 0.000340, 0.000190, 0.000050, 0.000020, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    ],
    dtype=np.float64,
)

# Standard sRGB (IEC 61966-2-1) D65 XYZ → linear-RGB matrix.
_XYZ_TO_LINEAR_RGB: NDArray[np.float64] = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ],
    dtype=np.float64,
)


def blackbody_rgb(temperature: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map a temperature array to linear sRGB.

    Parameters
    ----------
    temperature:
        Shape (N,), Kelvin. Values outside ``[1000, 40000]`` are clipped.

    Returns
    -------
    rgb:
        Shape (N, 3), each channel in ``[0, 1]``, the brightest channel at 1
        for every entry. ``C∞`` in ``T`` so colour gradients stay smooth.
    """
    t = np.clip(np.asarray(temperature, dtype=np.float64), _T_MIN, _T_MAX)

    lam = _CIE_LAMBDA_NM * 1.0e-9  # nm → m
    # x = hc / (λ k_B T), shape (N, 41); use expm1 for numerical stability.
    x = (_PLANCK_H * _LIGHT_C) / (lam[None, :] * _BOLTZMANN_K * t[..., None])
    # Drop the constant 2hc² prefactor — it cancels in the per-row peak
    # normalisation below.
    spectral = 1.0 / (lam[None, :] ** 5 * np.expm1(x))

    xyz = np.stack(
        [
            (spectral * _CIE_X[None, :]).sum(axis=-1),
            (spectral * _CIE_Y[None, :]).sum(axis=-1),
            (spectral * _CIE_Z[None, :]).sum(axis=-1),
        ],
        axis=-1,
    )

    rgb_linear = xyz @ _XYZ_TO_LINEAR_RGB.T
    rgb_linear = np.maximum(rgb_linear, 0.0)  # clamp out-of-gamut negatives.

    peak = np.max(rgb_linear, axis=-1, keepdims=True)
    return rgb_linear / np.maximum(peak, 1e-30)
