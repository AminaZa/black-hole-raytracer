"""Blackbody temperature → sRGB lookup.

Uses the Tanner Helland approximation: a piecewise polynomial fit to the CIE
XYZ → sRGB conversion of a Planck spectrum. Inputs are in Kelvin; outputs are
linear RGB in [0, 1]. The fit is calibrated for 1000 K – 40 000 K and the
input is clipped to that range.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


_T_MIN: float = 1000.0
_T_MAX: float = 40000.0


def blackbody_rgb(temperature: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map a temperature array to linear RGB.

    Parameters
    ----------
    temperature:
        Shape (N,), Kelvin. Values outside ``[1000, 40000]`` are clipped.

    Returns
    -------
    rgb:
        Shape (N, 3), each channel in ``[0, 1]``.
    """
    t = np.asarray(temperature, dtype=np.float64)
    t = np.clip(t, _T_MIN, _T_MAX) / 100.0

    safe_t60 = np.maximum(t - 60.0, 1e-9)
    safe_t10 = np.maximum(t - 10.0, 1e-9)
    safe_t = np.maximum(t, 1e-9)

    r_high = np.clip(329.698727446 * safe_t60 ** -0.1332047592, 0.0, 255.0)
    red = np.where(t <= 66.0, 255.0, r_high)

    g_low = np.clip(99.4708025861 * np.log(safe_t) - 161.1195681661, 0.0, 255.0)
    g_high = np.clip(288.1221695283 * safe_t60 ** -0.0755148492, 0.0, 255.0)
    green = np.where(t <= 66.0, g_low, g_high)

    b_mid = np.clip(138.5177312231 * np.log(safe_t10) - 305.0447927307, 0.0, 255.0)
    blue = np.where(t < 19.0, 0.0, np.where(t >= 66.0, 255.0, b_mid))

    rgb = np.stack([red, green, blue], axis=-1) / 255.0
    return rgb
