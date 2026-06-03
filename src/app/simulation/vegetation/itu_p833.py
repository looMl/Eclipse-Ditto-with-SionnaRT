"""
ITU-R P.833 vegetation attenuation model — Modified Exponential Decay (MED) form.

Reference: ITU-R P.833-9 (09/2013), §3.2
Validity: 30 MHz - 60 GHz (implementation limited to sub-6 GHz).
"""

import numpy as np

# (a, b, A_max_dB) per leaf state.
# Specific attenuation: R [dB/m] = a · f_GHz^b
# Maximum attenuation: A_max [dB] (saturation ceiling for a deep stand)
# Coefficients from ITU-R P.833-9 guidance and measurement fits.
_COEFFS: dict = {
    "in_leaf": (0.20, 0.30, 40.0),
    "out_of_leaf": (0.16, 0.18, 25.0),
}

_FREQ_MAX_HZ = 7.125e9  # FR2 boundary; extrapolation above this is out of scope


def excess_loss_db(
    depth_eff_m: float | np.ndarray,
    freq_hz: float,
    leaf_state: str = "in_leaf",
) -> float | np.ndarray:
    """Returns ITU-R P.833 MED excess attenuation [dB] for the given depth and frequency.

    A = A_max · (1 - exp(-R · d / A_max)),  R = a · f_GHz^b  [dB/m]

    Raises ValueError for freq_hz > 7.125 GHz (FR2 out of scope) or unknown leaf_state.
    """
    if freq_hz > _FREQ_MAX_HZ:
        raise ValueError(
            f"freq_hz={freq_hz:.3e} exceeds {_FREQ_MAX_HZ:.3e} Hz (7.125 GHz). "
            "FR2/mmWave is out of scope for v1 — update coefficients before use."
        )
    if leaf_state not in _COEFFS:
        raise ValueError(
            f"leaf_state must be 'in_leaf' or 'out_of_leaf', got {leaf_state!r}."
        )

    a, b, a_max = _COEFFS[leaf_state]
    f_ghz = freq_hz / 1e9

    # Specific attenuation [dB/m]
    R = a * (f_ghz**b)

    depth = np.asarray(depth_eff_m, dtype=np.float64)
    return a_max * (1.0 - np.exp(-R * depth / a_max))
