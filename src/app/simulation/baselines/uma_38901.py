"""3GPP TR 38.901 Urban Macro (UMa) closed-form RSRP baseline.

Implements UMa LOS/NLOS path loss (Table 7.4.1-1), LOS probability
(Table 7.4.2-1), and the element antenna pattern (Table 7.3-1) so that
ray-traced RSRP values can be compared against a standard closed-form
model. Reference: 3GPP TR 38.901 V17.0.0 (2022-03).

Applicability bounds: 10 m <= d_2D <= 5000 m, 1.5 m <= h_UT <= 22.5 m,
h_BS = 25 m nominal (formulas remain monotonic outside these bounds but
are no longer the calibrated 3GPP model).
"""

import numpy as np

C_LIGHT = 3.0e8


def _d_breakpoint(
    h_bs_m: float, h_ut_m: float, f_hz: float, h_e_m: float = 1.0
) -> float:
    h_bs_eff = max(h_bs_m - h_e_m, 0.1)
    h_ut_eff = max(h_ut_m - h_e_m, 0.1)
    return 4.0 * h_bs_eff * h_ut_eff * f_hz / C_LIGHT


def pl_los_uma(
    d_2d: float, d_3d: float, h_bs: float, h_ut: float, f_ghz: float
) -> float:
    """UMa LOS path loss in dB (TR 38.901 Eq 7.4.1-1)."""
    f_hz = f_ghz * 1e9
    d_bp = _d_breakpoint(h_bs, h_ut, f_hz)
    d_3d = max(d_3d, 1.0)
    if d_2d < d_bp:
        return 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
    return (
        28.0
        + 40.0 * np.log10(d_3d)
        + 20.0 * np.log10(f_ghz)
        - 9.0 * np.log10(d_bp**2 + (h_bs - h_ut) ** 2)
    )


def pl_nlos_uma(
    d_2d: float, d_3d: float, h_bs: float, h_ut: float, f_ghz: float
) -> float:
    """UMa NLOS path loss in dB (TR 38.901 Eq 7.4.1-1, NLOS = max(LOS, NLOS'))."""
    pl_nlos_prime = (
        13.54
        + 39.08 * np.log10(max(d_3d, 1.0))
        + 20.0 * np.log10(f_ghz)
        - 0.6 * (h_ut - 1.5)
    )
    return max(pl_los_uma(d_2d, d_3d, h_bs, h_ut, f_ghz), pl_nlos_prime)


def los_probability_uma(d_2d: float, h_ut: float) -> float:
    """UMa LOS probability (TR 38.901 Eq 7.4.2-1)."""
    if d_2d <= 18.0:
        return 1.0
    c_prime = 0.0 if h_ut <= 13.0 else ((h_ut - 13.0) / 10.0) ** 1.5
    p1 = 18.0 / d_2d + (1.0 - 18.0 / d_2d) * np.exp(-d_2d / 63.0)
    return p1 * (1.0 + c_prime * 1.25 * (d_2d / 100.0) ** 3 * np.exp(-d_2d / 150.0))


def _tr38901_element_gain_db(
    theta_deg: float, phi_deg: float, max_gain_dbi: float = 8.0
) -> float:
    """Single-element gain pattern (TR 38.901 Table 7.3-1).

    theta_deg in [0, 180] with 90 = horizon, phi_deg in [-180, 180] with 0 = boresight.
    """
    a_v = -min(12.0 * ((theta_deg - 90.0) / 65.0) ** 2, 30.0)
    a_h = -min(12.0 * (phi_deg / 65.0) ** 2, 30.0)
    a = -min(-(a_v + a_h), 30.0)
    return max_gain_dbi + a


def _world_to_local_direction(
    d_world: np.ndarray, yaw_rad: float, pitch_rad: float, roll_rad: float
) -> np.ndarray:
    """Inverse-rotate a unit world-frame direction into the TX local frame.

    Sionna composes orientation as R = R_z(yaw) R_y(pitch) R_x(roll). To
    move a world direction into the local frame we apply R^T.
    """
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    r = rz @ ry @ rx
    return r.T @ d_world


def antenna_gain_dbi(
    d_world: np.ndarray,
    orientation_rad: tuple[float, float, float],
    num_rows: int = 4,
    num_cols: int = 1,
    max_element_gain: float = 8.0,
) -> float:
    """Effective TX antenna gain in dBi toward `d_world` for a tr38901 panel.

    Combines element pattern with broadside array gain 10*log10(N_rows * N_cols).
    """
    n = np.linalg.norm(d_world)
    if n < 1e-9:
        return max_element_gain + 10.0 * np.log10(num_rows * num_cols)
    yaw, pitch, roll = orientation_rad
    d_local = _world_to_local_direction(d_world / n, yaw, pitch, roll)
    # Local: +x boresight, +z zenith. Sionna spherical: theta from +z, phi in xy from +x.
    theta_deg = float(np.degrees(np.arccos(np.clip(d_local[2], -1.0, 1.0))))
    phi_deg = float(np.degrees(np.arctan2(d_local[1], d_local[0])))
    elem_gain = _tr38901_element_gain_db(theta_deg, phi_deg, max_element_gain)
    array_gain = 10.0 * np.log10(num_rows * num_cols)
    return elem_gain + array_gain


def predict_rsrp_uma(
    tx_position: np.ndarray,
    tx_orientation_rad: tuple[float, float, float],
    tx_power_dbm: float,
    rx_position: np.ndarray,
    f_ghz: float = 1.8,
    num_subcarriers: int = 900,
    num_rows: int = 4,
    num_cols: int = 1,
) -> dict[str, float]:
    """Closed-form RSRP estimate using 3GPP TR 38.901 UMa.

    Returns the LOS-probability-weighted expected path loss (averaged in linear
    power, not dB, since that is the physically meaningful average over the
    LOS/NLOS shadowing distribution).
    """
    diff = np.asarray(rx_position, dtype=float) - np.asarray(tx_position, dtype=float)
    d_3d = float(np.linalg.norm(diff))
    d_2d = float(np.linalg.norm(diff[:2]))
    h_bs = float(tx_position[2])
    h_ut = float(rx_position[2])

    pl_los = pl_los_uma(d_2d, d_3d, h_bs, h_ut, f_ghz)
    pl_nlos = pl_nlos_uma(d_2d, d_3d, h_bs, h_ut, f_ghz)
    p_los = los_probability_uma(d_2d, h_ut)

    g_los = 10 ** (-pl_los / 10.0)
    g_nlos = 10 ** (-pl_nlos / 10.0)
    pl_eff = -10.0 * np.log10(p_los * g_los + (1.0 - p_los) * g_nlos)

    tx_gain = antenna_gain_dbi(
        diff, tx_orientation_rad, num_rows=num_rows, num_cols=num_cols
    )
    subcarrier_offset = 10.0 * np.log10(num_subcarriers)
    rsrp = tx_power_dbm + tx_gain - pl_eff - subcarrier_offset

    return {
        "rsrp_dbm": float(rsrp),
        "pl_los_db": float(pl_los),
        "pl_nlos_db": float(pl_nlos),
        "p_los": float(p_los),
        "pl_effective_db": float(pl_eff),
        "tx_gain_dbi": float(tx_gain),
        "d_2d_m": d_2d,
        "d_3d_m": d_3d,
    }
