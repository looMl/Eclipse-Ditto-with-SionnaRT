import sys
from typing import Any
import numpy as np
import sionna.rt as rt
import argparse
from loguru import logger
from app.simulation.engine import SimulationEngine
from app.simulation.scene_manager import SceneManager
from app.config import get_settings, get_project_root
from app.geomap_processor.utils.vegetation_field import VegetationField
from app.simulation.vegetation.vegetation_path_integrator import PathDepthIntegrator
from app.simulation.vegetation.itu_p833 import excess_loss_db
from app.simulation.baselines.uma_38901 import predict_rsrp_uma


def _collapse_dense_paths(
    paths,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Collapse full Sionna path tensors to (num_tgt, num_src, num_paths) form.

    When paths.synthetic_array is False, Sionna emits tensors with separate
    pattern and array-size axes that must be squeezed out before use.
    Returns (vertices, valid, types, src, tgt, max_depth).
    """
    vertices = np.array(paths.vertices)
    valid = np.array(paths.valid)
    types = np.array(paths.interactions)
    src = np.array(paths.sources).T
    tgt = np.array(paths.targets).T
    max_depth = vertices.shape[0]

    if not paths.synthetic_array:
        num_rx = paths.num_rx
        rx_array_size = paths.rx_array.array_size
        num_rx_patterns = len(paths.rx_array.antenna_pattern.patterns)
        num_tx = paths.num_tx
        tx_array_size = paths.tx_array.array_size
        num_tx_patterns = len(paths.tx_array.antenna_pattern.patterns)
        num_tgt = tgt.shape[0]
        num_src = src.shape[0]
        vertices = vertices.reshape(
            max_depth,
            num_rx,
            num_rx_patterns,
            rx_array_size,
            num_tx,
            num_tx_patterns,
            tx_array_size,
            -1,
            3,
        )[:, :, 0, :, :, 0, :, :, :].reshape(max_depth, num_tgt, num_src, -1, 3)
        valid = valid.reshape(
            num_rx,
            num_rx_patterns,
            rx_array_size,
            num_tx,
            num_tx_patterns,
            tx_array_size,
            -1,
        )[:, 0, :, :, 0, :, :].reshape(num_tgt, num_src, -1)
        types = types.reshape(
            max_depth,
            num_rx,
            num_rx_patterns,
            rx_array_size,
            num_tx,
            num_tx_patterns,
            tx_array_size,
            -1,
        )[:, :, 0, :, :, 0, :, :].reshape(max_depth, num_tgt, num_src, -1)

    return vertices, valid, types, src, tgt, max_depth


def _veg_per_path_attenuation_linear(
    paths,
    integrator: PathDepthIntegrator,
    freq_hz: float,
    leaf_state: str,
) -> np.ndarray:
    """Per-(tx, path) linear vegetation attenuation factor in [0, 1].

    Walks the path graph emitted by Sionna's PathSolver, builds a flat list of
    every ray segment with bookkeeping back to (tx, path), runs the per-segment
    integrator once, and reduces depths back per path. Geometry mirrors
    ``sionna.rt.paths_to_segments`` but preserves the path-id mapping needed
    to apply attenuation to per-path power before aggregation.
    """
    vertices, valid, types, src, tgt, max_depth = _collapse_dense_paths(paths)

    num_tgt, num_src, num_paths = valid.shape
    none_type = int(rt.InteractionType.NONE)

    if num_paths == 0:
        return np.ones((num_src, num_paths), dtype="float64")

    p0_list, p1_list, tx_list, path_list = [], [], [], []
    for rx_i in range(num_tgt):
        for tx_i in range(num_src):
            for p in range(num_paths):
                if not valid[rx_i, tx_i, p]:
                    continue
                start = src[tx_i]
                for i in range(max_depth):
                    if int(types[i, rx_i, tx_i, p]) == none_type:
                        break
                    end = vertices[i, rx_i, tx_i, p]
                    p0_list.append(start)
                    p1_list.append(end)
                    tx_list.append(tx_i)
                    path_list.append(p)
                    start = end
                p0_list.append(start)
                p1_list.append(tgt[rx_i])
                tx_list.append(tx_i)
                path_list.append(p)

    atten = np.ones((num_src, num_paths), dtype="float64")
    if not p0_list:
        return atten

    p0 = np.asarray(p0_list, dtype="float32")
    p1 = np.asarray(p1_list, dtype="float32")
    tx_idx = np.asarray(tx_list, dtype="int64")
    path_idx = np.asarray(path_list, dtype="int64")

    seg_depths = integrator.integrate_segments(p0, p1)

    flat_idx = tx_idx * num_paths + path_idx
    total_depths = np.bincount(
        flat_idx, weights=seg_depths, minlength=num_src * num_paths
    ).reshape(num_src, num_paths)

    atten_db = excess_loss_db(total_depths, freq_hz, leaf_state)
    return 10.0 ** (-atten_db / 10.0)


_Z90 = 1.2816  # Φ⁻¹(0.90) — used for the 80 % log-normal shadowing interval


def _los_flags(paths) -> np.ndarray:
    """Per-TX boolean: True if any valid path to the RX has no interactions (LOS)."""
    none_type = int(rt.InteractionType.NONE)
    _, valid, types, _, _, _ = _collapse_dense_paths(paths)
    # LOS: valid path whose first-depth interaction type is NONE (no bounces)
    is_los = valid & (types[0] == none_type)  # (num_tgt, num_src, num_paths)
    return np.any(is_los, axis=(0, 2))  # (num_src,)


def measure_rsrp(
    x: float,
    y: float,
    z: float = 1.5,
    skip_vegetation: bool = False,
) -> list[dict[str, Any]]:
    """
    Measures the Reference Signal Received Power (RSRP) in dBm at a given scene position.
    """
    SimulationEngine.initialize()

    manager = SceneManager()
    scene = manager.load_scene()

    pos = [x, y, z]

    logger.info(f"Measuring RSRP at x={x}, y={y}, z={z}")

    rx = rt.Receiver(name="rx", position=pos)
    scene.add(rx)

    # Receiver as 2x2 MIMO Smartphone
    scene.rx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",  # 3GPP standard for sub-6 GHz User Equipment
        polarization="cross",
    )

    logger.info("Computing propagation paths...")
    solver = rt.PathSolver()
    ds_cfg = getattr(get_settings().sionnart, "diffuse_scattering", None)
    diffuse = ds_cfg is not None and getattr(ds_cfg, "enabled", False)
    paths = solver(
        scene,
        max_depth=get_settings().sionnart.coverage.max_depth,
        samples_per_src=get_settings().sionnart.coverage.samples_per_tx,
        max_num_paths_per_src=get_settings().sionnart.coverage.max_num_paths_per_src,
        diffuse_reflection=diffuse,
    )
    los_flags = _los_flags(paths)

    # Vegetation correction setup (no-op when disabled or field absent)
    veg_integrator = None
    veg_freq_hz = 1.8e9
    veg_leaf_state = "in_leaf"
    veg_mode = "per_link"
    veg_cfg = getattr(get_settings().sionnart, "vegetation", None)
    if (
        not skip_vegetation
        and veg_cfg is not None
        and getattr(veg_cfg, "enabled", False)
    ):
        npz_path = get_project_root() / "scene" / "mesh" / "vegetation_field.npz"
        if npz_path.exists():
            _field = VegetationField.load(npz_path)
            veg_integrator = PathDepthIntegrator(
                _field, step_m=getattr(veg_cfg, "raster_step_m", 1.0)
            )
            veg_freq_hz = getattr(veg_cfg, "frequency_hz", 1.8e9)
            veg_leaf_state = getattr(veg_cfg, "leaf_state", "in_leaf")
            veg_mode = getattr(veg_cfg, "mode", "per_link")
        else:
            logger.warning(
                "Vegetation field not found; skipping vegetation correction."
            )

    # Per-path channel power (keep path axis until veg attenuation has been applied)
    a_data = paths.a
    if isinstance(a_data, tuple):
        real = np.array(a_data[0])
        imag = np.array(a_data[1])
        power_per_path = real**2 + imag**2
    else:
        a = np.array(a_data)
        power_per_path = np.abs(a) ** 2

    veg_per_path_active = veg_integrator is not None and veg_mode == "per_path"
    if veg_per_path_active:
        logger.info("Applying per-path vegetation attenuation...")
        atten_lin = _veg_per_path_attenuation_linear(
            paths, veg_integrator, veg_freq_hz, veg_leaf_state
        )
        # power_per_path shape: (num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths)
        # atten_lin shape:      (num_tx, num_paths) → broadcast over rx/rx_ant/tx_ant.
        power_per_path = power_per_path * atten_lin[None, None, :, None, :]

    power_linear = np.sum(power_per_path, axis=-1)

    transmitters = list(scene.transmitters.values())
    results = []

    # Shadowing interval parameters
    shadow_cfg = getattr(get_settings().sionnart, "shadowing", None)
    sigma_db = (
        float(getattr(shadow_cfg, "sigma_db", 6.0))
        if shadow_cfg and getattr(shadow_cfg, "enabled", True)
        else 0.0
    )

    handset_cfg = getattr(get_settings().sionnart, "handset", None)
    body_loss_db = (
        float(getattr(handset_cfg, "body_loss_db", 0.0)) if handset_cfg else 0.0
    )

    # 4G LTE Subcarrier Configuration
    # 10 MHz = 50 PRBs * 12 = 600 subcarriers
    # 15 MHz = 75 PRBs * 12 = 900 subcarriers
    NUM_SUBCARRIERS = 900
    SUBCARRIER_POWER_OFFSET_DB = 10 * np.log10(NUM_SUBCARRIERS)
    CARRIER_FREQ_GHZ = veg_freq_hz / 1e9

    for i, tx in enumerate(transmitters):
        # Shape: (num_rx, num_rx_ant, num_tx, num_tx_ant)
        antenna_pairs_power = power_linear[0, :, i, :]
        # Sum power across all TX antennas for each RX antenna
        power_per_rx_antenna = np.sum(antenna_pairs_power, axis=1)
        # Averaging the received power across the active branches
        total_channel_gain_linear = np.mean(power_per_rx_antenna)

        # Convert channel gain to dB
        if total_channel_gain_linear > 0:
            gain_db = 10 * np.log10(total_channel_gain_linear)
        else:
            gain_db = -150.0  # Floor for no signal

        pl_db = -gain_db

        # Total Transmit Power
        tx_power_total_dbm = float(np.array(tx.power_dbm).flatten()[0])

        # RSRP = Total TX Power + Channel Gain - Subcarrier Offset - Body Loss
        rsrp_dbm = (
            tx_power_total_dbm
            + float(gain_db)
            - SUBCARRIER_POWER_OFFSET_DB
            - body_loss_db
        )

        tx_pos = np.array(tx.position, dtype="float32").flatten()[:3]
        if veg_integrator is not None and not veg_per_path_active:
            rx_pos = np.array([[x, y, z]], dtype="float32")
            depth = veg_integrator.integrate(tx_pos, rx_pos)[0]
            rsrp_dbm -= float(excess_loss_db(depth, veg_freq_hz, veg_leaf_state))

        tx_orient = np.array(tx.orientation, dtype="float64").flatten()[:3]
        baseline = predict_rsrp_uma(
            tx_position=tx_pos.astype("float64"),
            tx_orientation_rad=(
                float(tx_orient[0]),
                float(tx_orient[1]),
                float(tx_orient[2]),
            ),
            tx_power_dbm=tx_power_total_dbm,
            rx_position=np.array([x, y, z], dtype="float64"),
            f_ghz=CARRIER_FREQ_GHZ,
            num_subcarriers=NUM_SUBCARRIERS,
        )

        results.append(
            {
                "thingId": tx.name.replace("_", ":").replace("__", "."),
                "name": tx.name,
                "rsrp_dbm": float(rsrp_dbm),
                "rsrp_10_dbm": float(rsrp_dbm) - _Z90 * sigma_db,
                "rsrp_90_dbm": float(rsrp_dbm) + _Z90 * sigma_db,
                "sigma_shadow_db": sigma_db,
                "pathloss_db": float(pl_db),
                "rsrp_baseline_dbm": baseline["rsrp_dbm"],
                "p_los": baseline["p_los"],
                "d_3d_m": baseline["d_3d_m"],
                "has_los_path": bool(los_flags[i]),
            }
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure RSRP at a specific Geo-coordinate."
    )
    parser.add_argument(
        "--lat", type=float, required=True, help="Latitude of the measurement point"
    )
    parser.add_argument(
        "--lon", type=float, required=True, help="Longitude of the measurement point"
    )
    parser.add_argument(
        "--height",
        type=float,
        default=1.5,
        help="Height above ground in meters (default: 1.5)",
    )
    parser.add_argument(
        "--no-vegetation",
        action="store_true",
        help="Disable vegetation attenuation correction.",
    )
    args = parser.parse_args()

    try:
        manager = SceneManager()
        transformer, (ox, oy) = manager.get_transformer()
        x, y = transformer.transform(args.lon, args.lat)
        local_x, local_y = x - ox, y - oy
        logger.info(
            f"Converted ({args.lat}, {args.lon}) -> scene-local ({local_x:.1f}, {local_y:.1f}) m"
        )

        results = measure_rsrp(
            local_x,
            local_y,
            args.height,
            skip_vegetation=args.no_vegetation,
        )

        header_width = 105
        print("\n" + "=" * header_width)
        print(f" RSRP MEASUREMENT AT: {args.lat}, {args.lon} (h={args.height}m)")
        print("=" * header_width)
        sigma = results[0]["sigma_shadow_db"] if results else 0.0
        print(
            f"{'Transmitter Name':<30} | {'RT (dBm)':>10} | {'[p10,p90]':>14} | "
            f"{'UMa (dBm)':>10} | {'PL (dB)':>8} | {'d3D (m)':>8} | {'P_LOS':>5}"
        )
        print("-" * header_width)

        for r in sorted(results, key=lambda x: x["rsrp_dbm"], reverse=True):
            interval = f"[{r['rsrp_10_dbm']:5.1f},{r['rsrp_90_dbm']:5.1f}]"
            print(
                f"{r['name']:<30} | {r['rsrp_dbm']:10.2f} | {interval:>14} | "
                f"{r['rsrp_baseline_dbm']:10.2f} | {r['pathloss_db']:8.2f} | "
                f"{r['d_3d_m']:8.1f} | {r['p_los']:5.2f}"
            )

        if results:
            best_rt = max(results, key=lambda x: x["rsrp_dbm"])
            best_uma = max(results, key=lambda x: x["rsrp_baseline_dbm"])
            print("-" * header_width)
            print(
                f"STRONGEST SERVER (RT):  {best_rt['name']} at {best_rt['rsrp_dbm']:.2f} dBm"
            )
            print(
                f"STRONGEST SERVER (UMa): {best_uma['name']} at {best_uma['rsrp_baseline_dbm']:.2f} dBm"
            )
        print("=" * header_width + "\n")

    except Exception as e:
        logger.exception(f"RSRP measurement failed: {e}")
        sys.exit(1)
