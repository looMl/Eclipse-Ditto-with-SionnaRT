import sys
import numpy as np
import sionna.rt as rt
import argparse
from loguru import logger
from app.simulation.engine import SimulationEngine
from app.simulation.scene_manager import SceneManager
from app.config import settings, get_project_root
from app.geomap_processor.utils.vegetation_field import VegetationField
from app.simulation.vegetation_path_integrator import PathDepthIntegrator
from app.simulation.itu_p833 import excess_loss_db
from app.simulation.baselines.uma_38901 import predict_rsrp_uma


def measure_rsrp(x: float, y: float, z: float = 1.5, skip_vegetation: bool = False):
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
    ds_cfg = getattr(settings.sionnart, "diffuse_scattering", None)
    diffuse = ds_cfg is not None and getattr(ds_cfg, "enabled", False)
    paths = solver(
        scene,
        max_depth=settings.sionnart.coverage.max_depth,
        samples_per_src=settings.sionnart.coverage.samples_per_tx,
        max_num_paths_per_src=settings.sionnart.coverage.max_num_paths_per_src,
        diffuse_reflection=diffuse,
    )

    # Calculate channel gain from paths
    a_data = paths.a

    if isinstance(a_data, tuple):
        real = np.array(a_data[0])
        imag = np.array(a_data[1])
        power_linear = np.sum(real**2 + imag**2, axis=-1)
    else:
        a = np.array(a_data)
        power_linear = np.sum(np.abs(a) ** 2, axis=-1)

    transmitters = list(scene.transmitters.values())
    results = []

    # Vegetation correction setup (no-op when disabled or field absent)
    veg_integrator = None
    veg_freq_hz = 1.8e9
    veg_leaf_state = "in_leaf"
    veg_cfg = getattr(settings.sionnart, "vegetation", None)
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
        else:
            logger.warning(
                "Vegetation field not found; skipping vegetation correction."
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

        # RSRP = Total TX Power + Channel Gain - Subcarrier Offset
        rsrp_dbm = tx_power_total_dbm + float(gain_db) - SUBCARRIER_POWER_OFFSET_DB

        tx_pos = np.array(tx.position, dtype="float32").flatten()[:3]
        if veg_integrator is not None:
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
                "pathloss_db": float(pl_db),
                "rsrp_baseline_dbm": baseline["rsrp_dbm"],
                "p_los": baseline["p_los"],
                "d_3d_m": baseline["d_3d_m"],
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
            f"Converted ({args.lat}, {args.lon}) → scene-local ({local_x:.1f}, {local_y:.1f}) m"
        )

        results = measure_rsrp(
            local_x, local_y, args.height, skip_vegetation=args.no_vegetation
        )

        header_width = 90
        print("\n" + "=" * header_width)
        print(f" RSRP MEASUREMENT AT: {args.lat}, {args.lon} (h={args.height}m)")
        print("=" * header_width)
        print(
            f"{'Transmitter Name':<30} | {'RT (dBm)':>10} | {'UMa (dBm)':>10} | "
            f"{'PL (dB)':>8} | {'d3D (m)':>8} | {'P_LOS':>5}"
        )
        print("-" * header_width)

        for r in sorted(results, key=lambda x: x["rsrp_dbm"], reverse=True):
            print(
                f"{r['name']:<30} | {r['rsrp_dbm']:10.2f} | {r['rsrp_baseline_dbm']:10.2f} | "
                f"{r['pathloss_db']:8.2f} | {r['d_3d_m']:8.1f} | {r['p_los']:5.2f}"
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
