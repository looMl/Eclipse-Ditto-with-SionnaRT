import sys
import numpy as np
import sionna.rt as rt
import argparse
from loguru import logger
from app.simulation.engine import SimulationEngine
from app.simulation.scene_manager import SceneManager
from app.config import settings


def measure_rsrp(lat: float, lon: float, height_m: float = 1.5):
    """
    Measures the Reference Signal Received Power (RSRP) in dBm at a given geo-position.
    """
    SimulationEngine.initialize()

    manager = SceneManager()
    scene = manager.load_scene()

    transformer, (ox, oy) = manager.get_transformer()
    px, py = transformer.transform(lon, lat)
    pos = [px - ox, py - oy, height_m]

    logger.info(f"Measuring RSRP at lat={lat}, lon={lon}, height={height_m}")
    logger.debug(f"Calculated scene position: {pos}")

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
    paths = solver(
        scene,
        max_depth=settings.sionnart.coverage.max_depth,
        samples_per_src=settings.sionnart.coverage.samples_per_tx,
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

    # 4G LTE Subcarrier Configuration
    # 10 MHz = 50 PRBs * 12 = 600 subcarriers
    # 15 MHz = 75 PRBs * 12 = 900 subcarriers
    NUM_SUBCARRIERS = 900
    SUBCARRIER_POWER_OFFSET_DB = 10 * np.log10(NUM_SUBCARRIERS)

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

        results.append(
            {
                "thingId": tx.name.replace("_", ":").replace("__", "."),
                "name": tx.name,
                "rsrp_dbm": float(rsrp_dbm),
                "pathloss_db": float(pl_db),
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

    args = parser.parse_args()

    try:
        results = measure_rsrp(args.lat, args.lon, args.height)

        print("\n" + "=" * 62)
        print(f" RSRP MEASUREMENT AT: {args.lat}, {args.lon} (h={args.height}m)")
        print("=" * 62)
        print(f"{'Transmitter Name':<30} | {'RSRP (dBm)':>10} | {'PL (dB)':>8}")
        print("-" * 62)

        for r in sorted(results, key=lambda x: x["rsrp_dbm"], reverse=True):
            print(f"{r['name']:<30} | {r['rsrp_dbm']:10.2f} | {r['pathloss_db']:8.2f}")

        if results:
            best = max(results, key=lambda x: x["rsrp_dbm"])
            print("-" * 62)
            print(f"STRONGEST SERVER: {best['name']} at {best['rsrp_dbm']:.2f} dBm")
        print("=" * 62 + "\n")

    except Exception as e:
        logger.exception(f"RSRP measurement failed: {e}")
        sys.exit(1)
