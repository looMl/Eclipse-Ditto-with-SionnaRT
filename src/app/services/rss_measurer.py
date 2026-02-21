import sys
import numpy as np
import sionna.rt as rt
import argparse
from loguru import logger
from app.simulation.engine import SimulationEngine
from app.simulation.scene_manager import SceneManager
from app.config import settings


def measure_rss(lat: float, lon: float, height_m: float = 1.5):
    """
    Measures the Received Signal Strength (RSS) in dBm at a given geo-position.
    """
    SimulationEngine.initialize()

    manager = SceneManager()
    scene = manager.load_scene()

    transformer, (ox, oy) = manager.get_transformer()
    px, py = transformer.transform(lon, lat)
    pos = [px - ox, py - oy, height_m]

    logger.info(f"Measuring RSS at lat={lat}, lon={lon}, height={height_m}")
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

    # Calculate RSS from paths
    a_data = paths.a

    # In some mitsuba variants, paths.a is a tuple (real, imag)
    if isinstance(a_data, tuple):
        real = np.array(a_data[0])
        imag = np.array(a_data[1])
        # power = sum(real^2 + imag^2) over paths (-1)
        power_linear = np.sum(real**2 + imag**2, axis=-1)
    else:
        a = np.array(a_data)
        power_linear = np.sum(np.abs(a) ** 2, axis=-1)

    transmitters = list(scene.transmitters.values())
    results = []

    for i, tx in enumerate(transmitters):
        # Based on observed shape: (num_rx, num_rx_ant, num_tx, num_tx_ant)
        # power_linear[0, :, i, :] gives antenna pairs for receiver 0 and transmitter i
        antenna_pairs_power = power_linear[0, :, i, :]
        power_per_rx_antenna = np.sum(antenna_pairs_power, axis=1)
        total_channel_gain_linear = np.max(power_per_rx_antenna)

        # Convert to dB
        if total_channel_gain_linear > 0:
            gain_db = 10 * np.log10(total_channel_gain_linear)
        else:
            gain_db = -140.0  # Floor for no signal

        pl_db = -gain_db

        # RSS (dBm) = P_tx (dBm) + Gain (dB)
        tx_power = float(np.array(tx.power_dbm).flatten()[0])
        rss_dbm = tx_power + float(gain_db)

        results.append(
            {
                "thingId": tx.name.replace("_", ":").replace("__", "."),
                "name": tx.name,
                "rss_dbm": float(rss_dbm),
                "pathloss_db": float(pl_db),
            }
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure RSS at a specific Geo-coordinate."
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

    # Example usage: uv run -m app.services.rss_measurer --lat 46.0668 --lon 11.1155 --height 1.5
    args = parser.parse_args()

    try:
        results = measure_rss(args.lat, args.lon, args.height)

        print("\n" + "=" * 60)
        print(f" RSS MEASUREMENT AT: {args.lat}, {args.lon} (h={args.height}m)")
        print("=" * 60)
        print(f"{'Transmitter Name':<30} | {'RSS (dBm)':>10} | {'PL (dB)':>8}")
        print("-" * 60)

        for r in sorted(results, key=lambda x: x["rss_dbm"], reverse=True):
            print(f"{r['name']:<30} | {r['rss_dbm']:10.2f} | {r['pathloss_db']:8.2f}")

        if results:
            best = max(results, key=lambda x: x["rss_dbm"])
            print("-" * 60)
            print(f"STRONGEST SERVER: {best['name']} at {best['rss_dbm']:.2f} dBm")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.exception(f"RSS measurement failed: {e}")
        sys.exit(1)
