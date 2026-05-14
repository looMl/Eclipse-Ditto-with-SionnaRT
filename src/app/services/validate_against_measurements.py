"""Validate the RT simulation against georeferenced LTE serving-cell measurements.

For each row in the input CSV, converts (lat, lon) to scene-local (x, y, z),
runs ``measure_rsrp`` to obtain RT + UMa predictions for every TX, and records:

* best-server RT error (model's strongest TX vs measured RSRP)
* best-server UMa error (baseline's strongest TX vs measured RSRP)
* per-cell RT error  (RSRP predicted at the *correct* TX from the ECI mapping,
  when available — gives a stronger validation than best-server)

The script is resumable: per-point results land in ``per_point.json`` as soon as
they are computed, so Ctrl+C and reruns skip already-processed rows.

Run with ``--limit 20 --stride 8`` for a stratified smoke test over the dataset.
"""

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from loguru import logger  # noqa: E402

from app.config import get_project_root  # noqa: E402
from app.services.rsrp_measurer import _Z90, measure_rsrp  # noqa: E402
from app.simulation.scene_manager import SceneManager  # noqa: E402


DEFAULT_MEASUREMENTS = "NetworkSurveyData/measurements.csv"
DEFAULT_MAPPING = "NetworkSurveyData/eci_to_thingid.json"
DEFAULT_OUT_DIR = "renders/validation"


def _resolve_local_xy(lat: float, lon: float, manager: SceneManager) -> tuple:
    transformer, (ox, oy) = manager.get_transformer()
    x, y = transformer.transform(lon, lat)
    return x - ox, y - oy


def _atomic_write_json(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _load_cache(path: Path) -> Dict[str, dict]:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}


def _select_rows(rows: List[dict], limit: Optional[int], stride: int) -> List[tuple]:
    """Returns [(global_row_idx, row_dict), ...] using a stride and optional cap."""
    selected = [(i, r) for i, r in enumerate(rows) if i % stride == 0]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _antenna_numeric_id(thingid: str) -> Optional[str]:
    """Extract the numeric part from either thingId format.

    Handles both 'com.sionna:antenna_13610097146' (mapping)
    and 'com:sionna:antenna:13610097152::sector:0' (measure_rsrp results).
    """
    for sep in ("antenna_", "antenna:"):
        if sep in thingid:
            return thingid.split(sep, 1)[1].split(":")[0].split("_")[0]
    return None


def _process_row(
    row: dict,
    manager: SceneManager,
    mapping: Dict[str, dict],
) -> dict:
    lat = float(row["lat"])
    lon = float(row["lon"])
    height = float(row["height"])
    measured = float(row["measured_rsrp_dbm"])
    eci = row.get("serving_cell_id", "")

    x, y = _resolve_local_xy(lat, lon, manager)
    results = measure_rsrp(x, y, height)
    if not results:
        return {"error": "measure_rsrp returned no transmitters"}

    best_rt = max(results, key=lambda r: r["rsrp_dbm"])
    best_uma = max(results, key=lambda r: r["rsrp_baseline_dbm"])

    within = (
        best_rt["rsrp_10_dbm"] <= measured <= best_rt["rsrp_90_dbm"]
        if "rsrp_10_dbm" in best_rt
        else None
    )
    out = {
        "lat": lat,
        "lon": lon,
        "height": height,
        "measured_rsrp_dbm": measured,
        "serving_cell_id": eci,
        "best_rt_thingid": best_rt["thingId"],
        "best_rt_rsrp_dbm": best_rt["rsrp_dbm"],
        "best_rt_rsrp_10_dbm": best_rt.get("rsrp_10_dbm"),
        "best_rt_rsrp_90_dbm": best_rt.get("rsrp_90_dbm"),
        "within_80pct_interval": within,
        "best_rt_error_db": best_rt["rsrp_dbm"] - measured,
        "best_rt_d3d_m": best_rt["d_3d_m"],
        "best_rt_plos": best_rt["p_los"],
        "los_class": "LOS" if best_rt.get("has_los_path", False) else "NLOS",
        "best_uma_thingid": best_uma["thingId"],
        "best_uma_rsrp_dbm": best_uma["rsrp_baseline_dbm"],
        "best_uma_error_db": best_uma["rsrp_baseline_dbm"] - measured,
        "per_cell_thingid": None,
        "per_cell_rt_rsrp_dbm": None,
        "per_cell_rt_error_db": None,
    }

    map_entry = mapping.get(eci) if mapping else None
    mapped_tid = (
        map_entry.get("thingId")
        if map_entry and map_entry.get("status") == "matched"
        else None
    )
    if mapped_tid:
        target_id = _antenna_numeric_id(mapped_tid)
        # Accept any sector of the matched antenna and take the strongest.
        # Sector-level selection by bearing is unreliable in urban NLOS — the
        # geometrically-facing sector is often not the dominant propagation path
        # (Piazza Bra sector audit confirmed this). We constrain to the correct
        # antenna and let the RT model determine which sector is dominant.
        antenna_results = [
            r for r in results if _antenna_numeric_id(r["thingId"]) == target_id
        ]
        if antenna_results:
            match = max(antenna_results, key=lambda r: r["rsrp_dbm"])
            out["per_cell_thingid"] = match["thingId"]
            out["per_cell_rt_rsrp_dbm"] = match["rsrp_dbm"]
            out["per_cell_rt_error_db"] = match["rsrp_dbm"] - measured

    return out


# ---------------------------------------------------------------------------
# Aggregation & plotting
# ---------------------------------------------------------------------------


def _errors(records: List[dict], key: str) -> np.ndarray:
    return np.array(
        [r[key] for r in records if r.get(key) is not None], dtype="float64"
    )


def _interval_coverage(records: List[dict]) -> str:
    flags = [
        r["within_80pct_interval"]
        for r in records
        if r.get("within_80pct_interval") is not None
    ]
    if not flags:
        return "  80% interval coverage: N/A"
    pct = 100.0 * sum(flags) / len(flags)
    sigma = records[0].get("best_rt_rsrp_90_dbm", 0) - records[0].get(
        "best_rt_rsrp_dbm", 0
    )
    return f"  80% interval coverage: {pct:.1f}% of {len(flags)} points  (sigma={sigma / _Z90:.1f} dB, expected ~80%)"


def _summary(name: str, errs: np.ndarray) -> str:
    if errs.size == 0:
        return f"{name:>28}: N=0 (no data)"
    rmse = float(np.sqrt(np.mean(errs**2)))
    bias = float(np.mean(errs))
    mae = float(np.mean(np.abs(errs)))
    p10, p50, p90 = np.percentile(np.abs(errs), [10, 50, 90])
    return (
        f"{name:>28}: N={errs.size:>3}  RMSE={rmse:6.2f}  bias={bias:+6.2f}  "
        f"MAE={mae:5.2f}  |e| p10/50/90={p10:4.1f}/{p50:4.1f}/{p90:4.1f}"
    )


def _centered_summary(name: str, errs: np.ndarray) -> str:
    label = name + " (centered)"
    if errs.size == 0:
        return f"{label:>28}: N=0 (no data)"
    bias = float(np.mean(errs))
    crmse = float(np.sqrt(np.mean((errs - bias) ** 2)))
    return (
        f"{label:>28}: RMSE={crmse:6.2f}"
        f"  (after global {-bias:+.2f} dB offset calibration)"
    )


def _cdf_ks_metrics(
    predictions: np.ndarray, measurements: np.ndarray
) -> tuple[float, float]:
    """Two-sample KS distance and p-value between prediction and measurement ECDFs."""
    stat, pval = ks_2samp(predictions, measurements)
    return float(stat), float(pval)


def _strat_line(label: str, errs: np.ndarray) -> str:
    if errs.size == 0:
        return f"  {label}: N=0 (no data)"
    bias = float(np.mean(errs))
    rmse = float(np.sqrt(np.mean(errs**2)))
    crmse = float(np.sqrt(np.mean((errs - bias) ** 2)))
    return f"  {label} (N={errs.size:>3}): RMSE={rmse:6.2f}  centered={crmse:6.2f}  bias={bias:+6.2f}"


def _plot_error_cdf(records: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [
        ("best_rt_error_db", "RT best-server"),
        ("best_uma_error_db", "UMa best-server"),
        ("per_cell_rt_error_db", "RT per-cell (mapped)"),
    ]:
        errs = np.abs(_errors(records, key))
        if errs.size == 0:
            continue
        xs = np.sort(errs)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=f"{label} (N={errs.size})")
    ax.set_xlabel("|prediction - measurement|  [dB]")
    ax.set_ylabel("CDF")
    ax.set_title("Validation error CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_scatter(records: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    measured = np.array([r["measured_rsrp_dbm"] for r in records])
    rt = np.array([r["best_rt_rsrp_dbm"] for r in records])
    uma = np.array([r["best_uma_rsrp_dbm"] for r in records])
    ax.scatter(measured, rt, alpha=0.6, label="RT best-server", s=24)
    ax.scatter(measured, uma, alpha=0.4, label="UMa best-server", s=24, marker="x")
    lo = min(measured.min(), rt.min(), uma.min()) - 2
    hi = max(measured.max(), rt.max(), uma.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured RSRP  [dBm]")
    ax.set_ylabel("Predicted RSRP  [dBm]")
    ax.set_title("Predicted vs measured")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_error_vs_distance(records: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    d = np.array([r["best_rt_d3d_m"] for r in records])
    e_rt = np.array([r["best_rt_error_db"] for r in records])
    e_uma = np.array([r["best_uma_error_db"] for r in records])
    ax.scatter(d, e_rt, alpha=0.6, label="RT best-server", s=24)
    ax.scatter(d, e_uma, alpha=0.4, label="UMa best-server", s=24, marker="x")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("3D distance to predicted serving TX  [m]")
    ax.set_ylabel("prediction - measurement  [dB]")
    ax.set_title("Signed error vs distance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_cdf_comparison(
    measured: np.ndarray,
    rt_raw: np.ndarray,
    rt_centered: np.ndarray,
    uma_raw: np.ndarray,
    path: Path,
) -> None:
    def _ecdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    fig, ax = plt.subplots(figsize=(9, 5))
    xs, ys = _ecdf(measured)
    ax.plot(xs, ys, color="black", linewidth=2, label="Measured")
    xs, ys = _ecdf(rt_raw)
    ax.plot(xs, ys, color="steelblue", linewidth=1.5, label="RT best-server raw")
    xs, ys = _ecdf(rt_centered)
    ax.plot(
        xs,
        ys,
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        label="RT best-server centered",
    )
    xs, ys = _ecdf(uma_raw)
    ax.plot(xs, ys, color="crimson", linewidth=1.5, label="UMa best-server raw")
    ax.set_xlim(-130, -40)
    ax.set_xlabel("RSRP [dBm]")
    ax.set_ylabel("CDF")
    ax.set_title("ECDF comparison: predicted vs measured RSRP")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--mapping", default=DEFAULT_MAPPING)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max points to process (after stride)."
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Process every Nth row (1 = all rows)."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Discard the per_point.json cache before running.",
    )
    args = parser.parse_args()

    src_root = get_project_root()  # …/src
    repo_root = src_root.parent  # …/  (NetworkSurveyData lives here)
    out_dir = src_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "per_point.json"
    if args.reset and cache_path.exists():
        cache_path.unlink()
    cache = _load_cache(cache_path)

    mapping = {}
    map_path = repo_root / args.mapping
    if map_path.exists():
        with map_path.open() as f:
            mapping = json.load(f)
        n_matched = sum(1 for v in mapping.values() if v.get("status") == "matched")
        logger.info(f"Loaded ECI mapping: {n_matched}/{len(mapping)} cells matched")
    else:
        logger.warning(f"No mapping at {map_path}; per-cell metric will be skipped.")

    meas_path = repo_root / args.measurements
    with meas_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = _select_rows(rows, args.limit, args.stride)
    logger.info(
        f"Selected {len(selected)} / {len(rows)} measurement rows "
        f"(stride={args.stride}, limit={args.limit})"
    )

    manager = SceneManager()

    t0 = time.perf_counter()
    for n, (idx, row) in enumerate(selected, 1):
        key = str(idx)
        if key in cache and "error" not in cache[key]:
            continue
        logger.info(
            f"[{n}/{len(selected)}] row #{idx}  measured={row['measured_rsrp_dbm']} dBm"
        )
        try:
            cache[key] = _process_row(row, manager, mapping)
        except Exception as e:
            logger.exception(f"row #{idx} failed: {e}")
            cache[key] = {"error": str(e)}
        _atomic_write_json(cache_path, cache)
    dt = time.perf_counter() - t0
    logger.success(
        f"Processed selection in {dt:.1f} s ({dt / max(len(selected), 1):.1f} s/point)"
    )

    # Aggregate
    records = [v for v in cache.values() if "error" not in v]
    rt_errs = _errors(records, "best_rt_error_db")
    uma_errs = _errors(records, "best_uma_error_db")
    cell_errs = _errors(records, "per_cell_rt_error_db")

    measured = np.array([r["measured_rsrp_dbm"] for r in records], dtype="float64")
    rt_raw = np.array([r["best_rt_rsrp_dbm"] for r in records], dtype="float64")
    uma_raw = np.array([r["best_uma_rsrp_dbm"] for r in records], dtype="float64")
    bias_rt = float(np.mean(rt_errs)) if rt_errs.size > 0 else 0.0
    rt_centered = rt_raw - bias_rt

    los_records = [r for r in records if r.get("los_class") == "LOS"]
    nlos_records = [r for r in records if r.get("los_class") == "NLOS"]
    rt_los_errs = _errors(los_records, "best_rt_error_db")
    rt_nlos_errs = _errors(nlos_records, "best_rt_error_db")

    summary_lines = [
        f"Validation against {meas_path.name}",
        f"  total points cached: {len(records)}",
        "",
        _summary("RT best-server (raw)", rt_errs),
        _centered_summary("RT best-server", rt_errs),
        "",
        _summary("UMa best-server (raw)", uma_errs),
        _centered_summary("UMa best-server", uma_errs),
        "",
        _summary("RT per-cell (raw)", cell_errs),
        _centered_summary("RT per-cell", cell_errs),
        "",
        _interval_coverage(records),
        "",
        "LOS / NLOS stratification (RT best-server):",
        _strat_line("LOS", rt_los_errs),
        _strat_line("NLOS", rt_nlos_errs),
    ]
    if len(records) >= 2:
        ks_rt_raw = _cdf_ks_metrics(rt_raw, measured)
        ks_rt_cen = _cdf_ks_metrics(rt_centered, measured)
        ks_uma_raw = _cdf_ks_metrics(uma_raw, measured)
        summary_lines += [
            "",
            "CDF agreement (KS distance, lower is better):",
            f"  {'RT best-server raw':>24}: D={ks_rt_raw[0]:.3f}  (p={ks_rt_raw[1]:.3f})",
            f"  {'RT best-server centered':>24}: D={ks_rt_cen[0]:.3f}  (p={ks_rt_cen[1]:.3f})",
            f"  {'UMa best-server raw':>24}: D={ks_uma_raw[0]:.3f}  (p={ks_uma_raw[1]:.3f})",
        ]
    summary = "\n".join(summary_lines)
    print("\n" + summary + "\n")
    (out_dir / "aggregate.txt").write_text(summary + "\n")

    # Wide CSV
    if records:
        fields = list(records[0].keys())
        with (out_dir / "per_point.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(records)

    if len(records) >= 2:
        _plot_error_cdf(records, out_dir / "error_cdf.png")
        _plot_scatter(records, out_dir / "scatter_predicted_vs_measured.png")
        _plot_error_vs_distance(records, out_dir / "error_vs_distance.png")
        _plot_cdf_comparison(
            measured, rt_raw, rt_centered, uma_raw, out_dir / "cdf_comparison.png"
        )
        logger.success(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
