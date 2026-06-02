"""GPR spatial correction for RT validation residuals.

Fits a Gaussian Process on (lon, lat) → RT prediction error, then evaluates
via leave-one-out cross-validation to estimate how much a spatial correction
layer reduces centered RMSE over the raw RT output.

Usage:
    python -m app.services.gpr_analysis
    python -m app.services.gpr_analysis --validation-dir renders/validation_phase3_fixed
"""

import argparse
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from app.config import get_project_root


DEFAULT_VALIDATION_DIR = "renders/validation_phase3_fixed"
_METRES_PER_DEGREE = 111319.5  # WGS84 metres per degree of latitude at the equator


def _load_records(path: Path) -> list:
    with path.open() as f:
        data = json.load(f)
    return [
        v
        for v in data.values()
        if "error" not in v and v.get("best_rt_error_db") is not None
    ]


def _to_metres(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Equirectangular projection to local metres, centred on dataset mean."""
    lon0, lat0 = lons.mean(), lats.mean()
    x = (lons - lon0) * np.cos(np.radians(lat0)) * _METRES_PER_DEGREE
    y = (lats - lat0) * _METRES_PER_DEGREE
    return np.column_stack([x, y])


def _stats(errors: np.ndarray) -> dict:
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    crmse = float(np.sqrt(np.mean((errors - bias) ** 2)))
    mae = float(np.mean(np.abs(errors)))
    return {"bias": bias, "rmse": rmse, "crmse": crmse, "mae": mae}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", default=DEFAULT_VALIDATION_DIR)
    args = parser.parse_args()

    src_root = get_project_root()
    val_dir = src_root / args.validation_dir
    records = _load_records(val_dir / "per_point.json")
    print(f"Loaded {len(records)} points from {val_dir.name}")

    lons = np.array([r["lon"] for r in records])
    lats = np.array([r["lat"] for r in records])
    errors = np.array([r["best_rt_error_db"] for r in records])
    X = _to_metres(lons, lats)

    # Step 1: fit on full dataset to optimise hyperparameters
    kernel = RBF(length_scale=150.0, length_scale_bounds=(20.0, 1000.0)) + WhiteKernel(
        noise_level=30.0, noise_level_bounds=(0.1, 300.0)
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True, alpha=1e-6
    )
    gpr.fit(X, errors)
    print(f"Optimised kernel: {gpr.kernel_}")

    # Step 2: LOO-CV with fixed (optimised) hyperparameters
    # Re-optimising per fold would be O(n^4); fixing hyperparameters is standard.
    loo_pred = np.zeros(len(errors))
    fixed_kernel = gpr.kernel_
    for i in range(len(errors)):
        mask = np.ones(len(errors), dtype=bool)
        mask[i] = False
        gpr_i = GaussianProcessRegressor(
            kernel=fixed_kernel, optimizer=None, normalize_y=True, alpha=1e-6
        )
        gpr_i.fit(X[mask], errors[mask])
        loo_pred[i] = gpr_i.predict(X[[i]])[0]

    corrected = errors - loo_pred

    # Geographic 4-fold CV (NE/NW/SE/SW quadrants) — conservative estimate.
    # Trains on 3 quadrants, predicts the 4th. Tests generalization to unmeasured areas.
    lon_mid, lat_mid = lons.mean(), lats.mean()
    geo_folds = (lons > lon_mid).astype(int) * 2 + (lats > lat_mid).astype(int)
    geo_pred = np.zeros(len(errors))
    for fold_id in range(4):
        test_mask = geo_folds == fold_id
        train_mask = ~test_mask
        if test_mask.sum() < 2:
            geo_pred[test_mask] = 0.0
            continue
        gpr_geo = GaussianProcessRegressor(
            kernel=fixed_kernel, optimizer=None, normalize_y=True, alpha=1e-6
        )
        gpr_geo.fit(X[train_mask], errors[train_mask])
        geo_pred[test_mask] = gpr_geo.predict(X[test_mask])
    geo_corrected = errors - geo_pred

    orig = _stats(errors)
    corr = _stats(corrected)
    geo = _stats(geo_corrected)
    delta_loo = orig["crmse"] - corr["crmse"]
    delta_geo = orig["crmse"] - geo["crmse"]

    fold_counts = [int((geo_folds == k).sum()) for k in range(4)]

    lines = [
        "GPR Spatial Correction Results",
        f"  N = {len(errors)}  |  kernel: {gpr.kernel_}",
        f"  Geographic folds (SW/NW/SE/NE): {'/'.join(str(c) for c in fold_counts)} points",
        "",
        f"  {'':32}  RMSE    bias   centered    MAE",
        f"  {'RT best-server (original)':32}  {orig['rmse']:5.2f}  {orig['bias']:+6.2f}  {orig['crmse']:8.2f}  {orig['mae']:5.2f}",
        f"  {'GPR corrected (point LOO-CV)':32}  {corr['rmse']:5.2f}  {corr['bias']:+6.2f}  {corr['crmse']:8.2f}  {corr['mae']:5.2f}",
        f"  {'GPR corrected (geo 4-fold CV)':32}  {geo['rmse']:5.2f}  {geo['bias']:+6.2f}  {geo['crmse']:8.2f}  {geo['mae']:5.2f}",
        "",
        f"  Point LOO-CV improvement:  {orig['crmse']:.2f} -> {corr['crmse']:.2f} dB  (delta = {delta_loo:+.2f} dB)",
        f"  Geo 4-fold CV improvement: {orig['crmse']:.2f} -> {geo['crmse']:.2f} dB  (delta = {delta_geo:+.2f} dB)",
        "",
        "  Note: point LOO-CV is optimistic (exploits walk-test neighbour clustering).",
        "  Geo 4-fold CV is the conservative estimate for unmeasured-area prediction.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    report_path = val_dir / "gpr_correction.txt"
    tmp_path = report_path.with_suffix(".txt.tmp")
    tmp_path.write_text(report + "\n")
    tmp_path.replace(report_path)

    # Spatial error maps
    vabs = max(
        np.percentile(np.abs(errors), 95),
        np.percentile(np.abs(corrected), 95),
        np.percentile(np.abs(geo_corrected), 95),
    )
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    for ax, vals, title in [
        (ax1, errors, "RT errors — original"),
        (ax2, corrected, "GPR corrected (point LOO-CV)"),
        (ax3, geo_corrected, "GPR corrected (geo 4-fold CV)"),
    ]:
        sc = ax.scatter(
            lons, lats, c=vals, cmap="RdBu_r", vmin=-vabs, vmax=vabs, s=35, alpha=0.8
        )
        plt.colorbar(sc, ax=ax, label="Error [dB]")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(val_dir / "gpr_spatial_error_map.png", dpi=120)
    plt.close(fig)
    print(f"\nResults saved to {val_dir}/")


if __name__ == "__main__":
    main()
