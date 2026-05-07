"""
Module 3c — Temporal Graph Dataset
=====================================

Converts time-series ward feature snapshots into sliding-window samples
that the ST-GNN consumes.

Data layout expected on disk
──────────────────────────────
  Parquet (from Module 1 Airflow DAG):
    ward_features_history.parquet
      Columns: ward_id, snapshot_ts (ISO-8601), ndvi, ndbi, impervious_pct,
               lst_celsius, rainfall_mm_24h, pm25_ugm3, population_density,
               heat_stress_score, flood_risk_score

  One row per (ward, snapshot).  Snapshots arrive hourly from the
  daily_feature_pipeline_dag.  Missing snapshots are forward-filled.

Sliding window sample
──────────────────────
  At each snapshot index t, a sample is:
    x_seq  : (N, T, F)  — last T snapshots of F features for all N wards
    y_heat : (N,)       — heat_stress_score at t  (regression target 0–100)
    y_flood: (N,)       — flood_risk_score  at t  (regression target 0–100)
    t_idx  : int        — snapshot index (useful for debugging)

Fully synthetic fallback
─────────────────────────
  When no Parquet is present, the dataset generates a synthetic time series
  (200 snapshots × 198 wards) with realistic diurnal / seasonal patterns so
  the entire training pipeline works offline.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

_MODULE_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "8_data"))

# Feature columns (order matches NODE_FEAT_DIM=12 in model.py)
FEATURE_COLS = [
    "ndvi", "ndbi", "impervious_norm",
    "lst_norm", "rainfall_log", "pm25_log",
    "pop_log", "area_norm", "industrial_weight",
    "lon_norm", "lat_norm", "heat_proxy",
]
TARGET_COLS  = ["heat_stress_score", "flood_risk_score"]
NUM_WARDS    = 198
T_WINDOW     = 8    # history timesteps per sample


# ---------------------------------------------------------------------------
# Feature normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived / normalised feature columns to a ward-snapshot DataFrame.
    Operates in-place on a copy.
    """
    out = df.copy()

    def _log(col: str, eps: float = 0.1, scale: float = 6.0) -> pd.Series:
        return (out[col].clip(lower=0).add(eps).apply(math.log) / scale).astype("float32")

    def _clip(col: str, lo: float, hi: float) -> pd.Series:
        return ((out[col].clip(lo, hi) - lo) / (hi - lo + 1e-8)).astype("float32")

    out["impervious_norm"]   = _clip("impervious_pct", 0, 100)
    out["lst_norm"]          = _clip("lst_celsius",   20, 50)
    out["rainfall_log"]      = _log("rainfall_mm_24h")
    out["pm25_log"]          = _log("pm25_ugm3", scale=6.0)
    out["pop_log"]           = _log("population_density", scale=12.0)

    # Synthetic / static columns if not present
    if "area_norm" not in out.columns:
        out["area_norm"] = 0.5
    if "industrial_weight" not in out.columns:
        out["industrial_weight"] = 0.0
    if "lon_norm" not in out.columns:
        out["lon_norm"] = _clip("centroid_lon", 77.45, 77.78) if "centroid_lon" in out.columns else 0.5
    if "lat_norm" not in out.columns:
        out["lat_norm"] = _clip("centroid_lat", 12.83, 13.14) if "centroid_lat" in out.columns else 0.5
    if "heat_proxy" not in out.columns:
        out["heat_proxy"] = out["lst_norm"]

    # Default targets if missing
    if "heat_stress_score" not in out.columns:
        out["heat_stress_score"] = (out["lst_norm"] * 100).clip(0, 100)
    if "flood_risk_score" not in out.columns:
        out["flood_risk_score"] = (out["rainfall_log"] * 30).clip(0, 100)

    return out


# ---------------------------------------------------------------------------
# Synthetic time-series generator
# ---------------------------------------------------------------------------

def _synthetic_history(
    n_snapshots: int = 200,
    n_wards: int = NUM_WARDS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic (n_snapshots × n_wards) time-series DataFrame.

    Adds diurnal temperature variation, random rainfall events, and
    slow-moving spatial patterns to create realistic training data.
    """
    rng = np.random.default_rng(seed)

    # Build static ward properties
    cols = 14
    rows_g = math.ceil(n_wards / cols)
    records = []
    for idx in range(n_wards):
        r, c = divmod(idx, cols)
        cbd_dist  = math.sqrt((r - 7) ** 2 + (c - 7) ** 2)
        ind_score = math.exp(-cbd_dist / 4) + 0.5 * math.exp(-((r-11)**2+(c-2)**2)/4)
        base_lst  = 35 + 5 * min(1.0, ind_score) + rng.normal(0, 1.5)
        base_ndvi = 0.4 - 0.25 * min(1.0, ind_score) + rng.uniform(-0.05, 0.05)
        records.append({
            "ward_id":           idx + 1,
            "base_lst":          float(np.clip(base_lst, 22, 48)),
            "base_ndvi":         float(np.clip(base_ndvi, 0.02, 0.75)),
            "base_pm25":         float(np.clip(40 + 25 * min(1, ind_score) + rng.normal(0, 5), 10, 150)),
            "industrial_weight": float(np.clip(ind_score, 0, 1)),
            "area_norm":         0.5,
            "lon_norm":          c / (cols - 1),
            "lat_norm":          r / (rows_g - 1),
        })
    static_df = pd.DataFrame(records)

    # Build time-varying snapshots
    all_rows = []
    base_ts  = pd.Timestamp("2024-01-01")
    for snap in range(n_snapshots):
        hour      = snap % 24
        day       = snap // 24
        diurnal   = 4.0 * math.sin(math.pi * (hour - 6) / 12)
        rain_event = rng.random() < 0.08   # 8% chance per hour

        for _, w in static_df.iterrows():
            noise_lst  = rng.normal(0, 0.5)
            lst = w["base_lst"] + diurnal + noise_lst
            ndvi = w["base_ndvi"] + rng.uniform(-0.02, 0.02)
            ndbi = float(np.clip(-ndvi * 0.8 + 0.1 + w["industrial_weight"] * 0.2, -0.5, 0.6))
            rainfall  = float(rng.exponential(8.0) if rain_event else rng.exponential(0.3))
            pm25 = w["base_pm25"] * (1 + 0.2 * math.sin(math.pi * hour / 12)) + rng.normal(0, 3)

            heat_stress = float(np.clip(
                0.35 * (lst - 22) / 26 * 100
                + 0.3 * w["industrial_weight"] * 100
                - 0.2 * ndvi * 100
                + rng.normal(0, 3), 0, 100
            ))
            flood_risk = float(np.clip(
                min(1, rainfall / 20) * 80
                + 0.1 * (1 - ndvi) * 30
                + rng.normal(0, 2), 0, 100
            ))

            all_rows.append({
                "ward_id":            int(w["ward_id"]),
                "snapshot_ts":        (base_ts + pd.Timedelta(hours=snap)).isoformat(),
                "snapshot_idx":       snap,
                "ndvi":               round(float(np.clip(ndvi, 0.02, 0.75)), 4),
                "ndbi":               round(ndbi, 4),
                "impervious_pct":     round(float(np.clip((ndbi + 1) / 2 * 100, 30, 98)), 2),
                "lst_celsius":        round(float(np.clip(lst, 22, 48)), 2),
                "rainfall_mm_24h":    round(rainfall, 3),
                "pm25_ugm3":          round(float(np.clip(pm25, 5, 200)), 2),
                "population_density": float(rng.lognormal(8.5 - 0.5 * w["industrial_weight"], 0.5)),
                "industrial_weight":  float(w["industrial_weight"]),
                "area_norm":          float(w["area_norm"]),
                "lon_norm":           float(w["lon_norm"]),
                "lat_norm":           float(w["lat_norm"]),
                "heat_stress_score":  round(heat_stress, 2),
                "flood_risk_score":   round(flood_risk, 2),
            })

    df = pd.DataFrame(all_rows)
    logger.info(
        f"Synthetic history | snapshots={n_snapshots}  wards={n_wards}  "
        f"total_rows={len(df)}"
    )
    return df


# ---------------------------------------------------------------------------
# WardTemporalDataset
# ---------------------------------------------------------------------------

class WardTemporalDataset(Dataset):
    """
    Sliding-window temporal graph dataset.

    Each __getitem__ returns one window of T consecutive snapshots for
    all N wards:
      x_seq  : (N, T, F) float32
      y_heat : (N,)       float32 — 0..100
      y_flood: (N,)       float32 — 0..100
      t_idx  : int
    """

    def __init__(
        self,
        source: Optional[str | Path | pd.DataFrame] = None,
        t_window: int = T_WINDOW,
        n_snapshots: int = 200,
        seed: int = 42,
    ) -> None:
        self.t_window = t_window
        df = self._load(source, n_snapshots, seed)
        df = _normalise_snapshot(df)

        # Build (snapshot_idx, ward_id) pivot tensors
        # Sort wards consistently
        self._ward_ids = sorted(df["ward_id"].unique())
        self._n_wards  = len(self._ward_ids)
        self._snaps    = sorted(df["snapshot_idx"].unique())
        self._n_snaps  = len(self._snaps)

        # Pivot: (n_snaps, n_wards, n_features)
        ward_idx_map = {w: i for i, w in enumerate(self._ward_ids)}
        snap_idx_map = {s: i for i, s in enumerate(self._snaps)}

        feat_arr   = np.zeros((self._n_snaps, self._n_wards, len(FEATURE_COLS)), dtype=np.float32)
        target_arr = np.zeros((self._n_snaps, self._n_wards, 2), dtype=np.float32)

        for _, row in df.iterrows():
            si = snap_idx_map[row["snapshot_idx"]]
            wi = ward_idx_map[row["ward_id"]]
            for fi, col in enumerate(FEATURE_COLS):
                feat_arr[si, wi, fi] = float(row.get(col, 0.0))
            target_arr[si, wi, 0] = float(row.get("heat_stress_score", 0.0))
            target_arr[si, wi, 1] = float(row.get("flood_risk_score",  0.0))

        # Per-feature standard scaling (zero mean, unit variance)
        flat = feat_arr.reshape(-1, len(FEATURE_COLS))
        self._feat_mean = flat.mean(axis=0)
        self._feat_std  = flat.std(axis=0) + 1e-8
        feat_arr = (feat_arr - self._feat_mean) / self._feat_std

        self._feat   = torch.from_numpy(feat_arr)    # (S, N, F)
        self._target = torch.from_numpy(target_arr)  # (S, N, 2)

        # Valid windows: need T past steps + 1 target step
        self._valid_starts = list(range(t_window, self._n_snaps))

        logger.info(
            f"WardTemporalDataset | wards={self._n_wards}  "
            f"snapshots={self._n_snaps}  samples={len(self)}"
        )

    # ------------------------------------------------------------------
    def _load(
        self,
        source: Optional[str | Path | pd.DataFrame],
        n_snapshots: int,
        seed: int,
    ) -> pd.DataFrame:
        if source is None:
            return _synthetic_history(n_snapshots=n_snapshots, seed=seed)
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            path = Path(source)
            if not path.exists():
                logger.warning(f"{path} not found — using synthetic history")
                return _synthetic_history(n_snapshots=n_snapshots, seed=seed)
            df = pd.read_parquet(path)

        if "snapshot_idx" not in df.columns:
            # Derive integer index from sorted timestamp
            ts = pd.to_datetime(df["snapshot_ts"])
            unique_ts = sorted(ts.unique())
            ts_map = {t: i for i, t in enumerate(unique_ts)}
            df["snapshot_idx"] = ts.map(ts_map)
        return df

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        t = self._valid_starts[idx]
        x_seq  = self._feat[t - self.t_window : t]       # (T, N, F)
        x_seq  = x_seq.permute(1, 0, 2)                  # (N, T, F)
        y      = self._target[t]                          # (N, 2)
        y_heat  = y[:, 0]
        y_flood = y[:, 1]
        return x_seq, y_heat, y_flood, t

    # ------------------------------------------------------------------
    def get_ward_ids(self) -> List[int]:
        return list(self._ward_ids)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    parquet_path: Optional[str | Path] = None,
    t_window: int = T_WINDOW,
    n_snapshots: int = 200,
    val_fraction: float = 0.20,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders using a chronological split.

    The last `val_fraction` of the timeline is held out for validation
    (no temporal leakage).
    """
    full_ds = WardTemporalDataset(
        source=parquet_path, t_window=t_window,
        n_snapshots=n_snapshots, seed=seed,
    )
    n_total = len(full_ds)
    n_val   = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    # Pure temporal split: first 80% train, last 20% val (no leakage)
    train_ds = torch.utils.data.Subset(full_ds, list(range(n_train)))
    val_ds   = torch.utils.data.Subset(full_ds, list(range(n_train, n_total)))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    logger.info(
        f"DataLoaders | train={n_train}  val={n_val}  "
        f"batch={batch_size}"
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Delta Lake integration helper
# ---------------------------------------------------------------------------

def read_latest_features_from_delta(delta_path: str, n_wards: int = 198):
    """
    Read latest ward features from Delta Lake.
    Returns pandas DataFrame or None if Delta Lake is unavailable.
    Falls back gracefully — caller should use synthetic data if None returned.
    """
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(f"{delta_path}/ward_features")
        df = dt.to_pandas()
        logger.info(f"Loaded {len(df)} rows from Delta Lake ward_features")
        return df
    except Exception as e:
        logger.warning(f"Delta Lake unavailable ({e}), caller should use synthetic fallback")
        return None


# ---------------------------------------------------------------------------
# __main__ — smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger

    logger.info("WardTemporalDataset smoke test")

    ds = WardTemporalDataset(n_snapshots=50)
    assert len(ds) == 50 - T_WINDOW, f"expected {50 - T_WINDOW} samples"

    x, yh, yf, t = ds[0]
    assert x.shape  == (NUM_WARDS, T_WINDOW, len(FEATURE_COLS)), f"x shape: {x.shape}"
    assert yh.shape == (NUM_WARDS,), f"y_heat shape: {yh.shape}"
    assert yf.shape == (NUM_WARDS,), f"y_flood shape: {yf.shape}"
    logger.info(
        f"Sample 0 | t={t}  x_mean={x.mean():.3f}  "
        f"heat μ={yh.mean():.1f}  flood μ={yf.mean():.1f}"
    )

    train_dl, val_dl = make_dataloaders(n_snapshots=50, batch_size=4)
    bx, byh, byf, bt = next(iter(train_dl))
    assert bx.shape == (4, NUM_WARDS, T_WINDOW, len(FEATURE_COLS))
    logger.info(f"Batch shape: {bx.shape}")
    logger.success("dataset.py smoke test passed.")
