"""
Module 1c — Spark Batch Processor
==================================

PySpark job that:

  1. Loads Landsat 8/9 band data (real GeoTIFFs from GEE or the synthetic
     .npz fallback produced by 8_data/sample_landsat_loader.py).
  2. Computes per-pixel spectral indices:
       NDVI = (B5 - B4) / (B5 + B4)      ← vegetation
       NDBI = (B6 - B5) / (B6 + B5)      ← built-up / impervious
  3. Converts thermal band B10 to Land Surface Temperature in Celsius
     using the Collection-2 scale factors.
  4. Spatially joins each BBMP ward polygon with the raster pixels that
     fall within it and aggregates mean NDVI, NDBI, impervious %, and LST
     per ward.
  5. Writes the result to a Delta Lake table `satellite_features`.
  6. Also writes the result to PostgreSQL for the FastAPI backend.

Impervious % estimation
───────────────────────
  impervious_pct = clip((NDBI + 1) / 2 × 100, 0, 100)
  This is a simplified linear transform; for production a calibrated
  non-linear model trained against NLCD ground truth would be used.

Offline fallback
────────────────
When neither real GeoTIFFs nor the synthetic .npz exist, the job generates
a fully synthetic per-ward feature table using the same statistical
distributions as mock_ward_features.py so the pipeline never stalls.

Usage:
    # Full pipeline (auto-detects data):
    python 1_data_pipeline/spark_batch_processor.py

    # Force synthetic data (skip raster loading):
    python 1_data_pipeline/spark_batch_processor.py --synthetic

    # Write to Postgres only (skip Delta Lake):
    python 1_data_pipeline/spark_batch_processor.py --no-delta

    # Dry run (print results, write nothing):
    python 1_data_pipeline/spark_batch_processor.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project paths
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "8_data"))

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class SparkSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ROOT / ".env"), extra="ignore")

    delta_lake_root: str = "/data/delta"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "namma_clima_grid"
    postgres_user: str = "ncg_user"
    postgres_password: str = "change_me_in_production"


_cfg = SparkSettings()

# ---------------------------------------------------------------------------
# Landsat Collection-2 scale factors
# ---------------------------------------------------------------------------
SR_SCALE  = 0.0000275   # Surface reflectance scale
SR_OFFSET = -0.2        # Surface reflectance offset
ST_SCALE  = 0.00341802  # ST_B10 scale (Kelvin)
ST_OFFSET = 149.0       # ST_B10 offset (Kelvin)


# ---------------------------------------------------------------------------
# Raster utilities (numpy-based, no GDAL required for the npz path)
# ---------------------------------------------------------------------------

def load_bands(scene_dir: Path) -> Optional[dict[str, np.ndarray]]:
    """
    Load Landsat bands from disk.  Tries GeoTIFFs first, then .npz fallback.
    Returns None if neither is available.
    """
    # GeoTIFF path (from GEE download)
    tif_files = sorted(scene_dir.glob("*_B*.tif"))
    if tif_files:
        try:
            import rasterio  # type: ignore
            bands: dict[str, np.ndarray] = {}
            for b in ["B2", "B3", "B4", "B5", "B6", "B7", "B10"]:
                matches = [f for f in tif_files if f.name.endswith(f"_{b}.tif")]
                if not matches:
                    raise FileNotFoundError(f"Missing {b}")
                with rasterio.open(matches[0]) as src:
                    bands[b] = src.read(1).astype(np.float32)
            logger.info(f"Loaded real GeoTIFF bands from {scene_dir}")
            return bands
        except (ImportError, FileNotFoundError) as e:
            logger.warning(f"GeoTIFF load failed ({e}), trying .npz")

    # .npz path (synthetic fallback)
    npz = scene_dir / "synthetic_landsat_scene.npz"
    if npz.exists():
        data = np.load(npz)
        bands = {b: data[b].astype(np.float32) for b in ["B2","B3","B4","B5","B6","B7","B10"]}
        logger.info(f"Loaded synthetic .npz bands from {npz}")
        return bands

    return None


def compute_indices(
    bands: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute NDVI, NDBI, and LST from Landsat DN arrays.

    Returns:
        ndvi  : float32 array, range approx -1..1
        ndbi  : float32 array, range approx -1..1
        lst_c : float32 array, Land Surface Temp in Celsius
    """
    # Convert DN → physical reflectance
    b4 = bands["B4"] * SR_SCALE + SR_OFFSET   # Red
    b5 = bands["B5"] * SR_SCALE + SR_OFFSET   # NIR
    b6 = bands["B6"] * SR_SCALE + SR_OFFSET   # SWIR-1

    # NDVI
    denom_vi = b5 + b4
    ndvi = np.where(denom_vi != 0, (b5 - b4) / denom_vi, 0.0).astype(np.float32)

    # NDBI
    denom_bi = b6 + b5
    ndbi = np.where(denom_bi != 0, (b6 - b5) / denom_bi, 0.0).astype(np.float32)

    # LST — Celsius
    lst_k  = bands["B10"] * ST_SCALE + ST_OFFSET
    lst_c  = (lst_k - 273.15).astype(np.float32)

    return ndvi, ndbi, lst_c


def impervious_from_ndbi(ndbi: np.ndarray) -> np.ndarray:
    """
    Estimate impervious surface % from NDBI.

    Linear mapping: NDBI=-1 → 0%, NDBI=+1 → 100%.
    """
    return np.clip((ndbi + 1.0) / 2.0 * 100.0, 0.0, 100.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Ward–pixel spatial join (lightweight grid-based approach)
# ---------------------------------------------------------------------------

def ward_stats_from_raster(
    ndvi: np.ndarray,
    ndbi: np.ndarray,
    lst_c: np.ndarray,
    n_wards: int = 198,
    scene_date: Optional[str] = None,
    source_scene: str = "SYNTHETIC",
) -> List[dict]:
    """
    Assign pixels to wards using a simple grid tiling (matching the synthetic
    ward grid in _common.py).  For real GeoTIFFs with known CRS a proper
    GeoPandas spatial join would be used instead.

    Returns a list of per-ward dicts.
    """
    import math
    h, w = ndvi.shape
    cols = 14
    rows_grid = math.ceil(n_wards / cols)
    dy = h / rows_grid
    dx = w / cols

    if scene_date is None:
        scene_date = date.today().isoformat()

    records: List[dict] = []
    for ward_idx in range(n_wards):
        r = ward_idx // cols
        c = ward_idx % cols
        r0, r1 = int(r * dy),      int((r + 1) * dy)
        c0, c1 = int(c * dx),      int((c + 1) * dx)

        patch_ndvi = ndvi[r0:r1, c0:c1]
        patch_ndbi = ndbi[r0:r1, c0:c1]
        patch_lst  = lst_c[r0:r1, c0:c1]
        patch_imp  = impervious_from_ndbi(patch_ndbi)

        records.append({
            "ward_id":        ward_idx + 1,
            "scene_date":     scene_date,
            "ndvi":           round(float(np.nanmean(patch_ndvi)), 4),
            "ndbi":           round(float(np.nanmean(patch_ndbi)), 4),
            "impervious_pct": round(float(np.nanmean(patch_imp)),  2),
            "lst_celsius":    round(float(np.nanmean(patch_lst)),  2),
            "source_scene":   source_scene,
        })
    return records


# ---------------------------------------------------------------------------
# Synthetic fallback — no raster data at all
# ---------------------------------------------------------------------------

def _synthetic_ward_stats(n_wards: int = 198) -> List[dict]:
    """
    Produce synthetic satellite features using the same statistical model as
    mock_ward_features.py when no real or synthetic raster is available.
    """
    import math
    rng = np.random.default_rng(42)
    cols = 14
    today = date.today().isoformat()

    records = []
    for idx in range(n_wards):
        r = idx // cols
        c = idx % cols
        cbd  = math.exp(-(((r-7)**2 + (c-7)**2) / 6))
        peen = math.exp(-(((r-11)**2 + (c-2)**2) / 5))
        eco  = math.exp(-(((r-1)**2 + (c-11)**2) / 5))
        industrial = min(1.0, 0.6*cbd + 0.8*peen + 0.7*eco)

        ndvi = float(np.clip(rng.uniform(0.1, 0.5) - 0.25*industrial, 0.02, 0.75))
        ndbi = float(np.clip(rng.uniform(-0.2, 0.4) + 0.2*industrial, -0.5, 0.6))
        imp  = float(np.clip((ndbi + 1) / 2 * 100, 30, 98))
        lst  = float(np.clip(rng.normal(35, 4) + 5*industrial - 6*ndvi, 22, 48))

        records.append({
            "ward_id":        idx + 1,
            "scene_date":     today,
            "ndvi":           round(ndvi, 4),
            "ndbi":           round(ndbi, 4),
            "impervious_pct": round(imp, 2),
            "lst_celsius":    round(lst, 2),
            "source_scene":   "SYNTHETIC-FALLBACK",
        })
    return records


# ---------------------------------------------------------------------------
# Write to PostgreSQL
# ---------------------------------------------------------------------------

def write_to_postgres(records: List[dict]) -> None:
    """Upsert satellite features into the PostgreSQL ward_satellite_features table."""
    try:
        import psycopg2  # type: ignore
        from psycopg2.extras import execute_values
    except ImportError:
        logger.warning("psycopg2 not installed — skipping Postgres write")
        return

    dsn = (
        f"host={_cfg.postgres_host} port={_cfg.postgres_port} "
        f"dbname={_cfg.postgres_db} user={_cfg.postgres_user} "
        f"password={_cfg.postgres_password}"
    )
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ward_satellite_features (
                ward_id        INTEGER PRIMARY KEY,
                scene_date     DATE,
                ndvi           DOUBLE PRECISION,
                ndbi           DOUBLE PRECISION,
                impervious_pct DOUBLE PRECISION,
                lst_celsius    DOUBLE PRECISION,
                source_scene   TEXT,
                updated_at     TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        rows = [
            (
                r["ward_id"], r["scene_date"], r["ndvi"], r["ndbi"],
                r["impervious_pct"], r["lst_celsius"], r["source_scene"],
            )
            for r in records
        ]
        execute_values(cur, """
            INSERT INTO ward_satellite_features
                (ward_id, scene_date, ndvi, ndbi, impervious_pct, lst_celsius, source_scene)
            VALUES %s
            ON CONFLICT (ward_id) DO UPDATE SET
                scene_date     = EXCLUDED.scene_date,
                ndvi           = EXCLUDED.ndvi,
                ndbi           = EXCLUDED.ndbi,
                impervious_pct = EXCLUDED.impervious_pct,
                lst_celsius    = EXCLUDED.lst_celsius,
                source_scene   = EXCLUDED.source_scene,
                updated_at     = NOW()
        """, rows)

        conn.commit()
        cur.close()
        conn.close()
        logger.success(f"Upserted {len(rows)} rows → ward_satellite_features")
    except Exception as exc:
        logger.error(f"Postgres write failed: {exc}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    scene_dir: Path,
    force_synthetic: bool = False,
    write_delta: bool = True,
    dry_run: bool = False,
) -> List[dict]:
    """
    Execute the full satellite feature extraction pipeline.

    Returns the list of per-ward feature dicts.
    """
    # 1. Load or synthesise raster data
    if force_synthetic:
        logger.info("Forcing synthetic ward stats (--synthetic flag set)")
        records = _synthetic_ward_stats()
        source = "SYNTHETIC-FORCED"
    else:
        bands = load_bands(scene_dir)
        if bands is None:
            logger.warning(
                f"No raster data found in {scene_dir}. "
                "Using fully synthetic fallback."
            )
            records = _synthetic_ward_stats()
            source = "SYNTHETIC-FALLBACK"
        else:
            logger.info("Computing spectral indices from raster data...")
            ndvi, ndbi, lst_c = compute_indices(bands)
            logger.info(
                f"NDVI  μ={ndvi.mean():.3f}  σ={ndvi.std():.3f}  "
                f"range=[{ndvi.min():.2f}, {ndvi.max():.2f}]"
            )
            logger.info(
                f"NDBI  μ={ndbi.mean():.3f}  σ={ndbi.std():.3f}"
            )
            logger.info(
                f"LST   μ={lst_c.mean():.1f}°C  σ={lst_c.std():.1f}°C"
            )
            records = ward_stats_from_raster(ndvi, ndbi, lst_c)
            source = records[0]["source_scene"] if records else "RASTER"

    logger.info(
        f"Computed features for {len(records)} wards | source={source}"
    )

    if dry_run:
        logger.info("Dry run — printing first 5 ward records:")
        for r in records[:5]:
            logger.info(f"  {r}")
        return records

    # 2. Write to Delta Lake via Spark
    if write_delta:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent))
            from delta_lake_writer import DeltaLakeWriter, get_spark, initialise_tables  # type: ignore

            spark = get_spark("SatelliteFeatureJob", delta_root=_cfg.delta_lake_root)
            from pyspark.sql import Row
            rows = [Row(**{k: v for k, v in r.items()}) for r in records]
            df = spark.createDataFrame(rows)

            writer = DeltaLakeWriter(spark, root=_cfg.delta_lake_root)
            initialise_tables(writer)
            writer.write_batch(
                "satellite_features", df,
                mode="overwrite",
                merge_key="ward_id",
            )
            spark.stop()
        except Exception as exc:
            logger.error(f"Delta write failed ({exc}) — skipping Delta, writing Postgres only")

    # 3. Write to PostgreSQL
    write_to_postgres(records)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid — Spark satellite feature extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scene-dir", type=Path,
        default=_ROOT / "8_data" / "landsat",
        help="Directory containing Landsat band files",
    )
    p.add_argument("--synthetic", action="store_true",
                   help="Skip raster loading; use synthetic data only")
    p.add_argument("--no-delta", action="store_true",
                   help="Skip Delta Lake write (Postgres only)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print results without writing anywhere")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Ensure synthetic scene exists if no real data
    scene_dir: Path = args.scene_dir
    if not args.synthetic and not scene_dir.exists():
        logger.info(f"Scene dir {scene_dir} not found — generating synthetic scene...")
        scene_dir.mkdir(parents=True, exist_ok=True)
        # Import and run sample_landsat_loader fallback inline
        sys.path.insert(0, str(_ROOT / "8_data"))
        try:
            from sample_landsat_loader import _synthesize_scene  # type: ignore
            _synthesize_scene(scene_dir, size=256)
        except ImportError:
            logger.warning("sample_landsat_loader not importable; using pure synthetic path")
            args.synthetic = True

    records = run_pipeline(
        scene_dir=scene_dir,
        force_synthetic=args.synthetic,
        write_delta=not args.no_delta,
        dry_run=args.dry_run,
    )
    logger.success(f"Pipeline complete | {len(records)} ward records processed.")
