"""
Module 1g — Satellite Data Ingestion (Landsat 8/9 via GEE or fallback)
========================================================================

Fetches Landsat 8/9 L2 data for Bengaluru bbox (77.35,12.77,77.82,13.18)
using Google Earth Engine and computes:
  - NDVI  (Normalized Difference Vegetation Index)
  - NDBI  (Normalized Difference Built-up Index)
  - LST_celsius (Land Surface Temperature)
  - MNDWI (Modified Normalized Difference Water Index)

Saves results to Delta Lake at $DELTA_LAKE_PATH/satellite_features
Falls back to CSV if delta-lake is not installed.

Usage:
    python 1_data_pipeline/satellite_ingestion.py
    python 1_data_pipeline/satellite_ingestion.py --fallback    # synthetic data
    python 1_data_pipeline/satellite_ingestion.py --dry-run     # print only
    python 1_data_pipeline/satellite_ingestion.py --days 30     # last 30 days
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

# Load project .env when running the pipeline locally. Docker/Airflow can still
# pass these values as real environment variables.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception as exc:  # noqa: BLE001
    logger.debug(f".env load skipped: {exc}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BENGALURU_BBOX = (77.35, 12.77, 77.82, 13.18)   # (west, south, east, north)
DELTA_LAKE_PATH = os.getenv("DELTA_LAKE_PATH", "/tmp/ncg_delta_lake")
N_WARDS = 198

# 198 ward centroids generated on a grid over Bengaluru
def _generate_ward_centroids(n: int = N_WARDS) -> List[Dict]:
    west, south, east, north = BENGALURU_BBOX
    cols = 14
    rows = math.ceil(n / cols)
    centroids = []
    for idx in range(n):
        r, c = divmod(idx, cols)
        lat = south + (north - south) * (r / (rows - 1)) if rows > 1 else (south + north) / 2
        lon = west  + (east  - west)  * (c / (cols - 1)) if cols > 1 else (west  + east)  / 2
        centroids.append({"ward_id": idx + 1, "lat": round(lat, 5), "lon": round(lon, 5)})
    return centroids


WARD_CENTROIDS = _generate_ward_centroids()


# ---------------------------------------------------------------------------
# GEE Fetch
# ---------------------------------------------------------------------------

def _gee_fetch(days: int = 30) -> Optional[List[Dict]]:
    """
    Try to fetch Landsat 8/9 L2 data from Google Earth Engine.
    Returns list of ward feature dicts, or None on failure.
    """
    try:
        import ee  # type: ignore
    except ImportError:
        logger.warning("earthengine-api not installed — will use fallback")
        return None

    try:
        service_account = os.getenv("GEE_SERVICE_ACCOUNT")
        key_file = os.getenv("GEE_KEY_FILE")
        if service_account and key_file and Path(key_file).exists():
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(credentials)
            logger.info(f"GEE service account authenticated: {service_account}")
        else:
            ee.Initialize()
            logger.info("GEE authenticated with local/default credentials")
    except Exception as exc:
        logger.warning(f"GEE auth failed ({exc}) — will use fallback")
        return None

    try:
        end_date   = date.today()
        start_date = end_date - timedelta(days=days)
        bbox = ee.Geometry.Rectangle(list(BENGALURU_BBOX))

        # Combine Landsat 8 + 9 L2
        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterBounds(bbox)
            .filterDate(start_date.isoformat(), end_date.isoformat())
            .filter(ee.Filter.lt("CLOUD_COVER", 30))
            .sort("CLOUD_COVER")
        )

        count = collection.size().getInfo()
        if count == 0:
            logger.warning(f"No Landsat scenes found in last {days} days")
            return None

        logger.info(f"Found {count} Landsat scenes — using least cloudy")
        image = collection.first()

        # Scale factors for Collection 2 Level-2
        optical = image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]) \
                       .multiply(0.0000275).add(-0.2)
        thermal = image.select("ST_B10").multiply(0.00341802).add(149.0)

        b4 = optical.select("SR_B4")   # Red
        b5 = optical.select("SR_B5")   # NIR
        b6 = optical.select("SR_B6")   # SWIR-1
        b3 = optical.select("SR_B3")   # Green

        ndvi  = b5.subtract(b4).divide(b5.add(b4)).rename("NDVI")
        ndbi  = b6.subtract(b5).divide(b6.add(b5)).rename("NDBI")
        mndwi = b3.subtract(b6).divide(b3.add(b6)).rename("MNDWI")
        lst_c = thermal.subtract(273.15).rename("LST_celsius")

        indices = ndvi.addBands(ndbi).addBands(mndwi).addBands(lst_c)

        records = []
        today = date.today().isoformat()
        for ward in WARD_CENTROIDS:
            point = ee.Geometry.Point([ward["lon"], ward["lat"]])
            try:
                sample = indices.sample(region=point, scale=30, numPixels=1) \
                                .first().getInfo()
                props = sample.get("properties", {})
                records.append({
                    "ward_id":     ward["ward_id"],
                    "scene_date":  today,
                    "lat":         ward["lat"],
                    "lon":         ward["lon"],
                    "ndvi":        round(float(props.get("NDVI", 0.2)), 4),
                    "ndbi":        round(float(props.get("NDBI", 0.0)), 4),
                    "mndwi":       round(float(props.get("MNDWI", -0.1)), 4),
                    "lst_celsius": round(float(props.get("LST_celsius", 32.0)), 2),
                    "source":      "GEE-Landsat",
                })
            except Exception as ward_exc:
                logger.debug(f"GEE sample failed for ward {ward['ward_id']}: {ward_exc}")

        if records:
            logger.success(f"GEE fetch complete | {len(records)} wards")
        return records if records else None

    except Exception as exc:
        logger.error(f"GEE processing failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def _synthetic_features(n_wards: int = N_WARDS) -> List[Dict]:
    """
    Generate synthetic satellite features for 198 wards.
    Uses realistic NDVI/LST ranges for urban Bengaluru.
    """
    rng = np.random.default_rng(int(datetime.now().timestamp()) % 100000)
    cols = 14
    today = date.today().isoformat()
    records = []

    for idx in range(n_wards):
        r, c = divmod(idx, cols)
        # Urban heat island effect: CBD (7,7) is hottest
        cbd_dist  = math.sqrt((r - 7)**2 + (c - 7)**2)
        peenya    = math.exp(-((r - 11)**2 + (c - 2)**2) / 5.0)
        industrial = min(1.0, 0.3 * math.exp(-cbd_dist / 5) + 0.7 * peenya)

        ndvi  = float(np.clip(rng.uniform(0.05, 0.55) - 0.25 * industrial, 0.02, 0.70))
        ndbi  = float(np.clip(rng.uniform(-0.25, 0.45) + 0.2 * industrial, -0.5, 0.6))
        mndwi = float(np.clip(rng.uniform(-0.4, 0.1) - 0.05 * industrial, -0.6, 0.3))
        lst   = float(np.clip(rng.normal(34, 4) + 6 * industrial - 5 * ndvi, 22, 50))

        records.append({
            "ward_id":     idx + 1,
            "scene_date":  today,
            "lat":         WARD_CENTROIDS[idx]["lat"],
            "lon":         WARD_CENTROIDS[idx]["lon"],
            "ndvi":        round(ndvi, 4),
            "ndbi":        round(ndbi, 4),
            "mndwi":       round(mndwi, 4),
            "lst_celsius": round(lst, 2),
            "source":      "SYNTHETIC-FALLBACK",
        })
    return records


# ---------------------------------------------------------------------------
# Delta Lake writer
# ---------------------------------------------------------------------------

def _write_delta(records: List[Dict]) -> bool:
    """Write records to Delta Lake. Returns True on success."""
    try:
        from deltalake import DeltaTable, write_deltalake  # type: ignore
        import pandas as pd

        df = pd.DataFrame(records)
        delta_path = str(Path(DELTA_LAKE_PATH) / "satellite_features")
        Path(delta_path).mkdir(parents=True, exist_ok=True)

        write_deltalake(delta_path, df, mode="overwrite")
        logger.success(f"Wrote {len(records)} rows to Delta Lake at {delta_path}")
        return True
    except ImportError:
        logger.warning("deltalake not installed — falling back to CSV")
        return False
    except Exception as exc:
        logger.error(f"Delta Lake write failed: {exc}")
        return False


def _write_csv(records: List[Dict]) -> None:
    """CSV fallback writer."""
    try:
        import csv
        csv_path = Path(DELTA_LAKE_PATH) / "satellite_features.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        logger.success(f"CSV fallback: wrote {len(records)} rows to {csv_path}")
    except Exception as exc:
        logger.error(f"CSV write failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(fallback: bool = False, days: int = 30, dry_run: bool = False) -> List[Dict]:
    """
    Run satellite ingestion.

    Args:
        fallback: Skip GEE and use synthetic data directly.
        days:     Number of days to look back for Landsat scenes.
        dry_run:  Print results without writing.
    """
    records: Optional[List[Dict]] = None

    if not fallback:
        logger.info(f"Attempting GEE Landsat fetch (last {days} days)...")
        records = _gee_fetch(days=days)

    if records is None:
        logger.info("Using synthetic satellite features as fallback")
        records = _synthetic_features()

    logger.info(f"Prepared {len(records)} ward satellite records")

    if dry_run:
        for r in records[:5]:
            logger.info(f"  DRY-RUN {r}")
        logger.info(f"  ... {len(records) - 5} more records")
        return records

    success = _write_delta(records)
    if not success:
        _write_csv(records)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid — Satellite data ingestion (Landsat 8/9 via GEE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fallback", action="store_true",
                   help="Skip GEE; generate synthetic satellite features")
    p.add_argument("--days",    type=int, default=30,
                   help="Days to look back for Landsat scenes")
    p.add_argument("--dry-run", action="store_true",
                   help="Print results without writing to storage")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    records = run(fallback=args.fallback, days=args.days, dry_run=args.dry_run)
    logger.success(f"Satellite ingestion complete | {len(records)} wards processed")
