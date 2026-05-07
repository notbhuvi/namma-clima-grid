"""
Module 7b — Mock Ward Feature Matrix Generator
==============================================

Generates a synthetic feature matrix for all 198 BBMP wards using realistic
statistical distributions, and writes the result as both CSV and Parquet so
downstream modules (ST-GNN training, RL environment, API warm cache) can
consume it without needing live satellite or IoT data.

Feature schema (one row per ward):
    ward_id              : int         (1..198)
    ward_code            : str         ("BBMP-001")
    ward_name            : str
    centroid_lon         : float
    centroid_lat         : float
    lst_celsius          : float       Surface temp, Normal(35, 4), hotter for industrial wards
    ndvi                 : float       Uniform(0.1, 0.5), lower in dense urban cores
    impervious_pct       : float       Uniform(40, 90)
    rainfall_mm_24h      : float       Gamma-distributed monsoon proxy
    flood_reports_7d     : int         Poisson(lambda=2) — boosted in low-NDVI wards
    pm25_ugm3            : float       LogNormal — higher near industrial ring
    population_density   : float       people / km^2
    area_km2             : float       ward area
    heat_stress_score    : float       derived [0..100]
    flood_risk_score     : float       derived [0..100]

Usage:
    python 8_data/mock_ward_features.py                       # default output
    python 8_data/mock_ward_features.py --seed 7 --monsoon    # monsoon mode
    python 8_data/mock_ward_features.py --out ./my_features
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from _common import (
    BENGALURU_BBOX,
    MODULE_DIR,
    NUM_WARDS,
    logger,
    synthetic_ward_catalogue,
)


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------
def _industrial_weight(row: int, col: int) -> float:
    """
    Return a 0..1 weight marking industrial/commercial hot zones.
    Two hot "belts" are modelled: Peenya (NW) and Electronic City (SE),
    plus the CBD core.
    """
    # CBD (near grid center)
    cbd = math.exp(-(((row - 7) ** 2 + (col - 7) ** 2) / 6))
    # Peenya NW belt
    peenya = math.exp(-(((row - 11) ** 2 + (col - 2) ** 2) / 5))
    # Electronic City / Bommanahalli SE
    ecity = math.exp(-(((row - 1) ** 2 + (col - 11) ** 2) / 5))
    return float(min(1.0, 0.6 * cbd + 0.8 * peenya + 0.7 * ecity))


def _ward_area_km2(row: int, col: int, cols: int, rows: int) -> float:
    """Approximate area of one grid cell of the Bengaluru bbox in km^2."""
    lat_km = 111.32
    lon_km = 111.32 * math.cos(math.radians((BENGALURU_BBOX.south + BENGALURU_BBOX.north) / 2))
    dy_deg = (BENGALURU_BBOX.north - BENGALURU_BBOX.south) / rows
    dx_deg = (BENGALURU_BBOX.east - BENGALURU_BBOX.west) / cols
    return round((dy_deg * lat_km) * (dx_deg * lon_km), 3)


def generate_ward_features(seed: int = 42, monsoon: bool = False) -> pd.DataFrame:
    """
    Build a deterministic synthetic feature matrix for all 198 wards.

    Args:
        seed: numpy RNG seed for reproducibility.
        monsoon: if True, rainfall + flood reports are scaled up.

    Returns:
        pandas.DataFrame with one row per ward.
    """
    rng = np.random.default_rng(seed)
    catalogue = synthetic_ward_catalogue()

    cols = 14
    rows = math.ceil(NUM_WARDS / cols)

    records: List[dict] = []
    for rec in catalogue:
        r = int(rec["grid_row"])
        c = int(rec["grid_col"])
        industrial = _industrial_weight(r, c)

        # --- Impervious % — higher in industrial zones
        impervious_pct = float(np.clip(
            rng.uniform(40, 90) + 10 * industrial,
            40.0, 98.0,
        ))

        # --- NDVI — inversely correlated with imperviousness
        ndvi = float(np.clip(
            rng.uniform(0.1, 0.5) - 0.25 * industrial + rng.normal(0, 0.03),
            0.02, 0.75,
        ))

        # --- LST — industrial + low-NDVI wards run hot
        lst_celsius = float(np.clip(
            rng.normal(35.0, 4.0) + 5.0 * industrial - 6.0 * ndvi,
            22.0, 48.0,
        ))

        # --- Rainfall (24h) — Gamma, scaled in monsoon mode
        shape = 2.0
        scale = 4.0 if monsoon else 1.0
        rainfall_mm_24h = float(rng.gamma(shape, scale))

        # --- Flood reports (7d) — Poisson, more likely in high-impervious wards
        lam = 2.0 + (impervious_pct - 60.0) / 15.0
        if monsoon:
            lam *= 3.0
        flood_reports_7d = int(rng.poisson(max(0.1, lam)))

        # --- PM2.5 — LogNormal, boosted industrially
        pm25 = float(np.clip(
            rng.lognormal(mean=3.4 + 0.5 * industrial, sigma=0.35),
            8.0, 350.0,
        ))

        # --- Population density + area
        area_km2 = _ward_area_km2(r, c, cols, rows)
        pop_density = float(np.clip(
            rng.normal(14000, 5000) + 4000 * (1 - industrial),
            2000, 45000,
        ))

        # --- Derived stress scores (0..100)
        #    Heat stress is driven by LST + impervious - NDVI
        heat_raw = 0.55 * (lst_celsius - 25) / 20 + 0.30 * (impervious_pct / 100) + 0.15 * (1 - ndvi)
        heat_stress_score = float(np.clip(heat_raw * 100, 0, 100))

        #    Flood risk is driven by rainfall + impervious - NDVI (vegetation retention)
        rain_norm = min(1.0, rainfall_mm_24h / 60.0)
        flood_raw = 0.55 * rain_norm + 0.30 * (impervious_pct / 100) + 0.15 * (1 - ndvi)
        flood_risk_score = float(np.clip(flood_raw * 100, 0, 100))

        records.append({
            "ward_id": rec["ward_id"],
            "ward_code": rec["ward_code"],
            "ward_name": rec["ward_name"],
            "centroid_lon": rec["centroid"][0],
            "centroid_lat": rec["centroid"][1],
            "lst_celsius": round(lst_celsius, 2),
            "ndvi": round(ndvi, 3),
            "impervious_pct": round(impervious_pct, 2),
            "rainfall_mm_24h": round(rainfall_mm_24h, 2),
            "flood_reports_7d": flood_reports_7d,
            "pm25_ugm3": round(pm25, 2),
            "population_density": round(pop_density, 1),
            "area_km2": area_km2,
            "heat_stress_score": round(heat_stress_score, 2),
            "flood_risk_score": round(flood_risk_score, 2),
            "industrial_weight": round(industrial, 3),
        })

    df = pd.DataFrame.from_records(records)
    logger.info(
        f"Generated {len(df)} ward rows | "
        f"LST μ={df.lst_celsius.mean():.1f}°C | "
        f"NDVI μ={df.ndvi.mean():.2f} | "
        f"HeatStress μ={df.heat_stress_score.mean():.1f} | "
        f"FloodRisk μ={df.flood_risk_score.mean():.1f}"
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate mock BBMP ward feature matrix.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--monsoon", action="store_true", help="Scale up rainfall + flood reports")
    p.add_argument(
        "--out",
        type=Path,
        default=MODULE_DIR,
        help="Output directory (default: 8_data/)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = generate_ward_features(seed=args.seed, monsoon=args.monsoon)

    csv_path = args.out / "ward_features_mock.csv"
    parquet_path = args.out / "ward_features_mock.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    logger.success(f"Wrote {csv_path}")
    logger.success(f"Wrote {parquet_path}")
    logger.info("Top 5 most heat-stressed wards:")
    for _, row in df.nlargest(5, "heat_stress_score").iterrows():
        logger.info(
            f"  #{row.ward_id:3d} {row.ward_name:<25s} "
            f"heat={row.heat_stress_score:5.1f}  flood={row.flood_risk_score:5.1f}"
        )


if __name__ == "__main__":
    main()
