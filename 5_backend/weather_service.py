"""
Weather Service — Real Open-Meteo Integration
===============================================
Fetches live weather for all 198 BBMP wards using the Open-Meteo API.

• Free — no API key required
• Queries a 5×4 representative grid across Bengaluru in one HTTP call
• Each ward maps to its nearest grid point
• Results cached for 10 minutes (thread-safe)
• Falls back gracefully to None on any network/parse error

Usage:
    from weather_service import get_live_ward_weather
    df = get_live_ward_weather()   # → pd.DataFrame or None
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ── Ward catalogue ─────────────────────────────────────────────────────────
_DATA_DIR = str(Path(__file__).resolve().parent.parent / "8_data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

# ── Cache ──────────────────────────────────────────────────────────────────
_CACHE_TTL_SEC = 600          # refresh every 10 minutes
_cache_lock    = Lock()
_cache: Dict   = {"df": None, "fetched_at": 0.0, "status": "not_fetched"}
_satellite_cache: Dict = {"df": None, "fetched_at": 0.0, "status": "not_fetched"}


# ── Bengaluru representative weather grid (5 lat × 4 lon = 20 points) ─────
# Evenly covers the BBMP bounding box (12.84–13.13 lat, 77.46–77.77 lon)
_GRID_LATS = [12.840, 12.913, 12.986, 13.059, 13.130]
_GRID_LONS = [77.462, 77.563, 77.664, 77.768]


def _nearest_grid(lat: float, lon: float) -> Tuple[float, float]:
    """Return the (lat, lon) of the nearest grid point."""
    best = min(
        ((gl, go) for gl in _GRID_LATS for go in _GRID_LONS),
        key=lambda p: (p[0] - lat) ** 2 + (p[1] - lon) ** 2,
    )
    return best


def _fetch_open_meteo() -> Optional[Dict[Tuple[float, float], dict]]:
    """
    Fetch current weather + 24h rainfall for all 20 grid points in ONE request.
    Returns {(lat, lon): {temperature_c, rainfall_mm, humidity_pct}} or None.
    """
    lats = ",".join(str(x) for x in _GRID_LATS for _ in _GRID_LONS)
    lons = ",".join(str(x) for x in _GRID_LONS * len(_GRID_LATS))

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
        "&timezone=Asia/Kolkata&forecast_days=1"
    )

    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[weather_service] Open-Meteo fetch failed: {exc}", flush=True)
        return None

    # data is a list when multiple locations are queried
    if isinstance(data, dict):
        data = [data]

    result: Dict[Tuple[float, float], dict] = {}
    for i, loc in enumerate(data):
        try:
            gl = _GRID_LATS[i // len(_GRID_LONS)]
            go = _GRID_LONS[i %  len(_GRID_LONS)]
            cur = loc.get("current", {})
            daily = loc.get("daily", {})
            result[(gl, go)] = {
                "temperature_c":  cur.get("temperature_2m", 30.0),
                "humidity_pct":   cur.get("relative_humidity_2m", 60.0),
                "current_precip": cur.get("precipitation", 0.0),
                "rainfall_mm":    (daily.get("precipitation_sum") or [0.0])[0],
                "temp_max":       (daily.get("temperature_2m_max") or [32.0])[0],
                "temp_min":       (daily.get("temperature_2m_min") or [22.0])[0],
                "wind_speed":     cur.get("wind_speed_10m", 5.0),
            }
        except Exception:
            continue

    return result if result else None


def _read_satellite_features() -> Optional[pd.DataFrame]:
    """Load latest per-ward satellite features from Postgres or Delta Lake."""
    age = time.time() - _satellite_cache["fetched_at"]
    if _satellite_cache["df"] is not None and age < _CACHE_TTL_SEC:
        return _satellite_cache["df"]

    postgres_df = None

    # Prefer PostgreSQL when it contains real materialised features.
    try:
        import psycopg2  # type: ignore
        from config import get_settings

        settings = get_settings()
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout=2,
        )
        df = pd.read_sql(
            """
            SELECT ward_id, scene_date, ndvi, ndbi, impervious_pct,
                   lst_celsius, source_scene, updated_at
            FROM ward_satellite_features
            ORDER BY ward_id
            """,
            conn,
        )
        conn.close()
        if not df.empty:
            postgres_df = df
            sources = set(df.get("source_scene", pd.Series(dtype=str)).dropna().astype(str))
            has_real_source = any("GEE" in src or "Landsat" in src for src in sources)
            if has_real_source:
                _satellite_cache.update({"df": df, "fetched_at": time.time(), "status": "postgres"})
                return df
    except Exception:
        pass

    # Fall back to the direct GEE ingestion Delta output. This commonly has the
    # freshest real satellite data when satellite_ingestion.py was run directly.
    delta_roots = [
        os.getenv("DELTA_LAKE_PATH"),
        os.getenv("DELTA_LAKE_ROOT"),
        "/tmp/ncg_delta_lake",
        "/data/delta",
    ]
    for root in [r for r in delta_roots if r]:
        delta_path = Path(root) / "satellite_features"
        if not delta_path.exists():
            continue
        try:
            from deltalake import DeltaTable  # type: ignore

            df = DeltaTable(str(delta_path)).to_pandas()
            if not df.empty:
                if "source" in df.columns and "source_scene" not in df.columns:
                    df = df.rename(columns={"source": "source_scene"})
                keep = [
                    col for col in [
                        "ward_id", "scene_date", "ndvi", "ndbi", "impervious_pct",
                        "lst_celsius", "source_scene",
                    ]
                    if col in df.columns
                ]
                df = df[keep].sort_values("ward_id")
                sources = set(df.get("source_scene", pd.Series(dtype=str)).dropna().astype(str))
                has_real_source = any("GEE" in src or "Landsat" in src for src in sources)
                if has_real_source or postgres_df is None:
                    _satellite_cache.update({"df": df, "fetched_at": time.time(), "status": "delta"})
                    return df
        except Exception:
            continue

    if postgres_df is not None:
        _satellite_cache.update({"df": postgres_df, "fetched_at": time.time(), "status": "postgres_synthetic"})
        return postgres_df

    _satellite_cache.update({"df": None, "fetched_at": time.time(), "status": "unavailable"})
    return None


def _build_ward_df(grid_data: Dict[Tuple[float, float], dict]) -> pd.DataFrame:
    """
    Build a per-ward DataFrame by:
      1. Mapping each ward to its nearest grid point
      2. Adjusting temperature for Urban Heat Island effect
         (LST = air temp + UHI offset based on impervious surface %)
      3. Merging real GEE/Landsat NDVI, impervious %, and LST when available
    """
    try:
        from _common import synthetic_ward_catalogue
        catalogue = synthetic_ward_catalogue()
    except ImportError:
        return pd.DataFrame()

    try:
        from mock_ward_features import generate_ward_features
        static_df = generate_ward_features(seed=42)
        static_map = {
            row.ward_id: {
                "ndvi":           row.ndvi,
                "impervious_pct": row.impervious_pct,
                "pm25_ugm3":      row.pm25_ugm3,
                "population_density": row.population_density,
                "industrial_weight":  row.industrial_weight,
            }
            for _, row in static_df.iterrows()
        }
    except Exception:
        static_map = {}

    satellite_df = _read_satellite_features()
    satellite_map = {}
    if satellite_df is not None and not satellite_df.empty:
        satellite_map = {
            int(row.ward_id): row.to_dict()
            for _, row in satellite_df.iterrows()
        }

    records = []
    for ward in catalogue:
        ward_id  = ward["ward_id"]
        lat, lon = ward["centroid"][1], ward["centroid"][0]

        grid_pt  = _nearest_grid(lat, lon)
        weather  = grid_data.get(grid_pt, {"temperature_c": 28.0, "rainfall_mm": 0.0,
                                            "humidity_pct": 65.0, "current_precip": 0.0,
                                            "temp_max": 32.0, "temp_min": 22.0, "wind_speed": 5.0})

        static = static_map.get(ward_id, {})
        ndvi             = static.get("ndvi",               0.30)
        impervious_pct   = static.get("impervious_pct",     55.0)
        industrial_w     = static.get("industrial_weight",  0.2)
        pm25             = static.get("pm25_ugm3",          45.0)
        pop_density      = static.get("population_density", 14000.0)
        satellite = satellite_map.get(ward_id, {})
        satellite_source = satellite.get("source_scene") or satellite.get("source")
        if satellite:
            ndvi = satellite.get("ndvi", ndvi)
            impervious_pct = satellite.get("impervious_pct", impervious_pct)

        # ── Urban Heat Island offset ──────────────────────────────────────
        # Dense urban wards are 2–6°C hotter than air temp (well-documented
        # in Bengaluru literature). Impervious surface % drives the offset.
        uhi_offset  = 1.5 + (impervious_pct / 100.0) * 4.5 + industrial_w * 2.0
        lst_celsius = weather["temperature_c"] + uhi_offset - ndvi * 3.0
        if satellite.get("lst_celsius") is not None:
            lst_celsius = satellite["lst_celsius"]

        # ── Rainfall: use 24h sum, boost if currently raining ────────────
        rainfall = weather["rainfall_mm"]
        if weather["current_precip"] > 0:
            rainfall = max(rainfall, weather["current_precip"] * 4)  # extrapolate

        records.append({
            "ward_id":          ward_id,
            "ward_name":        ward["ward_name"],
            "centroid_lon":     lon,
            "centroid_lat":     lat,
            # Real weather
            "temperature_c":    round(weather["temperature_c"], 1),
            "humidity_pct":     round(weather["humidity_pct"], 1),
            "rainfall_mm":      round(rainfall, 2),
            "current_precip":   round(weather["current_precip"], 2),
            "wind_speed":       round(weather["wind_speed"], 1),
            "temp_max":         round(weather["temp_max"], 1),
            "temp_min":         round(weather["temp_min"], 1),
            # Derived
            "lst_celsius":      round(lst_celsius, 2),
            "ndvi":               round(ndvi, 3),
            "impervious_pct":     round(impervious_pct, 2),
            "pm25_ugm3":          round(pm25, 2),
            "population_density": round(pop_density, 1),
            "industrial_weight":  round(industrial_w, 3),
            "uhi_offset":         round(uhi_offset, 2),
            "satellite_source":    satellite_source or "synthetic_static",
            "satellite_scene_date": str(satellite.get("scene_date")) if satellite.get("scene_date") is not None else None,
            # Grid source
            "weather_grid_lat": grid_pt[0],
            "weather_grid_lon": grid_pt[1],
        })

    return pd.DataFrame(records)


def get_live_ward_weather(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame with live weather for all 198 wards.
    Cached for 10 minutes. Returns None if the API is unreachable.
    """
    global _cache

    with _cache_lock:
        age = time.time() - _cache["fetched_at"]
        if not force_refresh and _cache["df"] is not None and age < _CACHE_TTL_SEC:
            return _cache["df"]

        grid_data = _fetch_open_meteo()
        if grid_data is None:
            _cache["status"] = "api_error"
            return _cache["df"]     # return stale cache if available

        df = _build_ward_df(grid_data)
        if df.empty:
            _cache["status"] = "build_error"
            return _cache["df"]

        _cache["df"]         = df
        _cache["fetched_at"] = time.time()
        _cache["status"]     = "live"

        n_pts   = len(grid_data)
        avg_tmp = df["temperature_c"].mean()
        avg_rn  = df["rainfall_mm"].mean()
        print(
            f"[weather_service] Refreshed: {n_pts} grid points | "
            f"avg temp={avg_tmp:.1f}°C | avg rainfall={avg_rn:.1f}mm",
            flush=True,
        )
        return df


def get_cache_status() -> dict:
    """Return cache metadata for the /health endpoint."""
    with _cache_lock:
        age = time.time() - _cache["fetched_at"]
        return {
            "status":         _cache["status"],
            "age_sec":        round(age, 1),
            "cache_ttl_sec":  _CACHE_TTL_SEC,
            "wards_cached":   len(_cache["df"]) if _cache["df"] is not None else 0,
            "last_temp_avg":  round(_cache["df"]["temperature_c"].mean(), 1)
                              if _cache["df"] is not None else None,
            "last_rain_avg":  round(_cache["df"]["rainfall_mm"].mean(), 2)
                              if _cache["df"] is not None else None,
        }


# ── CLI quick test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = get_live_ward_weather(force_refresh=True)
    if df is not None:
        print(f"\nLive weather for {len(df)} wards:")
        print(df[["ward_id", "ward_name", "temperature_c", "lst_celsius",
                   "rainfall_mm", "humidity_pct"]].head(10).to_string(index=False))
        print(f"\nAvg air temp: {df.temperature_c.mean():.1f}°C")
        print(f"Avg LST:      {df.lst_celsius.mean():.1f}°C")
        print(f"Avg rainfall: {df.rainfall_mm.mean():.2f}mm")
        print(f"Max rainfall: {df.rainfall_mm.max():.2f}mm (Ward {df.loc[df.rainfall_mm.idxmax(), 'ward_name']})")
    else:
        print("Weather fetch failed")
