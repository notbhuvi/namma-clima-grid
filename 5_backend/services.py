"""
Module 5 — Service Layer
==========================

Orchestrates calls to the ML inference wrappers (Modules 2-4) and exposes
a clean async interface for the FastAPI routes.

Design principles:
  * Models are loaded once at startup via init_services().
  * All heavy inference runs in thread-pool executors so the async event
    loop is never blocked.
  * A synthetic fallback is always available so the API starts even when
    no trained checkpoints exist.
"""
from __future__ import annotations

import asyncio
import math
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

_MODULE_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent

# Make Modules 2-4 importable
for _mod in ["2_thermal_vision", "3_st_gnn", "4_rl_optimizer", "8_data"]:
    _path = str(_PROJECT_ROOT / _mod)
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ---------------------------------------------------------------------------
# Ward catalogue (shared across services)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ward_catalogue() -> Dict[int, str]:
    """Return {ward_id: ward_name} for all 198 wards."""
    try:
        from _common import synthetic_ward_catalogue  # type: ignore
        return {w["ward_id"]: w["ward_name"] for w in synthetic_ward_catalogue()}
    except ImportError:
        return {i: f"Ward-{i}" for i in range(1, 199)}


# ---------------------------------------------------------------------------
# Synthetic fallback data
# ---------------------------------------------------------------------------

def _synthetic_ward_risk(n: int = 198, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic risk DataFrame when models are unavailable."""
    rng = np.random.default_rng(seed)
    names = _ward_catalogue()
    cols = 14
    records = []
    for i in range(n):
        r, c = divmod(i, cols)
        cbd  = math.exp(-((r - 7)**2 + (c - 7)**2) / 8)
        heat  = float(np.clip(rng.normal(45, 15) + 15 * cbd, 0, 100))
        flood = float(np.clip(rng.normal(30, 12) + 10 * rng.random(), 0, 100))
        combined = 0.6 * heat + 0.4 * flood
        if combined >= 75:
            level = "critical"
        elif combined >= 50:
            level = "high"
        elif combined >= 25:
            level = "medium"
        else:
            level = "low"
        records.append({
            "ward_id":           i + 1,
            "ward_name":         names.get(i + 1, f"Ward-{i+1}"),
            "heat_stress_score": round(heat, 2),
            "flood_risk_score":  round(flood, 2),
            "risk_level":        level,
            "lst_pred_celsius":  round(float(rng.normal(35, 4) + 4 * cbd), 2),
            "ndvi":              round(float(np.clip(0.4 - 0.25 * cbd + rng.uniform(-0.05, 0.05), 0, 1)), 3),
            "impervious_pct":    round(float(np.clip(40 + 40 * cbd + rng.uniform(-5, 5), 0, 100)), 1),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Service state (module-level singletons)
# ---------------------------------------------------------------------------

class _State:
    thermal_vision = None   # ThermalVisionInference
    st_gnn         = None   # STGNNInference
    rl_optimizer   = None   # GreenInfraOptimizer
    started_at:    float = 0.0
    models_loaded: Dict[str, bool] = {}


_state = _State()


# ---------------------------------------------------------------------------
# Init / shutdown
# ---------------------------------------------------------------------------

def init_services() -> None:
    """
    Load all ML models.  Called once during FastAPI lifespan startup.
    Each model uses its own fallback chain (checkpoint → MLflow → untrained).
    """
    _state.started_at = time.time()

    # Module 2 — Thermal Vision
    try:
        _load_thermal_module()
        _state.models_loaded["thermal_vision"] = True
    except Exception as exc:
        logger.warning(f"Thermal Vision load failed: {exc}")
        _state.models_loaded["thermal_vision"] = False

    # Module 3 — ST-GNN
    try:
        _load_gnn_module()
        _state.models_loaded["st_gnn"] = True
    except Exception as exc:
        logger.warning(f"ST-GNN load failed: {exc}")
        _state.models_loaded["st_gnn"] = False

    # Module 4 — RL Optimizer
    try:
        _load_rl_module()
        _state.models_loaded["rl_optimizer"] = True
    except Exception as exc:
        logger.warning(f"RL Optimizer load failed: {exc}")
        _state.models_loaded["rl_optimizer"] = False

    logger.info(f"Services initialised | {_state.models_loaded}")


# Flat module names that each inference file imports — must be flushed
# from sys.modules before loading so each module gets its own copy.
_SHARED_MODULE_NAMES = [
    "model", "dataset", "inference", "graph_builder",
    "environment", "_common", "train",
]


def _isolated_load(module_dir: str, inference_file: str, module_alias: str) -> object:
    """
    Load an inference module from an explicit file path with full sys.path
    and sys.modules isolation so that name-colliding flat imports (model,
    dataset, etc.) resolve against the correct module directory.
    """
    import importlib.util, sys
    data_dir = str(_PROJECT_ROOT / "8_data")

    # 1. Save state
    _orig_path    = sys.path[:]
    _saved_mods   = {k: sys.modules.pop(k) for k in list(sys.modules) if k in _SHARED_MODULE_NAMES}

    # 2. Put the right dirs at the front
    sys.path[:] = [module_dir, data_dir] + [p for p in _orig_path if p not in (module_dir, data_dir)]

    try:
        spec = importlib.util.spec_from_file_location(module_alias, inference_file)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[module_alias] = mod
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    finally:
        # 3. Restore path and evict any newly cached flat names so next
        #    loader starts clean, but keep the aliased module itself.
        sys.path[:] = _orig_path
        for k in _SHARED_MODULE_NAMES:
            sys.modules.pop(k, None)
        sys.modules.update(_saved_mods)


def _load_thermal_module() -> None:
    mod = _isolated_load(
        module_dir     = str(_PROJECT_ROOT / "2_thermal_vision"),
        inference_file = str(_PROJECT_ROOT / "2_thermal_vision" / "inference.py"),
        module_alias   = "thermal_inference",
    )
    _state.thermal_vision = mod.load_default_inference()  # type: ignore


def _load_gnn_module() -> None:
    mod = _isolated_load(
        module_dir     = str(_PROJECT_ROOT / "3_st_gnn"),
        inference_file = str(_PROJECT_ROOT / "3_st_gnn" / "inference.py"),
        module_alias   = "stgnn_inference",
    )
    _state.st_gnn = mod.load_default_inference()  # type: ignore


def _load_rl_module() -> None:
    mod = _isolated_load(
        module_dir     = str(_PROJECT_ROOT / "4_rl_optimizer"),
        inference_file = str(_PROJECT_ROOT / "4_rl_optimizer" / "inference.py"),
        module_alias   = "rl_inference",
    )
    _state.rl_optimizer = mod.load_default_optimizer()  # type: ignore


def shutdown_services() -> None:
    """Cleanup on application shutdown."""
    logger.info("Services shutting down")
    _state.thermal_vision = None
    _state.st_gnn         = None
    _state.rl_optimizer   = None


# ---------------------------------------------------------------------------
# Ward risk (synchronous — called via run_in_executor)
# ---------------------------------------------------------------------------

def _read_ward_risks_from_db(ward_ids: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
    """
    Read the latest ward risk scores from PostgreSQL ward_risk_scores table.
    Returns None if the table is empty or unavailable.
    """
    try:
        import os
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            dbname=os.getenv("POSTGRES_DB", "namma_clima_grid"),
            user=os.getenv("POSTGRES_USER", "ncg_user"),
            password=os.getenv("POSTGRES_PASSWORD", "change_me_in_production"),
        )
        where = ""
        params: list = []
        if ward_ids:
            placeholders = ",".join(["%s"] * len(ward_ids))
            where = f"WHERE r.ward_id IN ({placeholders})"
            params = list(ward_ids)

        sql = f"""
            SELECT
                r.ward_id,
                COALESCE(w.ward_name, 'Ward-' || r.ward_id::text) AS ward_name,
                r.heat_stress_score,
                r.flood_risk_score,
                r.risk_level,
                r.lst_pred_celsius,
                r.ndvi,
                r.impervious_pct,
                r.updated_at
            FROM ward_risk_scores r
            LEFT JOIN wards w ON w.ward_id = r.ward_id
            {where}
            ORDER BY r.ward_id
        """
        df = pd.read_sql(sql, conn, params=params)
        conn.close()
        if df.empty:
            return None
        logger.debug(f"Loaded {len(df)} ward risk rows from PostgreSQL (updated: {df['updated_at'].max()})")
        return df
    except Exception as exc:
        logger.debug(f"DB read skipped ({exc})")
        return None


def _weather_features_to_risk(ward_ids: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
    """
    Build a live feature DataFrame using real Open-Meteo weather data.
    Falls back to PostgreSQL weather_readings if available, otherwise
    uses the Open-Meteo API directly (no key required).
    Returns None if all sources fail.
    """
    # ── Try Open-Meteo directly (primary live source) ────────────────────
    try:
        from weather_service import get_live_ward_weather
        df = get_live_ward_weather()
        if df is not None and not df.empty:
            # Only rename rainfall — GNN needs lst_celsius (not lst_pred_celsius)
            df = df.rename(columns={"rainfall_mm": "rainfall_mm_24h"})
            df["data_source"] = "live_weather"
            if ward_ids:
                df = df[df["ward_id"].isin(ward_ids)]
            logger.info(
                f"Live Open-Meteo weather loaded for {len(df)} wards | "
                f"avg_temp={df['temperature_c'].mean():.1f}°C | "
                f"avg_rain={df['rainfall_mm_24h'].mean():.2f}mm"
            )
            return df
    except Exception as exc:
        logger.debug(f"Open-Meteo weather fetch skipped: {exc}")

    # ── Fallback: try PostgreSQL weather_readings ────────────────────────
    try:
        import os, psycopg2
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            dbname=os.getenv("POSTGRES_DB", "namma_clima_grid"),
            user=os.getenv("POSTGRES_USER", "ncg_user"),
            password=os.getenv("POSTGRES_PASSWORD", "change_me_in_production"),
            connect_timeout=2,
        )
        weather_sql = """
            SELECT DISTINCT ON (ward_id)
                ward_id,
                temperature_2m   AS temperature_c,
                humidity_pct,
                precipitation_mm AS rainfall_mm_24h,
                timestamp        AS obs_time
            FROM weather_readings
            ORDER BY ward_id, timestamp DESC
        """
        scores_sql = "SELECT ward_id, lst_pred_celsius, ndvi, impervious_pct FROM ward_risk_scores"
        wdf = pd.read_sql(weather_sql, conn)
        sdf = pd.read_sql(scores_sql, conn)
        conn.close()

        if wdf.empty:
            return None

        merged = wdf.merge(sdf, on="ward_id", how="left")
        merged["lst_pred_celsius"] = merged["lst_pred_celsius"].fillna(merged["temperature_c"])
        merged["ndvi"]             = merged["ndvi"].fillna(0.3)
        merged["impervious_pct"]   = merged["impervious_pct"].fillna(40.0)
        merged["pm25"]             = 35.0

        if ward_ids:
            merged = merged[merged["ward_id"].isin(ward_ids)]

        logger.debug(f"Built live weather features from PostgreSQL for {len(merged)} wards")
        return merged
    except Exception as exc:
        logger.debug(f"PostgreSQL weather feature build skipped ({exc})")
        return None


def get_ward_risks_sync(ward_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Return a DataFrame of ward risk scores.

    Priority order (most-live → most-synthetic):
      1. PostgreSQL ward_risk_scores  — written by pipeline / Airflow DAG
      2. ST-GNN on live weather       — real Open-Meteo observations as features
      3. ST-GNN on synthetic features — model-generated without real weather
      4. Deterministic synthetic      — seed=42 fallback, always available
    """
    names = _ward_catalogue()

    # ── 1. Try PostgreSQL (most live: updated by pipeline / Airflow DAG) ────
    db_df = _read_ward_risks_from_db(ward_ids)
    if db_df is not None and not db_df.empty:
        if "ward_name" not in db_df.columns:
            db_df["ward_name"] = db_df["ward_id"].map(names)
        return db_df

    # ── 2. ST-GNN on live weather readings ───────────────────────────────────
    if _state.st_gnn is not None:
        live_features = _weather_features_to_risk(ward_ids)
        if live_features is not None:
            try:
                result = _state.st_gnn.predict_snapshot(live_features)
                result["ward_name"]  = result["ward_id"].map(names)
                result["data_source"] = "live_weather"
                # Attach real temperature/rainfall for the API response
                weather_cols = live_features[["ward_id", "temperature_c",
                                              "rainfall_mm_24h", "humidity_pct"]].copy()
                result = result.merge(weather_cols, on="ward_id", how="left")
                if ward_ids:
                    result = result[result["ward_id"].isin(ward_ids)]
                logger.info("Serving ST-GNN predictions on live Open-Meteo weather features")
                return result
            except Exception as exc:
                logger.warning(f"GNN on live features failed: {exc}")

    # ── 3. ST-GNN on synthetic features ─────────────────────────────────────
    if _state.st_gnn is not None:
        try:
            features_df = _synthetic_ward_risk()
            result = _state.st_gnn.predict_snapshot(features_df)
            result["ward_name"]   = result["ward_id"].map(names)
            result["data_source"] = "model_generated"
            if ward_ids:
                result = result[result["ward_id"].isin(ward_ids)]
            logger.info("Serving ST-GNN predictions on synthetic features")
            return result
        except Exception as exc:
            logger.warning(f"GNN prediction failed: {exc}")

    # ── 4. Deterministic synthetic fallback ──────────────────────────────────
    logger.info("Serving synthetic fallback ward risk data (no DB / model available)")
    df = _synthetic_ward_risk()
    df["ward_name"]   = df["ward_id"].map(names)
    df["data_source"] = "model_generated"
    if ward_ids:
        df = df[df["ward_id"].isin(ward_ids)]
    return df


async def get_ward_risks(ward_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Async wrapper — runs sync inference in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_ward_risks_sync, ward_ids)


# ---------------------------------------------------------------------------
# Forecast (synchronous)
# ---------------------------------------------------------------------------

def get_forecasts_sync(
    ward_ids: Optional[List[int]] = None,
    horizon_hrs: int = 6,
) -> List[dict]:
    """
    Generate per-ward hourly forecasts.

    If the GNN is available, the forecast is a simple perturbation of the
    current prediction (placeholder for a proper temporal forecasting model).
    """
    current = get_ward_risks_sync(ward_ids)
    rng = np.random.default_rng(int(time.time()) % 10000)
    forecasts = []

    for _, row in current.iterrows():
        for h in range(1, horizon_hrs + 1):
            # Simple diurnal pattern + noise
            diurnal = 3.0 * math.sin(math.pi * (h % 24 - 6) / 12)
            heat = float(np.clip(
                row["heat_stress_score"] + diurnal + rng.normal(0, 2), 0, 100
            ))
            flood = float(np.clip(
                row["flood_risk_score"] + rng.normal(0, 3), 0, 100
            ))
            conf = max(0.3, 1.0 - h * 0.08)
            forecasts.append({
                "ward_id":               int(row["ward_id"]),
                "hour":                  h,
                "heat_stress_predicted": round(heat, 2),
                "flood_risk_predicted":  round(flood, 2),
                "confidence":            round(conf, 2),
            })
    return forecasts


async def get_forecasts(
    ward_ids: Optional[List[int]] = None,
    horizon_hrs: int = 6,
) -> List[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_forecasts_sync, ward_ids, horizon_hrs)


# ---------------------------------------------------------------------------
# Interventions (synchronous)
# ---------------------------------------------------------------------------

def get_recommendations_sync(
    budget: int = 60,
    top_k: int = 10,
    ward_ids: Optional[List[int]] = None,
) -> List[dict]:
    """
    Get RL intervention recommendations.

    Falls back to a heuristic ranking if the RL model is unavailable.
    """
    current = get_ward_risks_sync()

    if _state.rl_optimizer is not None:
        try:
            recs = _state.rl_optimizer.recommend_from_dataframe(
                current, budget=budget, top_k=top_k
            )
            names = _ward_catalogue()
            result = []
            rl_ward_ids = set()
            for r in recs:
                result.append({
                    "priority_rank":            r.priority_rank,
                    "ward_id":                  r.ward_id,
                    "ward_name":                names.get(r.ward_id),
                    "intervention_type":        r.intervention_type,
                    "expected_heat_reduction":  r.expected_heat_reduction,
                    "expected_flood_reduction": r.expected_flood_reduction,
                    "cost":                     r.cost,
                    "ward_heat_stress":         r.ward_heat_stress,
                    "ward_flood_risk":          r.ward_flood_risk,
                })
                rl_ward_ids.add(r.ward_id)

            # If RL returned fewer than top_k, fill remaining with heuristics
            if len(result) < top_k:
                spent = sum(r["cost"] for r in result)
                remaining_budget = budget - spent
                remaining_k      = top_k - len(result)
                heuristic = _heuristic_recommendations(
                    current[~current["ward_id"].isin(rl_ward_ids)],
                    budget=remaining_budget,
                    top_k=remaining_k,
                )
                # Re-rank sequentially
                for h in heuristic:
                    h["priority_rank"] = len(result) + 1
                    result.append(h)

            return result
        except Exception as exc:
            logger.warning(f"RL recommendation failed: {exc}")

    # Heuristic fallback: rank by combined risk, assign best interventions
    return _heuristic_recommendations(current, budget, top_k)


def _heuristic_recommendations(
    df: pd.DataFrame, budget: int, top_k: int
) -> List[dict]:
    """Simple rule-based fallback when RL model is unavailable."""
    intervention_map = [
        {"name": "tree_planting",      "heat_red": 9.0,  "flood_red": 0.0,  "cost": 2},
        {"name": "cool_pavement",      "heat_red": 10.0, "flood_red": 0.0,  "cost": 2},
        {"name": "green_roof",         "heat_red": 5.6,  "flood_red": 0.0,  "cost": 3},
        {"name": "permeable_pavement", "heat_red": 0.0,  "flood_red": 15.0, "cost": 4},
        {"name": "urban_wetland",      "heat_red": 3.0,  "flood_red": 25.0, "cost": 6},
    ]
    names = _ward_catalogue()

    df = df.copy()
    df["combined"] = 0.6 * df["heat_stress_score"] + 0.4 * df["flood_risk_score"]
    df = df.sort_values("combined", ascending=False)

    results = []
    spent = 0
    rank  = 1

    for _, row in df.iterrows():
        if rank > top_k or spent >= budget:
            break
        h = float(row["heat_stress_score"])
        f = float(row["flood_risk_score"])

        # Pick best-fit intervention by risk profile
        if h >= 70 and f < 40:
            iv = intervention_map[1]   # cool_pavement — extreme heat, low flood
        elif h >= 60:
            iv = intervention_map[0]   # tree_planting — high heat
        elif f >= 55:
            iv = intervention_map[4]   # urban_wetland — high flood
        elif f >= 40:
            iv = intervention_map[3]   # permeable_pavement — moderate flood
        else:
            iv = intervention_map[2]   # green_roof — moderate heat

        if spent + iv["cost"] > budget:
            # Try cheaper options
            for fallback in sorted(intervention_map, key=lambda x: x["cost"]):
                if spent + fallback["cost"] <= budget:
                    iv = fallback
                    break
            else:
                break

        results.append({
            "priority_rank":            rank,
            "ward_id":                  int(row["ward_id"]),
            "ward_name":                names.get(int(row["ward_id"])),
            "intervention_type":        iv["name"],
            "expected_heat_reduction":  iv["heat_red"],
            "expected_flood_reduction": iv["flood_red"],
            "cost":                     iv["cost"],
            "ward_heat_stress":         round(float(row["heat_stress_score"]), 1),
            "ward_flood_risk":          round(float(row["flood_risk_score"]), 1),
        })
        spent += iv["cost"]
        rank  += 1

    return results


async def get_recommendations(
    budget: int = 60, top_k: int = 10, ward_ids: Optional[List[int]] = None
) -> List[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, get_recommendations_sync, budget, top_k, ward_ids
    )


# ---------------------------------------------------------------------------
# Alert generation (for WebSocket)
# ---------------------------------------------------------------------------

def generate_alerts_sync() -> List[dict]:
    """Scan current ward risks and generate alerts for critical wards."""
    df = get_ward_risks_sync()
    names = _ward_catalogue()
    alerts = []
    now = datetime.now(timezone.utc)

    for _, row in df.iterrows():
        h = float(row["heat_stress_score"])
        f = float(row["flood_risk_score"])
        wid = int(row["ward_id"])
        wname = names.get(wid, f"Ward-{wid}")

        if h >= 75:
            alerts.append({
                "ward_id":    wid,
                "alert_type": "heat_wave",
                "severity":   "critical",
                "message":    f"Extreme heat stress in {wname}: {h:.0f}/100",
                "value":      round(h, 1),
                "threshold":  75.0,
                "timestamp":  now.isoformat(),
            })
        elif h >= 60:
            alerts.append({
                "ward_id":    wid,
                "alert_type": "heat_wave",
                "severity":   "high",
                "message":    f"High heat stress in {wname}: {h:.0f}/100",
                "value":      round(h, 1),
                "threshold":  60.0,
                "timestamp":  now.isoformat(),
            })

        if f >= 70:
            alerts.append({
                "ward_id":    wid,
                "alert_type": "flood_warning",
                "severity":   "critical",
                "message":    f"Flood risk critical in {wname}: {f:.0f}/100",
                "value":      round(f, 1),
                "threshold":  70.0,
                "timestamp":  now.isoformat(),
            })
        elif f >= 50:
            alerts.append({
                "ward_id":    wid,
                "alert_type": "flood_warning",
                "severity":   "high",
                "message":    f"Elevated flood risk in {wname}: {f:.0f}/100",
                "value":      round(f, 1),
                "threshold":  50.0,
                "timestamp":  now.isoformat(),
            })

    return alerts


async def generate_alerts() -> List[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_alerts_sync)


# ---------------------------------------------------------------------------
# Cached helpers (used by bbmp and research routes)
# ---------------------------------------------------------------------------

async def get_ward_risks_cached() -> List[dict]:
    """
    Returns all ward risks as a list of dicts (suitable for JSON responses).
    Wraps the existing get_ward_risks() DataFrame returning function.
    """
    df = await get_ward_risks()
    return df.to_dict(orient="records")


async def get_interventions_cached() -> List[dict]:
    """
    Returns intervention recommendations as a list of dicts.
    Wraps the existing get_recommendations() function with default budget.
    """
    return await get_recommendations(budget=500, top_k=198)


# ---------------------------------------------------------------------------
# Health metadata
# ---------------------------------------------------------------------------

def get_health() -> dict:
    return {
        "models_loaded": _state.models_loaded,
        "uptime_sec":    round(time.time() - _state.started_at, 1),
    }
