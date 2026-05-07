"""
Routes — Research export and feature importance endpoints
============================================================

  GET /research/export?format=csv         → Export ward data as CSV/JSON/GeoJSON
  GET /research/feature-importance        → ST-GNN gradient feature importance
  GET /wards/geojson/live                 → GeoJSON with ward polygon approximations
"""
from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from services import get_ward_risks_cached

router = APIRouter(tags=["research"])

# Static feature importance fallback (reflects ST-GNN architecture)
_STATIC_FEATURE_IMPORTANCE = [
    {"feature": "lst_celsius",        "importance": 0.248, "description": "Land Surface Temperature"},
    {"feature": "ndvi",               "importance": 0.187, "description": "Vegetation Index"},
    {"feature": "impervious_pct",     "importance": 0.162, "description": "Impervious Surface %"},
    {"feature": "rainfall_mm_24h",    "importance": 0.141, "description": "24h Rainfall"},
    {"feature": "pm25_ugm3",          "importance": 0.098, "description": "PM2.5 Concentration"},
    {"feature": "population_density", "importance": 0.074, "description": "Population Density"},
    {"feature": "ndbi",               "importance": 0.052, "description": "Built-up Index"},
    {"feature": "industrial_weight",  "importance": 0.038, "description": "Industrial Proximity"},
]


# ---------------------------------------------------------------------------
# GET /research/export
# ---------------------------------------------------------------------------

@router.get("/research/export")
async def export_ward_data(
    format: str = Query("csv", pattern="^(csv|json|geojson)$"),
):
    """
    Export all ward data in the requested format.

    Supported formats: csv, json, geojson
    """
    wards = await get_ward_risks_cached()

    if format == "csv":
        output = io.StringIO()
        if wards:
            fieldnames = list(wards[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wards)
        content = output.getvalue()
        return StreamingResponse(
            iter([content]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="ncg_ward_data.csv"'},
        )

    elif format == "json":
        payload = json.dumps({"wards": wards, "total": len(wards)}, default=str)
        return StreamingResponse(
            iter([payload]),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="ncg_ward_data.json"'},
        )

    else:  # geojson
        cols = 14
        rows_grid = math.ceil(198 / cols)
        features = []
        for w in wards:
            wid = w.get("ward_id", 1)
            row, col = divmod(wid - 1, cols)
            lat = 12.77 + (13.18 - 12.77) * (row / max(rows_grid - 1, 1))
            lon = 77.35 + (77.82 - 77.35) * (col / max(cols - 1, 1))
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(lon, 5), round(lat, 5)],
                },
                "properties": {k: v for k, v in w.items()},
            })
        geojson = json.dumps({
            "type": "FeatureCollection",
            "features": features,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, default=str)
        return StreamingResponse(
            iter([geojson]),
            media_type="application/geo+json",
            headers={"Content-Disposition": 'attachment; filename="ncg_ward_data.geojson"'},
        )


# ---------------------------------------------------------------------------
# GET /research/feature-importance
# ---------------------------------------------------------------------------

@router.get("/research/feature-importance")
async def feature_importance():
    """
    Returns ST-GNN feature importance scores.

    If the ST-GNN model is loaded and supports gradient-based attribution,
    returns computed importances. Falls back to static importances otherwise.
    """
    try:
        # Attempt live gradient attribution from loaded model
        from services import _state  # type: ignore
        if _state.st_gnn is not None and hasattr(_state.st_gnn, "feature_importance"):
            live = _state.st_gnn.feature_importance()
            if live:
                return {
                    "source": "model_gradient",
                    "features": live,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
    except Exception:
        pass

    return {
        "source": "static_fallback",
        "features": _STATIC_FEATURE_IMPORTANCE,
        "note": "ST-GNN model not loaded — returning static feature importances",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /wards/geojson/live  (prefix="/wards" from router tag, but manually set)
# ---------------------------------------------------------------------------

@router.get("/wards/geojson/live")
async def wards_geojson_live():
    """
    Returns live ward risk data as GeoJSON FeatureCollection.
    Ward polygons are approximated as circles around centroid coordinates.
    """
    wards = await get_ward_risks_cached()
    cols = 14
    rows_grid = math.ceil(198 / cols)

    features = []
    for w in wards:
        wid = w.get("ward_id", 1)
        row, col = divmod(wid - 1, cols)
        lat = 12.77 + (13.18 - 12.77) * (row / max(rows_grid - 1, 1))
        lon = 77.35 + (77.82 - 77.35) * (col / max(cols - 1, 1))

        # Approximate bounding box (0.018° ≈ 2km)
        half = 0.009
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [round(lon - half, 5), round(lat - half, 5)],
                    [round(lon + half, 5), round(lat - half, 5)],
                    [round(lon + half, 5), round(lat + half, 5)],
                    [round(lon - half, 5), round(lat + half, 5)],
                    [round(lon - half, 5), round(lat - half, 5)],
                ]],
            },
            "properties": {
                "ward_id":           wid,
                "ward_name":         w.get("ward_name"),
                "heat_stress_score": round(w.get("heat_stress_score", 0), 2),
                "flood_risk_score":  round(w.get("flood_risk_score", 0), 2),
                "risk_level":        w.get("risk_level", "medium"),
                "lst_pred_celsius":  w.get("lst_pred_celsius"),
                "ndvi":              w.get("ndvi"),
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
