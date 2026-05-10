"""
Routes — BBMP Councillor Dashboard endpoints
==============================================

  GET /bbmp/priority-wards?limit=20&min_heat_score=70
      Returns wards sorted by composite risk (heat × 0.6 + flood × 0.4)

  GET /bbmp/intervention-roi-map
      Returns a GeoJSON FeatureCollection with ward polygon approximations
      and ROI scores for each ward
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from auth import require_role
from services import get_ward_risks_cached, get_interventions_cached

router = APIRouter(prefix="/bbmp", tags=["bbmp"])


# ---------------------------------------------------------------------------
# POST /bbmp/broadcast-alert  — BBMP officials push alert to all citizens
# ---------------------------------------------------------------------------

class BroadcastAlertRequest(BaseModel):
    ward_id:    int   = Field(..., ge=1, le=198)
    alert_type: str   = Field(..., description="heat_wave | flood_warning | air_quality | advisory")
    severity:   str   = Field(..., description="low | medium | high | critical")
    message:    str   = Field(..., max_length=300)


@router.post("/broadcast-alert")
async def broadcast_alert(
    body: BroadcastAlertRequest,
    _auth = Depends(require_role("admin")),
):
    """
    BBMP officials send an alert to all connected citizen app users.
    Requires an admin bearer token.
    """
    from routes.websocket import inject_alert
    from datetime import datetime, timezone

    alert = {
        "ward_id":    body.ward_id,
        "alert_type": body.alert_type,
        "severity":   body.severity,
        "message":    f"📢 BBMP Official Alert: {body.message}",
        "value":      0.0,
        "threshold":  0.0,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "source":     "bbmp_official",
    }
    await inject_alert(alert)
    return {
        "status":       "broadcast",
        "alert":        alert,
        "recipients":   "all_connected_citizens",
    }


# ---------------------------------------------------------------------------
# GET /bbmp/priority-wards
# ---------------------------------------------------------------------------

@router.get("/priority-wards")
async def priority_wards(
    limit: int           = Query(20, ge=1, le=198),
    min_heat_score: float = Query(0.0, ge=0, le=100),
):
    """
    Return wards sorted by composite risk (heat×0.6 + flood×0.4).
    Optionally filter by minimum heat stress score.
    """
    wards = await get_ward_risks_cached()

    # Filter
    if min_heat_score > 0:
        wards = [w for w in wards if w.get("heat_stress_score", 0) >= min_heat_score]

    # Sort by composite risk (descending)
    wards = sorted(
        wards,
        key=lambda w: 0.6 * w.get("heat_stress_score", 0) + 0.4 * w.get("flood_risk_score", 0),
        reverse=True,
    )[:limit]

    result = []
    for rank, w in enumerate(wards, 1):
        composite = round(
            0.6 * w.get("heat_stress_score", 0) + 0.4 * w.get("flood_risk_score", 0), 2
        )
        result.append({
            "priority_rank":      rank,
            "ward_id":            w.get("ward_id"),
            "ward_name":          w.get("ward_name"),
            "heat_stress_score":  round(w.get("heat_stress_score", 0), 2),
            "flood_risk_score":   round(w.get("flood_risk_score", 0), 2),
            "composite_risk":     composite,
            "risk_level":         w.get("risk_level", "medium"),
            "lst_pred_celsius":   w.get("lst_pred_celsius"),
            "ndvi":               w.get("ndvi"),
            "impervious_pct":     w.get("impervious_pct"),
        })

    return {
        "priority_wards": result,
        "total": len(result),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /bbmp/intervention-roi-map
# ---------------------------------------------------------------------------

@router.get("/intervention-roi-map")
async def intervention_roi_map():
    """
    Returns a GeoJSON FeatureCollection with approximate ward polygon centroids
    and ROI scores for prioritising green-infra investments.

    ROI = (expected_heat_reduction + expected_flood_reduction) / cost
    """
    wards        = await get_ward_risks_cached()
    interventions = await get_interventions_cached()

    # Build a ward_id → intervention lookup
    ward_iv_map = {}
    for iv in interventions:
        wid = iv.get("ward_id")
        if wid not in ward_iv_map:
            ward_iv_map[wid] = iv

    features = []
    for w in wards:
        wid = w.get("ward_id", 0)
        # Approximate ward centroid from grid position (14 cols)
        cols = 14
        row, col = divmod(wid - 1, cols)
        rows_grid = math.ceil(198 / cols)
        lat = 12.77 + (13.18 - 12.77) * (row / max(rows_grid - 1, 1))
        lon = 77.35 + (77.82 - 77.35) * (col / max(cols - 1, 1))

        iv = ward_iv_map.get(wid, {})
        cost = iv.get("cost", 1) or 1
        heat_red  = iv.get("expected_heat_reduction", 0.0)
        flood_red = iv.get("expected_flood_reduction", 0.0)
        roi = round((heat_red + flood_red) / cost, 3)

        composite = round(
            0.6 * w.get("heat_stress_score", 0) + 0.4 * w.get("flood_risk_score", 0), 2
        )

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(lon, 5), round(lat, 5)],
            },
            "properties": {
                "ward_id":               wid,
                "ward_name":             w.get("ward_name"),
                "composite_risk":        composite,
                "heat_stress_score":     round(w.get("heat_stress_score", 0), 2),
                "flood_risk_score":      round(w.get("flood_risk_score", 0), 2),
                "risk_level":            w.get("risk_level", "medium"),
                "recommended_intervention": iv.get("intervention_type"),
                "expected_heat_reduction":  heat_red,
                "expected_flood_reduction": flood_red,
                "cost":                  cost,
                "roi_score":             roi,
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_wards": len(features),
    }
