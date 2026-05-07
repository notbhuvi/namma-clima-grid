"""
Routes — Ward risk & forecast endpoints
=========================================

  GET  /wards/risk            → all 198 wards with heat/flood scores
  GET  /wards/risk/{ward_id}  → single ward detail + recent history
  POST /wards/forecast        → hourly heat/flood forecast
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from schemas import (
    ForecastRequest, ForecastResponse, WardDetailResponse,
    WardForecast, WardRisk, WardRiskResponse, WardRiskSnapshot,
)
from services import get_ward_risks, get_forecasts, _ward_catalogue

router = APIRouter(prefix="/wards", tags=["wards"])


# ---------------------------------------------------------------------------
# GET /wards/risk
# ---------------------------------------------------------------------------

@router.get("/risk", response_model=WardRiskResponse)
async def list_ward_risks(
    risk_level: Optional[str] = Query(None, description="Filter: low|medium|high|critical"),
    sort_by:    str           = Query("heat_stress_score", description="Sort column"),
    limit:      int           = Query(198, ge=1, le=198),
):
    """Return risk scores for all wards."""
    df = await get_ward_risks()

    if risk_level:
        df = df[df["risk_level"] == risk_level]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    df = df.head(limit)
    names = _ward_catalogue()

    wards = []
    for _, row in df.iterrows():
        # Use the DB timestamp if present, else fall back to now()
        raw_ts = row.get("updated_at")
        if raw_ts is not None:
            try:
                updated = (raw_ts if hasattr(raw_ts, "tzinfo")
                           else datetime.fromisoformat(str(raw_ts)))
            except Exception:
                updated = datetime.now(timezone.utc)
        else:
            updated = datetime.now(timezone.utc)

        # Live weather fields (None when source is synthetic)
        temp_c   = float(row["temperature_c"])   if "temperature_c"   in row and row["temperature_c"]   is not None and str(row["temperature_c"])   != "nan" else None
        rain_mm  = float(row["rainfall_mm_24h"]) if "rainfall_mm_24h" in row and row["rainfall_mm_24h"] is not None and str(row["rainfall_mm_24h"]) != "nan" else None
        humid    = float(row["humidity_pct"])    if "humidity_pct"    in row and row["humidity_pct"]    is not None and str(row["humidity_pct"])    != "nan" else None

        wards.append(WardRisk(
            ward_id=int(row["ward_id"]),
            ward_name=names.get(int(row["ward_id"])),
            heat_stress_score=round(float(row["heat_stress_score"]), 2),
            flood_risk_score=round(float(row["flood_risk_score"]), 2),
            risk_level=str(row.get("risk_level", "medium")),
            lst_pred_celsius=row.get("lst_pred_celsius"),
            ndvi=row.get("ndvi"),
            impervious_pct=row.get("impervious_pct"),
            updated_at=updated,
            temperature_c=round(temp_c, 1)  if temp_c  is not None else None,
            rainfall_mm=round(rain_mm, 2)   if rain_mm is not None else None,
            humidity_pct=round(humid, 1)    if humid   is not None else None,
        ))

    # Indicate data freshness tier in the response
    has_db_ts = "updated_at" in df.columns and df["updated_at"].notna().any()
    df_source = df["data_source"].iloc[0] if "data_source" in df.columns else "model_generated"
    if has_db_ts:
        source = "postgresql_live"
    else:
        source = df_source  # "live_weather" or "model_generated"

    return WardRiskResponse(
        wards=wards,
        total=len(wards),
        timestamp=datetime.now(timezone.utc),
        source=source,
    )


# ---------------------------------------------------------------------------
# GET /wards/risk/{ward_id}
# ---------------------------------------------------------------------------

@router.get("/risk/{ward_id}", response_model=WardDetailResponse)
async def get_ward_detail(ward_id: int):
    """Return detailed risk info for a specific ward."""
    if ward_id < 1 or ward_id > 198:
        raise HTTPException(status_code=404, detail=f"Ward {ward_id} not found")

    df = await get_ward_risks(ward_ids=[ward_id])
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Ward {ward_id} not found")

    row = df.iloc[0]
    names = _ward_catalogue()

    ward = WardRisk(
        ward_id=int(row["ward_id"]),
        ward_name=names.get(ward_id),
        heat_stress_score=round(float(row["heat_stress_score"]), 2),
        flood_risk_score=round(float(row["flood_risk_score"]), 2),
        risk_level=row.get("risk_level", "medium"),
        lst_pred_celsius=row.get("lst_pred_celsius"),
        ndvi=row.get("ndvi"),
        impervious_pct=row.get("impervious_pct"),
        updated_at=datetime.now(timezone.utc),
    )

    return WardDetailResponse(ward=ward)


# ---------------------------------------------------------------------------
# POST /wards/forecast
# ---------------------------------------------------------------------------

@router.post("/forecast", response_model=ForecastResponse)
async def create_forecast(body: ForecastRequest):
    """Generate hourly heat/flood forecasts for selected wards."""
    raw = await get_forecasts(
        ward_ids=body.ward_ids,
        horizon_hrs=body.horizon_hrs,
    )
    forecasts = [WardForecast(**f) for f in raw]
    return ForecastResponse(
        forecasts=forecasts,
        horizon_hrs=body.horizon_hrs,
        generated_at=datetime.now(timezone.utc),
    )
