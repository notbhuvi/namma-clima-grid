"""
Module 5 — Pydantic Schemas
=============================

Request / response models for all FastAPI endpoints.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    low      = "low"
    medium   = "medium"
    high     = "high"
    critical = "critical"


class InterventionType(str, Enum):
    tree_planting       = "tree_planting"
    green_roof          = "green_roof"
    permeable_pavement  = "permeable_pavement"
    urban_wetland       = "urban_wetland"
    cool_pavement       = "cool_pavement"


# ---------------------------------------------------------------------------
# Ward schemas
# ---------------------------------------------------------------------------

class WardRisk(BaseModel):
    """Risk scores for a single BBMP ward."""
    ward_id:            int   = Field(..., ge=1, le=198)
    ward_name:          Optional[str] = None
    heat_stress_score:  float = Field(..., ge=0, le=100, description="UHI stress 0–100")
    flood_risk_score:   float = Field(..., ge=0, le=100, description="Flash flood risk 0–100")
    risk_level:         RiskLevel
    lst_pred_celsius:   Optional[float] = Field(None, description="Predicted LST (°C)")
    ndvi:               Optional[float] = None
    impervious_pct:     Optional[float] = None
    updated_at:         Optional[datetime] = None
    # Live weather fields (present when source == "live_weather")
    temperature_c:      Optional[float] = Field(None, description="Real air temperature (°C)")
    rainfall_mm:        Optional[float] = Field(None, description="Real 24h rainfall (mm)")
    humidity_pct:       Optional[float] = Field(None, description="Relative humidity (%)")


class WardRiskResponse(BaseModel):
    """GET /wards/risk response."""
    wards:         List[WardRisk]
    total:         int
    timestamp:     datetime
    model_version: Optional[str] = None
    source:        Optional[str] = None   # "postgresql_live" | "live_weather" | "model_generated"


class WardRiskSnapshot(BaseModel):
    """One historical risk reading for time-series charts."""
    timestamp:         datetime
    heat_stress_score: float
    flood_risk_score:  float


class WardDetailResponse(BaseModel):
    """GET /wards/{ward_id} response."""
    ward:              WardRisk
    recent_history:    List[WardRiskSnapshot] = []
    active_interventions: List[str] = []


# ---------------------------------------------------------------------------
# Forecast schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    """POST /wards/forecast request."""
    ward_ids:    Optional[List[int]] = Field(None, description="Specific wards (None = all)")
    horizon_hrs: int = Field(6, ge=1, le=72, description="Forecast horizon in hours")


class WardForecast(BaseModel):
    ward_id:               int
    hour:                  int
    heat_stress_predicted: float
    flood_risk_predicted:  float
    confidence:            float = Field(..., ge=0, le=1)


class ForecastResponse(BaseModel):
    forecasts:    List[WardForecast]
    horizon_hrs:  int
    generated_at: datetime


# ---------------------------------------------------------------------------
# Intervention schemas
# ---------------------------------------------------------------------------

class InterventionRequest(BaseModel):
    """POST /interventions/recommend request."""
    budget:     int   = Field(60, ge=1, le=500, description="Total cost budget")
    top_k:      int   = Field(10, ge=1, le=50, description="Max recommendations")
    ward_ids:   Optional[List[int]] = Field(None, description="Focus on specific wards")


class RecommendedIntervention(BaseModel):
    priority_rank:              int
    ward_id:                    int
    ward_name:                  Optional[str] = None
    intervention_type:          InterventionType
    expected_heat_reduction:    float = Field(..., description="Heat-stress reduction (points)")
    expected_flood_reduction:   float = Field(..., description="Flood-risk reduction (points)")
    cost:                       int
    ward_heat_stress:           float
    ward_flood_risk:            float


class InterventionResponse(BaseModel):
    recommendations:  List[RecommendedIntervention]
    budget_used:      int
    budget_remaining: int
    generated_at:     datetime


# ---------------------------------------------------------------------------
# Alert schemas (WebSocket)
# ---------------------------------------------------------------------------

class WardAlert(BaseModel):
    """Real-time alert pushed over WebSocket."""
    ward_id:     int
    alert_type:  str = Field(..., description="heat_wave | flood_warning | air_quality")
    severity:    RiskLevel
    message:     str
    value:       float
    threshold:   float
    timestamp:   datetime


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:       str = "ok"
    version:      str
    models_loaded: dict
    uptime_sec:   float
