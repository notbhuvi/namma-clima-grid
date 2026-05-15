"""
Routes — Intervention recommendation endpoints
=================================================

  POST /interventions/recommend  → RL-optimised green infra recommendations
  GET  /interventions/types      → available intervention catalogue
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter
from fastapi import Query

from schemas import (
    InterventionRequest, InterventionResponse,
    RecommendedIntervention, InterventionType,
)
from services import get_recommendations, compare_optimizer_with_greedy

router = APIRouter(prefix="/interventions", tags=["interventions"])


# ---------------------------------------------------------------------------
# Intervention catalogue (static)
# ---------------------------------------------------------------------------

INTERVENTION_CATALOGUE = [
    {
        "type":        "tree_planting",
        "display":     "Tree Planting",
        "description": "Plant native trees in public spaces and road medians",
        "cost":        2,
        "heat_effect": "NDVI +0.08, LST −1.5°C, heat stress −6",
        "flood_effect": "Minimal",
    },
    {
        "type":        "green_roof",
        "display":     "Green Roof",
        "description": "Install vegetated roof systems on public buildings",
        "cost":        3,
        "heat_effect": "NDVI +0.04, LST −0.8°C, impervious −5%",
        "flood_effect": "Moderate runoff reduction",
    },
    {
        "type":        "permeable_pavement",
        "display":     "Permeable Pavement",
        "description": "Replace impervious surfaces with permeable alternatives",
        "cost":        4,
        "heat_effect": "Minimal",
        "flood_effect": "Impervious −12%, flood risk −15",
    },
    {
        "type":        "urban_wetland",
        "display":     "Urban Wetland",
        "description": "Construct stormwater wetlands in low-lying areas",
        "cost":        6,
        "heat_effect": "NDVI +0.05, heat stress −3",
        "flood_effect": "Flood risk −25 (highest impact)",
    },
    {
        "type":        "cool_pavement",
        "display":     "Cool Pavement",
        "description": "Apply reflective coating to roads and parking lots",
        "cost":        2,
        "heat_effect": "LST −1.0°C, heat stress −8",
        "flood_effect": "None",
    },
]


# ---------------------------------------------------------------------------
# GET /interventions/types
# ---------------------------------------------------------------------------

@router.get("/types")
async def list_intervention_types() -> List[dict]:
    """Return the catalogue of available green infrastructure interventions."""
    return INTERVENTION_CATALOGUE


# ---------------------------------------------------------------------------
# POST /interventions/recommend
# ---------------------------------------------------------------------------

@router.post("/recommend", response_model=InterventionResponse)
async def recommend_interventions(body: InterventionRequest):
    """
    Use the trained RL policy to recommend optimal green infrastructure
    interventions for reducing city-wide heat stress and flood risk.
    """
    raw = await get_recommendations(
        budget=body.budget,
        top_k=body.top_k,
        ward_ids=body.ward_ids,
    )

    total_cost = sum(r["cost"] for r in raw)

    recommendations = [
        RecommendedIntervention(
            priority_rank=r["priority_rank"],
            ward_id=r["ward_id"],
            ward_name=r.get("ward_name"),
            intervention_type=r["intervention_type"],
            expected_heat_reduction=r["expected_heat_reduction"],
            expected_flood_reduction=r["expected_flood_reduction"],
            cost=r["cost"],
            ward_heat_stress=r["ward_heat_stress"],
            ward_flood_risk=r["ward_flood_risk"],
        )
        for r in raw
    ]

    return InterventionResponse(
        recommendations=recommendations,
        budget_used=total_cost,
        budget_remaining=body.budget - total_cost,
        generated_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# GET /interventions/optimizer-comparison
# ---------------------------------------------------------------------------

@router.get("/optimizer-comparison")
async def optimizer_comparison(
    budget: int = Query(60, ge=1, le=500),
    top_k: int = Query(10, ge=1, le=50),
):
    """
    Compare current optimizer recommendations against a transparent greedy
    baseline. Useful for demos and model defensibility.
    """
    return await compare_optimizer_with_greedy(budget=budget, top_k=top_k)
