from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("NCG_SKIP_MODEL_LOAD", "true")
os.environ.setdefault("NCG_SKIP_KAFKA_CONSUMER", "true")
os.environ.setdefault("AUTH_REQUIRED", "true")
os.environ.setdefault("ADMIN_API_KEY", "ci-admin-token-please-rotate")

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "5_backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402


def test_health_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["version"]
    assert body["status"] in {"ok", "degraded"}
    assert set(body["models_loaded"]) == {"thermal_vision", "st_gnn", "rl_optimizer"}
    assert set(body["dependencies"]) == {"postgres", "kafka"}


def test_ward_risk_fallback_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/wards/risk", params={"limit": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["wards"]) == 3
    assert body["source"] == "model_generated"
    assert body["prediction_confidence"]["tier"] == "low"
    assert {"ward_id", "heat_stress_score", "flood_risk_score", "risk_level"} <= set(body["wards"][0])
    assert {"data_source", "confidence_tier", "confidence_score"} <= set(body["wards"][0])


def test_intervention_catalogue_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/interventions/types")

    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 5
    assert {item["type"] for item in body} >= {
        "tree_planting",
        "green_roof",
        "permeable_pavement",
        "urban_wetland",
        "cool_pavement",
    }


def test_model_validation_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/research/model-validation")

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_level"] == "prototype_synthetic_validation"
    assert "important_caveat" in body
    assert {"cnn_vit_thermal", "st_gnn", "ppo_optimizer", "citizen_image_classifier"} <= set(
        body["model_metrics"]
    )


def test_optimizer_comparison_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/interventions/optimizer-comparison", params={"budget": 20, "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["budget"] == 20
    assert body["optimizer"]["totals"]["budget_used"] <= 20
    assert body["greedy_baseline"]["totals"]["budget_used"] <= 20
    assert "optimizer_minus_greedy" in body


def test_admin_endpoints_require_bearer_token() -> None:
    with TestClient(app) as client:
        response = client.get("/reports/recent")
        assert response.status_code == 401

        response = client.get(
            "/reports/recent",
            headers={"Authorization": "Bearer ci-admin-token-please-rotate"},
        )
        assert response.status_code == 200
        assert response.json()["reports"] == []
