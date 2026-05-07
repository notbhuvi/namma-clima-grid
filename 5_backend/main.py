"""
Module 5 — FastAPI Application
=================================

NammaClimaGrid Backend API

Endpoints
──────────
  GET   /health                     → service health + model status
  GET   /wards/risk                 → all 198 wards with risk scores
  GET   /wards/risk/{ward_id}       → single ward detail
  POST  /wards/forecast             → hourly heat/flood forecast
  GET   /interventions/types        → intervention catalogue
  POST  /interventions/recommend    → RL-optimised recommendations
  WS    /ws/alerts                  → real-time ward alert stream
  GET   /bbmp/priority-wards        → BBMP councillor priority list
  GET   /bbmp/intervention-roi-map  → GeoJSON ROI map
  GET   /research/export            → CSV/JSON/GeoJSON data export
  GET   /research/feature-importance → ST-GNN feature importances
  GET   /wards/geojson/live         → Live GeoJSON ward data
  POST  /reports/                   → Submit citizen report
  GET   /reports/ward/{id}/count    → Ward report counts

Start
──────
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Ensure this module's directory is on sys.path for sibling imports
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from schemas  import HealthResponse
from services import init_services, shutdown_services, get_health

from routes.wards         import router as wards_router
from routes.interventions import router as interventions_router
from routes.websocket     import (
    router as ws_router,
    start_alert_background_task,
    stop_alert_background_task,
)
from routes.bbmp     import router as bbmp_router
from routes.research import router as research_router
from routes.reports  import router as reports_router

APP_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, clean up on shutdown."""
    logger.info("NammaClimaGrid API starting …")
    init_services()
    start_alert_background_task()
    logger.success(f"API ready | version={APP_VERSION}")
    yield
    stop_alert_background_task()
    shutdown_services()
    logger.info("API shut down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NammaClimaGrid API",
    description=(
        "Bengaluru thermal-hydrological resilience API.\n\n"
        "Provides per-ward Urban Heat Island stress scores, flash-flood risk "
        "predictions, hourly forecasts, and RL-optimised green infrastructure "
        "intervention recommendations for all 198 BBMP wards."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Flutter web and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(wards_router)
app.include_router(interventions_router)
app.include_router(ws_router)
app.include_router(bbmp_router)
app.include_router(research_router)
app.include_router(reports_router)


# ---------------------------------------------------------------------------
# Health endpoint (on root router)
# ---------------------------------------------------------------------------

def _check_postgres() -> bool:
    """Quick liveness check for PostgreSQL."""
    try:
        import psycopg2  # type: ignore
        conn = psycopg2.connect(
            "postgresql://ncg_user:change_me_in_production@localhost:5432/namma_clima_grid",
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


def _check_kafka() -> bool:
    """Quick liveness check for Kafka."""
    try:
        from kafka import KafkaAdminClient  # type: ignore
        from kafka.errors import NoBrokersAvailable  # type: ignore
        admin = KafkaAdminClient(
            bootstrap_servers="localhost:9092",
            request_timeout_ms=2000,
            api_version_auto_timeout_ms=2000,
        )
        admin.close()
        return True
    except Exception:
        return False


def _check_model_files() -> dict:
    """Check which model checkpoint files exist."""
    _PROJECT_ROOT = _MODULE_DIR.parent
    checkpoints = {
        "thermal": _PROJECT_ROOT / "2_thermal_vision" / "checkpoints" / "best_model.pt",
        "stgnn":   _PROJECT_ROOT / "3_st_gnn"         / "checkpoints" / "gnn_best.pt",
        "rl":      _PROJECT_ROOT / "4_rl_optimizer"   / "checkpoints" / "ppo_best.pt",
    }
    return {name: path.exists() for name, path in checkpoints.items()}


@app.get("/health", tags=["meta"])
async def health():
    """
    Extended service health check — reports:
    - model load status
    - uptime
    - postgres connectivity
    - kafka connectivity
    - model checkpoint file presence
    """
    info = get_health()
    postgres_ok = _check_postgres()
    kafka_ok    = _check_kafka()
    model_files = _check_model_files()

    overall_status = "ok" if postgres_ok else "degraded"

    return {
        "status":        overall_status,
        "version":       APP_VERSION,
        "uptime_sec":    info["uptime_sec"],
        "models_loaded": info["models_loaded"],
        "model_files":   model_files,
        "dependencies": {
            "postgres": "ok" if postgres_ok else "unavailable",
            "kafka":    "ok" if kafka_ok    else "unavailable",
        },
    }


@app.get("/api", tags=["meta"])
async def root():
    """API landing page."""
    return {
        "service":  "NammaClimaGrid API",
        "version":  APP_VERSION,
        "docs":     "/docs",
        "health":   "/health",
        "endpoints": [
            "GET  /wards/risk",
            "GET  /wards/risk/{ward_id}",
            "POST /wards/forecast",
            "GET  /wards/geojson/live",
            "GET  /interventions/types",
            "POST /interventions/recommend",
            "WS   /ws/alerts",
            "GET  /bbmp/priority-wards",
            "GET  /bbmp/intervention-roi-map",
            "GET  /research/export",
            "GET  /research/feature-importance",
            "POST /reports/",
            "GET  /reports/ward/{ward_id}/count",
        ],
    }


# ---------------------------------------------------------------------------
# BBMP dashboard — served at /bbmp
# ---------------------------------------------------------------------------

_STATIC_DIR = _MODULE_DIR / "static"

@app.get("/bbmp", include_in_schema=False)
async def bbmp_dashboard():
    """Redirect /bbmp → /bbmp/"""
    return RedirectResponse(url="/bbmp/")

@app.get("/bbmp/", include_in_schema=False)
async def bbmp_dashboard_index():
    """Serve the BBMP councillor dashboard."""
    return FileResponse(str(_STATIC_DIR / "bbmp.html"))

if _STATIC_DIR.exists():
    app.mount("/bbmp-static", StaticFiles(directory=str(_STATIC_DIR)), name="bbmp-static")

# ---------------------------------------------------------------------------
# Flutter web app — served at root (must be LAST so API routes take priority)
# ---------------------------------------------------------------------------

_FLUTTER_BUILD = _MODULE_DIR.parent / "6_flutter_app" / "build" / "web"
if _FLUTTER_BUILD.exists():
    app.mount("/", StaticFiles(directory=str(_FLUTTER_BUILD), html=True), name="flutter")
    logger.info(f"Serving Flutter web app from {_FLUTTER_BUILD}")
