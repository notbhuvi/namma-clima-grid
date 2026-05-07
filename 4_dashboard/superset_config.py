"""
4_dashboard/superset_config.py
================================
Apache Superset configuration for NammaClimaGrid.

Connects Superset (running at http://localhost:8089) to:
  - PostgreSQL/PostGIS  — ward features, risk scores, interventions
  - NammaClimaGrid API  — live predictions via HTTP datasource

Usage
─────
  # Copy into the Superset container and import:
  docker cp 4_dashboard/superset_config.py ncg-superset:/app/superset_config.py
  docker exec ncg-superset superset fab create-admin \
      --username admin --firstname Admin --lastname User \
      --email admin@namma.local --password admin
  docker exec ncg-superset superset db upgrade
  docker exec ncg-superset superset init

  # Import dashboard:
  docker exec ncg-superset superset import-dashboards \
      -p /app/dashboards/ward_stress_dashboard.json
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv("SUPERSET_SECRET_KEY", "namma-clima-grid-superset-secret-change-in-prod")
SQLALCHEMY_DATABASE_URI = os.getenv(
    "SUPERSET_META_DB",
    "postgresql+psycopg2://ncg_user:ncg_pass@localhost:5432/superset",
)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
FEATURE_FLAGS = {
    "ENABLE_TEMPLATE_PROCESSING": True,   # Jinja in SQL Lab
    "DASHBOARD_NATIVE_FILTERS": True,
    "DASHBOARD_CROSS_FILTERS": True,
    "DATAPANEL_CLOSED_BY_DEFAULT": False,
    "ESCAPE_MARKDOWN_HTML": False,
}

# ---------------------------------------------------------------------------
# Cache (Redis)
# ---------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = 1

CACHE_CONFIG = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_DEFAULT_TIMEOUT": 300,
    "CACHE_KEY_PREFIX": "superset_",
    "CACHE_REDIS_URL": f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
}
DATA_CACHE_CONFIG  = {**CACHE_CONFIG, "CACHE_KEY_PREFIX": "superset_data_"}
RESULTS_BACKEND    = None   # use default filesystem backend

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
WTF_CSRF_ENABLED          = True
SESSION_COOKIE_HTTPONLY   = True
SESSION_COOKIE_SAMESITE   = "Lax"
ENABLE_PROXY_FIX          = True

# Allow embedding in iframe for the Flutter dashboard tab
HTTP_HEADERS              = {}   # add X-Frame-Options if needed in prod

# ---------------------------------------------------------------------------
# NammaClimaGrid — PostgreSQL database connection info
# (register this in Superset UI: Data → Databases → + Database)
# ---------------------------------------------------------------------------
NAMMA_DB = {
    "database_name": "NammaClimaGrid",
    "sqlalchemy_uri": os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://ncg_user:ncg_pass@localhost:5432/namma_clima",
    ),
    "description": "Bengaluru ward risk scores, intervention history, IoT sensor data",
    "expose_in_sqllab": True,
    "allow_ctas": True,
    "allow_cvas": True,
    "allow_dml": False,
}

# ---------------------------------------------------------------------------
# Mapbox (optional — for geo charts)
# ---------------------------------------------------------------------------
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY", "")

# ---------------------------------------------------------------------------
# Email alerts (optional)
# ---------------------------------------------------------------------------
SMTP_HOST     = os.getenv("SMTP_HOST", "")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_STARTTLS = True
SMTP_SSL      = False
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_MAIL_FROM = "alerts@namma-clima-grid.local"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
import logging
LOG_LEVEL = logging.INFO
