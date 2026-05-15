"""
Runtime configuration for the NammaClimaGrid API.

The backend should not hardcode service locations or credentials in route
modules. This small settings layer keeps local defaults convenient while
letting Docker, CI, and production deployments override values with env vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv


_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: str = "") -> Tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    environment: str
    allow_all_cors: bool
    cors_origins: Tuple[str, ...]
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    kafka_bootstrap_servers: str
    kafka_alerts_topic: str
    kafka_reports_topic: str
    upload_dir: Path
    skip_model_load: bool
    skip_kafka_consumer: bool
    require_external_services: bool
    auth_required: bool
    admin_api_key: str
    research_api_key: str
    ward_risk_max_age_minutes: int
    ward_risk_min_fresh_ratio: float

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        environment=os.getenv("NCG_ENV", os.getenv("ENVIRONMENT", "development")),
        allow_all_cors=_bool_env("ALLOW_ALL_CORS", False),
        cors_origins=_csv_env(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:3001,"
            "http://localhost:8000,http://127.0.0.1:8000",
        ),
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_db=os.getenv("POSTGRES_DB", "namma_clima_grid"),
        postgres_user=os.getenv("POSTGRES_USER", "ncg_user"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
        kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        kafka_alerts_topic=os.getenv("KAFKA_TOPIC_ALERTS", "climate-alerts"),
        kafka_reports_topic=os.getenv("KAFKA_TOPIC_CITIZEN_REPORTS", "citizen-reports"),
        upload_dir=Path(
            os.getenv("UPLOAD_DIR", str(_MODULE_DIR / "static" / "uploads"))
        ),
        skip_model_load=_bool_env("NCG_SKIP_MODEL_LOAD", False),
        skip_kafka_consumer=_bool_env("NCG_SKIP_KAFKA_CONSUMER", False),
        require_external_services=_bool_env("REQUIRE_EXTERNAL_SERVICES", False),
        auth_required=_bool_env("AUTH_REQUIRED", False),
        admin_api_key=os.getenv("ADMIN_API_KEY", ""),
        research_api_key=os.getenv("RESEARCH_API_KEY", ""),
        ward_risk_max_age_minutes=int(os.getenv("WARD_RISK_MAX_AGE_MINUTES", "60")),
        ward_risk_min_fresh_ratio=float(os.getenv("WARD_RISK_MIN_FRESH_RATIO", "0.95")),
    )
