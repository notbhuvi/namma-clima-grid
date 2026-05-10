#!/usr/bin/env python3
"""Validate production environment variables before deployment."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REQUIRED = [
    "NCG_ENV",
    "AUTH_REQUIRED",
    "REQUIRE_EXTERNAL_SERVICES",
    "ALLOW_ALL_CORS",
    "CORS_ORIGINS",
    "ADMIN_API_KEY",
    "SECRET_KEY",
    "POSTGRES_HOST",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "KAFKA_BOOTSTRAP_SERVERS",
    "MLFLOW_TRACKING_URI",
    "SUPERSET_SECRET_KEY",
]

PLACEHOLDER_RE = re.compile(r"(change_me|replace_with|your-|example|admin123)", re.I)


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Optional .env file to validate")
    args = parser.parse_args()

    values = dict(os.environ)
    if args.env_file:
        values.update(load_env_file(args.env_file))

    errors: list[str] = []
    warnings: list[str] = []

    for key in REQUIRED:
        value = values.get(key, "").strip()
        if not value:
            errors.append(f"{key} is required")
            continue
        if PLACEHOLDER_RE.search(value):
            errors.append(f"{key} still contains a placeholder value")

    if values.get("NCG_ENV") != "production":
        warnings.append("NCG_ENV is not 'production'")
    if values.get("AUTH_REQUIRED", "").lower() != "true":
        errors.append("AUTH_REQUIRED must be true in production")
    if values.get("REQUIRE_EXTERNAL_SERVICES", "").lower() != "true":
        errors.append("REQUIRE_EXTERNAL_SERVICES must be true in production")
    if values.get("ALLOW_ALL_CORS", "").lower() == "true":
        errors.append("ALLOW_ALL_CORS must be false in production")

    for key in ("ADMIN_API_KEY", "SECRET_KEY", "POSTGRES_PASSWORD", "SUPERSET_SECRET_KEY"):
        value = values.get(key, "")
        if value and len(value) < 24:
            errors.append(f"{key} should be at least 24 characters")

    origins = values.get("CORS_ORIGINS", "")
    if "*" in origins:
        errors.append("CORS_ORIGINS must not contain '*' in production")
    if "localhost" in origins or "127.0.0.1" in origins:
        warnings.append("CORS_ORIGINS contains a local development origin")

    for item in warnings:
        print(f"WARN: {item}")
    for item in errors:
        print(f"ERROR: {item}")

    if errors:
        print(f"Production env validation failed: {len(errors)} error(s)")
        return 1

    print("Production env validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
