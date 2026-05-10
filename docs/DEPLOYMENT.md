# Production Deployment Notes

NammaClimaGrid should run in production with external services required and
auth enabled:

```bash
python scripts/check_production_env.py --env-file .env.production
docker compose up -d api postgres redis kafka mlflow airflow-webserver airflow-scheduler superset grafana
```

Minimum production controls:

- `AUTH_REQUIRED=true`
- `REQUIRE_EXTERNAL_SERVICES=true`
- `ALLOW_ALL_CORS=false`
- Strong `ADMIN_API_KEY`, `SECRET_KEY`, `POSTGRES_PASSWORD`, and `SUPERSET_SECRET_KEY`
- Explicit `CORS_ORIGINS` for the deployed web domain
- Real Postgres/Kafka/MLflow endpoints, not localhost defaults

Operational checks:

```bash
curl -fsS http://localhost:8000/health
NCG_SKIP_MODEL_LOAD=true NCG_SKIP_KAFKA_CONSUMER=true python -m pytest -q 5_backend/tests
python 8_mlops/generate_quality_reports.py
python 8_mlops/generate_quality_reports.py --input /path/to/real_ward_features.parquet
```

The BBMP dashboard no longer contains a hardcoded browser password. Officials
must enter the configured `ADMIN_API_KEY`; protected moderation and broadcast
endpoints require `Authorization: Bearer <ADMIN_API_KEY>`.
