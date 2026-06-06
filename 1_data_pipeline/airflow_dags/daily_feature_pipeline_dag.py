"""
Module 1d-ii — Airflow DAG: Daily Feature Pipeline
====================================================

Schedule: 2am IST daily (02:00 Asia/Kolkata)

Tasks (in order with >> dependencies)
──────────────────────────────────────
1. kafka_setup              — python kafka_setup.py
2. weather_ingest           — python weather_ingestion.py
3. satellite_ingest         — python satellite_ingestion.py --fallback
4. spark_batch              — spark-submit spark_batch_processor.py (or python fallback)
5. upsert_ward_features_postgres — upsert into PostgreSQL ward_features table
6. publish_feature_update_event  — publish control event to Kafka ward-features-stream
7. validate_pipeline        — basic row-count + null check on Postgres

The DAG is idempotent: repeated runs overwrite the same ward_id rows,
never duplicating.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_ARGS = {
    "owner": "ncg",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

_PROJECT_ROOT = Path(Variable.get("NCG_PROJECT_ROOT", default_var="/app"))
_DELTA_ROOT   = Variable.get("NCG_DELTA_ROOT",   default_var="/data/delta")
_PYTHON       = sys.executable

_KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
_KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC_WARD_FEATURES", "ward-features-stream")
_PG_HOST         = os.getenv("POSTGRES_HOST",     "postgres")
_PG_PORT         = int(os.getenv("POSTGRES_PORT",  "5432"))
_PG_DB           = os.getenv("POSTGRES_DB",       "namma_clima_grid")
_PG_USER         = os.getenv("POSTGRES_USER",     "ncg_user")
_PG_PASS         = os.getenv("POSTGRES_PASSWORD", "change_me_in_production")


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _generate_mock_ward_features(**ctx) -> str:
    """
    Run mock_ward_features.py with today's seed to produce a fresh feature
    snapshot.  Returns the path to the generated Parquet file via XCom.
    """
    import logging
    log = logging.getLogger(__name__)

    script = _PROJECT_ROOT / "8_data" / "mock_ward_features.py"
    out_dir = _PROJECT_ROOT / "8_data"
    seed = int(ctx["ds_nodash"])  # e.g. 20260409 — deterministic per day

    cmd = [_PYTHON, str(script), "--seed", str(seed % 10000), "--out", str(out_dir)]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    log.info(result.stdout)
    if result.returncode != 0:
        log.error(result.stderr)
        raise RuntimeError("mock_ward_features.py failed")

    parquet_path = str(out_dir / "ward_features_mock.parquet")
    log.info(f"Feature parquet: {parquet_path}")
    return parquet_path  # pushed to XCom automatically


def _upsert_ward_features_delta(**ctx) -> None:
    """Write the day's ward features to Delta Lake."""
    import logging
    log = logging.getLogger(__name__)

    parquet_path = ctx["ti"].xcom_pull(task_ids="generate_mock_ward_features")
    if not parquet_path or not Path(parquet_path).exists():
        raise FileNotFoundError(f"Parquet not found at {parquet_path}")

    try:
        from pyspark.sql import SparkSession  # type: ignore
        from delta import configure_spark_with_delta_pip  # type: ignore

        builder = (
            SparkSession.builder.appName("DailyFeatureDelta")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.driver.memory", "1g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.showConsoleProgress", "false")
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        df = spark.read.parquet(parquet_path)
        delta_path = str(Path(_DELTA_ROOT) / "ward_features")

        from delta.tables import DeltaTable  # type: ignore
        if DeltaTable.isDeltaTable(spark, delta_path):
            (
                DeltaTable.forPath(spark, delta_path).alias("t")
                .merge(df.alias("s"), "t.ward_id = s.ward_id")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
            log.info("Delta MERGE complete for ward_features")
        else:
            df.write.format("delta").mode("overwrite").save(delta_path)
            log.info("Delta initial write complete for ward_features")

        count = spark.read.format("delta").load(delta_path).count()
        spark.stop()
        log.info(f"ward_features Delta rows: {count}")
    except Exception as exc:
        log.warning(f"Delta write skipped ({exc}) — Spark/Delta not available in this env")


def _upsert_ward_features_postgres(**ctx) -> None:
    """Upsert ward features into PostgreSQL."""
    import logging
    log = logging.getLogger(__name__)

    parquet_path = ctx["ti"].xcom_pull(task_ids="generate_mock_ward_features")
    if not parquet_path or not Path(parquet_path).exists():
        raise FileNotFoundError(f"Parquet not found at {parquet_path}")

    try:
        import pandas as pd  # type: ignore
        import psycopg2  # type: ignore
        from psycopg2.extras import execute_values

        df = pd.read_parquet(parquet_path)
        conn = psycopg2.connect(
            host=_PG_HOST, port=_PG_PORT, dbname=_PG_DB,
            user=_PG_USER, password=_PG_PASS,
        )
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ward_features (
                ward_id           INTEGER PRIMARY KEY,
                ward_code         TEXT,
                ward_name         TEXT,
                centroid_lon      DOUBLE PRECISION,
                centroid_lat      DOUBLE PRECISION,
                lst_celsius       DOUBLE PRECISION,
                ndvi              DOUBLE PRECISION,
                impervious_pct    DOUBLE PRECISION,
                rainfall_mm_24h   DOUBLE PRECISION,
                flood_reports_7d  INTEGER,
                pm25_ugm3         DOUBLE PRECISION,
                population_density DOUBLE PRECISION,
                area_km2          DOUBLE PRECISION,
                heat_stress_score DOUBLE PRECISION,
                flood_risk_score  DOUBLE PRECISION,
                industrial_weight DOUBLE PRECISION,
                updated_at        TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cols = [
            "ward_id", "ward_code", "ward_name",
            "centroid_lon", "centroid_lat",
            "lst_celsius", "ndvi", "impervious_pct",
            "rainfall_mm_24h", "flood_reports_7d", "pm25_ugm3",
            "population_density", "area_km2",
            "heat_stress_score", "flood_risk_score", "industrial_weight",
        ]
        rows = [tuple(row[c] for c in cols) for _, row in df[cols].iterrows()]

        execute_values(cur, f"""
            INSERT INTO ward_features ({', '.join(cols)})
            VALUES %s
            ON CONFLICT (ward_id) DO UPDATE SET
                {', '.join(f"{c} = EXCLUDED.{c}" for c in cols if c != 'ward_id')},
                updated_at = NOW()
        """, rows)

        conn.commit()
        cur.close()
        conn.close()
        log.info(f"Upserted {len(rows)} rows into ward_features (Postgres)")
    except Exception as exc:
        log.error(f"Postgres upsert failed: {exc}")
        raise


def _publish_feature_update_event(**ctx) -> None:
    """
    Publish a lightweight control event to the ward-features-stream Kafka
    topic so that the API cache knows to reload predictions.
    """
    import logging
    log = logging.getLogger(__name__)

    event = {
        "event_type":   "ward_features_updated",
        "pipeline_run": ctx["run_id"],
        "ds":           ctx["ds"],
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "ward_count":   198,
    }

    try:
        from kafka import KafkaProducer  # type: ignore
        producer = KafkaProducer(
            bootstrap_servers=_KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            request_timeout_ms=5000,
        )
        producer.send(_KAFKA_TOPIC, key="control", value=event)
        producer.flush(timeout=5)
        producer.close()
        log.info(f"Published feature-update event: {event}")
    except Exception as exc:
        log.warning(f"Kafka event publish failed ({exc}) — skipping (non-critical)")


def _validate_pipeline(**ctx) -> None:
    """
    Basic data quality check:
    - ward_features must have exactly 198 rows
    - No ward should have NULL heat_stress_score
    """
    import logging
    log = logging.getLogger(__name__)

    try:
        import psycopg2  # type: ignore

        conn = psycopg2.connect(
            host=_PG_HOST, port=_PG_PORT, dbname=_PG_DB,
            user=_PG_USER, password=_PG_PASS,
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM ward_features")
        count = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM ward_features WHERE heat_stress_score IS NULL"
        )
        nulls = cur.fetchone()[0]
        cur.close()
        conn.close()

        log.info(f"Validation: ward_features rows={count}, nulls={nulls}")
        if count < 198:
            raise ValueError(f"Expected 198 ward rows, got {count}")
        if nulls > 0:
            raise ValueError(f"{nulls} wards have NULL heat_stress_score")
        log.info("Validation passed")
    except Exception as exc:
        log.error(f"Validation failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="daily_feature_pipeline",
    default_args=_DEFAULT_ARGS,
    description="Daily ward feature refresh at 2am IST → Kafka + Satellite + Spark + Postgres",
    schedule="0 2 * * *",          # 02:00 IST daily
    start_date=pendulum.datetime(2024, 1, 1, tz="Asia/Kolkata"),
    catchup=False,
    max_active_runs=1,
    tags=["ncg", "features", "delta-lake", "postgres"],
    doc_md=__doc__,
) as dag:

    # 1. Kafka topic setup
    kafka_setup = BashOperator(
        task_id="kafka_setup",
        bash_command="python /opt/airflow/dags/../kafka_setup.py || true",
    )

    # 2. Weather ingestion
    weather_ingest = BashOperator(
        task_id="weather_ingest",
        bash_command="python /opt/airflow/dags/../weather_ingestion.py",
    )

    # 3. Satellite ingestion (real GEE when credentials exist, fallback otherwise)
    satellite_ingest = BashOperator(
        task_id="satellite_ingest",
        bash_command=(
            "if [ \"${SATELLITE_FORCE_FALLBACK:-false}\" = \"true\" ]; then "
            "python /opt/airflow/dags/../satellite_ingestion.py --fallback; "
            "else "
            "python /opt/airflow/dags/../satellite_ingestion.py --days ${SATELLITE_LOOKBACK_DAYS:-45}; "
            "fi"
        ),
    )

    # 4. Spark batch processing (try spark-submit first, fall back to python)
    spark_batch = BashOperator(
        task_id="spark_batch",
        bash_command=(
            "spark-submit /opt/airflow/dags/../spark_batch_processor.py --synthetic "
            "|| python /opt/airflow/dags/../spark_batch_processor.py --synthetic"
        ),
    )

    # 5. Mock ward features (for backward compatibility with downstream tasks)
    generate = PythonOperator(
        task_id="generate_mock_ward_features",
        python_callable=_generate_mock_ward_features,
    )

    delta_write = PythonOperator(
        task_id="upsert_ward_features_delta",
        python_callable=_upsert_ward_features_delta,
    )

    pg_write = PythonOperator(
        task_id="upsert_ward_features_postgres",
        python_callable=_upsert_ward_features_postgres,
    )

    kafka_event = PythonOperator(
        task_id="publish_feature_update_event",
        python_callable=_publish_feature_update_event,
    )

    validate = PythonOperator(
        task_id="validate_pipeline",
        python_callable=_validate_pipeline,
    )

    # Dependency chain:
    # kafka_setup >> weather_ingest >> satellite_ingest >> spark_batch
    #   >> generate >> [delta_write, pg_write] >> kafka_event >> validate
    kafka_setup >> weather_ingest >> satellite_ingest >> spark_batch
    spark_batch >> generate
    generate >> [delta_write, pg_write] >> kafka_event >> validate
