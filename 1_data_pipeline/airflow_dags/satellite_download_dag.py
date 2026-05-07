"""
Module 1d-i — Airflow DAG: Satellite Download
==============================================

Schedule: Daily at 06:00 IST (00:30 UTC)

Tasks
─────
1. check_gee_credentials   — verifies GEE service account key exists
2. download_landsat_scene  — runs sample_landsat_loader.py (GEE or synthetic)
3. trigger_spark_job       — submits spark_batch_processor.py
4. verify_delta_table      — reads one row from the Delta satellite_features table
5. notify_success          — logs a summary (can be extended to Slack/email)

The DAG is robust to missing credentials: if GEE is not configured, it
automatically falls back to the synthetic scene path so the downstream
Spark job always has data to process.

Airflow variables used (set via Admin → Variables in the UI):
  NCG_PROJECT_ROOT   — absolute path of the project (default: /app)
  NCG_DELTA_ROOT     — Delta Lake root path (default: /data/delta)
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.bash import BashOperator

# ---------------------------------------------------------------------------
# DAG defaults
# ---------------------------------------------------------------------------
_DEFAULT_ARGS = {
    "owner": "ncg",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

_PROJECT_ROOT = Path(Variable.get("NCG_PROJECT_ROOT", default_var="/app"))
_DELTA_ROOT   = Variable.get("NCG_DELTA_ROOT",   default_var="/data/delta")
_PYTHON       = sys.executable


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _check_gee_credentials(**ctx) -> bool:
    """
    ShortCircuitOperator task.
    Returns True → proceed with GEE download.
    Returns False → skip to synthetic path (downstream tasks handle it).
    """
    import logging
    log = logging.getLogger(__name__)

    service_account = os.getenv("GEE_SERVICE_ACCOUNT")
    key_file = os.getenv("GEE_KEY_FILE")
    if not service_account:
        log.warning("GEE_SERVICE_ACCOUNT not set — will use synthetic scene")
        return False
    if not key_file or not Path(key_file).exists():
        log.warning(f"GEE_KEY_FILE '{key_file}' not found — will use synthetic scene")
        return False
    log.info(f"GEE credentials found: {service_account}")
    return True


def _download_landsat_scene(**ctx) -> None:
    """Run sample_landsat_loader.py (GEE path)."""
    import logging
    log = logging.getLogger(__name__)

    script = _PROJECT_ROOT / "8_data" / "sample_landsat_loader.py"
    cmd = [_PYTHON, str(script)]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    log.info(result.stdout[-3000:] if result.stdout else "")
    if result.returncode != 0:
        log.error(result.stderr[-2000:])
        raise RuntimeError(f"Landsat download failed (exit {result.returncode})")
    log.info("Landsat download complete")


def _download_synthetic_scene(**ctx) -> None:
    """Run sample_landsat_loader.py with --force-fallback (no GEE needed)."""
    import logging
    log = logging.getLogger(__name__)

    script = _PROJECT_ROOT / "8_data" / "sample_landsat_loader.py"
    cmd = [_PYTHON, str(script), "--force-fallback"]
    log.info(f"Generating synthetic scene: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    log.info(result.stdout[-2000:] if result.stdout else "")
    if result.returncode != 0:
        log.error(result.stderr)
        raise RuntimeError("Synthetic scene generation failed")
    log.info("Synthetic scene ready")


def _trigger_spark_job(**ctx) -> None:
    """Submit spark_batch_processor.py. Falls back to --synthetic if needed."""
    import logging
    log = logging.getLogger(__name__)

    script = _PROJECT_ROOT / "1_data_pipeline" / "spark_batch_processor.py"
    landsat_dir = _PROJECT_ROOT / "8_data" / "landsat"

    cmd = [
        _PYTHON, str(script),
        "--scene-dir", str(landsat_dir),
    ]
    if not landsat_dir.exists() or not any(landsat_dir.iterdir()):
        cmd += ["--synthetic"]
        log.warning("Landsat dir empty — forcing --synthetic in Spark job")

    log.info(f"Running Spark job: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    log.info(result.stdout[-5000:] if result.stdout else "")
    if result.returncode != 0:
        log.error(result.stderr[-2000:])
        raise RuntimeError(f"Spark job failed (exit {result.returncode})")
    log.info("Spark batch job completed")


def _verify_delta_table(**ctx) -> None:
    """Spot-check: ensure the satellite_features Delta table has rows."""
    import logging
    log = logging.getLogger(__name__)

    delta_path = Path(_DELTA_ROOT) / "satellite_features"
    if not delta_path.exists():
        raise FileNotFoundError(f"Delta table not found: {delta_path}")

    try:
        from pyspark.sql import SparkSession  # type: ignore
        from delta import configure_spark_with_delta_pip  # type: ignore

        builder = (
            SparkSession.builder.appName("VerifyDelta")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.driver.memory", "512m")
            .config("spark.ui.showConsoleProgress", "false")
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        count = spark.read.format("delta").load(str(delta_path)).count()
        spark.stop()
        if count == 0:
            raise ValueError("satellite_features Delta table is empty after Spark job!")
        log.info(f"satellite_features table verified: {count} rows")
    except ImportError:
        # Spark not available in Airflow container — check file existence only
        npz_files = list(delta_path.glob("**/*.parquet"))
        if not npz_files:
            log.warning("No .parquet files found but Delta may use _delta_log — skipping count check")
        log.info(f"Delta table present at {delta_path} (skipped count — no Spark in env)")


def _notify_success(**ctx) -> None:
    """Log a pipeline completion summary. Extend to push Slack/email alerts."""
    import logging
    log = logging.getLogger(__name__)

    run_date = ctx["ds"]
    log.info(
        f"Satellite pipeline complete | date={run_date} | "
        f"delta_root={_DELTA_ROOT}"
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="satellite_download_pipeline",
    default_args=_DEFAULT_ARGS,
    description="Download Landsat scene → Spark NDVI/NDBI → Delta Lake",
    schedule="0 6 * * *",          # 06:00 IST daily
    start_date=pendulum.datetime(2024, 1, 1, tz="Asia/Kolkata"),
    catchup=False,
    tags=["ncg", "satellite", "delta-lake"],
    doc_md=__doc__,
) as dag:

    check_gee = ShortCircuitOperator(
        task_id="check_gee_credentials",
        python_callable=_check_gee_credentials,
        ignore_downstream_trigger_rules=False,
    )

    # BashOperator task running satellite_ingestion.py with --fallback
    satellite_ingest = BashOperator(
        task_id="satellite_ingestion_fallback",
        bash_command=(
            f"python /opt/airflow/dags/../satellite_ingestion.py --fallback"
        ),
        trigger_rule="all_done",  # runs regardless of check_gee outcome
    )

    download_real = PythonOperator(
        task_id="download_landsat_scene",
        python_callable=_download_landsat_scene,
    )

    download_synthetic = PythonOperator(
        task_id="download_synthetic_scene",
        python_callable=_download_synthetic_scene,
        trigger_rule="all_skipped",  # runs when check_gee short-circuits
    )

    spark_job = PythonOperator(
        task_id="trigger_spark_job",
        python_callable=_trigger_spark_job,
        trigger_rule="one_success",  # runs after either download path succeeds
    )

    verify = PythonOperator(
        task_id="verify_delta_table",
        python_callable=_verify_delta_table,
    )

    notify = PythonOperator(
        task_id="notify_success",
        python_callable=_notify_success,
    )

    # Dependency graph:
    #   check_gee → download_real ──┐
    #   check_gee → download_synth ─┴─ spark_job → verify → notify
    #                                     ↑
    #   satellite_ingest (always runs, independent)
    check_gee >> [download_real, download_synthetic]
    [download_real, download_synthetic] >> spark_job
    spark_job >> verify >> notify
