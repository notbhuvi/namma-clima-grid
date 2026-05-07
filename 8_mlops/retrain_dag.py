"""
Module 8c — Weekly Retrain DAG
================================

Airflow DAG that runs every Sunday at 02:00 IST (20:30 UTC Saturday) to:

  1. check_drift          — run PSI drift check on all three models
  2. retrain_thermal      — retrain CNN-ViT on latest ward_features parquet
  3. retrain_gnn          — retrain ST-GNN on latest ward_features_history
  4. retrain_rl           — retrain PPO policy (shorter run: 100k steps)
  5. promote_thermal      — MLflow Staging → Production if threshold met
  6. promote_gnn          — MLflow Staging → Production
  7. promote_rl           — MLflow Staging → Production
  8. smoke_test           — hit /health endpoint to confirm new models loaded
  9. notify               — send completion summary to log (+ Slack if configured)

Drift gate
───────────
  If no model exceeds PSI 0.1, only the drift check runs and the DAG
  short-circuits (skips retrain). This prevents unnecessary compute on
  stable weeks.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths to module training scripts
_THERMAL_TRAIN = str(_PROJECT_ROOT / "2_thermal_vision" / "train.py")
_GNN_TRAIN     = str(_PROJECT_ROOT / "3_st_gnn"         / "train.py")
_RL_TRAIN      = str(_PROJECT_ROOT / "4_rl_optimizer"   / "train.py")
_REGISTRY      = str(_PROJECT_ROOT / "8_mlops"          / "model_registry.py")
_MONITORING    = str(_PROJECT_ROOT / "8_mlops"          / "monitoring.py")

_PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner":            "ncg-mlops",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=15),
    "email_on_failure": False,
}


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    """Run a subprocess command, raising on non-zero exit."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    result.check_returncode()


def check_drift_task(**context) -> bool:
    """
    Run PSI drift checks for all three models.
    Returns True if any model shows significant drift (PSI > 0.1),
    which triggers the retrain branch.
    """
    drift_detected = False
    for model in ["thermal_vision", "st_gnn", "rl"]:
        try:
            result = subprocess.run(
                [_PYTHON, _MONITORING, "--check-drift", "--model", model],
                capture_output=True, text=True
            )
            if "WARN" in result.stdout or "CRITICAL" in result.stdout:
                drift_detected = True
                logger.info(f"Drift detected for model '{model}'")
        except Exception as exc:
            logger.warning(f"Drift check failed for {model}: {exc}")

    context["ti"].xcom_push(key="drift_detected", value=drift_detected)
    logger.info(f"Drift detected: {drift_detected}")
    return drift_detected   # ShortCircuitOperator will skip downstream if False


def retrain_thermal(**_) -> None:
    _run([
        _PYTHON, _THERMAL_TRAIN,
        "--epochs", "30",
        "--n-aug", "8",
        "--checkpoint-dir", str(_PROJECT_ROOT / "2_thermal_vision" / "checkpoints"),
    ])


def retrain_gnn(**_) -> None:
    _run([
        _PYTHON, _GNN_TRAIN,
        "--epochs", "40",
        "--n-snapshots", "300",
        "--checkpoint-dir", str(_PROJECT_ROOT / "3_st_gnn" / "checkpoints"),
    ])


def retrain_rl(**_) -> None:
    _run([
        _PYTHON, _RL_TRAIN,
        "--timesteps", "100000",
        "--checkpoint-dir", str(_PROJECT_ROOT / "4_rl_optimizer" / "checkpoints"),
    ])


def promote_model(model_key: str) -> callable:
    def _task(**_) -> None:
        _run([_PYTHON, _REGISTRY, "--promote", model_key])
    _task.__name__ = f"promote_{model_key}"
    return _task


def smoke_test(**_) -> None:
    """Hit the FastAPI /health endpoint to confirm new models are loaded."""
    import httpx  # type: ignore
    try:
        resp = httpx.get("http://localhost:8000/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Health check passed | models={data.get('models_loaded')}")
    except Exception as exc:
        logger.warning(f"Smoke test failed: {exc} — API may need restart")


def notify(**context) -> None:
    run_id    = context["run_id"]
    exec_date = context["execution_date"]
    logger.success(
        f"Weekly retrain DAG complete | run_id={run_id} | "
        f"exec_date={exec_date}"
    )
    # Optional Slack hook — set SLACK_WEBHOOK_URL in .env to enable
    import os
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if webhook:
        try:
            import httpx  # type: ignore
            httpx.post(webhook, json={
                "text": f"NammaClimaGrid weekly retrain complete ✓\n"
                        f"Run: {run_id}\n"
                        f"Date: {exec_date}"
            }, timeout=5)
        except Exception as exc:
            logger.warning(f"Slack notify failed: {exc}")


def save_baselines(**_) -> None:
    """Save updated baselines after successful retrain."""
    for model in ["thermal_vision", "st_gnn", "rl"]:
        try:
            _run([_PYTHON, _MONITORING, "--save-baseline", "--model", model])
        except Exception as exc:
            logger.warning(f"Baseline save failed for {model}: {exc}")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="ncg_weekly_retrain",
    description="NammaClimaGrid — weekly model retrain and promotion",
    schedule_interval="30 20 * * 0",   # Sunday 02:00 IST = 20:30 UTC Saturday
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "retrain"],
    max_active_runs=1,
) as dag:

    # 1. Drift gate — short-circuits if no drift detected
    t_drift = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=check_drift_task,
        provide_context=True,
        ignore_downstream_trigger_rules=True,
    )

    # 2. Parallel retrains
    t_retrain_thermal = PythonOperator(
        task_id="retrain_thermal_vision",
        python_callable=retrain_thermal,
    )
    t_retrain_gnn = PythonOperator(
        task_id="retrain_st_gnn",
        python_callable=retrain_gnn,
    )
    t_retrain_rl = PythonOperator(
        task_id="retrain_rl_optimizer",
        python_callable=retrain_rl,
    )

    # 3. Parallel promotions (after all retrains succeed)
    t_promote_thermal = PythonOperator(
        task_id="promote_thermal_vision",
        python_callable=promote_model("thermal_vision"),
    )
    t_promote_gnn = PythonOperator(
        task_id="promote_st_gnn",
        python_callable=promote_model("st_gnn"),
    )
    t_promote_rl = PythonOperator(
        task_id="promote_rl",
        python_callable=promote_model("rl"),
    )

    # 4. Save new baselines
    t_baselines = PythonOperator(
        task_id="save_baselines",
        python_callable=save_baselines,
    )

    # 5. API smoke test
    t_smoke = PythonOperator(
        task_id="api_smoke_test",
        python_callable=smoke_test,
    )

    # 6. Notify
    t_notify = PythonOperator(
        task_id="notify_completion",
        python_callable=notify,
        provide_context=True,
        trigger_rule="all_done",   # run even if some promotions failed
    )

    # DAG topology
    t_drift >> [t_retrain_thermal, t_retrain_gnn, t_retrain_rl]
    t_retrain_thermal >> t_promote_thermal
    t_retrain_gnn     >> t_promote_gnn
    t_retrain_rl      >> t_promote_rl
    [t_promote_thermal, t_promote_gnn, t_promote_rl] >> t_baselines
    t_baselines >> t_smoke >> t_notify
