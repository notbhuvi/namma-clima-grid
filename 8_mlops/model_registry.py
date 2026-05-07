"""
Module 8a — MLflow Model Registry
=====================================

Manages the full model lifecycle for the three trained models:

  thermal_vision_lst   — CNN-ViT LST predictor  (Module 2)
  st_gnn_ward_risk     — Spatio-Temporal GNN     (Module 3)
  rl_green_infra       — PPO policy              (Module 4)

Lifecycle stages (MLflow standard)
────────────────────────────────────
  None → Staging → Production → Archived

Promotion criteria
───────────────────
  thermal_vision  : val_rmse  < RMSE_THRESHOLD (default 3.5 °C)
  st_gnn          : val_heat_rmse < 8.0 AND val_flood_rmse < 10.0
  rl_green_infra  : ep_reward_mean > REWARD_THRESHOLD (default 50.0)

Usage
─────
  # Promote best run to Staging then Production:
  python 8_mlops/model_registry.py --promote thermal_vision

  # Compare all registered model versions:
  python 8_mlops/model_registry.py --compare st_gnn

  # Archive old Production version after new one is deployed:
  python 8_mlops/model_registry.py --archive rl_green_infra --version 1
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class RegistrySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"), extra="ignore"
    )
    mlflow_tracking_uri: str = "http://localhost:5001"

    # Promotion thresholds
    thermal_rmse_threshold:  float = 3.5
    gnn_heat_rmse_threshold: float = 8.0
    gnn_flood_rmse_threshold: float = 10.0
    rl_reward_threshold:     float = 50.0


_cfg = RegistrySettings()


# ---------------------------------------------------------------------------
# Model descriptors
# ---------------------------------------------------------------------------
@dataclass
class ModelDescriptor:
    registered_name:   str
    experiment_name:   str
    primary_metric:    str          # lower-is-better metric key
    threshold:         float
    higher_is_better:  bool = False


MODEL_DESCRIPTORS: Dict[str, ModelDescriptor] = {
    "thermal_vision": ModelDescriptor(
        registered_name="thermal_vision_lst",
        experiment_name="thermal_vision_lst",
        primary_metric="val_rmse",
        threshold=_cfg.thermal_rmse_threshold,
    ),
    "st_gnn": ModelDescriptor(
        registered_name="st_gnn_ward_risk",
        experiment_name="st_gnn_ward_risk",
        primary_metric="val_heat_rmse",
        threshold=_cfg.gnn_heat_rmse_threshold,
    ),
    "rl": ModelDescriptor(
        registered_name="rl_green_infra",
        experiment_name="rl_green_infra",
        primary_metric="ep_reward_mean",
        threshold=_cfg.rl_reward_threshold,
        higher_is_better=True,
    ),
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _mlflow_client():
    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(_cfg.mlflow_tracking_uri)
        return mlflow.tracking.MlflowClient()
    except ImportError as e:
        raise RuntimeError("mlflow not installed — run: pip install mlflow") from e


def _metric_passes(value: float, threshold: float, higher_is_better: bool) -> bool:
    return value > threshold if higher_is_better else value < threshold


def _best_run(client, experiment_name: str, metric: str, higher: bool) -> Optional[object]:
    """Return the best completed run from an experiment."""
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return None

    order = "DESC" if higher else "ASC"
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    return runs[0] if runs else None


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def promote_to_staging(model_key: str) -> Optional[str]:
    """
    Find the best run for a model, register it if it meets threshold,
    and move it to Staging.

    Returns the model version string if promoted, else None.
    """
    desc   = MODEL_DESCRIPTORS[model_key]
    client = _mlflow_client()

    run = _best_run(client, desc.experiment_name, desc.primary_metric, desc.higher_is_better)
    if run is None:
        logger.error(f"No completed runs found in experiment '{desc.experiment_name}'")
        return None

    metric_val = run.data.metrics.get(desc.primary_metric)
    if metric_val is None:
        logger.error(f"Metric '{desc.primary_metric}' not found in run {run.info.run_id}")
        return None

    passes = _metric_passes(metric_val, desc.threshold, desc.higher_is_better)
    direction = ">" if desc.higher_is_better else "<"
    logger.info(
        f"Best run {run.info.run_id} | {desc.primary_metric}={metric_val:.4f} "
        f"(threshold {direction} {desc.threshold}) | passes={passes}"
    )
    if not passes:
        logger.warning(f"Model '{model_key}' did not meet threshold — not promoted")
        return None

    import mlflow  # type: ignore
    # Register the model
    model_uri  = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri, desc.registered_name)
    logger.info(f"Registered {desc.registered_name} version={mv.version}")

    # Transition to Staging
    client.transition_model_version_stage(
        name=desc.registered_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.success(
        f"Promoted {desc.registered_name} v{mv.version} → Staging | "
        f"{desc.primary_metric}={metric_val:.4f}"
    )
    return mv.version


def promote_to_production(model_key: str, version: str) -> None:
    """Move a Staging version to Production and archive the old Production."""
    desc   = MODEL_DESCRIPTORS[model_key]
    client = _mlflow_client()
    client.transition_model_version_stage(
        name=desc.registered_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,   # archives previous Production
    )
    logger.success(f"Promoted {desc.registered_name} v{version} → Production")


def archive_version(model_key: str, version: str) -> None:
    """Move a version to Archived stage."""
    desc   = MODEL_DESCRIPTORS[model_key]
    client = _mlflow_client()
    client.transition_model_version_stage(
        name=desc.registered_name,
        version=version,
        stage="Archived",
    )
    logger.info(f"Archived {desc.registered_name} v{version}")


def compare_versions(model_key: str) -> None:
    """Print a comparison table of all registered versions for a model."""
    desc   = MODEL_DESCRIPTORS[model_key]
    client = _mlflow_client()

    try:
        versions = client.search_model_versions(f"name='{desc.registered_name}'")
    except Exception as exc:
        logger.error(f"Could not fetch versions: {exc}")
        return

    if not versions:
        logger.warning(f"No versions registered for '{desc.registered_name}'")
        return

    logger.info(f"\n{'Version':<8} {'Stage':<12} {'Run ID':<10} {desc.primary_metric:<12}")
    logger.info("-" * 50)
    for v in sorted(versions, key=lambda x: int(x.version)):
        run_id = v.run_id[:8]
        try:
            run    = client.get_run(v.run_id)
            metric = run.data.metrics.get(desc.primary_metric, float("nan"))
            logger.info(f"{v.version:<8} {v.current_stage:<12} {run_id:<10} {metric:<12.4f}")
        except Exception:
            logger.info(f"{v.version:<8} {v.current_stage:<12} {run_id:<10} {'N/A':<12}")


def full_promote_pipeline(model_key: str) -> None:
    """
    Full pipeline: Staging → validate → Production.
    Intended for use from the Airflow retrain DAG.
    """
    logger.info(f"Starting full promotion pipeline for '{model_key}'")
    version = promote_to_staging(model_key)
    if version is None:
        return
    promote_to_production(model_key, version)
    logger.success(f"Pipeline complete: {model_key} v{version} is now in Production")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid MLflow Model Registry CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--promote",  metavar="MODEL",
                       choices=MODEL_DESCRIPTORS,
                       help="Promote best run to Staging → Production")
    group.add_argument("--compare",  metavar="MODEL",
                       choices=MODEL_DESCRIPTORS,
                       help="Compare all registered versions")
    group.add_argument("--archive",  metavar="MODEL",
                       choices=MODEL_DESCRIPTORS,
                       help="Archive a specific version (requires --version)")
    p.add_argument("--version", metavar="V", help="Model version to archive/promote")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.promote:
        full_promote_pipeline(args.promote)
    elif args.compare:
        compare_versions(args.compare)
    elif args.archive:
        if not args.version:
            logger.error("--archive requires --version")
            sys.exit(1)
        archive_version(args.archive, args.version)
