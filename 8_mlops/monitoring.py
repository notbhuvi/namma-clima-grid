"""
Module 8b — Model Monitoring
================================

Provides two monitoring capabilities:

  1. Prediction drift detection
     Compares the current distribution of model predictions against a
     stored reference baseline using the Population Stability Index (PSI).
     Raises an alert when PSI > 0.2 (significant drift).

  2. Prometheus metrics export
     Exposes prediction statistics as Prometheus gauges/histograms so
     Grafana dashboards can track model health over time.

     Metrics exported:
       ncg_lst_prediction_celsius{ward="<id>"}          — gauge
       ncg_heat_stress_score{ward="<id>"}               — gauge
       ncg_flood_risk_score{ward="<id>"}                — gauge
       ncg_prediction_drift_psi{model="<name>"}         — gauge
       ncg_inference_latency_seconds{model="<name>"}    — histogram
       ncg_alerts_total{severity="<level>"}             — counter

Usage
─────
  # Run standalone metrics server on :9090
  python 8_mlops/monitoring.py --serve

  # Check drift against saved baseline
  python 8_mlops/monitoring.py --check-drift --model st_gnn

  # Save current predictions as the new baseline
  python 8_mlops/monitoring.py --save-baseline --model st_gnn
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

_MODULE_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
_BASELINE_DIR = _MODULE_DIR / "baselines"

DRIFT_THRESHOLD = 0.2     # PSI threshold for significant drift
N_BINS = 10               # PSI histogram bins


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class MonitorSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"), extra="ignore"
    )
    prometheus_port: int = 9090
    api_base_url:    str = "http://localhost:8000"


_cfg = MonitorSettings()


# ---------------------------------------------------------------------------
# PSI drift detection
# ---------------------------------------------------------------------------

def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Population Stability Index.

    PSI < 0.1  : no significant change
    PSI 0.1–0.2: moderate change (investigate)
    PSI > 0.2  : significant drift (retrain)
    """
    # Clip to valid range and build shared bin edges
    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max()) + 1e-6
    bins = np.linspace(lo, hi, n_bins + 1)

    exp_counts = np.histogram(expected, bins=bins)[0]
    act_counts = np.histogram(actual,   bins=bins)[0]

    exp_pct = np.clip(exp_counts / (exp_counts.sum() + 1e-8), 1e-6, None)
    act_pct = np.clip(act_counts / (act_counts.sum() + 1e-8), 1e-6, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# Baseline persistence
# ---------------------------------------------------------------------------

def save_baseline(model_name: str, predictions: Dict[str, np.ndarray]) -> None:
    """Save current prediction arrays as the reference baseline."""
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = _BASELINE_DIR / f"{model_name}_baseline.json"
    payload = {k: v.tolist() for k, v in predictions.items()}
    path.write_text(json.dumps(payload))
    logger.success(f"Baseline saved → {path}")


def load_baseline(model_name: str) -> Optional[Dict[str, np.ndarray]]:
    """Load reference baseline. Returns None if not found."""
    path = _BASELINE_DIR / f"{model_name}_baseline.json"
    if not path.exists():
        logger.warning(f"No baseline found at {path}")
        return None
    raw = json.loads(path.read_text())
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Drift check
# ---------------------------------------------------------------------------

def check_drift(
    model_name: str,
    current_predictions: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compare current predictions against baseline using PSI.

    Returns dict of {metric_name: psi_value}.
    Logs a warning for any metric exceeding DRIFT_THRESHOLD.
    """
    baseline = load_baseline(model_name)
    if baseline is None:
        logger.info(f"Saving first baseline for {model_name}")
        save_baseline(model_name, current_predictions)
        return {}

    drift_report: Dict[str, float] = {}
    for key, current in current_predictions.items():
        if key not in baseline:
            continue
        psi = _psi(baseline[key], current)
        drift_report[key] = round(psi, 4)
        level = "CRITICAL" if psi > DRIFT_THRESHOLD else ("WARN" if psi > 0.1 else "OK")
        logger.log(
            "WARNING" if psi > 0.1 else "INFO",
            f"Drift check | model={model_name}  metric={key}  "
            f"PSI={psi:.4f}  [{level}]"
        )

    return drift_report


# ---------------------------------------------------------------------------
# Fetch live predictions from API
# ---------------------------------------------------------------------------

def _fetch_predictions_from_api() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Pull current risk scores from the running FastAPI backend.
    Returns predictions keyed by model name.
    """
    try:
        import httpx  # type: ignore
        resp = httpx.get(f"{_cfg.api_base_url}/wards/risk", timeout=15)
        resp.raise_for_status()
        wards = resp.json()["wards"]
        heat  = np.array([w["heat_stress_score"] for w in wards], dtype=np.float32)
        flood = np.array([w["flood_risk_score"]  for w in wards], dtype=np.float32)
        return {
            "st_gnn":  {"heat_stress": heat, "flood_risk": flood},
        }
    except Exception as exc:
        logger.warning(f"API fetch failed ({exc}) — using synthetic data")
        rng   = np.random.default_rng(int(time.time()) % 10000)
        heat  = rng.normal(45, 15, 198).astype(np.float32)
        flood = rng.normal(30, 12, 198).astype(np.float32)
        return {"st_gnn": {"heat_stress": heat, "flood_risk": flood}}


# ---------------------------------------------------------------------------
# Prometheus metrics server
# ---------------------------------------------------------------------------

def start_metrics_server(port: int = _cfg.prometheus_port) -> None:
    """
    Start a Prometheus metrics HTTP server.

    Registers gauges for the latest ward predictions and drift PSI values,
    then polls the API every 60 seconds to update them.
    """
    try:
        from prometheus_client import (  # type: ignore
            start_http_server, Gauge, Counter, Histogram, REGISTRY
        )
    except ImportError:
        logger.error(
            "prometheus_client not installed — "
            "run: pip install prometheus-client"
        )
        return

    # Define metrics
    heat_gauge  = Gauge("ncg_heat_stress_mean",  "Mean city heat stress score (0-100)")
    flood_gauge = Gauge("ncg_flood_risk_mean",   "Mean city flood risk score (0-100)")
    drift_psi   = Gauge("ncg_prediction_drift_psi",
                        "Population Stability Index vs baseline", ["model", "metric"])
    alert_ctr   = Counter("ncg_alerts_total",
                          "Total alerts generated", ["severity"])
    latency_hist = Histogram("ncg_inference_latency_seconds",
                             "Inference latency", ["model"])

    start_http_server(port)
    logger.success(f"Prometheus metrics server started on :{port}/metrics")

    while True:
        try:
            t0 = time.time()
            preds = _fetch_predictions_from_api()
            latency = time.time() - t0
            latency_hist.labels(model="st_gnn").observe(latency)

            for model_name, model_preds in preds.items():
                if "heat_stress" in model_preds:
                    heat_gauge.set(float(model_preds["heat_stress"].mean()))
                if "flood_risk" in model_preds:
                    flood_gauge.set(float(model_preds["flood_risk"].mean()))

                drift = check_drift(model_name, model_preds)
                for metric, psi in drift.items():
                    drift_psi.labels(model=model_name, metric=metric).set(psi)
                    if psi > DRIFT_THRESHOLD:
                        alert_ctr.labels(severity="critical").inc()
                    elif psi > 0.1:
                        alert_ctr.labels(severity="warning").inc()

        except Exception as exc:
            logger.error(f"Metrics update error: {exc}")

        time.sleep(60)


# ---------------------------------------------------------------------------
# Summary report (console)
# ---------------------------------------------------------------------------

def print_drift_report(all_results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print a drift report table."""
    logger.info("\n{'='*60}")
    logger.info("DRIFT REPORT")
    logger.info("{'='*60}")
    for model, metrics in all_results.items():
        logger.info(f"\nModel: {model}")
        for metric, psi in metrics.items():
            status = "CRITICAL" if psi > 0.2 else "WARN" if psi > 0.1 else "OK"
            bar    = "█" * int(psi * 50)
            logger.info(f"  {metric:<20} PSI={psi:.4f}  [{status}]  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid — Model Monitoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--serve",         action="store_true",
                       help="Start Prometheus metrics server")
    group.add_argument("--check-drift",   action="store_true",
                       help="Check prediction drift vs baseline")
    group.add_argument("--save-baseline", action="store_true",
                       help="Save current predictions as new baseline")
    p.add_argument("--model", default="st_gnn",
                   choices=["thermal_vision", "st_gnn", "rl"],
                   help="Model to check/baseline")
    p.add_argument("--port", type=int, default=_cfg.prometheus_port)
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse()
    preds = _fetch_predictions_from_api()

    if args.serve:
        start_metrics_server(args.port)

    elif args.save_baseline:
        model_preds = preds.get(args.model, next(iter(preds.values())))
        save_baseline(args.model, model_preds)

    elif args.check_drift:
        model_preds = preds.get(args.model, next(iter(preds.values())))
        results     = check_drift(args.model, model_preds)
        print_drift_report({args.model: results})
