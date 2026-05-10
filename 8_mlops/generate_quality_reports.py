#!/usr/bin/env python3
"""Generate data quality and model checkpoint evaluation reports."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "8_data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

RANGES = {
    "ward_id": (1, 198),
    "centroid_lat": (12.77, 13.18),
    "centroid_lon": (77.35, 77.82),
    "lst_celsius": (15, 60),
    "ndvi": (0, 1),
    "impervious_pct": (0, 100),
    "rainfall_mm_24h": (0, 500),
    "pm25_ugm3": (0, 500),
    "population_density": (0, 100000),
    "heat_stress_score": (0, 100),
    "flood_risk_score": (0, 100),
}

CHECKPOINTS = {
    "thermal_vision": ROOT / "2_thermal_vision" / "checkpoints" / "best_model.pt",
    "st_gnn": ROOT / "3_st_gnn" / "checkpoints" / "gnn_best.pt",
    "rl_optimizer": ROOT / "4_rl_optimizer" / "checkpoints" / "ppo_best.pt",
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def load_features(seed: int, monsoon: bool, input_path: Path | None = None):
    if input_path is not None:
        import pandas as pd

        if input_path.suffix.lower() == ".csv":
            return pd.read_csv(input_path)
        if input_path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(input_path)
        raise ValueError("--input must be a .csv or .parquet file")

    from mock_ward_features import generate_ward_features

    return generate_ward_features(seed=seed, monsoon=monsoon)


def data_quality(df) -> dict[str, Any]:
    issues: list[str] = []
    null_counts = {col: int(count) for col, count in df.isna().sum().items() if count}

    if len(df) != 198:
        issues.append(f"Expected 198 wards, found {len(df)}")
    if "ward_id" in df and df["ward_id"].duplicated().any():
        issues.append("ward_id contains duplicates")
    if set(range(1, 199)) - set(df["ward_id"].astype(int).tolist()):
        issues.append("ward_id sequence is incomplete")
    if null_counts:
        issues.append(f"Null values present: {null_counts}")

    range_violations: dict[str, int] = {}
    for col, (lo, hi) in RANGES.items():
        if col not in df:
            issues.append(f"Missing required column: {col}")
            continue
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if not bad.empty:
            range_violations[col] = int(len(bad))
            issues.append(f"{col} has {len(bad)} value(s) outside [{lo}, {hi}]")

    correlations = {
        "heat_vs_lst": float(df["heat_stress_score"].corr(df["lst_celsius"])),
        "flood_vs_rain": float(df["flood_risk_score"].corr(df["rainfall_mm_24h"])),
        "heat_vs_impervious": float(df["heat_stress_score"].corr(df["impervious_pct"])),
    }

    return {
        "status": "pass" if not issues else "fail",
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "null_counts": null_counts,
        "range_violations": range_violations,
        "correlations": {k: round(v, 4) if not math.isnan(v) else None for k, v in correlations.items()},
        "issues": issues,
        "summary_stats": {
            col: {
                "min": float(df[col].min()),
                "mean": float(df[col].mean()),
                "max": float(df[col].max()),
            }
            for col in [
                "lst_celsius",
                "ndvi",
                "impervious_pct",
                "rainfall_mm_24h",
                "heat_stress_score",
                "flood_risk_score",
            ]
            if col in df
        },
    }


def model_report() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "status": "warn",
            "issues": [f"torch unavailable; checkpoint metadata not inspected: {exc}"],
            "models": {},
        }

    issues: list[str] = []
    models: dict[str, Any] = {}
    for name, path in CHECKPOINTS.items():
        entry: dict[str, Any] = {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else 0,
            "metadata": {},
        }
        if not path.exists():
            issues.append(f"{name} checkpoint missing")
            models[name] = entry
            continue
        try:
            ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict):
                metadata_keys = [
                    "epoch",
                    "val_rmse",
                    "combined_RMSE",
                    "ep_reward_mean",
                    "best_metric",
                    "metrics",
                    "config",
                ]
                entry["metadata"] = {
                    key: _to_jsonable(ckpt[key])
                    for key in metadata_keys
                    if key in ckpt
                }
        except Exception as exc:
            issues.append(f"{name} checkpoint could not be inspected: {exc}")
        models[name] = entry

    return {
        "status": "pass" if not issues else "warn",
        "models": models,
        "issues": issues,
    }


def write_reports(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_report.json").write_text(json.dumps(payload, indent=2, default=str))

    dq = payload["data_quality"]
    mr = payload["model_evaluation"]
    lines = [
        "# NammaClimaGrid Quality Report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Data Quality",
        "",
        f"- Status: {dq['status']}",
        f"- Rows: {dq['row_count']}",
        f"- Columns: {dq['column_count']}",
        f"- Issues: {len(dq['issues'])}",
        "",
        "## Model Checkpoints",
        "",
        f"- Status: {mr['status']}",
    ]
    for name, model in mr["models"].items():
        lines.append(
            f"- {name}: exists={model['exists']} size_mb={model['size_mb']} "
            f"metadata={model['metadata']}"
        )
    if dq["issues"] or mr["issues"]:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in dq["issues"] + mr["issues"])
    else:
        lines.extend(["", "No blocking data/model quality issues detected in this run."])

    (out_dir / "quality_report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "industry_readiness")
    parser.add_argument("--input", type=Path, help="Optional real ward feature CSV/Parquet to validate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monsoon", action="store_true")
    args = parser.parse_args()

    df = load_features(seed=args.seed, monsoon=args.monsoon, input_path=args.input)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.input) if args.input else "synthetic_ward_features",
        "data_quality": data_quality(df),
        "model_evaluation": model_report(),
    }
    write_reports(payload, args.out)
    print(f"Wrote {args.out / 'quality_report.md'}")
    print(f"Wrote {args.out / 'quality_report.json'}")
    return 0 if payload["data_quality"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
