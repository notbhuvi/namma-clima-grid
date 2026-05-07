"""
Helper — Writes 8_data/bengaluru_wards.geojson.

If the real BBMP ward boundary file is placed alongside as
`bengaluru_wards_real.geojson`, this script will copy it into place.
Otherwise it emits a synthetic 198-cell grid covering the Bengaluru bbox
so that Module 3 (graph builder) and Module 5 (API geojson endpoint) can
run without the official dataset.

Usage:
    python 8_data/generate_bengaluru_wards_geojson.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from _common import MODULE_DIR, ensure_synthetic_geojson, logger

TARGET = MODULE_DIR / "bengaluru_wards.geojson"
REAL_SRC = MODULE_DIR / "bengaluru_wards_real.geojson"


def main() -> None:
    if REAL_SRC.exists():
        shutil.copyfile(REAL_SRC, TARGET)
        feature_count = len(json.loads(TARGET.read_text()).get("features", []))
        logger.success(f"Copied real BBMP ward boundaries → {TARGET} ({feature_count} features)")
        return

    logger.info("Real BBMP shapefile not found — generating synthetic 198-cell grid.")
    ensure_synthetic_geojson(TARGET)
    feature_count = len(json.loads(TARGET.read_text()).get("features", []))
    logger.success(f"Wrote synthetic ward geojson → {TARGET} ({feature_count} features)")


if __name__ == "__main__":
    main()
