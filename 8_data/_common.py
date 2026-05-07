"""
Shared helpers for Module 7 (mock data generators).

Centralises:
  * .env loading
  * logger configuration
  * Bengaluru bounding box
  * a deterministic list of 198 synthetic BBMP-like ward IDs + names
  * a small built-in geojson fallback when the real shapefile isn't present
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODULE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = MODULE_DIR.parent
ENV_FILE: Path = PROJECT_ROOT / ".env"

# Load .env once on import (harmless if missing).
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(
    lambda m: print(m, end=""),
    level=_LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
)

# ---------------------------------------------------------------------------
# Bengaluru bounding box (from .env with safe defaults)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.west + self.east) / 2.0, (self.south + self.north) / 2.0)


BENGALURU_BBOX = BBox(
    west=float(os.getenv("BENGALURU_BBOX_WEST", 77.45)),
    south=float(os.getenv("BENGALURU_BBOX_SOUTH", 12.83)),
    east=float(os.getenv("BENGALURU_BBOX_EAST", 77.78)),
    north=float(os.getenv("BENGALURU_BBOX_NORTH", 13.14)),
)

# Number of BBMP wards.
NUM_WARDS: int = 198

# ---------------------------------------------------------------------------
# Synthetic ward catalogue
# ---------------------------------------------------------------------------
# A handful of real BBMP ward names — repeated/suffixed to fill 198 slots so
# the system has stable, human-readable identifiers without requiring the full
# official list (which is distributed as a separate GeoJSON asset).
_REAL_WARD_NAME_SEEDS: List[str] = [
    "Kempegowda", "Chickpet", "Shivajinagar", "Ulsoor", "Malleshwaram",
    "Rajajinagar", "Basavanagudi", "Jayanagar", "BTM Layout", "Koramangala",
    "HSR Layout", "Bommanahalli", "Bellandur", "Marathahalli", "Whitefield",
    "KR Puram", "Hebbal", "Yelahanka", "Sahakar Nagar", "Vidyaranyapura",
    "Peenya", "Mahalakshmipuram", "Vijayanagar", "Govindraj Nagar",
    "Nagarbhavi", "Kengeri", "RR Nagar", "Padmanabhanagar", "Banashankari",
    "JP Nagar", "Hongasandra", "Begur", "Singasandra", "Bilekahalli",
]


def synthetic_ward_catalogue() -> List[Dict[str, object]]:
    """
    Return 198 ward records with stable IDs and pseudo-real names.

    Each record: {ward_id: int, ward_code: str, ward_name: str, centroid: (lon,lat)}
    Centroids are placed on a ~14x15 regular grid inside the Bengaluru bbox so
    downstream code has deterministic coordinates even without the real shapefile.
    """
    import math

    records: List[Dict[str, object]] = []
    cols = 14
    rows = math.ceil(NUM_WARDS / cols)  # 15 rows × 14 cols = 210 >= 198
    dx = (BENGALURU_BBOX.east - BENGALURU_BBOX.west) / cols
    dy = (BENGALURU_BBOX.north - BENGALURU_BBOX.south) / rows

    for idx in range(NUM_WARDS):
        r, c = divmod(idx, cols)
        lon = BENGALURU_BBOX.west + (c + 0.5) * dx
        lat = BENGALURU_BBOX.south + (r + 0.5) * dy
        seed_name = _REAL_WARD_NAME_SEEDS[idx % len(_REAL_WARD_NAME_SEEDS)]
        suffix_num = (idx // len(_REAL_WARD_NAME_SEEDS)) + 1
        ward_name = seed_name if suffix_num == 1 else f"{seed_name} {suffix_num}"
        records.append({
            "ward_id": idx + 1,
            "ward_code": f"BBMP-{idx + 1:03d}",
            "ward_name": ward_name,
            "centroid": (round(lon, 6), round(lat, 6)),
            "grid_row": r,
            "grid_col": c,
        })
    return records


def synthetic_ward_geojson() -> Dict[str, object]:
    """
    Build a GeoJSON FeatureCollection of 198 rectangular ward polygons tiling
    the Bengaluru bbox. Used as a fallback when the real BBMP shapefile is
    not present on disk.
    """
    import math

    cols = 14
    rows = math.ceil(NUM_WARDS / cols)
    dx = (BENGALURU_BBOX.east - BENGALURU_BBOX.west) / cols
    dy = (BENGALURU_BBOX.north - BENGALURU_BBOX.south) / rows

    features: List[Dict[str, object]] = []
    for rec in synthetic_ward_catalogue():
        r, c = rec["grid_row"], rec["grid_col"]
        w = BENGALURU_BBOX.west + c * dx
        e = w + dx
        s = BENGALURU_BBOX.south + r * dy
        n = s + dy
        polygon = [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
        features.append({
            "type": "Feature",
            "properties": {
                "ward_id": rec["ward_id"],
                "ward_code": rec["ward_code"],
                "ward_name": rec["ward_name"],
                "centroid_lon": rec["centroid"][0],
                "centroid_lat": rec["centroid"][1],
            },
            "geometry": {"type": "Polygon", "coordinates": polygon},
        })
    return {"type": "FeatureCollection", "features": features}


def ensure_synthetic_geojson(path: Path) -> Path:
    """Write the synthetic wards geojson if it doesn't already exist."""
    if path.exists():
        logger.debug(f"GeoJSON already present at {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(synthetic_ward_geojson()))
    logger.info(f"Wrote synthetic ward geojson -> {path}")
    return path
