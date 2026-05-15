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
# Real BBMP 198-ward names from the 2009-2023 ward list. Geometry is still a
# synthetic grid fallback unless an official ward GeoJSON is supplied, but names
# should never look like "Banashankari 4" in demos.
BBMP_2009_2023_WARD_NAMES: Dict[int, str] = {
    1: "Kempegowda Ward",
    2: "Chowdeshwari Ward",
    3: "Attur",
    4: "Yelahanka Satellite Town",
    5: "Jakkur",
    6: "Thanisandra",
    7: "Byatarayanapura",
    8: "Kodigehalli",
    9: "Vidyaranyapura",
    10: "Doddabommasandra",
    11: "Kuvempunagar",
    12: "Shettyhalli",
    13: "Mallasandra",
    14: "Bagalagunte",
    15: "T. Dasarahalli",
    16: "Jalahalli",
    17: "J. P. Park",
    18: "Radhakrishna Temple",
    19: "Sanjaynagar",
    20: "Ganganagar",
    21: "Hebbala",
    22: "Vishwanatha Nagenahalli",
    23: "Nagavara",
    24: "HBR Layout",
    25: "Horamavu",
    26: "Ramamurthynagar",
    27: "Banaswadi",
    28: "Kammanahalli",
    29: "Kacharakanahalli",
    30: "Kadugondanahalli",
    31: "Kushalnagar",
    32: "Kavalbyrasandra",
    33: "Manorayanapalya",
    34: "Gangenahalli",
    35: "Aramanenagar",
    36: "Mathikere",
    37: "Yeshwanthpura",
    38: "H. M. T.",
    39: "Chokkasandra",
    40: "Dodda Bidarkallu",
    41: "Peenya Industrial Area",
    42: "Lakshmidevinagar",
    43: "Nandini Layout",
    44: "Marappana Palya",
    45: "Malleshwaram",
    46: "Jayachamarajendranagar",
    47: "Devarajeevanahalli",
    48: "Muneshwaranagar",
    49: "Lingarajapuram",
    50: "Benniganahalli",
    51: "Vijinapura",
    52: "Krishnarajapuram",
    53: "Basavanapura",
    54: "Hoodi",
    55: "Devasandra",
    56: "A. Narayanapura",
    57: "C. V. Raman Nagar",
    58: "Hosathippasandra",
    59: "Maruthisevanagar",
    60: "Sagayarapuram",
    61: "S. K. Garden",
    62: "Ramaswamypalya",
    63: "Jayamahal",
    64: "Rajamahal",
    65: "Kadumalleshwara",
    66: "Subrahmanyanagar",
    67: "Nagapura",
    68: "Mahalakshmipuram",
    69: "Laggere",
    70: "Rajagopalanagar",
    71: "Hegganahalli",
    72: "Herohalli",
    73: "Kottigepalya",
    74: "Shakthiganapathinagar",
    75: "Shankara Matha",
    76: "Gayathrinagar",
    77: "Dattathreya Temple",
    78: "Pulakeshinagar",
    79: "Sarvagnanagar",
    80: "Hoysalanagar",
    81: "Vignananagar",
    82: "Garudacharpalya",
    83: "Kadugudi",
    84: "Hagadooru",
    85: "Doddanekkundi",
    86: "Marathahalli",
    87: "HAL Airport",
    88: "Jeevanabima Nagar",
    89: "Jogupalya",
    90: "Ulsoor",
    91: "Bharathinagar",
    92: "Shivajinagar",
    93: "Vasanthnagar",
    94: "Gandhinagar",
    95: "Subhashnagar",
    96: "Okalipuram",
    97: "Dayanandanagar",
    98: "Prakashnagar",
    99: "Rajajinagar",
    100: "Basaveshwaranagar",
    101: "Kamakshipalya",
    102: "Vrishabhavathi",
    103: "Kaveripura",
    104: "Govindarajanagar",
    105: "Agrahara Dasarahalli",
    106: "Dr. Rajkumar",
    107: "Shivanagar",
    108: "Srirama Mandir",
    109: "Chickpete",
    110: "Sampangiramanagar",
    111: "Shanthalanagar",
    112: "Domlur",
    113: "Konena Agrahara",
    114: "Agaram",
    115: "Vannarpet",
    116: "Neelasandra",
    117: "Shanthinagar",
    118: "Sudhamanagar",
    119: "Dharmarayaswamy Temple",
    120: "Cottonpet",
    121: "Binnipete",
    122: "Kempapura Agrahara",
    123: "Vijaynagar",
    124: "Hosahalli",
    125: "Marenahalli",
    126: "Maruthi Mandira",
    127: "Moodalapalya",
    128: "Nagarabhavi",
    129: "Jnanabharathi",
    130: "Ullalu",
    131: "Nayandahalli",
    132: "Attiguppe",
    133: "Hampinagar",
    134: "Bapujinagar",
    135: "Padarayanapura",
    136: "Jagjivanram Nagar",
    137: "Rayapuram",
    138: "Chalavadipalya",
    139: "Krishnarajendra Market",
    140: "Chamarajapet",
    141: "Azadnagar",
    142: "Sunkenahalli",
    143: "Visvesvarapuram",
    144: "Siddapura",
    145: "Hombegowdanagar",
    146: "Lakkasandra",
    147: "Adugodi",
    148: "Ejipura",
    149: "Varthur",
    150: "Bellandur",
    151: "Koramangala",
    152: "Suddaguntepalya",
    153: "Jayanagar",
    154: "Basavanagudi",
    155: "Hanumanthanagar",
    156: "Srinagar",
    157: "Gali Anjaneya Temple",
    158: "Deepanjalinagar",
    159: "Kengeri",
    160: "Rajarajeshwarinagar",
    161: "Hosakerehalli",
    162: "Girinagar",
    163: "Kathriguppe",
    164: "Vidyapeetha",
    165: "Ganesh Mandira",
    166: "Karisandra",
    167: "Yediyur",
    168: "Pattabhiramnagar",
    169: "Byrasandra",
    170: "Jayanagar East",
    171: "Gurappanapalya",
    172: "Madiwala",
    173: "Jakkasandra",
    174: "HSR Layout",
    175: "Bommanahalli",
    176: "BTM Layout",
    177: "J. P. Nagar",
    178: "Sarakki",
    179: "Shakambarinagar",
    180: "Banashankari Temple",
    181: "Kumaraswamy Layout",
    182: "Padmanabhanagar",
    183: "Chikkalasandra",
    184: "Uttarahalli",
    185: "Yelachenahalli",
    186: "Jaraganahalli",
    187: "Puttenahalli",
    188: "Bilekahalli",
    189: "Hongasandra",
    190: "Mangammanapalya",
    191: "Singasandra",
    192: "Begur",
    193: "Arakere",
    194: "Gottigere",
    195: "Konanakunte",
    196: "Anjanapura",
    197: "Vasanthapura",
    198: "Hemmigepura",
}


def synthetic_ward_catalogue() -> List[Dict[str, object]]:
    """
    Return 198 ward records with stable IDs and real BBMP ward names.

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
        ward_name = BBMP_2009_2023_WARD_NAMES.get(idx + 1, f"Ward {idx + 1}")
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
