"""
Flood Image Classifier  (with adaptive feedback learning)
===========================================================
Analyses an uploaded image and predicts whether it shows flooding.

Feedback loop
─────────────
Every time BBMP marks a report as "fake" or "confirmed real", that
signal is written to  static/feedback/feedback_log.json.

On each classification call the module reads accumulated feedback and
derives a  bias_adjustment  (shift on the raw flood score) so that
repeated false-positives push the threshold up and missed floods push
it down.  The adjustment is recalculated automatically — no manual
retraining step needed.

Returns:
  {
    "is_flood":    bool,
    "confidence":  float,   # 0.0 – 1.0
    "label":       str,     # "flood" | "no_flood"
    "details":     dict,
  }
"""
from __future__ import annotations

import io
import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, Any, List

# ── Paths ──────────────────────────────────────────────────────────────────
_BASE_DIR      = os.path.dirname(__file__)
_FEEDBACK_DIR  = os.path.join(_BASE_DIR, "static", "feedback")
_FEEDBACK_FILE = os.path.join(_FEEDBACK_DIR, "feedback_log.json")
os.makedirs(_FEEDBACK_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Feedback store (persisted JSON)
# ---------------------------------------------------------------------------

def _load_feedback() -> List[Dict]:
    if not os.path.exists(_FEEDBACK_FILE):
        return []
    try:
        with open(_FEEDBACK_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_feedback(entries: List[Dict]) -> None:
    with open(_FEEDBACK_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def record_feedback(report_id: int, label: str, image_path: str | None = None) -> Dict:
    """
    Record BBMP feedback for a report.

    label : "fake"  → was NOT a flood (false positive)
            "real"  → confirmed flood (true positive)
            "clear" → remove previous flag
    """
    entries = _load_feedback()

    # Remove any prior entry for this report_id
    entries = [e for e in entries if e.get("report_id") != report_id]

    if label != "clear":
        entries.append({
            "report_id":  report_id,
            "label":      label,        # "fake" | "real"
            "image_path": image_path,
            "flagged_at": datetime.now(timezone.utc).isoformat(),
        })

    _save_feedback(entries)

    # Return updated stats
    stats = _compute_stats(entries)
    return {
        "recorded":    True,
        "report_id":   report_id,
        "label":       label,
        "total_feedback": len(entries),
        "model_bias":  stats["bias"],
        "fake_count":  stats["fake_count"],
        "real_count":  stats["real_count"],
    }


def _compute_stats(entries: List[Dict]) -> Dict:
    """
    Derive a bias_adjustment from accumulated feedback.

    Logic:
      • Each "fake" record means classifier over-predicted → bias DOWN
      • Each "real" record means classifier correct (or under-predicted) → bias UP (slight)
      • Recent feedback is weighted 2× more than older entries
      • Bias is clamped to [-0.25, +0.15]
    """
    if not entries:
        return {"bias": 0.0, "fake_count": 0, "real_count": 0}

    now = datetime.now(timezone.utc)
    fake_w = 0.0
    real_w = 0.0

    for e in entries:
        try:
            age_days = (now - datetime.fromisoformat(e["flagged_at"])).days
        except Exception:
            age_days = 30
        weight = 2.0 if age_days < 3 else 1.0

        if e["label"] == "fake":
            fake_w += weight
        elif e["label"] == "real":
            real_w += weight

    total_w = fake_w + real_w
    if total_w == 0:
        return {"bias": 0.0, "fake_count": 0, "real_count": 0}

    # Net bias: fakes pull down, reals push up slightly
    # Each fake = -0.03, each real = +0.01  (asymmetric — we prefer caution)
    raw_bias = (real_w * 0.01 - fake_w * 0.03)
    bias = max(-0.25, min(0.15, raw_bias))

    return {
        "bias":       round(bias, 4),
        "fake_count": sum(1 for e in entries if e["label"] == "fake"),
        "real_count": sum(1 for e in entries if e["label"] == "real"),
    }


def get_feedback_stats() -> Dict:
    """Return current feedback stats for the BBMP dashboard."""
    entries = _load_feedback()
    stats = _compute_stats(entries)
    stats["total_feedback"] = len(entries)
    stats["entries"] = entries[-20:]   # last 20 for display
    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Classify image bytes as flood / no_flood.
    Applies learned bias from BBMP feedback automatically.
    Never raises — returns error dict on failure.
    """
    try:
        return _analyse(image_bytes)
    except Exception as exc:
        return {
            "is_flood":   False,
            "confidence": 0.0,
            "label":      "error",
            "details":    {"error": str(exc)},
        }


# ---------------------------------------------------------------------------
# Internal analysis
# ---------------------------------------------------------------------------

def _analyse(image_bytes: bytes) -> Dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {
            "is_flood":   False,
            "confidence": 0.0,
            "label":      "no_flood",
            "details":    {"note": "PIL unavailable, defaulting to no_flood"},
        }

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)

    width, height = img.size
    pixels = list(img.getdata())

    # Focus on bottom 60% of image (ground/water level)
    lower_start  = int(height * 0.4) * width
    lower_pixels = pixels[lower_start:]

    # ── Colour-feature scoring ────────────────────────────────────────────
    water_score = mud_score = dark_score = green_score = dry_score = 0.0

    n = len(lower_pixels)
    for r, g, b in lower_pixels:
        brightness = (r + g + b) / 3.0

        # Water: blue dominant, moderate brightness
        if b > r + 15 and b > g + 5 and 40 < brightness < 180:
            water_score += 1
        # Murky flood water: blue-grey with brown tint
        if abs(r - g) < 30 and abs(g - b) < 30 and brightness < 130:
            water_score += 0.6
        # Muddy / brown flood water
        if r > g > b and r > 80 and b < 100 and (r - b) > 30:
            mud_score += 1
        # Very dark (deep water, wet asphalt)
        if brightness < 50:
            dark_score += 0.5
        # Vegetation (green dominant — reduces flood likelihood)
        if g > r + 20 and g > b + 20:
            green_score += 1
        # Dry / warm (sandy, dry road)
        if r > 160 and g > 130 and b < 110:
            dry_score += 0.5

    ws  = water_score / n
    ms  = mud_score   / n
    ds  = dark_score  / n
    gs  = green_score / n
    dry = dry_score   / n

    # ── Composite flood score ─────────────────────────────────────────────
    raw_flood = (
        ws  * 0.45 +
        ms  * 0.30 +
        ds  * 0.10 -
        gs  * 0.15 -
        dry * 0.10
    )
    raw_flood = max(0.0, min(1.0, raw_flood))

    # Apply adaptive bias from BBMP feedback
    feedback_entries = _load_feedback()
    stats = _compute_stats(feedback_entries)
    bias  = stats["bias"]

    adjusted_flood = max(0.0, min(1.0, raw_flood + bias))

    # Sigmoid sharpening
    confidence = _sigmoid(adjusted_flood * 10 - 2.5)
    is_flood   = confidence >= 0.52

    return {
        "is_flood":   is_flood,
        "confidence": round(confidence, 3),
        "label":      "flood" if is_flood else "no_flood",
        "details": {
            "water_px_pct":    round(ws  * 100, 1),
            "mud_px_pct":      round(ms  * 100, 1),
            "dark_px_pct":     round(ds  * 100, 1),
            "green_px_pct":    round(gs  * 100, 1),
            "dry_px_pct":      round(dry * 100, 1),
            "raw_flood_score": round(raw_flood, 3),
            "bias_applied":    round(bias, 4),
            "model_feedback":  {
                "fake_count": stats["fake_count"],
                "real_count": stats["real_count"],
            },
        },
    }


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
