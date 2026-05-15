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


def classify_air_quality_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Detect visible smoke/smog/haze in an air-quality report image.

    This is a conservative visual heuristic, not an AQI sensor reading. It
    exists so air-quality reports are not mislabeled as flood "clear" simply
    because no flooding was detected.
    """
    try:
        return _analyse_air_quality(image_bytes)
    except Exception as exc:
        return {
            "is_poor_air_quality": False,
            "confidence": 0.0,
            "label": "air_quality_unknown",
            "details": {"error": str(exc)},
        }


def classify_report_image(report_type: str, image_bytes: bytes) -> Dict[str, Any]:
    """
    Run the detector that matches the citizen-selected report category.

    These are lightweight visual heuristics calibrated against local sample
    incident photos. They are intentionally category-specific: a tree-fall
    image should never be interpreted through a flood-only lens.
    """
    try:
        auto = classify_incident_image(image_bytes)
        selected = _analyse_selected_report_type(report_type, image_bytes)
        if selected is None:
            return {
                "label": "image_received",
                "confidence": 0.0,
                "details": {
                    "note": "No category-specific detector for this report type.",
                    "auto_label": auto["label"],
                    "auto_confidence": auto["confidence"],
                },
            }

        selected_positive = selected["label"] not in {
            "no_visible_air_pollution",
            "no_visible_heat_wave",
            "no_visible_tree_fall",
            "no_visible_waterlogging",
            "no_visible_flood",
        }
        auto_is_different_positive = (
            auto["label"] != selected["label"]
            and auto["label"] != "uncertain_incident"
            and auto["confidence"] >= max(0.68, (selected.get("confidence") or 0) + 0.10)
        )
        if (not selected_positive and auto["confidence"] >= 0.62) or auto_is_different_positive:
            auto["details"]["selected_report_type"] = report_type
            auto["details"]["selected_detector_label"] = selected["label"]
            auto["details"]["auto_override"] = True
            return auto
        selected["details"]["auto_label"] = auto["label"]
        selected["details"]["auto_confidence"] = auto["confidence"]
        selected["details"]["auto_override"] = False
        return selected
        return {
            "label": "image_received",
            "confidence": 0.0,
            "details": {"note": "No category-specific detector for this report type."},
        }
    except Exception as exc:
        return {
            "label": "analysis_error",
            "confidence": 0.0,
            "details": {"error": str(exc)},
        }


def _analyse_selected_report_type(report_type: str, image_bytes: bytes) -> Dict[str, Any] | None:
    if report_type == "air_quality":
        return classify_air_quality_image(image_bytes)
    if report_type == "heat_wave":
        return _analyse_heat_wave(image_bytes)
    if report_type == "tree_fall":
        return _analyse_tree_fall(image_bytes)
    if report_type == "waterlogging":
        return _analyse_water_incident(image_bytes, label="waterlogging")
    if report_type == "flood":
        return _analyse_water_incident(image_bytes, label="flood")
    return None


def classify_incident_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Multi-class civic incident router.

    This is still a transparent prototype heuristic, not a trained universal
    vision model. It prevents the most damaging demo failure: running every
    image through flood-only logic and mislabelling AQI or tree-fall photos.
    """
    candidates = [
        classify_air_quality_image(image_bytes),
        _analyse_tree_fall(image_bytes),
        _analyse_water_incident(image_bytes, label="flood"),
        _analyse_water_incident(image_bytes, label="waterlogging"),
        _analyse_heat_wave(image_bytes),
    ]
    positive = [
        c for c in candidates
        if c["label"] not in {
            "no_visible_air_pollution",
            "no_visible_tree_fall",
            "no_visible_flood",
            "no_visible_waterlogging",
            "no_visible_heat_wave",
        }
    ]
    if not positive:
        best = max(candidates, key=lambda c: c["details"].get("score", c.get("confidence", 0.0)))
        return {
            "label": "uncertain_incident",
            "confidence": 0.0,
            "details": {
                "best_negative_label": best["label"],
                "candidate_labels": [c["label"] for c in candidates],
            },
        }

    def priority(c: Dict[str, Any]) -> tuple[float, int]:
        label = c["label"]
        # Distinct visual classes first, then water severity.
        order = {
            "poor_air_quality": 5,
            "tree_fall_detected": 4,
            "flood": 3,
            "waterlogging": 2,
            "heat_wave_dry_conditions": 1,
        }
        return (float(c.get("confidence") or 0.0), order.get(label, 0))

    best = max(positive, key=priority)
    best["details"]["candidate_labels"] = [c["label"] for c in candidates]
    return best


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
        channel_range = max(r, g, b) - min(r, g, b)

        # Water: blue/cyan dominant, moderate brightness.
        if b > r + 20 and b >= g and 45 < brightness < 190:
            water_score += 1
        # Murky flood water: low-saturation cool grey, not generic dark/brown.
        if channel_range < 24 and b >= r and g >= r and 55 < brightness < 150:
            water_score += 0.6
        # Muddy standing water. This is supporting evidence only; mud alone
        # should not classify a fallen tree or dry soil as flooding.
        if r > g > b and 70 < brightness < 150 and (r - b) > 35:
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

    # ── Composite flood evidence ──────────────────────────────────────────
    # The old implementation ran tiny colour ratios through a sigmoid, which
    # produced authoritative-looking percentages from weak evidence. Keep the
    # score literal and require visible water coverage before raising alerts.
    raw_flood = (
        ws  * 2.2 +
        ms  * 0.75 +
        ds  * 0.25 -
        gs  * 1.1 -
        dry * 0.8
    )
    raw_flood = max(0.0, min(1.0, raw_flood))

    # Apply adaptive bias from BBMP feedback
    feedback_entries = _load_feedback()
    stats = _compute_stats(feedback_entries)
    bias  = stats["bias"]

    adjusted_flood = max(0.0, min(1.0, raw_flood + bias))

    feature = _feature_summary(image_bytes)
    haze_veto = (
        feature["all_gray_haze_pct"] >= 85
        and feature["lower_white_foam_pct"] < 5
        and feature["lower_water_pct"] < 5
    )
    min_water_pct = 8.0
    min_evidence = 0.30
    is_flood = (
        ws * 100 >= min_water_pct
        and adjusted_flood >= min_evidence
        and not haze_veto
    )
    confidence = _flood_confidence(adjusted_flood) if is_flood else 0.0

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
            "adjusted_flood_score": round(adjusted_flood, 3),
            "haze_veto": haze_veto,
            "min_water_px_pct": min_water_pct,
            "min_flood_score": min_evidence,
            "bias_applied":    round(bias, 4),
            "model_feedback":  {
                "fake_count": stats["fake_count"],
                "real_count": stats["real_count"],
            },
        },
    }


def _flood_confidence(score: float) -> float:
    """
    Convert flood evidence to a conservative display confidence.

    This is not a calibrated ML probability; it is only exposed after the
    threshold has passed, so the UI does not show fake precision for clear
    or unrelated images.
    """
    return max(0.55, min(0.95, 0.55 + (score - 0.30) * 0.57))


def _analyse_air_quality(image_bytes: bytes) -> Dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {
            "is_poor_air_quality": False,
            "confidence": 0.0,
            "label": "air_quality_unknown",
            "details": {"note": "PIL unavailable, skipping air-quality image analysis"},
        }

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    pixels = list(img.getdata())
    n = len(pixels)

    orange_haze = gray_haze = dark_smoke = clear_sky = green = 0.0
    for r, g, b in pixels:
        brightness = (r + g + b) / 3.0
        channel_range = max(r, g, b) - min(r, g, b)

        # Dust/smog/smoke often casts the full frame orange/brown.
        if r > g + 12 and g > b + 12 and 70 < brightness < 230:
            orange_haze += 1
        # Grey haze/smog: low contrast, low saturation, middle brightness.
        if channel_range < 35 and 70 < brightness < 220:
            gray_haze += 1
        # Heavy smoke can be visibly dark without being green vegetation.
        if brightness < 60 and not (g > r + 20 and g > b + 20):
            dark_smoke += 1
        if b > r + 15 and b > g + 10 and brightness > 100:
            clear_sky += 1
        if g > r + 20 and g > b + 20:
            green += 1

    orange_pct = orange_haze / n
    gray_pct = gray_haze / n
    dark_pct = dark_smoke / n
    clear_sky_pct = clear_sky / n
    green_pct = green / n

    evidence = (
        orange_pct * 0.75 +
        gray_pct * 0.55 +
        dark_pct * 0.20 -
        clear_sky_pct * 0.45 -
        green_pct * 0.15
    )
    civic_features = _feature_summary(image_bytes)
    tree_like_obstruction = (
        civic_features["all_green_pct"] >= 2.0
        and civic_features["lower_brown_trunk_pct"] >= 24.0
        and civic_features["lower_water_pct"] + civic_features["lower_murky_pct"] < 18.0
    )
    if tree_like_obstruction:
        evidence -= 0.25
    evidence = max(0.0, min(1.0, evidence))
    is_poor = not tree_like_obstruction and (orange_pct >= 0.30 or gray_pct >= 0.45 or evidence >= 0.35)
    confidence = max(0.55, min(0.95, 0.55 + evidence * 0.45)) if is_poor else 0.0

    return {
        "is_poor_air_quality": is_poor,
        "confidence": round(confidence, 3),
        "label": "poor_air_quality" if is_poor else "no_visible_air_pollution",
        "details": {
            "orange_haze_pct": round(orange_pct * 100, 1),
            "gray_haze_pct": round(gray_pct * 100, 1),
            "dark_smoke_pct": round(dark_pct * 100, 1),
            "clear_sky_pct": round(clear_sky_pct * 100, 1),
            "green_px_pct": round(green_pct * 100, 1),
            "air_quality_score": round(evidence, 3),
            "tree_like_obstruction_veto": tree_like_obstruction,
            "note": "Visual smoke/smog heuristic; not a sensor AQI reading.",
        },
    }


def _feature_summary(image_bytes: bytes) -> Dict[str, float]:
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    width, height = img.size
    pixels = list(img.getdata())
    lower_pixels = pixels[int(height * 0.4) * width:]

    def ratios(sample: list[tuple[int, int, int]], prefix: str) -> Dict[str, float]:
        counts = {
            "water": 0.0,
            "murky": 0.0,
            "mud": 0.0,
            "dark": 0.0,
            "green": 0.0,
            "dry": 0.0,
            "orange_haze": 0.0,
            "gray_haze": 0.0,
            "clear_sky": 0.0,
            "pale_crack": 0.0,
            "brown_trunk": 0.0,
            "white_foam": 0.0,
            "asphalt": 0.0,
        }
        n = len(sample)
        for r, g, b in sample:
            brightness = (r + g + b) / 3.0
            channel_range = max(r, g, b) - min(r, g, b)

            if b > r + 20 and b >= g and 45 < brightness < 190:
                counts["water"] += 1
            if channel_range < 24 and b >= r and g >= r and 55 < brightness < 150:
                counts["murky"] += 1
            if r > g > b and 70 < brightness < 150 and (r - b) > 35:
                counts["mud"] += 1
            if brightness < 50:
                counts["dark"] += 1
            if g > r + 20 and g > b + 20:
                counts["green"] += 1
            if r > 160 and g > 130 and b < 120:
                counts["dry"] += 1
            if r > g + 12 and g > b + 12 and 70 < brightness < 230:
                counts["orange_haze"] += 1
            if channel_range < 35 and 70 < brightness < 220:
                counts["gray_haze"] += 1
            if b > r + 15 and b > g + 10 and brightness > 100:
                counts["clear_sky"] += 1
            if r > 145 and g > 125 and b > 95 and channel_range < 65:
                counts["pale_crack"] += 1
            if (
                35 < brightness < 150
                and r >= g - 8
                and g >= b - 18
                and (r - b) < 85
                and not (b > r + 20 and b >= g)
            ):
                counts["brown_trunk"] += 1
            if brightness > 165 and channel_range < 45:
                counts["white_foam"] += 1
            if channel_range < 30 and 35 < brightness < 115:
                counts["asphalt"] += 1

        return {f"{prefix}_{k}_pct": round(v / n * 100, 1) for k, v in counts.items()}

    result = ratios(pixels, "all")
    result.update(ratios(lower_pixels, "lower"))
    return result


def _with_common_result(
    label: str,
    positive_label: str,
    negative_label: str,
    score: float,
    is_positive: bool,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    confidence = max(0.55, min(0.95, 0.55 + score * 0.40)) if is_positive else 0.0
    return {
        "label": positive_label if is_positive else negative_label,
        "confidence": round(confidence, 3),
        "details": {
            "category": label,
            "score": round(score, 3),
            **details,
        },
    }


def _analyse_water_incident(image_bytes: bytes, label: str) -> Dict[str, Any]:
    features = _feature_summary(image_bytes)
    lower_water = features["lower_water_pct"]
    lower_murky = features["lower_murky_pct"]
    lower_foam = features["lower_white_foam_pct"]
    lower_gray = features["lower_gray_haze_pct"]
    lower_pale = features["lower_pale_crack_pct"]
    lower_green = features["lower_green_pct"]
    lower_trunk = features["lower_brown_trunk_pct"]
    all_gray = features["all_gray_haze_pct"]

    standing_water = lower_water + lower_murky
    fast_water = lower_foam
    score = max(
        standing_water / 35.0,
        fast_water / 35.0,
        (lower_gray + lower_foam + lower_murky) / 95.0,
    )
    score -= min(0.35, lower_green / 80.0)
    score -= 0.30 if lower_pale > 60 and standing_water < 10 and lower_foam < 30 else 0.0
    score -= 0.40 if all_gray > 88 and lower_foam < 6 and standing_water < 15 else 0.0
    score -= 0.28 if lower_green > 2 and lower_trunk > 24 and label == "flood" else 0.0
    score = max(0.0, min(1.0, score))

    haze_false_positive = all_gray > 88 and lower_foam < 6 and lower_water < 5
    tree_like_obstruction = lower_green > 2 and lower_trunk > 24 and standing_water < 18
    moving_road_water = (
        lower_foam >= 18
        and lower_gray >= 35
        and lower_pale < 66
        and not haze_false_positive
        and not tree_like_obstruction
    )
    if label == "flood":
        is_positive = (
            (standing_water >= 18 and not haze_false_positive and not tree_like_obstruction)
            or (lower_foam >= 30 and not haze_false_positive and not tree_like_obstruction)
        )
    else:
        is_positive = (
            (standing_water >= 18 and not haze_false_positive and not tree_like_obstruction)
            or (lower_foam >= 12 and lower_pale < 60 and not tree_like_obstruction)
            or (lower_foam >= 30 and not haze_false_positive and not tree_like_obstruction)
            or moving_road_water
            or (
                lower_gray >= 55
                and (lower_foam >= 8 or lower_murky >= 5)
                and lower_pale < 60
                and not haze_false_positive
                and not tree_like_obstruction
            )
        )
    positive = "waterlogging" if label == "waterlogging" else "flood"
    negative = "no_visible_waterlogging" if label == "waterlogging" else "no_visible_flood"
    details = {
        **features,
        "moving_road_water": moving_road_water,
        "haze_false_positive_veto": haze_false_positive,
        "tree_like_obstruction_veto": tree_like_obstruction,
    }
    return _with_common_result(label, positive, negative, score, is_positive, details)


def _analyse_heat_wave(image_bytes: bytes) -> Dict[str, Any]:
    features = _feature_summary(image_bytes)
    cracked = features["lower_pale_crack_pct"]
    dry = features["lower_dry_pct"]
    orange = features["all_orange_haze_pct"]
    water = features["lower_water_pct"] + features["lower_murky_pct"]
    score = max(cracked / 75.0, (dry + orange * 0.5) / 60.0)
    score -= min(0.45, water / 45.0)
    score = max(0.0, min(1.0, score))
    is_positive = (cracked >= 55 or dry >= 25 or orange >= 35) and water < 12
    return _with_common_result(
        "heat_wave",
        "heat_wave_dry_conditions",
        "no_visible_heat_wave",
        score,
        is_positive,
        features,
    )


def _analyse_tree_fall(image_bytes: bytes) -> Dict[str, Any]:
    features = _feature_summary(image_bytes)
    trunk = features["lower_brown_trunk_pct"]
    asphalt = features["lower_asphalt_pct"]
    green = features["all_green_pct"]
    foam = features["lower_white_foam_pct"]
    orange = features["all_orange_haze_pct"]
    pale = features["lower_pale_crack_pct"]
    murky = features["lower_murky_pct"]
    water = features["lower_water_pct"] + features["lower_murky_pct"]

    # Tree-fall detection used to treat any dark/brown/green lower-frame image
    # as a fallen tree. Keep it conservative: require road/ground obstruction
    # plus visible vegetation/branch evidence, and reject flood/heat/haze lookalikes.
    score = (
        max(0.0, trunk - 24.0) / 42.0 * 0.42
        + max(0.0, asphalt - 14.0) / 48.0 * 0.18
        + min(green, 6.0) / 6.0 * 0.32
        + max(0.0, 25.0 - orange) / 25.0 * 0.08
    )
    score -= min(0.08, foam / 160.0)
    score -= 0.08 if pale >= 60 else 0.0
    score -= 0.18 if orange >= 25 else 0.0
    score -= 0.12 if murky >= 18 and green < 0.3 else 0.0
    score -= 0.30 if water >= 18 else 0.0
    # This is a heuristic detector, not a trained probability model. Cap the
    # evidence so the UI never shows fake 90%+ certainty for tree-fall photos.
    score = max(0.0, min(0.78, score))
    is_positive = (
        trunk >= 24
        and green >= 1.5
        and asphalt >= 12
        and water < 18
        and orange < 25
        and score >= 0.20
    )
    return _with_common_result(
        "tree_fall",
        "tree_fall_detected",
        "no_visible_tree_fall",
        score,
        is_positive,
        features,
    )
