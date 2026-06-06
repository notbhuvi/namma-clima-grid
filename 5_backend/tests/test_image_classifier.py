from __future__ import annotations

import io
import os
import sys
from pathlib import Path

from PIL import Image
import pytest

os.environ.setdefault("NCG_SKIP_MODEL_LOAD", "true")
os.environ.setdefault("NCG_SKIP_KAFKA_CONSUMER", "true")
os.environ.setdefault("AUTH_REQUIRED", "true")
os.environ.setdefault("ADMIN_API_KEY", "ci-admin-token-please-rotate")

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "5_backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from image_classifier import (  # noqa: E402
    classify_air_quality_image,
    classify_image,
    classify_incident_image,
    classify_report_image,
)
from routes.reports import (  # noqa: E402
    _apply_flood_prediction_guard,
    _report_type_from_ai_label,
    _validate_report_location,
)


def _jpeg_bytes(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (224, 224), top)
    for y in range(90, 224):
        for x in range(224):
            img.putpixel((x, y), bottom)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_green_tree_fall_like_image_is_not_flood() -> None:
    result = classify_image(_jpeg_bytes((125, 155, 120), (45, 110, 42)))

    assert result["label"] == "no_flood"
    assert result["is_flood"] is False
    assert result["confidence"] == 0.0
    assert result["details"]["green_px_pct"] > 90


def test_brown_tree_or_soil_like_image_is_not_flood() -> None:
    result = classify_image(_jpeg_bytes((150, 150, 140), (120, 82, 48)))

    assert result["label"] == "no_flood"
    assert result["is_flood"] is False
    assert result["confidence"] == 0.0
    assert result["details"]["water_px_pct"] == 0.0


def test_visible_blue_standing_water_is_flood() -> None:
    result = classify_image(_jpeg_bytes((150, 165, 180), (55, 105, 150)))

    assert result["label"] == "flood"
    assert result["is_flood"] is True
    assert result["confidence"] >= 0.55
    assert result["details"]["water_px_pct"] > 90


def test_tree_fall_report_is_not_rewritten_to_flood_on_weak_evidence() -> None:
    result = {
        "is_flood": True,
        "confidence": 0.7,
        "details": {},
    }

    flood_predicted, flood_confidence = _apply_flood_prediction_guard(
        "tree_fall", result
    )

    assert flood_predicted is False
    assert flood_confidence is None
    assert "auto_alert_suppressed" in result["details"]


def test_flood_report_keeps_flood_prediction() -> None:
    result = {
        "is_flood": True,
        "confidence": 0.7,
        "details": {},
    }

    flood_predicted, flood_confidence = _apply_flood_prediction_guard(
        "flood", result
    )

    assert flood_predicted is True
    assert flood_confidence == 0.7


def test_location_mismatch_auto_overrides_to_expected_ward() -> None:
    # Synthetic ward 1 centroid is near the south-west corner of the app bbox.
    check = _validate_report_location(
        selected_ward_id=2,
        latitude=12.840333,
        longitude=77.461786,
    )

    assert check["location_mismatch"] is False
    assert check["expected_ward_id"] == 1
    assert check["expected_ward_name"]
    assert check["resolved_ward_id"] == 1
    assert check["submitted_ward_overridden"] is True
    assert check["fake_reason"] is None


def test_location_match_is_not_fake() -> None:
    check = _validate_report_location(
        selected_ward_id=1,
        latitude=12.840333,
        longitude=77.461786,
    )

    assert check["location_mismatch"] is False
    assert check["fake_reason"] is None
    assert check["resolved_ward_id"] == 1


def test_orange_haze_image_is_poor_air_quality() -> None:
    result = classify_air_quality_image(
        _jpeg_bytes((205, 145, 95), (150, 98, 62))
    )

    assert result["label"] == "poor_air_quality"
    assert result["is_poor_air_quality"] is True
    assert result["confidence"] >= 0.55
    assert result["details"]["orange_haze_pct"] > 80


def test_clear_blue_sky_is_not_poor_air_quality() -> None:
    result = classify_air_quality_image(
        _jpeg_bytes((85, 145, 210), (90, 150, 215))
    )

    assert result["label"] == "no_visible_air_pollution"
    assert result["is_poor_air_quality"] is False
    assert result["confidence"] == 0.0


@pytest.mark.parametrize(
    ("report_type", "path", "expected"),
    [
        ("air_quality", "/Users/bhuvanesh/Downloads/aq1.jpeg", "poor_air_quality"),
        ("air_quality", "/Users/bhuvanesh/Downloads/aq2.jpg", "poor_air_quality"),
        ("waterlogging", "/Users/bhuvanesh/Downloads/wl1.jpeg", "waterlogging"),
        ("flood", "/Users/bhuvanesh/Downloads/ff.jpeg", "flood"),
        ("flood", "/Users/bhuvanesh/Downloads/ff2.webp", "flood"),
        ("heat_wave", "/Users/bhuvanesh/Downloads/hw.jpeg", "heat_wave_dry_conditions"),
        ("tree_fall", "/Users/bhuvanesh/Downloads/tf1.jpeg", "tree_fall_detected"),
        ("tree_fall", "/Users/bhuvanesh/Downloads/tf2.jpg", "tree_fall_detected"),
    ],
)
def test_local_calibration_images_match_report_categories(
    report_type: str,
    path: str,
    expected: str,
) -> None:
    image_path = Path(path)
    if not image_path.exists():
        pytest.skip(f"Local calibration image missing: {path}")

    result = classify_report_image(report_type, image_path.read_bytes())

    assert result["label"] == expected
    assert result["confidence"] >= 0.55


@pytest.mark.parametrize(
    "path",
    [
        "/Users/bhuvanesh/Downloads/aq1.jpeg",
        "/Users/bhuvanesh/Downloads/aq2.jpg",
        "/Users/bhuvanesh/Downloads/wl1.jpeg",
        "/Users/bhuvanesh/Downloads/ff.jpeg",
        "/Users/bhuvanesh/Downloads/ff2.webp",
        "/Users/bhuvanesh/Downloads/hw.jpeg",
    ],
)
def test_non_tree_calibration_images_are_not_tree_fall(path: str) -> None:
    image_path = Path(path)
    if not image_path.exists():
        pytest.skip(f"Local calibration image missing: {path}")

    result = classify_report_image("tree_fall", image_path.read_bytes())

    assert result["label"] == "no_visible_tree_fall"
    assert result["confidence"] == 0.0


@pytest.mark.parametrize(
    ("report_type", "path", "expected_label", "expected_auto_label"),
    [
        ("air_quality", "/Users/bhuvanesh/Downloads/AQI1.jpg", "poor_air_quality", "poor_air_quality"),
        ("air_quality", "/Users/bhuvanesh/Downloads/AQI2.jpg", "poor_air_quality", "poor_air_quality"),
        ("waterlogging", "/Users/bhuvanesh/Downloads/WATERLOG1.jpeg", "waterlogging", "waterlogging"),
        ("waterlogging", "/Users/bhuvanesh/Downloads/WATERLOG2.jpeg", "waterlogging", "waterlogging"),
        ("tree_fall", "/Users/bhuvanesh/Downloads/FALLEN TREE.jpg", "tree_fall_detected", "tree_fall_detected"),
        ("flood", "/Users/bhuvanesh/Downloads/FLOOD1.jpg", "flood", "flood"),
        ("flood", "/Users/bhuvanesh/Downloads/FLOOD2.jpg", "flood", "flood"),
    ],
)
def test_user_supplied_civic_images_are_classified_correctly(
    report_type: str,
    path: str,
    expected_label: str,
    expected_auto_label: str,
) -> None:
    image_path = Path(path)
    if not image_path.exists():
        pytest.skip(f"Local civic image missing: {path}")

    image_bytes = image_path.read_bytes()
    selected_result = classify_report_image(report_type, image_bytes)
    auto_result = classify_incident_image(image_bytes)

    assert selected_result["label"] == expected_label
    assert selected_result["confidence"] >= 0.55
    assert auto_result["label"] == expected_auto_label
    assert auto_result["confidence"] >= 0.55


def test_ai_label_maps_to_correct_report_type() -> None:
    assert _report_type_from_ai_label("poor_air_quality") == "air_quality"
    assert _report_type_from_ai_label("tree_fall_detected") == "tree_fall"
    assert _report_type_from_ai_label("waterlogging") == "waterlogging"
    assert _report_type_from_ai_label("flood") == "flood"
