"""
Routes — Citizen Report endpoints
====================================

  POST /reports/              → Submit a citizen report (JSON or multipart with image)
  POST /reports/with-image    → Submit report + image → runs flood classifier → auto-alert
  GET  /reports/recent        → Last N reports (with flood_predicted flag)
  GET  /reports/ward/{ward_id}/count  → Report counts by type for a ward
"""
from __future__ import annotations

import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from auth import require_role
from config import get_settings

router = APIRouter(prefix="/reports", tags=["reports"])

UPLOAD_DIR = str(get_settings().upload_dir)
os.makedirs(UPLOAD_DIR, exist_ok=True)

FLOOD_LIKE_REPORT_TYPES = {"flood", "waterlogging"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "8_data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

_KNOWN_LOCATION_WARD_OVERRIDES = [
    {
        "ward_id": 111,
        "ward_name": "Shanthalanagar",
        "lat_min": 12.9600,
        "lat_max": 12.9825,
        "lon_min": 77.5900,
        "lon_max": 77.6075,
        "note": "Central Bengaluru / UB City / Cubbon Park belt",
    },
]


# ---------------------------------------------------------------------------
# Pydantic model (JSON-body endpoint)
# ---------------------------------------------------------------------------

class CitizenReport(BaseModel):
    ward_id:     Optional[int]  = Field(None, ge=1, le=198)
    latitude:    float          = Field(..., ge=12.77, le=13.18)
    longitude:   float          = Field(..., ge=77.35, le=77.82)
    report_type: str            = Field(...)
    severity:    int            = Field(..., ge=1, le=5)
    description: Optional[str] = Field(None, max_length=1000)
    image_url:   Optional[str] = Field(None, max_length=500)


def _apply_flood_prediction_guard(
    report_type: str,
    result: dict,
) -> tuple[bool, Optional[float]]:
    if report_type not in FLOOD_LIKE_REPORT_TYPES:
        result["details"]["auto_alert_suppressed"] = (
            f"Report type '{report_type}' is not flood-like; flood alerts are "
            "only emitted for flood and waterlogging reports."
        )
        return False, None

    flood_predicted = result["is_flood"]
    flood_confidence = result["confidence"] if flood_predicted else None
    return flood_predicted, flood_confidence


def _analyse_saved_report_image(report_type: str, image_url: Optional[str]) -> Optional[dict]:
    if not image_url:
        return None
    try:
        from image_classifier import classify_report_image

        filename = os.path.basename(image_url)
        filepath = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(filepath):
            return None
        with open(filepath, "rb") as f:
            return classify_report_image(report_type, f.read())
    except Exception:
        return None


def _report_type_from_ai_label(label: str) -> Optional[str]:
    return {
        "poor_air_quality": "air_quality",
        "heat_wave_dry_conditions": "heat_wave",
        "tree_fall_detected": "tree_fall",
        "waterlogging": "waterlogging",
        "flood": "flood",
    }.get(label)


def _derive_effective_ai_result(report_type: str, ai_result: dict) -> dict:
    detected_type = _report_type_from_ai_label(ai_result.get("label", ""))
    auto_override = bool(ai_result.get("details", {}).get("auto_override"))
    effective_type = (
        detected_type
        if detected_type and (auto_override or detected_type in FLOOD_LIKE_REPORT_TYPES)
        else report_type
    )
    flood_predicted = (
        effective_type in FLOOD_LIKE_REPORT_TYPES
        and ai_result.get("label") in ("flood", "waterlogging")
    )
    flood_confidence = ai_result.get("confidence") if flood_predicted else None
    return {
        "effective_type": effective_type,
        "flood_predicted": flood_predicted,
        "flood_confidence": flood_confidence,
    }


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_ward_for_location(latitude: float, longitude: float) -> Optional[dict]:
    for override in _KNOWN_LOCATION_WARD_OVERRIDES:
        if (
            override["lat_min"] <= latitude <= override["lat_max"]
            and override["lon_min"] <= longitude <= override["lon_max"]
        ):
            return {
                "ward_id": int(override["ward_id"]),
                "ward_name": str(override["ward_name"]),
                "distance_km": 0.0,
                "source": "known_location_override",
                "note": override["note"],
            }

    try:
        from _common import synthetic_ward_catalogue  # type: ignore
    except Exception:
        return None

    nearest = None
    for ward in synthetic_ward_catalogue():
        centroid_lon, centroid_lat = ward["centroid"]
        distance = _distance_km(latitude, longitude, centroid_lat, centroid_lon)
        if nearest is None or distance < nearest["distance_km"]:
            nearest = {
                "ward_id": int(ward["ward_id"]),
                "ward_name": str(ward["ward_name"]),
                "distance_km": round(distance, 3),
            }
    return nearest


def _validate_report_location(
    selected_ward_id: Optional[int],
    latitude: float,
    longitude: float,
) -> dict:
    expected = _nearest_ward_for_location(latitude, longitude)
    if expected is None:
        return {
            "location_mismatch": False,
            "expected_ward_id": None,
            "expected_ward_name": None,
            "resolved_ward_id": selected_ward_id,
            "resolved_ward_name": None,
            "distance_km": None,
            "fake_reason": None,
            "ward_auto_selected": False,
        }

    mismatch = selected_ward_id is not None and expected["ward_id"] != selected_ward_id
    return {
        "location_mismatch": False,
        "expected_ward_id": expected["ward_id"],
        "expected_ward_name": expected["ward_name"],
        "resolved_ward_id": expected["ward_id"],
        "resolved_ward_name": expected["ward_name"],
        "distance_km": expected["distance_km"],
        "fake_reason": None,
        "ward_auto_selected": True,
        "submitted_ward_id": selected_ward_id,
        "submitted_ward_overridden": mismatch,
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS citizen_reports (
                id               SERIAL PRIMARY KEY,
                ward_id          INTEGER NOT NULL,
                latitude         DOUBLE PRECISION,
                longitude        DOUBLE PRECISION,
                report_type      TEXT NOT NULL,
                severity         INTEGER CHECK (severity BETWEEN 1 AND 5),
                description      TEXT,
                image_url        TEXT,
                flood_predicted  BOOLEAN DEFAULT FALSE,
                flood_confidence DOUBLE PRECISION,
                ai_label         TEXT,
                ai_confidence    DOUBLE PRECISION,
                ai_details       JSONB,
                is_fake          BOOLEAN DEFAULT FALSE,
                fake_reason      TEXT,
                expected_ward_id INTEGER,
                expected_ward_name TEXT,
                location_distance_km DOUBLE PRECISION,
                reported_at      TIMESTAMPTZ DEFAULT NOW(),
                source           TEXT DEFAULT 'api'
            )
        """)
        # Add new columns to existing tables (idempotent)
        for col, defn in [
            ("flood_predicted",  "BOOLEAN DEFAULT FALSE"),
            ("flood_confidence", "DOUBLE PRECISION"),
            ("ai_label",         "TEXT"),
            ("ai_confidence",    "DOUBLE PRECISION"),
            ("ai_details",       "JSONB"),
            ("is_fake",          "BOOLEAN DEFAULT FALSE"),
            ("bbmp_confirmed",   "BOOLEAN DEFAULT FALSE"),
            ("fake_reason",      "TEXT"),
            ("expected_ward_id", "INTEGER"),
            ("expected_ward_name", "TEXT"),
            ("location_distance_km", "DOUBLE PRECISION"),
        ]:
            try:
                cur.execute(
                    f"ALTER TABLE citizen_reports ADD COLUMN IF NOT EXISTS {col} {defn}"
                )
            except Exception:
                pass
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_citizen_reports_ward_ts
            ON citizen_reports (ward_id, reported_at DESC)
        """)
    conn.commit()


def _save_report(
    ward_id: int,
    latitude: float,
    longitude: float,
    report_type: str,
    severity: int,
    description: Optional[str],
    image_url: Optional[str],
    flood_predicted: bool = False,
    flood_confidence: Optional[float] = None,
    ai_label: Optional[str] = None,
    ai_confidence: Optional[float] = None,
    ai_details: Optional[dict] = None,
    is_fake: bool = False,
    fake_reason: Optional[str] = None,
    expected_ward_id: Optional[int] = None,
    expected_ward_name: Optional[str] = None,
    location_distance_km: Optional[float] = None,
) -> int:
    try:
        import psycopg2
    except ImportError:
        raise HTTPException(status_code=500, detail="psycopg2 not installed")

    try:
        conn = psycopg2.connect(get_settings().postgres_dsn)
        _ensure_table(conn)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO citizen_reports
                    (ward_id, latitude, longitude, report_type, severity,
                     description, image_url, flood_predicted, flood_confidence,
                     ai_label, ai_confidence, ai_details, is_fake, fake_reason,
                     expected_ward_id, expected_ward_name, location_distance_km)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                ward_id, latitude, longitude, report_type, severity,
                description, image_url, flood_predicted, flood_confidence,
                ai_label, ai_confidence,
                json.dumps(ai_details) if ai_details is not None else None,
                is_fake, fake_reason, expected_ward_id, expected_ward_name,
                location_distance_km,
            ))
            row_id = cur.fetchone()[0]
        conn.commit()
        conn.close()
        return row_id
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


async def _broadcast_flood_alert(ward_id: int, ward_name: str,
                                  confidence: float, report_id: int,
                                  image_url: Optional[str]) -> None:
    """Push a flood alert to all WebSocket clients."""
    try:
        from routes.websocket import inject_alert
        await inject_alert({
            "ward_id":    ward_id,
            "alert_type": "flood_warning",
            "severity":   "critical" if confidence >= 0.75 else "high",
            "message":    f"🌊 Citizen-reported flooding in {ward_name} "
                          f"(AI confidence: {confidence*100:.0f}%)",
            "value":      round(confidence * 100, 1),
            "threshold":  52.0,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "image_url":  image_url,
            "report_id":  report_id,
            "source":     "citizen_image",
        })
    except Exception as exc:
        import logging
        logging.warning(f"Alert broadcast failed: {exc}")


# ---------------------------------------------------------------------------
# GET /reports/resolve-ward  — derive ward from GPS coordinates
# ---------------------------------------------------------------------------

@router.get("/resolve-ward")
async def resolve_ward(latitude: float, longitude: float):
    if not (12.77 <= latitude <= 13.18 and 77.35 <= longitude <= 77.82):
        raise HTTPException(
            status_code=400,
            detail="Coordinates are outside supported Bengaluru bounds",
        )
    ward = _nearest_ward_for_location(latitude, longitude)
    if ward is None:
        raise HTTPException(status_code=404, detail="Could not resolve ward")
    return {
        "ward_id": ward["ward_id"],
        "ward_name": ward["ward_name"],
        "distance_km": ward["distance_km"],
        "latitude": latitude,
        "longitude": longitude,
        "source": "coordinates",
    }


# ---------------------------------------------------------------------------
# POST /reports/  (JSON body — legacy)
# ---------------------------------------------------------------------------

@router.post("/")
async def submit_report(body: CitizenReport):
    location_check = _validate_report_location(
        body.ward_id, body.latitude, body.longitude
    )
    resolved_ward_id = location_check["resolved_ward_id"] or body.ward_id
    if resolved_ward_id is None:
        raise HTTPException(status_code=400, detail="Could not resolve ward from coordinates")
    report_id = _save_report(
        ward_id=resolved_ward_id, latitude=body.latitude, longitude=body.longitude,
        report_type=body.report_type, severity=body.severity,
        description=body.description, image_url=body.image_url,
        is_fake=False,
        fake_reason=location_check["fake_reason"],
        expected_ward_id=location_check["expected_ward_id"],
        expected_ward_name=location_check["expected_ward_name"],
        location_distance_km=location_check["distance_km"],
    )
    return {
        "status":       "accepted",
        "report_id":    report_id,
        "ward_id":      resolved_ward_id,
        "ward_name":    location_check["resolved_ward_name"],
        "report_type":  body.report_type,
        "severity":     body.severity,
        "location_check": location_check,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "message":      "Thank you! Ward was selected automatically from your location.",
    }


# ---------------------------------------------------------------------------
# POST /reports/with-image  (multipart — image + flood AI)
# ---------------------------------------------------------------------------

@router.post("/with-image")
async def submit_report_with_image(
    ward_id:     Optional[int]  = Form(None),
    latitude:    float          = Form(...),
    longitude:   float          = Form(...),
    report_type: str            = Form(...),
    severity:    int            = Form(...),
    description: Optional[str] = Form(None),
    image:       UploadFile     = File(...),
):
    """
    Submit a citizen report with an image.
    The image is run through the flood classifier.
    If flood is detected, an alert is automatically broadcast to all
    connected citizens and the BBMP dashboard.
    """
    # 1. Read and validate image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await image.read()
    if len(image_bytes) > 10 * 1024 * 1024:   # 10 MB limit
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    # 2. Save image to static/uploads
    ext      = image.filename.rsplit(".", 1)[-1].lower() if "." in image.filename else "jpg"
    filename = f"report_{uuid.uuid4().hex[:12]}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    image_url = f"/bbmp-static/uploads/{filename}"

    # 3. Run report-type image analysis
    from image_classifier import classify_report_image
    ai_result = classify_report_image(report_type, image_bytes)
    derived = _derive_effective_ai_result(report_type, ai_result)
    effective_type = derived["effective_type"]
    flood_predicted = derived["flood_predicted"]
    flood_confidence = derived["flood_confidence"]

    # 4. Save to DB
    from services import _ward_catalogue
    names     = _ward_catalogue()
    location_check = _validate_report_location(ward_id, latitude, longitude)
    resolved_ward_id = location_check["resolved_ward_id"] or ward_id
    if resolved_ward_id is None:
        raise HTTPException(status_code=400, detail="Could not resolve ward from coordinates")
    ward_name = names.get(resolved_ward_id, f"Ward {resolved_ward_id}")

    report_id = _save_report(
        ward_id=resolved_ward_id, latitude=latitude, longitude=longitude,
        report_type=effective_type, severity=severity,
        description=description, image_url=image_url,
        flood_predicted=flood_predicted,
        flood_confidence=flood_confidence,
        ai_label=ai_result["label"],
        ai_confidence=ai_result["confidence"] or None,
        ai_details=ai_result["details"],
        is_fake=False,
        fake_reason=location_check["fake_reason"],
        expected_ward_id=location_check["expected_ward_id"],
        expected_ward_name=location_check["expected_ward_name"],
        location_distance_km=location_check["distance_km"],
    )

    # 5. Broadcast alert if flood detected
    if flood_predicted and not location_check["location_mismatch"]:
        await _broadcast_flood_alert(
            ward_id=resolved_ward_id, ward_name=ward_name,
            confidence=flood_confidence, report_id=report_id,
            image_url=image_url,
        )

    return {
        "status":           "accepted",
        "report_id":        report_id,
        "ward_id":          resolved_ward_id,
        "ward_name":        ward_name,
        "report_type":      effective_type,
        "severity":         severity,
        "image_url":        image_url,
        "flood_predicted":  flood_predicted,
        "flood_confidence": flood_confidence,
        "ai_label":         ai_result["label"],
        "ai_confidence":    ai_result["confidence"],
        "ai_details":       ai_result["details"],
        "ai_corrected_report_type": effective_type != report_type,
        "location_check":   location_check,
        "alert_sent":       flood_predicted and not location_check["location_mismatch"],
        "submitted_at":     datetime.now(timezone.utc).isoformat(),
        "message": (
            f"⚠️ Flood detected ({flood_confidence*100:.0f}% confidence)! "
            f"Alert sent to all citizens and BBMP. Ward selected automatically from location."
            if flood_predicted
            else "Thank you! Ward was selected automatically from your location."
        ),
    }


# ---------------------------------------------------------------------------
# GET /reports/recent
# ---------------------------------------------------------------------------

@router.get("/recent")
async def recent_reports(limit: int = 20, _auth = Depends(require_role("admin"))):
    """Return the most recent citizen reports for the BBMP dashboard."""
    try:
        import psycopg2
        conn = psycopg2.connect(get_settings().postgres_dsn)
        _ensure_table(conn)
        # Ensure new columns exist before querying them
        with conn.cursor() as cur:
            for col, defn in [
                ("is_fake",        "BOOLEAN DEFAULT FALSE"),
                ("bbmp_confirmed", "BOOLEAN DEFAULT FALSE"),
                ("ai_label",       "TEXT"),
                ("ai_confidence",  "DOUBLE PRECISION"),
                ("ai_details",     "JSONB"),
                ("fake_reason",    "TEXT"),
                ("expected_ward_id", "INTEGER"),
                ("expected_ward_name", "TEXT"),
                ("location_distance_km", "DOUBLE PRECISION"),
            ]:
                try:
                    cur.execute(f"ALTER TABLE citizen_reports ADD COLUMN IF NOT EXISTS {col} {defn}")
                except Exception:
                    pass
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ward_id, report_type, severity, description,
                       image_url, flood_predicted, flood_confidence, reported_at,
                       COALESCE(is_fake, FALSE), COALESCE(bbmp_confirmed, FALSE),
                       ai_label, ai_confidence, ai_details,
                       fake_reason, expected_ward_id, expected_ward_name,
                       location_distance_km
                FROM   citizen_reports
                ORDER  BY reported_at DESC
                LIMIT  %s
            """, (limit,))
            rows = cur.fetchall()
        reports = []
        for r in rows:
            report = {
                "id":               r[0],
                "ward_id":          r[1],
                "report_type":      r[2],
                "severity":         r[3],
                "description":      r[4],
                "image_url":        r[5],
                "flood_predicted":  r[6],
                "flood_confidence": round(float(r[7]), 3) if r[7] else None,
                "reported_at":      r[8].isoformat() if r[8] else None,
                "is_fake":          r[9],
                "bbmp_confirmed":   r[10],
                "ai_label":         r[11],
                "ai_confidence":    round(float(r[12]), 3) if r[12] else None,
                "ai_details":       r[13],
                "fake_reason":      r[14],
                "expected_ward_id":  r[15],
                "expected_ward_name": r[16],
                "location_distance_km": round(float(r[17]), 3) if r[17] else None,
            }
            if report["report_type"] in {
                "air_quality", "heat_wave", "tree_fall", "waterlogging", "flood"
            } and report["image_url"]:
                ai = _analyse_saved_report_image(report["report_type"], report["image_url"])
                if ai:
                    derived = _derive_effective_ai_result(report["report_type"], ai)
                    effective_type = derived["effective_type"]
                    flood_predicted = derived["flood_predicted"]
                    flood_confidence = derived["flood_confidence"]

                    report["ai_label"] = ai["label"]
                    report["ai_confidence"] = ai["confidence"] or None
                    report["ai_details"] = ai["details"]
                    report["report_type"] = effective_type
                    report["flood_predicted"] = flood_predicted
                    report["flood_confidence"] = flood_confidence

                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE citizen_reports
                            SET report_type = %s,
                                flood_predicted = %s,
                                flood_confidence = %s,
                                ai_label = %s,
                                ai_confidence = %s,
                                ai_details = %s
                            WHERE id = %s
                        """, (
                            effective_type,
                            flood_predicted,
                            flood_confidence,
                            ai["label"],
                            ai["confidence"] or None,
                            json.dumps(ai["details"]),
                            report["id"],
                        ))
                    conn.commit()
            reports.append(report)
        conn.close()
    except Exception:
        reports = []

    return {"reports": reports, "total": len(reports)}


# ---------------------------------------------------------------------------
# DELETE /reports/{report_id}  — BBMP: remove a report
# ---------------------------------------------------------------------------

@router.delete("/{report_id}")
async def delete_report(report_id: int, _auth = Depends(require_role("admin"))):
    """
    BBMP officials can permanently delete a citizen report.
    Also clears any associated feedback entry.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(get_settings().postgres_dsn)
        _ensure_table(conn)
        with conn.cursor() as cur:
            # Fetch image_url first (to optionally delete file)
            cur.execute("SELECT image_url FROM citizen_reports WHERE id = %s", (report_id,))
            row = cur.fetchone()
            if not row:
                conn.close()
                raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
            image_url = row[0]

            cur.execute("DELETE FROM citizen_reports WHERE id = %s", (report_id,))
        conn.commit()
        conn.close()

        # Clear feedback entry for this report
        from image_classifier import record_feedback
        record_feedback(report_id, "clear")

        # Optionally delete the uploaded file
        if image_url:
            filename = image_url.split("/")[-1]
            filepath = os.path.join(UPLOAD_DIR, filename)
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass

        return {"status": "deleted", "report_id": report_id}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


# ---------------------------------------------------------------------------
# POST /reports/{report_id}/flag-fake  — BBMP: mark as fake → retrain
# ---------------------------------------------------------------------------

@router.post("/{report_id}/flag-fake")
async def flag_fake(report_id: int, _auth = Depends(require_role("admin"))):
    """
    BBMP marks a citizen report as fake/incorrect.
    Feedback is recorded and the classifier bias is updated immediately.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(get_settings().postgres_dsn)
        _ensure_table(conn)

        # Ensure columns exist
        with conn.cursor() as cur:
            for col, defn in [
                ("is_fake",       "BOOLEAN DEFAULT FALSE"),
                ("fake_flagged_at", "TIMESTAMPTZ"),
            ]:
                try:
                    cur.execute(f"ALTER TABLE citizen_reports ADD COLUMN IF NOT EXISTS {col} {defn}")
                except Exception:
                    pass

            cur.execute("""
                UPDATE citizen_reports
                SET is_fake = TRUE, fake_flagged_at = NOW()
                WHERE id = %s
                RETURNING id, image_url, ward_id
            """, (report_id,))
            row = cur.fetchone()
            if not row:
                conn.close()
                raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
            _, image_url, ward_id = row
        conn.commit()
        conn.close()

        # Record feedback → update model bias
        from image_classifier import record_feedback
        image_path = os.path.join(UPLOAD_DIR, image_url.split("/")[-1]) if image_url else None
        stats = record_feedback(report_id, "fake", image_path)

        return {
            "status":      "flagged_fake",
            "report_id":   report_id,
            "ward_id":     ward_id,
            "model_update": {
                "bias":        stats["model_bias"],
                "fake_count":  stats["fake_count"],
                "real_count":  stats["real_count"],
                "message":     f"Model bias adjusted to {stats['model_bias']:+.4f} "
                               f"({stats['fake_count']} fakes, {stats['real_count']} reals in training set)",
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error: {exc}")


# ---------------------------------------------------------------------------
# POST /reports/{report_id}/confirm-real  — BBMP: confirm genuine flood
# ---------------------------------------------------------------------------

@router.post("/{report_id}/confirm-real")
async def confirm_real(report_id: int, _auth = Depends(require_role("admin"))):
    """
    BBMP confirms a citizen report is genuine flood.
    Provides positive feedback to the classifier.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(get_settings().postgres_dsn)
        _ensure_table(conn)

        with conn.cursor() as cur:
            for col, defn in [
                ("is_fake",         "BOOLEAN DEFAULT FALSE"),
                ("bbmp_confirmed",  "BOOLEAN DEFAULT FALSE"),
                ("confirmed_at",    "TIMESTAMPTZ"),
            ]:
                try:
                    cur.execute(f"ALTER TABLE citizen_reports ADD COLUMN IF NOT EXISTS {col} {defn}")
                except Exception:
                    pass

            cur.execute("""
                UPDATE citizen_reports
                SET bbmp_confirmed = TRUE, is_fake = FALSE, confirmed_at = NOW()
                WHERE id = %s
                RETURNING id, image_url, ward_id
            """, (report_id,))
            row = cur.fetchone()
            if not row:
                conn.close()
                raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
            _, image_url, ward_id = row
        conn.commit()
        conn.close()

        from image_classifier import record_feedback
        image_path = os.path.join(UPLOAD_DIR, image_url.split("/")[-1]) if image_url else None
        stats = record_feedback(report_id, "real", image_path)

        return {
            "status":    "confirmed_real",
            "report_id": report_id,
            "ward_id":   ward_id,
            "model_update": {
                "bias":       stats["model_bias"],
                "fake_count": stats["fake_count"],
                "real_count": stats["real_count"],
                "message":    f"Positive signal recorded. Bias: {stats['model_bias']:+.4f}",
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error: {exc}")


# ---------------------------------------------------------------------------
# GET /reports/model-stats  — feedback + bias info for BBMP dashboard
# ---------------------------------------------------------------------------

@router.get("/model-stats")
async def model_stats(_auth = Depends(require_role("admin"))):
    """Return current classifier feedback stats."""
    from image_classifier import get_feedback_stats
    return get_feedback_stats()


# ---------------------------------------------------------------------------
# GET /reports/ward/{ward_id}/count
# ---------------------------------------------------------------------------

@router.get("/ward/{ward_id}/count")
async def ward_report_count(ward_id: int, days: int = 7):
    if ward_id < 1 or ward_id > 198:
        raise HTTPException(status_code=404, detail=f"Ward {ward_id} not found")

    try:
        import psycopg2
        conn = psycopg2.connect(get_settings().postgres_dsn)
        _ensure_table(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT report_type, COUNT(*), AVG(severity), MAX(reported_at)
                FROM   citizen_reports
                WHERE  ward_id = %s
                  AND  reported_at >= NOW() - INTERVAL '%s days'
                GROUP  BY report_type
                ORDER  BY COUNT(*) DESC
            """, (ward_id, days))
            rows = cur.fetchall()
        conn.close()
        report_types = [
            {
                "report_type":  r[0],
                "count":        int(r[1]),
                "avg_severity": round(float(r[2]), 1),
                "latest_at":    r[3].isoformat() if r[3] else None,
            }
            for r in rows
        ]
        total = sum(r["count"] for r in report_types)
    except Exception:
        report_types, total = [], 0

    return {
        "ward_id":       ward_id,
        "days":          days,
        "total_reports": total,
        "by_type":       report_types,
        "queried_at":    datetime.now(timezone.utc).isoformat(),
    }
