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
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from auth import require_role
from config import get_settings

router = APIRouter(prefix="/reports", tags=["reports"])

UPLOAD_DIR = str(get_settings().upload_dir)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic model (JSON-body endpoint)
# ---------------------------------------------------------------------------

class CitizenReport(BaseModel):
    ward_id:     int            = Field(..., ge=1, le=198)
    latitude:    float          = Field(..., ge=12.77, le=13.18)
    longitude:   float          = Field(..., ge=77.35, le=77.82)
    report_type: str            = Field(...)
    severity:    int            = Field(..., ge=1, le=5)
    description: Optional[str] = Field(None, max_length=1000)
    image_url:   Optional[str] = Field(None, max_length=500)


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
                reported_at      TIMESTAMPTZ DEFAULT NOW(),
                source           TEXT DEFAULT 'api'
            )
        """)
        # Add new columns to existing tables (idempotent)
        for col, defn in [
            ("flood_predicted",  "BOOLEAN DEFAULT FALSE"),
            ("flood_confidence", "DOUBLE PRECISION"),
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
                     description, image_url, flood_predicted, flood_confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                ward_id, latitude, longitude, report_type, severity,
                description, image_url, flood_predicted, flood_confidence,
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
# POST /reports/  (JSON body — legacy)
# ---------------------------------------------------------------------------

@router.post("/")
async def submit_report(body: CitizenReport):
    report_id = _save_report(
        ward_id=body.ward_id, latitude=body.latitude, longitude=body.longitude,
        report_type=body.report_type, severity=body.severity,
        description=body.description, image_url=body.image_url,
    )
    return {
        "status":       "accepted",
        "report_id":    report_id,
        "ward_id":      body.ward_id,
        "report_type":  body.report_type,
        "severity":     body.severity,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "message":      "Thank you! Your report has been recorded.",
    }


# ---------------------------------------------------------------------------
# POST /reports/with-image  (multipart — image + flood AI)
# ---------------------------------------------------------------------------

@router.post("/with-image")
async def submit_report_with_image(
    ward_id:     int            = Form(...),
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

    # 3. Run flood classifier
    from image_classifier import classify_image
    result = classify_image(image_bytes)

    flood_predicted  = result["is_flood"]
    flood_confidence = result["confidence"]
    effective_type   = "flood" if flood_predicted and report_type not in ("flood", "waterlogging") \
                       else report_type

    # 4. Save to DB
    from services import _ward_catalogue
    names     = _ward_catalogue()
    ward_name = names.get(ward_id, f"Ward {ward_id}")

    report_id = _save_report(
        ward_id=ward_id, latitude=latitude, longitude=longitude,
        report_type=effective_type, severity=severity,
        description=description, image_url=image_url,
        flood_predicted=flood_predicted,
        flood_confidence=flood_confidence,
    )

    # 5. Broadcast alert if flood detected
    if flood_predicted:
        await _broadcast_flood_alert(
            ward_id=ward_id, ward_name=ward_name,
            confidence=flood_confidence, report_id=report_id,
            image_url=image_url,
        )

    return {
        "status":           "accepted",
        "report_id":        report_id,
        "ward_id":          ward_id,
        "report_type":      effective_type,
        "severity":         severity,
        "image_url":        image_url,
        "flood_predicted":  flood_predicted,
        "flood_confidence": flood_confidence,
        "ai_label":         result["label"],
        "ai_details":       result["details"],
        "alert_sent":       flood_predicted,
        "submitted_at":     datetime.now(timezone.utc).isoformat(),
        "message": (
            f"⚠️ Flood detected ({flood_confidence*100:.0f}% confidence)! "
            f"Alert sent to all citizens and BBMP."
            if flood_predicted
            else "Thank you! Your report has been recorded."
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
                       COALESCE(is_fake, FALSE), COALESCE(bbmp_confirmed, FALSE)
                FROM   citizen_reports
                ORDER  BY reported_at DESC
                LIMIT  %s
            """, (limit,))
            rows = cur.fetchall()
        conn.close()
        reports = [
            {
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
            }
            for r in rows
        ]
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
