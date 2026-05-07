"""
Module 1f — Real Weather Data Ingestion
=========================================

Fetches current weather from the Open-Meteo free API for 10 representative
Bengaluru ward centroids and:

  1. Writes readings to PostgreSQL table `weather_readings`
  2. Publishes JSON events to Kafka topic `iot-sensor-stream`

Variables fetched: temperature_2m, relative_humidity_2m, precipitation,
wind_speed_10m, surface_pressure, cloud_cover

Usage:
    python 1_data_pipeline/weather_ingestion.py
    python 1_data_pipeline/weather_ingestion.py --loop      # hourly continuous
    python 1_data_pipeline/weather_ingestion.py --dry-run   # print only
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
POSTGRES_DSN  = "postgresql://ncg_user:change_me_in_production@localhost:5432/namma_clima_grid"
KAFKA_TOPIC   = "iot-sensor-stream"
KAFKA_SERVERS = "localhost:9092"

WEATHER_VARIABLES = (
    "temperature_2m,relative_humidity_2m,precipitation,"
    "wind_speed_10m,surface_pressure,cloud_cover"
)

# 10 representative Bengaluru ward centroids
WARD_CENTROIDS: List[Dict] = [
    {"ward_id": 12,  "ward_name": "MG Road / CBD",       "lat": 12.9756, "lon": 77.6050},
    {"ward_id": 27,  "ward_name": "Koramangala",          "lat": 12.9352, "lon": 77.6245},
    {"ward_id": 33,  "ward_name": "HSR Layout",           "lat": 12.9116, "lon": 77.6389},
    {"ward_id": 45,  "ward_name": "Electronic City",      "lat": 12.8452, "lon": 77.6602},
    {"ward_id": 58,  "ward_name": "Whitefield",           "lat": 12.9698, "lon": 77.7500},
    {"ward_id": 71,  "ward_name": "Hebbal",               "lat": 13.0359, "lon": 77.5970},
    {"ward_id": 89,  "ward_name": "Yelahanka",            "lat": 13.1007, "lon": 77.5963},
    {"ward_id": 102, "ward_name": "Peenya Industrial",    "lat": 13.0287, "lon": 77.5200},
    {"ward_id": 115, "ward_name": "Rajajinagar",          "lat": 12.9916, "lon": 77.5529},
    {"ward_id": 130, "ward_name": "Jayanagar",            "lat": 12.9250, "lon": 77.5938},
]


# ---------------------------------------------------------------------------
# Open-Meteo fetch
# ---------------------------------------------------------------------------

def fetch_weather_for_ward(ward: Dict, client: httpx.Client) -> Optional[Dict]:
    """Fetch current weather for a single ward centroid."""
    try:
        resp = client.get(
            OPENMETEO_URL,
            params={
                "latitude":  ward["lat"],
                "longitude": ward["lon"],
                "current":   WEATHER_VARIABLES,
                "timezone":  "Asia/Kolkata",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})

        reading = {
            "ward_id":          ward["ward_id"],
            "ward_name":        ward["ward_name"],
            "lat":              ward["lat"],
            "lon":              ward["lon"],
            "timestamp":        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "temperature_2m":   float(current.get("temperature_2m", 28.0)),
            "humidity_pct":     float(current.get("relative_humidity_2m", 60.0)),
            "precipitation_mm": float(current.get("precipitation", 0.0)),
            "wind_speed_kmh":   float(current.get("wind_speed_10m", 5.0)),
            "surface_pressure": float(current.get("surface_pressure", 1013.0)),
            "cloud_cover_pct":  float(current.get("cloud_cover", 30.0)),
            "source":           "open-meteo",
        }
        return reading
    except Exception as exc:
        logger.warning(f"Open-Meteo fetch failed for ward {ward['ward_id']}: {exc}")
        return None


def fetch_all_wards(client: httpx.Client) -> List[Dict]:
    """Fetch weather for all ward centroids."""
    readings = []
    for ward in WARD_CENTROIDS:
        reading = fetch_weather_for_ward(ward, client)
        if reading:
            readings.append(reading)
            logger.debug(
                f"Ward {ward['ward_id']:3d} | T={reading['temperature_2m']:.1f}°C  "
                f"H={reading['humidity_pct']:.0f}%  "
                f"P={reading['precipitation_mm']:.1f}mm"
            )
        else:
            logger.warning(f"Skipping ward {ward['ward_id']} (fetch failed)")
    return readings


# ---------------------------------------------------------------------------
# PostgreSQL writer
# ---------------------------------------------------------------------------

def ensure_table(conn) -> None:
    """Create weather_readings table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS weather_readings (
                id                SERIAL PRIMARY KEY,
                ward_id           INTEGER NOT NULL,
                ward_name         TEXT,
                lat               DOUBLE PRECISION,
                lon               DOUBLE PRECISION,
                timestamp         TIMESTAMPTZ NOT NULL,
                temperature_2m    DOUBLE PRECISION,
                humidity_pct      DOUBLE PRECISION,
                precipitation_mm  DOUBLE PRECISION,
                wind_speed_kmh    DOUBLE PRECISION,
                surface_pressure  DOUBLE PRECISION,
                cloud_cover_pct   DOUBLE PRECISION,
                source            TEXT DEFAULT 'open-meteo',
                ingested_at       TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_weather_ward_ts
            ON weather_readings (ward_id, timestamp DESC)
        """)
    conn.commit()


def write_to_postgres(readings: List[Dict]) -> None:
    """Insert weather readings into PostgreSQL."""
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        logger.warning("psycopg2 not installed — skipping PostgreSQL write")
        return

    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        ensure_table(conn)

        cols = [
            "ward_id", "ward_name", "lat", "lon", "timestamp",
            "temperature_2m", "humidity_pct", "precipitation_mm",
            "wind_speed_kmh", "surface_pressure", "cloud_cover_pct", "source",
        ]
        rows = [tuple(r[c] for c in cols) for r in readings]
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"INSERT INTO weather_readings ({', '.join(cols)}) VALUES %s",
                rows,
            )
        conn.commit()
        conn.close()
        logger.success(f"Inserted {len(rows)} rows into weather_readings (PostgreSQL)")
    except Exception as exc:
        logger.error(f"PostgreSQL write failed: {exc}")


# ---------------------------------------------------------------------------
# Kafka publisher
# ---------------------------------------------------------------------------

def publish_to_kafka(readings: List[Dict]) -> bool:
    """Publish readings to Kafka iot-sensor-stream topic. Returns True if successful."""
    try:
        from kafka import KafkaProducer
    except ImportError:
        logger.warning("kafka-python not installed — skipping Kafka publish")
        return False

    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            request_timeout_ms=5000,
            api_version_auto_timeout_ms=5000,
        )
        for reading in readings:
            producer.send(
                KAFKA_TOPIC,
                key=str(reading["ward_id"]),
                value=reading,
            )
        producer.flush(timeout=10)
        producer.close()
        logger.success(f"Published {len(readings)} weather readings to Kafka topic '{KAFKA_TOPIC}'")
        return True
    except Exception as exc:
        logger.warning(f"Kafka publish failed ({exc}) — continuing without Kafka")
        return False


# ---------------------------------------------------------------------------
# Ward risk score refresh — writes live scores back to ward_risk_scores
# ---------------------------------------------------------------------------

def refresh_ward_risk_scores(readings: List[Dict]) -> None:
    """
    Compute and UPSERT heat_stress_score + flood_risk_score into the
    ward_risk_scores table based on the latest weather readings.

    Formula (physics-informed, no ML needed for the 10 observed wards;
    the remaining 188 wards keep their existing model-generated values):

      heat_stress = temperature_percentile_score + humidity_penalty
        temperature_percentile_score = clip((T - 28) / (42 - 28) * 100, 0, 100)
        humidity_penalty             = clip((H - 30) / 70 * 15, 0, 15)

      flood_risk  = precipitation_score
        precipitation_score = clip(P / 50 * 100, 0, 100)
        (50 mm in current hour = 100% flood risk)

    risk_level thresholds: combined = 0.6*heat + 0.4*flood
      >= 75 → critical | >= 50 → high | >= 25 → medium | else low
    """
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        logger.warning("psycopg2 not available — skipping ward_risk_scores refresh")
        return

    if not readings:
        return

    rows = []
    for r in readings:
        t    = float(r.get("temperature_2m", 30))
        h    = float(r.get("humidity_pct", 50))
        prec = float(r.get("precipitation_mm", 0))

        heat = min(100.0, max(0.0, (t - 28) / (42 - 28) * 100))
        heat += min(15.0, max(0.0, (h - 30) / 70 * 15))
        heat = min(100.0, heat)

        flood = min(100.0, max(0.0, prec / 20 * 100))

        combined = 0.6 * heat + 0.4 * flood
        if combined >= 75:
            risk_level = "critical"
        elif combined >= 50:
            risk_level = "high"
        elif combined >= 25:
            risk_level = "medium"
        else:
            risk_level = "low"

        rows.append((
            r["ward_id"],
            round(heat, 2),
            round(flood, 2),
            risk_level,
            float(r.get("temperature_2m", t)),  # lst_pred_celsius proxy
        ))

    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO ward_risk_scores
                    (ward_id, heat_stress_score, flood_risk_score, risk_level,
                     lst_pred_celsius, updated_at)
                VALUES %s
                ON CONFLICT (ward_id)
                DO UPDATE SET
                    heat_stress_score = EXCLUDED.heat_stress_score,
                    flood_risk_score  = EXCLUDED.flood_risk_score,
                    risk_level        = EXCLUDED.risk_level,
                    lst_pred_celsius  = EXCLUDED.lst_pred_celsius,
                    updated_at        = now()
                """,
                rows,
                template="(%s, %s, %s, %s, %s, now())",
            )
        conn.commit()
        conn.close()
        logger.success(
            f"Refreshed ward_risk_scores for {len(rows)} wards "
            f"(wards: {[r[0] for r in rows]})"
        )
    except Exception as exc:
        logger.error(f"ward_risk_scores refresh failed: {exc}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_once(dry_run: bool = False) -> List[Dict]:
    """Fetch weather for all wards and write to Postgres + Kafka."""
    logger.info(f"Starting weather ingestion for {len(WARD_CENTROIDS)} wards...")

    with httpx.Client() as client:
        readings = fetch_all_wards(client)

    if not readings:
        logger.error("No readings fetched — check network / Open-Meteo API")
        return []

    logger.info(f"Fetched {len(readings)} weather readings")

    if dry_run:
        for r in readings:
            logger.info(f"  DRY-RUN ward={r['ward_id']} T={r['temperature_2m']:.1f}°C")
        return readings

    write_to_postgres(readings)
    publish_to_kafka(readings)
    refresh_ward_risk_scores(readings)   # ← update live scores in DB

    return readings


def run_loop() -> None:
    """Run weather ingestion in a continuous hourly loop."""
    logger.info("Starting continuous hourly weather ingestion loop...")
    while True:
        try:
            readings = run_once()
            logger.info(f"Cycle complete | {len(readings)} readings | sleeping 3600s")
        except KeyboardInterrupt:
            logger.info("Weather ingestion loop stopped by user")
            break
        except Exception as exc:
            logger.error(f"Ingestion cycle failed: {exc}")
        time.sleep(3600)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid — Real weather data ingestion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--loop",    action="store_true",
                   help="Run continuously every hour (default: run once and exit)")
    p.add_argument("--dry-run", action="store_true",
                   help="Fetch and print without writing to Postgres or Kafka")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.loop:
        run_loop()
    else:
        readings = run_once(dry_run=args.dry_run)
        logger.success(f"Weather ingestion complete | {len(readings)} wards processed")
