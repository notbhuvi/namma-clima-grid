"""
Module 1a — Kafka Producer (IoT Sensor Stream)
===============================================

Simulates 10 virtual IoT sensor nodes distributed across real Bengaluru
ward locations. For every emission cycle it:

  1. Fetches current weather from the Open-Meteo free API for each node's
     coordinates (temperature, humidity, precipitation).
  2. Adds realistic sensor noise + diurnal variation.
  3. Synthesises PM2.5 from a log-normal distribution (wards near the
     Peenya industrial belt run hotter and dirtier).
  4. Publishes a JSON message to the Kafka topic `iot-sensor-stream`.
  5. If Kafka is unreachable, falls back to appending readings to a CSV
     file so downstream modules always have data to work with.

Message schema:
    {
        "ward_id":       int,
        "sensor_id":     "SENSOR-0007",
        "timestamp":     "2026-04-09T08:32:05Z",
        "lat":           12.9716,
        "lon":           77.5946,
        "temperature_c": 31.4,
        "humidity_pct":  64.2,
        "pm25":          48.7,
        "rainfall_mm":   0.0
    }

Usage:
    # Stream forever to Kafka (default):
    python 1_data_pipeline/kafka_producer.py

    # Run for 5 minutes then exit:
    python 1_data_pipeline/kafka_producer.py --duration 300

    # Force offline CSV mode:
    python 1_data_pipeline/kafka_producer.py --offline

    # Faster demo (10 s interval):
    python 1_data_pipeline/kafka_producer.py --interval 10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class ProducerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_iot: str = "iot-sensor-stream"
    openmeteo_base_url: str = "https://api.open-meteo.com/v1/forecast"


_cfg = ProducerSettings()

CSV_FALLBACK = Path(__file__).parent.parent / "8_data" / "mock_iot_stream.csv"
CSV_FIELDS = [
    "ward_id", "sensor_id", "timestamp", "lat", "lon",
    "temperature_c", "humidity_pct", "pm25", "rainfall_mm",
]

# ---------------------------------------------------------------------------
# Sensor catalogue — 10 fixed nodes at real Bengaluru landmarks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorNode:
    sensor_id: str
    ward_id: int
    ward_name: str
    lat: float
    lon: float
    # Industrial proximity weight 0..1 — controls PM2.5 baseline
    industrial_weight: float = 0.2


SENSOR_NODES: List[SensorNode] = [
    SensorNode("SENSOR-0001", 12,  "MG Road / CBD",        12.9756, 77.6050, 0.30),
    SensorNode("SENSOR-0002", 27,  "Koramangala",          12.9352, 77.6245, 0.15),
    SensorNode("SENSOR-0003", 33,  "HSR Layout",           12.9116, 77.6389, 0.10),
    SensorNode("SENSOR-0004", 45,  "Electronic City",      12.8452, 77.6602, 0.60),
    SensorNode("SENSOR-0005", 58,  "Whitefield",           12.9698, 77.7500, 0.40),
    SensorNode("SENSOR-0006", 71,  "Hebbal",               13.0359, 77.5970, 0.20),
    SensorNode("SENSOR-0007", 89,  "Yelahanka",            13.1007, 77.5963, 0.15),
    SensorNode("SENSOR-0008", 102, "Peenya Industrial",    13.0287, 77.5200, 0.95),
    SensorNode("SENSOR-0009", 115, "Rajajinagar",          12.9916, 77.5529, 0.35),
    SensorNode("SENSOR-0010", 130, "Jayanagar",            12.9250, 77.5938, 0.10),
]

# ---------------------------------------------------------------------------
# Open-Meteo fetcher
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=False,
)
def _fetch_weather(
    lat: float, lon: float, client: httpx.Client
) -> Optional[Dict[str, float]]:
    """
    Fetch current weather from Open-Meteo for one coordinate pair.
    Returns None on failure (caller uses synthesised fallback).
    """
    try:
        resp = client.get(
            _cfg.openmeteo_base_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation",
                "timezone": "Asia/Kolkata",
            },
            timeout=8.0,
        )
        resp.raise_for_status()
        current = resp.json().get("current") or {}
        return {
            "temperature_c": float(current.get("temperature_2m", 28.0)),
            "humidity_pct":  float(current.get("relative_humidity_2m", 60.0)),
            "rainfall_mm":   float(current.get("precipitation", 0.0)),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Open-Meteo failed for ({lat},{lon}): {exc}")
        return None


def _diurnal_weather() -> Dict[str, float]:
    """Synthesise a plausible current reading when Open-Meteo is unavailable."""
    hour = datetime.now(timezone.utc).astimezone().hour
    phase = math.sin((hour - 8) / 24.0 * 2 * math.pi)
    return {
        "temperature_c": round(27.0 + 6.0 * phase + random.gauss(0, 0.5), 2),
        "humidity_pct":  round(max(25.0, 68.0 - 8.0 * phase + random.gauss(0, 2)), 2),
        "rainfall_mm":   round(max(0.0, random.gauss(0.0, 0.2)), 2),
    }


# ---------------------------------------------------------------------------
# Reading builder
# ---------------------------------------------------------------------------
def build_reading(
    node: SensorNode,
    client: Optional[httpx.Client],
) -> Dict[str, object]:
    """
    Compose a complete sensor payload for one node.

    - Tries Open-Meteo first; falls back to diurnal synthesis.
    - Adds per-sensor Gaussian noise so readings are distinct.
    - Synthesises PM2.5 from a log-normal distribution biased by
      industrial_weight.
    """
    if client is not None:
        weather = _fetch_weather(node.lat, node.lon, client)
    else:
        weather = None

    if weather is None:
        weather = _diurnal_weather()

    temperature = weather["temperature_c"] + random.gauss(0, 0.4)
    humidity    = max(0.0, min(100.0, weather["humidity_pct"] + random.gauss(0, 1.5)))
    rainfall    = max(0.0, weather["rainfall_mm"] + abs(random.gauss(0, 0.05)))

    # PM2.5: log-normal baseline shifted upward for industrial wards
    pm25_mean = 3.2 + 0.8 * node.industrial_weight
    pm25 = max(5.0, random.lognormvariate(pm25_mean, 0.35))

    return {
        "ward_id":       node.ward_id,
        "sensor_id":     node.sensor_id,
        "timestamp":     datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lat":           node.lat,
        "lon":           node.lon,
        "temperature_c": round(temperature, 2),
        "humidity_pct":  round(humidity, 2),
        "pm25":          round(pm25, 2),
        "rainfall_mm":   round(rainfall, 3),
    }


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------
class KafkaSink:
    """Publish readings to Kafka. Fails fast if the broker is unreachable."""

    def __init__(self) -> None:
        from confluent_kafka import Producer  # type: ignore

        self._producer = Producer({
            "bootstrap.servers": _cfg.kafka_bootstrap_servers,
            "client.id": "ncg-iot-producer",
            "linger.ms": 100,
            "acks": "1",
        })
        # Probe connectivity — raises if broker is down
        self._producer.list_topics(timeout=5.0)
        logger.success(
            f"KafkaSink connected → {_cfg.kafka_bootstrap_servers} "
            f"topic={_cfg.kafka_topic_iot}"
        )

    def _on_delivery(self, err, msg) -> None:
        if err:
            logger.error(f"Kafka delivery error: {err}")

    def emit(self, reading: Dict[str, object]) -> None:
        self._producer.produce(
            _cfg.kafka_topic_iot,
            key=str(reading["sensor_id"]),
            value=json.dumps(reading),
            callback=self._on_delivery,
        )
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush(5.0)

    def close(self) -> None:
        self.flush()


class CsvSink:
    """Append readings to a CSV file (offline fallback)."""

    def __init__(self, path: Path = CSV_FALLBACK) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        self._fh = path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        if new_file:
            self._writer.writeheader()
        logger.success(f"CsvSink → {path}")

    def emit(self, reading: Dict[str, object]) -> None:
        self._writer.writerow({k: reading[k] for k in CSV_FIELDS})
        self._fh.flush()

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def make_sink(offline: bool) -> KafkaSink | CsvSink:
    if offline:
        return CsvSink()
    try:
        return KafkaSink()
    except Exception as exc:
        logger.warning(f"Kafka unavailable ({exc}) — falling back to CSV")
        return CsvSink()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(interval: float, duration: Optional[float], offline: bool) -> None:
    """
    Emit one batch of readings (one per sensor) every `interval` seconds.

    Args:
        interval: seconds between batches.
        duration: total runtime in seconds; None = run forever.
        offline:  force CSV sink.
    """
    sink = make_sink(offline)
    stop_flag = {"v": False}

    def _handle_signal(sig, _frame) -> None:
        logger.warning(f"Signal {sig} received, shutting down.")
        stop_flag["v"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    t0 = time.monotonic()
    tick = 0

    with httpx.Client() as http:
        while not stop_flag["v"]:
            tick += 1
            batch_ts = datetime.now(timezone.utc).strftime("%H:%M:%S")

            for node in SENSOR_NODES:
                reading = build_reading(node, http)
                sink.emit(reading)
                logger.debug(
                    f"[{batch_ts}] {node.sensor_id}  "
                    f"T={reading['temperature_c']:5.1f}°C  "
                    f"H={reading['humidity_pct']:5.1f}%  "
                    f"R={reading['rainfall_mm']:4.2f}mm  "
                    f"PM2.5={reading['pm25']:6.1f}"
                )

            sink.flush()
            logger.info(
                f"Tick {tick:4d} | {batch_ts} | "
                f"published {len(SENSOR_NODES)} readings"
            )

            if duration is not None and (time.monotonic() - t0) >= duration:
                logger.info("Duration limit reached.")
                break

            # Interruptible sleep in 0.5s chunks
            slept = 0.0
            while slept < interval and not stop_flag["v"]:
                time.sleep(min(0.5, interval - slept))
                slept += 0.5

    sink.close()
    logger.success("Kafka producer stopped cleanly.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid — IoT sensor Kafka producer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--interval", type=float, default=60.0,
                   help="Seconds between emission cycles")
    p.add_argument("--duration", type=float, default=None,
                   help="Total runtime in seconds (omit for forever)")
    p.add_argument("--offline", action="store_true",
                   help="Force CSV sink; skip Kafka entirely")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logger.info(
        f"NammaClimaGrid IoT Producer starting | "
        f"nodes={len(SENSOR_NODES)} interval={args.interval}s "
        f"duration={args.duration} offline={args.offline}"
    )
    try:
        run(interval=args.interval, duration=args.duration, offline=args.offline)
    except KeyboardInterrupt:
        sys.exit(0)
