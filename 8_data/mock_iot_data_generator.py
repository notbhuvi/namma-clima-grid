"""
Module 7a — Mock IoT Data Generator
===================================

Simulates 10 virtual IoT sensor nodes placed across real Bengaluru coordinates.
For each node we fetch a live weather snapshot from the Open-Meteo free API,
layer on realistic noise + a diurnal cycle, and emit readings to Kafka.

If Kafka is unreachable, readings are appended to a rolling CSV instead so
the pipeline still produces data in an offline environment.

Reading schema (JSON):
    {
        "ward_id":        int,
        "sensor_id":      "SENSOR-0007",
        "timestamp":      "2026-04-09T14:32:05Z",
        "lat":            12.9716,
        "lon":            77.5946,
        "temperature_c":  31.4,
        "humidity_pct":   64.2,
        "pm25":           48.7,
        "rainfall_mm":    0.0
    }

Usage:
    # Stream to Kafka forever (default):
    python 8_data/mock_iot_data_generator.py

    # Run for 15 minutes then exit:
    python 8_data/mock_iot_data_generator.py --duration 900

    # Force offline mode (no Kafka) — write CSV only:
    python 8_data/mock_iot_data_generator.py --offline

    # Emit faster (useful for demos):
    python 8_data/mock_iot_data_generator.py --interval 5
"""
from __future__ import annotations

import argparse
import csv
import json
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
from tenacity import retry, stop_after_attempt, wait_exponential

from _common import MODULE_DIR, logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_IOT", "iot-sensor-stream")
OPENMETEO_URL = os.getenv(
    "OPENMETEO_BASE_URL", "https://api.open-meteo.com/v1/forecast"
)

CSV_FALLBACK_PATH = MODULE_DIR / "mock_iot_stream.csv"
CSV_FIELDS = [
    "ward_id", "sensor_id", "timestamp", "lat", "lon",
    "temperature_c", "humidity_pct", "pm25", "rainfall_mm",
]


# ---------------------------------------------------------------------------
# Sensor catalogue — 10 real Bengaluru locations, mapped to synthetic ward_ids
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorNode:
    sensor_id: str
    ward_id: int
    ward_name: str
    lat: float
    lon: float


SENSOR_NODES: List[SensorNode] = [
    SensorNode("SENSOR-0001", 12,  "MG Road (CBD)",        12.9756, 77.6050),
    SensorNode("SENSOR-0002", 27,  "Koramangala",          12.9352, 77.6245),
    SensorNode("SENSOR-0003", 33,  "HSR Layout",           12.9116, 77.6389),
    SensorNode("SENSOR-0004", 45,  "Electronic City",      12.8452, 77.6602),
    SensorNode("SENSOR-0005", 58,  "Whitefield",           12.9698, 77.7500),
    SensorNode("SENSOR-0006", 71,  "Hebbal",               13.0359, 77.5970),
    SensorNode("SENSOR-0007", 89,  "Yelahanka",            13.1007, 77.5963),
    SensorNode("SENSOR-0008", 102, "Peenya Industrial",    13.0287, 77.5200),
    SensorNode("SENSOR-0009", 115, "Rajajinagar",          12.9916, 77.5529),
    SensorNode("SENSOR-0010", 130, "Jayanagar",            12.9250, 77.5938),
]


# ---------------------------------------------------------------------------
# Open-Meteo client (with retry/backoff)
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _fetch_open_meteo(lat: float, lon: float, client: httpx.Client) -> Dict[str, float]:
    """Fetch current-weather snapshot from Open-Meteo. Raises on failure."""
    resp = client.get(
        OPENMETEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation",
            "timezone": "Asia/Kolkata",
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    current = resp.json().get("current", {}) or {}
    return {
        "temperature_c": float(current.get("temperature_2m", 28.0)),
        "humidity_pct":  float(current.get("relative_humidity_2m", 60.0)),
        "rainfall_mm":   float(current.get("precipitation", 0.0)),
    }


# ---------------------------------------------------------------------------
# Reading fabrication
# ---------------------------------------------------------------------------
def _synthesize_offline_weather(node: SensorNode) -> Dict[str, float]:
    """
    If Open-Meteo is unreachable, synthesise a plausible reading based on
    the current hour of day (diurnal cycle) so the stream keeps flowing.
    """
    hour = datetime.now(timezone.utc).astimezone().hour
    # Diurnal sinusoid peaking ~14:00 IST
    import math
    phase = math.sin((hour - 8) / 24.0 * 2 * math.pi)
    base_temp = 27.0 + 6.0 * phase
    return {
        "temperature_c": round(base_temp + random.gauss(0, 0.8), 2),
        "humidity_pct":  round(max(25.0, 70.0 - 10 * phase + random.gauss(0, 3)), 2),
        "rainfall_mm":   round(max(0.0, random.gauss(0.0, 0.3)), 2),
    }


def build_reading(node: SensorNode, client: Optional[httpx.Client]) -> Dict[str, object]:
    """Assemble a complete sensor payload for one node."""
    try:
        if client is None:
            raise RuntimeError("no http client")
        weather = _fetch_open_meteo(node.lat, node.lon, client)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{node.sensor_id}] Open-Meteo fallback: {exc}")
        weather = _synthesize_offline_weather(node)

    # Layer on sensor noise so each node's reading is distinct
    temperature = weather["temperature_c"] + random.gauss(0, 0.4)
    humidity    = max(0.0, min(100.0, weather["humidity_pct"] + random.gauss(0, 1.5)))
    rainfall    = max(0.0, weather["rainfall_mm"] + random.gauss(0, 0.05))
    # PM2.5 is not provided by free Open-Meteo current; synthesise realistically.
    pm25 = max(5.0, random.lognormvariate(3.4, 0.35))

    return {
        "ward_id":       node.ward_id,
        "sensor_id":     node.sensor_id,
        "timestamp":     datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lat":           node.lat,
        "lon":           node.lon,
        "temperature_c": round(temperature, 2),
        "humidity_pct":  round(humidity, 2),
        "pm25":          round(pm25, 2),
        "rainfall_mm":   round(rainfall, 2),
    }


# ---------------------------------------------------------------------------
# Sinks — Kafka primary, CSV fallback
# ---------------------------------------------------------------------------
class Sink:
    """Abstract sink. `emit()` must never raise."""

    def emit(self, reading: Dict[str, object]) -> None:  # pragma: no cover
        raise NotImplementedError

    def flush(self) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass


class KafkaSink(Sink):
    """Publishes readings to a Kafka topic via confluent-kafka."""

    def __init__(self, bootstrap: str, topic: str) -> None:
        try:
            from confluent_kafka import Producer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "confluent-kafka is not installed; install requirements.txt"
            ) from e

        self.topic = topic
        self._producer = Producer({
            "bootstrap.servers": bootstrap,
            "client.id": "namma-mock-iot",
            "linger.ms": 50,
        })
        # Probe the broker — fail fast so we can fall back to CSV.
        try:
            self._producer.list_topics(timeout=3.0)
        except Exception as e:
            raise RuntimeError(f"Kafka unreachable at {bootstrap}: {e}") from e
        logger.success(f"KafkaSink connected → {bootstrap} / topic={topic}")

    def _delivery(self, err, msg) -> None:
        if err is not None:
            logger.error(f"Kafka delivery failed: {err}")

    def emit(self, reading: Dict[str, object]) -> None:
        try:
            self._producer.produce(
                self.topic,
                key=str(reading["sensor_id"]),
                value=json.dumps(reading),
                callback=self._delivery,
            )
            self._producer.poll(0)
        except BufferError:
            self._producer.flush(2.0)

    def flush(self) -> None:
        self._producer.flush(5.0)

    def close(self) -> None:
        self.flush()


class CsvSink(Sink):
    """Appends readings to a CSV file. Used when Kafka is not available."""

    def __init__(self, path: Path) -> None:
        self.path = path
        new_file = not path.exists()
        self._fh = path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        if new_file:
            self._writer.writeheader()
        logger.success(f"CsvSink writing → {path}")

    def emit(self, reading: Dict[str, object]) -> None:
        self._writer.writerow({k: reading[k] for k in CSV_FIELDS})
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover
            pass


def make_sink(offline: bool) -> Sink:
    """Try Kafka first; fall back to CSV on any failure (or when --offline)."""
    if not offline:
        try:
            return KafkaSink(KAFKA_BOOTSTRAP, KAFKA_TOPIC)
        except Exception as e:
            logger.warning(f"Kafka unavailable ({e}) — falling back to CSV")
    return CsvSink(CSV_FALLBACK_PATH)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(interval: float, duration: Optional[float], offline: bool) -> None:
    """
    Emit one reading per sensor every `interval` seconds.

    Args:
        interval: seconds between emission cycles.
        duration: total runtime in seconds; None = run forever.
        offline:  force CSV sink.
    """
    sink = make_sink(offline)
    stop_flag = {"stop": False}

    def _handle_sig(signum, _frame) -> None:  # noqa: ANN001
        logger.warning(f"Received signal {signum}, shutting down...")
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    start_ts = time.time()
    tick = 0
    with httpx.Client() as client:
        while not stop_flag["stop"]:
            tick += 1
            for node in SENSOR_NODES:
                reading = build_reading(node, client)
                sink.emit(reading)
                logger.debug(
                    f"tick={tick} {reading['sensor_id']} "
                    f"T={reading['temperature_c']}°C "
                    f"H={reading['humidity_pct']}% "
                    f"R={reading['rainfall_mm']}mm"
                )
            sink.flush()
            logger.info(f"Tick {tick}: emitted {len(SENSOR_NODES)} readings")

            if duration is not None and (time.time() - start_ts) >= duration:
                logger.info("Duration reached, stopping.")
                break

            # Sleep in small chunks so Ctrl-C is responsive
            slept = 0.0
            while slept < interval and not stop_flag["stop"]:
                time.sleep(min(0.5, interval - slept))
                slept += 0.5

    sink.close()
    logger.success("IoT generator stopped cleanly.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock Bengaluru IoT sensor stream.")
    p.add_argument("--interval", type=float, default=60.0,
                   help="Seconds between emission cycles (default 60)")
    p.add_argument("--duration", type=float, default=None,
                   help="Total runtime in seconds (default: run forever)")
    p.add_argument("--offline", action="store_true",
                   help="Force CSV sink; don't even try Kafka")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info(
        f"Starting mock IoT generator | nodes={len(SENSOR_NODES)} "
        f"interval={args.interval}s duration={args.duration} offline={args.offline}"
    )
    try:
        run(interval=args.interval, duration=args.duration, offline=args.offline)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
