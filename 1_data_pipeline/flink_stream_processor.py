"""
Module 1b — Flink Stream Processor (PyFlink)
=============================================

Consumes the `iot-sensor-stream` Kafka topic, applies 15-minute tumbling
window aggregations per ward_id, detects climate anomalies, and writes:

  → ward-features-stream   : aggregated per-ward features every 15 min
  → climate-alerts          : anomaly events (T > 40°C or rainfall spike > 20mm)

Architecture
────────────
  Kafka (iot-sensor-stream)
       │
       ▼
  Flink DataStream (parse JSON)
       │
       ├─► 15-min TumblingEventTimeWindows per ward_id
       │         → aggregate: avg_temp, avg_humidity, avg_pm25,
       │                       sum_rainfall, count
       │         → emit to ward-features-stream
       │
       └─► Anomaly filter (T > 40°C or rainfall > 20mm)
                 → emit to climate-alerts

Offline fallback
────────────────
When Kafka is unreachable or PyFlink JARs are missing, the script falls
back to a pure-Python windowing loop that reads from the CSV fallback
file produced by kafka_producer.py --offline.

Usage:
    # Run against live Kafka + Flink:
    python 1_data_pipeline/flink_stream_processor.py

    # Force offline mode (reads CSV, no Kafka needed):
    python 1_data_pipeline/flink_stream_processor.py --offline

    # One-shot offline test (process existing CSV and exit):
    python 1_data_pipeline/flink_stream_processor.py --offline --once
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class FlinkSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_iot: str = "iot-sensor-stream"
    kafka_topic_ward_features: str = "ward-features-stream"
    kafka_topic_alerts: str = "climate-alerts"


_cfg = FlinkSettings()

WINDOW_MINUTES = 15  # tumbling window width
TEMP_ANOMALY_C = 40.0
RAIN_ANOMALY_MM = 20.0

# ---------------------------------------------------------------------------
# Jar / connector resolution
# ---------------------------------------------------------------------------

def _flink_jar_dir() -> Path:
    """Return the directory where Flink's built-in connectors live."""
    try:
        import pyflink  # type: ignore
        flink_home = Path(pyflink.__file__).parent
        jar_dir = flink_home / "lib"
        if jar_dir.exists():
            return jar_dir
    except ImportError:
        pass
    return Path("/opt/flink/lib")


def _connector_jars() -> str:
    """
    Build a semicolon-separated list of JAR URIs required by the Kafka
    SQL connector. Uses whatever is available in the PyFlink package; emits
    a warning (not an error) if jars are missing so the offline path can
    proceed.
    """
    jar_dir = _flink_jar_dir()
    patterns = [
        "flink-sql-connector-kafka*.jar",
        "flink-connector-kafka*.jar",
        "kafka-clients*.jar",
    ]
    found: List[str] = []
    for pat in patterns:
        for match in jar_dir.glob(pat):
            found.append(match.as_uri())
    if not found:
        logger.warning(
            f"No Kafka connector JARs found in {jar_dir}. "
            "Falling back to offline mode."
        )
    return ";".join(found)


# ---------------------------------------------------------------------------
# PyFlink path — runs when Kafka + Flink are available
# ---------------------------------------------------------------------------

def _run_flink() -> None:
    """
    Execute the full streaming job via PyFlink Table API + Kafka connector.
    """
    from pyflink.common import WatermarkStrategy, Duration, Types  # type: ignore
    from pyflink.datastream import StreamExecutionEnvironment  # type: ignore
    from pyflink.datastream.connectors.kafka import (  # type: ignore
        KafkaSource,
        KafkaSink,
        KafkaRecordSerializationSchema,
        KafkaOffsetsInitializer,
        DeliveryGuarantee,
    )
    from pyflink.datastream.window import TumblingEventTimeWindows, Time  # type: ignore
    from pyflink.datastream.functions import ProcessWindowFunction  # type: ignore
    from pyflink.common.serialization import SimpleStringSchema  # type: ignore

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.enable_checkpointing(60_000)  # checkpoint every 60 s

    jars = _connector_jars()
    if jars:
        env.add_jars(*jars.split(";"))

    # ── Source ──────────────────────────────────────────────────────────────
    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(_cfg.kafka_bootstrap_servers)
        .set_topics(_cfg.kafka_topic_iot)
        .set_group_id("ncg-flink-ward-aggregator")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    wm_strategy = WatermarkStrategy.for_bounded_out_of_orderness(
        Duration.of_seconds(30)
    ).with_timestamp_assigner(
        lambda event, _prev: _extract_ts_ms(event)
    )

    raw_stream = env.from_source(source, wm_strategy, "KafkaIoTSource")

    # ── Parse JSON ──────────────────────────────────────────────────────────
    def _parse(raw: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    parsed = raw_stream.map(_parse).filter(lambda x: x is not None)

    # ── Key by ward_id ──────────────────────────────────────────────────────
    keyed = parsed.key_by(lambda r: str(r["ward_id"]), key_type=Types.STRING())

    # ── 15-minute tumbling window ────────────────────────────────────────────
    class WardAggregateWindow(ProcessWindowFunction):
        def process(self, key, context, elements):
            records = list(elements)
            ward_id = int(key)
            window_start = datetime.fromtimestamp(
                context.window().start / 1000, tz=timezone.utc
            ).isoformat()
            window_end = datetime.fromtimestamp(
                context.window().end / 1000, tz=timezone.utc
            ).isoformat()
            n = len(records)
            avg_temp     = sum(r["temperature_c"] for r in records) / n
            avg_humidity = sum(r["humidity_pct"]  for r in records) / n
            avg_pm25     = sum(r["pm25"]          for r in records) / n
            sum_rain     = sum(r["rainfall_mm"]   for r in records)
            has_anomaly  = int(
                avg_temp > TEMP_ANOMALY_C or sum_rain > RAIN_ANOMALY_MM
            )

            payload = json.dumps({
                "ward_id":           ward_id,
                "window_start":      window_start,
                "window_end":        window_end,
                "avg_temperature_c": round(avg_temp, 2),
                "avg_humidity_pct":  round(avg_humidity, 2),
                "avg_pm25":          round(avg_pm25, 2),
                "sum_rainfall_mm":   round(sum_rain, 3),
                "reading_count":     n,
                "has_anomaly":       has_anomaly,
            })
            yield payload

    windowed = keyed.window(
        TumblingEventTimeWindows.of(Time.minutes(WINDOW_MINUTES))
    ).process(WardAggregateWindow(), output_type=Types.STRING())

    # ── Anomaly stream ───────────────────────────────────────────────────────
    def _is_anomaly(msg: str) -> bool:
        try:
            return bool(json.loads(msg).get("has_anomaly"))
        except Exception:
            return False

    alerts = windowed.filter(_is_anomaly)

    # ── Sinks ────────────────────────────────────────────────────────────────
    def _make_kafka_sink(topic: str) -> KafkaSink:
        return (
            KafkaSink.builder()
            .set_bootstrap_servers(_cfg.kafka_bootstrap_servers)
            .set_record_serializer(
                KafkaRecordSerializationSchema.builder()
                .set_topic(topic)
                .set_value_serialization_schema(SimpleStringSchema())
                .build()
            )
            .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
            .build()
        )

    windowed.sink_to(_make_kafka_sink(_cfg.kafka_topic_ward_features))
    alerts.sink_to(_make_kafka_sink(_cfg.kafka_topic_alerts))

    logger.success("Flink job graph assembled — submitting...")
    env.execute("NammaClimaGrid Ward Aggregator")


def _extract_ts_ms(event: str) -> int:
    """Extract epoch milliseconds from a JSON IoT reading's timestamp field."""
    try:
        ts_str = json.loads(event).get("timestamp", "")
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Offline / CSV fallback path — pure Python windowing
# ---------------------------------------------------------------------------

class _Window:
    """Accumulator for one 15-minute ward window."""
    __slots__ = ("ward_id", "start_ts", "readings")

    def __init__(self, ward_id: int, start_ts: float) -> None:
        self.ward_id = ward_id
        self.start_ts = start_ts
        self.readings: List[Dict[str, Any]] = []

    def flush(self) -> Dict[str, Any]:
        n = len(self.readings)
        if n == 0:
            return {}
        avg_temp     = sum(r["temperature_c"] for r in self.readings) / n
        avg_humidity = sum(r["humidity_pct"]  for r in self.readings) / n
        avg_pm25     = sum(r["pm25"]          for r in self.readings) / n
        sum_rain     = sum(r["rainfall_mm"]   for r in self.readings)
        has_anomaly  = int(avg_temp > TEMP_ANOMALY_C or sum_rain > RAIN_ANOMALY_MM)
        window_start = datetime.fromtimestamp(
            self.start_ts, tz=timezone.utc
        ).isoformat()
        window_end = datetime.fromtimestamp(
            self.start_ts + WINDOW_MINUTES * 60, tz=timezone.utc
        ).isoformat()
        return {
            "ward_id":           self.ward_id,
            "window_start":      window_start,
            "window_end":        window_end,
            "avg_temperature_c": round(avg_temp, 2),
            "avg_humidity_pct":  round(avg_humidity, 2),
            "avg_pm25":          round(avg_pm25, 2),
            "sum_rainfall_mm":   round(sum_rain, 3),
            "reading_count":     n,
            "has_anomaly":       has_anomaly,
        }


def _run_offline(csv_path: Path, once: bool = False) -> None:
    """
    Pure-Python offline processor.

    Reads from csv_path in a tail-follow loop (or one-shot if once=True),
    groups readings into 15-minute tumbling windows per ward, and prints
    aggregated features + anomaly alerts to stdout.
    """
    def _try_kafka_produce(topic: str, payload: Dict[str, Any]) -> None:
        """Publish to Kafka using kafka-python; silently skips if Kafka unavailable."""
        try:
            from kafka import KafkaProducer  # type: ignore
            prod = KafkaProducer(
                bootstrap_servers=_cfg.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=3000,
            )
            prod.send(topic, value=payload)
            prod.flush(timeout=2)
            prod.close()
        except Exception:
            pass  # best effort; offline mode doesn't require Kafka

    if not csv_path.exists():
        logger.error(f"CSV not found at {csv_path}. Run kafka_producer.py --offline first.")
        return

    logger.info(f"Offline mode | reading {csv_path}")

    # windows[ward_id] = _Window
    windows: Dict[int, _Window] = {}
    WINDOW_SECS = WINDOW_MINUTES * 60.0

    def _bucket(ts: float) -> float:
        return ts - (ts % WINDOW_SECS)

    def _process_row(row: Dict[str, str]) -> None:
        ward_id = int(row["ward_id"])
        try:
            ts = datetime.fromisoformat(
                row["timestamp"].replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            ts = time.time()

        reading = {
            "temperature_c": float(row.get("temperature_c", 0)),
            "humidity_pct":  float(row.get("humidity_pct", 0)),
            "pm25":          float(row.get("pm25", 0)),
            "rainfall_mm":   float(row.get("rainfall_mm", 0)),
        }
        bucket = _bucket(ts)
        win = windows.get(ward_id)

        if win is None or win.start_ts != bucket:
            # Flush old window
            if win is not None:
                agg = win.flush()
                if agg:
                    logger.info(f"Ward {agg['ward_id']:3d} window → {agg}")
                    _try_kafka_produce(_cfg.kafka_topic_ward_features, agg)
                    if agg.get("has_anomaly"):
                        logger.warning(f"⚠  ANOMALY ward={agg['ward_id']} → {agg}")
                        _try_kafka_produce(_cfg.kafka_topic_alerts, agg)
            windows[ward_id] = _Window(ward_id, bucket)
            windows[ward_id].readings.append(reading)
        else:
            win.readings.append(reading)

    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            _process_row(row)

        if once:
            # Flush remaining open windows
            for win in windows.values():
                agg = win.flush()
                if agg:
                    logger.info(f"Final flush ward {agg['ward_id']} → {agg}")
            logger.success("One-shot offline processing complete.")
            return

        # tail-follow: keep reading new lines as the file grows
        logger.info("Tailing CSV for new lines (Ctrl-C to stop)...")
        try:
            while True:
                pos = fh.tell()
                line = fh.readline()
                if not line:
                    time.sleep(2.0)
                    continue
                # Re-parse single line via DictReader trick
                import io
                row_reader = csv.DictReader(
                    io.StringIO(f"{','.join(reader.fieldnames)}\n{line}"),
                    fieldnames=reader.fieldnames,
                )
                next(row_reader)  # skip header replica
                row = next(row_reader, None)
                if row:
                    _process_row(row)
        except KeyboardInterrupt:
            logger.info("Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid Flink stream processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--offline", action="store_true",
                   help="Use CSV fallback instead of live Kafka + Flink")
    p.add_argument("--once", action="store_true",
                   help="(offline only) Process existing CSV once and exit")
    p.add_argument(
        "--csv", type=Path,
        default=Path(__file__).parent.parent / "8_data" / "mock_iot_stream.csv",
        help="CSV file to read in offline mode",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.offline:
        logger.info("Running in offline CSV mode")
        _run_offline(args.csv, once=args.once)
        sys.exit(0)

    # Try live Flink; fall back to offline on any import / connectivity error
    try:
        import pyflink  # noqa: F401  # type: ignore
        logger.info("PyFlink available — running streaming job")
        _run_flink()
    except ImportError:
        logger.warning("PyFlink not installed — switching to offline CSV mode")
        _run_offline(args.csv, once=False)
    except Exception as exc:
        logger.error(f"Flink job failed ({exc}) — switching to offline CSV mode")
        _run_offline(args.csv, once=False)
