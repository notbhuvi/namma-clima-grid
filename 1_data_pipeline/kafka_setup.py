"""
Module 1h — Kafka Topic Setup
================================

Creates all required Kafka topics for NammaClimaGrid using kafka-python
KafkaAdminClient. Has retry logic and handles TopicAlreadyExistsError
gracefully.

Topics created:
  iot-sensor-stream     (4 partitions) — IoT sensor readings
  ward-features-stream  (4 partitions) — aggregated ward features
  climate-alerts        (2 partitions) — anomaly/alert events
  citizen-reports       (2 partitions) — citizen-submitted reports
  satellite-ingestion   (2 partitions) — satellite pipeline events
  model-predictions     (4 partitions) — ML model output events

Usage:
    python 1_data_pipeline/kafka_setup.py
    python 1_data_pipeline/kafka_setup.py --bootstrap-servers kafka:29092
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

from loguru import logger

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 5

# Topic definitions: name → partitions
TOPICS: Dict[str, int] = {
    "iot-sensor-stream":    4,
    "ward-features-stream": 4,
    "climate-alerts":       2,
    "citizen-reports":      2,
    "satellite-ingestion":  2,
    "model-predictions":    4,
}


def create_topics(bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS) -> bool:
    """
    Create all NammaClimaGrid Kafka topics.

    Retries up to MAX_RETRIES times if the broker is not yet ready.
    Returns True if all topics exist (created or already existed).
    """
    try:
        from kafka import KafkaAdminClient
        from kafka.admin import NewTopic
        from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable
    except ImportError:
        logger.error("kafka-python not installed — run: pip install kafka-python")
        return False

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(
            f"Kafka topic setup attempt {attempt}/{MAX_RETRIES} "
            f"| broker={bootstrap_servers}"
        )
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=bootstrap_servers,
                client_id="ncg-topic-setup",
                request_timeout_ms=10000,
                connections_max_idle_ms=15000,
            )

            created = []
            skipped = []
            failed  = []

            # Create topics one-by-one so per-topic errors are handled cleanly
            for topic_name, partitions in TOPICS.items():
                new_topic = NewTopic(
                    name=topic_name,
                    num_partitions=partitions,
                    replication_factor=1,
                )
                try:
                    admin.create_topics([new_topic], validate_only=False)
                    created.append(topic_name)
                except TopicAlreadyExistsError:
                    skipped.append(topic_name)
                except Exception as exc:
                    # kafka-python sometimes bundles TopicAlreadyExists into
                    # a generic exception; treat it as a skip if the message says so
                    if "already exists" in str(exc).lower() or "error_code=36" in str(exc):
                        skipped.append(topic_name)
                    else:
                        logger.error(f"Failed to create topic '{topic_name}': {exc}")
                        failed.append(topic_name)

            admin.close()

            if created:
                logger.success(f"Created topics: {created}")
            if skipped:
                logger.info(f"Already exist (skipped): {skipped}")
            if failed:
                logger.error(f"Failed to create: {failed}")
                return False

            logger.success(
                f"Kafka topic setup complete | "
                f"created={len(created)} skipped={len(skipped)}"
            )
            return True

        except NoBrokersAvailable:
            logger.warning(
                f"No Kafka brokers available (attempt {attempt}/{MAX_RETRIES}). "
                f"Retrying in {RETRY_WAIT_SECONDS}s..."
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)
        except Exception as exc:
            logger.error(f"Kafka admin client error: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

    logger.error(
        f"Failed to set up Kafka topics after {MAX_RETRIES} attempts. "
        "Is the Kafka broker running?"
    )
    return False


def list_topics(bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS) -> None:
    """List all existing Kafka topics (for diagnostic use)."""
    try:
        from kafka import KafkaAdminClient
        admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers, client_id="ncg-list")
        topics = admin.list_topics()
        admin.close()
        logger.info(f"Existing Kafka topics: {sorted(topics)}")
    except Exception as exc:
        logger.error(f"Could not list topics: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid — Kafka topic initialisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--bootstrap-servers",
        default=KAFKA_BOOTSTRAP_SERVERS,
        help="Kafka bootstrap servers (comma-separated)",
    )
    p.add_argument("--list", action="store_true",
                   help="List existing topics and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.list:
        list_topics(args.bootstrap_servers)
    else:
        success = create_topics(args.bootstrap_servers)
        if not success:
            import sys
            sys.exit(1)
