"""
Routes — WebSocket real-time alert stream
============================================

  WS /ws/alerts  → pushes ward alerts every 30 seconds
                   + forwards messages from Kafka `climate-alerts` topic

Clients receive JSON messages matching the WardAlert schema:
  {
    "ward_id": 42,
    "alert_type": "heat_wave",
    "severity": "critical",
    "message": "Extreme heat stress in Koramangala: 82/100",
    "value": 82.3,
    "threshold": 75.0,
    "timestamp": "2024-06-15T10:30:00Z"
  }

Multiple clients can connect simultaneously; each gets the same alerts.
A background thread consumes the Kafka `climate-alerts` topic and forwards
any messages to all connected WebSocket clients.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from config import get_settings
from services import generate_alerts

router = APIRouter(tags=["websocket"])

# Active connections
_connections: Set[WebSocket] = set()

# Background alert task handle
_alert_task = None
ALERT_INTERVAL_SEC = 15

# Kafka consumer thread handle
_kafka_thread: threading.Thread | None = None
_kafka_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

async def _broadcast(payload: str) -> None:
    """Send a message to all connected clients, removing dead connections."""
    dead: List[WebSocket] = []
    for ws in _connections:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connections.discard(ws)


async def inject_alert(alert: dict) -> None:
    """
    Inject a single alert from any route (e.g. citizen image upload or
    BBMP official broadcast) and push it to all connected clients immediately.
    """
    payload = json.dumps({"alerts": [alert], "count": 1})
    await _broadcast(payload)
    logger.info(f"Injected alert → {len(_connections)} clients | type={alert.get('alert_type')}")


# ---------------------------------------------------------------------------
# Background alert loop
# ---------------------------------------------------------------------------

async def _alert_loop() -> None:
    """Periodically generate alerts and broadcast to connected clients."""
    while True:
        await asyncio.sleep(ALERT_INTERVAL_SEC)
        if not _connections:
            continue
        try:
            alerts = await generate_alerts()
            if alerts:
                payload = json.dumps({"alerts": alerts, "count": len(alerts)})
                await _broadcast(payload)
                logger.debug(f"Broadcast {len(alerts)} alerts to {len(_connections)} clients")
        except Exception as exc:
            logger.error(f"Alert broadcast error: {exc}")


def start_alert_background_task() -> None:
    """Start the background alert broadcaster (called once at startup)."""
    global _alert_task
    if _alert_task is None:
        _alert_task = asyncio.create_task(_alert_loop())
        logger.info("Alert broadcast background task started")
    if get_settings().skip_kafka_consumer:
        logger.info("NCG_SKIP_KAFKA_CONSUMER=true — Kafka alert consumer disabled")
        return
    _start_kafka_consumer_thread()


def stop_alert_background_task() -> None:
    """Cancel the background task on shutdown."""
    global _alert_task
    if _alert_task is not None:
        _alert_task.cancel()
        _alert_task = None
    _stop_kafka_consumer_thread()


# ---------------------------------------------------------------------------
# Kafka consumer background thread
# ---------------------------------------------------------------------------

def _kafka_consumer_thread(loop: asyncio.AbstractEventLoop) -> None:
    """
    Background thread that reads from the `climate-alerts` Kafka topic and
    broadcasts each message to all connected WebSocket clients.

    Runs in a daemon thread (not async) to avoid blocking the event loop.
    Skips gracefully if Kafka is unavailable.
    """
    try:
        from kafka import KafkaConsumer  # type: ignore
        from kafka.errors import NoBrokersAvailable  # type: ignore
    except ImportError:
        logger.warning("kafka-python not installed — Kafka alert consumer disabled")
        return

    consumer = None
    try:
        settings = get_settings()
        consumer = KafkaConsumer(
            settings.kafka_alerts_topic,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id="ncg-ws-alert-forwarder",
            auto_offset_reset="latest",
            consumer_timeout_ms=1000,       # poll timeout; allows stop check
            value_deserializer=lambda b: b.decode("utf-8", errors="replace"),
            request_timeout_ms=10000,
            api_version_auto_timeout_ms=5000,
        )
        logger.info(
            f"Kafka consumer connected to topic '{settings.kafka_alerts_topic}' — "
            "forwarding alerts to WebSocket clients"
        )
    except NoBrokersAvailable:
        logger.warning(
            "Kafka broker not available — "
            "WebSocket Kafka alert forwarding disabled"
        )
        return
    except Exception as exc:
        logger.warning(f"Kafka consumer init failed ({exc}) — disabling Kafka forwarding")
        return

    while not _kafka_stop_event.is_set():
        try:
            for message in consumer:
                if _kafka_stop_event.is_set():
                    break
                if not _connections:
                    continue
                try:
                    payload_str = message.value
                    # Validate it is valid JSON before broadcasting
                    json.loads(payload_str)
                    # Schedule broadcast on the event loop (thread-safe)
                    asyncio.run_coroutine_threadsafe(
                        _broadcast(payload_str), loop
                    )
                except json.JSONDecodeError:
                    pass
                except Exception as exc:
                    logger.debug(f"Kafka message forward error: {exc}")
        except Exception as exc:
            if not _kafka_stop_event.is_set():
                logger.debug(f"Kafka consumer poll error: {exc}")

    try:
        consumer.close()
    except Exception:
        pass
    logger.info("Kafka alert consumer thread stopped")


def _start_kafka_consumer_thread() -> None:
    """Start the Kafka consumer daemon thread."""
    global _kafka_thread
    _kafka_stop_event.clear()
    if _kafka_thread is None or not _kafka_thread.is_alive():
        loop = asyncio.get_event_loop()
        _kafka_thread = threading.Thread(
            target=_kafka_consumer_thread,
            args=(loop,),
            daemon=True,
            name="ncg-kafka-alert-consumer",
        )
        _kafka_thread.start()
        logger.info("Kafka alert consumer thread started")


def _stop_kafka_consumer_thread() -> None:
    """Signal the Kafka consumer thread to stop."""
    _kafka_stop_event.set()
    if _kafka_thread is not None and _kafka_thread.is_alive():
        _kafka_thread.join(timeout=5.0)
    logger.info("Kafka alert consumer thread signalled to stop")


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket):
    """
    Real-time ward alert stream.

    On connect: sends current alerts immediately.
    Then: receives periodic broadcast updates every 30s.
    Client can send `{"ping": true}` to keep alive.
    """
    await websocket.accept()
    _connections.add(websocket)
    logger.info(f"WebSocket client connected | total={len(_connections)}")

    # Send initial alerts immediately
    try:
        alerts = await generate_alerts()
        await websocket.send_text(
            json.dumps({"alerts": alerts, "count": len(alerts)})
        )
    except Exception as exc:
        logger.warning(f"Initial alert send failed: {exc}")

    # Keep connection alive, handle client messages
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("ping"):
                    await websocket.send_text(json.dumps({"pong": True}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _connections.discard(websocket)
        logger.info(f"WebSocket client disconnected | total={len(_connections)}")
