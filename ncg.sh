#!/usr/bin/env bash
# =============================================================================
# NammaClimaGrid — Unified CLI
# =============================================================================
# Usage: ./ncg.sh <command> [args]
#
# Commands:
#   infra-up      Start all Docker infrastructure services
#   infra-down    Stop all services
#   kafka-setup   Create / verify all required Kafka topics
#   pipeline      Run full data ingestion (weather + satellite)
#   train         Train ML models (all | thermal | stgnn | rl)
#   backend       Start FastAPI backend (port 8000)
#   flutter       Build and serve Flutter web app (port 3000)
#   status        Show health of all running services
#   full          Start everything (infra + pipeline + backend + flutter)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Python interpreter ──────────────────────────────────────────────────────
if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
    VENV_PY="$SCRIPT_DIR/.venv/bin/python"
elif [[ -f "$SCRIPT_DIR/.venv/bin/python3" ]]; then
    VENV_PY="$SCRIPT_DIR/.venv/bin/python3"
else
    VENV_PY="$(which python3 2>/dev/null || which python)"
fi

# ── Flutter binary ──────────────────────────────────────────────────────────
FLUTTER_BIN="$(which flutter 2>/dev/null || echo 'flutter')"

# ── Helpers ─────────────────────────────────────────────────────────────────
_green()  { echo -e "\033[0;32m$*\033[0m"; }
_yellow() { echo -e "\033[1;33m$*\033[0m"; }
_red()    { echo -e "\033[0;31m$*\033[0m"; }
_bold()   { echo -e "\033[1m$*\033[0m"; }

_check_docker() {
    if ! docker info >/dev/null 2>&1; then
        _red "Docker daemon is not running. Start Docker Desktop first."
        exit 1
    fi
}

_http_ok() {
    curl -sf --max-time 3 "$1" >/dev/null 2>&1
}

# ── Commands ─────────────────────────────────────────────────────────────────

cmd_infra_up() {
    _check_docker
    _bold "Starting NammaClimaGrid infrastructure..."
    docker compose up -d \
        postgres redis kafka mlflow \
        airflow-webserver airflow-scheduler \
        superset grafana
    echo ""
    _yellow "Waiting 20s for services to initialise..."
    sleep 20
    # Create Kafka topics
    cmd_kafka_setup
    _green "Infrastructure ready."
    echo ""
    _bold "Service URLs:"
    echo "  Kafka broker ......... localhost:9092"
    echo "  PostgreSQL ........... localhost:5432"
    echo "  Redis ................ localhost:6379"
    echo "  MLflow ............... http://localhost:5001"
    echo "  Airflow .............. http://localhost:8088  (admin / admin)"
    echo "  Superset ............. http://localhost:8089  (admin / admin)"
    echo "  Grafana .............. http://localhost:3002  (admin / ncg_admin)"
    echo "  Flutter App .......... http://localhost:3001  (after: ./ncg.sh backend && ./ncg.sh flutter)"
}

cmd_infra_down() {
    _check_docker
    _bold "Stopping all NammaClimaGrid containers..."
    docker compose down
    _green "All containers stopped."
}

cmd_kafka_setup() {
    _bold "Setting up Kafka topics..."
    "$VENV_PY" 1_data_pipeline/kafka_setup.py \
        --bootstrap-servers localhost:9092 2>&1 || true
}

cmd_pipeline() {
    _bold "Running data ingestion pipeline..."
    echo ""

    echo "── Weather ingestion (Open-Meteo) ─────────────────────────────"
    "$VENV_PY" 1_data_pipeline/weather_ingestion.py 2>&1 || \
        _yellow "  Weather ingestion failed (non-critical)"

    echo ""
    echo "── Satellite ingestion (fallback mode) ───────────────────────"
    "$VENV_PY" 1_data_pipeline/satellite_ingestion.py --fallback 2>&1 || \
        _yellow "  Satellite ingestion failed (non-critical)"

    _green "Data pipeline complete."
}

cmd_train() {
    MODULE="${1:-all}"
    _bold "Training ML modules: $MODULE"
    echo ""

    case "$MODULE" in
        all)
            echo "── Module 2: CNN-ViT Thermal Vision ──────────────────────────"
            (cd 2_thermal_vision && "$VENV_PY" train.py) 2>&1
            _green "  CNN-ViT training done"
            echo ""
            echo "── Module 3: ST-GNN Heat+Flood Forecaster ────────────────────"
            (cd 3_st_gnn && "$VENV_PY" train.py) 2>&1
            _green "  ST-GNN training done"
            echo ""
            echo "── Module 4: PPO RL Green-Infra Optimizer ────────────────────"
            (cd 4_rl_optimizer && "$VENV_PY" train.py) 2>&1
            _green "  PPO RL training done"
            ;;
        thermal)
            echo "── Module 2: CNN-ViT Thermal Vision ──────────────────────────"
            (cd 2_thermal_vision && "$VENV_PY" train.py) 2>&1
            _green "  CNN-ViT training done"
            ;;
        stgnn)
            echo "── Module 3: ST-GNN Heat+Flood Forecaster ────────────────────"
            (cd 3_st_gnn && "$VENV_PY" train.py) 2>&1
            _green "  ST-GNN training done"
            ;;
        rl)
            echo "── Module 4: PPO RL Green-Infra Optimizer ────────────────────"
            (cd 4_rl_optimizer && "$VENV_PY" train.py) 2>&1
            _green "  PPO RL training done"
            ;;
        *)
            _red "Unknown module '$MODULE'. Use: all | thermal | stgnn | rl"
            exit 1
            ;;
    esac
    echo ""
    _green "Training complete. Checkpoints saved in each module's checkpoints/ dir."
    echo "  2_thermal_vision/checkpoints/best_model.pt"
    echo "  3_st_gnn/checkpoints/gnn_best.pt"
    echo "  4_rl_optimizer/checkpoints/ppo_best.pt"
    echo "  MLflow experiments: http://localhost:5001"
}

cmd_backend() {
    _bold "Starting FastAPI backend on http://localhost:8000 ..."
    echo "  API docs: http://localhost:8000/docs"
    echo "  Press Ctrl+C to stop."
    echo ""
    (cd 5_backend && "$VENV_PY" -m uvicorn main:app \
        --host 0.0.0.0 --port 8000 --reload)
}

cmd_flutter() {
    _bold "Building Flutter web app..."
    (cd 6_flutter_app && "$FLUTTER_BIN" build web --release --base-href "/" 2>&1) || {
        _red "Flutter build failed. Is Flutter installed? Run: which flutter"
        exit 1
    }
    _green "Build complete."
    echo ""
    _bold "Serving Flutter app at http://localhost:3001 ..."
    echo "  Press Ctrl+C to stop."
    "$VENV_PY" -m http.server 3001 --directory 6_flutter_app/build/web
}

cmd_status() {
    _bold "=== NammaClimaGrid System Status ==="
    echo ""

    # Docker containers
    echo "Infrastructure (Docker):"
    if docker info >/dev/null 2>&1; then
        docker ps --format "  {{.Names}}: {{.Status}}" 2>/dev/null | \
            grep "ncg-" | sort || echo "  No NCG containers running"
    else
        _yellow "  Docker not running"
    fi
    echo ""

    # Services
    echo "Services:"
    _http_ok "http://localhost:8000/api" && \
        echo "  FastAPI backend: UP  → http://localhost:8000/docs" || \
        _yellow "  FastAPI backend: DOWN"

    if _http_ok "http://localhost:3001/"; then
        echo "  Flutter app:     UP  → http://localhost:3001"
    elif _http_ok "http://localhost:8000/"; then
        echo "  Flutter app:     UP  → http://localhost:8000  (served by FastAPI)"
    else
        _yellow "  Flutter app:     DOWN"
    fi

    _http_ok "http://localhost:5001/" && \
        echo "  MLflow:          UP  → http://localhost:5001" || \
        _yellow "  MLflow:          DOWN"

    _http_ok "http://localhost:8088/" && \
        echo "  Airflow:         UP  → http://localhost:8088  (admin/admin)" || \
        _yellow "  Airflow:         DOWN"

    _http_ok "http://localhost:8089/" && \
        echo "  Superset:        UP  → http://localhost:8089  (admin/admin)" || \
        _yellow "  Superset:        DOWN"

    _http_ok "http://localhost:3002/" && \
        echo "  Grafana:         UP  → http://localhost:3002  (admin/ncg_admin)" || \
        _yellow "  Grafana:         DOWN"

    # ML models
    echo ""
    echo "ML Checkpoints:"
    [[ -f 2_thermal_vision/checkpoints/best_model.pt ]] && \
        echo "  CNN-ViT model:   OK  (best_model.pt)" || \
        _yellow "  CNN-ViT model:   MISSING — run: ./ncg.sh train thermal"
    [[ -f 3_st_gnn/checkpoints/gnn_best.pt ]] && \
        echo "  ST-GNN model:    OK  (gnn_best.pt)" || \
        _yellow "  ST-GNN model:    MISSING — run: ./ncg.sh train stgnn"
    [[ -f 4_rl_optimizer/checkpoints/ppo_best.pt ]] && \
        echo "  PPO-RL model:    OK  (ppo_best.pt)" || \
        _yellow "  PPO-RL model:    MISSING — run: ./ncg.sh train rl"

    # Kafka topics
    echo ""
    echo "Kafka Topics:"
    "$VENV_PY" -c "
from kafka import KafkaAdminClient
try:
    a = KafkaAdminClient(bootstrap_servers='localhost:9092', request_timeout_ms=2000)
    topics = [t for t in sorted(a.list_topics()) if not t.startswith('__')]
    a.close()
    for t in topics: print(f'  {t}')
except Exception as e:
    print(f'  Kafka unavailable: {e}')
" 2>/dev/null || _yellow "  Kafka unavailable"
}

cmd_full() {
    _bold "=== Starting full NammaClimaGrid stack ==="
    echo ""
    cmd_infra_up
    echo ""
    cmd_pipeline
    echo ""
    _bold "Starting backend and Flutter in background..."
    (cd 5_backend && "$VENV_PY" -m uvicorn main:app \
        --host 0.0.0.0 --port 8000 2>&1 | \
        grep -v "^INFO:" &)
    sleep 3
    # Serve pre-built Flutter (rebuild only if needed)
    if [[ -d 6_flutter_app/build/web ]]; then
        "$VENV_PY" -m http.server 3001 --directory 6_flutter_app/build/web > /dev/null 2>&1 &
    else
        _yellow "Flutter web not built — run: ./ncg.sh flutter"
    fi
    echo ""
    sleep 2
    cmd_status
    echo ""
    _green "=== NammaClimaGrid is running ==="
    echo ""
    echo "  Flutter App: http://localhost:3001"
    echo "  API docs:    http://localhost:8000/docs"
    echo "  MLflow:      http://localhost:5001"
    echo "  Grafana:     http://localhost:3002"
    echo "  Superset:    http://localhost:8089  (admin / admin)"
    echo ""
    _yellow "To stop: ./ncg.sh infra-down  (or Ctrl+C for foreground processes)"
}

# ── Main dispatch ─────────────────────────────────────────────────────────────

COMMAND="${1:-help}"
shift 2>/dev/null || true

case "$COMMAND" in
    infra-up)    cmd_infra_up ;;
    infra-down)  cmd_infra_down ;;
    kafka-setup) cmd_kafka_setup ;;
    pipeline)    cmd_pipeline ;;
    train)       cmd_train "${1:-all}" ;;
    backend)     cmd_backend ;;
    flutter)     cmd_flutter ;;
    status)      cmd_status ;;
    full)        cmd_full ;;
    help|--help|-h|*)
        _bold "NammaClimaGrid CLI"
        echo ""
        echo "Usage: ./ncg.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  infra-up              Start all Docker services (Kafka, Postgres, MLflow, etc.)"
        echo "  infra-down            Stop all Docker services"
        echo "  kafka-setup           Create / verify Kafka topics"
        echo "  pipeline              Run data ingestion (weather + satellite)"
        echo "  train [module]        Train ML models  [all | thermal | stgnn | rl]"
        echo "  backend               Start FastAPI backend on :8000"
        echo "  flutter               Build + serve Flutter web app on :3000"
        echo "  status                Show health of all services + checkpoints"
        echo "  full                  Start everything end-to-end"
        echo ""
        echo "Examples:"
        echo "  ./ncg.sh infra-up                # start infrastructure"
        echo "  ./ncg.sh pipeline                # run data ingestion"
        echo "  ./ncg.sh train thermal           # train only CNN-ViT"
        echo "  ./ncg.sh backend                 # start API server"
        echo "  ./ncg.sh status                  # check everything"
        echo "  ./ncg.sh full                    # start the whole stack"
        echo ""
        echo "Service URLs (when running):"
        echo "  Flutter App  → http://localhost:3001"
        echo "  API (docs)   → http://localhost:8000/docs"
        echo "  MLflow       → http://localhost:5001"
        echo "  Airflow      → http://localhost:8088  (admin / admin)"
        echo "  Superset     → http://localhost:8089  (admin / admin)"
        echo "  Grafana      → http://localhost:3002  (admin / ncg_admin)"
        ;;
esac
