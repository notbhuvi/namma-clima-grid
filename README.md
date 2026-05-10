# NammaClimaGrid

> **Bengaluru Thermal-Hydrological Resilience Platform**  
> Real-time Urban Heat Island stress and flash-flood risk prediction for all 198 BBMP wards — powered by a CNN-ViT vision encoder, Spatio-Temporal GNN, and a PPO-trained green infrastructure optimizer.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          NammaClimaGrid                                  │
│                                                                         │
│  IoT Sensors ──► Kafka ──► Flink (15-min windows) ──► ward-features    │
│  Open-Meteo ──►           ──► Spark batch ──────────► Delta Lake        │
│  Landsat/GEE ──►                                    ──► PostgreSQL      │
│                                        │                                │
│  Module 2: CNN-ViT ◄── satellite patches ◄── Landsat bands             │
│     └─► ward_embedding (128-dim) + lst_pred                             │
│                                        │                                │
│  Module 3: ST-GNN ◄── ward_features time-series + ward_embedding       │
│     └─► heat_stress_score + flood_risk_score (per ward, per hour)      │
│                                        │                                │
│  Module 4: PPO Agent ◄── GNN risk scores                               │
│     └─► intervention recommendations (ward × intervention × budget)    │
│                                        │                                │
│  Module 5: FastAPI ◄────────────────────┘                              │
│     ├─ GET  /wards/risk                                                 │
│     ├─ POST /wards/forecast                                             │
│     ├─ POST /interventions/recommend                                    │
│     └─ WS   /ws/alerts                                                  │
│                          │                                              │
│  Module 6: Flutter App ◄─┘   (iOS / Android / Web)                    │
│                                                                         │
│  Module 8: MLOps                                                        │
│     ├─ MLflow model registry (Staging → Production promotion)           │
│     ├─ PSI drift detection + Prometheus metrics                         │
│     └─ Airflow weekly retrain DAG                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

| Tool | Version |
|------|---------|
| Docker Desktop | ≥ 4.28 (Apple Silicon M-series supported) |
| Python | 3.9+ |
| Flutter | ≥ 3.10 |

### 2. Start the infrastructure stack

```bash
cp .env.template .env        # edit passwords if needed
docker compose up -d
```

Services started:

| Service | URL |
|---------|-----|
| FastAPI backend | `http://localhost:8000` |
| Kafka broker | `localhost:9092` |
| PostgreSQL | `localhost:5432` |
| Redis | `localhost:6379` |
| MLflow UI | `http://localhost:5001` |
| Apache Flink | `http://localhost:8081` |
| Airflow | `http://localhost:8082` (admin / admin) |
| Apache Superset | `http://localhost:8088` |

### 3. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate

pip install -r 1_data_pipeline/requirements.txt
pip install -r 2_thermal_vision/requirements.txt
pip install -r 3_st_gnn/requirements.txt
pip install -r 4_rl_optimizer/requirements.txt
pip install -r 5_backend/requirements.txt
```

### 4. Run the data pipeline

```bash
# Stream synthetic IoT data to Kafka
python 1_data_pipeline/kafka_producer.py

# Process with Flink (CSV fallback if Flink not running)
python 1_data_pipeline/flink_stream_processor.py

# Batch Landsat processing → Delta Lake
python 1_data_pipeline/spark_batch_processor.py --force-fallback
```

### 5. Train the models

```bash
# Module 2 — CNN-ViT (synthetic data, ~5 min on CPU)
python 2_thermal_vision/train.py --epochs 20 --no-mlflow

# Module 3 — ST-GNN (~2 min on CPU)
python 3_st_gnn/train.py --epochs 20 --n-snapshots 100 --no-mlflow

# Module 4 — PPO (~3 min on CPU)
python 4_rl_optimizer/train.py --timesteps 50000 --no-mlflow
```

### 6. Start the API

```bash
cd 5_backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 6.1 Operational smoke tests

The backend has a lightweight smoke suite that skips model loading and Kafka
threads so it can run quickly in CI:

```bash
NCG_SKIP_MODEL_LOAD=true NCG_SKIP_KAFKA_CONSUMER=true \
  python -m pytest -q 5_backend/tests
```

The Docker Compose file now includes the real Module 5 API service. For
production-like deployments, set `REQUIRE_EXTERNAL_SERVICES=true`,
`ALLOW_ALL_CORS=false`, and provide explicit `CORS_ORIGINS`, database,
Kafka, and secret values through environment variables.

### 7. Run the Flutter app

```bash
cd 6_flutter_app
flutter pub get
flutter run --dart-define=API_BASE_URL=http://localhost:8000
```

---

## Module Reference

### Module 1 — Data Pipeline (`1_data_pipeline/`)

| File | Description |
|------|-------------|
| `kafka_producer.py` | 10 synthetic sensor nodes → Open-Meteo API → Kafka `iot-sensor-stream` |
| `flink_stream_processor.py` | PyFlink 15-min tumbling windows → ward aggregates + anomaly alerts |
| `spark_batch_processor.py` | Landsat bands → NDVI/NDBI/LST → Delta Lake + PostgreSQL |
| `delta_lake_writer.py` | DeltaLakeWriter: write_batch / write_streaming / time_travel / vacuum |
| `airflow_dags/satellite_download_dag.py` | Daily 02:00 IST: GEE download → Spark → Delta verify |
| `airflow_dags/daily_feature_pipeline_dag.py` | Hourly: mock features → Delta MERGE + Postgres upsert + Kafka event |

### Module 2 — Thermal Vision Engine (`2_thermal_vision/`)

| File | Description |
|------|-------------|
| `model.py` | `HybridLSTPredictor`: ResNet50 (channel-adapted) + 6-layer ViT → LST + heat_stress + 128-dim ward_embedding |
| `dataset.py` | `WardPatchDataset`: synthetic 7-channel 64×64 patches per ward with realistic texture |
| `train.py` | AdamW + cosine LR, MSE+MAE loss, early stopping, MLflow tracking |
| `inference.py` | `ThermalVisionInference`: `predict_dataframe()` + `encode_node_features()` for GNN |

### Module 3 — Spatio-Temporal GNN (`3_st_gnn/`)

| File | Description |
|------|-------------|
| `graph_builder.py` | k-NN + distance-threshold ward adjacency, haversine edge weights, `.pt` cache |
| `model.py` | `SpatioTemporalGNN`: 3× GraphSAGE → temporal multi-head attention → dual sigmoid heads |
| `dataset.py` | `WardTemporalDataset`: sliding window (N=198, T=8, F=12) with synthetic diurnal time series |
| `train.py` | Dual-task MSE (λ=0.6 heat + 0.4 flood), early stopping, MLflow |
| `inference.py` | `STGNNInference`: `predict_snapshot(df)` + `encode_node_embeddings()` |

### Module 4 — RL Green Infra Optimizer (`4_rl_optimizer/`)

| File | Description |
|------|-------------|
| `environment.py` | `BengaluruClimaEnv`: 198 wards × 5 interventions = 990-action Gymnasium env |
| `model.py` | `ActorCritic`: per-ward MLP encoder + global max-pool + actor/critic heads, budget masking |
| `train.py` | PPO-clip: GAE, mini-batch updates, gradient clipping, MLflow |
| `inference.py` | `GreenInfraOptimizer`: `recommend(heat, flood, budget)` → `List[RecommendedIntervention]` |

**Intervention types:**

| Type | Cost | Heat effect | Flood effect |
|------|------|-------------|--------------|
| `tree_planting` | 2 | NDVI +0.08, LST −1.5°C | — |
| `green_roof` | 3 | NDVI +0.04, LST −0.8°C | impervious −5% |
| `permeable_pavement` | 4 | — | impervious −12%, flood −15 |
| `urban_wetland` | 6 | NDVI +0.05, heat −3 | flood −25 |
| `cool_pavement` | 2 | LST −1.0°C, heat −8 | — |

### Module 5 — FastAPI Backend (`5_backend/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + model load status |
| `/wards/risk` | GET | All 198 wards with heat/flood scores |
| `/wards/risk/{id}` | GET | Single ward detail |
| `/wards/forecast` | POST | Hourly forecast (1–72h horizon) |
| `/interventions/types` | GET | Intervention catalogue |
| `/interventions/recommend` | POST | RL-optimised budget allocation |
| `/ws/alerts` | WS | Real-time critical ward alerts (30s broadcast) |

### Module 6 — Flutter App (`6_flutter_app/`)

Three screens:
- **Dashboard** — 198-ward grid sorted by risk, city-wide averages, risk-level filter chips
- **Interventions** — Budget slider, RL recommendation list with expected reductions
- **Alerts** — Live WebSocket stream, severity badges, swipe-to-dismiss

### Module 8 — MLOps (`8_mlops/`)

| File | Description |
|------|-------------|
| `model_registry.py` | MLflow model promotion: Staging → Production with metric thresholds |
| `monitoring.py` | PSI drift detection + Prometheus metrics server (`:9090/metrics`) |
| `retrain_dag.py` | Airflow DAG: weekly retrain → promote → smoke test (Sunday 02:00 IST) |

---

## Environment Variables

Copy `.env.template` to `.env` and configure:

```
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
POSTGRES_HOST=localhost
DELTA_LAKE_ROOT=/data/delta
MLFLOW_TRACKING_URI=http://localhost:5001
THERMAL_VISION_CHECKPOINT=2_thermal_vision/checkpoints/best_model.pt
STGNN_CHECKPOINT=3_st_gnn/checkpoints/gnn_best.pt
RL_CHECKPOINT=4_rl_optimizer/checkpoints/ppo_best.pt
```

---

## Offline / Synthetic Mode

Every module has a complete synthetic fallback — no external APIs or trained models required:

| Module | Fallback |
|--------|----------|
| IoT sensors | Diurnal temperature synthesis, CSV sink |
| Open-Meteo | Synthetic weather from ward coordinates |
| Landsat/GEE | Synthetic `.npz` raster with NDVI/NDBI/LST |
| Module 2 | `ThermalVisionInference.untrained()` |
| Module 3 | `STGNNInference.untrained()` + synthetic time series |
| Module 4 | `GreenInfraOptimizer.untrained()` + heuristic fallback |
| Module 5 | Synthetic `_synthetic_ward_risk()` DataFrame |

---

## Data Flow Summary

```
Sensor / API data
      │
      ▼ (Module 1)
Delta Lake snapshots (ward_features, raw_iot, satellite_features)
      │
      ├──► Module 2: CNN-ViT encodes satellite patches → ward_embedding (128-dim)
      │
      ├──► Module 3: ST-GNN processes time-series + embeddings
      │          └─► heat_stress_score, flood_risk_score per ward
      │
      ├──► Module 4: PPO agent reads GNN scores
      │          └─► ranked intervention recommendations
      │
      └──► Module 5: FastAPI aggregates all outputs
                 └─► Module 6: Flutter app renders to field officers
```

---

## Project Structure

```
namma-clima-grid/
├── .env.template
├── docker-compose.yml
├── 1_data_pipeline/
├── 2_thermal_vision/
├── 3_st_gnn/
├── 4_rl_optimizer/
├── 5_backend/
├── 6_flutter_app/
├── 7_infrastructure/
│   └── postgres/
├── 8_data/               ← synthetic data generators + ward catalogue
└── 8_mlops/
```
