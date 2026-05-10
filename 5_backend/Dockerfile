FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY 5_backend/requirements.txt /tmp/requirements-backend.txt
COPY 2_thermal_vision/requirements.txt /tmp/requirements-thermal.txt
COPY 3_st_gnn/requirements.txt /tmp/requirements-stgnn.txt
COPY 4_rl_optimizer/requirements.txt /tmp/requirements-rl.txt

RUN pip install --upgrade pip \
    && pip install \
        -r /tmp/requirements-backend.txt \
        -r /tmp/requirements-thermal.txt \
        -r /tmp/requirements-stgnn.txt \
        -r /tmp/requirements-rl.txt

COPY 2_thermal_vision ./2_thermal_vision
COPY 3_st_gnn ./3_st_gnn
COPY 4_rl_optimizer ./4_rl_optimizer
COPY 5_backend ./5_backend
COPY 8_data ./8_data

WORKDIR /app/5_backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
