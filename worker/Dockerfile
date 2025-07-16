##############################################################################
#  Base image – installs runtime deps, copies ONNX model & proto,            #
#  generates stubs (NO torch anywhere).                                      #
##############################################################################
FROM python:3.10-slim AS base
ARG GRPC_HEALTH_PROBE_VERSION=v0.4.22

WORKDIR /app/worker
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ── tiny gRPC health‑probe binary and network tools ─────────────────────────
RUN set -e; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        iproute2 && \
    wget -qO /usr/local/bin/grpc_health_probe \
        https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /usr/local/bin/grpc_health_probe && \
    apt-get purge -y --auto-remove wget && \
    rm -rf /var/lib/apt/lists/*

# ── runtime‑only Python packages (CPU build of ORT) ─────────────────────────
RUN pip install --no-cache-dir \
        onnxruntime \
        transformers \
        grpcio grpcio-tools grpcio-health-checking \
        watchdog

# Optional: hide “None of PyTorch, TensorFlow …” message
ENV TRANSFORMERS_NO_FRAMEWORK_WARNING=1

# ── copy the ONNX model so it is always inside the image ───────────────────
COPY worker/models /app/models

# ── copy proto files and generate stubs (to /app) ──────────────────────────
COPY proto /app/proto
RUN python -m grpc_tools.protoc \
        -I /app/proto \
        --python_out=/app \
        --grpc_python_out=/app \
        /app/proto/inference.proto

##############################################################################
#  Production stage – code is baked in, good for CI / CD.                    #
##############################################################################
FROM base AS prod
COPY worker /app/worker
WORKDIR /app/worker
CMD ["python", "-u", "worker.py"]

##############################################################################
#  Development stage – no COPY of code, relies on host bind‑mount.           #
##############################################################################
FROM base AS dev
WORKDIR /app/worker
CMD ["python", "-u", "worker.py"]