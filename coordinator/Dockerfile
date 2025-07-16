###############  common base (deps & stubs only) #################
FROM python:3.10-slim AS base
WORKDIR /app/coordinator
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# OS‑level tools needed
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps that change rarely
RUN pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        grpcio grpcio-tools grpcio-health-checking

# Generate gRPC stubs once (re‑use cache unless proto changes)
COPY proto /app/proto
RUN python -m grpc_tools.protoc \
        -I /app/proto \
        --python_out=/app \
        --grpc_python_out=/app \
        /app/proto/inference.proto

EXPOSE 8000

###############  production image (immutable) ####################
FROM base AS prod
COPY coordinator /app/coordinator
WORKDIR /app/coordinator
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]

###############  development image ##################
FROM base AS dev
WORKDIR /app/coordinator
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]