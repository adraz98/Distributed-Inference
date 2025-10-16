# Distributed AI Inference System – Coding Challenge Submission

> **Candidate:** *Adrian Azemi*\
> **Stack:** FastAPI · gRPC · asyncio · Docker Compose · Streamlit

---

## Introduction

Build a **simulated, fault‑tolerant distributed inference pipeline** on a single machine. The system:

- receives REST inference requests.
- micro‑batches and dispatch them to a fleet of worker processes via gRPC.
- survives worker crashes and network hiccups (retries (=redirects) & health checks).
- exposes logs, metrics and a live dashboard (containing also the logs).

---

## High‑Level Architecture

```text

                             ┌──────────────────────────────┐
                             │        Dashboard (UI)        │
                             │          Streamlit           │
                             └──────────────┬───────────────┘
                                            │
                           REST (polls `/status`, `/recent_requests`)
                                            │
                                            ▼
┌──────────────────┐      HTTP/REST      ┌───────────────────────────────────────────────┐      gRPC        ┌──────────────────┐
│  Test Script     │────────────────────►│            Coordinator (FastAPI + asyncio)    │─────────────────►│    Worker 1      │
│ (bursty load)    │                     │  • Micro-batch queue                          │      gRPC        │  (DistilBERT)    │
└──────────────────┘                     │  • Round-robin load-balancer                  │─────────────────►│    Worker 2      │
                                         │  • Retry logic & heart-beats                  │      gRPC        │  (DistilBERT)    │
                                         │  • Structured JSON logging                    │─────────────────►│    Worker 3      │
                                         └───────────────────────────────────────────────┘                  │  (DistilBERT)    │
                                                                                                            └──────────────────┘

```

- **Coordinator** – Stateless FastAPI app. Queues requests, builds micro‑batches (size/time), load‑balances across healthy workers, handles retries (=redirections to other healthy workers), and conducts logging.
- **Workers** – Lightweight Python processes running DistilBERT‑SST2 in **ONNXRuntime‑CPU** classifying a text sequence into **positive** or **negative**. Each worker exposes a gRPC service generated from `proto/inference.proto`.
- **Client & Load Test** – A small Python test script firing bursty traffic to validate throughput and fault‑tolerance.
- **Dashboard** – Streamlit front‑end pulling `/status` & `/recent_requests` from the coordinator for near‑real‑time monitoring.

---

## Why gRPC Internally?

| Feature                                            | gRPC (HTTP/2)      | Classic REST (HTTP/1.1)   |
| -------------------------------------------------- | ------------------ | ------------------------- |
| **Multiplexing** (many concurrent streams per TCP) | ✔                  | ❌ (head‑of‑line blocking) |
| **Binary Framing** (ProtoBuf)                      | ✔ (compact & fast) | ❌ (text JSON)             |
| **Bi‑directional Streaming**                       | ✔                  | ❌                         |
| **Code Generation** (typed stubs)                  | ✔                  | ❌                         |
| **Low‑latency & CPU**                              | ✔                  | ❌                         |

REST remains for external clients conducting inference requests because it is human‑friendly and trivial to test/observe with `curl` or a browser.

---

## Repository Layout

```text
├── coordinator/          # FastAPI service (app.py) + gRPC proto
│   ├── proto/            
│   ├── app.py            # Coordinator logic
│   └── Dockerfile        # Coordinator Docker image
│
├── worker/               # gRPC worker service
│   ├── models/           # distilbert-sst2.onnx model
│   ├── proto/            
│   ├── worker.py         # Worker logic
│   └── Dockerfile        # Worker Docker image
│
├── monitor/              # Streamlit dashboard UI
│   ├── dashboard.py      # Streamlit app
│   └── Dockerfile        # Dashboard Docker image
│
├── proto/                # Shared proto definitions
│   └── inference.proto   # gRPC service definitions
│
├── test/                 # Test client
│   ├── test_client.py    # Bursty load test script
│   └── Dockerfile        # Test client Docker image
│
├── docs/                 # Documentation assets
│   └── images/           # Example dashboard screenshots
│
├── docker-compose.yml    # Orchestration for coordinator, workers, dashboard, test client
└── README.md             # (this file)
```

---

## Quick Start & Testing (🛠 Docker Compose)

> **Prerequisites:** Docker ≥ 28, Docker Compose v2.

```bash
# 1️⃣ Clone
$ git clone https://github.com/yourname/distributed‑ai‑challenge.git
$ cd distributed‑ai‑challenge

# 2️⃣ Build & Run (coordinator + 3 workers + dashboard + test client)
$ docker compose up --build

# 4️⃣ Open dashboard Streamlit UI)
http://localhost:8501  # (containing ALL live and historical logged information)
```

The compose file ensures that all services are started automatically in the correct order. **After running the above commands,
you can observe the results of the test script in the Streamlit dashboard.**

### Expected and exemplary outputs

When running Docker Compose with the default parameters, you will simulate ideal network conditions. In this case, the 
output should resemble the following:

![expected with ideal network.png](docs/images/expected%20with%20ideal%20network.png)

If you adjust the parameters to simulate network failures and delays, the output may vary significantly. This is 
particularly noticeable if all three workers fail simultaneously for a short period. Although Docker automatically 
restarts failed workers, many requests may still be lost during such events. The same can hold when by chance unusually 
many simulated network delays trigger the worker timeouts. Below is an example of potential output under simulated network
failures and delays (with the commented out parameters in `docker-compose.yml`):

![example simulated network failure.png](docs/images/example%20simulated%20network%20failure.png)

---

## Configuration

All knobs are environment variables ‑‑ override them in `docker‑compose.yml`.

| Variable               | Default                                     | Scope       | Description                                     |
| ---------------------- |---------------------------------------------| ----------- | ----------------------------------------------- |
| **WORKER\_TIMEOUT**    | `3`                                         | coordinator | Seconds to wait for a single gRPC call          |
| **BATCH\_SIZE**        | `3`                                         | coordinator | Maximum micro‑batch size                        |
| **BATCH\_TIMEOUT**     | `0.15`                                      | coordinator | Max age (s) of oldest queued task before dispatch |
| **WORKERS**            | `worker1:50051,worker2:50051,worker3:50051` | coordinator | Comma‑separated gRPC addresses                  |
| **REQUEST\_LOG\_SIZE** | `5000`                                      | coordinator | Ring buffer length for `/recent_requests`       |
| **CRASH\_RATE**        | `0` respectively `0.05` (for every worker)  | worker      | Probability ∈ [0,1] that a worker crashes  |
| **MAX\_DELAY**         | `0` respectively `4` (for every worker)     | worker      | Synthetic network/compute latency [s] upper‑bound |


---

## REST API Reference

### `POST /infer`

| Purpose      | Submit a single inference request                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------- |
| Request Body | `{ "input": "<text>" }`                                                                           |
| Success 200  | `{ "request_id": "<uuid>", "output": "<text>", "worker": "<id>", "success": true, "retries": 0 }` |
| Failure 5xx  | `{ "detail": "Inference request timed out" }`                                                     |

> The call blocks until the task succeeds **or** ≈ `5 × WORKER_TIMEOUT` (≈ 15 s default).

---

### `GET /status`

Returns coordinator health snapshot, worker states and aggregated counters. Example:

```json
{
  "workers": [
    { "id": "worker1", "alive": true,  "tasks_processed": 1287, "last_seen": 1721128100.23 },
    { "id": "worker2", "alive": true,  "tasks_processed": 1293, "last_seen": 1721128100.18 },
    { "id": "worker3", "alive": true,  "tasks_processed": 1286, "last_seen": 1721128099.95 }
  ],
  "tasks_submitted": 3900,
  "tasks_completed": 3886,
  "tasks_failed": 14,
  "tasks_retried": 71,
  "queue_length": 0
}
```

---

## Failure & Latency Simulation

Uncomment or tweak the following in `docker‑compose.yml`:

```yaml
environment:
  CRASH_RATE: "0.05"   # 5 % chance the worker exits mid‑batch
  MAX_DELAY:  "2"      # Each request delayed ∈ [0 .. 2] s
```

*Docker will auto‑restart crashed containers*. You can monitor the impact live on the Streamlit dashboard – look for ❌ workers, retries and failed requests and observe how Docker brings them back up.

---

## Monitoring Dashboard

```bash
http://localhost:8501
```

The UI shows:

- **System Health** – ✅/❌ per worker + last heartbeat
- **Core Metrics** – submitted/completed/failed/retried counters, current queue length
- **Pending Requests** – ⏳ queued & ⚙️ in‑flight requests currently being processed by workers (live table)
- **Historical Logs** – 5 000 most‑recent requests incl. latency, retries, assigned workers, errors, truncated input & model output

An illustration of the dashboard depicting all these aspects at once is shown below:

![example simulated network failure pending.png](docs/images/example%20simulated%20network%20failure%20pending.png)

---

*Thanks for reviewing my submission!*

yolo3