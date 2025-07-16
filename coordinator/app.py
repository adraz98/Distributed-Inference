"""
Coordinator – FastAPI front‑end + gRPC dispatcher
Handles micro‑batching and fair round‑robin load‑balancing to the workers.
"""

from __future__ import annotations

import os
import asyncio
import time
import uuid
import logging
import json
from collections import deque
from typing import List

import grpc
from fastapi import FastAPI, HTTPException

import inference_pb2
import inference_pb2_grpc

# ──────────────────────────── logging ────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("coordinator")
uvicorn_error = logging.getLogger("uvicorn.error")
uvicorn_error.disabled = True
uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.disabled = True

# ───────────────────────── configuration ─────────────────────────
WORKER_TIMEOUT = float(os.getenv("WORKER_TIMEOUT", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "3"))              # max items per batch
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.15"))   # seconds from first arrival

worker_addresses = os.getenv("WORKERS", "")
worker_addresses = worker_addresses.split(",") if worker_addresses else [
    "worker1:50051",
    "worker2:50051",
    "worker3:50051",
]

REQUEST_LOG_SIZE = int(os.getenv("REQUEST_LOG_SIZE", "5000"))
recent_requests: deque[dict] = deque(maxlen=REQUEST_LOG_SIZE)

# helper: seconds → milliseconds (1 decimal)
def _ms(sec: float) -> float:
    return round(sec * 1_000, 1)

# ─────────────────────── gRPC stubs for workers ───────────────────────
workers: List[dict] = []
for addr in worker_addresses:
    wid = addr.split(":")[0]
    channel = grpc.aio.insecure_channel(addr)
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    workers.append(
        {
            "id": wid,
            "stub": stub,
            "alive": True,
            "tasks_processed": 0,
            "last_seen": time.time(),
        }
    )

next_worker_index = 0  # round‑robin pointer (global across requests)
queue: asyncio.Queue["Task"] = asyncio.Queue()

# ─────────────────────────── global counters ───────────────────────────
tasks_submitted = tasks_completed = tasks_failed = tasks_retried = 0  # noqa: E401

# ──────────────────────────── live state ───────────────────────────────
# request_id → {"id", "input", "status": "queued|processing", "worker", "timestamp"}
pending_tasks: dict[str, dict] = {}

# ───────────────────────────── FastAPI app ─────────────────────────────
app = FastAPI()


class Task:
    """Container for one user request living inside the coordinator."""

    def __init__(self, request_id: str, input_text: str):
        self.id: str = request_id
        self.input_text: str = input_text
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.start_time: float = time.time()
        self.attempts: int = 0
        self.worker_assigned: str | None = None
        self.result: dict | None = None


# ────────────────────────── util: dashboard log ──────────────────────────
def _record_request(
    task: Task,
    *,
    success: bool,
    output: str | None = None,
    error: str | None = None,
    retries: int | None = None,
    latency: float | None = None,
) -> None:
    """Append a request entry to the in‑memory log **and**
    emit a structured log line with the essential metadata required
    for external log aggregation / monitoring."""
    entry = {
        "timestamp": time.time(),
        "request_id": task.id,
        "input": task.input_text,
        "output": output,
        "worker": task.worker_assigned,
        "success": success,
        "error": error,
        "retries": retries,
        "latency_ms": _ms(
            latency if latency is not None else time.time() - task.start_time
        ),
    }
    recent_requests.append(entry)

    # 🔹 Emit one JSON log line per task with the requested metadata.
    logger.info(
        json.dumps(
            {
                "event": "task_log",
                "request_id": task.id,
                "worker": task.worker_assigned,
                "latency_ms": entry["latency_ms"],
                "retries": retries,
            }
        )
    )


# ────────────────────────────── endpoints ───────────────────────────────
@app.post("/infer")
async def infer_endpoint(data: dict):
    """Accept a single inference request."""
    global tasks_submitted, tasks_failed

    text = data.get("input")
    if text is None:
        raise HTTPException(status_code=400, detail="No input provided")

    request_id = str(uuid.uuid4())
    task = Task(request_id, text)
    tasks_submitted += 1

    # mark as queued
    pending_tasks[request_id] = {
        "id": request_id,
        "input": text,
        "status": "queued",
        "worker": None,
        "timestamp": task.start_time,
    }

    await queue.put(task)

    try:
        result = await asyncio.wait_for(task.future, timeout=WORKER_TIMEOUT * 5)
        return result
    except asyncio.TimeoutError:
        tasks_failed += 1
        logger.error(json.dumps({"event": "request_timeout", "request_id": request_id}))
        raise HTTPException(status_code=500, detail="Inference request timed out")


@app.get("/status")
async def status():
    """Quick stats endpoint for dashboards / health probes."""
    return {
        "workers": [
            {
                "id": w["id"],
                "alive": w["alive"],
                "tasks_processed": w["tasks_processed"],
                "last_seen": w["last_seen"],
            }
            for w in workers
        ],
        "tasks_submitted": tasks_submitted,
        "tasks_completed": tasks_completed,
        "tasks_failed": tasks_failed,
        "tasks_retried": tasks_retried,
        "queue_length": queue.qsize(),
        "pending_requests": list(pending_tasks.values()),
        "recent_requests": list(reversed(list(recent_requests)[-10:])),
    }


@app.get("/recent_requests")
async def recent_requests_endpoint(limit: int = 5000):
    """Return the most recent `limit` requests (max REQUEST_LOG_SIZE)."""
    limit = max(1, min(limit, REQUEST_LOG_SIZE))
    return list(reversed(list(recent_requests)[-limit:]))


# ───────────────────── dispatcher / batching logic ─────────────────────
async def dispatcher_loop() -> None:
    """Pull tasks from the queue, bundle into micro‑batches, send to workers."""
    global next_worker_index

    while True:
        # ───   Take first task ─────────────────────────────────────
        first_task: Task = await queue.get()
        # Now we ensure that no empty batches are sent
        batch: List[Task] = [first_task]
        batch_start = time.time()

        # ───  Collect micro‑batch until size or timeout ───────────
        while len(batch) < BATCH_SIZE:
            remaining = BATCH_TIMEOUT - (time.time() - batch_start)
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break

        # ───  Determine currently alive workers ─────────────────────
        alive = [w for w in workers if w["alive"]]
        if not alive:
            for t in batch:
                _fail_task(t, error="No alive workers available")
            continue

        # ───  Build fair round‑robin order snapshot ──────────────
        start = next_worker_index % len(alive)
        worker_order = alive[start:] + alive[:start]
        # advance global pointer for next batch
        next_worker_index = (next_worker_index + 1) % len(alive)

        logger.debug(
            json.dumps(
                {
                    "event": "dispatch_batch",
                    "batch_size": len(batch),
                    "worker_rr": [w["id"] for w in worker_order],
                }
            )
        )

        # ─── 5️⃣  Send to handler ─────────────────────────────────────
        await _handle_batch(batch, worker_order)


async def _handle_batch(batch: List[Task], worker_order: List[dict]) -> None:
    """
    Try the given batch on each worker in *worker_order* until one succeeds.

    `worker_order` is a pre‑computed, fair, round‑robin snapshot produced
    by `dispatcher_loop()`. This guarantees that both the first attempt and
    any subsequent retries follow the same global scheduling policy.
    """
    global tasks_completed, tasks_failed, tasks_retried

    # iterate over the snapshot; skip workers that have already been marked dead
    for attempt, worker in enumerate(worker_order):
        if not worker["alive"]:
            continue

        req_ids = [t.id for t in batch]
        inputs = [t.input_text for t in batch]

        # update per‑task bookkeeping
        for t in batch:
            t.attempts = attempt + 1
            info = pending_tasks.get(t.id)
            if info:
                info["status"] = "processing"
                info["worker"] = worker["id"]

        request = inference_pb2.InferRequest(request_ids=req_ids, input_texts=inputs)

        try:
            response = await asyncio.wait_for(
                worker["stub"].Infer(request), timeout=WORKER_TIMEOUT
            )

            if response.success and len(response.output_texts) == len(batch):
                worker_latency = time.time() - batch[0].start_time

                # bookkeeping on success
                worker["tasks_processed"] += len(batch)
                now = time.time()
                for t, out in zip(batch, response.output_texts):
                    t.worker_assigned = worker["id"]
                    t.result = {
                        "request_id": t.id,
                        "output": out,
                        "worker": worker["id"],
                        "success": True,
                        "retries": t.attempts - 1,
                    }
                    tasks_completed += 1
                    _record_request(
                        t,
                        success=True,
                        output=out,
                        retries=t.attempts - 1,
                        latency=now - t.start_time,
                    )
                    pending_tasks.pop(t.id, None)
                    t.future.set_result(t.result)

                logger.info(
                    json.dumps(
                        {
                            "event": "batch_ok",
                            "worker": worker["id"],
                            "batch_size": len(batch),
                            "latency": worker_latency,
                            "retries": attempt,
                        }
                    )
                )
                return  # batch done

        except asyncio.TimeoutError:
            tasks_retried += len(batch)
            worker["alive"] = False
            logger.warning(
                json.dumps(
                    {
                        "event": "worker_timeout",
                        "worker": worker["id"],
                        "attempt": attempt,
                        "batch_size": len(batch),
                    }
                )
            )

        except grpc.RpcError as e:
            tasks_retried += len(batch)
            worker["alive"] = False
            logger.warning(
                json.dumps(
                    {
                        "event": "rpc_error",
                        "worker": worker["id"],
                        "details": str(e)[:200],
                    }
                )
            )

    # If we reach here, *all* workers in the snapshot failed
    for t in batch:
        _fail_task(
            t,
            error="All previously alive workers failed",
            retries=t.attempts - 1,
        )


def _fail_task(t: Task, *, error: str, retries: int | None = None) -> None:
    """Mark a task as failed and resolve its Future."""
    global tasks_failed
    tasks_failed += 1
    t.result = {"request_id": t.id, "success": False, "error": error}
    _record_request(
        t,
        success=False,
        error=error,
        retries=retries,
        latency=time.time() - t.start_time,
    )
    pending_tasks.pop(t.id, None)
    if not t.future.done():
        t.future.set_result(t.result)
    logger.error(json.dumps({"request_id": t.id, "error": error}))


# ───────────────────── heartbeat loop ─────────────────────
async def heartbeat_loop() -> None:
    """Ping each worker every few seconds – simple liveness check."""
    while True:
        for w in workers:
            prev_alive = w["alive"]
            try:
                ok = await asyncio.wait_for(
                    w["stub"].HealthCheck(inference_pb2.HealthRequest()), timeout=1.0
                )
                if ok.ok:
                    w["alive"] = True
                    w["last_seen"] = time.time()
            except Exception:
                w["alive"] = False

            if prev_alive != w["alive"]:
                logger.warning(
                    json.dumps(
                        {
                            "event": "worker_up" if w["alive"] else "worker_down",
                            "worker": w["id"],
                        }
                    )
                )

        await asyncio.sleep(5)


# ─────────────────────────── FastAPI hooks ──────────────────────────────
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(dispatcher_loop())
    asyncio.create_task(heartbeat_loop())
    logger.info(f"Coordinator up – workers: {[w['id'] for w in workers]}")
