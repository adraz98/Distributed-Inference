"""
Worker – gRPC server that executes a **batched** sentiment model in ONNXRuntime.
"""

from __future__ import annotations

import os
import asyncio
import json
import logging
import random

import grpc
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast

# ── gRPC stubs ─────────────────────────────────────────────────────────────
import inference_pb2
import inference_pb2_grpc

# ── health service ─────────────────────────────────────────────────────────
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

# ── logging & env ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("worker")

WORKER_ID = os.getenv("WORKER_ID", "worker")
CRASH_RATE = float(os.getenv("CRASH_RATE", "0.1"))   # hard‑failure probability
MAX_DELAY = float(os.getenv("MAX_DELAY", "4.0"))     # artificial latency (seconds)
SERVICE_FQN = "inference.InferenceService"

# ── device & ONNXRuntime session ──────────────────────────────────────────
providers = (
    ["CUDAExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers()
    else ["CPUExecutionProvider"]
)
logger.info(f"ONNXRuntime providers = {providers}")

MODEL_PATH = "/app/models/distilbert-sst2.onnx"
ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


# ── inference service implementation ──────────────────────────────────────
class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    async def Infer(self, request, context):
        req_ids = list(request.request_ids)
        texts = list(request.input_texts)

        logger.info(
            json.dumps(
                {
                    "event": "inference_request",
                    "worker": WORKER_ID,
                    "batch_size": len(texts),
                }
            )
        )

        # ── simulate hard crash ───────────────────────────────────────────
        if random.random() < CRASH_RATE:
            logger.error("Simulated hard crash triggered!")
            os._exit(1)

        # ── simulate network / compute delay ──────────────────────────────
        if MAX_DELAY > 0:
            await asyncio.sleep(random.uniform(0, MAX_DELAY))

        # ── actual model forward pass ─────────────────────────────────────
        try:
            inputs = tokenizer(texts, return_tensors="np", truncation=True, padding=True)
            ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

            logits = ort_session.run(None, ort_inputs)[0]
            preds = np.argmax(logits, axis=1)
            results = ["NEGATIVE" if p == 0 else "POSITIVE" for p in preds]

            logger.info(
                json.dumps(
                    {
                        "event": "inference_result",
                        "worker": WORKER_ID,
                        "batch_size": len(results),
                    }
                )
            )
            return inference_pb2.InferResponse(
                request_ids=req_ids,
                output_texts=results,
                worker_id=WORKER_ID,
                success=True,
            )

        except Exception:
            logger.exception("Unexpected error during ONNX forward pass")
            return inference_pb2.InferResponse(
                request_ids=req_ids,
                output_texts=[],
                worker_id=WORKER_ID,
                success=False,
            )

    async def HealthCheck(self, request, context):
        return inference_pb2.HealthResponse(ok=True, worker_id=WORKER_ID)


# ── gRPC server startup ───────────────────────────────────────────────────
async def serve() -> None:
    server = grpc.aio.server()
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set(SERVICE_FQN, health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port("[::]:50051")
    logger.info(f"Worker {WORKER_ID} starting on :50051")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())