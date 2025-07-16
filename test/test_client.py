import requests
import random
from concurrent.futures import ThreadPoolExecutor

COORDINATOR_URL = "http://coordinator:8000/infer"

# Test inputs
texts = [
    "I love this product, it is fantastic!",
    "This is the worst experience ever.",
    "Not sure how I feel about this.",
    "Absolutely wonderful performance.",
    "Terrible outcome, very disappointed."
]

def send_request(text: str, index: int) -> None:
    payload = {"input": text}
    try:
        resp = requests.post(COORDINATOR_URL, json=payload, timeout=120)
        log_prefix = f"[Request {index}] Input: {text!r} -> "

        if resp.status_code == 200:
            data = resp.json()
            if data.get("success", False):
                print(
                    f"{log_prefix}Success: output={data.get('output')!r} "
                    f"(handled by {data.get('worker')}, retries={data.get('retries')})"
                )
            else:
                print(f"{log_prefix}Failed: {data.get('error')}")
        else:
            print(f"{log_prefix}HTTP {resp.status_code}: {resp.text}")
    except Exception as exc:
        print(f"{log_prefix}Exception: {exc}")

# Fire off a burst of concurrent requests
num_requests = 100
inputs = [random.choice(texts) for _ in range(num_requests)]

print(f"Sending {num_requests} requests to {COORDINATOR_URL}...")
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(send_request, inputs[i], i) for i in range(num_requests)]
    for future in futures:
        future.result()

print("Test traffic completed.")