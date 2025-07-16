import time
import requests
import streamlit as st
import pandas as pd

COORDINATOR_URL = "http://coordinator:8000"

st.set_page_config(page_title="Inference Dashboard", layout="wide")
st.title("Distributed Inference System Monitor")

# ───────────────────── central data fetch ─────────────────────
def fetch_status():
    s = requests.get(f"{COORDINATOR_URL}/status", timeout=1).json()
    # pull extended request history
    r = requests.get(f"{COORDINATOR_URL}/recent_requests?limit=5000", timeout=1).json()
    s["recent_requests"] = r
    return s


try:
    data = fetch_status()
except Exception as e:
    st.error(f"Unable to reach coordinator: {e}")
    st.stop()

# ───────────────────────── headline metrics ─────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tasks Submitted", data["tasks_submitted"])
col2.metric("Tasks Completed", data["tasks_completed"])
col3.metric("Tasks Failed", data["tasks_failed"])
col4.metric("Tasks Retried", data["tasks_retried"])
col5.metric("Queue Length", data["queue_length"])

# ───────────────────────── worker table ─────────────────────────
st.markdown("### Workers")
for w in data["workers"]:
    w["last_seen_s_ago"] = round(time.time() - w["last_seen"], 1) if w["last_seen"] else None
st.table(
    [
        {
            "Worker": w["id"],
            "Alive": "✅" if w["alive"] else "❌",
            "Load (processed)": w["tasks_processed"],
            "Last heartbeat (s)": w["last_seen_s_ago"],
        }
        for w in data["workers"]
    ]
)

# ───────────────────────── pending requests ─────────────────────────
st.markdown("### Pending Requests (in‑flight & queued)")
pending = data.get("pending_requests", [])

if not pending:
    st.info("No pending requests 🎉")
else:
    pdf = pd.DataFrame(pending)
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], unit="s").dt.strftime("%Y‑%m‑%d %H:%M:%S")
    pdf["input"] = pdf["input"].str.slice(0, 120)
    pdf = pdf.sort_values("timestamp", ascending=False)

    status_emoji = {"queued": "🕒 queued", "processing": "⚙️ processing"}
    pdf["status"] = pdf["status"].map(status_emoji)

    st.dataframe(
        pdf[["id", "timestamp", "input", "status", "worker"]],
        height=300,
        use_container_width=True,
    )

# ───────────────────────── recent request log ─────────────────────────
st.markdown("### Recent Inference Requests")
if not data["recent_requests"]:
    st.info("No requests logged yet.")
else:
    df = pd.DataFrame(data["recent_requests"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y‑%m‑%d %H:%M:%S")
    df["input"] = df["input"].str.slice(0, 120)
    df["output"] = df["output"].fillna("")
    df = df.sort_values("timestamp", ascending=False)

    st.dataframe(
        df[
            [
                "timestamp",
                "request_id",
                "latency_ms",
                "input",
                "output",
                "worker",
                "success",
                "retries",
                "error",
            ]
        ],
        height=420,
        use_container_width=True,
    )

# ───────────────────────── manual refresh ─────────────────────────
if st.button("Refresh"):
    st.experimental_rerun()
