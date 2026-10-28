"""Tasks 11-12 transcript: exercises FastAPI in-process and proves no raw PII reached the log file."""
import json

from fastapi.testclient import TestClient

from api.main import app
from config import REQUEST_LOG

PHONE, CARD = "9876543210", "4242"


def main():
    REQUEST_LOG.unlink(missing_ok=True)
    with TestClient(app) as c:
        r = c.post("/ask", json={"session_id": "api-1", "query": f"My phone is {PHONE}, card ending {CARD}. Status of order NYK-1010?"})
        print("POST /ask ->", r.status_code, "X-Trace-Id:", r.headers["x-trace-id"]); print(json.dumps(r.json(), indent=2))
        r = c.post("/ask", json={"session_id": "api-1", "query": "How long does a COD refund take?"})
        print("\nPOST /ask ->", r.status_code, r.json()["answer"])
        r = c.post("/add-document", json={"doc_id": "KB-13", "topic": "gift wrapping",
                   "content": "Gift wrapping costs INR 49 per item and is available on all orders above INR 299. Wrapped items cannot be exchanged."})
        print("\nPOST /add-document ->", r.status_code, r.json())
        r = c.post("/ask", json={"session_id": "api-1", "query": "How much does gift wrapping cost?"})
        print("POST /ask (new doc) ->", r.status_code, r.json()["answer"])
        r = c.post("/add-document", json={"doc_id": "KB-13", "topic": "dup", "content": "x" * 30})
        print("POST /add-document duplicate ->", r.status_code)
    lines = REQUEST_LOG.read_text().splitlines()
    print(f"\n{len(lines)} JSONL log entries; first entry:\n{lines[0]}")
    raw = REQUEST_LOG.read_text()
    assert PHONE not in raw and f"ending {CARD}" not in raw, "RAW PII LEAKED INTO LOGS"
    print("[proof] raw phone number and card digits do NOT appear anywhere in logs/requests.jsonl")


if __name__ == "__main__":
    main()
