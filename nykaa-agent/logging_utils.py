"""One JSON-Lines entry per request (Task 12). The query is masked INSIDE the logger, so a caller
cannot accidentally write a raw phone number or card digits to disk."""
import json
import threading
import time

from agent.guardrails import mask_pii
from config import REQUEST_LOG


class RequestLogger:
    def __init__(self, path=REQUEST_LOG):
        self.path, self._lock = path, threading.Lock()

    def log(self, trace_id, endpoint, raw_text="", **fields):
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "trace_id": trace_id, "endpoint": endpoint,
                 "query_masked": mask_pii(raw_text)[0], **fields}
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
