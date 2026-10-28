"""Conversation memory persisted to a JSON file (Task 8). Stores only already-masked text."""
import json
import os
import threading
from pathlib import Path


class JsonMemory:
    def __init__(self, path):
        self.path, self._lock = Path(path), threading.Lock()

    def _read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {"sessions": {}}

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.path)

    def load(self, session_id: str) -> dict:
        with self._lock:
            return self._read()["sessions"].get(session_id, {"turns": [], "last_record_id": None})

    def append_turn(self, session_id, user, agent, intent, record_id=None) -> None:
        with self._lock:
            data = self._read()
            s = data["sessions"].setdefault(session_id, {"turns": [], "last_record_id": None})
            s["turns"].append({"user": user, "agent": agent, "intent": intent})
            if record_id:
                s["last_record_id"] = record_id
            self._write(data)

    def reset(self, session_id: str) -> None:
        with self._lock:
            data = self._read()
            data["sessions"].pop(session_id, None)
            self._write(data)
