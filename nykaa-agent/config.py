"""Paths and global switches. MOCK_LLM=1 (default) needs no API key and no network."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS, LOGS, DATA, TRANSCRIPTS = (ROOT / d for d in ("artifacts", "logs", "data", "transcripts"))
CHROMA_DIR = ROOT / "chroma_db"
CHECKPOINT_DB = ROOT / "checkpoints.sqlite"
MEMORY_FILE = DATA / "conversations.json"
REQUEST_LOG = LOGS / "requests.jsonl"
RAG_CONFIG_FILE = ARTIFACTS / "rag_config.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"
for _d in (ARTIFACTS, LOGS, DATA, TRANSCRIPTS):
    _d.mkdir(parents=True, exist_ok=True)


def load_rag_config() -> dict:
    """Strategy + calibrated thresholds, written by `python -m evaluation.rag_eval`."""
    if not RAG_CONFIG_FILE.exists():
        raise RuntimeError("artifacts/rag_config.json missing - run `python -m evaluation.rag_eval` first "
                           "(it calibrates the 'I don't know' threshold from measured similarities).")
    return json.loads(RAG_CONFIG_FILE.read_text())
