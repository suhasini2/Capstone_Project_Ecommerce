"""Two chunking strategies (Task 3). Pure Python, no heavy imports."""
import re


def fixed_chunks(text: str, size: int = 30, overlap: int = 10) -> list:
    """Fixed-size windows of `size` words with `overlap` shared words (word-boundary safe)."""
    assert 0 <= overlap < size
    words, step, i, out = text.split(), size - overlap, 0, []
    while words:
        out.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += step
    return out


def sentence_chunks(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


STRATEGIES = {"fixed": fixed_chunks, "sentence": sentence_chunks}


def chunk_document(doc: dict, strategy: str) -> list:
    """-> [{'chunk_id', 'doc_id', 'text'}] with ids that map back to the parent document."""
    return [{"chunk_id": f"{doc['doc_id']}::{strategy}::{i}", "doc_id": doc["doc_id"], "text": t}
            for i, t in enumerate(STRATEGIES[strategy](doc["content"]))]
