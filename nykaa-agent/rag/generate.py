"""Grounded generation with an empirically calibrated 'I don't know' threshold (Task 4)."""
from config import MOCK_LLM

FALLBACK = "I don't know based on our policy documents. I can connect you with a human support agent."
PROMPT = ("Answer the question using ONLY the context below. If the context does not contain the answer, "
          "say you don't know.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:")


def generate(query: str, contexts: list) -> str:
    if MOCK_LLM:  # deterministic extractive 'LLM': the best-matching chunk, verbatim
        return contexts[0]
    from llm import complete
    return complete(PROMPT.format(context="\n".join(contexts), query=query))


def answer_query(kb, strategy: str, query: str, threshold: float, k: int = 3) -> dict:
    hits = kb.search(strategy, query, k)
    top = hits[0]["score"] if hits else 0.0
    if not hits or top < threshold:
        return {"answer": FALLBACK, "fallback": True, "hits": hits, "top_score": top}
    usable = [h["text"] for h in hits if h["score"] >= threshold]
    return {"answer": generate(query, usable), "fallback": False, "hits": hits, "top_score": top}
