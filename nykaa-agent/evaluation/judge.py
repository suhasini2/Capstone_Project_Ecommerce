"""LLM-as-judge for the RAG triad (Task 13). Under MOCK_LLM the judge is a deterministic heuristic:
  context_relevance = cosine(query, best retrieved context)
  groundedness      = share of answer content tokens found in the retrieved contexts (a refusal makes no claims -> 1.0)
  answer_relevance  = cosine(query, answer); for a query that SHOULD be refused: 1.0 if refused else 0.0
With MOCK_LLM=0 the same JUDGE_PROMPT is sent to a real model."""
import json

from agent.guardrails import content_tokens
from config import MOCK_LLM

JUDGE_PROMPT = """You are a strict evaluator of a RAG support agent. Score each from 0.0 to 1.0.
- context_relevance: does the retrieved context contain what is needed to answer the question?
- groundedness: is every claim in the answer supported by the context?
- answer_relevance: does the answer directly address the question?
Question: {query}
Context: {context}
Answer: {answer}
Reply with JSON only: {{"context_relevance": x, "groundedness": y, "answer_relevance": z}}"""


def _clip(x):
    return round(max(0.0, min(1.0, float(x))), 2)


def judge(query: str, contexts: list, answer: str, refused: bool, expect_refusal: bool) -> dict:
    if not MOCK_LLM:
        from llm import complete
        raw = complete(JUDGE_PROMPT.format(query=query, context="\n".join(contexts), answer=answer))
        return {k: _clip(v) for k, v in json.loads(raw).items()}
    from rag.embeddings import embed
    q = embed([query])[0]
    ctx_rel = max((float(q @ v) for v in embed(contexts)), default=0.0) if contexts else 0.0
    if refused:
        ground = 1.0
        ans_rel = 1.0 if expect_refusal else 0.0
    else:
        a = content_tokens(answer)
        c = set().union(*(content_tokens(x) for x in contexts)) if contexts else set()
        ground = len(a & c) / len(a) if a else 0.0
        ans_rel = 0.0 if expect_refusal else float(q @ embed([answer])[0])
    return {"context_relevance": _clip(ctx_rel), "groundedness": _clip(ground), "answer_relevance": _clip(ans_rel)}
