# Nykaa Domain Support Agent (LangGraph)

**Track completed: Nykaa (E-commerce & Retail).** Everything runs under `MOCK_LLM=1` (the default): **zero API keys, zero network** after the one-time download of the free `all-MiniLM-L6-v2` embedding model.

## Dataset design choices (Part 1, Task 1) - reproducible
| Choice | Value |
|---|---|
| Seed / size | `SEED = 42`, `NUM_ORDERS = 48` |
| Category weights | Apparel .22, Electronics .18, Home .15, Footwear .20, Beauty .25 |
| Status weights | Placed .15, Shipped .25, Delivered .40, Returned .10, Refunded .10 |
| Delay probability | 0.20 per order (resulting `delayed_shipment=True` rate: 14.58%, inside the 10-30% band) |
| Order value (INR) | per category: Apparel 499-5999, Electronics 999-39999, Home 399-12999, Footwear 699-7999, Beauty 199-3999 |
| Reasoning | a Nykaa basket spans a Rs 199 beauty item to a Rs 40k electronics item, so each category gets its own realistic band |

Records are never hand-edited; `validate_orders()` asserts every threshold in the brief.

## Run it
```bash
pip install -r requirements.txt
python -m tests.test_logic          # pure-logic tests (or: pytest -q)
python run_all.py                   # runs every task demo, writes transcripts/*.txt, fills the Results section below
```
Individual pieces: `python -m evaluation.rag_eval` (must run first: it calibrates the threshold),
`python -m scripts.demo_agent routing|memory|guardrails`, `uvicorn api.main:app --port 8080`,
`python -m mcp_layer.demo`, `python -m resilience.checkpoint_demo`, `python -m resilience.retry_timeout_demo`.

## Architecture
```
START -> memory_load -> input_guard -> router --(conditional edge)--> rag_node -> output_guard -> finalize -> END
                                              |-> order_node ------------------------------------> finalize
                                              '-> finalize (blocked: prompt injection)
```
7 nodes, 1 conditional edge. Router is rule-based (deterministic under MOCK_LLM).

| Task | Where |
|---|---|
| 1 dataset | `dataset.py` |
| 2 KB (12 docs) | `knowledge_base/documents.py` |
| 3 two chunkings, two Chroma collections | `rag/chunking.py`, `rag/store.py` |
| 4 grounded generation + calibrated threshold | `rag/generate.py`, `evaluation/rag_eval.py` |
| 5 P@3 / R@3 both collections | `evaluation/rag_eval.py`, `rag/metrics.py` |
| 6 escalation score | `agent/tools.py` |
| 7 LangGraph, conditional routing | `agent/graph.py`, `agent/router.py` |
| 8 JSON-file memory | `agent/memory.py` |
| 9 JSON Schema | `agent/schema.py` |
| 10 guardrails | `agent/guardrails.py` (+ nodes in `graph.py`) |
| 11-12 FastAPI + JSONL logs | `api/main.py`, `logging_utils.py` |
| 13 RAG triad (15 queries) | `evaluation/triad.py`, `judge.py` |
| 14 MCP | `mcp_layer/server.py`, `client.py`, `demo.py` |
| 15 SQLite checkpointing | `resilience/checkpoint_demo.py` |
| 16 retries + timeouts | `resilience/policies.py`, `retry_timeout_demo.py` |

## Escalation score (Task 6)
`escalation_score = 0.6 * delayed_shipment + 0.4 * min(days_since_created / 30, 1)`.
Threshold = 80th percentile (nearest-rank) of the scores in my generated dataset; escalate when score > threshold.
Limitation: the score ignores status, so an old Delivered order can still score high.

## Retry / timeout settings (Task 16)
max_attempts 4, initial_interval 0.2 s, backoff_factor 2.0, max_interval 1.0 s, jitter on. Per-node timeout is
thread-based (a Python thread cannot be killed, so the abandoned call finishes in the background); the global
timeout is cooperative: the caller errors at the deadline and every node checks the deadline before starting.

## Honest limitations
- `MOCK_LLM` generation is extractive (returns the best chunk) and the triad judge is a deterministic heuristic
  (cosine + token overlap), not a real LLM judge. A real model is optional via `MOCK_LLM=0` + `GROQ_API_KEY` (untested).
- Customer name / address are out of scope for masking (free text), as stated in the brief.
- Added documents (`/add-document`) live in Chroma for the process lifetime only.

## Results (auto-filled by `python run_all.py`)
<!-- RESULTS:START -->
### RAG comparison and calibration

| collection | mean P@3 | mean R@3 |
|---|---|---|
| fixed | 0.556 | 1.000 |
| sentence | 0.750 | 1.000 |

Recommendation: deploy the 'sentence' collection - mean P@3 0.750 / R@3 1.000 versus 0.556 / 1.000 for 'fixed'.

| collection | in-scope top-1 cosines | out-of-scope top-1 cosines | threshold |
|---|---|---|---|
| fixed | [0.518, 0.686, 0.746, 0.617] | [0.129, 0.024, 0.21] | 0.364 |
| sentence | [0.523, 0.8, 0.76, 0.589] | [0.096, 0.073, 0.234] | 0.378 |

### RAG triad (mock judge, 15 queries)

| context relevance | groundedness | answer relevance |
|---|---|---|
| 0.579 | 1.0 | 0.684 |

<!-- RESULTS:END -->

## Analysis (write in your own words)
Recommendation on chunking strategy, citing the two sets of numbers above: _TODO_
