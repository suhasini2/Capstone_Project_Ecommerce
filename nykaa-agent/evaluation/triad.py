"""Task 13: RAG-triad over 15 queries.  Run: python -m evaluation.triad"""
import json
import statistics as st

from agent.factory import build_agent
from config import ARTIFACTS
from evaluation.eval_data import TRIAD_QUERIES
from evaluation.judge import judge


def main():
    agent = build_agent()
    rows = []
    print(f"{'#':>2} {'ctx_rel':>7} {'ground':>6} {'ans_rel':>7}  refused  top-doc  query")
    for i, (q, exp_doc, exp_refuse) in enumerate(TRIAD_QUERIES, 1):
        out = agent.ask_full(f"triad-{i}", q)
        resp, ctx = out["response"], [c["text"] for c in out.get("contexts", [])]
        s = judge(q, ctx, resp["answer"], resp["refused"], exp_refuse)
        top_doc = out["contexts"][0]["doc_id"] if out.get("contexts") else "-"
        rows.append({"query": q, "expected_doc": exp_doc, "top_doc": top_doc, "refused": resp["refused"], **s})
        print(f"{i:>2} {s['context_relevance']:>7.2f} {s['groundedness']:>6.2f} {s['answer_relevance']:>7.2f}  "
              f"{str(resp['refused']):7}  {top_doc:6}  {q}")
    avg = {k: round(st.mean(r[k] for r in rows), 3) for k in ("context_relevance", "groundedness", "answer_relevance")}
    print("\nAVERAGES over 15 queries:", avg)
    (ARTIFACTS / "triad_results.json").write_text(json.dumps({"rows": rows, "averages": avg}, indent=2))
    (ARTIFACTS / "results_triad.md").write_text(
        "### RAG triad (mock judge, 15 queries)\n\n| context relevance | groundedness | answer relevance |\n|---|---|---|\n"
        f"| {avg['context_relevance']} | {avg['groundedness']} | {avg['answer_relevance']} |\n")


if __name__ == "__main__":
    main()
