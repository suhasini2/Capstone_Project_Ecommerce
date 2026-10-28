"""Tasks 3-5: index both chunkings, calibrate the 'I don't know' threshold, compare P@3/R@3, demo grounded generation.
Run:  python -m evaluation.rag_eval"""
import json
import statistics as st

from config import RAG_CONFIG_FILE, ARTIFACTS
from evaluation.eval_data import (CALIBRATION_IN, CALIBRATION_OUT, DEMO_OUT_OF_SCOPE, RETRIEVAL_QUERIES)
from knowledge_base.documents import KB_DOCUMENTS
from rag.chunking import chunk_document
from rag.generate import answer_query
from rag.metrics import dedup_docs, precision_recall
from rag.store import VectorKB


def main():
    kb = VectorKB(rebuild_docs=KB_DOCUMENTS)
    results, summary_lines = {}, []
    for s in ("fixed", "sentence"):
        print(f"[index] {s:8s}: {sum(len(chunk_document(d, s)) for d in KB_DOCUMENTS)} chunks from {len(KB_DOCUMENTS)} docs")
    print("\nsample retrieval (sentence collection):", [(h['doc_id'], round(h['score'], 3)) for h in
                                                      kb.search('sentence', RETRIEVAL_QUERIES[0][0])])

    # ---- Task 5: Precision@3 / Recall@3, document level, per collection
    for s in ("fixed", "sentence"):
        print(f"\n=== Collection '{s}': Precision@3 / Recall@3 (chunks -> parent docs -> dedup) ===")
        ps, rs = [], []
        for q, rel in RETRIEVAL_QUERIES:
            docs = dedup_docs(kb.search(s, q, 3))
            m = precision_recall(docs, rel)
            ps.append(m["p"]); rs.append(m["r"])
            print(f"Q: {q}\n   retrieved docs={docs} relevant={sorted(rel)}\n   P@3 = {m['p_str']}   R@3 = {m['r_str']}")
        results[s] = {"P": st.mean(ps), "R": st.mean(rs)}
        print(f"MEAN  P@3={results[s]['P']:.3f}  R@3={results[s]['R']:.3f}")

    best = max(("sentence", "fixed"), key=lambda s: results[s]["P"] + results[s]["R"])  # tie -> sentence
    other = "fixed" if best == "sentence" else "sentence"
    rec = (f"Recommendation: deploy the '{best}' collection - mean P@3 {results[best]['P']:.3f} / R@3 {results[best]['R']:.3f} "
           f"versus {results[other]['P']:.3f} / {results[other]['R']:.3f} for '{other}'.")
    print("\n" + rec)

    # ---- Task 4: threshold calibration, measured per collection
    thresholds = {}
    for s in ("fixed", "sentence"):
        ins = [kb.search(s, q, 1)[0]["score"] for q in CALIBRATION_IN]
        outs = [kb.search(s, q, 1)[0]["score"] for q in CALIBRATION_OUT]
        lo_in, hi_out = min(ins), max(outs)
        if lo_in > hi_out:
            thr = (lo_in + hi_out) / 2
            note = "clusters separate cleanly -> midpoint of (lowest in-scope, highest out-of-scope)"
        else:
            thr = (st.mean(ins) + st.mean(outs)) / 2
            note = "WARNING clusters overlap -> midpoint of the two cluster means"
        thresholds[s] = round(thr, 3)
        print(f"\n[calibration:{s}] in-scope top-1 cosine  = {[round(x, 3) for x in ins]}"
              f"\n[calibration:{s}] out-of-scope top-1     = {[round(x, 3) for x in outs]}"
              f"\n[calibration:{s}] chosen threshold = {thresholds[s]}  ({note})")
        summary_lines.append(f"| {s} | {[round(x,3) for x in ins]} | {[round(x,3) for x in outs]} | {thresholds[s]} |")

    RAG_CONFIG_FILE.write_text(json.dumps({"strategy": best, "thresholds": thresholds,
                                           "precision_recall": results}, indent=2))

    # ---- Task 4: grounded generation demo with the recommended collection
    thr = thresholds[best]
    print(f"\n=== Grounded generation ({best} collection, threshold {thr}) ===")
    for q, _ in RETRIEVAL_QUERIES:
        r = answer_query(kb, best, q, thr)
        print(f"Q: {q}\n   top-1 sim={r['top_score']:.3f} fallback={r['fallback']}\n   A: {r['answer']}")
    r = answer_query(kb, best, DEMO_OUT_OF_SCOPE, thr)
    print(f"Q (out of scope): {DEMO_OUT_OF_SCOPE}\n   top-1 sim={r['top_score']:.3f} fallback={r['fallback']}\n   A: {r['answer']}")

    md = ["### RAG comparison and calibration", "",
          "| collection | mean P@3 | mean R@3 |", "|---|---|---|",
          *[f"| {s} | {results[s]['P']:.3f} | {results[s]['R']:.3f} |" for s in results], "",
          rec, "", "| collection | in-scope top-1 cosines | out-of-scope top-1 cosines | threshold |", "|---|---|---|---|",
          *summary_lines, ""]
    (ARTIFACTS / "results_rag.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
