"""Document-level Precision@k / Recall@k (Task 5): chunks -> parent docs -> dedup -> score."""


def dedup_docs(hits: list) -> list:
    seen, out = set(), []
    for h in hits:
        if h["doc_id"] not in seen:
            seen.add(h["doc_id"])
            out.append(h["doc_id"])
    return out


def precision_recall(retrieved_docs: list, relevant: set) -> dict:
    found = [d for d in retrieved_docs if d in relevant]
    p = len(found) / len(retrieved_docs) if retrieved_docs else 0.0
    r = len(found) / len(relevant) if relevant else 0.0
    return {"p": p, "r": r, "found": len(found), "retrieved": len(retrieved_docs), "relevant": len(relevant),
            "p_str": f"{len(found)}/{len(retrieved_docs)} = {p:.2f}",
            "r_str": f"{len(found)}/{len(relevant)} = {r:.2f}"}
