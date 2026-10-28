"""Two separate ChromaDB collections, one per chunking strategy (Task 3)."""
import numpy as np
from config import CHROMA_DIR
from rag.chunking import chunk_document
from rag.embeddings import embed

COLLECTION_NAMES = {"fixed": "kb_fixed", "sentence": "kb_sentence"}


class VectorKB:
    def __init__(self, path=CHROMA_DIR, rebuild_docs=None):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(path))
        self.docs = {}
        self.cols = {}
        if rebuild_docs is not None:
            self.rebuild(rebuild_docs)
        else:
            for s, name in COLLECTION_NAMES.items():
                self.cols[s] = self.client.get_collection(name)

    def rebuild(self, docs):
        self.docs = {}
        for s, name in COLLECTION_NAMES.items():
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
            self.cols[s] = self.client.create_collection(name, metadata={"hnsw:space": "cosine"})
        self.add_documents(docs)

    def add_documents(self, docs) -> dict:
        added = {}
        for s in COLLECTION_NAMES:
            chunks = [c for d in docs for c in chunk_document(d, s)]
            if chunks:
                self.cols[s].add(ids=[c["chunk_id"] for c in chunks],
                                 documents=[c["text"] for c in chunks],
                                 embeddings=embed([c["text"] for c in chunks]).tolist(),
                                 metadatas=[{"doc_id": c["doc_id"]} for c in chunks])
            added[s] = len(chunks)
        for d in docs:
            self.docs[d["doc_id"]] = d
        return added

    def search(self, strategy: str, query: str, k: int = 3) -> list:
        """Top-k chunks with cosine similarity recomputed from vectors (independent of Chroma's distance convention)."""
        qv = embed([query])[0]
        res = self.cols[strategy].query(query_embeddings=[qv.tolist()], n_results=k,
                                        include=["documents", "metadatas", "embeddings"])
        vecs = np.asarray(res["embeddings"][0])
        hits = []
        for i, text in enumerate(res["documents"][0]):
            hits.append({"chunk_id": res["ids"][0][i], "doc_id": res["metadatas"][0][i]["doc_id"],
                         "text": text, "score": float(np.dot(qv, vecs[i]))})
        return sorted(hits, key=lambda h: -h["score"])
