"""Free local SentenceTransformers embeddings (lazy import so pure-logic tests need no model)."""
from config import EMBED_MODEL

_model = None


def embed(texts: list):
    """-> L2-normalised float32 array (n, dim); dot product == cosine similarity."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
