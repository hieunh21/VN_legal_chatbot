from sentence_transformers import CrossEncoder

from config.settings import settings

model = CrossEncoder(settings.reranker_model)


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """Rerank chunks by relevance to query. Returns top_n results."""
    if not chunks:
        return []
    pairs = [(query, chunk["context"]) for chunk in chunks]
    scores = model.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

