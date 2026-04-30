from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from config.settings import settings

model = SentenceTransformer(settings.embedding_model, device="cpu")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def embed(text: str, return_sparse: bool = False):
    """Embed a single text string."""
    dense_vec = model.encode(text, normalize_embeddings=True).tolist()
    if not return_sparse:
        return dense_vec
        
    sparse_emb = list(sparse_model.embed([text]))[0]
    sparse_vec = {"indices": sparse_emb.indices.tolist(), "values": sparse_emb.values.tolist()}
    return dense_vec, sparse_vec


def embed_batch(texts: list[str], return_sparse: bool = False):
    """Embed a batch of text strings."""
    dense_vecs = model.encode(texts, normalize_embeddings=True).tolist()
    if not return_sparse:
        return dense_vecs
        
    sparse_embs = list(sparse_model.embed(texts))
    
    results = []
    for d_vec, s_emb in zip(dense_vecs, sparse_embs):
        s_vec = {"indices": s_emb.indices.tolist(), "values": s_emb.values.tolist()}
        results.append((d_vec, s_vec))
    return results

