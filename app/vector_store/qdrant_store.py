from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

from config.settings import settings

client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

COLLECTION = settings.qdrant_collection
CACHE_COLLECTION = "legal_cache"
VECTOR_SIZE = 1024  # BGE-M3 output dimension


def ensure_collection():
    """Create collection if it doesn't exist."""
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": models.VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            }
        )
    
    if not client.collection_exists(CACHE_COLLECTION):
        client.create_collection(
            collection_name=CACHE_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert(points: list[PointStruct]):
    """Upsert points into the collection."""
    client.upsert(collection_name=COLLECTION, points=points)


def search(dense_vector: list[float], sparse_vector: dict = None, top_k: int = 10) -> list[dict]:
    """Sync Hybrid Search (Dense + Sparse RRF)."""
    
    if not sparse_vector:
        # Fallback to pure dense search
        results = client.query_points(
            collection_name=COLLECTION,
            query=dense_vector,
            using="dense", # Bắt buộc chỉ định khi collection có named vectors
            limit=top_k,
            with_payload=True,
        )
        return [{"score": r.score, **r.payload} for r in results.points]

    # RRF Hybrid Search
    prefetch_dense = models.Prefetch(query=dense_vector, using="dense", limit=top_k * 2)
    prefetch_sparse = models.Prefetch(query=models.SparseVector(**sparse_vector), using="sparse", limit=top_k * 2)
    
    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[prefetch_dense, prefetch_sparse],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [{"score": r.score, **r.payload} for r in results.points]


def upsert_cache(query_vector: list[float], answer: str, query_text: str = "", sources: list[dict] = None, cache_type: str = "auto_generated"):
    """Save an answer against the query vector in the cache with rich metadata."""
    point_id = str(uuid.uuid4())
    
    legal_basis = []
    if sources:
        for s in sources:
            basis = f"{s.get('article', '')} | {s.get('chapter', '')}".strip()
            if basis and basis != "|":
                legal_basis.append(basis)

    payload = {
        "question": query_text,
        "answer": answer,
        "legal_basis": list(set(legal_basis)), # Unique lists
        "cache_type": cache_type,
        "domain": "VN_Law"
    }
    
    point = PointStruct(id=point_id, vector=query_vector, payload=payload)
    client.upsert(collection_name=CACHE_COLLECTION, points=[point])


def search_cache(query_vector: list[float], threshold: float = 0.96) -> str | None:
    """Find a cached answer if similarity > threshold."""
    results = client.query_points(
        collection_name=CACHE_COLLECTION,
        query=query_vector,
        limit=1,
        with_payload=True,
    )
    if results.points and results.points[0].score >= threshold:
        return results.points[0].payload.get("answer")
    return None
