from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config.settings import settings

client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

COLLECTION = settings.qdrant_collection
VECTOR_SIZE = 1024  # BGE-M3 output dimension


def ensure_collection():
    """Create collection if it doesn't exist."""
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert(points: list[PointStruct]):
    """Upsert points into the collection."""
    client.upsert(collection_name=COLLECTION, points=points)


def search(vector: list[float], top_k: int = 20) -> list[dict]:
    """Search for similar vectors, return list of payloads with scores."""
    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"score": r.score, **r.payload}
        for r in results.points
    ]
