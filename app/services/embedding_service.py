from sentence_transformers import SentenceTransformer

from config.settings import settings

model = SentenceTransformer(settings.embedding_model)


def embed(text: str) -> list[float]:
    """Embed a single text string."""
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings."""
    return model.encode(texts, normalize_embeddings=True).tolist()

