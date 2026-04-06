from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/legal_chatbot"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "legal_chunks"

    # HuggingFace
    hf_api_token: str = ""
    hf_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # Models
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    model_config = {"env_file": ".env"}


settings = Settings()
