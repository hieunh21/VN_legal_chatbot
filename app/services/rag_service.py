from app.services import embedding_service, reranker_service, llm_service
from app.vector_store import qdrant_store
from app.models.message import Message

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp luật Việt Nam. "
    "Trả lời câu hỏi dựa trên các điều luật được cung cấp bên dưới. "
    "Nếu không tìm thấy thông tin liên quan, hãy nói rõ rằng bạn không có đủ dữ liệu để trả lời. "
    "Trả lời bằng tiếng Việt, chính xác và dễ hiểu."
)


def answer(query: str, history: list[Message]) -> dict:
    """Full RAG pipeline: embed → search → rerank → build context → LLM."""
    # 1. Embed query
    query_vector = embedding_service.embed(query)

    # 2. Vector search
    candidates = qdrant_store.search(query_vector, top_k=20)

    # 3. Rerank
    top_chunks = reranker_service.rerank(query, candidates, top_n=5)

    # 4. Build context
    context = "\n\n".join(
        f"[{chunk.get('title', '')}]\n{chunk['context']}"
        for chunk in top_chunks
    )

    # 5. Build messages
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nTài liệu tham khảo:\n{context}"}]

    for msg in history[-10:]:  # last 10 messages for context window
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": query})

    # 6. Call LLM
    response = llm_service.generate(messages)

    # 7. Build sources
    sources = [
        {"title": c.get("title", ""), "article": c.get("article", ""), "chapter": c.get("chapter", "")}
        for c in top_chunks
    ]

    return {"answer": response, "sources": sources}
