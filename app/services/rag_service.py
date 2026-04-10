import json
from collections.abc import Iterator

from app.models.message import Message
from app.services import embedding_service, llm_service, reranker_service
from app.vector_store import qdrant_store

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp luật Việt Nam. "
    "Nhiệm vụ của bạn là giải đáp thắc mắc pháp lý của người dùng một cách chính xác, ngắn gọn và dễ hiểu."
)


import math

def _format_sources(top_chunks: list[dict]) -> list[dict]:
    sources = []
    for c in top_chunks:
        # Chuyển đổi rerank_score (logits) sang % (sigmoid)
        score = c.get("rerank_score", 0.0)
        relevance = round((1 / (1 + math.exp(-score))) * 100, 1)
        sources.append({
            "title": c.get("title", ""),
            "article": c.get("article", ""),
            "chapter": c.get("chapter", ""),
            "relevance": relevance
        })
    return sources


def _retrieve(query: str) -> tuple[list[dict], list[dict]]:
    """Embed → search → rerank. Returns (top_chunks, sources)."""
    query_vector = embedding_service.embed(query)
    candidates = qdrant_store.search(query_vector, top_k=10)
    top_chunks = reranker_service.rerank(query, candidates, top_n=5)
    return top_chunks, _format_sources(top_chunks)





def _build_messages(query: str, history: list[Message], context: str) -> list[dict]:
    """Build the prompt message list for the LLM."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # 1. Đưa lịch sử trò chuyện vào
    for msg in history[-10:]:
        msgs.append({"role": msg.role, "content": msg.content})
        
    # 2. Đóng gói Context và Query chung vào tin nhắn cuối cùng để AI không bối rối
    final_user_prompt = (
        "Dựa vào các phần tài liệu tham khảo dưới đây để trả lời câu hỏi. "
        "Nếu tài liệu không đủ thông tin, hãy nói rõ là không thấy trong tài liệu.\n\n"
        f"--- TÀI LIỆU THAM KHẢO ---\n{context}\n\n"
        f"--- CÂU HỎI --- \n{query}"
    )
    msgs.append({"role": "user", "content": final_user_prompt})
    return msgs


def answer(query: str, history: list[Message]) -> dict:
    """Full RAG pipeline — non-streaming."""
    top_chunks, sources = _retrieve(query)
    context = "\n\n".join(f"[{c.get('title', '')}]\n{c['context']}" for c in top_chunks)
    messages = _build_messages(query, history, context)
    return {"answer": llm_service.generate(messages), "sources": sources}


def answer_stream(query: str, history: list[Message]) -> Iterator[str]:
    """Streaming RAG pipeline. Yields SSE-formatted strings."""
    top_chunks, sources = _retrieve(query)
    context = "\n\n".join(f"[{c.get('title', '')}]\n{c['context']}" for c in top_chunks)
    messages = _build_messages(query, history, context)

    for token in llm_service.generate_stream(messages):
        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


