import json
from collections.abc import Iterator

from app.models.message import Message
from app.services import embedding_service, llm_service, reranker_service, gemini_service
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


def _retrieve(query: str, query_vector: list[float] = None) -> tuple[list[dict], list[dict]]:
    """Selective Query Rewriting & Selective Reranking with Hybrid Search."""
    
    # Sinh song song Dense và Sparse Vector cho câu hỏi gốc
    dense_vec, sparse_vec = embedding_service.embed(query, return_sparse=True)
        
    # ==========================================
    # ROUTE 1: FAST PATH (Đường trơn)
    # ==========================================
    fast_candidates = qdrant_store.search(dense_vector=dense_vec, sparse_vector=sparse_vec, top_k=5)
    fast_top_chunks = reranker_service.rerank(query, fast_candidates, top_n=5)
    
    # Đo nhiệt độ tin cậy từ Sigmoid Reranker
    highest_score = fast_top_chunks[0].get("rerank_score", 0.0) if fast_top_chunks else -10.0
    highest_relevance = (1 / (1 + math.exp(-highest_score))) * 100
    
    if highest_relevance >= 80.0:
        print(f"\n[FAST PATH] Tìm thấy đoạn luật quá khớp (Trust = {highest_relevance:.1f}%). Bỏ qua hoàn toàn thuật toán Gemini!")
        return fast_top_chunks, _format_sources(fast_top_chunks)
        
    # ==========================================
    # ROUTE 2: HEAVY PATH (Đường hiểm trở)
    # ==========================================
    print(f"\n[HEAVY PATH] Câu hỏi hóc búa (Trust = {highest_relevance:.1f}%) -> Kích hoạt Gemini và Đào bới diện rộng...")
    
    variations = gemini_service.generate_multi_queries(query)
    
    print("\n" + "="*50)
    print("GEMINI ĐÃ MỞ RỘNG CÂU HỎI THÀNH CÁC BIẾN THỂ:")
    for i, var in enumerate(variations):
        print(f" {i+1}. {var}")
    print("="*50 + "\n")

    all_queries = [query] + variations

    # Tạo vector hàng loạt (Cả Dense và Sparse)
    query_vectors = embedding_service.embed_batch(all_queries, return_sparse=True)

    all_candidates = []
    seen_texts = set()

    for d_vec, s_vec in query_vectors:
        results = qdrant_store.search(dense_vector=d_vec, sparse_vector=s_vec, top_k=8)
        for r in results:
            chunk_content = r.get("context", "")
            if chunk_content and chunk_content not in seen_texts:
                seen_texts.add(chunk_content)
                all_candidates.append(r)

    # => TÍNH NĂNG MỚI: SELECTIVE RERANKING
    # Sắp xếp sơ bộ bằng độ đo Cosine (Càng gần 1.0 càng tốt)
    all_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    # Vứt 25 cục rác thấp nhất đi, chỉ lấy 15 cục Qdrant điểm cao đầu bảng cho Reranker chạy
    top_15_candidates = all_candidates[:15]

    heavy_top_chunks = reranker_service.rerank(query, top_15_candidates, top_n=5)
    
    return heavy_top_chunks, _format_sources(heavy_top_chunks)





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
    # 1. Semantic Cache Check
    query_vector = embedding_service.embed(query)
    cached_answer = qdrant_store.search_cache(query_vector)
    if cached_answer:
        return {"answer": cached_answer, "sources": []}

    # 2. RAG Pipeline
    top_chunks, sources = _retrieve(query, query_vector)
    context = "\n\n".join(f"[{c.get('title', '')}]\n{c['context']}" for c in top_chunks)
    messages = _build_messages(query, history, context)
    
    # Generate and Save
    ans = llm_service.generate(messages)
    qdrant_store.upsert_cache(query_vector, ans, query_text=query, sources=sources)
    return {"answer": ans, "sources": sources}


def answer_stream(query: str, history: list[Message]) -> Iterator[str]:
    """Streaming RAG pipeline. Yields SSE-formatted strings."""
    # 1. Semantic Cache Check
    query_vector = embedding_service.embed(query)
    cached_answer = qdrant_store.search_cache(query_vector)
    
    if cached_answer:
        print("\n⚡ [CACHE HIT] Đã tìm thấy câu trả lời trong bộ nhớ đệm!\n")
        yield f"data: {json.dumps({'type': 'token', 'content': cached_answer}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 2. Normal Pipeline if Cache Miss
    top_chunks, sources = _retrieve(query, query_vector)
    context = "\n\n".join(f"[{c.get('title', '')}]\n{c['context']}" for c in top_chunks)
    messages = _build_messages(query, history, context)

    full_answer_list = []
    for token in llm_service.generate_stream(messages):
        full_answer_list.append(token)
        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

    # Save final answer to cache
    qdrant_store.upsert_cache(
        query_vector=query_vector, 
        answer="".join(full_answer_list), 
        query_text=query, 
        sources=sources
    )

    yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


