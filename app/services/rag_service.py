import json
import math
from collections.abc import Iterator
import re
from app.models.message import Message
from app.services import embedding_service, llm_service, reranker_service, gemini_service
from app.vector_store import qdrant_store

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp luật Việt Nam, chuyên về Luật Trật tự an toàn giao thông đường bộ 2024 và Luật Bảo vệ quyền lợi người tiêu dùng 2023.

## Nguyên tắc trả lời
- Chỉ trả lời dựa trên nội dung điều luật được cung cấp trong ngữ cảnh bên dưới.
- Nếu ngữ cảnh không có thông tin liên quan, trả lời đúng một câu: "Tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu."
- Tuyệt đối không suy diễn, không bổ sung thông tin từ kiến thức bên ngoài ngữ cảnh.

## Trích dẫn điều luật
- Chỉ trích dẫn khi ngữ cảnh có ghi rõ tên luật, số điều, số khoản.
- Chép nguyên văn số điều, số khoản từ ngữ cảnh — không được tự suy ra hay điều chỉnh con số.
- Nếu ngữ cảnh không ghi rõ số khoản, chỉ trích dẫn tên điều, không được tự thêm "khoản X" hay "điểm Y".

## Format trả lời
- Ngắn gọn, trực tiếp vào câu hỏi, không lan man.
- Dùng ngôn ngữ đơn giản, dễ hiểu."""


def _format_sources(top_chunks: list[dict]) -> list[dict]:
    sources = []
    for c in top_chunks:
        score = c.get("rerank_score", 0.0)
        relevance = round((1 / (1 + math.exp(-score))) * 100, 1)
        sources.append({
            "title"    : c.get("title", ""),
            "article"  : c.get("article", ""),
            "chapter"  : c.get("chapter", ""),
            "law_name" : c.get("law_name", ""),
            "relevance": relevance,
            "content"  : c.get("context", "")
        })
    return sources


def _build_context(top_chunks: list[dict]) -> str:
    parts = []
    for c in top_chunks:
        law_name = c.get("law_name", "")
        article  = c.get("article", "")
        title    = c.get("title", "")
        content  = c.get("context", "")

        # Thay "1. " → "Khoản 1. " để LLM không nhầm với số điều
        content = re.sub(r'^(\d+)\.\s', r'Khoản \1. ', content, flags=re.MULTILINE)

        # Thêm "Điểm" trước ký tự a), b), c)...: "a) " → "Điểm a) "
        content = re.sub(r'^([a-zđ])\)\s', r'Điểm \1) ', content, flags=re.MULTILINE)

        if law_name and article:
            header = f"[Nguồn: {law_name} — {article}]"
        else:
            header = f"[{title}]"

        parts.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(parts)


def _rewrite_query(query: str, history: list[Message]) -> str:
    """
    Dùng Gemini viết lại câu hỏi mơ hồ thành câu hỏi độc lập.
    Chỉ gọi khi có history — câu đầu tiên không cần rewrite.
    """
    if not history:
        return query

    # Trích xuất 4 tin nhắn gần nhất thành chuỗi văn bản cho Prompt
    history_text = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in history[-4:]])
    
    rewritten = gemini_service.rewrite_query(query, history_text)
    print(f"  [Rewrite (Gemini)] '{query}' → '{rewritten}'")
    return rewritten


def _retrieve(query: str, history: list[Message] = None, enable_multi_query: bool = True) -> tuple[list[dict], list[dict]]:
    """Selective Query Rewriting & Selective Reranking with Hybrid Search."""

    search_query = _rewrite_query(query, history or [])

    dense_vec, sparse_vec = embedding_service.embed(search_query, return_sparse=True)

    # ==========================================
    # ROUTE 1: FAST PATH
    # ==========================================
    fast_candidates = qdrant_store.search(dense_vector=dense_vec, sparse_vector=sparse_vec, top_k=10)
    fast_top_chunks = reranker_service.rerank(search_query, fast_candidates, top_n=5)

    highest_score     = fast_top_chunks[0].get("rerank_score", 0.0) if fast_top_chunks else -10.0
    highest_relevance = (1 / (1 + math.exp(-highest_score))) * 100

    if highest_relevance >= 80.0 or not enable_multi_query:
        if not enable_multi_query and highest_relevance < 80.0:
            print(f"\n[FAST PATH] Độ tin cậy thấp ({highest_relevance:.1f}%) nhưng Gemini ĐÃ BỊ TẮT.")
        else:
            print(f"\n[FAST PATH] Trust = {highest_relevance:.1f}%. Bỏ qua Gemini!")
        return fast_top_chunks, _format_sources(fast_top_chunks)

    # ==========================================
    # ROUTE 2: HEAVY PATH
    # ==========================================
    print(f"\n[HEAVY PATH] Trust = {highest_relevance:.1f}% -> Kích hoạt Gemini...")

    variations  = gemini_service.generate_multi_queries(search_query)
    all_queries = [search_query] + variations

    print("\n" + "=" * 50)
    print("GEMINI MỞ RỘNG CÂU HỎI:")
    for i, var in enumerate(variations):
        print(f"  {i+1}. {var}")
    print("=" * 50 + "\n")

    query_vectors = embedding_service.embed_batch(all_queries, return_sparse=True)

    all_candidates = []
    seen_texts     = set()

    for d_vec, s_vec in query_vectors:
        results = qdrant_store.search(dense_vector=d_vec, sparse_vector=s_vec, top_k=8)
        for r in results:
            chunk_content = r.get("context", "")
            if chunk_content and chunk_content not in seen_texts:
                seen_texts.add(chunk_content)
                all_candidates.append(r)

    all_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top_8_candidates  = all_candidates[:8]
    heavy_top_chunks  = reranker_service.rerank(search_query, top_8_candidates, top_n=5)

    return heavy_top_chunks, _format_sources(heavy_top_chunks)


def _build_messages(query: str, history: list[Message], context: str) -> list[dict]:
    """Build the prompt message list for the LLM."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history[-10:]:
        msgs.append({"role": msg.role, "content": msg.content})

    final_user_prompt = (
    "Dưới đây là TẤT CẢ tài liệu bạn được phép dùng để trả lời. "
    "KHÔNG được dùng bất kỳ thông tin nào ngoài những đoạn văn dưới đây, "
    "kể cả khi bạn biết thông tin đó là đúng.\n\n"
    "=== TÀI LIỆU ===\n"
    f"{context}\n"
    "=== HẾT TÀI LIỆU ===\n\n"
    f"Câu hỏi: {query}\n\n"
    "Nhắc lại: Chỉ được trích dẫn điều luật có trong TÀI LIỆU ở trên."
    )
    msgs.append({"role": "user", "content": final_user_prompt})

    # print("\n" + "=" * 50)
    # print("🚀 NỘI DUNG PROMPT GỬI CHO LLM:")
    # for m in msgs:
    #     print(f"👉 [{m['role'].upper()}]:\n{m['content']}")
    #     print("-" * 30)
    # print("=" * 50 + "\n")

    return msgs


def answer(query: str, history: list[Message], enable_multi_query: bool = True) -> dict:
    """Full RAG pipeline — non-streaming."""
    query_vector  = embedding_service.embed(query)
    cached_answer = qdrant_store.search_cache(query_vector)
    if cached_answer:
        return {"answer": cached_answer, "sources": []}

    top_chunks, sources = _retrieve(query, history, enable_multi_query=enable_multi_query)
    context  = _build_context(top_chunks)   # ← dùng hàm mới thay vì join thẳng
    messages = _build_messages(query, history, context)

    ans = llm_service.generate(messages)
    # Tắt lưu tự động vào cache: qdrant_store.upsert_cache(query_vector, ans, query_text=query, sources=sources)
    return {"answer": ans, "sources": sources}


def answer_stream(query: str, history: list[Message], enable_multi_query: bool = True) -> Iterator[str]:
    """Streaming RAG pipeline. Yields SSE-formatted strings."""
    query_vector  = embedding_service.embed(query)
    cached_answer = qdrant_store.search_cache(query_vector)

    if cached_answer:
        print("\n⚡ [CACHE HIT]\n")
        yield f"data: {json.dumps({'type': 'token', 'content': cached_answer}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    top_chunks, sources = _retrieve(query, history, enable_multi_query=enable_multi_query)
    context  = _build_context(top_chunks)   # ← dùng hàm mới
    messages = _build_messages(query, history, context)

    full_answer_list = []
    for token in llm_service.generate_stream(messages):
        full_answer_list.append(token)
        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

    # Tắt lưu tự động vào cache từ chat:
    # qdrant_store.upsert_cache(
    #     query_vector=query_vector,
    #     answer="".join(full_answer_list),
    #     query_text=query,
    #     sources=sources
    # )

    yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"