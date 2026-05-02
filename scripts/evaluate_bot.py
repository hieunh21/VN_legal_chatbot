import os
import sys
import json
import time
import io
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services import rag_service
from app.vector_store import qdrant_store
from qdrant_client.models import Filter

DATASET_PATH  = "evaluate/golden_dataset.json"
PROGRESS_PATH = "evaluate/eval_progress.json"
REPORT_PATH   = "evaluate/eval_report.json"

# ======================== SHOPAIKEY CONFIG ========================
ai = genai.Client(
    api_key=os.getenv("SHOPAIKEY_API_KEY"),
    http_options={
        "base_url": "https://api.shopaikey.com",
        "timeout": 60000,
    },
)

GEMINI_MODEL = "gemini-2.5-flash"

CONFIG = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=4000,
    response_mime_type="application/json", # <--- DÒNG PHÉP THUẬT NẰM Ở ĐÂY
    thinking_config=types.ThinkingConfig(thinking_budget=0),
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
    ],
)

DELAY       = 5.0
MAX_RETRIES = 3


# ======================== CHECKPOINT ========================
def load_progress() -> dict:
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Resume] Tìm thấy {len(data)} câu đã chạy từ lần trước.")
        return data
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_report(report: dict):
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"💾 Báo cáo đã lưu: {REPORT_PATH}")


# ======================== HELPERS ========================
def clear_cache():
    try:
        if qdrant_store.client.collection_exists(qdrant_store.CACHE_COLLECTION):
            qdrant_store.client.delete(
                collection_name=qdrant_store.CACHE_COLLECTION,
                points_selector=Filter()
            )
            print("Đã làm sạch semantic cache.")
    except Exception as e:
        print(f"Lỗi khi xóa cache: {e}")


def ask_gemini(prompt: str) -> str:
    import json
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(DELAY)
            resp = ai.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=CONFIG,
            )
            text = (resp.text or "").strip()
            
            # KIỂM TRA NHANH: Nếu API trả về chuỗi rỗng
            if not text:
                raise ValueError("API trả về chuỗi rỗng.")
                
            # Cắt bỏ markdown (nếu có) trước khi test
            test_text = text
            if test_text.startswith("```json"): 
                test_text = test_text[7:]
            if test_text.endswith("```"): 
                test_text = test_text[:-3]
            test_text = test_text.strip()
            
            # Thử parse nhanh. Nếu Gemini gen thiếu dấu }, nó sẽ văng lỗi nhảy xuống except và Retry!
            json.loads(test_text) 
            
            return text # Nếu parse thành công thì mới trả về text

        except Exception as e:
            print(f"  [Retry {attempt}] API đứt đoạn hoặc JSON hỏng: {e}")
            time.sleep(attempt * 2)
            
    return ""


def parse_json_robust(text: str) -> dict:
    import re
    text = text.strip()
    clean_text = text
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:].strip()
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:].strip()
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3].strip()
    
    try:
        return json.loads(clean_text)
    except Exception as e1:
        try:
            # Sửa lỗi "Unterminated string" do Gemini tự ý thêm \n vào giữa chuỗi
            fixed_text = clean_text.replace('\n', ' ').replace('\r', '')
            return json.loads(fixed_text)
        except Exception:
            pass

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                fixed_match = match.group(0).replace('\n', ' ').replace('\r', '')
                return json.loads(fixed_match)
            except Exception as e2:
                print(f"  [Parse Error] Regex JSON parse failed: {e2}")
                pass
        print(f"  [Parse Error] Không thể parse JSON. Raw: >>>{text}<<<")
        raise e1


def build_full_context(sources: list) -> str:
    parts = []
    for s in sources:
        title   = s.get('title', '')
        article = s.get('article', '')
        chapter = s.get('chapter', '')
        content = s.get('content', '')
        parts.append(f"[Title: {title}]\n[Article: {article}]\n[Chapter: {chapter}]\n{content}")
    return "\n\n---\n\n".join(parts)


# ======================== METRIC 1: HIT RATE ========================
def compute_hit_rate(expected_article: str, sources: list) -> float:
    """Đánh giá hit rate với cơ chế fuzzy match (chuẩn hóa khoảng trắng)"""
    if not expected_article:
        return None
        
    # 1. Hàm phụ để dọn sạch chuỗi (xóa \n, biến nhiều khoảng trắng thành 1)
    def clean_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip())
        
    expected_clean = clean_text(expected_article)
    
    # 2. Bắt thêm từ khóa cốt lõi (Ví dụ: "điều 24")
    # Nếu expected là "Điều 24. Giải thích...", sẽ trích xuất ra chữ "điều 24"
    match = re.search(r'(điều\s+\d+[a-z]*)', expected_clean)
    short_expected = match.group(1) if match else expected_clean

    for s in sources:
        title_clean   = clean_text(s.get('title', ''))
        article_clean = clean_text(s.get('article', ''))
        
        # So khớp chuỗi đầy đủ đã dọn dẹp, hoặc chỉ cần có "điều 24" là tính HIT
        if (expected_clean in title_clean or 
            expected_clean in article_clean or 
            short_expected in title_clean or 
            short_expected in article_clean):
            return 1.0
            
    return 0.0

# ======================== METRIC 2: FAITHFULNESS ========================
FAITHFULNESS_PROMPT = """Bạn là chuyên gia đánh giá hệ thống RAG.

Nhiệm vụ: Kiểm tra từng khẳng định trong câu trả lời xem có cơ sở từ context không.

CONTEXT:
{context}

ANSWER:
{answer}

Hướng dẫn quan trọng:
1. Tách answer thành từng claim chứa nội dung thông tin cốt lõi.
2. supported=true nếu nội dung claim ĐỒNG NGHĨA hoặc CÓ THỂ SUY RA trực tiếp từ context.
3. BỎ QUA các sai sót nhỏ về format như: thiếu/thừa dấu phẩy trong tên đạo luật, dịch thuật từ ngữ (ví dụ: dùng "paragraph" thay cho "khoản", hoặc nhầm "điểm" thành "khoản") miễn là nội dung số và điều luật vẫn khớp với context.
4. supported=false CHỈ KHI claim tự bịa ra thông tin, sai lệch ý nghĩa pháp lý, hoặc trích dẫn sai số Điều/Khoản làm thay đổi hoàn toàn bản chất.

LƯU Ý QUAN TRỌNG: Bạn BẮT BUỘC phải escape (thoát) tất cả các dấu ngoặc kép (") thành (\") bên trong chuỗi giá trị (value) của JSON để không làm hỏng cấu trúc.

Chỉ trả về JSON, không giải thích thêm:
{{
  "claims": [
    {{"claim": "nội dung khẳng định", "supported": true}},
    {{"claim": "nội dung khẳng định", "supported": false}}
  ]
}}"""


def compute_faithfulness(context: str, answer: str) -> tuple:
    """Faithfulness = claim được hỗ trợ / tổng claim. Trả về (score, claims)."""
    if not answer.strip():
        return 0.0, []
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    try:
        raw = ask_gemini(prompt)
        if not raw:
            print("  [Faithfulness] Gemini trả về rỗng")
            return 0.0, []
        print(f"  [Faithfulness RAW] >>>{raw}<<<")
        data   = parse_json_robust(raw)
        claims = data.get("claims", [])
        if not claims:
            print("  [Faithfulness] Parse OK nhưng claims rỗng → mặc định 1.0")
            return None, []
        supported = sum(1 for c in claims if c.get("supported", False))
        return round(supported / len(claims), 3), claims
    except Exception as e:
        raw_text = raw if 'raw' in locals() else 'Rỗng hoặc Timeout'
        print(f"  [LỖI PARSE] Không thể đọc JSON. Lỗi Python: {e}")
        print(f"  [RAW TEXT TỪ API TRẢ VỀ]:\n>>>\n{raw_text}\n<<<")
        return None, []  # Hoặc return None, [], "" đối với hàm correctness


# ======================== METRIC 3: CONTEXT RELEVANCE ========================
CONTEXT_RELEVANCE_PROMPT = """Bạn là chuyên gia đánh giá hệ thống RAG.

Nhiệm vụ: Đánh giá context lấy về có chứa đủ thông tin để trả lời query không.

QUERY: {query}

CONTEXT:
{context}

Chỉ trả về JSON, không giải thích thêm:
{{
  "relevant_sentences": ["câu/đoạn liên quan trong context"],
  "score": <float 0.0-1.0>,
  "reason": "lý do ngắn gọn"
}}

Thang điểm:
- 1.0: Chứa đầy đủ thông tin chính xác để trả lời hoàn toàn
- 0.7: Có thông tin liên quan nhưng chưa đầy đủ
- 0.4: Chỉ liên quan một phần nhỏ
- 0.0: Hoàn toàn không liên quan"""


def compute_context_relevance(query: str, context: str) -> tuple:
    prompt = CONTEXT_RELEVANCE_PROMPT.format(query=query, context=context)
    try:
        raw    = ask_gemini(prompt)
        data   = parse_json_robust(raw)
        score  = round(float(data.get("score", 0.0)), 3)
        reason = data.get("reason", "")
        return score, reason
    except Exception as e:
        raw_text = raw if 'raw' in locals() else 'None'
        print(f"  [Context Relevance Error] {e} | Raw: {raw_text!r}")
        return None, ""


# ======================== METRIC 4: ANSWER RELEVANCE ========================
ANSWER_RELEVANCE_PROMPT = """Bạn là chuyên gia đánh giá hệ thống RAG.

Nhiệm vụ: Đánh giá câu trả lời có đúng trọng tâm câu hỏi không.

QUERY: {query}
ANSWER: {answer}

Chỉ trả về JSON, không giải thích thêm:
{{
  "score": <float 0.0-1.0>,
  "reason": "lý do ngắn gọn"
}}

Thang điểm:
- 1.0: Trả lời thẳng vào câu hỏi, đầy đủ, không lan man
- 0.7: Đúng hướng nhưng có phần dư thừa hoặc thiếu nhỏ
- 0.4: Chỉ trả lời một phần, lạc sang vấn đề khác
- 0.0: Từ chối trả lời hoặc hoàn toàn lạc đề"""


def compute_answer_relevance(query: str, answer: str) -> tuple:
    if not answer.strip():
        return 0.0, "Không có câu trả lời"
    prompt = ANSWER_RELEVANCE_PROMPT.format(query=query, answer=answer)
    try:
        raw    = ask_gemini(prompt)
        data   = parse_json_robust(raw)
        score  = round(float(data.get("score", 0.0)), 3)
        reason = data.get("reason", "")
        return score, reason
    except Exception as e:
        raw_text = raw if 'raw' in locals() else 'None'
        print(f"  [Answer Relevance Error] {e} | Raw: {raw_text!r}")
        return None, ""


# ======================== METRIC 5: ANSWER CORRECTNESS ========================
ANSWER_CORRECTNESS_PROMPT = """Bạn là chuyên gia đánh giá hệ thống RAG về pháp luật Việt Nam.

Nhiệm vụ: So sánh câu trả lời thực tế với câu trả lời chuẩn (ground truth).

QUERY: {query}

EXPECTED ANSWER (câu trả lời chuẩn):
{expected_answer}

ACTUAL ANSWER (câu trả lời cần đánh giá):
{actual_answer}

Hướng dẫn:
1. Liệt kê các điểm chính trong expected answer.
2. Với mỗi điểm, kiểm tra actual answer có đề cập đúng không.
3. Tính điểm dựa trên tỉ lệ điểm chính được trả lời đúng.

Chỉ trả về JSON, không giải thích thêm:
{{
  "key_points": [
    {{"point": "nội dung điểm chính", "covered": true}},
    {{"point": "nội dung điểm chính", "covered": false}}
  ],
  "score": <float 0.0-1.0>,
  "reason": "lý do ngắn gọn"
}}

Thang điểm score:
- 1.0: Trả lời đúng và đầy đủ tất cả điểm chính
- 0.7: Đúng phần lớn, thiếu vài chi tiết nhỏ
- 0.4: Đúng một phần, sai hoặc thiếu nhiều điểm quan trọng
- 0.0: Sai hoàn toàn hoặc không liên quan đến expected answer"""


def compute_answer_correctness(query: str, expected_answer: str, actual_answer: str) -> tuple:
    """
    So sánh actual_answer với expected_answer (ground truth).
    Trả về (score 0.0~1.0, danh sách key_points, lý do)
    """
    if not expected_answer.strip():
        return None, [], "Không có expected_answer trong dataset"
    if not actual_answer.strip():
        return 0.0, [], "Không có câu trả lời"

    prompt = ANSWER_CORRECTNESS_PROMPT.format(
        query=query,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
    )
    try:
        raw        = ask_gemini(prompt)
        data       = parse_json_robust(raw)
        score      = round(float(data.get("score", 0.0)), 3)
        key_points = data.get("key_points", [])
        reason     = data.get("reason", "")
        return score, key_points, reason
    except Exception as e:
        raw_text = raw if 'raw' in locals() else 'Rỗng hoặc Timeout'
        print(f"  [LỖI PARSE] Không thể đọc JSON. Lỗi Python: {e}")
        print(f"  [RAW TEXT TỪ API TRẢ VỀ]:\n>>>\n{raw_text}\n<<<")
        return None, [], ""  # Hoặc return None, [], "" đối với hàm correctness


# ======================== MAIN ========================
def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Không tìm thấy {DATASET_PATH}.")
        return

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    progress     = load_progress()
    already_done = set(progress.keys())

    clear_cache()

    print("=" * 70)
    print("BẮT ĐẦU CHẠY ĐÁNH GIÁ CHATBOT (5 METRICS)")
    print("=" * 70)

    order = 0
    for item in dataset:
        if not item.get('is_in_scope', True):
            continue

        query            = item['query']
        expected_article = item.get('expected_article', '')
        expected_answer  = item.get('expected_answer', '')  # ← ground truth
        order           += 1

        if query in already_done:
            print(f"\n[{order}] Bỏ qua (đã có): {query[:60]}")
            continue

        print(f"\n[{order}] Q: {query[:80]}")

        # --- Gọi RAG ---
        result        = rag_service.answer(query, [])
        actual_answer = result.get('answer', '')
        sources       = result.get('sources', [])
        full_context  = build_full_context(sources)

        # --- Metric 1: Hit Rate (không tốn API call) ---
        hit = compute_hit_rate(expected_article, sources)

        # --- Metric 2: Faithfulness ---
        f_score, claims = compute_faithfulness(full_context, actual_answer)
        unsupported     = [c['claim'] for c in claims if not c.get('supported')]

        # --- Metric 3: Context Relevance ---
        ctx_score, ctx_reason = compute_context_relevance(query, full_context)

        # --- Metric 4: Answer Relevance ---
        ans_score, ans_reason = compute_answer_relevance(query, actual_answer)

        # --- Metric 5: Answer Correctness (so với ground truth) ---
        cor_score, key_points, cor_reason = compute_answer_correctness(
            query, expected_answer, actual_answer
        )
        missed_points = [p['point'] for p in key_points if not p.get('covered')]

        print(f"  Hit Rate           : {' HIT' if hit else ' MISS'}")
        if f_score is not None:
            print(f"  Faithfulness       : {f_score:.3f}  {' hallucinated: ' + str(len(unsupported)) + ' claims' if unsupported else '✅'}")
        else:
            print(f"  Faithfulness       : — (Lỗi parse)")
            
        print(f"  Context Relevance  : {ctx_score:.3f}  | {ctx_reason[:65]}" if ctx_score is not None else "  Context Relevance  : — (Lỗi parse)")
        print(f"  Answer Relevance   : {ans_score:.3f}  | {ans_reason[:65]}" if ans_score is not None else "  Answer Relevance   : — (Lỗi parse)")
        
        if cor_score is not None:
            print(f"  Answer Correctness : {cor_score:.3f}  | {cor_reason[:65]}")
        else:
            print(f"  Answer Correctness : —  (không có expected_answer / lỗi parse)")

        # --- Lưu tiến trình ngay sau mỗi câu ---
        progress[query] = {
            "order"              : order,
            "expected_article"   : expected_article,
            "expected_answer"    : expected_answer,
            "actual_answer"      : actual_answer,
            "hit"                : hit,
            "faithfulness"       : f_score,
            "hallucinated_claims": unsupported,
            "faithfulness_claims": claims,
            "context_relevance"  : ctx_score,
            "context_reason"     : ctx_reason,
            "answer_relevance"   : ans_score,
            "answer_reason"      : ans_reason,
            "answer_correctness" : cor_score,
            "missed_points"      : missed_points,
            "key_points"         : key_points,
            "correctness_reason" : cor_reason,
        }
        save_progress(progress)

# ======================== BÁO CÁO ========================
    results = list(progress.values())
    if not results:
        print("\nKhông có câu hỏi hợp lệ nào.")
        return
 
    n = len(results)
 
    # [FIX] Hit rate chỉ tính những câu có ground truth (hit != None)
    hit_results = [r for r in results if r['hit'] is not None]
    avg_hit     = sum(r['hit'] for r in hit_results) / len(hit_results) if hit_results else 0.0
 
    f_results   = [r for r in results if r['faithfulness'] is not None]
    avg_f       = sum(r['faithfulness'] for r in f_results) / len(f_results) if f_results else 0.0
 
    ctx_results = [r for r in results if r['context_relevance'] is not None]
    avg_ctx     = sum(r['context_relevance'] for r in ctx_results) / len(ctx_results) if ctx_results else 0.0
 
    ans_results = [r for r in results if r['answer_relevance'] is not None]
    avg_ans     = sum(r['answer_relevance'] for r in ans_results) / len(ans_results) if ans_results else 0.0
 
    cor_results = [r for r in results if r['answer_correctness'] is not None]
    avg_cor     = sum(r['answer_correctness'] for r in cor_results) / len(cor_results) if cor_results else None
 
    failed_hits = [r['order'] for r in hit_results if r['hit'] == 0.0]
    low_faith   = [r for r in results if r['hallucinated_claims']]
    low_correct = [r for r in cor_results if r['answer_correctness'] < 0.5]
 
    print("\n\n" + "=" * 58)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 58)
    print(f"Tổng câu hỏi (in-scope)    : {n}")
    print(f"Hit Rate                   : {avg_hit:.1%}  ({int(sum(r['hit'] for r in hit_results))}/{len(hit_results)} câu có ground truth)")
    print(f"Avg Faithfulness           : {avg_f:.3f} / 1.000")
    print(f"Avg Context Relevance      : {avg_ctx:.3f} / 1.000")
    print(f"Avg Answer Relevance       : {avg_ans:.3f} / 1.000")
    if avg_cor is not None:
        print(f"Avg Answer Correctness     : {avg_cor:.3f} / 1.000  ({len(cor_results)} câu có ground truth)")
    else:
        print(f"Avg Answer Correctness     : —  (không có expected_answer)")
 
    if failed_hits:
        print(f"\n Miss retrieval ({len(failed_hits)} câu — thứ tự: {failed_hits[:10]})")
 
    if low_faith:
        print(f"\n Hallucination ({len(low_faith)} câu):")
        for r in low_faith[:3]:
            q = next(q for q, v in progress.items() if v is r)
            print(f"   Q: {q[:65]}")
            for h in r['hallucinated_claims'][:2]:
                print(f"      → Bịa: {h[:75]}")
 
    if low_correct:
        print(f"\n Trả lời sai/thiếu nhiều ({len(low_correct)} câu, score < 0.5):")
        for r in low_correct[:3]:
            q = next(q for q, v in progress.items() if v is r)
            print(f"   [{r['answer_correctness']:.2f}] Q: {q[:65]}")
            for p in r['missed_points'][:2]:
                print(f"      → Thiếu: {p[:75]}")
 
    print("=" * 58)
 
    save_report({
        "total"                  : n,
        "hit_rate_sample_size"   : len(hit_results),
        "avg_hit_rate"           : round(avg_hit, 4),
        "avg_faithfulness"       : round(avg_f, 4),
        "avg_context_relevance"  : round(avg_ctx, 4),
        "avg_answer_relevance"   : round(avg_ans, 4),
        "avg_answer_correctness" : round(avg_cor, 4) if avg_cor is not None else None,
        "correctness_sample_size": len(cor_results),
        "failed_hit_orders"      : failed_hits,
        "details"                : results,
    })
 
 
if __name__ == "__main__":
    main()
 