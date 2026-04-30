from google import genai
from config.settings import settings

# Khởi tạo client dùng SDK mới của Google
client = genai.Client(api_key=settings.gemini_api_key)

def generate_multi_queries(query: str) -> list[str]:
    """
    Sử dụng Gemini để sinh ra 2 biến thể của câu hỏi gốc.
    Các biến thể nhắm tới việc paraphrase, sử dụng từ đồng nghĩa, 
    hoặc chuẩn hoá thành thuật ngữ pháp lý.
    """
    if not settings.gemini_api_key:
        # Fallback nếu chưa có key, trả về đúng câu query gốc để không bị sập pipeline
        return [query]

    prompt = (
        "Bạn là một chuyên gia ngôn ngữ pháp lý Việt Nam. "
        "Dựa vào câu hỏi dưới đây của người dùng, hãy sinh ra đúng 2 biến thể nội dung "
        "dùng từ vựng chuẩn pháp luật, từ đồng nghĩa hoặc câu hỏi phụ (sub-questions) "
        "để tối ưu hoá cho hệ thống tìm kiếm Vector (Vector Search).\n\n"
        f"Câu hỏi gốc: {query}\n\n"
        "YÊU CẦU OUTUT:\n"
        "- Trả về chính xác 2 câu trên 2 dòng riêng biệt.\n"
        "- KHÔNG giải thích, KHÔNG đánh số, KHÔNG gạch đầu dòng.\n"
        "- Mỗi câu không nên quá dài."
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        # Tách kết quả theo dòng và lọc các dòng trống
        variants = [line.strip() for line in response.text.split("\n") if line.strip()]
        
        # Đề phòng Gemini sinh ra nhiều hơn/ít hơn hoặc kèm markdown
        return variants[:2] if variants else [query]
    except Exception as e:
        print(f"Lỗi khi gọi Gemini Multi-Query: {e}")
        return [query]  # Luôn có phương án an toàn

def rewrite_query(query: str, history_text: str) -> str:
    """
    Sử dụng Gemini để viết lại câu hỏi mơ hồ thành câu hỏi độc lập (dựa trên lịch sử chat).
    """
    if not settings.gemini_api_key:
        return query

    prompt = (
        "Nhiệm vụ: Viết lại câu hỏi mới nhất thành câu hỏi độc lập, "
        "đầy đủ ngữ cảnh dựa trên lịch sử hội thoại dưới đây.\n"
        "Chỉ trả về ĐÚNG MỘT câu hỏi mới, KHÔNG giải thích, KHÔNG thêm chữ thừa.\n\n"
        f"--- LỊCH SỬ CHAT GẦN ĐÂY ---\n{history_text}\n\n"
        f"--- CÂU HỎI MỚI NHẤT ---\nUser: {query}\n\n"
        "Câu hỏi viết lại:"
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={"temperature": 0.1} # Giảm sáng tạo để không chế cháo câu hỏi
        )
        return response.text.strip()
    except Exception as e:
        print(f"Lỗi khi gọi Gemini Rewrite: {e}")
        return query
