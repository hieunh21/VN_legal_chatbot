from google import genai
from config.settings import settings

# Khởi tạo client dùng SDK mới của Google
client = genai.Client(api_key=settings.gemini_api_key)

def generate_multi_queries(query: str) -> list[str]:
    """
    Sử dụng Gemini để sinh ra 3 biến thể của câu hỏi gốc.
    Các biến thể nhắm tới việc paraphrase, sử dụng từ đồng nghĩa, 
    hoặc chuẩn hoá thành thuật ngữ pháp lý.
    """
    if not settings.gemini_api_key:
        # Fallback nếu chưa có key, trả về đúng câu query gốc để không bị sập pipeline
        return [query]

    prompt = (
        "Bạn là một chuyên gia ngôn ngữ pháp lý Việt Nam. "
        "Dựa vào câu hỏi dưới đây của người dùng, hãy sinh ra đúng 3 biến thể nội dung "
        "dùng từ vựng chuẩn pháp luật, từ đồng nghĩa hoặc câu hỏi phụ (sub-questions) "
        "để tối ưu hoá cho hệ thống tìm kiếm Vector (Vector Search).\n\n"
        f"Câu hỏi gốc: {query}\n\n"
        "YÊU CẦU OUTUT:\n"
        "- Trả về chính xác 3 câu trên 3 dòng riêng biệt.\n"
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
        return variants[:3] if variants else [query]
    except Exception as e:
        print(f"Lỗi khi gọi Gemini Multi-Query: {e}")
        return [query]  # Luôn có phương án an toàn
