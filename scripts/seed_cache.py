import json
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from app.services import embedding_service
from app.vector_store import qdrant_store

DATA_PATH = "data/faq_seed.json"

def seed_cache():
    print(f"[*] Đang đọc dữ liệu FAQ từ {DATA_PATH}...")
    
    if not os.path.exists(DATA_PATH):
        print(f"[!] Lỗi: Không tìm thấy file {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    print(f"[*] Tìm thấy {len(faq_data)} câu hỏi. Bắt đầu đẩy lên Cache...")
    
    # Đảm bảo Collection legal_cache đã tồn tại trên Qdrant
    qdrant_store.ensure_collection()

    success_count = 0
    for item in faq_data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        legal_basis = item.get("legal_basis", [])

        if not question or not answer:
            continue
            
        print(f"  -> Đang Embed: {question[:50]}...")
        # 1. Nhúng nội dung câu hỏi thành Vector
        query_vector = embedding_service.embed(question)
        
        # Cái ngụy trang cho tham số sources truyền vào upsert_cache để hàm khỏi báo lỗi
        mock_sources = [{"article": lb, "chapter": ""} for lb in legal_basis]
        
        # 2. Upload thẳng lên Cache
        qdrant_store.upsert_cache(
            query_vector=query_vector,
            answer=answer,
            query_text=question,
            sources=mock_sources,
            cache_type="seeded" # Đánh dấu là được nạp bằng tay chứ không phải AI tự sinh
        )
        success_count += 1
        
    print(f"[+] Hoàn tất! Đã nạp thành công {success_count} / {len(faq_data)} bản ghi vào Qdrant Cache.")

if __name__ == "__main__":
    seed_cache()
