# VN Legal RAG Chatbot

Chatbot pháp luật Việt Nam dùng kiến trúc RAG:
- FastAPI cho backend API
- Streamlit cho giao diện chat
- PostgreSQL lưu session và lịch sử hội thoại
- Qdrant lưu vector embeddings
- Sentence Transformers + CrossEncoder cho retrieve/rerank
- Hugging Face Inference API cho sinh câu trả lời

## 1. Cấu trúc dự án

```
VN-legal-chatbot/
├─ app/
│  ├─ db/
│  ├─ models/
│  ├─ routers/
│  ├─ services/
│  └─ vector_store/
├─ config/
├─ data/
├─ data_processing/
├─ scripts/
├─ docker-compose.yml
├─ requirements.txt
└─ streamlit_app.py
```

## 2. Yêu cầu

- Python 3.10+
- Docker Desktop (để chạy PostgreSQL, Qdrant, pgAdmin)

## 3. Cài đặt

### Windows PowerShell

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## 4. Cấu hình môi trường

Tạo file `.env` ở root (tuỳ chỉnh khi cần):

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/legal_chatbot
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=legal_chunks

HF_API_TOKEN=your_hf_token
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct

EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

Lưu ý: token Hugging Face cần quyền gọi Inference API cho model bạn chọn.

## 5. Chạy hạ tầng phụ trợ

```powershell
docker compose up -d
```

Services mặc định:
- Qdrant: `http://localhost:6333`
- PostgreSQL: `localhost:5432`
- pgAdmin: `http://localhost:5050`

## 6. Khởi tạo database

```powershell
python -m scripts.init_db
```

## 7. Chuẩn bị và index dữ liệu luật

1. Tạo file JSON chunks từ notebook trong `data_processing/`.
2. Đảm bảo các file output JSON tồn tại đúng đường dẫn mà `scripts/index_data.py` đang đọc.
3. Chạy index:

```powershell
python -m scripts.index_data
```

## 8. Chạy backend FastAPI

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: `http://localhost:8000/docs`

## 9. Chạy giao diện Streamlit

Mở terminal khác, cùng môi trường ảo:

```powershell
streamlit run streamlit_app.py
```

UI mặc định: `http://localhost:8501`

## 10. API chính

- `POST /sessions`: tạo phiên chat mới
- `GET /sessions/{session_id}`: lấy thông tin phiên + history
- `POST /chat/{session_id}`: gửi câu hỏi và nhận câu trả lời + nguồn tham khảo

## 11. Luồng hoạt động RAG

1. Embed câu hỏi người dùng
2. Search top-k trong Qdrant
3. Rerank candidates bằng CrossEncoder
4. Ghép context luật liên quan
5. Gọi LLM (Hugging Face) để sinh câu trả lời
6. Lưu user/assistant messages vào PostgreSQL

## 12. Troubleshooting nhanh

- `ModuleNotFoundError`: chạy lại `pip install -r requirements.txt`
- `invalid escape sequence` với đường dẫn Windows:
	- Dùng dấu `/`, ví dụ `E:/VN-legal-chatbot/...`
	- Hoặc dùng raw string `r"E:\\VN-legal-chatbot\\..."`
- `FileNotFoundError` khi xử lý PDF: kiểm tra chính xác tên file và khoảng trắng thừa trong path

## 13. Lệnh thường dùng

```powershell
# Start infra
docker compose up -d

# Init DB
python -m scripts.init_db

# Index data
python -m scripts.index_data

# Run API
uvicorn app.main:app --reload

# Run UI
streamlit run streamlit_app.py
```