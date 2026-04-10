# VN Legal RAG Chatbot (Advanced Edition)

Đây là hệ thống Chatbot pháp luật Việt Nam được xây dựng đạt chuẩn **Advanced RAG**, tối ưu riêng cho môi trường tư vấn pháp lý với khả năng đáp ứng thời gian thực:
- **FastAPI**: Lõi Backend API hỗ trợ Real-time Streaming (Server-Sent Events).
- **Streamlit**: Giao diện người dùng tích hợp hiệu ứng Streaming và chấm điểm "🎯 Độ phù hợp" bằng Toán Sigmoid.
- **PostgreSQL**: Lưu trữ Lịch sử hội thoại vĩnh viễn (thiết kế chống sập DB khi ngắt kết nối).
- **Qdrant**: Vector Database tốc độ cao lưu trữ Embeddings của các bộ luật.
- **Google GenAI (Gemini)**: Đảm nhận tầng Pre-retrieval để viết lại truy vấn (Multi-Query Expansion).
- **Sentence Transformers (Qdrant/BAAI)**: Chạy Batch Embedding và Cross-Encoder Reranking diện rộng.
- **Hugging Face**: Kết nối `Qwen2.5-7B-Instruct` làm LLM tư duy, lý luận và trả lời.

## 1. Cấu trúc dự án

```text
VN-legal-chatbot/
├─ app/
│  ├─ db/
│  ├─ models/
│  ├─ routers/       (Chứa các endpoint REST & SSE Stream)
│  ├─ services/      (Lõi RAG, Gemini, LLM, Embedding, Reranker)
│  └─ vector_store/  (Giao tiếp Qdrant)
├─ config/           (Pydantic settings)
├─ data/
├─ data_processing/
├─ scripts/
├─ docker-compose.yml
├─ requirements.txt
└─ streamlit_app.py
```

## 2. Tiêu chuẩn Hệ thống

- Lập trình bằng hệ điều hành Win/Linux/MacOS với **Python 3.10+**.
- Dùng **Docker Desktop** để chạy nhanh bộ sậu (PostgreSQL, Qdrant, pgAdmin).

## 3. Hướng dẫn Cài đặt & Chạy (Clone Project)

### Bước 1: Khởi tạo môi trường (Dành cho Windows PowerShell)
```powershell
# Tạo môi trường ảo
python -m venv venv
./venv/Scripts/Activate.ps1

# Nạp thư viện (bao gồm cả google-genai mới nhất)
pip install -r requirements.txt
```

### Bước 2: Thiết lập Biến Môi Trường (API Keys)
Tạo file `.env` ở tại thư mục gốc của dự án `VN-legal-chatbot/.env` (coppy từ nội dung dưới, sau đó điền Key của bạn):

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/legal_chatbot

# Vector Store
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=legal_chunks

# LLM Sinh văn bản (Qwen)
HF_API_TOKEN=your_hugginface_token
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct

# Tác nhân Pre-retrieval Multi-Query (Mở rộng câu hỏi)
GEMINI_API_KEY=your_gemini_api_key

# Tác nhân Search
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### Bước 3: Đánh thức hệ thống nền tảng
Chạy các server lưu trữ bằng Docker:
```powershell
docker compose up -d
```
*Các địa chỉ mặc định: Qdrant `http://localhost:6333` | Postgres `localhost:5432` | pgAdmin `http://localhost:5050`*

### Bước 4: Chuẩn bị Dữ liệu Luật (Làm 1 lần)
1. Đảm bảo các file `.json` điều luật đã nằm trong file map ở `scripts/index_data.py`.
2. Nạp schema bảng Chat vào PostgresSQL: `python -m scripts.init_db`
3. Vector (Embed) đẩy vào Qdrant: `python -m scripts.index_data`

### Bước 5: Kích hoạt Hệ thống Hai Lớp
- **Bật Backend (Terminal 1):**
  ```powershell
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  ```
  *(Xem API ngầm tại Swagger: `http://localhost:8000/docs`)*

- **Bật Frontend Giao diện (Terminal 2):** Mở 1 tab terminal khác (nhớ `activate` venv).
  ```powershell
  streamlit run streamlit_app.py
  ```
  *(Truy cập giao diện: `http://localhost:8501`)*

---

## 4. Luồng Hoạt Động (Tiêu chuẩn Advanced RAG)
Nếu muốn nghiên cứu mã nguồn, hãy xem ở `app/services/rag_service.py` với 7 trạm kiểm duyệt dữ liệu đỉnh cao:

1. **Pre-Retrieval (Multi-Query Expansion):** Hệ thống chọc vào API Gemini sinh 3 biến thể pháp lý của câu hỏi gốc để bọc lót lỗi dùng từ tiếng lóng của User.
2. **Batch Embedding:** Code nhúng gộp cục bộ (4 câu liền 1 lúc) để không ép xung phần cứng CPU.
3. **Parallel Search:** Bắn lên Qdrant thu về 4 luồng ứng viên.
4. **Deduplication (Trị trùng lặp):** Lọc mảng bằng hàm Hashing tự nhiên (ngăn ngừa các biến thể gom về cùng 1 luật).
5. **Cross-Encoder Reranking:** Từ 40 điều luật rác, mô hình lọc lõi chỉ chọn đúng Top 5 điều luật xuất sắc nhất.
6. **Bottom-Bound Prompting:** Các tài liệu Luật tìm được sẽ bị đính khống chế vào mẩu tin nhắn mới nhất nhằm trị bệnh Context Bleeding (LLM ngáo khi trộn History).
7. **Realtime SSE Streaming:** Đẩy chữ liên tục ra API để Streamlit in từng ký tự cho User xem mà không bị nghẽn (Zero Perceived Latency).

---

## 5. Troubleshooting (Bắt Lỗi Thường Gặp)

- Lỗi **`503 UNAVAILABLE`** ở ngầm Terminal: Model Gemini đang nghẽn server, hệ thống lúc này đã kích hoạt "Fallback", chỉ nhúng 1 câu hỏi gốc và mọi thứ bảo đảm không hề bị crash!
- Xảy ra `ModuleNotFoundError`: Chắc chắn bạn chưa Update lại file `requirements.txt`. Gõ lại `pip install -rc requirements.txt`.
- Dấu gạch chéo thư mục bị sai: Ở Windows, luôn xài raw string `r"C:\\..."` hoặc `/` để nạp dữ liệu từ PDF.