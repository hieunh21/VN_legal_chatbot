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
HF_MODEL_ID=Qwen/Qwen2.5-72B-Instruct

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

## 4. Luồng Hoạt Động (Tiêu chuẩn Advanced RAG 2.0)
Nếu muốn nghiên cứu mã nguồn, hãy xem ở `app/services/rag_service.py` với các chốt chặn xử lý dữ liệu:

1. **Semantic Caching (Bộ Nhớ Đệm Ngữ Nghĩa):** Trước khi chạy truy vấn, hệ thống kiểm tra kho Cache nhúng bằng vector ngữ nghĩa. Giúp trả kết quả cực nhanh (~0.01 giây), bỏ qua toàn bộ RAG lõi tốn CPU.
2. **Hybrid Search (Tìm Kiếm Lai Cao Cấp):** Kết hợp đồng thời 2 công nghệ: Vector Dense (BGE-M3) để hiểu ngữ cảnh sâu + Vector Sparse (BM25 - FastEmbed) để ráp chính xác từ khoá khó. Ghép lại bằng thuật toán nạp chồng RRF (Reciprocal Rank Fusion).
3. **Selective Query Rewriting (Viết lại Câu Hỏi Có Chọn Lọc):**
   - **Fast Path (Đường trơn):** Tìm nhấp nháy 1 lần. Nếu Reranker chấm điểm > 0.8 (Tính theo Sigmoid Model), lập tức trả về (Bỏ qua Gemini nhọc nhằn).
   - **Heavy Path (Đường cày):** Chỉ mở Gemini để sinh 3 câu đồng nghĩa (Multi-Query) khi độ chính xác đường trơn quá thấp. Tránh lãng phí Token vô tội vạ.
4. **Selective Reranking (Lọc Xếp Hạng Thông Minh):** Vắt kiệt lại 15 mảnh tin cậy nhất từ bộ lọc Hybrid RRF, và chỉ đưa 15 mẫu này cho Cross-Encoder chấm lại điểm chuẩn xác tuyệt đối, tiết kiệm tối đa RAM so với Rerank toàn bộ.
5. **Generative Evaluation Data (Dữ Liệu Đánh Giá):** Đã xây dựng sẵn một bộ đánh giá kiểm thử tiêu chuẩn **Golden Set 90 Items** (`evaluate/golden_set_luat_chatbot_90_items.json`). Bộ dữ liệu thiết kế có đối chiếu Căn cứ, hỗ trợ phục vụ cho việc chấm điểm Pipeline Retrieval sau này.
6. **Realtime SSE Streaming:** Đẩy chữ liên tục ra API để Streamlit in từng ký tự cho User xem mà không bị nghẽn (Zero Perceived Latency).

---

## 5. Troubleshooting (Bắt Lỗi Thường Gặp)

- Lỗi **`503 UNAVAILABLE`** ở ngầm Terminal: Model Gemini đang nghẽn server, hệ thống lúc này đã kích hoạt "Fallback", chỉ nhúng 1 câu hỏi gốc và mọi thứ bảo đảm không hề bị crash!
- Xảy ra `ModuleNotFoundError`: Chắc chắn bạn chưa Update lại file `requirements.txt`. Gõ lại `pip install -rc requirements.txt`.
- Dấu gạch chéo thư mục bị sai: Ở Windows, luôn xài raw string `r"C:\\..."` hoặc `/` để nạp dữ liệu từ PDF.