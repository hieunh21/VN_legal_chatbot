# VN Legal RAG Chatbot (Advanced Edition)

Đây là hệ thống Chatbot pháp luật Việt Nam được xây dựng đạt chuẩn **Advanced RAG**, tối ưu riêng cho môi trường tư vấn pháp lý với khả năng đáp ứng thời gian thực:
- **FastAPI**: Lõi Backend API hỗ trợ Real-time Streaming (Server-Sent Events).
- **Streamlit**: Giao diện người dùng tích hợp hiệu ứng Streaming và chấm điểm "🎯 Độ phù hợp" bằng Toán Sigmoid.
- **PostgreSQL**: Lưu trữ Lịch sử hội thoại vĩnh viễn (thiết kế chống sập DB khi ngắt kết nối).
- **Qdrant**: Vector Database tốc độ cao lưu trữ Embeddings của các bộ luật.
- **Google GenAI (Gemini)**: Đảm nhận tầng Pre-retrieval để khôi phục ngữ cảnh (Query Rewriting dựa trên lịch sử) và mở rộng câu hỏi (Multi-Query Expansion).
- **Sentence Transformers (Qdrant/BAAI)**: Chạy Batch Embedding và Cross-Encoder Reranking diện rộng.
- **Hugging Face**: Kết nối `Qwen2.5-72B-Instruct` làm LLM tư duy, lý luận và trả lời cuối cùng (temperature được tinh chỉnh khắt khe để tránh ảo giác).

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
1. Đảm bảo các file `.json` điều luật đã nằm trong file map ở `scripts/index_data.py` băng cách chạy 2 notebooks: data_process_luat_giao_thong.ipynb va data_process_luat_tieu_dung.ipynb.
2. Nạp schema bảng Chat vào PostgresSQL: `python -m scripts.init_db`
3. Vector (Embed) đẩy vào Qdrant: `python -m scripts.index_data`
4. `python -m scripts.seed_cache`

### Bước 5: Kích hoạt Hệ thống Hai Lớp
- **Bật Backend (Terminal 1) (nhớ `activate` venv):**
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

1. **Semantic Caching Chuyên Sâu:** Bộ nhớ đệm tĩnh, hiện chỉ được cấp nguồn từ dữ liệu FAQ chuẩn xác (`data/faq_seed.json`), bị khoá tự động "Save Cache" chiều từ Chatbot. Điều này tránh tình trạng "rác cache" do các câu chat ngẫu nhiên của người dùng. Trúng cache sẽ trả kết quả tức thì (~0.01s).
2. **History-Aware Query Rewriting (Khôi phục ngữ cảnh):** Sử dụng hệ thống **Gemini** (Google) để trích xuất 4 tin nhắn gần nhất và tự động viết thông dịch các câu hỏi phụ cộc lốc/thiếu chủ ngữ của người dùng thành một câu hỏi độc lập chuẩn pháp lý.
3. **Hybrid Search (Tìm Kiếm Lai Cao Cấp):** Kết hợp đồng thời 2 bản đồ: Vector Dense (BGE-M3) để hiểu ngữ cảnh sâu + Vector Sparse (BM25 - FastEmbed) để ráp độ tin cậy từ vựng.
4. **Selective Multi-Query & Routing (Định tuyến chọn lọc):** 
   - **Fast Path:** Nếu BGE Reranker chấm điểm Vector tìm được > 80%, dùng luôn tài liệu đó để sinh câu trả lời.
   - **Heavy Path:** Nếu điểm < 80%, nhường sân cho Gemini sinh thêm 2 biến thể câu hỏi (Multi-Query Expansion) để rà quét đáy Qdrant với mẻ lưới lớn.
5. **Selective Reranking (Lọc Xếp Hạng Thông Minh):** Vắt kiệt lại 8 mảnh tin cậy nhất từ bộ lọc Hybrid Search, và chỉ đưa 8 mẫu này cho Cross-Encoder chấm lại điểm chuẩn xác tuyệt đối, tiết kiệm tối đa RAM so với Rerank toàn bộ.
6. **Generative Evaluation Data (Dữ Liệu Đánh Giá):** Đã xây dựng sẵn một bộ đánh giá kiểm thử tiêu chuẩn **Golden Datset** (`data_processing/golden_dataset.json`). Bộ dữ liệu thiết kế có đối chiếu Căn cứ, hỗ trợ phục vụ cho việc chấm điểm Pipeline Retrieval sau này.
7. **Realtime SSE Streaming:** Đẩy chữ liên tục ra API để Streamlit in từng ký tự cho User xem (Zero Perceived Latency).

---

## 5. Troubleshooting (Bắt Lỗi Thường Gặp)

- Lỗi **`503 UNAVAILABLE`** ở ngầm Terminal: Model LLM đang nghẽn server chờ reset hoặc chưa đổi qua Model khác trong file `.env`.
- Xảy ra `ModuleNotFoundError`: Chắc chắn bạn chưa Update lại file `requirements.txt`.
- Dấu gạch chéo thư mục bị sai: Ở Windows, luôn xài raw string `r"C:\\..."` hoặc `/` để nạp dữ liệu từ PDF.

---

## 6. Hướng dẫn Chạy Thử nghiệm & Đánh giá (Evaluation)

Dự án tích hợp sẵn module benchmark để chạy bài test tự động trên bộ **Golden Dataset** (file `data_processing/golden_dataset.json`). Quá trình này sẽ gọi bot trả lời 100 câu hỏi và lưu lại toàn bộ tiến độ.

1. Hãy chắc chắn Backend (FastAPI) và các Server DB Docker (Qdrant, Postgres) **đang chạy**.
2. Mở một terminal mới (đã kích hoạt `. venv/Scripts/activate`):
   ```powershell
   python scripts/evaluate_bot.py
   ```
3. Xem kết quả đánh giá (Thời gian phản hồi, số token) sẽ được lưu lại dưới dạng file JSON tại thư mục `data_processing/` (lưu tự động vào `eval_progress.json` mỗi 5 mẫu tránh crash). Khi hoàn tất toàn bộ sẽ sinh ra file `eval_report.json`. Mọi thay đổi về code ở tương lai đều có thể chạy cái lại cái report này để so sánh độ trễ!