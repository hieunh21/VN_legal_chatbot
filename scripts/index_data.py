"""
Indexing pipeline: Load cleaned JSON files → Embed with BGE-M3 → Upsert to Qdrant.
Run once: python -m scripts.index_data
"""

import json
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client.models import PointStruct
from app.services.embedding_service import embed_batch
from app.vector_store.qdrant_store import ensure_collection, upsert

DATA_PATHS = [
    r"data_processing\output_qdrant_ready_luat_giao_thong_new.json",
    r"data_processing\output_qdrant_ready_luat_tieu_dung_new.json",
]

BATCH_SIZE = 32


def load_chunks(data_paths):
    all_chunks = []

    for path in data_paths:
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # thêm tên file nguồn nếu chưa có
        for chunk in chunks:
            if "source_file" not in chunk:
                chunk["source_file"] = os.path.basename(path)

        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {path}")

    return all_chunks


def main():
    chunks = load_chunks(DATA_PATHS)
    print(f"Total loaded: {len(chunks)} chunks")

    ensure_collection()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [chunk["context"] for chunk in batch]
        # Bật Sparse=True để lấy Multi-vector
        vectors = embed_batch(texts, return_sparse=True)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                # Đẩy 2 Vector vào Data Model dạng Dict
                vector={"dense": d_vec, "sparse": s_vec},
                payload=chunk,
            )
            for chunk, (d_vec, s_vec) in zip(batch, vectors)
        ]

        upsert(points)
        print(f"Indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print("Done! All chunks indexed to the same Qdrant collection.")


if __name__ == "__main__":
    main()