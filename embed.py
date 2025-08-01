import faiss
import json
import os
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === Load model embedding ===
model = SentenceTransformer("intfloat/e5-base-v2")  # Bisa diganti dengan 'bge-small-en-v1.5'

# === Load chunks dari JSONL ===
def load_chunks(file_path):
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["content"].strip():
                chunks.append(data)
    return chunks

# === Embedding dan indexing ===
def embed_and_index(chunks, index_path="wbs_faiss.index", metadata_path="wbs_metadata.pkl"):
    texts = [chunk["content"] for chunk in chunks]

    print(f"🔄 Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Simpan index
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index disimpan ke '{index_path}'")

    # Simpan metadata (dengan source & isi)
    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "content": chunk["content"]
        }
        for chunk in chunks
    ]
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ Metadata {len(metadata)} chunks disimpan ke '{metadata_path}'")

if __name__ == "__main__":
    chunks = load_chunks("chunks_wbs.jsonl")
    embed_and_index(chunks)
