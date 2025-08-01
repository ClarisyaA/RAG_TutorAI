import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]  # Dari paragraf → kalimat → kata
    )
    chunks = splitter.split_text(text)
    return chunks

def load_all_texts_from_folder(folder_path):
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt") and not file.endswith(".id.txt") and not file.endswith(".en.txt"):
            base_name = os.path.splitext(file)[0]
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                if text.strip():
                    texts.append({"source": base_name, "text": text})
    return texts

def save_chunks(chunks, out_file="chunks_wbs.jsonl"):
    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    folder_path = "ekstrak"
    all_docs = load_all_texts_from_folder(folder_path)

    all_chunks = []

    for doc in all_docs:
        print(f"🔍 Memproses file: {doc['source']}")
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": f"{doc['source']}_chunk_{i+1}",
                "content": chunk.strip()
            })

    save_chunks(all_chunks)
    print(f"✅ Total {len(all_chunks)} chunks disimpan ke 'chunks_wbs.jsonl'")
