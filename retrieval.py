import faiss
import pickle
import json
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator

# === Load FAISS index ===
index = faiss.read_index("wbs_faiss.index")

# === Load metadata chunks ===
with open("wbs_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# === Load model embedding ===
model = SentenceTransformer("intfloat/e5-base-v2")

# === Deteksi dan translasi pertanyaan ===
def translate_if_needed(text, target_lang="en"):
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    if lang == "id" and target_lang == "en":
        translated = GoogleTranslator(source='id', target='en').translate(text)
        return translated, "id"
    else:
        return text, lang

# === Retrieve top-k context ===
def retrieve_context(user_question, top_k=3):
    # Translate jika perlu
    question_translated, detected_lang = translate_if_needed(user_question)

    # Buat embedding dari pertanyaan
    question_embedding = model.encode([question_translated], convert_to_numpy=True)

    # FAISS search
    distances, indices = index.search(question_embedding, top_k)

    # Ambil chunk hasil
    results = [metadata[i] for i in indices[0]]
    return results, detected_lang
