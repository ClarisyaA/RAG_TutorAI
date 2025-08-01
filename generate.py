import requests
import textwrap
from retrieval import retrieve_context, translate_if_needed
from deep_translator import GoogleTranslator

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"  # Ganti sesuai model Ollama kamu, bisa juga 'mistral'

# === Prompt Builder ===
def build_prompt(chunks, question):
    prompt_lines = [
        "Berikut adalah beberapa referensi dari jurnal atau materi yang relevan.",
        "Jawablah pertanyaan hanya berdasarkan informasi berikut.",
        "Jika jawaban tidak ditemukan, jawab: \"Maaf, informasi tersebut tidak tersedia dalam materi.\"",
        "",
        "=== REFERENSI ==="
    ]

    for i, chunk in enumerate(chunks, 1):
        source = chunk["source"]
        content = chunk["content"].strip().replace("\n", " ")
        prompt_lines.append(f"[Sumber: {source}] Dikutip: \"{content}\"")

    prompt_lines += [
        "",
        "=== PERTANYAAN ===",
        question,
        "",
        "=== JAWABAN ==="
    ]

    return "\n".join(prompt_lines)

# === Kirim ke Ollama ===
def ask_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })

    if response.status_code == 200:
        return response.json()["response"]
    else:
        print("❌ Gagal meminta ke Ollama:", response.text)
        return None

# === Main Interaction ===
if __name__ == "__main__":
    question = input("❓ Masukkan pertanyaan: ").strip()

    if not question:
        print("⚠️ Pertanyaan kosong.")
        exit()

    # Retrieve context dan bahasa input
    chunks, input_lang = retrieve_context(question, top_k=3)

    # Translate pertanyaan jika perlu
    question_translated, _ = translate_if_needed(question)

    # Bangun prompt
    prompt = build_prompt(chunks, question_translated)

    print("\n🧠 Prompt yang dikirim ke LLM:\n")
    print(textwrap.indent(prompt, prefix="    "))

    # Dapatkan jawaban dari Ollama
    print("\n🤖 Jawaban dari LLM:\n")
    answer = ask_ollama(prompt)

    # Translate kembali jika input asli Indonesia
    if input_lang == "id":
        answer = GoogleTranslator(source='en', target='id').translate(answer)

    print(textwrap.fill(answer, width=90))
