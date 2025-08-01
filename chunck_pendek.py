import os
import re
import json
import spacy
from typing import List, Tuple, Dict
from tabulate import tabulate

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠ Jalankan: python -m spacy download en_core_web_sm")
    nlp = None


class FinalSemanticChunker:
    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size
        self.table_pattern = re.compile(r'\|.?\|.?\|')

    def detect_structure(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        structure = {'headers': [], 'lists': [], 'tables': []}
        lines = text.split('\n')
        pos = 0

        for i, line in enumerate(lines):
            start = pos
            end = start + len(line)

            if re.match(r'^\s*(#{1,6}|\d+\.)\s+', line):
                structure['headers'].append((start, end))
            elif re.match(r'^\s*[-+•]\s+', line) or re.match(r'^\s\d+\.\s+', line):
                structure['lists'].append((start, end))
            elif self.table_pattern.search(line):
                table_start = start
                table_end = end
                j = i + 1
                while j < len(lines) and (self.table_pattern.search(lines[j]) or lines[j].strip() == ''):
                    table_end = pos + len(lines[j])
                    j += 1
                structure['tables'].append((table_start, table_end))
            pos += len(line) + 1
        return structure

    def format_tables(self, text: str, table_spans: List[Tuple[int, int]]) -> Dict[int, str]:
        formatted = {}
        for idx, (start, end) in enumerate(table_spans):
            raw = text[start:end]
            lines = raw.strip().split('\n')
            rows = [line.split('|') for line in lines if '|' in line]
            cleaned = [[cell.strip() for cell in row if cell.strip()] for row in rows if row]
            if cleaned:
                tabular = tabulate(cleaned[1:], headers=cleaned[0], tablefmt='grid')
                formatted[start] = f"\n[TABEL {idx+1}]\n{tabular}\n[/TABEL {idx+1}]\n"
            else:
                formatted[start] = f"\n[TABEL {idx+1}]\n{raw}\n[/TABEL {idx+1}]\n"
        return formatted

    def replace_tables(self, text: str, tables: Dict[int, str], spans: List[Tuple[int, int]]) -> str:
        new_text = text
        offset = 0
        for (start, end) in spans:
            if start in tables:
                formatted = tables[start]
                new_text = new_text[:start+offset] + formatted + new_text[end+offset:]
                offset += len(formatted) - (end - start)
        return new_text

    def smart_chunking(self, text: str) -> List[str]:
        structure = self.detect_structure(text)
        tables = self.format_tables(text, structure['tables'])
        clean_text = self.replace_tables(text, tables, structure['tables'])

        # Hapus newline (\n) yang tidak berada dalam [TABEL]
        if '[TABEL' not in clean_text:
            clean_text = clean_text.replace('\n', ' ')
        else:
            parts = re.split(r'(\[TABEL.?\].?\[/TABEL.*?\])', clean_text, flags=re.DOTALL)
            cleaned_parts = [p.replace('\n', ' ') if not p.startswith('[TABEL') else p for p in parts]
            clean_text = ''.join(cleaned_parts)

        if nlp:
            doc = nlp(clean_text)
            all_sentences = [sent.text.strip() for sent in doc.sents]
        else:
            all_sentences = re.split(r'(?<=[.!?])\s+', clean_text)

        used = set()
        chunks = []
        current_chunk = []
        current_len = 0

        for sent in all_sentences:
            if not sent or sent in used:
                continue
            sent_len = len(sent)

            # Jika tabel → langsung simpan sebagai chunk sendiri
            if '[TABEL' in sent and '[/TABEL' in sent:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                        used.update(current_chunk)
                    current_chunk = []
                    current_len = 0
                chunks.append(sent)
                used.add(sent)
                continue

            # Jika melebihi batas
            if current_len + sent_len > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                    used.update(current_chunk)
                current_chunk = [sent]
                current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len

        # Sisa terakhir
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks



def save_chunks(chunks: List[Dict], out_file: str):
    with open(out_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            chunk['char_count'] = len(chunk['content'])
            chunk['word_count'] = len(chunk['content'].split())
            chunk['has_table'] = '[TABEL' in chunk['content']
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def analyze_quality(chunks: List[Dict]):
    total = len(chunks)
    print(f"\n📊 Analisis Chunking:")
    print(f"Total: {total}")
    print(f"Rata-rata panjang: {sum(len(c['content']) for c in chunks) / total:.1f}")
    print(f"Chunks dengan tabel: {sum('[TABEL' in c['content'] for c in chunks)}")
    print(f"Chunks < 100 char: {sum(len(c['content']) < 100 for c in chunks)}")
    print(f"Chunks > 800 char: {sum(len(c['content']) > 800 for c in chunks)}")


def load_texts(folder: str):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".txt") and not file.endswith(".id.txt") and not file.endswith(".en.txt"):
            path = os.path.join(folder, file)
            with open(path, 'r', encoding='utf-8') as f:
                teks = f.read()
                if teks.strip():
                    docs.append({'source': file.replace(".txt", ""), 'text': teks})
    return docs


if __name__ == "__main__":
    folder = "ekstrak"
    output_file = "chunck_wbs_pendek.jsonl"
    chunker = FinalSemanticChunker(chunk_size=500)

    all_docs = load_texts(folder)
    all_chunks = []

    for doc in all_docs:
        print(f"📄 Memproses: {doc['source']}")
        chunks = chunker.smart_chunking(doc['text'])
        doc_chunks = [{
            "source": doc['source'],
            "chunk_id": f"{doc['source']}chunk{i+1}",
            "content": c
        } for i, c in enumerate(chunks)]
        all_chunks.extend(doc_chunks)
        print(f"   ✅ {len(doc_chunks)} chunks")

        analyze_quality(doc_chunks)

    save_chunks(all_chunks, output_file)
    print(f"\n✅ Semua chunks disimpan ke {output_file}")