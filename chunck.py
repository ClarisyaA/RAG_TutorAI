import re
import os
import spacy
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

# Setup folder paths
MATERI_FOLDER = "ekstrak"
CHUNKS_FOLDER = "chunks"

# Create chunks folder if not exists
Path(CHUNKS_FOLDER).mkdir(parents=True, exist_ok=True)

class SemanticChunker:
    def __init__(self):
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_md")
            print("🧠 Loaded SpaCy full model (en_core_web_md)")
        except OSError:
            try:
                self.nlp = spacy.blank("en")
                self.nlp.add_pipe("sentencizer")
                print("🧠 Loaded SpaCy basic model (sentencizer)")
            except Exception as e:
                print(f"⚠️ SpaCy initialization failed: {e}")
                self.nlp = None

        self.has_spacy = self.nlp is not None

    def detect_document_structure(self, text):
        structure = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if re.match(r"^#{1,6}\s+", line):
                level = len(re.match(r"^#+", line).group())
                structure.append(
                    {"type": "heading", "level": level, "text": line, "line_num": i}
                )
            elif re.match(r"^\d+\.", line):
                structure.append({"type": "numbered_item", "text": line, "line_num": i})
            elif line.isupper() and len(line) > 10:
                structure.append({"type": "section_title", "text": line, "line_num": i})
            else:
                structure.append({"type": "paragraph", "text": line, "line_num": i})

        return structure

    def extract_concepts_enhanced(self, chunk_text):
        concepts = []
        regex_concepts = self.extract_concepts_simple(chunk_text)
        concepts.extend(regex_concepts)

        if self.has_spacy:
            spacy_concepts = self._extract_spacy_concepts(chunk_text)
            concepts.extend(spacy_concepts)

        unique_concepts = list(
            set([concept.lower().strip() for concept in concepts if concept.strip()])
        )
        return unique_concepts

    def _extract_spacy_concepts(self, text):
        concepts = []
        doc = self.nlp(text)

        pm_entity_types = ["ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "PERSON"]
        for ent in doc.ents:
            if ent.label_ in pm_entity_types:
                concepts.append(ent.text)

        pm_keywords = [
            "project", "management", "planning", "schedule", "wbs",
            "task", "milestone", "deliverable", "scope", "resource", "budget"
        ]

        for chunk in doc.noun_chunks:
            chunk_lower = chunk.text.lower()
            if any(keyword in chunk_lower for keyword in pm_keywords):
                concepts.append(chunk.text)

        for token in doc:
            if (
                token.pos_ in ["NOUN", "PROPN"]
                and not token.is_stop
                and not token.is_punct
                and len(token.text) > 3
            ):
                if any(pm_word in token.text.lower() for pm_word in pm_keywords):
                    concepts.append(token.lemma_)

        return concepts

    def _smart_sentence_split(self, text):
        if self.has_spacy:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            sentences = re.split(r"[.!?]+", text)
            return [s.strip() for s in sentences if s.strip()]

    def semantic_chunk(self, text, chunk_size=500, chunk_overlap=100):
        if self.has_spacy:
            print("🧠 Enhanced semantic chunking with SpaCy")
        else:
            print("🧠 Basic semantic chunking")

        structure = self.detect_document_structure(text)
        sections = []
        current_section = []
        current_heading = None

        for item in structure:
            if item["type"] in ["heading", "section_title"]:
                if current_section:
                    sections.append({
                        "heading": current_heading,
                        "content": "\n".join([s["text"] for s in current_section]),
                    })
                current_heading = item["text"]
                current_section = []
            else:
                current_section.append(item)

        if current_section:
            sections.append({
                "heading": current_heading,
                "content": "\n".join([s["text"] for s in current_section]),
            })

        enhanced_chunks = []

        if self.has_spacy:
            for section in sections:
                sentences = self._smart_sentence_split(section["content"])
                current_chunk_sentences = []
                current_chunk_size = 0

                for sentence in sentences:
                    sentence_size = len(sentence)
                    if current_chunk_size + sentence_size > chunk_size and current_chunk_sentences:
                        chunk_text = " ".join(current_chunk_sentences)
                        enhanced_chunk = self._create_enhanced_chunk(chunk_text, section["heading"])
                        enhanced_chunks.append(enhanced_chunk)
                        overlap_sentences = current_chunk_sentences[-1:] if chunk_overlap > 0 else []
                        current_chunk_sentences = overlap_sentences + [sentence]
                        current_chunk_size = sum(len(s) for s in current_chunk_sentences)
                    else:
                        current_chunk_sentences.append(sentence)
                        current_chunk_size += sentence_size

                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    enhanced_chunk = self._create_enhanced_chunk(chunk_text, section["heading"])
                    enhanced_chunks.append(enhanced_chunk)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""],
            )
            for section in sections:
                section_chunks = splitter.split_text(section["content"])
                for chunk in section_chunks:
                    enhanced_chunk = self._create_enhanced_chunk(chunk, section["heading"])
                    enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def extract_concepts_simple(self, chunk_text):
        concept_patterns = [
            r"work\s+breakdown\s+structure\w*", r"wbs\w*", r"project\s+management\w*",
            r"project\s+planning\w*", r"scope\s+management\w*", r"milestone\w*",
            r"deliverable\w*", r"stakeholder\w*", r"risk\s+management\w*", r"schedule\w*",
            r"resource\w*", r"budget\w*", r"quality\s+assurance\w*", r"communication\s+plan\w*",
            r"decomposition\w*", r"hierarchy\w*", r"task\s+breakdown\w*", r"project\s+lifecycle\w*",
            r"methodology\w*", r"framework\w*", r"agile\w*", r"waterfall\w*",
            r"gantt\s+chart\w*", r"critical\s+path\w*", r"cost\s+estimation\w*"
        ]

        concepts = []
        text_lower = chunk_text.lower()
        for pattern in concept_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            concepts.extend(matches)

        return list(set([concept.strip() for concept in concepts if concept.strip()]))

    def _create_enhanced_chunk(self, chunk_text, section_heading):
        concepts = self.extract_concepts_enhanced(chunk_text)
        metadata = {
            "chunk_type": "spacy_enhanced" if self.has_spacy else "semantic",
            "has_concepts": len(concepts) > 0,
            "concept_count": len(concepts),
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split()),
        }

        if self.has_spacy:
            doc = self.nlp(chunk_text)
            metadata.update({
                "sentence_count": len(list(doc.sents)),
                "entity_count": len(doc.ents),
                "has_entities": len(doc.ents) > 0,
            })

        return {
            "text": chunk_text,
            "section_heading": section_heading if section_heading else "Unknown Section",
            "concepts": concepts,
            "metadata": metadata,
        }

def process_files():
    print("🚀 Starting Enhanced Chunking Process")
    print(f"📁 Source folder: {MATERI_FOLDER}")
    print(f"📁 Destination folder: {CHUNKS_FOLDER}")
    
    # Get all text files in materi folder
    text_files = [f for f in os.listdir(MATERI_FOLDER) if f.endswith(".txt")]
    
    if not text_files:
        print("❌ No text files found in materi folder!")
        return
    
    print(f"📄 Found {len(text_files)} text files to process")
    chunker = SemanticChunker()
    
    for filename in text_files:
        file_path = os.path.join(MATERI_FOLDER, filename)
        output_path = os.path.join(CHUNKS_FOLDER, f"{os.path.splitext(filename)[0]}_chunks.jsonl")
        
        print(f"\n🔍 Processing: {filename}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            print(f"   ✅ Loaded text: {len(raw_text)} characters")
            
            enhanced_chunks = chunker.semantic_chunk(raw_text)
            print(f"   ✅ Created {len(enhanced_chunks)} enhanced chunks")
            
            # Save as JSONL (JSON Lines)
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in enhanced_chunks:
                    json_line = json.dumps(chunk, ensure_ascii=False)
                    f.write(json_line + "\n")
            print(f"   💾 Saved to: {output_path}")
            
            # Show preview
            print("\n   📊 Preview (first 2 chunks):")
            print("   " + "=" * 50)
            for i, chunk in enumerate(enhanced_chunks[:2]):
                print(f"   [Chunk {i+1}]")
                print(f"   Section: {chunk['section_heading'][:50]}...")
                print(f"   Concepts: {', '.join(chunk['concepts'][:3])}...")
                print(f"   Text: {chunk['text'][:100]}...")
                print("   " + "-" * 50)
                
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {str(e)}")
    
    print("\n✅ All files processed successfully!")

if __name__ == "__main__":
    import json
    
    print("=" * 70)
    print("🚀 ENHANCED CHUNKING - PROCESS FOLDER (JSONL OUTPUT)")
    print("=" * 70)
    
    process_files()
    
    print("\n" + "=" * 70)
    print("✅ Program completed!")
    print("=" * 70)