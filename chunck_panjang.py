import os
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from tabulate import tabulate

# Load spaCy model for sentence boundary detection
try:
    nlp = spacy.load("en_core_web_md")
    print("✅ spaCy model loaded successfully")
except IOError:
    print("⚠️ spaCy model not installed. Installing fallback sentence splitter...")
    nlp = None


class ImprovedSemanticChunker:
    """
    Improved semantic text chunker that prevents mid-sentence cuts and duplications.
    """
    
    def __init__(self, 
                 target_chunk_size: int = 500,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 800,
                 semantic_threshold: float = 0.3,
                 overlap_ratio: float = 0.1):
        """
        Initialize the chunker with configuration parameters.
        
        Args:
            target_chunk_size: Target size for each chunk in characters
            min_chunk_size: Minimum acceptable chunk size
            max_chunk_size: Maximum acceptable chunk size
            semantic_threshold: Threshold for semantic similarity (0-1)
            overlap_ratio: Ratio of overlap between chunks (0-0.3)
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.semantic_threshold = semantic_threshold
        self.overlap_ratio = min(overlap_ratio, 0.3)  # Cap at 30%
        
        # Enhanced sentence boundary patterns
        self.sentence_endings = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|'  # Period/!/? followed by space and capital
            r'(?<=[.!?])\s*\n+\s*(?=[A-Z])|'  # Period/!/? followed by newline and capital
            r'(?<=\.)\s*\n\s*\n+\s*(?=[A-Z])'  # Period followed by paragraph break
        )
        
        # Common abbreviations that don't end sentences
        self.abbreviations = {
            'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Sr.', 'Jr.',
            'etc.', 'i.e.', 'e.g.', 'vs.', 'Inc.', 'Corp.', 'Ltd.', 'Co.',
            'Fig.', 'Vol.', 'No.', 'pp.', 'p.',
            'Ph.D.', 'B.A.', 'M.A.', 'M.S.', 'B.S.', 'M.D.',
            'a.m.', 'p.m.', 'A.M.', 'P.M.',
            'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Jun.', 'Jul.', 
            'Aug.', 'Sep.', 'Sept.', 'Oct.', 'Nov.', 'Dec.'
        }
        
        # Header detection patterns
        self.header_patterns = [
            re.compile(r'^\s*#{1,6}\s+.+$', re.MULTILINE),  # Markdown headers
            re.compile(r'^\s*\d+(\.\d+)*\s+[A-Z].+$', re.MULTILINE),  # Numbered sections
            re.compile(r'^\s*[A-Z][A-Z\s]{2,}$', re.MULTILINE),  # ALL CAPS headers
            re.compile(r'^\s*[A-Z][^.!?]*:?\s*$', re.MULTILINE),  # Title case headers
            re.compile(r'^\s*[A-Z]+\s*$', re.MULTILINE)  # Single word headers
        ]
        
        # Table detection
        self.table_pattern = re.compile(r'^\s*\|.*\|.*\|\s*$', re.MULTILINE)
        
        # List detection
        self.list_pattern = re.compile(r'^\s*[-*+•]\s|^\s*\d+\.\s', re.MULTILINE)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = None

    def extract_sentences(self, text: str) -> List[Dict[str, any]]:
        """
        Extract sentences with precise boundary detection.
        
        Args:
            text: Input text to process
            
        Returns:
            List of sentence dictionaries with metadata
        """
        sentences = []
        
        if nlp:
            # Use spaCy for accurate sentence segmentation
            doc = nlp(text)
            for i, sent in enumerate(doc.sents):
                sentence_text = sent.text.strip()
                if sentence_text and len(sentence_text) > 3:  # Filter very short fragments
                    sentences.append({
                        'text': sentence_text,
                        'start': sent.start_char,
                        'end': sent.end_char,
                        'index': i,
                        'is_header': self._is_header_sentence(sentence_text),
                        'is_list_item': self._is_list_item(sentence_text),
                        'has_numeric': bool(re.search(r'\d+', sentence_text)),
                        'length': len(sentence_text),
                        'word_count': len(sentence_text.split()),
                        'ends_with_period': sentence_text.rstrip().endswith(('.', '!', '?'))
                    })
        else:
            # Fallback to improved regex-based splitting
            sentences = self._advanced_sentence_split(text)
        
        return sentences

    def _advanced_sentence_split(self, text: str) -> List[Dict[str, any]]:
        """Advanced sentence splitting with comprehensive boundary detection."""
        sentences = []
        
        # Normalize text
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Split into potential sentences
        potential_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        current_pos = 0
        for i, sent_text in enumerate(potential_sentences):
            sent_text = sent_text.strip()
            if not sent_text or len(sent_text) < 3:
                current_pos += len(sent_text) + 1
                continue
            
            # Check if it's a real sentence boundary
            if not self._is_abbreviation_ending(sent_text):
                start_pos = text.find(sent_text, current_pos)
                end_pos = start_pos + len(sent_text)
                
                sentences.append({
                    'text': sent_text,
                    'start': start_pos,
                    'end': end_pos,
                    'index': i,
                    'is_header': self._is_header_sentence(sent_text),
                    'is_list_item': self._is_list_item(sent_text),
                    'has_numeric': bool(re.search(r'\d+', sent_text)),
                    'length': len(sent_text),
                    'word_count': len(sent_text.split()),
                    'ends_with_period': sent_text.rstrip().endswith(('.', '!', '?'))
                })
                
                current_pos = end_pos
            else:
                current_pos += len(sent_text) + 1
        
        return sentences

    def _is_abbreviation_ending(self, text: str) -> bool:
        """Check if text ends with an abbreviation."""
        words = text.strip().split()
        if not words:
            return False
        
        last_word = words[-1]
        return last_word in self.abbreviations

    def _is_header_sentence(self, sentence: str) -> bool:
        """Enhanced header detection."""
        sentence = sentence.strip()
        
        # Check against patterns
        for pattern in self.header_patterns:
            if pattern.match(sentence):
                return True
        
        # Additional heuristics
        if (len(sentence.split()) <= 10 and  # Reasonably short
            sentence[0].isupper() and        # Starts with capital
            not sentence.endswith('.') and   # Doesn't end with period
            len(sentence) < 120 and          # Not too long
            ':' not in sentence[-10:]):      # No colon near end
            return True
        
        return False

    def _is_list_item(self, sentence: str) -> bool:
        """Check if sentence is a list item."""
        return bool(self.list_pattern.match(sentence.strip()))

    def calculate_semantic_similarity(self, sentences: List[Dict]) -> np.ndarray:
        """Calculate semantic similarity between sentences."""
        if len(sentences) < 2:
            return np.array([[1.0]])
        
        # Extract text for vectorization
        texts = [sent['text'] for sent in sentences]
        
        try:
            # Initialize vectorizer if needed
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=min(1000, len(texts) * 10),
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    token_pattern=r'\b\w+\b'
                )
            
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
        
        except Exception as e:
            print(f"⚠️ Warning: Similarity calculation failed: {e}")
            return np.eye(len(sentences))

    def find_optimal_chunks(self, sentences: List[Dict]) -> List[Tuple[int, int]]:
        """
        Find optimal chunk boundaries using semantic analysis and size constraints.
        
        Args:
            sentences: List of sentence dictionaries
            
        Returns:
            List of (start_idx, end_idx) tuples for chunks
        """
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [(0, 1)]
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_semantic_similarity(sentences)
        
        chunks = []
        current_start = 0
        current_size = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_size = sentence['length']
            
            # Check if adding this sentence would exceed maximum size
            if current_size + sentence_size > self.max_chunk_size and current_size > 0:
                # Must split before this sentence
                chunks.append((current_start, i))
                current_start = i
                current_size = sentence_size
            
            # Check if we've reached target size
            elif current_size + sentence_size >= self.target_chunk_size and current_size >= self.min_chunk_size:
                # Look for optimal split point
                split_point = self._find_best_split_point(
                    sentences, similarity_matrix, current_start, i + 1
                )
                
                if split_point > current_start:
                    chunks.append((current_start, split_point))
                    current_start = split_point
                    # Recalculate current size
                    current_size = sum(s['length'] for s in sentences[split_point:i+1])
                else:
                    current_size += sentence_size
            else:
                current_size += sentence_size
            
            i += 1
        
        # Add final chunk
        if current_start < len(sentences):
            chunks.append((current_start, len(sentences)))
        
        # Post-process to ensure minimum sizes
        chunks = self._merge_small_chunks(sentences, chunks)
        
        return chunks

    def _find_best_split_point(self, sentences: List[Dict], similarity_matrix: np.ndarray, 
                               start_idx: int, max_idx: int) -> int:
        """Find the best semantic split point within a range."""
        best_split = max_idx - 1
        best_score = float('-inf')
        
        # Define search range
        search_start = max(start_idx + 1, max_idx - 5)
        search_end = min(len(sentences), max_idx + 2)
        
        for candidate in range(search_start, search_end):
            if candidate >= len(sentences) or candidate <= start_idx:
                continue
            
            # Calculate chunk size
            chunk_size = sum(s['length'] for s in sentences[start_idx:candidate])
            
            # Skip if chunk would be too small or too large
            if chunk_size < self.min_chunk_size or chunk_size > self.max_chunk_size:
                continue
            
            # Calculate split score
            score = self._calculate_split_score(
                sentences, similarity_matrix, candidate, start_idx, chunk_size
            )
            
            if score > best_score:
                best_score = score
                best_split = candidate
        
        return best_split

    def _calculate_split_score(self, sentences: List[Dict], similarity_matrix: np.ndarray, 
                               split_idx: int, start_idx: int, chunk_size: int) -> float:
        """Calculate quality score for a potential split point."""
        score = 0.0
        
        # Semantic discontinuity score (higher is better for splitting)
        if split_idx > 0 and split_idx < len(similarity_matrix):
            semantic_gap = 1 - similarity_matrix[split_idx - 1][split_idx]
            score += semantic_gap * 0.4
        
        # Boundary bonus (prefer splits at natural boundaries)
        if split_idx < len(sentences):
            sentence = sentences[split_idx]
            
            # Header boundary
            if sentence.get('is_header', False):
                score += 0.3
            
            # Paragraph boundary (sentence starts with capital after period)
            if split_idx > 0:
                prev_sentence = sentences[split_idx - 1]
                if (prev_sentence.get('ends_with_period', False) and 
                    sentence['text'][0].isupper()):
                    score += 0.2
            
            # List boundary
            if sentence.get('is_list_item', False):
                score += 0.1
        
        # Size preference (prefer chunks closer to target size)
        size_ratio = chunk_size / self.target_chunk_size
        if 0.8 <= size_ratio <= 1.2:
            score += 0.2
        elif 0.6 <= size_ratio <= 1.4:
            score += 0.1
        
        return score

    def _merge_small_chunks(self, sentences: List[Dict], chunks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge chunks that are too small."""
        if not chunks:
            return chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            start, end = chunks[i]
            chunk_size = sum(s['length'] for s in sentences[start:end])
            
            # If chunk is too small, try to merge with next or previous
            if chunk_size < self.min_chunk_size and len(chunks) > 1:
                # Try to merge with next chunk
                if i + 1 < len(chunks):
                    next_start, next_end = chunks[i + 1]
                    combined_size = chunk_size + sum(s['length'] for s in sentences[next_start:next_end])
                    
                    if combined_size <= self.max_chunk_size:
                        merged_chunks.append((start, next_end))
                        i += 2  # Skip next chunk as it's merged
                        continue
                
                # Try to merge with previous chunk
                if merged_chunks and len(merged_chunks) > 0:
                    prev_start, prev_end = merged_chunks[-1]
                    combined_size = chunk_size + sum(s['length'] for s in sentences[prev_start:prev_end])
                    
                    if combined_size <= self.max_chunk_size:
                        merged_chunks[-1] = (prev_start, end)
                        i += 1
                        continue
            
            merged_chunks.append((start, end))
            i += 1
        
        return merged_chunks

    def create_chunks_with_overlap(self, sentences: List[Dict], chunk_boundaries: List[Tuple[int, int]]) -> List[str]:
        """Create final chunks with intelligent overlap to prevent information loss."""
        if not sentences or not chunk_boundaries:
            return []
        
        chunks = []
        processed_sentences: Set[int] = set()  # Track processed sentences to avoid duplication
        
        for i, (start_idx, end_idx) in enumerate(chunk_boundaries):
            # Calculate overlap
            overlap_size = int(self.overlap_ratio * (end_idx - start_idx))
            overlap_size = max(0, min(overlap_size, 3))  # Max 3 sentences overlap
            
            # Determine actual start (with overlap from previous chunk)
            actual_start = start_idx
            if i > 0 and overlap_size > 0:
                actual_start = max(0, start_idx - overlap_size)
                
                # Avoid duplicating sentences we've already fully processed
                while actual_start in processed_sentences and actual_start < start_idx:
                    actual_start += 1
            
            # Create chunk text
            chunk_sentences = sentences[actual_start:end_idx]
            chunk_text = self._assemble_chunk_text(chunk_sentences)
            
            # Only add if chunk meets minimum size and isn't empty
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(chunk_text.strip())
                
                # Mark sentences as processed (excluding overlap region)
                for sent_idx in range(start_idx, end_idx):
                    processed_sentences.add(sent_idx)
        
        return chunks

    def _assemble_chunk_text(self, sentences: List[Dict]) -> str:
        """Assemble sentences into coherent chunk text."""
        if not sentences:
            return ""
        
        text_parts = []
        
        for i, sentence in enumerate(sentences):
            sent_text = sentence['text'].strip()
            
            # Add appropriate spacing
            if i == 0:
                text_parts.append(sent_text)
            else:
                prev_sentence = sentences[i - 1]
                
                # Determine spacing based on context
                if (sentence.get('is_header', False) or 
                    prev_sentence.get('is_header', False) or
                    '\n\n' in prev_sentence['text'] or
                    sentence.get('is_list_item', False)):
                    # Add paragraph break
                    text_parts.append('\n\n' + sent_text)
                elif sentence.get('is_list_item', False) and not prev_sentence.get('is_list_item', False):
                    # Start of list
                    text_parts.append('\n' + sent_text)
                else:
                    # Normal sentence continuation
                    text_parts.append(' ' + sent_text)
        
        # Clean up the assembled text
        chunk_text = ''.join(text_parts)
        chunk_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', chunk_text)  # Normalize paragraph breaks
        chunk_text = re.sub(r' +', ' ', chunk_text)  # Normalize spaces
        
        return chunk_text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """
        Main chunking method that ensures no mid-sentence cuts.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of semantically coherent chunks without mid-sentence cuts
        """
        if not text or len(text.strip()) < 10:
            return [text.strip()] if text.strip() else []
        
        # Handle special case: text smaller than minimum chunk size
        if len(text) < self.min_chunk_size:
            return [text.strip()]
        
        # Extract sentences with metadata
        print("🔍 Extracting sentences...")
        sentences = self.extract_sentences(text)
        
        if not sentences:
            return [text.strip()]
        
        if len(sentences) == 1:
            return [sentences[0]['text']]
        
        print(f"   Found {len(sentences)} sentences")
        
        # Find optimal chunk boundaries
        print("🧠 Finding optimal chunk boundaries...")
        chunk_boundaries = self.find_optimal_chunks(sentences)
        
        print(f"   Identified {len(chunk_boundaries)} chunks")
        
        # Create final chunks with overlap
        print("📦 Creating final chunks...")
        chunks = self.create_chunks_with_overlap(sentences, chunk_boundaries)
        
        # Validate chunks (ensure no mid-sentence cuts)
        validated_chunks = self._validate_chunks(chunks)
        
        print(f"   Generated {len(validated_chunks)} validated chunks")
        
        return validated_chunks

    def _validate_chunks(self, chunks: List[str]) -> List[str]:
        """Validate chunks to ensure no mid-sentence cuts."""
        validated = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # Check if chunk starts and ends at sentence boundaries
            if len(chunk) >= self.min_chunk_size:
                # Ensure chunk ends with proper punctuation or is at document end
                if not chunk[-1] in '.!?':
                    # Try to find the last complete sentence
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    if len(sentences) > 1:
                        # Keep all complete sentences
                        complete_part = '. '.join(sentences[:-1])
                        if len(complete_part) >= self.min_chunk_size:
                            chunk = complete_part + '.'
                
                validated.append(chunk)
            elif validated:
                # Merge small chunk with previous one
                validated[-1] += ' ' + chunk
        
        return validated


def load_documents_from_folder(folder_path: str) -> List[Dict]:
    """Load all text documents from a folder."""
    documents = []
    supported_extensions = ['.txt']
    exclude_patterns = ['.id.', '.en.', '_backup', '_temp', '.jsonl']
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' not found!")
        return documents
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            if not any(pattern in filename.lower() for pattern in exclude_patterns):
                file_path = os.path.join(folder_path, filename)
                base_name = os.path.splitext(filename)[0]
                
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        if content.strip():
                            documents.append({
                                "source": base_name,
                                "text": content,
                                "file_size": len(content),
                                "file_path": file_path
                            })
                            print(f"✅ Loaded: {filename} ({len(content)} chars)")
                except Exception as e:
                    print(f"❌ Error reading {filename}: {e}")
    
    print(f"\n📚 Total {len(documents)} documents loaded successfully")
    return documents


def save_chunks_to_file(chunks: List[Dict], output_file: str = "improved_chunks.jsonl") -> bool:
    """Save chunks to JSONL file with comprehensive metadata."""
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            for chunk in chunks:
                content = chunk['content']
                
                # Add comprehensive metadata
                chunk.update({
                    'char_count': len(content),
                    'word_count': len(content.split()),
                    'sentence_count': len([s for s in re.split(r'[.!?]+', content) if s.strip()]),
                    'has_table': '[TABLE' in content or bool(re.search(r'\|.*\|.*\|', content)),
                    'has_list': bool(re.search(r'^\s*[-*+•]\s|^\s*\d+\.\s', content, re.MULTILINE)),
                    'has_code': '```' in content or 'code' in content.lower(),
                    'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
                    'starts_complete': content.strip()[0].isupper() if content.strip() else True,
                    'ends_complete': content.strip()[-1] in '.!?' if content.strip() else True,
                    'has_headers': bool(re.search(r'^\s*#+\s|^\s*[A-Z][A-Z\s]{2,}$', content, re.MULTILINE))
                })
                
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        
        print(f"💾 Chunks saved successfully to '{output_file}'")
        return True
    except Exception as e:
        print(f"❌ Error saving chunks: {e}")
        return False


def analyze_chunk_quality(chunks: List[Dict]) -> Dict:
    """Comprehensive analysis of chunk quality."""
    if not chunks:
        return {}
    
    char_lengths = [chunk['char_count'] for chunk in chunks]
    word_lengths = [chunk['word_count'] for chunk in chunks]
    
    # Quality metrics
    complete_start = sum(1 for c in chunks if c.get('starts_complete', False))
    complete_end = sum(1 for c in chunks if c.get('ends_complete', False))
    
    stats = {
        'total_chunks': len(chunks),
        'avg_char_length': sum(char_lengths) / len(chunks),
        'avg_word_length': sum(word_lengths) / len(chunks),
        'min_length': min(char_lengths),
        'max_length': max(char_lengths),
        'median_length': sorted(char_lengths)[len(char_lengths)//2],
        
        # Quality indicators
        'complete_sentences': complete_start + complete_end,
        'chunks_with_tables': sum(1 for c in chunks if c.get('has_table', False)),
        'chunks_with_lists': sum(1 for c in chunks if c.get('has_list', False)),
        'chunks_with_code': sum(1 for c in chunks if c.get('has_code', False)),
        'chunks_with_headers': sum(1 for c in chunks if c.get('has_headers', False)),
        
        # Size distribution
        'very_short_chunks': sum(1 for c in chunks if c['char_count'] < 200),
        'short_chunks': sum(1 for c in chunks if 200 <= c['char_count'] < 400),
        'optimal_chunks': sum(1 for c in chunks if 400 <= c['char_count'] <= 700),
        'long_chunks': sum(1 for c in chunks if 700 < c['char_count'] <= 900),
        'very_long_chunks': sum(1 for c in chunks if c['char_count'] > 900),
        
        # Quality score
        'quality_score': (complete_start + complete_end) / (2 * len(chunks)) * 100
    }
    
    print("\n📊 Improved Chunk Quality Analysis:")
    print(f"   📦 Total chunks: {stats['total_chunks']}")
    print(f"   📏 Length stats: avg={stats['avg_char_length']:.0f}, median={stats['median_length']}")
    print(f"   📊 Range: {stats['min_length']} - {stats['max_length']} characters")
    print(f"   🎯 Quality score: {stats['quality_score']:.1f}%")
    print(f"   ✅ Complete boundaries: {stats['complete_sentences']}/{stats['total_chunks']*2}")
    
    print(f"\n   📋 Content types:")
    print(f"      Tables: {stats['chunks_with_tables']}")
    print(f"      Lists: {stats['chunks_with_lists']}")
    print(f"      Headers: {stats['chunks_with_headers']}")
    print(f"      Code: {stats['chunks_with_code']}")
    
    print(f"\n   📏 Size distribution:")
    print(f"      Very short (<200): {stats['very_short_chunks']}")
    print(f"      Short (200-400): {stats['short_chunks']}")
    print(f"      Optimal (400-700): {stats['optimal_chunks']}")
    print(f"      Long (700-900): {stats['long_chunks']}")
    print(f"      Very long (>900): {stats['very_long_chunks']}")
    
    return stats


def main():
    """Main execution function with improved error handling."""
    print("🚀 Starting Improved Semantic Chunking...")
    print("=" * 50)
    
    # Configuration
    FOLDER_PATH = "ekstrak"
    TARGET_CHUNK_SIZE = 500
    MIN_CHUNK_SIZE = 200
    MAX_CHUNK_SIZE = 800
    SEMANTIC_THRESHOLD = 0.3
    OVERLAP_RATIO = 0.1  # 10% overlap
    OUTPUT_FILE = "chunck_wbs_panjang.jsonl"
    
    # Load documents
    print(f"📁 Loading documents from '{FOLDER_PATH}'...")
    documents = load_documents_from_folder(FOLDER_PATH)
    
    if not documents:
        print("❌ No documents found to process!")
        return
    
    # Initialize chunker
    chunker = ImprovedSemanticChunker(
        target_chunk_size=TARGET_CHUNK_SIZE,
        min_chunk_size=MIN_CHUNK_SIZE,
        max_chunk_size=MAX_CHUNK_SIZE,
        semantic_threshold=SEMANTIC_THRESHOLD,
        overlap_ratio=OVERLAP_RATIO
    )
    
    all_chunks = []
    total_input_chars = 0
    total_output_chars = 0
    
    # Process each document
    for doc_idx, doc in enumerate(documents, 1):
        print(f"\n📄 Processing document {doc_idx}/{len(documents)}: {doc['source']}")
        print(f"   📊 Size: {doc['file_size']:,} characters")
        
        total_input_chars += doc['file_size']
        
        try:
            # Chunk the document
            chunks = chunker.chunk_text(doc["text"])
            
            if not chunks:
                print(f"   ⚠️ No chunks generated for {doc['source']}")
                continue
            
            # Create chunk objects with metadata
            doc_chunks = 0
            for i, chunk_content in enumerate(chunks):
                if chunk_content.strip():
                    chunk_data = {
                        "source": doc["source"],
                        "chunk_id": f"{doc['source']}_chunk_{i+1:03d}",
                        "content": chunk_content,
                        "chunk_index": i + 1,
                        "original_file": doc.get("file_path", ""),
                        "document_size": doc['file_size']
                    }
                    all_chunks.append(chunk_data)
                    doc_chunks += 1
                    total_output_chars += len(chunk_content)
            
            print(f"   ✅ Generated {doc_chunks} chunks")
            
            # Show sample of first chunk for verification
            if chunks and len(chunks[0]) > 100:
                sample = chunks[0][:100].replace('\n', ' ')
                print(f"   📝 First chunk preview: {sample}...")
            
        except Exception as e:
            print(f"   ❌ Error processing {doc['source']}: {str(e)}")
            import traceback
            print(f"   🐛 Traceback: {traceback.format_exc()}")
    
    # Save results and analyze
    if all_chunks:
        print(f"\n💾 Saving {len(all_chunks)} chunks...")
        success = save_chunks_to_file(all_chunks, OUTPUT_FILE)
        
        if success:
            print(f"\n🎉 CHUNKING COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"📊 Processing Summary:")
            print(f"   📄 Documents processed: {len(documents)}")
            print(f"   📦 Total chunks created: {len(all_chunks)}")
            print(f"   📈 Input characters: {total_input_chars:,}")
            print(f"   📉 Output characters: {total_output_chars:,}")
            print(f"   🔄 Compression ratio: {(total_output_chars/total_input_chars)*100:.1f}%")
            print(f"   💾 Output file: {OUTPUT_FILE}")
            
            # Perform quality analysis
            analyze_chunk_quality(all_chunks)
            
            # Show sample chunks for manual verification
            print(f"\n🔍 Sample Chunks for Verification:")
            print("-" * 50)
            for i, chunk in enumerate(all_chunks[:3]):
                print(f"\nChunk {i+1} ({chunk['source']}):")
                print(f"Length: {len(chunk['content'])} chars")
                content_preview = chunk['content'][:200].replace('\n', ' ')
                print(f"Preview: {content_preview}...")
                
                # Check for potential issues
                issues = []
                if not chunk['content'].strip()[0].isupper():
                    issues.append("doesn't start with capital")
                if not chunk['content'].strip()[-1] in '.!?':
                    issues.append("doesn't end with punctuation")
                if len(chunk['content']) < MIN_CHUNK_SIZE:
                    issues.append("too short")
                if len(chunk['content']) > MAX_CHUNK_SIZE:
                    issues.append("too long")
                
                if issues:
                    print(f"⚠️  Issues: {', '.join(issues)}")
                else:
                    print("✅ No issues detected")
        else:
            print("❌ Failed to save chunks!")
    else:
        print("❌ No chunks were generated!")


def chunk_text_improved(text: str, 
                       target_chunk_size: int = 500,
                       min_chunk_size: int = 200,
                       max_chunk_size: int = 800,
                       semantic_threshold: float = 0.3,
                       overlap_ratio: float = 0.1) -> List[str]:
    """
    Convenience function for improved semantic text chunking.
    
    Args:
        text: Input text to chunk
        target_chunk_size: Target chunk size in characters
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        semantic_threshold: Semantic similarity threshold
        overlap_ratio: Overlap ratio between chunks
        
    Returns:
        List of semantically coherent chunks without mid-sentence cuts
    """
    chunker = ImprovedSemanticChunker(
        target_chunk_size=target_chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        semantic_threshold=semantic_threshold,
        overlap_ratio=overlap_ratio
    )
    return chunker.chunk_text(text)


def validate_chunks(chunks: List[str]) -> Dict[str, any]:
    """
    Validate chunks for common issues.
    
    Args:
        chunks: List of chunk texts
        
    Returns:
        Validation report
    """
    report = {
        'total_chunks': len(chunks),
        'valid_chunks': 0,
        'issues': [],
        'chunk_issues': []
    }
    
    for i, chunk in enumerate(chunks):
        chunk_issues = []
        
        # Check if chunk is empty or too short
        if not chunk.strip():
            chunk_issues.append("empty")
        elif len(chunk.strip()) < 50:
            chunk_issues.append("too_short")
        
        # Check sentence boundaries
        if chunk.strip():
            first_char = chunk.strip()[0]
            last_char = chunk.strip()[-1]
            
            if not first_char.isupper():
                chunk_issues.append("invalid_start")
            
            if last_char not in '.!?':
                chunk_issues.append("invalid_end")
        
        # Check for potential mid-sentence cuts
        if ' and ' in chunk[-20:] or ' but ' in chunk[-20:] or ' or ' in chunk[-20:]:
            chunk_issues.append("potential_mid_sentence")
        
        # Check for incomplete sentences at the end
        sentences = re.split(r'[.!?]+', chunk)
        if len(sentences) > 1 and sentences[-1].strip() and len(sentences[-1].strip()) > 10:
            chunk_issues.append("incomplete_sentence")
        
        if not chunk_issues:
            report['valid_chunks'] += 1
        else:
            report['chunk_issues'].append({
                'chunk_index': i,
                'issues': chunk_issues,
                'length': len(chunk),
                'preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
            })
    
    # Overall issues
    if report['valid_chunks'] / report['total_chunks'] < 0.8:
        report['issues'].append("low_quality_ratio")
    
    # Calculate quality score
    report['quality_score'] = (report['valid_chunks'] / report['total_chunks']) * 100
    
    return report


def print_validation_report(report: Dict[str, any]):
    """Print a detailed validation report."""
    print(f"\n🔍 Chunk Validation Report:")
    print(f"   📦 Total chunks: {report['total_chunks']}")
    print(f"   ✅ Valid chunks: {report['valid_chunks']}")
    print(f"   🎯 Quality score: {report['quality_score']:.1f}%")
    
    if report['chunk_issues']:
        print(f"\n⚠️  Issues found in {len(report['chunk_issues'])} chunks:")
        for issue in report['chunk_issues'][:5]:  # Show first 5 issues
            print(f"   Chunk {issue['chunk_index']}: {', '.join(issue['issues'])}")
            print(f"      Length: {issue['length']} chars")
            print(f"      Preview: {issue['preview']}")
    else:
        print("\n✅ All chunks passed validation!")


if __name__ == "__main__":
    # Allow running specific functions for testing
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Test mode - chunk a sample text
            sample_text = """
            This is a sample document for testing the chunking algorithm. 
            It contains multiple sentences and paragraphs.
            
            ## Header Example
            
            This section demonstrates how headers are handled. The algorithm should 
            recognize headers as natural boundaries for chunking.
            
            ### Subsection
            
            Here we have more content that should be grouped appropriately. 
            The semantic similarity should help maintain coherent chunks.
            
            - This is a list item
            - Another list item
            - Final list item
            
            And here's the conclusion of our test document. It should end properly
            with complete sentences and proper punctuation.
            """
            
            print("🧪 Testing chunker with sample text...")
            chunks = chunk_text_improved(sample_text, target_chunk_size=300)
            
            print(f"\nGenerated {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
                print(chunk)
            
            # Validate results
            report = validate_chunks(chunks)
            print_validation_report(report)
        
        elif sys.argv[1] == "validate":
            # Validation mode - check existing chunks file
            try:
                chunks_data = []
                with open("improved_semantic_chunks.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        chunk_data = json.loads(line)
                        chunks_data.append(chunk_data['content'])
                
                report = validate_chunks(chunks_data)
                print_validation_report(report)
                
            except FileNotFoundError:
                print("❌ Chunks file not found. Run main chunking first.")
    else:
        # Normal execution
        main()