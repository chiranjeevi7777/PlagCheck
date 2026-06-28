import nltk
from typing import List, Dict, Any
from utils import logger

class DocumentChunker:
    """Splits normalized text into semantic chunks respecting sentence boundaries and paragraph structures."""

    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.target_chunk_size = chunk_size  # target size in words
        self.overlap_size = overlap          # overlap size in words
        self.max_chunk_size = 500            # hard cap in words
        self.min_chunk_size = 250            # soft floor in words

    def split_into_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits text into chunks of 300-500 words with ~50-word overlap.
        Preserves sentence boundaries by grouping sentences together.
        """
        if not text.strip():
            return []

        # Split into paragraphs based on newlines
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        
        # Tokenize sentences per paragraph to preserve boundaries
        all_sentences = []
        for p_idx, paragraph in enumerate(paragraphs):
            try:
                sentences = nltk.sent_tokenize(paragraph)
            except Exception as e:
                # Fallback to simple split if NLTK fails
                logger.warning(f"NLTK sentence tokenization failed, using regex fallback: {e}")
                sentences = [s.strip() for s in paragraph.split(". ") if s.strip()]
                # Add back periods
                for idx, s in enumerate(sentences):
                    if not s.endswith(".") and idx < len(sentences) - 1:
                        sentences[idx] += "."

            for s in sentences:
                s_stripped = s.strip()
                if s_stripped:
                    words = s_stripped.split()
                    all_sentences.append({
                        "text": s_stripped,
                        "paragraph_idx": p_idx,
                        "word_count": len(words)
                    })

        if not all_sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        chunk_idx = 0

        i = 0
        while i < len(all_sentences):
            sentence = all_sentences[i]
            s_word_count = sentence["word_count"]

            # If adding this sentence exceeds the max chunk size (500 words),
            # and we already have a reasonable chunk, we commit the current chunk.
            if current_word_count + s_word_count > self.max_chunk_size and current_word_count >= self.min_chunk_size:
                # Commit current chunk
                chunk_text = " ".join([s["text"] for s in current_chunk_sentences])
                chunks.append({
                    "id": f"chunk_{chunk_idx}",
                    "text": chunk_text,
                    "word_count": current_word_count,
                    "sentences": [s["text"] for s in current_chunk_sentences]
                })
                chunk_idx += 1

                # Calculate backtracking for overlap (target ~50 words)
                overlap_sentences = []
                overlap_words = 0
                # Go backwards from the end of current_chunk_sentences
                for s_back in reversed(current_chunk_sentences):
                    if overlap_words + s_back["word_count"] > self.overlap_size * 1.5:
                        break
                    overlap_sentences.insert(0, s_back)
                    overlap_words += s_back["word_count"]

                # Start new chunk with the overlap sentences
                current_chunk_sentences = overlap_sentences.copy()
                current_word_count = overlap_words

            # Add current sentence
            current_chunk_sentences.append(sentence)
            current_word_count += s_word_count
            i += 1

        # Add remaining sentences if any
        if current_chunk_sentences:
            chunk_text = " ".join([s["text"] for s in current_chunk_sentences])
            # If the final chunk is very small and we have previous chunks,
            # we can merge it into the last chunk if it doesn't exceed the max size
            if len(chunks) > 0 and current_word_count < 100:
                last_chunk = chunks[-1]
                if last_chunk["word_count"] + current_word_count <= self.max_chunk_size:
                    last_chunk["text"] += " " + chunk_text
                    last_chunk["word_count"] += current_word_count
                    last_chunk["sentences"].extend([s["text"] for s in current_chunk_sentences])
                else:
                    chunks.append({
                        "id": f"chunk_{chunk_idx}",
                        "text": chunk_text,
                        "word_count": current_word_count,
                        "sentences": [s["text"] for s in current_chunk_sentences]
                    })
            else:
                chunks.append({
                    "id": f"chunk_{chunk_idx}",
                    "text": chunk_text,
                    "word_count": current_word_count,
                    "sentences": [s["text"] for s in current_chunk_sentences]
                })

        logger.info(f"Split document into {len(chunks)} chunks.")
        return chunks
