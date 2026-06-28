"""
Document chunking service.

Splits normalised text into sentence-aware overlapping chunks
using NLTK tokenisation with a regex fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List

import nltk

from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Split normalised text into semantic chunks respecting sentence boundaries."""

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        max_chunk_size: int = 500,
        min_chunk_size: int = 250,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split(self, text: str) -> List[Dict[str, Any]]:
        """Return a list of chunk dicts with keys: id, text, word_count, sentences."""
        if not text.strip():
            return []

        all_sentences: List[Dict[str, Any]] = []
        for p_idx, paragraph in enumerate(text.split("\n")):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            try:
                sents = nltk.sent_tokenize(paragraph)
            except Exception as e:
                logger.warning(f"NLTK tokenise failed, using regex fallback: {e}")
                sents = [s.strip() for s in paragraph.split(". ") if s.strip()]
            for s in sents:
                s = s.strip()
                if s:
                    all_sentences.append(
                        {"text": s, "paragraph_idx": p_idx, "word_count": len(s.split())}
                    )

        if not all_sentences:
            return []

        chunks: List[Dict[str, Any]] = []
        current: List[Dict[str, Any]] = []
        current_wc = 0
        chunk_idx = 0

        for sentence in all_sentences:
            swc = sentence["word_count"]
            if current_wc + swc > self.max_chunk_size and current_wc >= self.min_chunk_size:
                chunks.append(self._build_chunk(chunk_idx, current, current_wc))
                chunk_idx += 1
                # Backtrack for overlap
                overlap_sents: List[Dict[str, Any]] = []
                overlap_wc = 0
                for s_back in reversed(current):
                    if overlap_wc + s_back["word_count"] > self.overlap * 1.5:
                        break
                    overlap_sents.insert(0, s_back)
                    overlap_wc += s_back["word_count"]
                current = overlap_sents
                current_wc = overlap_wc

            current.append(sentence)
            current_wc += swc

        if current:
            if chunks and current_wc < 100:
                last = chunks[-1]
                if last["word_count"] + current_wc <= self.max_chunk_size:
                    last["text"] += " " + " ".join(s["text"] for s in current)
                    last["word_count"] += current_wc
                    last["sentences"].extend(s["text"] for s in current)
                else:
                    chunks.append(self._build_chunk(chunk_idx, current, current_wc))
            else:
                chunks.append(self._build_chunk(chunk_idx, current, current_wc))

        logger.info(f"Split document into {len(chunks)} chunks.")
        return chunks

    @staticmethod
    def _build_chunk(
        idx: int, sentences: List[Dict[str, Any]], wc: int
    ) -> Dict[str, Any]:
        return {
            "id": f"chunk_{idx}",
            "text": " ".join(s["text"] for s in sentences),
            "word_count": wc,
            "sentences": [s["text"] for s in sentences],
        }
