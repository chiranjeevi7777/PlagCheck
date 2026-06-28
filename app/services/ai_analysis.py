"""
AI Writing Pattern Analysis Service.

Analyses text chunks for writing patterns statistically associated
with AI-generated content. Results are probabilistic estimates —
NOT proof of AI authorship.

Concurrency: chunk analysis runs in a ThreadPoolExecutor (Groq SDK is sync).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from app.llm.groq_client import GroqAPIClient
from app.schemas.analysis import AIChunkResult, classify_ai_probability
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an expert linguistic analyst specialising in writing style assessment.
Analyse the provided text for patterns commonly associated with AI-generated content.

Evaluate:
1. Predictability of wording and phrasing
2. Sentence structure repetition
3. Sentence length variation (or lack thereof)
4. Vocabulary diversity vs predictability
5. Formality level and tonal consistency
6. Naturalness vs mechanical flow
7. Generic textbook-style explanations
8. Overused polished transitions (Furthermore, Moreover, In conclusion)

IMPORTANT: You are detecting writing PATTERNS, not determining authorship.
Return ONLY valid JSON in exactly this format (no markdown, no extra text):
{
  "chunk_id": "string",
  "ai_probability": int (0-100),
  "confidence": int (0-100),
  "classification": "Very Low AI Writing Pattern"|"Low AI Writing Pattern"|"Moderate AI Writing Pattern"|"High AI Writing Pattern"|"Very High AI Writing Pattern",
  "reason": "string (2-3 sentences)",
  "features": ["string", ...]
}"""

_ALLOWED_CLASSES = {
    "very low ai writing pattern", "low ai writing pattern",
    "moderate ai writing pattern", "high ai writing pattern",
    "very high ai writing pattern",
}


class AIWritingPatternService:
    """Orchestrates parallel AI-pattern analysis across all document chunks."""

    def __init__(self, groq_client: GroqAPIClient, max_workers: int = 3) -> None:
        self._groq = groq_client
        self._max_workers = max_workers

    def analyze_document(
        self,
        chunks: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Analyse all chunks concurrently and return aggregated report."""
        N = len(chunks)
        if N == 0:
            return self._empty_report()

        logger.info(f"AI pattern analysis: {N} chunks, max_workers={self._max_workers}")
        results_map: Dict[str, AIChunkResult] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_chunk = {
                pool.submit(self._analyze_chunk, chunk): chunk for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                cid = chunk["id"]
                completed += 1
                try:
                    results_map[cid] = future.result()
                except Exception as e:
                    logger.error(f"Future error for chunk {cid}: {e}")
                    results_map[cid] = self._fallback(cid, str(e))
                if progress_callback:
                    progress_callback(completed, N, f"AI analysis {completed}/{N}")

        ordered: List[Dict[str, Any]] = []
        for chunk in chunks:
            cid = chunk["id"]
            res = results_map.get(cid, self._fallback(cid, "missing"))
            ordered.append({
                "chunk_id": cid,
                "chunk_text": chunk["text"],
                "word_count": chunk.get("word_count", len(chunk["text"].split())),
                "ai_probability": res.ai_probability,
                "confidence": res.confidence,
                "classification": res.classification,
                "reason": res.reason,
                "features": res.features,
            })
        return self._aggregate(ordered)

    # ── Private ───────────────────────────────────────────────────────────────

    def _analyze_chunk(self, chunk: Dict[str, Any]) -> AIChunkResult:
        cid = chunk["id"]
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"chunk_id: {cid}\n\n--- PASSAGE ---\n{chunk['text']}"},
        ]
        try:
            data = self._groq.call_parsed(messages)
            cls_lower = str(data.get("classification", "")).lower()
            matched = classify_ai_probability(int(data.get("ai_probability", 0)))
            for ac in _ALLOWED_CLASSES:
                if ac in cls_lower:
                    matched = " ".join(w.capitalize() for w in ac.split())
                    break
            data["classification"] = matched
            data["chunk_id"] = cid
            result = AIChunkResult(**data)
            logger.info(f"AI: {cid} -> {result.ai_probability}% ({result.classification})")
            return result
        except Exception as e:
            logger.error(f"AI analysis error for {cid}: {e}")
            return self._fallback(cid, str(e))

    @staticmethod
    def _fallback(chunk_id: str, reason: str) -> AIChunkResult:
        return AIChunkResult(
            chunk_id=chunk_id, ai_probability=0, confidence=0,
            classification="Very Low AI Writing Pattern",
            reason=f"Analysis failed: {reason}", features=[],
        )

    @staticmethod
    def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        N = len(results)
        if N == 0:
            return AIWritingPatternService._empty_report()
        probs = [r["ai_probability"] for r in results]
        confs = [r["confidence"] for r in results]
        overall = round(sum(probs) / N)
        freq: Dict[str, int] = {}
        for r in results:
            for f in r.get("features", []):
                freq[f] = freq.get(f, 0) + 1
        top_features = sorted(freq.items(), key=lambda x: -x[1])[:10]
        return {
            "overall_ai_score": overall,
            "average_confidence": round(sum(confs) / N),
            "overall_classification": classify_ai_probability(overall),
            "total_chunks": N,
            "very_high_probability_chunks": sum(1 for p in probs if p > 80),
            "high_probability_chunks": sum(1 for p in probs if 61 <= p <= 80),
            "moderate_probability_chunks": sum(1 for p in probs if 41 <= p <= 60),
            "low_probability_chunks": sum(1 for p in probs if 21 <= p <= 40),
            "very_low_probability_chunks": sum(1 for p in probs if p <= 20),
            "highest_chunk": max(results, key=lambda r: r["ai_probability"]),
            "lowest_chunk": min(results, key=lambda r: r["ai_probability"]),
            "top_features": [{"feature": f, "count": c} for f, c in top_features],
            "chunk_results": results,
            "disclaimer": (
                "This analysis estimates writing patterns commonly associated with "
                "AI-generated content. It should not be interpreted as definitive proof "
                "that any passage was written by AI."
            ),
        }

    @staticmethod
    def _empty_report() -> Dict[str, Any]:
        return {
            "overall_ai_score": 0, "average_confidence": 0,
            "overall_classification": "Very Low AI Writing Pattern",
            "total_chunks": 0, "very_high_probability_chunks": 0,
            "high_probability_chunks": 0, "moderate_probability_chunks": 0,
            "low_probability_chunks": 0, "very_low_probability_chunks": 0,
            "highest_chunk": {}, "lowest_chunk": {}, "top_features": [],
            "chunk_results": [],
            "disclaimer": (
                "This analysis estimates writing patterns commonly associated with "
                "AI-generated content. It should not be interpreted as definitive proof "
                "that any passage was written by AI."
            ),
        }
