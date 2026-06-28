"""
AI Writing Pattern Analysis Module
Analyzes text chunks for writing patterns commonly associated with AI-generated content.
Results are probability estimates and NOT definitive proof of AI authorship.
"""

import json
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import groq

from groq_client import GroqPlagiarismClient
from utils import logger


# ─────────────────────────────────────────────
# Pydantic response model
# ─────────────────────────────────────────────

class AIChunkResult(BaseModel):
    chunk_id: str
    ai_probability: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    classification: str
    reason: str
    features: List[str]


# ─────────────────────────────────────────────
# Classification helper
# ─────────────────────────────────────────────

def classify_ai_probability(probability: int) -> str:
    """Map a 0-100 probability to a human-readable classification."""
    if probability <= 20:
        return "Very Low AI Writing Pattern"
    elif probability <= 40:
        return "Low AI Writing Pattern"
    elif probability <= 60:
        return "Moderate AI Writing Pattern"
    elif probability <= 80:
        return "High AI Writing Pattern"
    else:
        return "Very High AI Writing Pattern"


# ─────────────────────────────────────────────
# Groq AI Analysis Client
# ─────────────────────────────────────────────

class AIPatternAnalyzer:
    """
    Analyzes text chunks for AI writing pattern indicators using Groq LLM.
    Results are probabilistic estimates based on writing style features.
    """

    _SYSTEM_PROMPT = (
        "You are an expert linguistic analyst specializing in writing style assessment.\n"
        "Analyze the provided text passage for writing patterns commonly associated with AI-generated content.\n"
        "Evaluate the following dimensions:\n"
        "  1. Predictability of wording and phrasing\n"
        "  2. Sentence structure repetition\n"
        "  3. Sentence length variation (or lack thereof)\n"
        "  4. Vocabulary diversity vs. predictability\n"
        "  5. Formality level and tonal consistency\n"
        "  6. Stylistic consistency across the passage\n"
        "  7. Naturalness vs. mechanical flow\n"
        "  8. Presence of generic, textbook-style explanations\n"
        "  9. Presence of overly polished transitions ('Furthermore', 'Moreover', 'In conclusion', etc.)\n"
        " 10. Likelihood that the passage follows common AI writing patterns\n\n"
        "IMPORTANT: You are detecting writing PATTERNS, not determining authorship definitively.\n"
        "Return ONLY a valid JSON object in exactly this format (no markdown, no extra text):\n"
        "{\n"
        '  "chunk_id": "string (the chunk_id you were given)",\n'
        '  "ai_probability": int (0-100, estimated probability of AI writing patterns),\n'
        '  "confidence": int (0-100, your confidence in this estimate),\n'
        '  "classification": "string (one of: Very Low AI Writing Pattern, Low AI Writing Pattern, '
        'Moderate AI Writing Pattern, High AI Writing Pattern, Very High AI Writing Pattern)",\n'
        '  "reason": "string (2-3 sentence explanation of detected patterns)",\n'
        '  "features": ["string", ...] (list of 2-6 specific detected stylistic features)\n'
        "}\n"
        "Never return markdown fences. Never explain outside the JSON."
    )

    _ALLOWED_CLASSIFICATIONS = {
        "very low ai writing pattern",
        "low ai writing pattern",
        "moderate ai writing pattern",
        "high ai writing pattern",
        "very high ai writing pattern",
    }

    def __init__(self, groq_client: GroqPlagiarismClient):
        self.groq_client = groq_client

    def analyze_chunk(self, chunk: Dict[str, Any]) -> AIChunkResult:
        """
        Analyze a single text chunk for AI writing patterns.
        Returns an AIChunkResult with probability, confidence, classification, and features.
        """
        chunk_id = chunk["id"]
        chunk_text = chunk["text"]

        user_content = (
            f"chunk_id: {chunk_id}\n\n"
            f"--- PASSAGE ---\n{chunk_text}"
        )
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw_response = self.groq_client._call_groq_api(messages)
            data = json.loads(raw_response)

            # Normalise classification
            cls_lower = str(data.get("classification", "")).lower().strip()
            matched_cls = classify_ai_probability(int(data.get("ai_probability", 0)))
            for allowed in self._ALLOWED_CLASSIFICATIONS:
                if allowed in cls_lower:
                    # Convert to title-case display
                    matched_cls = " ".join(w.capitalize() for w in allowed.split())
                    break
            data["classification"] = matched_cls

            # Ensure chunk_id matches the request
            data["chunk_id"] = chunk_id

            # Validate and return
            result = AIChunkResult(**data)
            logger.info(f"AI analysis for {chunk_id}: {result.ai_probability}% ({result.classification})")
            return result

        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error for chunk {chunk_id}: {je}")
            return self._fallback_result(chunk_id, "JSON parse error from Groq.")
        except Exception as e:
            logger.error(f"Error analyzing chunk {chunk_id}: {e}")
            return self._fallback_result(chunk_id, str(e))

    @staticmethod
    def _fallback_result(chunk_id: str, error_msg: str) -> AIChunkResult:
        """Return a safe default result on failure."""
        return AIChunkResult(
            chunk_id=chunk_id,
            ai_probability=0,
            confidence=0,
            classification="Very Low AI Writing Pattern",
            reason=f"Analysis failed: {error_msg}",
            features=[]
        )


# ─────────────────────────────────────────────
# Orchestrator / Report Aggregator
# ─────────────────────────────────────────────

class AIWritingPatternService:
    """
    Orchestrates AI writing pattern analysis across all chunks of a document.
    Uses thread-based concurrency for faster processing.
    """

    def __init__(self, groq_client: GroqPlagiarismClient, max_workers: int = 3):
        self.analyzer = AIPatternAnalyzer(groq_client)
        self.max_workers = max_workers

    def analyze_document(
        self,
        chunks: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes all chunks concurrently and produces an aggregated AI-pattern report.
        
        Returns a dict with:
          - overall_ai_score: int
          - average_confidence: int
          - classification: str
          - chunk_results: List[dict]
          - summary stats
        """
        N = len(chunks)
        if N == 0:
            return self._empty_report()

        logger.info(f"Starting AI writing pattern analysis on {N} chunks (max_workers={self.max_workers})")

        results_map: Dict[str, AIChunkResult] = {}
        completed_count = 0

        # Submit all chunks to the thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.analyzer.analyze_chunk, chunk): chunk
                for chunk in chunks
            }

            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                chunk_id = chunk["id"]
                completed_count += 1

                try:
                    result = future.result()
                    results_map[chunk_id] = result
                except Exception as e:
                    logger.error(f"Future error for chunk {chunk_id}: {e}")
                    results_map[chunk_id] = AIPatternAnalyzer._fallback_result(chunk_id, str(e))

                msg = (
                    f"AI Pattern Analysis: {completed_count}/{N} chunks processed"
                    f" — Chunk {chunk_id}"
                )
                if progress_callback:
                    progress_callback(completed_count, N, msg)

        # Rebuild in original order
        ordered_results: List[Dict[str, Any]] = []
        for chunk in chunks:
            cid = chunk["id"]
            res = results_map.get(cid)
            if res:
                ordered_results.append({
                    "chunk_id": cid,
                    "chunk_text": chunk["text"],
                    "word_count": chunk.get("word_count", len(chunk["text"].split())),
                    "ai_probability": res.ai_probability,
                    "confidence": res.confidence,
                    "classification": res.classification,
                    "reason": res.reason,
                    "features": res.features,
                })

        return self._aggregate(ordered_results)

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics from individual chunk results."""
        N = len(chunk_results)
        if N == 0:
            return AIWritingPatternService._empty_report()

        probs = [r["ai_probability"] for r in chunk_results]
        confs = [r["confidence"] for r in chunk_results]

        overall_score = round(sum(probs) / N)
        avg_confidence = round(sum(confs) / N)
        overall_classification = classify_ai_probability(overall_score)

        # Bucket counts
        very_high = sum(1 for p in probs if p > 80)
        high       = sum(1 for p in probs if 61 <= p <= 80)
        moderate   = sum(1 for p in probs if 41 <= p <= 60)
        low        = sum(1 for p in probs if 21 <= p <= 40)
        very_low   = sum(1 for p in probs if p <= 20)

        highest_chunk = max(chunk_results, key=lambda x: x["ai_probability"])
        lowest_chunk  = min(chunk_results, key=lambda x: x["ai_probability"])

        # Collect all detected features with frequency count
        feature_freq: Dict[str, int] = {}
        for r in chunk_results:
            for feat in r.get("features", []):
                feature_freq[feat] = feature_freq.get(feat, 0) + 1

        top_features = sorted(feature_freq.items(), key=lambda x: -x[1])[:10]

        return {
            "overall_ai_score": overall_score,
            "average_confidence": avg_confidence,
            "overall_classification": overall_classification,
            "total_chunks": N,
            "very_high_probability_chunks": very_high,
            "high_probability_chunks": high,
            "moderate_probability_chunks": moderate,
            "low_probability_chunks": low,
            "very_low_probability_chunks": very_low,
            "highest_chunk": {
                "chunk_id": highest_chunk["chunk_id"],
                "ai_probability": highest_chunk["ai_probability"],
                "classification": highest_chunk["classification"],
            },
            "lowest_chunk": {
                "chunk_id": lowest_chunk["chunk_id"],
                "ai_probability": lowest_chunk["ai_probability"],
                "classification": lowest_chunk["classification"],
            },
            "top_features": [{"feature": f, "count": c} for f, c in top_features],
            "chunk_results": chunk_results,
            "disclaimer": (
                "This analysis estimates writing patterns commonly associated with "
                "AI-generated content. It should not be interpreted as definitive proof "
                "that any passage was written by AI."
            ),
        }

    @staticmethod
    def _empty_report() -> Dict[str, Any]:
        return {
            "overall_ai_score": 0,
            "average_confidence": 0,
            "overall_classification": "Very Low AI Writing Pattern",
            "total_chunks": 0,
            "very_high_probability_chunks": 0,
            "high_probability_chunks": 0,
            "moderate_probability_chunks": 0,
            "low_probability_chunks": 0,
            "very_low_probability_chunks": 0,
            "highest_chunk": {},
            "lowest_chunk": {},
            "top_features": [],
            "chunk_results": [],
            "disclaimer": (
                "This analysis estimates writing patterns commonly associated with "
                "AI-generated content. It should not be interpreted as definitive proof "
                "that any passage was written by AI."
            ),
        }
