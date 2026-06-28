"""
Groq Verification Service.

Groq ONLY verifies plagiarism — it never searches, retrieves, or ranks.
Input: (suspected chunk, reference abstract, candidate metadata)
Output: ExplainableComparisonResult with full scoring breakdown.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from app.llm.groq_client import GroqAPIClient
from app.schemas.analysis import (
    ChunkComparisonResult,
    ExplainableComparisonResult,
    SentenceMatch,
)
from app.schemas.retrieval import CandidatePaper
from app.core.logging import get_logger

logger = get_logger(__name__)

_ALLOWED_CLASSES = {
    "original", "minor similarity", "light rewrite", "heavy rewrite",
    "heavy paraphrasing", "near duplicate", "exact copy",
}

_SYSTEM_PROMPT = """\
You are an expert plagiarism verification system.
Compare the SUSPECTED passage against the REFERENCE passage.
Analyse: exact copying, paraphrasing, structural changes, semantic alignment.

Return ONLY a JSON object matching this exact schema:
{
  "semantic_similarity": int (0-100),
  "exact_copy": int (0-100),
  "paraphrase": int (0-100),
  "classification": "Original"|"Minor Similarity"|"Light Rewrite"|"Heavy Rewrite"|"Heavy Paraphrasing"|"Near Duplicate"|"Exact Copy",
  "confidence": int (0-100),
  "reason": "string (2-3 sentences explaining the overlap)",
  "sentence_matches": [
    {
      "suspected_sentence": "string",
      "original_sentence": "string",
      "similarity_score": int (0-100),
      "match_type": "exact_copy"|"paraphrase"
    }
  ]
}
Include sentence_matches only for pairs with similarity >= 50.
The response MUST be a single raw JSON object. Do NOT wrap the JSON in markdown code blocks or backticks (e.g. do NOT use ```json or ```). Start directly with { and end with }."""


class GroqVerifier:
    """
    Verification-only LLM client.

    Receives a pre-selected candidate paper (from reranker) and
    produces a detailed plagiarism assessment for the chunk.
    """

    def __init__(self, groq_client: GroqAPIClient) -> None:
        self._groq = groq_client

    def verify(
        self,
        suspected_text: str,
        candidate: CandidatePaper,
    ) -> ExplainableComparisonResult:
        """
        Verify suspected_text against candidate.abstract using Groq LLM.

        Returns an ExplainableComparisonResult that includes both the
        LLM judgment and the retrieval scoring provenance.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"--- REFERENCE PASSAGE ---\n{candidate.abstract}\n\n"
                    f"--- SUSPECTED PASSAGE ---\n{suspected_text}"
                ),
            },
        ]
        try:
            logger.info(
                f"Verifying chunk against '{candidate.title[:60]}' "
                f"(source={candidate.source})"
            )
            data = self._groq.call_parsed(messages)
            comparison = self._parse_comparison(data)
            return ExplainableComparisonResult(
                **comparison.model_dump(),
                source=candidate.source,
                source_url=candidate.url or "",
                citation_count=candidate.citation_count,
                year=candidate.year,
                embedding_score=candidate.embedding_score,
                reranker_score=candidate.reranker_score,
                keyword_overlap=candidate.keyword_overlap,
                combined_score=candidate.combined_score,
            )
        except Exception as e:
            logger.error(f"GroqVerifier error: {e}")
            return self._fallback_result(candidate)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_comparison(data: Dict[str, Any]) -> ChunkComparisonResult:
        """Parse and normalise the raw LLM dict into a ChunkComparisonResult."""
        cls_raw = str(data.get("classification", "Original")).lower().strip()
        matched_cls = "Original"
        for ac in _ALLOWED_CLASSES:
            if ac in cls_raw:
                matched_cls = " ".join(w.capitalize() for w in ac.split())
                break
        data["classification"] = matched_cls

        # Normalise sentence_matches
        raw_matches: List[Dict[str, Any]] = data.get("sentence_matches", [])
        validated_matches = []
        for m in raw_matches:
            try:
                validated_matches.append(SentenceMatch(**m))
            except Exception:
                pass
        data["sentence_matches"] = [m.model_dump() for m in validated_matches]
        return ChunkComparisonResult(**data)

    @staticmethod
    def _fallback_result(candidate: CandidatePaper) -> ExplainableComparisonResult:
        return ExplainableComparisonResult(
            semantic_similarity=0,
            exact_copy=0,
            paraphrase=0,
            classification="Original",
            confidence=0,
            reason="Verification failed — LLM call error.",
            sentence_matches=[],
            source=candidate.source,
            source_url=candidate.url or "",
            citation_count=candidate.citation_count,
            year=candidate.year,
        )
