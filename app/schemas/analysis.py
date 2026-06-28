"""
Analysis-layer schemas.

Pydantic models for LLM comparison results and AI writing pattern results.
Previously defined inline inside groq_client.py and ai_analyzer.py.
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


# ── Plagiarism comparison ─────────────────────────────────────────────────────

class SentenceMatch(BaseModel):
    """A single matching sentence pair between suspected and reference text."""

    suspected_sentence: str = Field(description="Sentence from the suspected document.")
    original_sentence: str = Field(description="Corresponding sentence from the reference.")
    similarity_score: int = Field(description="Sentence-level similarity score (0-100).")
    match_type: str = Field(description="'exact_copy' or 'paraphrase'.")


class ChunkComparisonResult(BaseModel):
    """Full comparison result for a single suspected chunk vs a reference abstract."""

    semantic_similarity: int = Field(description="Overall semantic similarity (0-100).")
    exact_copy: int = Field(description="Exact copy percentage (0-100).")
    paraphrase: int = Field(description="Paraphrase percentage (0-100).")
    classification: str = Field(
        description=(
            "One of: Original, Minor Similarity, Light Rewrite, Heavy Rewrite, "
            "Heavy Paraphrasing, Near Duplicate, Exact Copy."
        )
    )
    confidence: int = Field(description="LLM confidence in this result (0-100).")
    reason: str = Field(description="Brief explanation of detected overlap.")
    sentence_matches: List[SentenceMatch] = Field(
        default_factory=list,
        description="Specific matched sentence pairs (similarity >= 50).",
    )


class ExplainableComparisonResult(ChunkComparisonResult):
    """Comparison result augmented with retrieval provenance for explainability."""

    # Retrieval provenance
    source: str = ""              # retrieval source name
    source_url: str = ""
    citation_count: int = 0
    year: int | None = None

    # Retrieval scoring breakdown
    embedding_score: float = 0.0
    reranker_score: float = 0.0
    keyword_overlap: float = 0.0
    combined_score: float = 0.0


# ── AI writing pattern analysis ───────────────────────────────────────────────

_CLASSIFICATIONS = {
    "very low ai writing pattern",
    "low ai writing pattern",
    "moderate ai writing pattern",
    "high ai writing pattern",
    "very high ai writing pattern",
}


def classify_ai_probability(probability: int) -> str:
    """Map 0-100 AI probability to a human-readable classification label."""
    if probability <= 20:
        return "Very Low AI Writing Pattern"
    elif probability <= 40:
        return "Low AI Writing Pattern"
    elif probability <= 60:
        return "Moderate AI Writing Pattern"
    elif probability <= 80:
        return "High AI Writing Pattern"
    return "Very High AI Writing Pattern"


class AIChunkResult(BaseModel):
    """AI writing pattern assessment for a single text chunk."""

    chunk_id: str
    ai_probability: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    classification: str
    reason: str
    features: List[str]
