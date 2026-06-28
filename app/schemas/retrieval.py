"""
Retrieval-layer schemas.

These are the canonical data shapes for all retrieved academic papers,
query bundles, and ranked retrieval results used across the RAG pipeline.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class CandidatePaper(BaseModel):
    """A single retrieved academic paper from any retrieval source."""

    title: str
    abstract: str
    authors: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    citation_count: int = 0
    source: str  # "semantic_scholar" | "openalex" | "arxiv" | "crossref" | "core"
    source_confidence: float = 1.0  # Source reliability weight (0-1)

    # ── Scoring fields populated during ranking ──────────────────────────────
    embedding_score: float = 0.0   # Cosine similarity with chunk embedding
    reranker_score: float = 0.0    # CrossEncoder relevance score
    keyword_overlap: float = 0.0   # Normalised content-word overlap (0-1)
    combined_score: float = 0.0    # Final hybrid ranking score


class QueryBundle(BaseModel):
    """Four query variants generated from a single document chunk."""

    keyword_query: str   # 3-5 domain-specific keywords
    semantic_query: str  # Conceptually rephrased alternative
    expanded_query: str  # Synonyms and related terms added
    academic_query: str  # Formal/technical reformulation

    def all_queries(self) -> List[str]:
        """Return deduplicated list of all query variants."""
        seen: set[str] = set()
        result: List[str] = []
        for q in [
            self.keyword_query,
            self.semantic_query,
            self.expanded_query,
            self.academic_query,
        ]:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                result.append(q)
        return result


class RetrievalResult(BaseModel):
    """Aggregated retrieval result for a single document query."""

    query_bundle: QueryBundle
    candidates: List[CandidatePaper] = Field(default_factory=list)
    total_raw: int = 0        # Raw candidates before deduplication
    total_unique: int = 0     # Unique candidates after dedup
    sources_used: List[str] = Field(default_factory=list)
    retrieval_time_ms: float = 0.0
