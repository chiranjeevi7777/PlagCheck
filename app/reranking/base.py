"""Abstract base class for reranking services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.schemas.retrieval import CandidatePaper


class BaseReranker(ABC):
    """Abstract reranker — receives (query, candidates) and returns ranked list."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[CandidatePaper],
        top_k: int,
    ) -> List[CandidatePaper]:
        """
        Rerank *candidates* by relevance to *query*.

        Returns the top *top_k* candidates sorted by descending relevance.
        Must never raise — return *candidates[:top_k]* as fallback on error.
        """
        ...
