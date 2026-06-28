"""
Abstract base class for all retrieval clients.

Every retrieval source must implement this interface.
No client should know about or depend on another client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.schemas.retrieval import CandidatePaper


class BaseRetriever(ABC):
    """Abstract retrieval source client."""

    source_name: str = "unknown"

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[CandidatePaper]:
        """
        Search the source for papers matching *query*.

        Must return a (possibly empty) list on failure — never raise.
        """
        ...

    @abstractmethod
    def parse(self, raw: dict) -> CandidatePaper | None:
        """
        Parse a single raw API response item into a CandidatePaper.

        Return None if the item cannot be parsed (missing title/abstract).
        """
        ...

    def normalize(self, paper: CandidatePaper) -> CandidatePaper:
        """
        Normalise a parsed CandidatePaper (trim whitespace, clamp values, etc.).

        Override per-source if normalisation rules differ.
        """
        paper.title = paper.title.strip()
        paper.abstract = paper.abstract.strip()
        paper.authors = [a.strip() for a in paper.authors if a.strip()]
        return paper
