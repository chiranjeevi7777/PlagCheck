"""
OpenAlex retrieval client (async).

Free — no API key required.
Polite pool: include email in User-Agent for higher rate limits.
200M+ scholarly works across all disciplines.
"""

from __future__ import annotations

from typing import List, Optional
import httpx

from app.retrieval.base import BaseRetriever
from app.schemas.retrieval import CandidatePaper
from app.core.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.openalex.org/works"
_HEADERS = {
    "User-Agent": "PlagCheck-AI/2.0 (mailto:research@plagcheck.ai)"
}


class OpenAlexClient(BaseRetriever):
    """Async OpenAlex API client."""

    source_name = "openalex"

    def __init__(self, timeout: float = 12.0) -> None:
        self._timeout = timeout

    async def search(self, query: str, limit: int = 10) -> List[CandidatePaper]:
        params = {
            "search": query,
            "per-page": limit,
            "select": "title,abstract_inverted_index,authorships,doi,publication_year,cited_by_count,id",
            "filter": "has_abstract:true",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    _BASE_URL, params=params, headers=_HEADERS, follow_redirects=True
                )
                if response.status_code == 200:
                    items = response.json().get("results", [])
                    papers = [
                        p for item in items
                        if (p := self.parse(item)) is not None
                    ]
                    logger.info(
                        f"OpenAlex: {len(papers)} papers for '{query[:40]}'"
                    )
                    return papers
                else:
                    logger.warning(f"OpenAlex: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"OpenAlex search error: {e}")
        return []

    def parse(self, raw: dict) -> Optional[CandidatePaper]:
        title = (raw.get("title") or "").strip()
        # OpenAlex stores abstracts as inverted index — reconstruct
        abstract = self._reconstruct_abstract(raw.get("abstract_inverted_index") or {})
        if not title or not abstract:
            return None

        authors = [
            a.get("author", {}).get("display_name", "")
            for a in raw.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]
        doi = raw.get("doi") or ""
        url = f"https://doi.org/{doi}" if doi else raw.get("id", "")
        return self.normalize(
            CandidatePaper(
                title=title,
                abstract=abstract,
                authors=authors,
                url=url,
                doi=doi,
                year=raw.get("publication_year"),
                citation_count=raw.get("cited_by_count") or 0,
                source=self.source_name,
                source_confidence=0.90,
            )
        )

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict) -> str:
        """Reconstruct full abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)
