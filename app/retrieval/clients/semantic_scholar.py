"""
Semantic Scholar retrieval client (async).

Free tier — no API key required.
Rate limit: 100 requests / 5 minutes (unauthenticated).
"""

from __future__ import annotations

from typing import List, Optional

import httpx

from app.retrieval.base import BaseRetriever
from app.schemas.retrieval import CandidatePaper
from app.core.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,authors,url,year,citationCount"
_HEADERS = {
    "User-Agent": (
        "PlagCheck-AI/2.0 (academic plagiarism research; "
        "contact: research@plagcheck.ai)"
    )
}


class SemanticScholarClient(BaseRetriever):
    """Async Semantic Scholar API client."""

    source_name = "semantic_scholar"

    def __init__(self, timeout: float = 12.0) -> None:
        self._timeout = timeout

    async def search(self, query: str, limit: int = 10) -> List[CandidatePaper]:
        params = {"query": query, "limit": limit, "fields": _FIELDS}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(_BASE_URL, params=params, headers=_HEADERS)
                if response.status_code == 200:
                    data = response.json()
                    papers = [
                        p for item in data.get("data", [])
                        if (p := self.parse(item)) is not None
                    ]
                    logger.info(
                        f"SemanticScholar: {len(papers)} papers for '{query[:40]}'"
                    )
                    return papers
                elif response.status_code == 429:
                    logger.warning("SemanticScholar: rate limited (429).")
                else:
                    logger.warning(
                        f"SemanticScholar: HTTP {response.status_code} for '{query[:40]}'"
                    )
        except Exception as e:
            logger.warning(f"SemanticScholar search error: {e}")
        return []

    def parse(self, raw: dict) -> Optional[CandidatePaper]:
        title = (raw.get("title") or "").strip()
        abstract = (raw.get("abstract") or "").strip()
        if not title or not abstract:
            return None
        authors = [
            a["name"] for a in raw.get("authors", []) if a.get("name")
        ]
        return self.normalize(
            CandidatePaper(
                title=title,
                abstract=abstract,
                authors=authors,
                url=raw.get("url") or "",
                year=raw.get("year"),
                citation_count=raw.get("citationCount") or 0,
                source=self.source_name,
                source_confidence=0.95,
            )
        )
