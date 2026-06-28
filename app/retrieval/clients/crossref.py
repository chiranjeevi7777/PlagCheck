"""
Crossref retrieval client (async).

Free — no API key required (polite pool with mailto in User-Agent).
130M+ metadata records for academic journals and conference papers.
"""

from __future__ import annotations

from typing import List, Optional

import httpx

from app.retrieval.base import BaseRetriever
from app.schemas.retrieval import CandidatePaper
from app.core.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.crossref.org/works"
_HEADERS = {
    "User-Agent": "PlagCheck-AI/2.0 (mailto:research@plagcheck.ai)"
}


class CrossrefClient(BaseRetriever):
    """Async Crossref Works API client."""

    source_name = "crossref"

    def __init__(self, timeout: float = 12.0) -> None:
        self._timeout = timeout

    async def search(self, query: str, limit: int = 10) -> List[CandidatePaper]:
        params = {
            "query": query,
            "rows": limit,
            "select": "title,abstract,author,DOI,published,is-referenced-by-count,URL",
            "filter": "has-abstract:true",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    _BASE_URL, params=params, headers=_HEADERS
                )
                if response.status_code == 200:
                    items = response.json().get("message", {}).get("items", [])
                    papers = [
                        p for item in items
                        if (p := self.parse(item)) is not None
                    ]
                    logger.info(
                        f"Crossref: {len(papers)} papers for '{query[:40]}'"
                    )
                    return papers
                else:
                    logger.warning(f"Crossref: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"Crossref search error: {e}")
        return []

    def parse(self, raw: dict) -> Optional[CandidatePaper]:
        title_list = raw.get("title") or []
        title = (title_list[0] if title_list else "").strip()
        abstract = (raw.get("abstract") or "").strip()
        # Strip JATS XML tags sometimes returned by Crossref
        import re
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        if not title or not abstract:
            return None

        authors = []
        for a in raw.get("author") or []:
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        doi = raw.get("DOI", "")
        url = raw.get("URL") or (f"https://doi.org/{doi}" if doi else "")

        year: Optional[int] = None
        pub = raw.get("published", {}).get("date-parts", [[]])
        if pub and pub[0]:
            try:
                year = int(pub[0][0])
            except (ValueError, IndexError):
                pass

        return self.normalize(
            CandidatePaper(
                title=title,
                abstract=abstract,
                authors=authors,
                url=url,
                doi=doi,
                year=year,
                citation_count=raw.get("is-referenced-by-count") or 0,
                source=self.source_name,
                source_confidence=0.88,
            )
        )
