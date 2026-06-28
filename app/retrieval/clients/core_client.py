"""
CORE retrieval client (async).

CORE (core.ac.uk) — free for academic research with API key registration.
200M+ open access research papers. Replaces paid Brave Search.

API key registration: https://core.ac.uk/services/api
Set CORE_API_KEY in your .env file to enable this source.
If CORE_API_KEY is empty, this client is a no-op (returns empty list).
"""

from __future__ import annotations

from typing import List, Optional

import httpx

from app.retrieval.base import BaseRetriever
from app.schemas.retrieval import CandidatePaper
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.core.ac.uk/v3/search/outputs"


class CoreClient(BaseRetriever):
    """
    Async CORE API client.

    Enabled only when CORE_API_KEY is set in .env.
    Returns empty list gracefully when the key is absent or the API fails.
    """

    source_name = "core"

    def __init__(self, timeout: float = 12.0) -> None:
        self._timeout = timeout
        self._api_key = settings.core_api_key

    async def search(self, query: str, limit: int = 10) -> List[CandidatePaper]:
        if not self._api_key:
            # Graceful no-op — CORE_API_KEY not configured
            return []

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        params = {"q": query, "limit": limit}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(_BASE_URL, params=params, headers=headers)
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    papers = [
                        p for item in results
                        if (p := self.parse(item)) is not None
                    ]
                    logger.info(
                        f"CORE: {len(papers)} papers for '{query[:40]}'"
                    )
                    return papers
                elif response.status_code == 401:
                    logger.warning("CORE: Invalid API key (401). Check CORE_API_KEY.")
                elif response.status_code == 429:
                    logger.warning("CORE: Rate limited (429).")
                else:
                    logger.warning(f"CORE: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"CORE search error: {e}")
        return []

    def parse(self, raw: dict) -> Optional[CandidatePaper]:
        title = (raw.get("title") or "").strip()
        abstract = (raw.get("abstract") or "").strip()
        if not title or not abstract:
            return None

        authors = [
            a.get("name", "").strip()
            for a in raw.get("authors") or []
            if a.get("name")
        ]
        doi = raw.get("doi") or ""
        url = raw.get("sourceFulltextUrls", [None])[0] or raw.get("downloadUrl") or ""
        year: Optional[int] = None
        try:
            year_raw = raw.get("yearPublished")
            if year_raw:
                year = int(year_raw)
        except (ValueError, TypeError):
            pass

        return self.normalize(
            CandidatePaper(
                title=title,
                abstract=abstract,
                authors=authors,
                url=url,
                doi=doi,
                year=year,
                citation_count=raw.get("citationCount") or 0,
                source=self.source_name,
                source_confidence=0.82,
            )
        )
