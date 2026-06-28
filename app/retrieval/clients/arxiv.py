"""
arXiv retrieval client (async).

Free — no API key required.
Atom XML API. Best for CS, physics, math, and quantitative biology preprints.
Rate limit: 3 requests / second (enforced by caller concurrency).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Optional
from urllib.parse import quote_plus

import httpx

from app.retrieval.base import BaseRetriever
from app.schemas.retrieval import CandidatePaper
from app.core.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://export.arxiv.org/api/query"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivClient(BaseRetriever):
    """Async arXiv Atom API client."""

    source_name = "arxiv"

    def __init__(self, timeout: float = 12.0) -> None:
        self._timeout = timeout

    async def search(self, query: str, limit: int = 10) -> List[CandidatePaper]:
        params = {
            "search_query": f"all:{query}",
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(_BASE_URL, params=params)
                if response.status_code == 200:
                    papers = self._parse_atom(response.text)
                    logger.info(
                        f"arXiv: {len(papers)} papers for '{query[:40]}'"
                    )
                    return papers
                else:
                    logger.warning(f"arXiv: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"arXiv search error: {e}")
        return []

    def _parse_atom(self, xml_text: str) -> List[CandidatePaper]:
        papers: List[CandidatePaper] = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", _NS):
                paper = self.parse(entry)
                if paper is not None:
                    papers.append(paper)
        except ET.ParseError as e:
            logger.warning(f"arXiv XML parse error: {e}")
        return papers

    def parse(self, raw) -> Optional[CandidatePaper]:  # raw is ET.Element
        try:
            title = (raw.findtext("atom:title", namespaces=_NS) or "").strip().replace("\n", " ")
            abstract = (raw.findtext("atom:summary", namespaces=_NS) or "").strip().replace("\n", " ")
            if not title or not abstract:
                return None

            authors = [
                (a.findtext("atom:name", namespaces=_NS) or "").strip()
                for a in raw.findall("atom:author", _NS)
            ]
            # Use the HTML abstract URL as the paper URL
            entry_id = (raw.findtext("atom:id", namespaces=_NS) or "").strip()
            url = entry_id.replace("http://arxiv.org/abs/", "https://arxiv.org/abs/")

            # Extract year from published date
            published = raw.findtext("atom:published", namespaces=_NS) or ""
            year: Optional[int] = None
            if published:
                try:
                    year = int(published[:4])
                except ValueError:
                    pass

            return self.normalize(
                CandidatePaper(
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    url=url,
                    year=year,
                    citation_count=0,  # arXiv API does not return citation count
                    source=self.source_name,
                    source_confidence=0.85,
                )
            )
        except Exception as e:
            logger.warning(f"arXiv parse error for entry: {e}")
            return None
