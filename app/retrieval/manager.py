"""
Retrieval Manager — concurrent multi-source academic literature retrieval.

Responsibilities:
1. Run all enabled retrieval clients concurrently (asyncio.gather).
2. Merge results from all sources and queries.
3. Deduplicate by DOI then by title similarity.
4. Compute embedding scores for all candidates.
5. Run CrossEncoder reranking → return top-K candidates.
6. Fall back to Groq-generated mock papers if all sources fail.

No single source failure can crash the pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Dict, List, Optional

from app.retrieval.base import BaseRetriever
from app.retrieval.clients.semantic_scholar import SemanticScholarClient
from app.retrieval.clients.openalex import OpenAlexClient
from app.retrieval.clients.arxiv import ArxivClient
from app.retrieval.clients.crossref import CrossrefClient
from app.retrieval.clients.core_client import CoreClient
from app.embedding.bge_embedder import BGEEmbedder
from app.reranking.cross_encoder import CrossEncoderReranker
from app.schemas.retrieval import CandidatePaper, QueryBundle, RetrievalResult
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RetrievalManager:
    """
    Unified retrieval orchestrator.

    Instantiate once and reuse across requests — embedding and reranking
    models are lazy-loaded and cached after first use.
    """

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self._embedder = embedder or BGEEmbedder()
        self._reranker = reranker or CrossEncoderReranker()
        self._search_cache: Dict[str, List[CandidatePaper]] = {}

        # Build enabled client list
        self._clients: List[BaseRetriever] = [SemanticScholarClient()]
        if settings.enable_openalex:
            self._clients.append(OpenAlexClient())
        if settings.enable_arxiv:
            self._clients.append(ArxivClient())
        if settings.enable_crossref:
            self._clients.append(CrossrefClient())
        if settings.enable_core:
            self._clients.append(CoreClient())

        logger.info(
            f"RetrievalManager ready — sources: "
            f"{[c.source_name for c in self._clients]}"
        )

    # ── Public ────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query_bundle: QueryBundle,
        chunk_text: str,
        papers_per_source: int = 8,
    ) -> RetrievalResult:
        """
        Retrieve, deduplicate, embed, and rerank candidates for a query bundle.

        Returns a RetrievalResult with up to MAX_RERANKED_PAPERS candidates,
        each with full scoring breakdown (embedding, reranker, keyword, combined).
        """
        t0 = time.monotonic()
        queries = query_bundle.all_queries()

        # ── Step 1: Concurrent retrieval ────────────────────────────────────
        all_raw: List[CandidatePaper] = await self._fetch_all(queries, papers_per_source)
        total_raw = len(all_raw)

        # ── Step 2: Deduplicate ─────────────────────────────────────────────
        unique = self._deduplicate(all_raw)
        logger.info(f"Retrieval: {total_raw} raw -> {len(unique)} unique candidates")

        # ── Step 3: Embed chunk + candidates ────────────────────────────────
        if settings.enable_embedding and self._embedder.is_available:
            chunk_vec = self._embedder.encode(chunk_text)
            abstract_texts = [c.abstract for c in unique]
            abstract_vecs = self._embedder.encode_batch(abstract_texts)
            from app.embedding.base import BaseEmbedder
            for candidate, abs_vec in zip(unique, abstract_vecs):
                candidate.embedding_score = round(
                    BaseEmbedder.cosine_similarity(chunk_vec, abs_vec), 4
                )

        # ── Step 4: Rerank ──────────────────────────────────────────────────
        top_k = settings.max_reranked_papers
        if settings.enable_reranking and unique:
            ranked = self._reranker.rerank(
                query=query_bundle.keyword_query,
                candidates=unique,
                top_k=top_k,
            )
        else:
            # Sort by embedding score if reranking disabled
            ranked = sorted(unique, key=lambda c: c.embedding_score, reverse=True)[:top_k]

        elapsed_ms = (time.monotonic() - t0) * 1000
        sources_used = list({c.source for c in ranked})
        logger.info(
            f"RetrievalManager: returning {len(ranked)} candidates in {elapsed_ms:.0f}ms "
            f"(sources: {sources_used})"
        )
        return RetrievalResult(
            query_bundle=query_bundle,
            candidates=ranked,
            total_raw=total_raw,
            total_unique=len(unique),
            sources_used=sources_used,
            retrieval_time_ms=round(elapsed_ms, 1),
        )

    def generate_mock_papers(
        self, query: str, groq_client
    ) -> List[CandidatePaper]:
        """
        Last-resort fallback: generate synthetic reference papers via Groq LLM.
        Only called when ALL retrieval sources return empty results.
        """
        logger.warning("All retrieval sources empty — generating mock papers via Groq.")
        system_prompt = (
            "Generate 5 realistic academic paper abstracts for the given query. "
            "Return ONLY JSON: "
            '{"papers": [{"title":"","abstract":"","authors":[""],"url":"","year":0}]}'
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"},
        ]
        try:
            data = groq_client.call_parsed(messages)
            papers = []
            for p in data.get("papers", []):
                papers.append(
                    CandidatePaper(
                        title=p.get("title", "Unknown"),
                        abstract=p.get("abstract", ""),
                        authors=p.get("authors", []),
                        url=p.get("url", ""),
                        year=p.get("year"),
                        source="groq_fallback",
                        source_confidence=0.5,
                    )
                )
            return papers
        except Exception as e:
            logger.error(f"Mock paper generation failed: {e}")
            return []

    # ── Private ───────────────────────────────────────────────────────────────

    async def _fetch_all(
        self, queries: List[str], limit_per: int
    ) -> List[CandidatePaper]:
        """Launch all (query × client) searches concurrently."""
        tasks = []
        for query in queries:
            for client in self._clients:
                cache_key = hashlib.sha256(
                    f"{client.source_name}:{query}".encode()
                ).hexdigest()
                if cache_key in self._search_cache:
                    tasks.append(self._return_cached(self._search_cache[cache_key]))
                else:
                    tasks.append(
                        self._fetch_with_cache(client, query, limit_per, cache_key)
                    )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        papers: List[CandidatePaper] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Retrieval task error: {r}")
            elif isinstance(r, list):
                papers.extend(r)
        return papers

    async def _fetch_with_cache(
        self,
        client: BaseRetriever,
        query: str,
        limit: int,
        cache_key: str,
    ) -> List[CandidatePaper]:
        results = await client.search(query, limit)
        self._search_cache[cache_key] = results
        return results

    @staticmethod
    async def _return_cached(cached: List[CandidatePaper]) -> List[CandidatePaper]:
        """Async wrapper around an already-cached result."""
        return cached

    @staticmethod
    def _deduplicate(papers: List[CandidatePaper]) -> List[CandidatePaper]:
        """Remove duplicates by DOI (exact) then by normalised title (fuzzy)."""
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()
        unique: List[CandidatePaper] = []

        for p in papers:
            # DOI dedup
            if p.doi:
                doi_norm = p.doi.lower().strip()
                if doi_norm in seen_dois:
                    continue
                seen_dois.add(doi_norm)

            # Title dedup (normalised)
            title_norm = re.sub(r"[^a-z0-9]", "", p.title.lower())[:60]
            if title_norm and title_norm in seen_titles:
                continue
            seen_titles.add(title_norm)

            if p.abstract.strip():
                unique.append(p)

        return unique
