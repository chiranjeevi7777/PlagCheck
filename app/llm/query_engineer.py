"""
Query Engineering — generates multiple query variants from a document chunk.

A single Groq call produces four semantically diverse queries to maximise
retrieval recall across all configured academic sources.

Results are cached by the SHA-256 of the first 500 chars of the chunk text
so the same document segment never triggers a redundant LLM call.
"""

from __future__ import annotations

import hashlib
from typing import Dict

from app.llm.groq_client import GroqAPIClient
from app.schemas.retrieval import QueryBundle
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an academic information retrieval specialist.
Given a passage of text, generate FOUR distinct search queries to find related academic papers:

1. keyword_query   — 3-5 domain-specific technical keywords (shortest, most precise)
2. semantic_query  — full conceptual rephrasing suitable for semantic search
3. expanded_query  — keyword_query augmented with synonyms and related terms
4. academic_query  — formal/technical reformulation suited for scholarly databases

Return ONLY valid JSON matching this schema:
{
  "keyword_query": "string",
  "semantic_query": "string",
  "expanded_query": "string",
  "academic_query": "string"
}
The response MUST be a single raw JSON object. Do NOT wrap the JSON in markdown code blocks or backticks (e.g. do NOT use ```json or ```). Start directly with { and end with }."""

_FALLBACK_QUERY = "machine learning deep learning artificial intelligence"


class QueryEngineer:
    """
    Generates diverse query variants for multi-source academic retrieval.

    Uses a single Groq LLM call per unique text to produce four query types.
    Results are cached in-process to avoid repeated identical LLM calls.
    """

    def __init__(self, groq_client: GroqAPIClient) -> None:
        self._groq = groq_client
        self._cache: Dict[str, QueryBundle] = {}

    def generate(self, text: str) -> QueryBundle:
        """
        Generate a QueryBundle for *text*.

        Falls back gracefully to a keyword-only bundle on any LLM failure.
        """
        cache_key = hashlib.sha256(text[:500].encode()).hexdigest()
        if cache_key in self._cache:
            logger.info("QueryEngineer cache hit — skipping LLM call.")
            return self._cache[cache_key]

        snippet = text[:1500]
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Passage:\n{snippet}"},
        ]
        try:
            data = self._groq.call_parsed(messages)
            bundle = QueryBundle(
                keyword_query=data.get("keyword_query", _FALLBACK_QUERY).strip(),
                semantic_query=data.get("semantic_query", _FALLBACK_QUERY).strip(),
                expanded_query=data.get("expanded_query", _FALLBACK_QUERY).strip(),
                academic_query=data.get("academic_query", _FALLBACK_QUERY).strip(),
            )
            logger.info(f"Generated queries: {bundle.keyword_query!r} (and 3 variants)")
        except Exception as e:
            logger.warning(f"QueryEngineer LLM call failed ({e}). Using fallback queries.")
            bundle = QueryBundle(
                keyword_query=_FALLBACK_QUERY,
                semantic_query=_FALLBACK_QUERY,
                expanded_query=_FALLBACK_QUERY,
                academic_query=_FALLBACK_QUERY,
            )

        self._cache[cache_key] = bundle
        return bundle
