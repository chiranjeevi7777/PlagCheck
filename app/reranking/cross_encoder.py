"""
Cross-encoder reranking service.

Uses sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2 by default)
to produce relevance scores for (query, abstract) pairs.

The combined ranking score is:
  combined = 0.50 * reranker_score
           + 0.30 * embedding_cosine_similarity
           + 0.20 * keyword_overlap

Falls back to keyword-only ranking if model is unavailable.
Model is lazy-loaded and configurable via RERANKER_MODEL env var.
"""

from __future__ import annotations

import re
from typing import List, Set

import numpy as np

from app.reranking.base import BaseReranker
from app.schemas.retrieval import CandidatePaper
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_STOP_WORDS: Set[str] = {
    "the", "and", "that", "for", "with", "this", "from", "are", "was", "were",
    "have", "has", "had", "been", "they", "their", "there", "then", "about",
    "would", "could", "should", "your", "them", "some", "other", "into",
    "than", "its", "also", "these", "those", "such", "only", "over", "more",
    "most", "both", "each", "under", "between", "through", "during", "before",
    "after", "above", "below",
}


class CrossEncoderReranker(BaseReranker):
    """
    Hybrid reranker combining CrossEncoder scores, embedding cosine similarity,
    and keyword overlap into a single combined ranking score.

    Supports Hugging Face Inference API to offload deep learning computation,
    completely preventing OOM crashes in low-memory environments like Render.
    Falls back to keyword-only overlap comparison when both HF and local models are unavailable.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.reranker_model
        self._model = None
        self._available: bool | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[CandidatePaper],
        top_k: int = 10,
    ) -> List[CandidatePaper]:
        """
        Score all candidates and return top_k sorted by combined_score descending.
        Embedding scores must already be set on each candidate before calling.
        """
        if not candidates:
            return []

        query_words = self._content_words(query)
        abstracts = [c.abstract for c in candidates]
        reranker_scores: List[float] | None = None

        # 1. Try Hugging Face Inference API if enabled
        if settings.use_huggingface_api:
            reranker_scores = self._hf_api_call(query, abstracts)

        # 2. Try local model if API failed/disabled
        if reranker_scores is None:
            model = self._get_model()
            if model is not None:
                pairs = [(query, c.abstract[:512]) for c in candidates]
                try:
                    raw_scores = model.predict(pairs)
                    # Sigmoid to normalise raw cross-encoder scores to [0,1]
                    reranker_scores = [float(1 / (1 + np.exp(-s))) for s in raw_scores]
                except Exception as e:
                    logger.warning(f"CrossEncoder prediction error: {e}. Using keyword fallback.")

        # 3. Fallback: keyword overlap as reranker proxy
        if reranker_scores is None:
            reranker_scores = [
                self._keyword_overlap(query_words, self._content_words(c.abstract))
                for c in candidates
            ]

        # Combine scores
        for candidate, r_score in zip(candidates, reranker_scores):
            kw = self._keyword_overlap(query_words, self._content_words(candidate.abstract))
            candidate.reranker_score = round(r_score, 4)
            candidate.keyword_overlap = round(kw, 4)
            candidate.combined_score = round(
                0.50 * r_score + 0.30 * candidate.embedding_score + 0.20 * kw, 4
            )

        ranked = sorted(candidates, key=lambda c: c.combined_score, reverse=True)
        logger.info(
            f"Reranked {len(candidates)} candidates -> top {top_k}. "
            f"Best combined_score={ranked[0].combined_score:.3f}"
        )
        return ranked[:top_k]

    @property
    def is_available(self) -> bool:
        """True if either Hugging Face Inference API or the local model is available."""
        if settings.use_huggingface_api:
            return True
        return self._get_model() is not None

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._available is False:
            return None
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            logger.info(f"Loading reranker model: {self._model_name}")
            self._model = CrossEncoder(self._model_name)
            self._available = True
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load reranker '{self._model_name}': {e}. "
                "Using keyword-only fallback ranking."
            )
            self._available = False
        return self._model

    def _hf_api_call(self, query: str, abstracts: List[str]) -> List[float] | None:
        """Call Hugging Face Inference API to get cross-encoder rerank scores."""
        import httpx
        import time

        headers = {}
        if settings.hf_api_token:
            headers["Authorization"] = f"Bearer {settings.hf_api_token}"

        url = f"https://api-inference.huggingface.co/models/{self._model_name}"
        payload = {"inputs": [[query, abs[:512]] for abs in abstracts]}

        try:
            with httpx.Client(timeout=15.0) as client:
                for attempt in range(3):
                    response = client.post(url, json=payload, headers=headers)
                    if response.status_code == 200:
                        res = response.json()
                        scores: List[float] = []

                        if not isinstance(res, list):
                            logger.warning(f"HF reranker returned non-list: {res}")
                            return None

                        for item in res:
                            if isinstance(item, (int, float)):
                                scores.append(float(item))
                            elif isinstance(item, dict):
                                scores.append(float(item.get("score", 0.0)))
                            elif isinstance(item, list) and len(item) > 0:
                                first = item[0]
                                if isinstance(first, dict):
                                    scores.append(float(first.get("score", 0.0)))
                                elif isinstance(first, (int, float)):
                                    scores.append(float(first))
                                else:
                                    scores.append(0.0)
                            else:
                                scores.append(0.0)

                        if len(scores) == len(abstracts):
                            normalized_scores = []
                            for s in scores:
                                if s < 0.0 or s > 1.0:
                                    # Normalize raw logits to [0, 1] using standard sigmoid
                                    normalized_scores.append(float(1.0 / (1.0 + np.exp(-s))))
                                else:
                                    normalized_scores.append(s)
                            return normalized_scores
                        return None

                    elif response.status_code == 503:
                        try:
                            data = response.json()
                            est_time = data.get("estimated_time", 5.0)
                        except Exception:
                            est_time = 5.0
                        logger.info(f"HF model {self._model_name} is loading. Waiting {est_time:.1f}s...")
                        time.sleep(min(est_time, 5.0))
                    else:
                        logger.warning(f"HF Reranker status {response.status_code}: {response.text}")
                        break
        except Exception as e:
            logger.warning(f"HF Reranker connection error: {e}")
        return None

    @staticmethod
    def _content_words(text: str) -> Set[str]:
        words = re.findall(r"\b[a-z0-9]{3,}\b", text.lower())
        return {w for w in words if w not in _STOP_WORDS}

    @staticmethod
    def _keyword_overlap(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(a), len(b))

