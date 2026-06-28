"""
BGE Embedding Service using BAAI/bge-small-en-v1.5 (default).

Features:
- Lazy model loading (downloaded on first call, not at startup)
- SHA-256 keyed in-process cache (avoids re-encoding same text)
- Graceful degradation: returns zero-vectors if model unavailable
- Model configurable via EMBEDDING_MODEL env var

Memory footprint:
  bge-small-en-v1.5  ≈ 130 MB  (default, safe for Render free tier)
  bge-large-en-v1.5  ≈ 1.2 GB  (set EMBEDDING_MODEL= for high accuracy)
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np

from app.embedding.base import BaseEmbedder
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BGEEmbedder(BaseEmbedder):
    """
    Sentence-transformers BGE embedding service with in-process LRU cache.

    Falls back to zero-vectors when sentence-transformers is not installed
    or the model cannot be loaded, so the pipeline continues without crashing.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._model = None          # Lazy-loaded
        self._cache: Dict[str, np.ndarray] = {}
        self._available: bool | None = None  # None = not yet checked

    # ── Public ────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Return a normalised embedding for *text*, using cache when possible."""
        key = self._hash(text)
        if key not in self._cache:
            self._cache[key] = self._encode_single(text)
        return self._cache[key]

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Return normalised embeddings for all *texts*, using cache for known entries."""
        model = self._get_model()
        if model is None:
            return [np.zeros(384) for _ in texts]

        # Only encode texts not yet in cache
        missing_indices = [i for i, t in enumerate(texts) if self._hash(t) not in self._cache]
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            try:
                vectors = model.encode(
                    missing_texts,
                    batch_size=settings.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for idx, vec in zip(missing_indices, vectors):
                    self._cache[self._hash(texts[idx])] = vec
            except Exception as e:
                logger.warning(f"BGEEmbedder batch encode error: {e}")
                for idx in missing_indices:
                    self._cache[self._hash(texts[idx])] = np.zeros(384)

        return [self._cache[self._hash(t)] for t in texts]

    @property
    def is_available(self) -> bool:
        """True if the embedding model loaded successfully."""
        if self._available is None:
            self._available = self._get_model() is not None
        return self._available

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._available is False:
            return None
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            self._available = True
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load embedding model '{self._model_name}': {e}. "
                "Embedding scores will be 0.0 — pipeline continues."
            )
            self._available = False
        return self._model

    def _encode_single(self, text: str) -> np.ndarray:
        model = self._get_model()
        if model is None:
            return np.zeros(384)
        try:
            vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            return np.array(vec)
        except Exception as e:
            logger.warning(f"BGEEmbedder single encode error: {e}")
            return np.zeros(384)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text[:1000].encode()).hexdigest()
