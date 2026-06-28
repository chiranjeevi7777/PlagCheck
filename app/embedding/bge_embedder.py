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

    Supports Hugging Face Inference API to offload deep learning computation,
    completely preventing OOM crashes in low-memory environments like Render.
    Falls back to a signed hashing vectorizer for zero-dependency keyword cosine matching.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._model = None          # Lazy-loaded local model
        self._cache: Dict[str, np.ndarray] = {}
        self._available: bool | None = None  # None = not yet checked

    # ── Public ────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Return a normalised embedding for *text*, using cache when possible."""
        key = self._hash(text)
        if key in self._cache:
            return self._cache[key]

        # 1. Try Hugging Face Inference API if enabled
        if settings.use_huggingface_api:
            vecs = self._hf_api_call([text])
            if vecs is not None and len(vecs) == 1:
                self._cache[key] = vecs[0]
                return vecs[0]

        # 2. Try local model
        model = self._get_model()
        if model is not None:
            try:
                vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
                vec = np.array(vec, dtype=np.float32)
                self._cache[key] = vec
                return vec
            except Exception as e:
                logger.warning(f"BGEEmbedder local encode error: {e}")

        # 3. Fallback to signed hashing vectorizer
        vec = self._fallback_encode(text)
        self._cache[key] = vec
        return vec

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Return normalised embeddings for all *texts*, using cache for known entries."""
        # Find uncached texts
        missing_indices = [i for i, t in enumerate(texts) if self._hash(t) not in self._cache]
        if not missing_indices:
            return [self._cache[self._hash(t)] for t in texts]

        missing_texts = [texts[i] for i in missing_indices]

        # 1. Try Hugging Face Inference API if enabled
        if settings.use_huggingface_api:
            vecs = self._hf_api_call(missing_texts)
            if vecs is not None and len(vecs) == len(missing_texts):
                for idx, vec in zip(missing_indices, vecs):
                    self._cache[self._hash(texts[idx])] = vec
                return [self._cache[self._hash(t)] for t in texts]

        # 2. Try local model
        model = self._get_model()
        if model is not None:
            try:
                vectors = model.encode(
                    missing_texts,
                    batch_size=settings.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for idx, vec in zip(missing_indices, vectors):
                    self._cache[self._hash(texts[idx])] = np.array(vec, dtype=np.float32)
                return [self._cache[self._hash(t)] for t in texts]
            except Exception as e:
                logger.warning(f"BGEEmbedder local batch encode error: {e}")

        # 3. Fallback to signed hashing vectorizer for each missing text
        for idx in missing_indices:
            self._cache[self._hash(texts[idx])] = self._fallback_encode(texts[idx])

        return [self._cache[self._hash(t)] for t in texts]

    @property
    def is_available(self) -> bool:
        """True if either Hugging Face Inference API or the local model is available."""
        if settings.use_huggingface_api:
            return True
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
                "Bypass local model."
            )
            self._available = False
        return self._model

    def _hf_api_call(self, texts: List[str]) -> List[np.ndarray] | None:
        """Call Hugging Face Inference API to get embeddings for a batch of texts."""
        import httpx
        import time

        headers = {}
        if settings.hf_api_token:
            headers["Authorization"] = f"Bearer {settings.hf_api_token}"

        url = f"https://api-inference.huggingface.co/models/{self._model_name}"

        try:
            with httpx.Client(timeout=15.0) as client:
                for attempt in range(3):
                    response = client.post(url, json={"inputs": texts}, headers=headers)
                    if response.status_code == 200:
                        res = response.json()
                        if not isinstance(res, list):
                            logger.warning(f"HF Inference returned non-list: {res}")
                            return None

                        # Handle HF feature-extraction format (can return 1D/2D/3D lists)
                        if all(isinstance(x, (int, float)) for x in res):
                            # Response is a single vector (1D list of floats)
                            vec = np.array(res, dtype=np.float32)
                            norm = np.linalg.norm(vec)
                            if norm > 0:
                                vec = vec / norm
                            return [vec]

                        vectors: List[np.ndarray] = []
                        for item in res:
                            if not isinstance(item, list):
                                return None

                            if len(item) > 0 and isinstance(item[0], list):
                                # It's a sequence of embeddings (seq_len, dim) -> mean pool
                                arr = np.array(item, dtype=np.float32)
                                vec = np.mean(arr, axis=0)
                            else:
                                vec = np.array(item, dtype=np.float32)

                            norm = np.linalg.norm(vec)
                            if norm > 0:
                                vec = vec / norm
                            vectors.append(vec)

                        if len(vectors) == len(texts):
                            return vectors
                        return None

                    elif response.status_code == 503:
                        # Model is currently loading on HF servers
                        try:
                            data = response.json()
                            est_time = data.get("estimated_time", 5.0)
                        except Exception:
                            est_time = 5.0
                        logger.info(f"HF model {self._model_name} is loading. Waiting {est_time:.1f}s...")
                        time.sleep(min(est_time, 5.0))
                    else:
                        logger.warning(f"HF Inference status {response.status_code}: {response.text}")
                        break
        except Exception as e:
            logger.warning(f"HF Inference API error: {e}")
        return None

    def _fallback_encode(self, text: str, dim: int = 384) -> np.ndarray:
        """Signed hashing vectorizer as a fast, zero-dependency, zero-RAM semantic proxy."""
        import hashlib
        vec = np.zeros(dim, dtype=np.float32)
        words = [w.lower() for w in text.split() if len(w) > 2]
        if not words:
            return vec
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1 if ((h >> 8) & 1) else -1
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text[:1000].encode()).hexdigest()

