"""
Abstract base class for embedding services.

Implementations must encode text to normalised float vectors.
The architecture allows swapping BGE → OpenAI → Cohere embeddings
without changing any application logic — only swap the concrete class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    """Abstract embedding service."""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Return a normalised embedding vector for *text*."""
        ...

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Return normalised embedding vectors for a list of texts."""
        ...

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two normalised vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
