"""
Centralised application settings loaded from environment / .env file.

All configurable parameters live here. Never hardcode values elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Groq LLM ─────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_api_keys: str = Field(default="", validation_alias="GROQ_API_KEYS")
    groq_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL")
    fallback_models: str = Field(
        default="llama-3.3-70b-versatile,llama-3.1-8b-instant,qwen/qwen3-32b,qwen/qwen3.6-27b",
        validation_alias="FALLBACK_MODELS",
    )
    temperature: float = Field(default=0.0, validation_alias="TEMPERATURE")
    max_tokens: int = Field(default=1024, validation_alias="MAX_TOKENS")
    max_retries: int = Field(default=3, validation_alias="MAX_RETRIES")
    timeout_seconds: float = Field(default=30.0, validation_alias="TIMEOUT_SECONDS")

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=400, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, validation_alias="CHUNK_OVERLAP")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    max_retrieved_papers: int = Field(default=30, validation_alias="MAX_RETRIEVED_PAPERS")
    max_reranked_papers: int = Field(default=10, validation_alias="MAX_RERANKED_PAPERS")
    retrieval_timeout: float = Field(default=12.0, validation_alias="RETRIEVAL_TIMEOUT")

    # Source feature flags
    enable_openalex: bool = Field(default=True, validation_alias="ENABLE_OPENALEX")
    enable_arxiv: bool = Field(default=True, validation_alias="ENABLE_ARXIV")
    enable_crossref: bool = Field(default=True, validation_alias="ENABLE_CROSSREF")
    enable_core: bool = Field(default=False, validation_alias="ENABLE_CORE")
    core_api_key: str = Field(default="", validation_alias="CORE_API_KEY")

    # ── Embedding ─────────────────────────────────────────────────────────────
    enable_embedding: bool = Field(default=True, validation_alias="ENABLE_EMBEDDING")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", validation_alias="EMBEDDING_MODEL"
    )
    embedding_batch_size: int = Field(default=32, validation_alias="EMBEDDING_BATCH_SIZE")

    # ── Reranking ─────────────────────────────────────────────────────────────
    enable_reranking: bool = Field(default=True, validation_alias="ENABLE_RERANKING")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="RERANKER_MODEL",
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    upload_dir: str = Field(default="uploads", validation_alias="UPLOAD_DIR")
    report_dir: str = Field(default="reports", validation_alias="REPORT_DIR")

    # ── Properties ───────────────────────────────────────────────────────────
    @property
    def upload_path(self) -> Path:
        p = BASE_DIR / self.upload_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def report_path(self) -> Path:
        p = BASE_DIR / self.report_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_api_keys(self) -> list[str]:
        """Return deduplicated list of Groq API keys."""
        keys: list[str] = []
        for raw in (self.groq_api_keys, self.groq_api_key):
            for k in raw.split(","):
                k = k.strip()
                if k and k not in keys:
                    keys.append(k)
        return keys

    def get_fallback_models(self) -> list[str]:
        """Return ordered list of Groq model names to try."""
        models = [m.strip() for m in self.fallback_models.split(",") if m.strip()]
        if not models:
            models = [self.groq_model]
        if self.groq_model not in models:
            models.insert(0, self.groq_model)
        return models


# Singleton — import this everywhere
settings = Settings()
