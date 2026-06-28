import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory of the project
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Groq API settings
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_api_keys: str = Field(default="", validation_alias="GROQ_API_KEYS")
    groq_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL")
    fallback_models: str = Field(
        default="llama-3.3-70b-versatile,llama-3.1-8b-instant,qwen/qwen3-32b,qwen/qwen3.6-27b",
        validation_alias="FALLBACK_MODELS"
    )
    temperature: float = Field(default=0.0, validation_alias="TEMPERATURE")
    max_tokens: int = Field(default=1024, validation_alias="MAX_TOKENS")

    # System parameters
    chunk_size: int = Field(default=400, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, validation_alias="CHUNK_OVERLAP")
    max_retries: int = Field(default=3, validation_alias="MAX_RETRIES")
    timeout_seconds: float = Field(default=30.0, validation_alias="TIMEOUT_SECONDS")

    # Directory Paths (Relative to BASE_DIR or Absolute)
    upload_dir: str = Field(default="uploads", validation_alias="UPLOAD_DIR")
    report_dir: str = Field(default="reports", validation_alias="REPORT_DIR")

    @property
    def upload_path(self) -> Path:
        path = BASE_DIR / self.upload_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def report_path(self) -> Path:
        path = BASE_DIR / self.report_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

settings = Settings()
