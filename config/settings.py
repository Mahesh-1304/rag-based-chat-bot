# config/settings.py
"""
Configuration management for RAG Document Chatbot.
Uses environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ============ File Paths ============
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DOCS_DIR: Path = DATA_DIR / "raw_docs"
    PROCESSED_DOCS_DIR: Path = DATA_DIR / "processed_docs"
    VECTOR_STORE_DIR: Path = BASE_DIR / "embeddings" / "vector_store"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    MODEL_PATH: Optional[Path] = None

    # ============ Embedding Model ============
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_VECTOR_DIM: int = 384

    # ============ Chunking Parameters ============
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50

    # ============ Vector Store ============
    VECTOR_STORE_TYPE: str = "flat"   # "flat" | "hnsw" | "ivf"
    HNSW_M: int = 32
    IVF_NLIST: int = 100

    # ============ Retriever ============
    RETRIEVER_TOP_K: int = 3
    RETRIEVER_SCORE_THRESHOLD: float = 0.1

    # ============ Ollama LLM ============
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"   # Ollama OpenAI-compatible endpoint
    OLLAMA_MODEL: str = "llama3.2"                       # Change to any model you have pulled
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 512
    LLM_CONTEXT_LENGTH: int = 2048

    # ============ API Settings ============
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # ============ Caching ============
    ENABLE_CACHING: bool = True
    CACHE_MAX_SIZE: int = 1000
    REDIS_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"

    # ============ Logging ============
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ============ Feature Flags ============
    ENABLE_ASYNC: bool = True
    ENABLE_METRICS: bool = True
    ENABLE_VALIDATION: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **data):
        super().__init__(**data)
        self.PROCESSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()