"""Central configuration for NewsLens-AI backend.

All configuration is loaded from environment variables + optional .env file
via Pydantic Settings. The model_config.yaml is loaded and merged in at
startup via load_model_config().

Usage:
    from app.core.config import get_settings
    settings = get_settings()
    model_cfg = settings.load_model_config()
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Sub-models for model_config.yaml
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel):
    """Configuration for a single named provider instance."""

    provider: str
    model: str | None = None
    base_url: str | None = None
    context_window: int | None = None
    supports_vision: bool = False
    supports_tool_use: bool = False
    embedding_dim: int | None = None
    lang: str | None = None  # OCR language string (e.g. "eng+hin")


class ModelConfig(BaseModel):
    """Parsed content of model_config.yaml."""

    providers: dict[str, ProviderConfig] = {}
    task_bindings: dict[str, str] = {}

    def get_provider_for_task(self, task: str) -> ProviderConfig:
        """Resolve a task name to its ProviderConfig.

        Raises:
            ValueError: If task has no binding or the binding points to an
                        undefined provider.
        """
        binding = self.task_bindings.get(task)
        if not binding:
            raise ValueError(f"No task binding found for task: {task!r}")
        provider = self.providers.get(binding)
        if not provider:
            raise ValueError(
                f"Task {task!r} is bound to {binding!r} "
                f"but that provider is not defined in the providers section."
            )
        return provider


# ---------------------------------------------------------------------------
# Typed sub-settings (exposed as properties on Settings)
# ---------------------------------------------------------------------------


class DatabaseSettings(BaseModel):
    """MySQL connection settings."""

    host: str = "localhost"
    port: int = 3306
    user: str = "newslens"
    password: str = "newslens_pass"
    db: str = "newslens"

    @property
    def async_url(self) -> str:
        """Async SQLAlchemy URL (aiomysql driver)."""
        return (
            f"mysql+aiomysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}?charset=utf8mb4"
        )

    @property
    def sync_url(self) -> str:
        """Sync SQLAlchemy URL (pymysql driver) — for Alembic migrations."""
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}?charset=utf8mb4"
        )


class QdrantSettings(BaseModel):
    """Qdrant vector store settings."""

    host: str = "localhost"
    port: int = 6333
    api_key: str | None = None
    collection_name: str = "article_chunks"


class MinioSettings(BaseModel):
    """MinIO / S3-compatible object storage settings."""

    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin123"
    secure: bool = False
    bucket_pages: str = "newslens-pages"
    bucket_originals: str = "newslens-originals"


class RedisSettings(BaseModel):
    """Redis settings."""

    url: str = "redis://localhost:6379/0"


# ---------------------------------------------------------------------------
# Main Settings class
# ---------------------------------------------------------------------------


DEFAULT_PROVIDERS = {
    "ollama_chat": ProviderConfig(
        provider="ollama",
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        supports_tool_use=True,
    ),
    "ollama_vlm": ProviderConfig(
        provider="ollama",
        model="qwen2.5vl:7b",
        base_url="http://localhost:11434",
        supports_vision=True,
        supports_tool_use=True,
    ),
    "ollama_embed": ProviderConfig(
        provider="ollama",
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        embedding_dim=768,
    ),
    "local_embed_bge": ProviderConfig(
        provider="local_sentence_transformers",
        model="BAAI/bge-m3",
        embedding_dim=1024,
    ),
    "tesseract_ocr": ProviderConfig(
        provider="tesseract",
        lang="eng+hin",
    ),
    "mineru_parser": ProviderConfig(
        provider="mineru",
        lang="en+hi",
        supports_vision=True,
    ),
}

DEFAULT_TASK_BINDINGS = {
    "query_planner": "ollama_chat",
    "answerer": "ollama_chat",
    "layout_analysis": "mineru_parser",
    "document_parser": "mineru_parser",
    "article_segmentation": "ollama_chat",
    "metadata_extraction": "ollama_chat",
    "classification": "ollama_chat",
    "embedding": "local_embed_bge",
    "ocr": "mineru_parser",
}


def find_project_root() -> Path:
    """Locate the repository root directory by walking upward looking for project markers."""
    curr = Path.cwd().resolve()
    for p in [curr, *curr.parents]:
        if (p / "model_config.yaml").exists() or (p / "docker-compose.local.yml").exists():
            return p
    file_p = Path(__file__).resolve().parent
    for p in [file_p, *file_p.parents]:
        if (p / "model_config.yaml").exists() or (p / "docker-compose.local.yml").exists():
            return p
    return Path.cwd().resolve()


class Settings(BaseSettings):
    """NewsLens-AI application settings.

    Values are loaded from (in order of precedence):
    1. Environment variables
    2. .env file (if present in the working directory)
    3. Default values defined below
    """

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    app_name: str = "NewsLens-AI"
    app_debug: bool = False
    app_log_level: str = "INFO"
    app_secret_key: str = "change-me-in-production"
    testing: bool = False

    # --- Model config file path ---
    model_config_path: str = "model_config.yaml"

    # --- MySQL ---
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "newslens"
    mysql_password: str = "newslens_pass"
    mysql_db: str = "newslens"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "article_chunks"

    # --- MinIO ---
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_secure: bool = False
    minio_bucket_pages: str = "newslens-pages"
    minio_bucket_originals: str = "newslens-originals"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"

    # --- API Keys (hosted providers — optional) ---
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    voyage_api_key: str | None = None
    hf_token: str | None = None
    huggingface_token: str | None = None

    @field_validator(
        "groq_api_key",
        "gemini_api_key",
        "anthropic_api_key",
        "openai_api_key",
        "google_api_key",
        "voyage_api_key",
        "qdrant_api_key",
        "hf_token",
        "huggingface_token",
        mode="before",
    )
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    # --- Internal: cached model config ---
    _model_config_data: ModelConfig | None = None

    # --- Typed property accessors ---

    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings(
            host=self.mysql_host,
            port=self.mysql_port,
            user=self.mysql_user,
            password=self.mysql_password,
            db=self.mysql_db,
        )

    @property
    def qdrant(self) -> QdrantSettings:
        return QdrantSettings(
            host=self.qdrant_host,
            port=self.qdrant_port,
            api_key=self.qdrant_api_key,
            collection_name=self.qdrant_collection_name,
        )

    @property
    def minio(self) -> MinioSettings:
        return MinioSettings(
            endpoint=self.minio_endpoint,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            secure=self.minio_secure,
            bucket_pages=self.minio_bucket_pages,
            bucket_originals=self.minio_bucket_originals,
        )

    @property
    def redis(self) -> RedisSettings:
        return RedisSettings(url=self.redis_url)

    def load_model_config(self) -> ModelConfig:
        """Load and parse model_config.yaml with robust root discovery and default fallbacks."""
        if self._model_config_data is not None:
            return self._model_config_data

        candidate_paths: list[Path] = []
        if self.model_config_path:
            candidate_paths.append(Path(self.model_config_path))

        root = find_project_root()
        candidate_paths.extend(
            [
                root / "model_config.yaml",
                Path.cwd() / "model_config.yaml",
                Path.cwd() / "../model_config.yaml",
            ]
        )

        found_path: Path | None = None
        for p in candidate_paths:
            try:
                resolved = p.resolve()
                if resolved.is_file() and resolved.exists():
                    found_path = resolved
                    break
            except Exception:
                continue

        if found_path:
            with found_path.open(encoding="utf-8") as f:
                raw: dict[str, Any] = yaml.safe_load(f) or {}
            providers = {
                k: ProviderConfig(**v)
                for k, v in raw.get("providers", {}).items()
                if isinstance(v, dict)
            }
            task_bindings = dict(raw.get("task_bindings", {}))
        else:
            providers = dict(DEFAULT_PROVIDERS)
            task_bindings = dict(DEFAULT_TASK_BINDINGS)

        config = ModelConfig(
            providers=providers,
            task_bindings=task_bindings,
        )
        object.__setattr__(self, "_model_config_data", config)
        return config


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    s = Settings()
    token = s.hf_token or s.huggingface_token
    if token:
        import os

        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return s
