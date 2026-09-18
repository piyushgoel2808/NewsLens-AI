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
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)
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
    max_output_tokens: int | None = None
    is_reasoning_model: bool | None = None
    reasoning_headroom: int | None = None
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
    socket_path: str | None = None
    ssl: bool = False

    @property
    def async_url(self) -> str:
        """Async SQLAlchemy URL (aiomysql driver)."""
        if self.socket_path:
            return (
                f"mysql+aiomysql://{self.user}:{self.password}"
                f"@/{self.db}?unix_socket={self.socket_path}&charset=utf8mb4"
            )
        return (
            f"mysql+aiomysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}?charset=utf8mb4"
        )

    @property
    def sync_url(self) -> str:
        """Sync SQLAlchemy URL (pymysql driver) — for Alembic migrations."""
        if self.socket_path:
            return (
                f"mysql+pymysql://{self.user}:{self.password}"
                f"@/{self.db}?unix_socket={self.socket_path}&charset=utf8mb4"
            )
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
    collection_name_v2: str = "article_chunks_v2"
    https: bool = False


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
    "gemini_vision": ProviderConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        context_window=1048576,
        supports_vision=True,
        supports_tool_use=True,
    ),
    "gemini_flash": ProviderConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        context_window=1048576,
        supports_vision=True,
        supports_tool_use=True,
    ),
    "google_cloud_vision": ProviderConfig(
        provider="google_cloud_vision",
        supports_vision=True,
        supports_tool_use=False,
    ),
    "ollama_deepseek": ProviderConfig(
        provider="ollama",
        model="deepseek-r1:14b",
        base_url="http://localhost:11434",
        context_window=65536,
        supports_vision=False,
        supports_tool_use=True,
    ),
    "ollama_llama3": ProviderConfig(
        provider="ollama",
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        context_window=128000,
        supports_vision=False,
        supports_tool_use=True,
    ),
    "ollama_chat": ProviderConfig(
        provider="ollama",
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        context_window=128000,
        supports_vision=False,
        supports_tool_use=True,
    ),
    "ollama_qwen3vl": ProviderConfig(
        provider="ollama",
        model="qwen3-vl:latest",
        base_url="http://localhost:11434",
        context_window=32768,
        supports_vision=True,
        supports_tool_use=True,
    ),
    "ollama_vlm": ProviderConfig(
        provider="ollama",
        model="qwen2.5vl:7b",
        base_url="http://localhost:11434",
        context_window=32768,
        supports_vision=True,
        supports_tool_use=True,
    ),
    "ollama_embed": ProviderConfig(
        provider="ollama",
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        embedding_dim=768,
    ),
    "openrouter_gemma4_26b": ProviderConfig(
        provider="openrouter",
        model="google/gemma-4-26b-a4b-it:free",
        context_window=262144,
        supports_vision=True,
        supports_tool_use=True,
    ),
    "openrouter_nemotron": ProviderConfig(
        provider="openrouter",
        model="nvidia/nemotron-3.5-lightning:free",
        context_window=1000000,
        supports_vision=False,
        supports_tool_use=True,
    ),
    "nvidia_nemotron": ProviderConfig(
        provider="nvidia",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        context_window=128000,
        supports_vision=False,
        supports_tool_use=True,
    ),
    "nvidia_llama_vision": ProviderConfig(
        provider="nvidia",
        model="meta/llama-3.2-11b-vision-instruct",
        context_window=128000,
        supports_vision=True,
        supports_tool_use=True,
    ),
    "local_embed_bge": ProviderConfig(
        provider="local_sentence_transformers",
        model="BAAI/bge-m3",
        embedding_dim=1024,
    ),
    "docling_parser": ProviderConfig(
        provider="docling",
        model="doclaynet-rapidocr",
        supports_vision=True,
        supports_tool_use=False,
    ),
    "docling_cloud_parser": ProviderConfig(
        provider="docling_cloud",
        model="docling-ibm-cloud",
        supports_vision=True,
        supports_tool_use=False,
    ),
    "gemini_embedding": ProviderConfig(
        provider="gemini_embedding",
        model="gemini-embedding-001",
        supports_vision=False,
        supports_tool_use=False,
        embedding_dim=768,
    ),
}

DEFAULT_TASK_BINDINGS = {
    "layout_analysis": "docling_parser",
    "document_parser": "docling_parser",
    "ocr": "docling_parser",
    "embedding": "local_embed_bge",
    "query_planner": "ollama_llama3",
    "answerer": "ollama_llama3",
    "metadata_extraction": "ollama_llama3",
    "classification": "ollama_llama3",
    "article_segmentation": "ollama_deepseek",
    "visual_extraction": "ollama_qwen3vl",
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
    cors_allowed_origins: str = (
        "http://localhost:5173,http://localhost:5174,http://localhost:3000,"
        "http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:3000"
    )

    # --- Model config file path ---
    model_config_path: str = "model_config.yaml"

    # --- MySQL ---
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "newslens"
    mysql_password: str = "newslens_pass"
    mysql_db: str = "newslens"
    mysql_readonly_user: str | None = None
    mysql_readonly_password: str | None = None
    mysql_socket_path: str | None = None

    # --- Dynamic Tool Generation ---
    enable_dynamic_tools: bool = True
    dynamic_tool_timeout_seconds: int = 10
    dynamic_tool_max_memory_mb: int = 256
    dynamic_tool_max_retries: int = 2

    # --- Storage Backend (minio or gcs) ---
    storage_backend: str = "minio"
    gcs_project_id: str | None = None
    gcs_bucket_pages: str = "newslens-pages"
    gcs_bucket_originals: str = "newslens-originals"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "article_chunks"
    qdrant_collection_name_v2: str = "article_chunks_v2"
    qdrant_https: bool = False

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

    # --- API Keys & GCP Credentials (hosted providers — optional) ---
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    google_application_credentials: str | None = None
    gcp_service_account_key: str | None = None
    gcp_service_account_json: str | None = None
    gcp_project_id: str | None = None
    voyage_api_key: str | None = None
    hf_token: str | None = None
    huggingface_token: str | None = None

    openrouter_api_keys: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # --- Live Web / News Search Providers ---
    newsdata_api_key: str | None = None
    serper_api_key: str | None = None
    tavily_api_key: str | None = None

    # --- Hosted IBM Docling Cloud ---
    docling_api_key: str | None = None
    docling_service_url: str = "https://api.aws-c1.dcls.saas.ibm.com/20260918-1838-0663-6071-b702840c3d46"

    @field_validator(
        "groq_api_key",
        "gemini_api_key",
        "anthropic_api_key",
        "openai_api_key",
        "openrouter_api_keys",
        "nvidia_api_key",
        "newsdata_api_key",
        "serper_api_key",
        "tavily_api_key",
        "google_api_key",
        "google_application_credentials",
        "gcp_service_account_key",
        "gcp_service_account_json",
        "voyage_api_key",
        "qdrant_api_key",
        "hf_token",
        "huggingface_token",
        "docling_api_key",
        mode="before",
    )
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def get_openrouter_keys(self) -> list[str]:
        """Return parsed list of OpenRouter API keys for dual-account round-robin rotation."""
        if not self.openrouter_api_keys:
            return []
        return [k.strip() for k in self.openrouter_api_keys.split(",") if k.strip()]

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
            socket_path=self.mysql_socket_path,
        )

    @property
    def mysql_readonly_url(self) -> str:
        """Async connection string for read-only dynamic tool execution."""
        user = self.mysql_readonly_user or self.mysql_user
        pwd = self.mysql_readonly_password or self.mysql_password
        if self.mysql_socket_path:
            return (
                f"mysql+aiomysql://{user}:{pwd}"
                f"@/{self.mysql_db}?unix_socket={self.mysql_socket_path}&charset=utf8mb4"
            )
        return (
            f"mysql+aiomysql://{user}:{pwd}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )

    @property
    def qdrant(self) -> QdrantSettings:
        return QdrantSettings(
            host=self.qdrant_host,
            port=self.qdrant_port,
            api_key=self.qdrant_api_key,
            collection_name=self.qdrant_collection_name,
            collection_name_v2=self.qdrant_collection_name_v2,
            https=self.qdrant_https,
        )

    @property
    def bucket_pages(self) -> str:
        if (self.storage_backend or "").lower() == "gcs":
            return self.gcs_bucket_pages
        return self.minio_bucket_pages

    @property
    def bucket_originals(self) -> str:
        if (self.storage_backend or "").lower() == "gcs":
            return self.gcs_bucket_originals
        return self.minio_bucket_originals

    @property
    def minio(self) -> MinioSettings:
        return MinioSettings(
            endpoint=self.minio_endpoint,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            secure=self.minio_secure,
            bucket_pages=self.bucket_pages,
            bucket_originals=self.bucket_originals,
        )

    @property
    def redis(self) -> RedisSettings:
        return RedisSettings(url=self.redis_url)

    @property
    def cors_origins(self) -> list[str]:
        """Allowed origins for CORS middleware."""
        if self.app_debug:
            return ["*"]
        if not self.cors_allowed_origins:
            return ["*"]
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

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

    def save_model_config(self, config: ModelConfig) -> bool:
        """Serialize and persist ModelConfig to model_config.yaml on disk."""
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

        target_path: Path | None = None
        for p in candidate_paths:
            try:
                resolved = p.resolve()
                if resolved.is_file() and resolved.exists():
                    target_path = resolved
                    break
            except Exception:
                continue

        if not target_path:
            target_path = (root / "model_config.yaml").resolve()

        # Always update in-memory configuration so active session reflects new bindings
        object.__setattr__(self, "_model_config_data", config)

        try:
            raw_providers = {
                k: {
                    f: v
                    for f, v in p_cfg.model_dump().items()
                    if v is not None
                }
                for k, p_cfg in config.providers.items()
            }
            dump_data = {
                "providers": raw_providers,
                "task_bindings": config.task_bindings,
            }
            with target_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(dump_data, f, default_flow_style=False, sort_keys=False)

            return True
        except Exception as e:
            logger.warning(
                "Could not persist model_config.yaml to disk (check volume write permissions); in-memory bindings updated",
                extra={"target_path": str(target_path), "error": str(e)},
            )
            return False


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
