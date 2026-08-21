"""Tests for the config loading system."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.core.config import ModelConfig, ProviderConfig, Settings, get_settings


class TestSettings:
    """Test Settings loading and typed property accessors."""

    def test_get_settings_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_database_async_url_uses_aiomysql(self) -> None:
        settings = Settings(mysql_host="testhost", mysql_db="testdb")
        assert "aiomysql" in settings.database.async_url
        assert "testhost" in settings.database.async_url
        assert "testdb" in settings.database.async_url

    def test_database_sync_url_uses_pymysql(self) -> None:
        settings = Settings(mysql_host="testhost", mysql_db="testdb")
        assert "pymysql" in settings.database.sync_url

    def test_qdrant_settings_accessible(self) -> None:
        settings = Settings(qdrant_host="qdrant-server", qdrant_port=6333)
        assert settings.qdrant.host == "qdrant-server"
        assert settings.qdrant.port == 6333

    def test_minio_settings_accessible(self) -> None:
        settings = Settings(minio_endpoint="minio:9000")
        assert settings.minio.endpoint == "minio:9000"

    def test_redis_settings_accessible(self) -> None:
        settings = Settings(redis_url="redis://myredis:6379/1")
        assert settings.redis.url == "redis://myredis:6379/1"

    def test_optional_api_keys_default_to_none(self) -> None:
        settings = Settings()
        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None


class TestModelConfig:
    """Test model_config.yaml parsing."""

    def test_empty_model_config_returns_empty_model(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "model_config.yaml"
        cfg_file.write_text("{}")
        settings = Settings(model_config_path=str(cfg_file))
        cfg = settings.load_model_config()
        assert isinstance(cfg, ModelConfig)
        assert cfg.providers == {}
        assert cfg.task_bindings == {}

    def test_missing_config_file_returns_empty_model(self, tmp_path: Path) -> None:
        settings = Settings(model_config_path=str(tmp_path / "nonexistent.yaml"))
        cfg = settings.load_model_config()
        assert isinstance(cfg, ModelConfig)

    def test_providers_parsed_correctly(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "model_config.yaml"
        cfg_file.write_text(textwrap.dedent("""
            providers:
              my_ollama:
                provider: ollama
                model: llama3.2:3b
                supports_vision: false
                supports_tool_use: true
            task_bindings:
              query_planner: my_ollama
        """))
        settings = Settings(model_config_path=str(cfg_file))
        cfg = settings.load_model_config()

        assert "my_ollama" in cfg.providers
        provider = cfg.providers["my_ollama"]
        assert isinstance(provider, ProviderConfig)
        assert provider.provider == "ollama"
        assert provider.model == "llama3.2:3b"
        assert provider.supports_vision is False
        assert provider.supports_tool_use is True

    def test_task_bindings_parsed(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "model_config.yaml"
        cfg_file.write_text(textwrap.dedent("""
            providers:
              my_ollama:
                provider: ollama
                model: llama3.2:3b
            task_bindings:
              answerer: my_ollama
              embedding: my_ollama
        """))
        settings = Settings(model_config_path=str(cfg_file))
        cfg = settings.load_model_config()
        assert cfg.task_bindings["answerer"] == "my_ollama"
        assert cfg.task_bindings["embedding"] == "my_ollama"

    def test_get_provider_for_task_resolves_correctly(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "model_config.yaml"
        cfg_file.write_text(textwrap.dedent("""
            providers:
              my_provider:
                provider: ollama
                model: llama3.2:3b
            task_bindings:
              query_planner: my_provider
        """))
        settings = Settings(model_config_path=str(cfg_file))
        cfg = settings.load_model_config()
        provider_cfg = cfg.get_provider_for_task("query_planner")
        assert provider_cfg.provider == "ollama"

    def test_get_provider_for_unknown_task_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "model_config.yaml"
        cfg_file.write_text("{}")
        settings = Settings(model_config_path=str(cfg_file))
        cfg = settings.load_model_config()
        with pytest.raises(ValueError, match="No task binding"):
            cfg.get_provider_for_task("nonexistent_task")

    def test_get_provider_for_undefined_provider_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "model_config.yaml"
        cfg_file.write_text(textwrap.dedent("""
            providers: {}
            task_bindings:
              query_planner: undefined_provider
        """))
        settings = Settings(model_config_path=str(cfg_file))
        cfg = settings.load_model_config()
        with pytest.raises(ValueError, match="not defined"):
            cfg.get_provider_for_task("query_planner")

    def test_model_config_cached_after_first_load(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "model_config.yaml"
        cfg_file.write_text("{}")
        settings = Settings(model_config_path=str(cfg_file))
        cfg1 = settings.load_model_config()
        cfg2 = settings.load_model_config()
        assert cfg1 is cfg2  # Same object — cached
