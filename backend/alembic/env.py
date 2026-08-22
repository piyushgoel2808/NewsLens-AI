"""Alembic environment configuration for NewsLens-AI.

Uses synchronous pymysql driver for migrations (Alembic is synchronous).
DB URL is read from app Settings so it stays consistent with the application.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine, pool

from alembic import context

# Add backend/ to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.models import Base  # noqa: E402 — imports all models for autogenerate

# Alembic Config object (gives access to values within alembic.ini)
config = context.config

# Override sqlalchemy.url from app Settings (sync pymysql URL for migrations)
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database.sync_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a DB connection (generates SQL script)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = create_engine(
        config.get_main_option("sqlalchemy.url"),  # type: ignore[arg-type]
        poolclass=pool.NullPool,  # No pooling for migration runs
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
