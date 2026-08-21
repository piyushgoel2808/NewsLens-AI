"""SQLAlchemy async engine, Base class, and session factory for NewsLens-AI.

All database access goes through get_db() (a FastAPI dependency).
The sync engine is used ONLY by Alembic migrations (not the application).
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Standard naming convention for Alembic autogenerate + constraint names
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# Module-level engine and session factory (initialized by lifespan)
_engine = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_db(database_url: str) -> None:
    """Initialize the async engine and session factory. Call once at startup."""
    global _engine, _async_session_factory
    _engine = create_async_engine(
        database_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    _async_session_factory = async_sessionmaker(
        _engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield a managed async DB session."""
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def close_db() -> None:
    """Close the async engine. Call at shutdown."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
