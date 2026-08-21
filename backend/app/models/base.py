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


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the global async session factory, initializing if needed."""
    global _async_session_factory
    if _async_session_factory is None:
        from app.core.config import get_settings

        init_db(get_settings().database.async_url)
    assert _async_session_factory is not None
    return _async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield a managed async DB session."""
    factory = get_session_factory()
    async with factory() as session:
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
