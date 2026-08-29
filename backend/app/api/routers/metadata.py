"""FastAPI router for Entity and Topic metadata discovery."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.base import get_db
from app.models.entity import ArticleEntity, Entity, Topic

router = APIRouter(prefix="/api", tags=["metadata"])


@router.get("/entities", summary="Search and list entities across all newspapers")
async def list_entities(
    q: str | None = Query(None, description="Search query for entity name"),
    type: str | None = Query(None, description="Filter by type (person, org, location, misc)"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List entities sorted alphabetically or matching search term."""
    stmt = select(Entity)
    if q:
        stmt = stmt.where(Entity.name.ilike(f"%{q}%"))
    if type:
        stmt = stmt.where(Entity.type == type)

    stmt = stmt.order_by(Entity.name).limit(limit)
    res = await db.execute(stmt)
    entities = res.scalars().all()

    return [
        {
            "id": e.id,
            "name": e.name,
            "type": e.type,
            "canonical_id": e.canonical_id,
        }
        for e in entities
    ]


@router.get("/topics", summary="List all topic taxonomy categories")
async def list_topics(
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List all hierarchical topics."""
    stmt = select(Topic).order_by(Topic.name)
    res = await db.execute(stmt)
    topics = res.scalars().all()

    return [
        {
            "id": t.id,
            "name": t.name,
            "taxonomy_path": t.taxonomy_path,
        }
        for t in topics
    ]


@router.get(
    "/articles/{article_id}/entities",
    summary="Get entities mentioned in a specific article",
)
async def get_article_entities(
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List all named entities linked to an article with mention counts and salience."""
    stmt = (
        select(ArticleEntity)
        .where(ArticleEntity.article_id == article_id)
        .options(selectinload(ArticleEntity.entity))
        .order_by(desc(ArticleEntity.salience_score))
    )
    res = await db.execute(stmt)
    art_entities = res.scalars().all()

    return [
        {
            "entity_id": ae.entity_id,
            "name": ae.entity.name if ae.entity else "",
            "type": ae.entity.type if ae.entity else "misc",
            "mention_count": ae.mention_count,
            "salience_score": ae.salience_score,
        }
        for ae in art_entities
    ]


@router.get(
    "/entities/{entity_name}/graph",
    summary="Get multi-hop co-occurrence knowledge graph for an entity",
)
async def get_entity_graph(
    entity_name: str,
    depth: int = Query(2, ge=1, le=3),
    top_neighbors: int = Query(15, ge=1, le=50),
    min_cooccurrence: int = Query(1, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return multi-hop knowledge graph topology connecting entities through shared newspaper stories."""
    from app.models.base import _async_session_factory
    from app.retrieval.entity_filter import EntitySearchEngine

    if _async_session_factory is None:
        return {"root_entity": entity_name, "nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}

    engine = EntitySearchEngine(session_factory=_async_session_factory)
    return await engine.expand_entity_cooccurrence_graph(
        entity_name=entity_name,
        depth=depth,
        top_neighbors=top_neighbors,
        min_cooccurrence=min_cooccurrence,
    )

