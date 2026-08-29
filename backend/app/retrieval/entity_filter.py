"""Entity-Based Search Engine for structured entity and taxonomy queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.article import Article
from app.models.entity import ArticleEntity, ArticleTopic, Entity, Topic
from app.models.newspaper import Issue

logger = get_logger(__name__)


@dataclass
class EntitySearchResult:
    """An article result matched via entity and taxonomy filters."""

    article_id: int
    headline: str
    byline_author: str | None
    section: str | None
    article_type: str
    prominence_score: float
    entity_name: str
    entity_type: str
    mention_count: int
    salience_score: float
    newspaper_name: str
    issue_date: str
    pages: list[int]
    summary: str
    issue_id: int = 0
    bboxes: list[dict[str, Any]] = field(default_factory=list)


class EntitySearchEngine:
    """Retrieves articles based on structured entity mentions and topic taxonomy."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def search_by_entity(
        self,
        entity_name: str | None = None,
        entity_type: str | None = None,
        topic_name: str | None = None,
        min_salience: float = 0.0,
        newspaper_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        top_k: int = 10,
    ) -> list[EntitySearchResult]:
        """Search articles containing specified entities or topic taxonomy."""
        async with self._session_factory() as db:
            stmt = (
                select(ArticleEntity)
                .join(Entity, ArticleEntity.entity_id == Entity.id)
                .join(Article, ArticleEntity.article_id == Article.id)
                .join(Issue, Article.issue_id == Issue.id)
                .options(
                    selectinload(ArticleEntity.entity),
                    selectinload(ArticleEntity.article)
                    .selectinload(Article.issue)
                    .selectinload(Issue.newspaper),
                    selectinload(ArticleEntity.article).selectinload(Article.article_pages),
                )
            )

            if entity_name:
                stmt = stmt.where(Entity.name.ilike(f"%{entity_name}%"))
            if entity_type:
                stmt = stmt.where(Entity.type == entity_type)
            if min_salience > 0.0:
                stmt = stmt.where(ArticleEntity.salience_score >= min_salience)
            if newspaper_id:
                stmt = stmt.where(Issue.newspaper_id == newspaper_id)
            if date_from:
                stmt = stmt.where(Issue.issue_date >= date_from)
            if date_to:
                stmt = stmt.where(Issue.issue_date <= date_to)

            if topic_name:
                stmt = (
                    stmt.join(ArticleTopic, ArticleTopic.article_id == Article.id)
                    .join(Topic, ArticleTopic.topic_id == Topic.id)
                    .where(Topic.name.ilike(f"%{topic_name}%"))
                )

            stmt = stmt.order_by(
                desc(ArticleEntity.salience_score),
                desc(ArticleEntity.mention_count),
            ).limit(top_k)

            res = await db.execute(stmt)
            records = res.scalars().all()

            results: list[EntitySearchResult] = []
            for ae in records:
                art = ae.article
                if not art:
                    continue

                np_name = (
                    art.issue.newspaper.name if art.issue and art.issue.newspaper else "Daily News"
                )
                issue_date = str(art.issue.issue_date) if art.issue else ""
                pages_list = (
                    sorted({ap.page_number for ap in art.article_pages})
                    if art.article_pages
                    else []
                )

                bboxes_list: list[dict[str, Any]] = []
                if art.article_pages:
                    for ap in art.article_pages:
                        if ap.bbox_json:
                            if isinstance(ap.bbox_json, list):
                                bboxes_list.extend(ap.bbox_json)
                            elif isinstance(ap.bbox_json, dict):
                                bboxes_list.append(ap.bbox_json)

                results.append(
                    EntitySearchResult(
                        article_id=art.id,
                        headline=art.headline or "Untitled",
                        byline_author=art.byline_author,
                        section=art.section,
                        article_type=art.article_type,
                        prominence_score=art.prominence_score,
                        entity_name=ae.entity.name if ae.entity else "",
                        entity_type=ae.entity.type if ae.entity else "misc",
                        mention_count=ae.mention_count,
                        salience_score=ae.salience_score,
                        newspaper_name=np_name,
                        issue_date=issue_date,
                        pages=pages_list,
                        summary=art.summary or (art.full_text[:250] if art.full_text else ""),
                        issue_id=art.issue_id,
                        bboxes=bboxes_list,
                    )
                )

            logger.info(
                "Entity search executed",
                extra={
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "hits": len(results),
                },
            )

            return results

    async def expand_entity_cooccurrence_graph(
        self,
        entity_name: str,
        depth: int = 2,
        top_neighbors: int = 15,
        min_cooccurrence: int = 1,
    ) -> dict[str, Any]:
        """Multi-Hop Knowledge Graph Traversal: Expand entity co-occurrence graph across articles."""
        async with self._session_factory() as db:
            # 1. Locate focal entity (matching name)
            root_stmt = select(Entity).where(Entity.name.ilike(f"%{entity_name}%")).limit(5)
            root_res = await db.execute(root_stmt)
            root_entities = list(root_res.scalars().all())

            if not root_entities:
                return {
                    "root_entity": entity_name,
                    "nodes": [],
                    "edges": [],
                    "total_nodes": 0,
                    "total_edges": 0,
                }

            root_ids = {e.id for e in root_entities}
            visited_entity_ids = set(root_ids)
            nodes_map: dict[int, dict[str, Any]] = {
                e.id: {
                    "id": str(e.id),
                    "name": e.name,
                    "type": e.type,
                    "depth": 0,
                    "article_count": 0,
                }
                for e in root_entities
            }
            edges_map: dict[tuple[int, int], dict[str, Any]] = {}

            current_frontier = set(root_ids)

            # Traversal loop up to depth
            for current_depth in range(1, min(depth, 3) + 1):
                if not current_frontier:
                    break

                # Find all articles containing frontier entities
                art_stmt = (
                    select(ArticleEntity.article_id, ArticleEntity.entity_id)
                    .where(ArticleEntity.entity_id.in_(current_frontier))
                )
                art_res = await db.execute(art_stmt)
                frontier_articles: dict[int, set[int]] = {}
                for art_id, ent_id in art_res.all():
                    frontier_articles.setdefault(ent_id, set()).add(art_id)

                all_art_ids = {art_id for arts in frontier_articles.values() for art_id in arts}
                if not all_art_ids:
                    break

                # Find all co-occurring entities in those articles
                cooc_stmt = (
                    select(
                        ArticleEntity.article_id,
                        ArticleEntity.entity_id,
                        ArticleEntity.salience_score,
                        Entity.name,
                        Entity.type,
                    )
                    .join(Entity, ArticleEntity.entity_id == Entity.id)
                    .where(ArticleEntity.article_id.in_(all_art_ids))
                )
                cooc_res = await db.execute(cooc_stmt)
                rows = cooc_res.all()

                next_frontier: set[int] = set()

                # Group by article to build co-occurrence pairs
                art_to_entities: dict[int, list[tuple[int, str, str, float]]] = {}
                for art_id, ent_id, salience, name, ent_type in rows:
                    art_to_entities.setdefault(art_id, []).append((ent_id, name, ent_type, salience))
                    if ent_id not in nodes_map:
                        nodes_map[ent_id] = {
                            "id": str(ent_id),
                            "name": name,
                            "type": ent_type,
                            "depth": current_depth,
                            "article_count": 0,
                        }
                    nodes_map[ent_id]["article_count"] += 1

                for art_id, ent_list in art_to_entities.items():
                    for i in range(len(ent_list)):
                        for j in range(i + 1, len(ent_list)):
                            e1, n1, t1, s1 = ent_list[i]
                            e2, n2, t2, s2 = ent_list[j]

                            # Ensure edge touches the frontier at this step
                            if e1 not in current_frontier and e2 not in current_frontier:
                                continue

                            u, v = min(e1, e2), max(e1, e2)
                            pair_key = (u, v)
                            if pair_key not in edges_map:
                                edges_map[pair_key] = {
                                    "source": str(u),
                                    "target": str(v),
                                    "source_name": n1 if u == e1 else n2,
                                    "target_name": n2 if u == e1 else n1,
                                    "weight": 0,
                                    "articles": set(),
                                }
                            edges_map[pair_key]["weight"] += 1
                            edges_map[pair_key]["articles"].add(art_id)

                            if e1 not in visited_entity_ids:
                                next_frontier.add(e1)
                                visited_entity_ids.add(e1)
                            if e2 not in visited_entity_ids:
                                next_frontier.add(e2)
                                visited_entity_ids.add(e2)

                current_frontier = next_frontier

            # Filter edges by min_cooccurrence and prune top neighbors
            sorted_edges = sorted(
                edges_map.values(),
                key=lambda x: x["weight"],
                reverse=True,
            )
            filtered_edges = [
                {
                    "source": e["source"],
                    "target": e["target"],
                    "source_name": e["source_name"],
                    "target_name": e["target_name"],
                    "weight": e["weight"],
                    "article_count": len(e["articles"]),
                }
                for e in sorted_edges
                if e["weight"] >= min_cooccurrence
            ][:top_neighbors * 2]

            active_node_ids = {e["source"] for e in filtered_edges} | {e["target"] for e in filtered_edges}
            active_node_ids |= {str(i) for i in root_ids}

            filtered_nodes = [
                nodes_map[int(nid)]
                for nid in active_node_ids
                if int(nid) in nodes_map
            ]

            return {
                "root_entity": root_entities[0].name,
                "nodes": filtered_nodes,
                "edges": filtered_edges,
                "total_nodes": len(filtered_nodes),
                "total_edges": len(filtered_edges),
            }

