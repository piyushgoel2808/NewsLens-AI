"""Entity-Based Search Engine for structured entity and taxonomy queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.article import Article
from app.models.entity import ArticleEntity, ArticleTopic, Entity, Topic
from app.models.newspaper import Issue

logger = get_logger(__name__)

_HONORIFIC_PREFIX_RE = re.compile(
    r"^(?:chief\s+minister|prime\s+minister|union\s+minister|minister|president|governor|"
    r"senator|justice|judge|general|colonel|dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|shri|smt\.?|cm|pm)\s+",
    re.IGNORECASE,
)


def _clean_entity_query(entity_name: str) -> list[str]:
    """Extract candidate entity search terms after stripping titles and splitting 'or'."""
    raw = (entity_name or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"\s+or\s+", raw, flags=re.I) if p.strip()]
    candidates: list[str] = []
    for p in parts:
        cleaned = _HONORIFIC_PREFIX_RE.sub("", p).strip("?:!.,\"' ")
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
        if p not in candidates:
            candidates.append(p)
        tokens = [t for t in re.findall(r"[A-Za-z0-9]{4,}", cleaned) if len(t) >= 4]
        if len(tokens) >= 2 and tokens[-1] not in candidates:
            candidates.append(tokens[-1])
    return candidates


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
    """Retrieves articles based on structured entity mentions and topic taxonomy with full-text fallback."""

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
        """Search articles containing specified entities or topic taxonomy, falling back to article text if NER table is sparse."""
        candidates = _clean_entity_query(entity_name or "")
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

            if candidates:
                stmt = stmt.where(or_(*(Entity.name.ilike(f"%{c}%") for c in candidates)))
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
            seen_art_ids: set[int] = set()
            for ae in records:
                art = ae.article
                if not art or art.id in seen_art_ids:
                    continue
                seen_art_ids.add(art.id)

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
                        entity_name=ae.entity.name if ae.entity else (entity_name or ""),
                        entity_type=ae.entity.type if ae.entity else (entity_type or "person"),
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

            # Fallback: If ArticleEntity join returned 0 rows, search Article headline/summary/full_text directly
            if not results and candidates:
                primary_terms = candidates[:2]
                text_conds = []
                for term in primary_terms:
                    text_conds.append(Article.headline.ilike(f"%{term}%"))
                    text_conds.append(Article.summary.ilike(f"%{term}%"))
                    text_conds.append(Article.full_text.ilike(f"%{term}%"))

                art_stmt = (
                    select(Article)
                    .join(Issue, Article.issue_id == Issue.id)
                    .where(or_(*text_conds))
                    .options(
                        selectinload(Article.issue).selectinload(Issue.newspaper),
                        selectinload(Article.article_pages),
                    )
                )
                if newspaper_id:
                    art_stmt = art_stmt.where(Issue.newspaper_id == newspaper_id)
                if date_from:
                    art_stmt = art_stmt.where(Issue.issue_date >= date_from)
                if date_to:
                    art_stmt = art_stmt.where(Issue.issue_date <= date_to)

                art_stmt = art_stmt.order_by(desc(Article.prominence_score)).limit(top_k)
                art_res = await db.execute(art_stmt)
                for art in art_res.scalars().all():
                    if art.id in seen_art_ids:
                        continue
                    seen_art_ids.add(art.id)
                    np_name = (
                        art.issue.newspaper.name if art.issue and art.issue.newspaper else "Daily News"
                    )
                    issue_date = str(art.issue.issue_date) if art.issue else ""
                    pages_list = (
                        sorted({ap.page_number for ap in art.article_pages})
                        if art.article_pages
                        else [1]
                    )
                    bboxes_list = []
                    if art.article_pages:
                        for ap in art.article_pages:
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
                            article_type=art.article_type or "news",
                            prominence_score=art.prominence_score or 0.5,
                            entity_name=candidates[0],
                            entity_type=entity_type or "entity",
                            mention_count=1,
                            salience_score=0.75,
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

            for current_depth in range(1, min(depth, 3) + 1):
                if not current_frontier:
                    break

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
                    if ent_id not in visited_entity_ids:
                        next_frontier.add(ent_id)

                for _art_id, ents in art_to_entities.items():
                    for i in range(len(ents)):
                        for j in range(i + 1, len(ents)):
                            id_a, _name_a, _, sal_a = ents[i]
                            id_b, _name_b, _, sal_b = ents[j]
                            if id_a == id_b:
                                continue
                            pair = (min(id_a, id_b), max(id_a, id_b))
                            if pair not in edges_map:
                                edges_map[pair] = {
                                    "source": str(pair[0]),
                                    "target": str(pair[1]),
                                    "weight": 0,
                                    "avg_salience": 0.0,
                                }
                            edges_map[pair]["weight"] += 1
                            edges_map[pair]["avg_salience"] = round(
                                (edges_map[pair]["avg_salience"] + (sal_a + sal_b) / 2.0) / 2.0,
                                3,
                            )

                visited_entity_ids.update(next_frontier)
                current_frontier = set(list(next_frontier)[:top_neighbors])

            filtered_edges = [
                edge for edge in edges_map.values() if edge["weight"] >= min_cooccurrence
            ]
            filtered_edges.sort(key=lambda x: x["weight"], reverse=True)
            filtered_edges = filtered_edges[: top_neighbors * 2]

            active_node_ids = {str(rid) for rid in root_ids}
            for edge in filtered_edges:
                active_node_ids.add(edge["source"])
                active_node_ids.add(edge["target"])

            final_nodes = [
                n for n in nodes_map.values() if n["id"] in active_node_ids
            ][: top_neighbors + len(root_ids)]

            return {
                "root_entity": entity_name,
                "nodes": final_nodes,
                "edges": filtered_edges,
                "total_nodes": len(final_nodes),
                "total_edges": len(filtered_edges),
            }
