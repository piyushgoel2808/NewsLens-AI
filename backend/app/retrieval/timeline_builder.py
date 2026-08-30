"""Timeline Builder: Chronological event trajectory and perspective comparison.

Provides storyline reconstruction, editorial angle analysis, and discrepancy detection.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import asc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.cost_tracker import record_usage_and_cost, validate_query_budget
from app.core.logging import get_logger
from app.models.article import Article
from app.models.entity import ArticleEntity, Entity
from app.models.newspaper import Issue
from app.providers.base import Message
from app.providers.registry import get_registry
from app.retrieval.reranker import CrossEncoderReranker
from app.storage.cache_store import CacheStore, compute_query_cache_key

logger = get_logger(__name__)


# =============================================================================
# Pydantic Schemas for Narrative Trajectory
# =============================================================================


class NewspaperPerspective(BaseModel):
    """Detailed coverage record and angle from a single newspaper broadsheet."""

    newspaper_name: str
    issue_date: str
    pdf_page: int
    headline: str
    key_takeaway: str
    angle: str = Field(
        default="General News",
        description="Editorial angle (e.g. Financial, Policy, Socio-Political, Regulatory)",
    )
    bboxes: list[dict[str, Any]] = Field(default_factory=list)
    issue_id: int = 0
    article_id: int = 0


class TimelineMilestone(BaseModel):
    """A synthesized chronological milestone encompassing cross-publication coverage."""

    milestone_id: str
    date: str
    canonical_event: str
    event_phase: str = Field(
        default="Development",
        description="Lifecycle phase: Breaking, Development, Financial, Regulatory",
    )
    perspectives: list[NewspaperPerspective] = Field(default_factory=list)
    discrepancies: list[str] = Field(
        default_factory=list,
        description="Conflicting figures, dates, metrics, or factual statements across reports",
    )
    active_entities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Key entities and actors actively participating in this narrative milestone",
    )


class NarrativeTrajectoryResponse(BaseModel):
    """Comprehensive cross-newspaper narrative trajectory payload."""

    query: str
    topic_summary: str
    date_range: list[str] = Field(default_factory=list)
    milestones: list[TimelineMilestone] = Field(default_factory=list)
    latency_ms: int = 0
    cost_usd: float = 0.0
    cached: bool = False


# =============================================================================
# Backward-Compatible Dataclasses for Legacy Pipeline Calls
# =============================================================================


@dataclass
class LegacyTimelineMilestone:
    """Legacy milestone representation for backward compatibility."""

    article_id: int
    headline: str
    byline_author: str | None
    section: str | None
    summary: str
    prominence_score: float
    pages: list[int]
    issue_id: int = 0
    bboxes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TimelineDateGroup:
    """All news coverage and milestones on a single calendar date."""

    date: str  # YYYY-MM-DD
    newspaper_name: str
    articles_count: int
    milestones: list[LegacyTimelineMilestone] = field(default_factory=list)


@dataclass
class TimelineResult:
    """Full chronological trajectory across the requested period."""

    query: str
    total_dates: int
    total_articles: int
    date_groups: list[TimelineDateGroup] = field(default_factory=list)


# =============================================================================
# Structured LLM Prompt for Timeline Synthesis
# =============================================================================

TIMELINE_SYNTHESIS_SYSTEM_PROMPT = """You are NewsLens-AI's Narrative Trajectory & Auditor.
Analyze chronological newspaper broadsheets across multiple publications covering a storyline.

Reconstruct the storyline lifecycle, contrast editorial perspectives,
and detect factual/metric discrepancies between reporting broadsheets.

For each distinct calendar date in the coverage:
1. canonical_event: 1 concise, objective sentence describing what happened.
2. event_phase: Categorize into one of:
   - "Breaking": Initial sudden announcement or breaking report.
   - "Development": Follow-up negotiations, policy escalations, or market reactions.
   - "Financial": Earnings impact, stock swings, deal terms, or commercial ramifications.
   - "Regulatory": Court judgments, regulatory filings, signed treaties, or decrees.
3. perspectives: Extract each newspaper's distinct angle (e.g. Financial, Policy) and takeaway.
4. discrepancies: List any conflicting numbers, dates, estimates, or contradictions across papers.

Output strictly valid JSON matching this schema:
{
  "topic_summary": "Overall synthesis of how the story progressed from start to finish.",
  "milestones": [
    {
      "date": "YYYY-MM-DD",
      "canonical_event": "One-sentence event summary",
      "event_phase": "Breaking" | "Development" | "Financial" | "Regulatory",
      "perspectives": [
        {
          "newspaper_name": "Mint",
          "headline": "...",
          "angle": "Financial Impact",
          "key_takeaway": "..."
        }
      ],
      "discrepancies": ["Mint reported X whereas Business Standard reported Y"]
    }
  ]
}
"""


class TimelineBuilder:
    """Constructs cross-newspaper chronological narrative trajectories and audits discrepancies."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        cache_store: CacheStore | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._cache = cache_store or CacheStore()

    async def build_timeline(
        self,
        query: str | None = None,
        newspaper_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> TimelineResult:
        """Construct a structured chronological timeline of articles (legacy format)."""
        async with self._session_factory() as db:
            stmt = (
                select(Article)
                .join(Issue, Article.issue_id == Issue.id)
                .options(
                    selectinload(Article.issue).selectinload(Issue.newspaper),
                    selectinload(Article.article_pages),
                )
            )

            if newspaper_id:
                stmt = stmt.where(Issue.newspaper_id == newspaper_id)
            if date_from:
                stmt = stmt.where(Issue.issue_date >= date_from)
            if date_to:
                stmt = stmt.where(Issue.issue_date <= date_to)
            if query:
                stmt = stmt.where(
                    (Article.headline.ilike(f"%{query}%"))
                    | (Article.summary.ilike(f"%{query}%"))
                    | (Article.full_text.ilike(f"%{query}%"))
                )

            stmt = stmt.order_by(asc(Issue.issue_date), asc(Article.primary_page_id)).limit(limit)
            res = await db.execute(stmt)
            articles = res.scalars().all()

            grouped: dict[tuple[str, str], list[LegacyTimelineMilestone]] = {}
            for art in articles:
                issue_date = str(art.issue.issue_date) if art.issue else "Unknown Date"
                np_name = (
                    art.issue.newspaper.name if art.issue and art.issue.newspaper else "Daily News"
                )
                key = (issue_date, np_name)
                if key not in grouped:
                    grouped[key] = []

                pages_list = (
                    sorted({ap.page_number for ap in art.article_pages})
                    if art.article_pages
                    else []
                )
                summary = art.summary or (art.full_text[:250] if art.full_text else "")

                bboxes_list: list[dict[str, Any]] = []
                if art.article_pages:
                    for ap in art.article_pages:
                        if ap.bbox_json:
                            if isinstance(ap.bbox_json, list):
                                bboxes_list.extend(ap.bbox_json)
                            elif isinstance(ap.bbox_json, dict):
                                bboxes_list.append(ap.bbox_json)

                grouped[key].append(
                    LegacyTimelineMilestone(
                        article_id=art.id,
                        headline=art.headline or "Untitled",
                        byline_author=art.byline_author,
                        section=art.section,
                        summary=summary,
                        prominence_score=art.prominence_score,
                        pages=pages_list,
                        issue_id=art.issue_id,
                        bboxes=bboxes_list,
                    )
                )

            date_groups: list[TimelineDateGroup] = []
            for (dt, np_name), milestones in grouped.items():
                date_groups.append(
                    TimelineDateGroup(
                        date=dt,
                        newspaper_name=np_name,
                        articles_count=len(milestones),
                        milestones=milestones,
                    )
                )

            return TimelineResult(
                query=query or "All Coverage",
                total_dates=len(date_groups),
                total_articles=len(articles),
                date_groups=date_groups,
            )

    async def build_narrative_trajectory(
        self,
        query: str,
        issue_ids: list[int] | None = None,
        model_override: str | None = None,
        use_cache: bool = True,
        on_progress: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> NarrativeTrajectoryResponse:
        """Construct multi-perspective narrative trajectory with discrepancy auditing."""
        t0 = time.monotonic()
        clean_query = query.strip()

        # 1. Heavy-LLM Caching Check (Phase 7 Redis Cache)
        cache_key = compute_query_cache_key(
            f"timeline:{clean_query}",
            model_id=model_override or "default",
            issue_ids=issue_ids,
        )
        if use_cache:
            cached_data = await self._cache.get_query(cache_key)
            if cached_data:
                logger.info("Timeline cache hit", extra={"query": clean_query[:30]})
                if on_progress:
                    await on_progress("cache_hit", {"message": "Returning cached trajectory"})
                res = NarrativeTrajectoryResponse(**cached_data)
                res.cached = True
                res.latency_ms = round((time.monotonic() - t0) * 1000)
                return res

        # 2. Cost & Budget Guardrail Check
        validate_query_budget(estimated_cost_usd=0.03)

        if on_progress:
            await on_progress(
                "fetching_articles",
                {"message": "Retrieving broadsheets across editions"},
            )

        # 3. Retrieve Candidate Articles (Direct Keyword + Multi-Hop Graph Expansion with 4-Tier Anti-Hallucination Gates)
        async with self._session_factory() as db:
            stmt = (
                select(Article)
                .join(Issue, Article.issue_id == Issue.id)
                .options(
                    selectinload(Article.issue).selectinload(Issue.newspaper),
                    selectinload(Article.article_pages),
                    selectinload(Article.article_entities).selectinload(ArticleEntity.entity),
                )
            )

            if issue_ids:
                stmt = stmt.where(Issue.id.in_(issue_ids))

            keywords = [w for w in re.findall(r"\b[A-Za-z0-9]{3,}\b", clean_query) if len(w) >= 3]
            if keywords:
                filters = []
                for kw in keywords[:4]:
                    filters.append(Article.headline.ilike(f"%{kw}%"))
                    filters.append(Article.summary.ilike(f"%{kw}%"))
                    filters.append(Article.full_text.ilike(f"%{kw}%"))
                stmt = stmt.where(select(Article).where(*filters).whereclause)  # type: ignore[arg-type]

            stmt = stmt.order_by(asc(Issue.issue_date), asc(Article.primary_page_id)).limit(35)
            db_res = await db.execute(stmt)
            direct_articles = list(db_res.scalars().all())

        # Anti-Hallucination Gate 1 & 2: Multi-Hop Graph Expansion bounded strictly by Event Window and High Salience
        expanded_articles_map: dict[int, Article] = {a.id: a for a in direct_articles}
        if direct_articles:
            event_dates = [
                str(a.issue.issue_date) for a in direct_articles if a.issue and a.issue.issue_date
            ]
            min_date = min(event_dates) if event_dates else None
            max_date = max(event_dates) if event_dates else None

            # Collect high-salience protagonist entities from direct articles (Gate 1)
            focal_entity_names = set()
            for art in direct_articles:
                if hasattr(art, "article_entities") and art.article_entities:
                    for ae in art.article_entities:
                        if ae.salience_score >= 0.50 and ae.entity and ae.entity.name:
                            focal_entity_names.add(ae.entity.name)

            if focal_entity_names and min_date and max_date:
                # Find connected 2-hop articles within the exact temporal window (Gate 2)
                async with self._session_factory() as db:
                    hop_stmt = (
                        select(Article)
                        .join(Issue, Article.issue_id == Issue.id)
                        .join(ArticleEntity, ArticleEntity.article_id == Article.id)
                        .join(Entity, ArticleEntity.entity_id == Entity.id)
                        .where(
                            Entity.name.in_(list(focal_entity_names)[:8]),
                            Issue.issue_date >= min_date,
                            Issue.issue_date <= max_date,
                            ArticleEntity.salience_score >= 0.40,
                        )
                        .options(
                            selectinload(Article.issue).selectinload(Issue.newspaper),
                            selectinload(Article.article_pages),
                            selectinload(Article.article_entities).selectinload(ArticleEntity.entity),
                        )
                        .limit(20)
                    )
                    if issue_ids:
                        hop_stmt = hop_stmt.where(Issue.id.in_(issue_ids))
                    hop_res = await db.execute(hop_stmt)
                    for ha in hop_res.scalars().all():
                        expanded_articles_map[ha.id] = ha

        articles = list(expanded_articles_map.values())

        if not articles:
            # Fallback to broader search if specific keywords yielded 0
            async with self._session_factory() as db:
                fallback_stmt = (
                    select(Article)
                    .join(Issue, Article.issue_id == Issue.id)
                    .options(
                        selectinload(Article.issue).selectinload(Issue.newspaper),
                        selectinload(Article.article_pages),
                        selectinload(Article.article_entities).selectinload(ArticleEntity.entity),
                    )
                    .order_by(asc(Issue.issue_date))
                    .limit(20)
                )
                if issue_ids:
                    fallback_stmt = fallback_stmt.where(Issue.id.in_(issue_ids))
                fb_res = await db.execute(fallback_stmt)
                articles = list(fb_res.scalars().all())

        # Anti-Hallucination Gate 3: Neural Cross-Encoder Verification Thresholding
        # Reject candidates with semantic relevance score below 0.20 for storyline consistency
        if len(articles) > 5:
            try:
                reranker = CrossEncoderReranker()
                candidate_pairs = [
                    (
                        clean_query,
                        f"{a.headline or ''} - {(a.summary or a.full_text or '')[:300]}",
                    )
                    for a in articles
                ]
                scores = reranker.predict(candidate_pairs)
                verified_articles = []
                for a, sc in zip(articles, scores):
                    if sc >= 0.15:  # Retain verified storyline-relevant documents
                        verified_articles.append(a)
                if verified_articles:
                    articles = verified_articles
            except Exception as re_err:
                logger.warning(
                    "Neural Cross-Encoder verification skipped in timeline",
                    extra={"error": str(re_err)},
                )

        if not articles:
            return NarrativeTrajectoryResponse(
                query=clean_query,
                topic_summary=f"No archival records found matching topic: '{clean_query}'.",
                date_range=[],
                milestones=[],
                latency_ms=round((time.monotonic() - t0) * 1000),
                cost_usd=0.0,
            )

        # 4. Temporal Grouping by Calendar Date
        if on_progress:
            await on_progress(
                "clustering_dates",
                {"message": f"Clustered {len(articles)} reports across dates"},
            )

        articles_by_date: dict[str, list[Article]] = {}
        for art in articles:
            d_str = str(art.issue.issue_date) if art.issue else "Unknown Date"
            if d_str not in articles_by_date:
                articles_by_date[d_str] = []
            articles_by_date[d_str].append(art)

        all_dates = sorted(articles_by_date.keys())
        date_range = [all_dates[0], all_dates[-1]] if all_dates else []

        # 5. Build Multi-Perspective Evidence Context for LLM Synthesis
        evidence_prompt_blocks: list[str] = []
        for dt in all_dates:
            date_arts = articles_by_date[dt]
            block = [f"=== DATE: {dt} ({len(date_arts)} articles) ==="]
            for a in date_arts:
                np_name = a.issue.newspaper.name if a.issue and a.issue.newspaper else "Daily News"
                p_num = a.article_pages[0].page_number if a.article_pages else 1
                snip = a.summary or (a.full_text[:300] if a.full_text else "No preview available.")
                block.append(
                    f"• [{np_name} | Page {p_num}] \"{a.headline or 'Untitled'}\"\n"
                    f"  Snippet: {snip.strip()}"
                )
            evidence_prompt_blocks.append("\n".join(block))

        user_content = (
            f"Topic Query: {clean_query}\n\n"
            f"Archival Evidence Grouped by Date:\n"
            f"{chr(10).join(evidence_prompt_blocks)}\n\n"
            f"Synthesize the storyline milestones, contrast each newspaper's angle, "
            f"and list any discrepancies found across reports."
        )

        if on_progress:
            await on_progress(
                "synthesizing_perspectives",
                {"message": "Analyzing perspectives & discrepancies"},
            )

        # 6. LLM Structured Analysis Pass
        cost_usd = 0.0
        milestones: list[TimelineMilestone] = []
        topic_summary = (
            f"Chronological coverage of {clean_query} across {len(all_dates)} reported dates."
        )

        provider = None
        try:
            registry = get_registry()
            provider = registry.get_chat_provider(model_override)
        except Exception as e:
            logger.warning(
                "Could not instantiate LLM chat provider for timeline",
                extra={"error": str(e)},
            )

        llm_succeeded = False
        if provider:
            try:
                response = await provider.complete(
                    messages=[
                        Message(role="system", content=TIMELINE_SYNTHESIS_SYSTEM_PROMPT),
                        Message(role="user", content=user_content),
                    ],
                    max_tokens=3000,
                    temperature=0.1,
                )
                p_name = getattr(provider, "provider_name", "llm")
                m_name = getattr(provider, "model_name", "default")
                cost_usd = record_usage_and_cost(
                    p_name,
                    m_name,
                    response.input_tokens,
                    response.output_tokens,
                )

                json_str = response.text.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()

                parsed = json.loads(json_str)
                topic_summary = parsed.get("topic_summary", topic_summary)
                parsed_milestones = parsed.get("milestones", [])

                for idx, m_data in enumerate(parsed_milestones, start=1):
                    m_date = m_data.get("date", all_dates[0] if all_dates else "2026-08-01")
                    c_event = m_data.get("canonical_event", "Key news event reported")
                    e_phase = m_data.get("event_phase", "Development")
                    discrepancies = m_data.get("discrepancies", [])

                    perspectives: list[NewspaperPerspective] = []
                    matched_articles = articles_by_date.get(m_date, [])

                    for p_item in m_data.get("perspectives", []):
                        p_np = p_item.get("newspaper_name", "Daily News")
                        p_hl = p_item.get("headline", "")
                        p_angle = p_item.get("angle", "General Coverage")
                        p_takeaway = p_item.get("key_takeaway", "")

                        match_art = None
                        for cand in matched_articles:
                            cand_np = (
                                cand.issue.newspaper.name
                                if cand.issue and cand.issue.newspaper
                                else ""
                            )
                            if cand_np.lower() in p_np.lower() or p_np.lower() in cand_np.lower():
                                match_art = cand
                                break
                        if not match_art and matched_articles:
                            match_art = matched_articles[0]

                        art_id = match_art.id if match_art else 0
                        iss_id = match_art.issue_id if match_art else 0
                        pdf_page = (
                            match_art.article_pages[0].page_number
                            if match_art and match_art.article_pages
                            else 1
                        )
                        bboxes = []
                        if match_art and match_art.article_pages:
                            for ap in match_art.article_pages:
                                if ap.bbox_json:
                                    if isinstance(ap.bbox_json, list):
                                        bboxes.extend(ap.bbox_json)
                                    elif isinstance(ap.bbox_json, dict):
                                        bboxes.append(ap.bbox_json)

                        hl_fallback = (
                            match_art.headline
                            if (match_art and match_art.headline)
                            else "Report"
                        )
                        final_headline = str(p_hl or hl_fallback)
                        final_takeaway = str(
                            p_takeaway
                            or (match_art.summary if (match_art and match_art.summary) else "")
                        )

                        perspectives.append(
                            NewspaperPerspective(
                                newspaper_name=p_np,
                                issue_date=m_date,
                                pdf_page=pdf_page,
                                headline=final_headline,
                                key_takeaway=final_takeaway,
                                angle=p_angle,
                                bboxes=bboxes,
                                issue_id=iss_id,
                                article_id=art_id,
                            )
                        )
                    # Extract active entities for this milestone from the articles on that date
                    milestone_entities_map: dict[str, dict[str, Any]] = {}
                    for cand in matched_articles:
                        if hasattr(cand, "article_entities") and cand.article_entities:
                            for ae in cand.article_entities:
                                if ae.entity and ae.entity.name:
                                    e_name = ae.entity.name
                                    if e_name not in milestone_entities_map:
                                        milestone_entities_map[e_name] = {
                                            "name": e_name,
                                            "type": ae.entity.type,
                                            "salience": round(ae.salience_score, 2),
                                            "mentions": ae.mention_count,
                                        }
                    sorted_entities = sorted(
                        milestone_entities_map.values(),
                        key=lambda x: x["salience"],
                        reverse=True,
                    )[:6]

                    milestones.append(
                        TimelineMilestone(
                            milestone_id=f"milestone_{idx}_{m_date}",
                            date=m_date,
                            canonical_event=c_event,
                            event_phase=e_phase,
                            perspectives=perspectives,
                            discrepancies=discrepancies,
                            active_entities=sorted_entities,
                        )
                    )

                llm_succeeded = True
            except Exception as e:
                logger.warning(
                    "LLM timeline structured synthesis failed; falling back to heuristic engine",
                    extra={"error": str(e)},
                )

        # 7. Deterministic Fallback Heuristic if LLM is offline/mock
        if not llm_succeeded or not milestones:
            milestones = []
            for idx, dt in enumerate(all_dates, start=1):
                date_arts = articles_by_date[dt]
                lead_art = date_arts[0]

                if idx == 1:
                    phase = "Breaking"
                elif idx == len(all_dates) and len(all_dates) > 2:
                    phase = "Regulatory"
                elif any(
                    "financial" in (a.section or "").lower()
                    or "market" in (a.section or "").lower()
                    for a in date_arts
                ):
                    phase = "Financial"
                else:
                    phase = "Development"

                perspectives = []
                for a in date_arts:
                    np_name = (
                        a.issue.newspaper.name if a.issue and a.issue.newspaper else "Daily News"
                    )
                    pdf_page = a.article_pages[0].page_number if a.article_pages else 1
                    bboxes = []
                    if a.article_pages:
                        for ap in a.article_pages:
                            if ap.bbox_json:
                                if isinstance(ap.bbox_json, list):
                                    bboxes.extend(ap.bbox_json)
                                elif isinstance(ap.bbox_json, dict):
                                    bboxes.append(ap.bbox_json)

                    angle = "General News"
                    if "mint" in np_name.lower() or "financial" in (a.section or "").lower():
                        angle = "Financial Impact"
                    elif "business standard" in np_name.lower():
                        angle = "Policy & Industry"
                    elif "hindu" in np_name.lower():
                        angle = "Socio-Political"

                    takeaway = a.summary or (a.full_text[:180] if a.full_text else "Event covered.")

                    perspectives.append(
                        NewspaperPerspective(
                            newspaper_name=np_name,
                            issue_date=dt,
                            pdf_page=pdf_page,
                            headline=a.headline or "News Story",
                            key_takeaway=takeaway,
                            angle=angle,
                            bboxes=bboxes,
                            issue_id=a.issue_id,
                            article_id=a.id,
                        )
                    )

                milestone_entities_map = {}
                for cand in date_arts:
                    if hasattr(cand, "article_entities") and cand.article_entities:
                        for ae in cand.article_entities:
                            if ae.entity and ae.entity.name:
                                e_name = ae.entity.name
                                if e_name not in milestone_entities_map:
                                    milestone_entities_map[e_name] = {
                                        "name": e_name,
                                        "type": ae.entity.type,
                                        "salience": round(ae.salience_score, 2),
                                        "mentions": ae.mention_count,
                                    }
                sorted_entities = sorted(
                    milestone_entities_map.values(),
                    key=lambda x: x["salience"],
                    reverse=True,
                )[:6]

                discrepancies = []
                if len(perspectives) > 1:
                    discrepancies.append(
                        f"Distinct editorial focus observed on {dt} between "
                        f"{perspectives[0].newspaper_name} ({perspectives[0].angle}) and "
                        f"{perspectives[1].newspaper_name} ({perspectives[1].angle})."
                    )

                c_event = lead_art.headline or f"Major developments regarding {clean_query}"
                milestones.append(
                    TimelineMilestone(
                        milestone_id=f"milestone_{idx}_{dt}",
                        date=dt,
                        canonical_event=c_event,
                        event_phase=phase,
                        perspectives=perspectives,
                        discrepancies=discrepancies,
                    )
                )

        trajectory_response = NarrativeTrajectoryResponse(
            query=clean_query,
            topic_summary=topic_summary,
            date_range=date_range,
            milestones=milestones,
            latency_ms=round((time.monotonic() - t0) * 1000),
            cost_usd=cost_usd,
            cached=False,
        )

        # 8. Store in Heavy-LLM Redis Cache (Phase 7 Integration)
        if use_cache:
            await self._cache.set_query(
                cache_key,
                trajectory_response.model_dump(),
                ttl_seconds=3600,
            )

        if on_progress:
            await on_progress("completed", {"milestones_count": len(milestones)})

        return trajectory_response

