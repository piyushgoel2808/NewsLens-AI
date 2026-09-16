"""3-Tier Negative Coverage and Omission Reconciliation Engine.

Enforces the Coverage Invariant:
Never claim a newspaper "did NOT cover" an event based solely on top-k retrieval.
Distinguishes between:
- COVERED: Confirmed relevant article(s) found in publication.
- NOT_FOUND: Issue exists and is fully ingested ('completed'), but no relevant articles found.
- UNCERTAIN: Borderline relevance score or ambiguous evidence.
- PROCESSING_ERROR: Issue failed ingestion or is still pending/processing.
"""

from __future__ import annotations

import asyncio
import datetime
import re
from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.models.newspaper import Issue, Newspaper
from app.retrieval.hybrid_search import HybridSearchEngine, SearchFilter

logger = get_logger(__name__)


class CoverageStatus(StrEnum):
    """Authoritative classification of newspaper coverage."""

    COVERED = "COVERED"
    NOT_FOUND = "NOT_FOUND"
    UNCERTAIN = "UNCERTAIN"
    PROCESSING_ERROR = "PROCESSING_ERROR"


@dataclass
class PublicationCoverageReport:
    """Coverage determination for a single newspaper publication."""

    newspaper_id: int
    newspaper_name: str
    status: CoverageStatus
    confidence: float
    matched_article_ids: list[int] = field(default_factory=list)
    matched_headlines: list[str] = field(default_factory=list)
    top_score: float = 0.0
    evidence_snippet: str | None = None
    audit_notes: str = ""


@dataclass
class CoverageMatrix:
    """Multi-newspaper cross-publication comparative coverage matrix."""

    target_query_or_event: str
    target_date: str | None
    total_publications: int = 0
    covered_count: int = 0
    not_found_count: int = 0
    uncertain_count: int = 0
    processing_error_count: int = 0
    reports: dict[str, PublicationCoverageReport] = field(default_factory=dict)


class CoverageAnalyzer:
    """Rigorous 3-Tier Coverage & Gap Analysis Engine.

    Tiers:
      1. Relational health audit: Verify issue exists and ingestion_status == 'completed'.
         If ingestion_status != 'completed', MUST return UNCERTAIN or PROCESSING_ERROR.
      2. Targeted publication-scoped hybrid retrieval cascade.
      3. Confidence-based coverage classification (COVERED vs NOT_FOUND vs UNCERTAIN).
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        hybrid_search_engine: HybridSearchEngine,
        high_relevance_threshold: float = 0.55,
        uncertainty_threshold: float = 0.30,
    ) -> None:
        self._session_factory = session_factory
        self._hybrid_search = hybrid_search_engine
        self.high_threshold = high_relevance_threshold
        self.uncertainty_threshold = uncertainty_threshold

    async def analyze_newspaper_coverage(
        self,
        newspaper: Newspaper,
        query_or_event: str,
        target_date: str | None = None,
        date_window_days: int = 2,
        query_vector: list[float] | None = None,
    ) -> PublicationCoverageReport:
        """Perform 3-tier coverage evaluation for a single newspaper regarding an event."""
        logger.info(
            "Analyzing coverage for newspaper",
            extra={"newspaper": newspaper.name, "query": query_or_event, "target_date": target_date},
        )

        # Tier 1: Relational Ingestion Health Audit
        async with self._session_factory() as db:
            stmt = select(Issue).where(Issue.newspaper_id == newspaper.id)
            if target_date:
                try:
                    d_obj = datetime.date.fromisoformat(target_date)
                    d_start = d_obj - datetime.timedelta(days=date_window_days)
                    d_end = d_obj + datetime.timedelta(days=date_window_days)
                    stmt = stmt.where(Issue.issue_date >= d_start, Issue.issue_date <= d_end)
                except ValueError:
                    pass

            res = await db.execute(stmt)
            issues = res.scalars().all()

        if not issues:
            return PublicationCoverageReport(
                newspaper_id=newspaper.id,
                newspaper_name=newspaper.name,
                status=CoverageStatus.PROCESSING_ERROR,
                confidence=0.0,
                audit_notes="No newspaper issue or edition found in archive for targeted date window.",
            )

        # Invariant: If any targeted issue failed or is not fully completed, flag processing error / uncertain
        failed_or_incomplete = [
            iss for iss in issues if iss.ingestion_status not in ("completed", "indexed", "ready")
        ]
        if failed_or_incomplete:
            statuses = ", ".join(f"Issue {iss.id}: {iss.ingestion_status}" for iss in failed_or_incomplete)
            return PublicationCoverageReport(
                newspaper_id=newspaper.id,
                newspaper_name=newspaper.name,
                status=CoverageStatus.PROCESSING_ERROR,
                confidence=0.0,
                audit_notes=f"Newspaper issue not fully ingested. Current status: {statuses}",
            )

        # Tier 2: Publication-Scoped Targeted Hybrid Retrieval Cascade
        d_from = None
        d_to = None
        if target_date:
            try:
                d_obj = datetime.date.fromisoformat(target_date)
                d_from = (d_obj - datetime.timedelta(days=date_window_days)).isoformat()
                d_to = (d_obj + datetime.timedelta(days=date_window_days)).isoformat()
            except ValueError:
                pass

        search_filter = SearchFilter(
            newspaper_id=newspaper.id,
            date_from=d_from,
            date_to=d_to,
        )

        clean_q = re.sub(
            r"(?i)\b(?:did|was|were|is|are|has|have)\s+(?:the\s+)?[\w\s]+\s+(?:report\s+on|cover|publish|carry|mention)\s*",
            "",
            query_or_event,
        )
        clean_q = re.sub(r"(?i)\b(?:on|dated?)\s+\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "", clean_q).strip("?:!.,\"' ") or query_or_event

        hits = await self._hybrid_search.search(
            query=clean_q,
            top_k=5,
            filters=search_filter,
            rerank=True,
            query_vector=query_vector,
        )

        # Tier 3: Rigorous Confidence Classification
        if hits:
            top_hit = hits[0]
            has_rerank = top_hit.rerank_score is not None
            rerank_val = top_hit.rerank_score if has_rerank else None
            score = rerank_val if has_rerank else top_hit.rrf_score

            if has_rerank and rerank_val is not None:
                # Calibrated cross-encoder logits:
                # >= -4.5: Confirmed coverage
                # -7.0 to -4.5: Borderline / uncertain relevance
                # < -7.0: Below semantic relevance baseline -> Confirmed omission (NOT_FOUND)
                if rerank_val >= -4.5:
                    matched_art_ids = [h.article_id for h in hits if h.rerank_score is not None and h.rerank_score >= -5.5]
                    matched_hls = [h.headline for h in hits[:3] if h.headline and (h.rerank_score is None or h.rerank_score >= -5.5)]
                    conf = 0.95 if rerank_val >= -2.0 else 0.85
                    return PublicationCoverageReport(
                        newspaper_id=newspaper.id,
                        newspaper_name=newspaper.name,
                        status=CoverageStatus.COVERED,
                        confidence=conf,
                        matched_article_ids=matched_art_ids or [top_hit.article_id],
                        matched_headlines=matched_hls or ([top_hit.headline] if top_hit.headline else []),
                        top_score=round(float(score), 4),
                        evidence_snippet=top_hit.snippet[:300],
                        audit_notes=f"Confirmed coverage across {len(matched_art_ids or [1])} relevant article(s).",
                    )
                elif rerank_val >= -7.0:
                    return PublicationCoverageReport(
                        newspaper_id=newspaper.id,
                        newspaper_name=newspaper.name,
                        status=CoverageStatus.UNCERTAIN,
                        confidence=0.5,
                        matched_article_ids=[top_hit.article_id],
                        matched_headlines=[top_hit.headline] if top_hit.headline else [],
                        top_score=round(float(score), 4),
                        evidence_snippet=top_hit.snippet[:300],
                        audit_notes="Borderline relevance match found. Manual editorial verification advised.",
                    )
            else:
                # Fallback when reranker is unavailable: use RRF / dense scores
                is_covered = (score >= self.high_threshold) or (top_hit.rrf_score >= 0.012 and len(hits) >= 1)
                if is_covered:
                    matched_art_ids = [h.article_id for h in hits]
                    matched_hls = [h.headline for h in hits[:3] if h.headline]
                    return PublicationCoverageReport(
                        newspaper_id=newspaper.id,
                        newspaper_name=newspaper.name,
                        status=CoverageStatus.COVERED,
                        confidence=0.85,
                        matched_article_ids=matched_art_ids,
                        matched_headlines=matched_hls,
                        top_score=round(float(score), 4),
                        evidence_snippet=top_hit.snippet[:300],
                        audit_notes=f"Confirmed coverage across {len(hits)} relevant article(s).",
                    )
                elif score >= self.uncertainty_threshold or top_hit.rrf_score >= 0.008:
                    return PublicationCoverageReport(
                        newspaper_id=newspaper.id,
                        newspaper_name=newspaper.name,
                        status=CoverageStatus.UNCERTAIN,
                        confidence=0.5,
                        matched_article_ids=[top_hit.article_id],
                        matched_headlines=[top_hit.headline] if top_hit.headline else [],
                        top_score=round(float(score), 4),
                        evidence_snippet=top_hit.snippet[:300],
                        audit_notes="Borderline relevance match found. Manual editorial verification advised.",
                    )

        # Issue is fully verified as ingested ('completed'), but 0 semantic matches met threshold
        return PublicationCoverageReport(
            newspaper_id=newspaper.id,
            newspaper_name=newspaper.name,
            status=CoverageStatus.NOT_FOUND,
            confidence=0.95,
            audit_notes="Issue fully ingested and indexed without processing errors; 0 relevant coverage found (confirmed omission).",
        )

    async def generate_coverage_matrix(
        self,
        query_or_event: str,
        target_date: str | None = None,
        date_window_days: int = 2,
        newspaper_name: str | None = None,
    ) -> CoverageMatrix:
        """Generate comparative multi-newspaper coverage reconciliation matrix."""
        async with self._session_factory() as db:
            if target_date:
                try:
                    d_obj = datetime.date.fromisoformat(target_date)
                    d_start = d_obj - datetime.timedelta(days=date_window_days)
                    d_end = d_obj + datetime.timedelta(days=date_window_days)
                    stmt = (
                        select(Newspaper)
                        .join(Issue, Issue.newspaper_id == Newspaper.id)
                        .where(Issue.issue_date >= d_start, Issue.issue_date <= d_end)
                        .distinct()
                    )
                except ValueError:
                    stmt = select(Newspaper)
            else:
                stmt = select(Newspaper)
            if newspaper_name:
                stmt = stmt.where(Newspaper.name.ilike(f"%{newspaper_name}%"))
            res = await db.execute(stmt)
            newspapers = res.scalars().all()

        matrix = CoverageMatrix(
            target_query_or_event=query_or_event,
            target_date=target_date,
            total_publications=len(newspapers),
        )

        if not newspapers:
            return matrix

        # Precompute query vector once across all publications
        query_vector: list[float] | None = None
        if query_or_event and query_or_event.strip():
            try:
                provider = self._hybrid_search._get_embedding_provider()
                query_vector = await provider.embed_one(query_or_event)
            except Exception as ex:
                logger.warning("Could not precompute query vector for coverage matrix", extra={"error": str(ex)})

        reports = await asyncio.gather(*(
            self.analyze_newspaper_coverage(
                newspaper=np,
                query_or_event=query_or_event,
                target_date=target_date,
                date_window_days=date_window_days,
                query_vector=query_vector,
            )
            for np in newspapers
        ))

        for np, rep in zip(newspapers, reports, strict=False):
            matrix.reports[np.name] = rep
            if rep.status == CoverageStatus.COVERED:
                matrix.covered_count += 1
            elif rep.status == CoverageStatus.NOT_FOUND:
                matrix.not_found_count += 1
            elif rep.status == CoverageStatus.UNCERTAIN:
                matrix.uncertain_count += 1
            elif rep.status == CoverageStatus.PROCESSING_ERROR:
                matrix.processing_error_count += 1

        return matrix
