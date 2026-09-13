"""SQL Analytics Engine: Quantitative trends, mention frequencies, and distribution metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.article import Article, ArticlePage, Photo
from app.models.entity import ArticleEntity, ArticleTopic, Entity
from app.models.newspaper import Issue, Newspaper

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "category_aliases.yaml"
_USER_QUERY_SYNONYMS: dict[str, str] = {}
_CATEGORY_KEYWORDS: dict[str, list[str]] = {}
_CANONICAL_CATEGORIES: list[str] = []

if _CONFIG_PATH.exists():
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            _USER_QUERY_SYNONYMS = {
                k.lower().strip(): v for k, v in cfg.get("user_query_synonyms", {}).items()
            }
            _CATEGORY_KEYWORDS = cfg.get("category_keywords", {})
            _CANONICAL_CATEGORIES = cfg.get("canonical_categories", [])
    except Exception as e:
        logger.warning("Failed to load category_aliases.yaml in sql_analytics", extra={"error": str(e)})



_COMMON_HEADLINE_VOCAB = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "from",
    "by", "of", "up", "out", "over", "into", "as", "is", "are", "was", "were",
    "stocks", "stock", "rally", "fall", "falls", "surge", "surges", "plunge", "plunges",
    "crash", "rise", "rises", "drop", "drops", "gain", "gains", "loss", "losses",
    "meet", "meets", "talks", "win", "wins", "lose", "loses", "protest", "protests",
    "attack", "attacks", "death", "killed", "dies", "dead", "open", "opens", "close", "closes",
    "war", "market", "markets", "govt", "government", "minister", "police", "report", "reports",
    "tax", "taxes", "rate", "rates", "cricket", "cup", "match", "final", "plan", "plans",
    "deal", "deals", "bill", "bills", "row", "scam", "held", "hit", "hits", "warns", "seeks",
    "calls", "poll", "polls", "vote", "votes", "gold", "bank", "banks", "fund", "funds",
    "growth", "inflation", "gdp", "trade", "india", "world", "local", "sports", "news",
    "tech", "ai", "auto", "power", "solar", "oil", "gas", "rupee", "dollar", "price", "prices",
    "high", "low", "record", "new", "top", "day", "year", "lead", "leads", "chief", "case",
    "court", "order", "plea", "bail", "jail", "probe", "fire", "blast", "road", "water",
    "hospital", "health", "cancer", "heart", "care", "cure", "drug", "drugs", "study", "recap",
}

_NEWS_STOP_WORDS: frozenset[str] = frozenset({
    "about", "above", "after", "again", "against", "all", "also", "and", "any", "are", "back", "been",
    "before", "being", "below", "between", "both", "came", "come", "could", "did", "does", "down",
    "during", "each", "even", "first", "from", "further", "had", "has", "have", "here", "into", "just",
    "like", "made", "make", "many", "more", "most", "much", "must", "new", "now", "only", "other",
    "our", "out", "over", "said", "same", "says", "should", "some", "still", "such", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "time", "under",
    "very", "was", "way", "well", "were", "what", "when", "where", "which", "while", "who", "whom",
    "will", "with", "would", "special", "feature", "today", "edition", "daily", "brief", "short", "news",
    "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
    "in short >>", "in short", "news in brief", "panaji", "margao", "vasco", "mapusa", "delhi", "mumbai",
})

_UBIQUITOUS_ENTITIES: frozenset[str] = frozenset({
    "india", "delhi", "goa", "mumbai", "new delhi", "panaji", "state", "centre", "government",
    "police", "court", "high court", "supreme court", "bjp", "congress", "general", "news",
})


def extract_substantive_tokens(text: str | None) -> set[str]:
    """Extract lowercase substantive alphanumeric tokens excluding stopwords."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in _NEWS_STOP_WORDS}


def extract_numeric_anchors(text: str | None) -> set[str]:
    """Extract numeric, percentage, and monetary quantity anchors from text."""
    if not text:
        return set()
    nums = re.findall(r"\b(?:\d+[\.,]?\d*(?:%|k|cr|crore|lakh|bn|m)?)\b", text.lower())
    return {n for n in nums if len(n) >= 2 or (n.isdigit() and int(n) >= 10)}


def sanitize_headline(
    headline: str | None,
    subheadline: str | None = None,
    byline_author: str | None = None,
    snippet: str | None = None,
) -> tuple[str, str | None]:
    """Cleanse headlines where doctor/author profile names were mistakenly extracted as the headline.

    Returns (cleaned_headline, effective_byline).
    """
    if not headline or not headline.strip():
        return ("Untitled Report", byline_author)

    hl = headline.strip()
    byline = (byline_author or "").strip() or None
    sub = (subheadline or "").strip()
    snip = (snippet or "").strip()
    if snip:
        snip = re.sub(r"\[Newspaper:.*?\]\s*", "", snip, flags=re.IGNORECASE)
        snip = re.sub(r"\[Page\(s\):.*?\]\s*", "", snip, flags=re.IGNORECASE)
        snip = re.sub(r"\[Exact Chunk Match\]:?\s*", "", snip, flags=re.IGNORECASE)
        snip = re.sub(r"\[Visual Data Asset:.*?\]\s*", "", snip, flags=re.IGNORECASE)
        snip = re.sub(r"\[Article Parent Context\]:?\s*", "", snip, flags=re.IGNORECASE)
        snip = re.sub(r"\[📷\s*Attached Image/Photo:.*?\]\s*", "", snip, flags=re.IGNORECASE)
        snip = re.sub(r"^[\<\#\s\.\,\-]+", "", snip).strip()

    # 1. Detect Doctor / Expert name box as headline (e.g. 'Dr. Smriti Naswa Singh')
    if re.match(r"^(?:Dr\.?|Doctor|Prof\.?|Professor)\s+[A-Z]", hl, re.IGNORECASE):
        effective_byline = hl if not byline else f"{hl} ({byline})"
        if sub and len(sub) > 5:
            cleaned_sub = re.sub(r"[\s\~\-]+$", "", sub).strip()
            return (cleaned_sub, effective_byline)
        elif snip:
            first_sent = re.split(r"[.\n]", snip)[0].strip()
            if len(first_sent) > 10:
                return (first_sent[:80], effective_byline)
        return (f"Medical Column: {hl}", effective_byline)

    # 2. Headline identical to byline author
    if byline and hl.lower() == byline.lower():
        if sub and len(sub) > 5:
            return (sub, byline)
        elif snip:
            first_sent = re.split(r"[.\n]", snip)[0].strip()
            if len(first_sent) > 10:
                return (first_sent[:80], byline)
        return (f"Profile: {hl}", byline)

    # 3. All-caps 2 or 3 word person name without verbs or news terms (e.g. 'UTHAMA SANKARANARAYANAN')
    words = hl.split()
    if (
        2 <= len(words) <= 3
        and hl.isupper()
        and all(w.isalpha() for w in words)
        and not any(w.lower() in _COMMON_HEADLINE_VOCAB for w in words)
    ):
        if snip:
            first_sent = re.split(r"[.\n]", snip)[0].strip()
            if len(first_sent) > 15:
                topic = re.sub(r"^(?:A|The)\s+", "", first_sent, flags=re.IGNORECASE)
                topic = topic[:70].strip()
                eff_byline = f"{hl} / {byline}" if byline else hl
                return (f"{topic.capitalize()}", eff_byline)
        return (f"Special Feature: {hl.title()}", byline or hl.title())

    return (hl, byline)


class SQLAnalyticsEngine:
    """Executes safe, parameterized aggregation queries for quantitative news trends."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_archive_metadata(self) -> dict[str, Any]:
        """Retrieve lightweight metadata summary of available dates, newspapers, and categories."""
        from app.models.article import ArticleCategory
        from app.models.newspaper import Newspaper

        async with self._session_factory() as db:
            stmt = (
                select(Issue.issue_date, Newspaper.name, Issue.id)
                .join(Newspaper, Issue.newspaper_id == Newspaper.id)
                .where(Issue.ingestion_status.in_(("completed", "indexed", "ready")))
                .order_by(Issue.issue_date.desc())
            )
            res = await db.execute(stmt)
            rows = res.all()
            by_date: dict[str, list[str]] = {}
            for dt, np_name, _ in rows:
                dt_str = str(dt)
                if np_name not in by_date.setdefault(dt_str, []):
                    by_date[dt_str].append(np_name)

            res2 = await db.execute(select(ArticleCategory.name).order_by(ArticleCategory.name))
            cats = list(res2.scalars().all())

            return {
                "available_dates": by_date,
                "categories": cats or _CANONICAL_CATEGORIES,
            }

    async def get_entity_mention_trends(
        self,
        entity_name: str,
        group_by_period: str = "month",  # "day", "month", "year"
    ) -> list[dict[str, Any]]:
        """Compute mention frequency trends over time for a specific entity."""
        async with self._session_factory() as db:
            stmt = (
                select(
                    Issue.issue_date,
                    func.count(ArticleEntity.article_id).label("article_count"),
                    func.sum(ArticleEntity.mention_count).label("total_mentions"),
                    func.avg(ArticleEntity.salience_score).label("avg_salience"),
                )
                .join(Article, ArticleEntity.article_id == Article.id)
                .join(Issue, Article.issue_id == Issue.id)
                .join(Entity, ArticleEntity.entity_id == Entity.id)
                .where(Entity.name.ilike(f"%{entity_name}%"))
                .group_by(Issue.issue_date)
                .order_by(Issue.issue_date)
            )

            res = await db.execute(stmt)
            rows = res.all()

            return [
                {
                    "date": str(r.issue_date),
                    "article_count": r.article_count,
                    "total_mentions": int(r.total_mentions or 0),
                    "avg_salience": round(float(r.avg_salience or 0.0), 3),
                }
                for r in rows
            ]

    async def get_topic_distribution(
        self,
        newspaper_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compute article volume breakdown across topic categories."""
        async with self._session_factory() as db:
            stmt = select(
                Article.section,
                Article.article_type,
                func.count(Article.id).label("count"),
                func.avg(Article.prominence_score).label("avg_prominence"),
            ).join(Issue, Article.issue_id == Issue.id)

            if newspaper_id:
                stmt = stmt.where(Issue.newspaper_id == newspaper_id)

            stmt = stmt.group_by(Article.section, Article.article_type).order_by(desc("count"))

            res = await db.execute(stmt)
            rows = res.all()

            return [
                {
                    "section": r.section or "General",
                    "article_type": r.article_type,
                    "count": r.count,
                    "avg_prominence": round(float(r.avg_prominence or 0.0), 3),
                }
                for r in rows
            ]

    async def get_frontpage_prominence_ratio(
        self,
        newspaper_id: int | None = None,
    ) -> dict[str, Any]:
        """Compute percentage of articles appearing on Page 1 vs inside pages."""
        async with self._session_factory() as db:
            stmt = (
                select(
                    ArticlePage.page_number,
                    func.count(ArticlePage.article_id.distinct()).label("article_count"),
                )
                .join(Article, ArticlePage.article_id == Article.id)
                .join(Issue, Article.issue_id == Issue.id)
            )

            if newspaper_id:
                stmt = stmt.where(Issue.newspaper_id == newspaper_id)

            stmt = stmt.group_by(ArticlePage.page_number).order_by(ArticlePage.page_number)
            res = await db.execute(stmt)
            rows = res.all()

            page_distribution = {r.page_number: r.article_count for r in rows}
            total_articles = sum(page_distribution.values())
            page_1_count = page_distribution.get(1, 0)
            ratio = (page_1_count / total_articles) if total_articles > 0 else 0.0

            return {
                "total_articles": total_articles,
                "frontpage_articles": page_1_count,
                "frontpage_ratio": round(ratio, 4),
                "page_distribution": page_distribution,
            }

    async def list_issue_articles(
        self,
        issue_id: int | None = None,
        newspaper_name: str | None = None,
        issue_date: str | None = None,
        section: str | None = None,
        page_filter: str | int | None = None,
        exclude_page_filter: str | int | None = None,
        category_filter: str | None = None,
        page_number: int | None = None,
        query: str | None = None,
    ) -> Any:
        """Return a structured manifest of all articles in an issue with section/type breakdowns."""
        from app.models.newspaper import Newspaper

        # Defensive fallback: if query is provided and explicit parameters are missing, extract them
        if query and not (issue_id or newspaper_name or issue_date):
            from app.agent.planner import extract_parameters_from_query
            extracted = extract_parameters_from_query(query)
            if not issue_id and extracted.get("issue_id"):
                issue_id = extracted["issue_id"]
            if not newspaper_name and extracted.get("newspaper_name"):
                newspaper_name = extracted["newspaper_name"]
            if not issue_date and extracted.get("issue_date"):
                issue_date = extracted["issue_date"]
            if not category_filter and extracted.get("category_filter"):
                category_filter = extracted["category_filter"]

        async with self._session_factory() as db:
            issue: Issue | None = None
            if issue_id:
                stmt = (
                    select(Issue)
                    .where(Issue.id == issue_id)
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)
                # If issue found, but newspaper_name was also specified and does not match:
                if (
                    issue
                    and newspaper_name
                    and issue.newspaper
                    and newspaper_name.lower() not in issue.newspaper.name.lower()
                    and issue.newspaper.name.lower() not in newspaper_name.lower()
                ):
                    issue = None

            # Fallback 1: Resolve by (newspaper_name, issue_date)
            if not issue and newspaper_name and issue_date:
                stmt = (
                    select(Issue)
                    .join(Newspaper)
                    .where(
                        Newspaper.name.ilike(f"%{newspaper_name}%"),
                        Issue.issue_date == issue_date,
                    )
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)

            # Fallback 2: Resolve by newspaper_name (latest issue) only if issue_date was NOT specified
            if not issue and newspaper_name and not issue_date:
                stmt = (
                    select(Issue)
                    .join(Newspaper)
                    .where(Newspaper.name.ilike(f"%{newspaper_name}%"))
                    .order_by(desc(Issue.issue_date), desc(Issue.id))
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)

            # Fallback 3: Resolve by issue_date (latest issue on that date) only if newspaper_name was NOT specified
            if not issue and issue_date and not newspaper_name:
                stmt = (
                    select(Issue)
                    .where(Issue.issue_date == issue_date)
                    .order_by(desc(Issue.id))
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)

            # Fallback 4: Resolve latest overall issue if no filters given
            if not issue and not issue_id and not newspaper_name and not issue_date:
                stmt = (
                    select(Issue)
                    .order_by(desc(Issue.id))
                    .limit(1)
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)

            if not issue:
                if issue_id and not newspaper_name and not issue_date:
                    return {
                        "error": f"Issue #{issue_id} was not found in the newspaper archive.",
                        "articles": [],
                    }
                elif newspaper_name and issue_date:
                    return {
                        "error": f"No issue found for '{newspaper_name}' on date {issue_date}.",
                        "articles": [],
                    }
                elif newspaper_name:
                    return {
                        "error": f"No issues found in archive for newspaper '{newspaper_name}'.",
                        "articles": [],
                    }
                return {"error": "No newspaper issues found in archive.", "articles": []}

            art_stmt = (
                select(Article)
                .where(Article.issue_id == issue.id)
                .options(
                    selectinload(Article.category),
                    selectinload(Article.article_topics).selectinload(ArticleTopic.topic),
                )
                .order_by(Article.primary_page_id, desc(Article.prominence_score))
            )
            art_res = await db.execute(art_stmt)
            articles = art_res.scalars().all()

            section_counts: dict[str, int] = {}
            type_counts: dict[str, int] = {}
            category_counts: dict[str, int] = {}
            manifest: list[dict[str, Any]] = []

            page_folio_map = {
                p.id: p.printed_page_number or str(p.page_number) for p in issue.pages
            }
            page_num_map = {p.id: p.page_number for p in issue.pages}

            for a in articles:
                sec = a.section or "General"
                section_counts[sec] = section_counts.get(sec, 0) + 1
                atype = a.article_type or "news"
                type_counts[atype] = type_counts.get(atype, 0) + 1
                cat_name = a.category.name if a.category else (a.section or "General")
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                art_topics = [at.topic.name for at in (a.article_topics or []) if at.topic]

                p_num = page_num_map.get(a.primary_page_id, 1) if a.primary_page_id else 1
                folio = (
                    page_folio_map.get(a.primary_page_id, str(p_num))
                    if a.primary_page_id
                    else str(p_num)
                )

                clean_hl, clean_byline = sanitize_headline(
                    a.headline,
                    a.subheadline,
                    a.byline_author,
                    (a.full_text[:300] if a.full_text else None),
                )

                manifest.append(
                    {
                        "id": a.id,
                        "headline": clean_hl,
                        "raw_headline": a.headline,
                        "subheadline": a.subheadline,
                        "section": sec,
                        "printed_section": a.printed_section,
                        "category": cat_name,
                        "topics": art_topics,
                        "article_type": atype,
                        "byline_author": clean_byline or a.byline_author,
                        "page_number": p_num,
                        "printed_page": folio,
                        "word_count": a.word_count,
                        "prominence_score": a.prominence_score,
                    }
                )

            filtered_manifest = manifest

            # 1. Apply section or category filter if requested
            if section:
                sec_target = section.strip().lower()
                filtered_manifest = [
                    m for m in filtered_manifest
                    if sec_target in str(m.get("section", "")).lower()
                    or sec_target in str(m.get("printed_section", "") or "").lower()
                ]
            elif category_filter:
                cat_raw = category_filter.strip().lower()
                target_canons: list[str] = []
                # If query refers to economics, finance, markets, or business, bridge both economic domains
                if any(w in cat_raw for w in ["econom", "financ", "business", "market"]):
                    target_canons = ["Business & Markets", "Economy & Policy"]
                else:
                    resolved_cat = _USER_QUERY_SYNONYMS.get(cat_raw, cat_raw).lower()
                    canon_key = next(
                        (c for c in _CANONICAL_CATEGORIES if c.lower() in (cat_raw, resolved_cat)),
                        None,
                    )
                    target_canons = [canon_key] if canon_key else [cat_raw]

                kw_cluster: list[str] = []
                for c_name in target_canons:
                    kw_cluster.extend(_CATEGORY_KEYWORDS.get(c_name, []))
                if not kw_cluster:
                    kw_cluster = [cat_raw]

                matched: list[dict[str, Any]] = []
                matched_ids: set[int] = set()

                for m in filtered_manifest:
                    m_id = m.get("id")
                    m_cat = str(m.get("category", "")).lower()
                    m_sec = str(m.get("section", "")).lower()
                    m_psec = str(m.get("printed_section", "") or "").lower()
                    m_topics = [t.lower() for t in m.get("topics", [])]
                    hl_sub = f"{m.get('headline', '')} {m.get('subheadline', '')}".lower()

                    structured_match = any(
                        tc.lower() in m_cat
                        or tc.lower() in m_sec
                        or tc.lower() in m_psec
                        or any(tc.lower() in t for t in m_topics)
                        for tc in target_canons
                    )
                    keyword_match = any(
                        re.search(r"\b" + re.escape(kw.lower()) + r"\b", hl_sub)
                        for kw in kw_cluster
                        if len(kw) >= 3
                    )

                    # Domain noise filter: Discard obvious event listings, ads, court notices, or tax stories from Health manifests
                    if any("health" in tc.lower() for tc in target_canons):
                        has_explicit_health_term = any(
                            h_pat in hl_sub
                            for h_pat in [
                                "health", "hospital", "doctor", "medicine", "medical",
                                "disease", "patient", "clinic", "treatment", "vaccine",
                                "pharma", "clinical", "surgery", "diet", "mental health", "typhoid"
                            ]
                        )
                        # Exclude conflicting primary categories (e.g. Politics, Entertainment, Crime) unless explicitly about health
                        is_conflicting_primary = any(
                            non_h in m_cat for non_h in ["politics", "entertainment", "crime", "sports"]
                        )
                        if is_conflicting_primary and not has_explicit_health_term:
                            continue

                        if any(
                            noise_pat in hl_sub
                            for noise_pat in [
                                "when: ", "where: ", "studio xo", "cases still pending",
                                "[advertisement]", "sit vacant", "excise duty", "tax revenue",
                                "indirect taxes", "net tax collections", "cricket", "hockey",
                                "police complaint", "singer files"
                            ]
                        ) and not has_explicit_health_term:
                            continue

                    if (structured_match or keyword_match) and m_id not in matched_ids:
                        matched.append(m)
                        if m_id is not None:
                            matched_ids.add(m_id)

                filtered_manifest = matched

            # 2. Apply positive page filter if requested
            if page_number is not None:
                filtered_manifest = [m for m in filtered_manifest if m.get("page_number") == page_number]
            elif page_filter is not None:
                p_raw = str(page_filter).strip().lower()
                p_target = p_raw.replace("page", "").replace("pg", "").strip()
                printed_matches = [
                    m
                    for m in filtered_manifest
                    if str(m.get("printed_page", "")).lower() == p_target
                    or str(m.get("printed_page", "")).lower() == f"page {p_target}"
                ]
                filtered_manifest = (
                    printed_matches
                    if printed_matches
                    else [m for m in filtered_manifest if str(m.get("page_number")) == p_target]
                )

            # 3. Apply negative page exclusion filter (hard safety net)
            if exclude_page_filter is not None:
                excl_raw = str(exclude_page_filter).strip().lower()
                excl_target = excl_raw.replace("page", "").replace("pg", "").strip()
                filtered_manifest = [
                    m for m in filtered_manifest
                    if str(m.get("page_number")) != excl_target
                    and str(m.get("printed_page", "")).lower() != excl_target
                    and str(m.get("printed_page", "")).lower() != f"page {excl_target}"
                ]

            if section is not None or page_number is not None:
                return filtered_manifest

            return {
                "issue_id": issue.id,
                "newspaper": issue.newspaper.name if issue.newspaper else "Newspaper",
                "issue_date": str(issue.issue_date),
                "page_filter": str(page_filter) if page_filter else None,
                "exclude_page_filter": str(exclude_page_filter) if exclude_page_filter else None,
                "category_filter": category_filter,
                "total_articles": len(filtered_manifest),
                "total_issue_articles": len(articles),
                "total_pages": len(issue.pages),
                "section_breakdown": section_counts,
                "type_breakdown": type_counts,
                "category_breakdown": category_counts,
                "articles": filtered_manifest,
            }

    async def count_articles(
        self,
        newspaper_name: str | None = None,
        issue_date: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        section: str | None = None,
        article_type: str | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        """Return exact article count and matching article records matching filters."""
        from app.models.newspaper import Newspaper
        from app.models.article import ArticlePage, ArticleCategory
        from sqlalchemy import or_

        norm_date = normalize_date_to_iso(issue_date) if issue_date else None
        norm_from = normalize_date_to_iso(date_from) if date_from else None
        norm_to = normalize_date_to_iso(date_to) if date_to else None

        async with self._session_factory() as db:
            stmt = select(func.count(Article.id)).join(Issue, Article.issue_id == Issue.id)
            if newspaper_name:
                stmt = stmt.join(Newspaper, Issue.newspaper_id == Newspaper.id).where(
                    Newspaper.name.ilike(f"%{newspaper_name}%")
                )
            if norm_date:
                stmt = stmt.where(Issue.issue_date == norm_date)
            else:
                if norm_from:
                    stmt = stmt.where(Issue.issue_date >= norm_from)
                if norm_to:
                    stmt = stmt.where(Issue.issue_date <= norm_to)
            if section:
                stmt = stmt.outerjoin(ArticleCategory, Article.category_id == ArticleCategory.id).where(
                    or_(Article.section.ilike(f"%{section}%"), ArticleCategory.name.ilike(f"%{section}%"))
                )
            if article_type:
                stmt = stmt.where(Article.article_type == article_type)

            res = await db.execute(stmt)
            count = res.scalar() or 0

            # Detailed matching articles query for verification and citation pills
            stmt_arts = (
                select(
                    Article.id,
                    Article.headline,
                    Article.section,
                    Article.article_type,
                    Article.word_count,
                    Issue.issue_date,
                    Newspaper.name.label("newspaper_name"),
                    func.coalesce(func.min(ArticlePage.page_number), 1).label("page_number"),
                )
                .join(Issue, Article.issue_id == Issue.id)
                .join(Newspaper, Issue.newspaper_id == Newspaper.id)
                .outerjoin(ArticlePage, Article.id == ArticlePage.article_id)
            )
            if newspaper_name:
                stmt_arts = stmt_arts.where(Newspaper.name.ilike(f"%{newspaper_name}%"))
            if norm_date:
                stmt_arts = stmt_arts.where(Issue.issue_date == norm_date)
            else:
                if norm_from:
                    stmt_arts = stmt_arts.where(Issue.issue_date >= norm_from)
                if norm_to:
                    stmt_arts = stmt_arts.where(Issue.issue_date <= norm_to)
            if section:
                stmt_arts = stmt_arts.outerjoin(ArticleCategory, Article.category_id == ArticleCategory.id).where(
                    or_(Article.section.ilike(f"%{section}%"), ArticleCategory.name.ilike(f"%{section}%"))
                )
            if article_type:
                stmt_arts = stmt_arts.where(Article.article_type == article_type)

            stmt_arts = stmt_arts.group_by(
                Article.id,
                Article.headline,
                Article.section,
                Article.article_type,
                Article.word_count,
                Issue.issue_date,
                Newspaper.name,
            ).order_by(Issue.issue_date.desc(), func.min(ArticlePage.page_number).asc()).limit(limit)

            res_arts = await db.execute(stmt_arts)
            rows = res_arts.all()

            articles = [
                {
                    "article_id": r.id,
                    "headline": r.headline or "Untitled Article",
                    "section": r.section or "General",
                    "article_type": r.article_type,
                    "word_count": r.word_count,
                    "issue_date": str(r.issue_date),
                    "newspaper_name": r.newspaper_name,
                    "page_number": r.page_number,
                }
                for r in rows
            ]

            return {
                "count": count,
                "articles": articles,
                "filters": {
                    "newspaper_name": newspaper_name,
                    "issue_date": norm_date,
                    "date_from": norm_from,
                    "date_to": norm_to,
                    "section": section,
                    "article_type": article_type,
                },
            }

    async def count_advertisements(
        self,
        newspaper_name: str | None = None,
        issue_date: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        issue_id: int | None = None,
    ) -> dict[str, Any]:
        """Return exact advertisement count and article records matching filters."""
        from app.models.newspaper import Newspaper
        from app.models.article import ArticlePage

        norm_date = normalize_date_to_iso(issue_date) if issue_date else None
        norm_from = normalize_date_to_iso(date_from) if date_from else None
        norm_to = normalize_date_to_iso(date_to) if date_to else None

        async with self._session_factory() as db:
            stmt = (
                select(
                    Article.id,
                    Article.headline,
                    Article.section,
                    Article.article_type,
                    Article.word_count,
                    Issue.issue_date,
                    Newspaper.name.label("newspaper_name"),
                    func.coalesce(func.min(ArticlePage.page_number), 1).label("page_number"),
                )
                .join(Issue, Article.issue_id == Issue.id)
                .join(Newspaper, Issue.newspaper_id == Newspaper.id)
                .outerjoin(ArticlePage, Article.id == ArticlePage.article_id)
                .where(Article.article_type == "advertisement")
            )
            if issue_id:
                stmt = stmt.where(Issue.id == issue_id)
            if newspaper_name:
                stmt = stmt.where(Newspaper.name.ilike(f"%{newspaper_name}%"))
            if norm_date:
                stmt = stmt.where(Issue.issue_date == norm_date)
            else:
                if norm_from:
                    stmt = stmt.where(Issue.issue_date >= norm_from)
                if norm_to:
                    stmt = stmt.where(Issue.issue_date <= norm_to)

            stmt = stmt.group_by(
                Article.id,
                Article.headline,
                Article.section,
                Article.article_type,
                Article.word_count,
                Issue.issue_date,
                Newspaper.name,
            ).order_by(Issue.issue_date.desc(), func.min(ArticlePage.page_number).asc())

            res = await db.execute(stmt)
            rows = res.all()

            advertisements = [
                {
                    "article_id": r.id,
                    "headline": r.headline or "[Advertisement]",
                    "section": r.section or "Advertisements & Notices",
                    "article_type": r.article_type,
                    "word_count": r.word_count,
                    "issue_date": str(r.issue_date),
                    "newspaper_name": r.newspaper_name,
                    "page_number": r.page_number,
                }
                for r in rows
            ]

            return {
                "count": len(advertisements),
                "advertisements": advertisements,
                "filters": {
                    "newspaper_name": newspaper_name,
                    "issue_date": norm_date,
                    "date_from": norm_from,
                    "date_to": norm_to,
                    "issue_id": issue_id,
                },
            }

    async def count_issues(
        self,
        newspaper_name: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """Return count of issues matching newspaper and/or date filters."""
        from app.models.newspaper import Newspaper

        norm_from = normalize_date_to_iso(date_from) if date_from else None
        norm_to = normalize_date_to_iso(date_to) if date_to else None

        async with self._session_factory() as db:
            stmt = select(func.count(Issue.id))
            if newspaper_name:
                stmt = stmt.join(Newspaper, Issue.newspaper_id == Newspaper.id).where(
                    Newspaper.name.ilike(f"%{newspaper_name}%")
                )
            if norm_from:
                stmt = stmt.where(Issue.issue_date >= norm_from)
            if norm_to:
                stmt = stmt.where(Issue.issue_date <= norm_to)

            res = await db.execute(stmt)
            count = res.scalar() or 0
            return {
                "count": count,
                "filters": {
                    "newspaper_name": newspaper_name,
                    "date_from": norm_from,
                    "date_to": norm_to,
                },
            }

    async def get_photo_counts_by_section(
        self,
        newspaper_name: str | None = None,
        issue_date: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        issue_id: int | None = None,
        section: str | None = None,
    ) -> dict[str, Any]:
        """Return photo count aggregated by section."""
        norm_date = normalize_date_to_iso(issue_date) if issue_date else None
        norm_from = normalize_date_to_iso(date_from) if date_from else None
        norm_to = normalize_date_to_iso(date_to) if date_to else None

        async with self._session_factory() as db:
            stmt = (
                select(
                    func.coalesce(Article.section, "Unassigned").label("section"),
                    func.count(Photo.id).label("photo_count"),
                )
                .select_from(Photo)
                .join(Article, Photo.article_id == Article.id)
                .join(Issue, Article.issue_id == Issue.id)
            )
            if issue_id:
                stmt = stmt.where(Issue.id == issue_id)
            if newspaper_name:
                stmt = stmt.join(Newspaper, Issue.newspaper_id == Newspaper.id).where(
                    Newspaper.name.ilike(f"%{newspaper_name}%")
                )
            if norm_date:
                stmt = stmt.where(Issue.issue_date == norm_date)
            else:
                if norm_from:
                    stmt = stmt.where(Issue.issue_date >= norm_from)
                if norm_to:
                    stmt = stmt.where(Issue.issue_date <= norm_to)
            if section:
                stmt = stmt.where(Article.section.ilike(f"%{section}%"))

            stmt = stmt.group_by(func.coalesce(Article.section, "Unassigned")).order_by(
                desc("photo_count")
            )

            res = await db.execute(stmt)
            rows = res.all()

            section_counts = [{"section": r[0], "count": int(r[1])} for r in rows]
            total_photos = sum(sc["count"] for sc in section_counts)
            by_section = {sc["section"]: sc["count"] for sc in section_counts}

            return {
                "total_photos": total_photos,
                "section_counts": section_counts,
                "by_section": by_section,
                "filters": {
                    "newspaper_name": newspaper_name,
                    "issue_date": norm_date or issue_date,
                    "date_from": norm_from or date_from,
                    "date_to": norm_to or date_to,
                    "issue_id": issue_id,
                    "section": section,
                },
            }

    async def get_issue_summary(
        self,
        newspaper_name: str | None = None,
        issue_date: str | None = None,
        issue_id: int | None = None,
        page_filter: str | int | None = None,
        exclude_page_filter: str | int | None = None,
        category_filter: str | None = None,
        query: str | None = None,
    ) -> Any:
        """Alias for list_issue_articles to maintain backward compatibility."""
        return await self.list_issue_articles(
            issue_id=issue_id,
            newspaper_name=newspaper_name,
            issue_date=issue_date,
            page_filter=page_filter,
            exclude_page_filter=exclude_page_filter,
            category_filter=category_filter,
            query=query,
        )

    async def _match_shared_articles(
        self,
        clean_source: list[dict[str, Any]],
        clean_comp: list[dict[str, Any]],
        min_overlap: float = 0.40,
    ) -> list[dict[str, Any]]:
        """Core 2-Tier Hybrid Matcher with 1-to-1 Bipartite Deduplication."""
        all_art_ids = [a["id"] for a in clean_source if a.get("id")] + [a["id"] for a in clean_comp if a.get("id")]
        entities_by_art: dict[int, set[str]] = {}
        if all_art_ids and self._session_factory:
            try:
                async with self._session_factory() as db:
                    stmt = (
                        select(ArticleEntity.article_id, Entity.name)
                        .join(Entity, Entity.id == ArticleEntity.entity_id)
                        .where(ArticleEntity.article_id.in_(all_art_ids))
                    )
                    res = await db.execute(stmt)
                    rows = res.all() if hasattr(res, "all") else []
                    for aid, ename in rows:
                        clean_e = str(ename).lower().strip()
                        if clean_e not in _UBIQUITOUS_ENTITIES and len(clean_e) >= 3:
                            entities_by_art.setdefault(aid, set()).add(clean_e)
            except Exception as e:
                logger.debug("Could not fetch article entities for shared coverage matching", extra={"error": str(e)})

        candidates: list[tuple[float, dict[str, Any], dict[str, Any], str, str]] = []
        for sa in clean_source:
            stoks = extract_substantive_tokens(sa.get("headline", ""))
            snums = extract_numeric_anchors(sa.get("headline", ""))
            sentities = entities_by_art.get(sa.get("id"), set())
            for ca in clean_comp:
                ctoks = extract_substantive_tokens(ca.get("headline", ""))
                cnums = extract_numeric_anchors(ca.get("headline", ""))
                centities = entities_by_art.get(ca.get("id"), set())

                common_toks = stoks & ctoks
                common_nums = snums & cnums
                score = 0.0
                mtype = None
                shared_kw = sorted(common_toks | common_nums)
                if not shared_kw and sentities and centities:
                    shared_kw = sorted(sentities & centities)
                topic_hint = ", ".join(shared_kw) if shared_kw else (sa.get("section") or "General")

                # Tier 1: Substantive token Jaccard & numerical anchors
                if len(common_toks) >= 2 and min(len(stoks), len(ctoks)) > 0:
                    jaccard = len(common_toks) / min(len(stoks), len(ctoks))
                    if jaccard >= min_overlap:
                        score = jaccard
                        mtype = "token_jaccard"
                    elif jaccard >= 0.25 and bool(common_nums):
                        score = min(0.95, jaccard + 0.35)
                        mtype = "numerical_anchor"

                # Tier 2: Entity & Named-Entity Intersection (for rewritten wires)
                if score < 0.50 and sentities and centities:
                    common_ents = sentities & centities
                    if len(common_ents) >= 2 and (bool(common_toks) or bool(common_nums)):
                        ent_score = min(0.90, 0.55 + len(common_ents) * 0.10)
                        if ent_score > score:
                            score = ent_score
                            mtype = f"entity_intersection ({len(common_ents)} entities)"
                            topic_hint = ", ".join(sorted(common_ents | common_toks))

                if score >= min_overlap:
                    candidates.append((score, sa, ca, mtype or "similarity_match", topic_hint))

        # Bipartite 1-to-1 Maximum Weight Matching (Greedy Deduplication)
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_s: set[int] = set()
        used_c: set[int] = set()
        shared_articles: list[dict[str, Any]] = []

        for score, sa, ca, mtype, topic_hint in candidates:
            s_id = sa.get("id")
            c_id = ca.get("id")
            if s_id in used_s or c_id in used_c:
                continue
            if s_id:
                used_s.add(s_id)
            if c_id:
                used_c.add(c_id)

            shared_articles.append({
                "source_headline": sa.get("headline"),
                "headline_a": sa.get("headline"),
                "source_page": sa.get("page_number", 1),
                "page_a": sa.get("page_number", 1),
                "source_section": sa.get("section", "General"),
                "section_a": sa.get("section", "General"),
                "source_category": sa.get("category", "General"),
                "source_article_id": s_id,
                "article_id_a": s_id,
                "source_snippet": sa.get("summary") or sa.get("snippet", ""),
                "comparison_headline": ca.get("headline"),
                "headline_b": ca.get("headline"),
                "matched_headline": ca.get("headline"),  # backward compat alias
                "comparison_page": ca.get("page_number", 1),
                "page_b": ca.get("page_number", 1),
                "comparison_section": ca.get("section", "General"),
                "section_b": ca.get("section", "General"),
                "comparison_category": ca.get("category", "General"),
                "comparison_article_id": c_id,
                "article_id_b": c_id,
                "comparison_snippet": ca.get("summary") or ca.get("snippet", ""),
                "confidence": round(score, 2),
                "overlap_score": round(score, 2),  # backward compat alias
                "match_type": mtype,
                "match_tier": mtype,
                "shared_keywords": topic_hint,
                "topic": topic_hint,
            })

        return shared_articles

    async def get_newspaper_shared_coverage(
        self,
        source_newspaper: str | None = None,
        comparison_newspaper: str | None = None,
        issue_date: str | None = None,
        category_filter: str | None = None,
        query: str | None = None,
        min_overlap: float = 0.40,
        newspaper_a: str | None = None,
        newspaper_b: str | None = None,
    ) -> dict[str, Any]:
        """Compute verified shared syndicated and similar wire coverage between two newspapers on a given date."""
        src_np = source_newspaper or newspaper_a or ""
        cmp_np = comparison_newspaper or newspaper_b or ""

        source_summary = await self.list_issue_articles(
            newspaper_name=src_np,
            issue_date=issue_date,
            category_filter=category_filter,
            query=query,
        )
        if not isinstance(source_summary, dict) or "error" in source_summary:
            err = source_summary.get("error", "Source issue not found") if isinstance(source_summary, dict) else "Source error"
            return {"error": f"Source publication '{src_np}': {err}"}

        comp_summary = await self.list_issue_articles(
            newspaper_name=cmp_np,
            issue_date=issue_date,
            category_filter=category_filter,
            query=query,
        )
        if not isinstance(comp_summary, dict) or "error" in comp_summary:
            err = comp_summary.get("error", "Comparison issue not found") if isinstance(comp_summary, dict) else "Comparison error"
            return {"error": f"Comparison publication '{cmp_np}': {err}"}

        source_articles = source_summary.get("articles", [])
        comp_articles = comp_summary.get("articles", [])

        noise_hls = {
            "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
            "in short >>", "in short", "news in brief", "panaji", "margao", "vasco", "mapusa", "delhi", "mumbai",
            "when: ", "where: ", "studio xo", "sit vacant", "[advertisement]", "results", "epaper.morningstandard.in",
        }
        clean_source = [
            a for a in source_articles
            if len((a.get("headline") or "").strip()) >= 10
            and (a.get("headline") or "").strip().lower() not in noise_hls
        ]
        clean_comp = [
            a for a in comp_articles
            if len((a.get("headline") or "").strip()) >= 10
            and (a.get("headline") or "").strip().lower() not in noise_hls
        ]

        shared_articles = await self._match_shared_articles(clean_source, clean_comp, min_overlap=min_overlap)
        for story in shared_articles:
            story["newspaper_a"] = source_summary.get("newspaper", src_np)
            story["newspaper_b"] = comp_summary.get("newspaper", cmp_np)
            story["source_newspaper"] = source_summary.get("newspaper", src_np)
            story["comparison_newspaper"] = comp_summary.get("newspaper", cmp_np)

        return {
            "source_newspaper": source_summary.get("newspaper", src_np),
            "comparison_newspaper": comp_summary.get("newspaper", cmp_np),
            "newspaper_a": source_summary.get("newspaper", src_np),
            "newspaper_b": comp_summary.get("newspaper", cmp_np),
            "issue_date": str(source_summary.get("issue_date", issue_date)),
            "total_source_articles": len(source_articles),
            "total_comparison_articles": len(comp_articles),
            "total_newspaper_a_articles": len(source_articles),
            "total_newspaper_b_articles": len(comp_articles),
            "shared_count": len(shared_articles),
            "shared_articles": shared_articles,
            "shared_stories": shared_articles,
        }

    async def get_newspaper_coverage_difference(
        self,
        source_newspaper: str,
        comparison_newspaper: str,
        issue_date: str | None = None,
    ) -> dict[str, Any]:
        """Compute verified differential coverage: articles in source_newspaper absent from comparison_newspaper."""
        source_summary = await self.list_issue_articles(
            newspaper_name=source_newspaper,
            issue_date=issue_date,
        )
        if not isinstance(source_summary, dict) or "error" in source_summary:
            err = source_summary.get("error", "Source issue not found") if isinstance(source_summary, dict) else "Source error"
            return {"error": f"Source publication '{source_newspaper}': {err}"}

        comp_summary = await self.list_issue_articles(
            newspaper_name=comparison_newspaper,
            issue_date=issue_date,
        )
        if not isinstance(comp_summary, dict) or "error" in comp_summary:
            err = comp_summary.get("error", "Comparison issue not found") if isinstance(comp_summary, dict) else "Comparison error"
            return {"error": f"Comparison publication '{comparison_newspaper}': {err}"}

        source_articles = source_summary.get("articles", [])
        comp_articles = comp_summary.get("articles", [])

        noise_hls = {
            "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
            "in short >>", "in short", "news in brief", "panaji", "margao", "vasco", "mapusa",
        }
        clean_source = [
            a for a in source_articles
            if len((a.get("headline") or "").strip()) >= 10
            and (a.get("headline") or "").strip().lower() not in noise_hls
        ]
        clean_comp = [
            a for a in comp_articles
            if len((a.get("headline") or "").strip()) >= 10
            and (a.get("headline") or "").strip().lower() not in noise_hls
        ]

        shared_articles = await self._match_shared_articles(clean_source, clean_comp, min_overlap=0.40)
        shared_source_ids = {a.get("source_article_id") for a in shared_articles if a.get("source_article_id")}
        shared_source_hls = {str(a.get("source_headline", "")).strip().lower() for a in shared_articles}

        exclusive_articles: list[dict[str, Any]] = []
        for sa in source_articles:
            s_id = sa.get("id")
            s_hl = str(sa.get("headline", "")).strip()
            if s_id in shared_source_ids or s_hl.lower() in shared_source_hls:
                continue
            if len(s_hl) < 10 or s_hl.lower() in noise_hls:
                continue

            exclusive_articles.append({
                "id": s_id,
                "headline": s_hl,
                "page_number": sa.get("page_number", 1),
                "printed_page": sa.get("printed_page", "1"),
                "section": sa.get("section", "General"),
                "category": sa.get("category", "General"),
                "snippet": sa.get("summary") or sa.get("snippet", ""),
            })

        return {
            "source_newspaper": source_summary.get("newspaper", source_newspaper),
            "comparison_newspaper": comp_summary.get("newspaper", comparison_newspaper),
            "issue_date": str(source_summary.get("issue_date", issue_date)),
            "total_source_articles": len(source_articles),
            "total_comparison_articles": len(comp_articles),
            "exclusive_count": len(exclusive_articles),
            "shared_count": len(shared_articles),
            "exclusive_articles": exclusive_articles,
            "shared_articles": shared_articles,
        }

    async def get_issues_by_date(self, issue_date: str) -> list[Issue]:
        """Retrieve all newspaper issues for a given date with eager newspaper and page relationships."""
        normalized_date = normalize_date_to_iso(issue_date) or issue_date
        async with self._session_factory() as db:
            stmt = (
                select(Issue)
                .where(Issue.issue_date == normalized_date)
                .options(
                    selectinload(Issue.newspaper),
                    selectinload(Issue.pages),
                )
                .order_by(Issue.id)
            )
            res = await db.execute(stmt)
            return list(res.scalars().all())

    async def get_newspaper_id_by_name(self, newspaper_name: str) -> int | None:
        """Resolve newspaper_name to primary key ID via case-insensitive lookup."""
        if not newspaper_name:
            return None
        async with self._session_factory() as db:
            stmt = select(Newspaper.id).where(Newspaper.name.ilike(f"%{newspaper_name.strip()}%")).limit(1)
            res = await db.execute(stmt)
            return res.scalar_one_or_none()


def normalize_date_to_iso(date_str: str | None) -> str | None:
    """Normalize user or parameter dates (YYYY-MM-DD, DD/MM/YYYY, D/M/YYYY) to ISO format."""
    if not date_str:
        return None
    s = date_str.strip()
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", s):
        parts = s.split("-")
        return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if month > 12 and day <= 12:
            day, month = month, day
        return f"{year:04d}-{month:02d}-{day:02d}"
    return s


