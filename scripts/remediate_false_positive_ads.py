"""NewsLens-AI False Positive Advertisement Remediation & Embedding Script.

Scans all articles falsely tagged with '[Advertisement]' due to naive substring
matching (e.g. 'ipo' matching 'bipolarity'). Restores them to genuine news articles,
generates missing contextual chunks, and indexes dense vector points into Qdrant.

Usage:
    uv run python scripts/remediate_false_positive_ads.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "backend"))
sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.ingestion.chunker import NewspaperChunker
from app.ingestion.detector import check_is_advertisement_text
from app.ingestion.embedder import ArticleEmbedder
from app.models.article import Article, ArticleChunk, ArticlePage
from app.models.base import get_session_factory, init_db
from app.models.entity import ArticleEntity, ArticleTopic
from app.models.newspaper import Issue, Newspaper
from app.storage.qdrant_store import QdrantStore

setup_logging("INFO")
logger = get_logger("remediate_ads")


async def remediate_false_positive_ads(dry_run: bool = False) -> None:
    settings = get_settings()
    init_db(settings.database.async_url)
    session_factory = get_session_factory()
    qdrant = QdrantStore(settings.qdrant)
    chunker = NewspaperChunker()

    async with session_factory() as db:
        embedder = ArticleEmbedder(db=db, qdrant=qdrant)

        stmt = (
            select(Article)
            .where(
                (Article.headline.like("[Advertisement]%"))
                | (Article.article_type == "advertisement")
            )
            .options(
                selectinload(Article.article_pages),
                selectinload(Article.article_entities).selectinload(ArticleEntity.entity),
                selectinload(Article.article_topics).selectinload(ArticleTopic.topic),
            )
        )
        res = await db.execute(stmt)
        candidates = res.scalars().all()
        logger.info(f"Found {len(candidates)} candidate advertisement articles to inspect.")

        remediated_count = 0
        embedded_count = 0
        total_chunks_created = 0

        for art in candidates:
            full_txt = art.full_text or ""
            words = full_txt.split()
            w_count = len(words)
            raw_hl = art.headline or ""

            # Explicit notice keywords
            is_explicit_notice = any(
                p in raw_hl.lower()
                for p in ["public notice", "tender notice", "e-tender", "statutory notice", "auction sale notice"]
            )
            if is_explicit_notice:
                continue

            # Evaluate against real multi-signal ad detector
            is_detector_ad = check_is_advertisement_text(full_txt, word_count=w_count)
            if is_detector_ad:
                continue

            # This is a false positive!
            remediated_count += 1
            clean_hl = raw_hl
            if clean_hl.startswith("[Advertisement] "):
                clean_hl = clean_hl[len("[Advertisement] "):]
            elif clean_hl.startswith("[Advertisement]"):
                clean_hl = clean_hl[len("[Advertisement]"):].strip()

            logger.info(
                f"Remediating Article {art.id}: '{clean_hl[:60]}...' "
                f"(byline={art.byline_author}, words={w_count})"
            )

            if not dry_run:
                art.headline = clean_hl
                if art.article_type == "advertisement":
                    art.article_type = "news"

                # Check if chunks already exist
                chunk_check = await db.execute(
                    select(ArticleChunk.id).where(ArticleChunk.article_id == art.id)
                )
                existing_chunk_ids = chunk_check.scalars().all()

                if not existing_chunk_ids:
                    issue_res = await db.execute(
                        select(Issue).where(Issue.id == art.issue_id).options(selectinload(Issue.newspaper))
                    )
                    issue = issue_res.scalar_one_or_none()
                    np_name = issue.newspaper.name if issue and issue.newspaper else "Daily News"
                    issue_dt_str = issue.issue_date.isoformat() if issue else ""

                    pages = [ap.page_number for ap in art.article_pages] or [1]
                    printed_pages = [
                        ap.printed_page_number or str(ap.page_number)
                        for ap in art.article_pages
                    ] or [str(p) for p in pages]

                    entities = [ae.entity.name for ae in art.article_entities if ae.entity]
                    topics = [at.topic.name for at in art.article_topics if at.topic]

                    chunks = chunker.chunk_article(
                        full_text=full_txt,
                        newspaper_name=np_name,
                        issue_date=issue_dt_str,
                        headline=clean_hl,
                        section=art.section or "National",
                        pages=pages,
                        printed_pages=printed_pages,
                    )

                    if chunks:
                        await embedder.embed_and_index_chunks(
                            article_id=art.id,
                            issue_id=art.issue_id,
                            newspaper_name=np_name,
                            issue_date=issue_dt_str,
                            headline=clean_hl,
                            section=art.section,
                            article_type=art.article_type,
                            prominence_score=art.prominence_score or 0.5,
                            page_numbers=pages,
                            printed_pages=printed_pages,
                            entities=entities,
                            topics=topics,
                            chunks=chunks,
                            has_photo=False,
                            has_table=False,
                            chunk_type="text",
                        )
                        embedded_count += 1
                        total_chunks_created += len(chunks)

        if not dry_run:
            await db.commit()
            logger.info(
                f"Remediation complete: {remediated_count} articles corrected, "
                f"{embedded_count} articles chunked & embedded ({total_chunks_created} total chunks created)."
            )
        else:
            logger.info(f"[DRY-RUN] Would remediate {remediated_count} articles.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remediate false-positive advertisement articles.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry-run without modifying DB.")
    args = parser.parse_args()

    asyncio.run(remediate_false_positive_ads(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
