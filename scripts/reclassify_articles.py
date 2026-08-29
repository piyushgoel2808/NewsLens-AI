"""NewsLens-AI Universal Article Re-Classification & Multi-Topic Migration Script.

Re-evaluates all articles in MySQL using the enhanced 12-domain multi-signal
probabilistic classifier:
1. Re-scores Headline (3x), Subheadline (2x), and Body (1x).
2. Applies domain context anchor dampening (metaphor collision disambiguation).
3. Sets accurate primary category (Sports, Science, Tech, Business, Entertainment, etc.).
4. Harmonizes physical section tags (preserving Front Page, Opinion & Editorial, etc.).
5. Populates secondary cross-cutting domains into Topic and ArticleTopic junction table.

Usage:
    uv run python scripts/reclassify_articles.py [--issue-id <ID>]
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
from app.ingestion.classifier import ArticleClassifier
from app.ingestion.cross_page_assembler import AssembledArticle
from app.models.article import Article, ArticleCategory
from app.models.base import get_session_factory, init_db
from app.models.entity import ArticleTopic, Topic
from app.models.newspaper import Issue, Page

setup_logging("INFO")
logger = get_logger("reclassify_articles")


async def reclassify_archive(issue_id_filter: int | None = None) -> None:
    settings = get_settings()
    init_db(settings.database.async_url)
    session_factory = get_session_factory()
    classifier = ArticleClassifier()

    async with session_factory() as db:
        stmt = (
            select(Article)
            .options(
                selectinload(Article.category),
                selectinload(Article.article_topics),
            )
        )
        if issue_id_filter:
            stmt = stmt.where(Article.issue_id == issue_id_filter)

        res = await db.execute(stmt)
        articles = res.scalars().all()

        logger.info(f"Loaded {len(articles)} articles for re-classification")

        updated_count = 0
        topics_added = 0
        cat_cache: dict[str, int] = {}
        topic_cache: dict[str, int] = {}

        # Preload Page ID -> Page Number map
        page_res = await db.execute(select(Page.id, Page.page_number))
        page_num_map = dict(page_res.all())

        # Preload categories
        cat_res = await db.execute(select(ArticleCategory))
        for c in cat_res.scalars().all():
            cat_cache[c.name] = c.id

        # Preload topics
        t_res = await db.execute(select(Topic))
        for t in t_res.scalars().all():
            topic_cache[t.name] = t.id

        for idx, art in enumerate(articles, 1):
            page_num = page_num_map.get(art.primary_page_id, 1) if art.primary_page_id else 1
            assembled = AssembledArticle(
                headline=art.headline or "Untitled",
                subheadline=art.subheadline,
                byline_author=art.byline_author,
                full_text=art.full_text or "",
                primary_page_number=page_num,
                word_count=art.word_count,
                printed_section=art.printed_section,
            )

            class_res = classifier.classify_and_score(
                article=assembled,
                printed_section=art.printed_section,
            )

            # 1. Update Primary Category
            if class_res.category:
                cat_id = cat_cache.get(class_res.category)
                if not cat_id:
                    new_cat = ArticleCategory(name=class_res.category)
                    db.add(new_cat)
                    await db.flush()
                    cat_id = new_cat.id
                    cat_cache[class_res.category] = cat_id
                art.category_id = cat_id
                art.category_confidence = class_res.category_confidence

            # 2. Update Section & Prominence
            art.section = class_res.section or art.section
            art.prominence_score = class_res.prominence_score

            # 3. Update Multi-Topic Secondary Categories
            if class_res.secondary_categories:
                existing_topic_ids = {at.topic_id for at in art.article_topics}
                for sec_cat, sec_conf in class_res.secondary_categories:
                    t_id = topic_cache.get(sec_cat)
                    if not t_id:
                        new_topic = Topic(name=sec_cat, taxonomy_path=f"Newsroom > {sec_cat}")
                        db.add(new_topic)
                        await db.flush()
                        t_id = new_topic.id
                        topic_cache[sec_cat] = t_id

                    if t_id not in existing_topic_ids:
                        at = ArticleTopic(
                            article_id=art.id,
                            topic_id=t_id,
                            confidence=sec_conf,
                        )
                        db.add(at)
                        topics_added += 1
                        existing_topic_ids.add(t_id)

            updated_count += 1
            if idx % 100 == 0:
                await db.commit()
                logger.info(f"Progress: {idx}/{len(articles)} articles re-classified...")

        await db.commit()
        logger.info(
            f"Re-classification complete! Processed {updated_count} articles, added {topics_added} secondary topic tags."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reclassify newspaper articles into 12 canonical domains")
    parser.add_argument("--issue-id", type=int, default=None, help="Optional issue ID filter")
    args = parser.parse_args()

    asyncio.run(reclassify_archive(args.issue_id))


if __name__ == "__main__":
    main()
