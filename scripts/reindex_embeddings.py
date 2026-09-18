#!/usr/bin/env python3
"""Background re-indexing utility for NewsLens-AI.

Reads existing ArticleChunk records from MySQL and re-embeds them using
Google Gemini Embedding (gemini-embedding-001, 768d MRL) into the Qdrant
article_chunks_v2 collection.

Usage:
    python scripts/reindex_embeddings.py [--limit N] [--batch-size 50] [--issue-id ID] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.article import Article, ArticleChunk, ArticlePage, Photo, ArticleTable
from app.models.newspaper import Issue, Newspaper
from app.providers.gemini_embedding_provider import GeminiEmbeddingProvider
from app.storage.base import VectorPoint
from app.models.base import close_db, get_session_factory
from app.storage.qdrant_store import QdrantStore

logger = get_logger("reindex_embeddings")


async def reindex(
    issue_id: int | None = None,
    limit: int | None = None,
    batch_size: int = 50,
    dry_run: bool = False,
) -> None:
    settings = get_settings()
    session_factory = get_session_factory()
    target_collection = getattr(settings.qdrant, "collection_name_v2", "article_chunks_v2")

    print(f"=== NewsLens-AI Re-indexing to {target_collection} (768d) ===")
    print(f"Batch size: {batch_size} | Issue filter: {issue_id or 'ALL'} | Limit: {limit or 'ALL'} | Dry Run: {dry_run}")

    # 1. Initialize Gemini embedding provider
    embed_provider = GeminiEmbeddingProvider(
        model="gemini-embedding-001",
        embedding_dim=768,
        api_key=settings.gemini_api_key or settings.google_api_key,
        service_account_info=settings.gcp_service_account_json or settings.gcp_service_account_key,
        project_id=settings.gcp_project_id,
    )
    print(f"Initialized provider: {embed_provider.provider_name} (dim: {embed_provider.embedding_dim})")

    # 2. Initialize Qdrant store and ensure collections
    qdrant = QdrantStore(settings.qdrant, embedding_dim=768)
    if not dry_run:
        await qdrant._ensure_collection()
        print(f"Ensured Qdrant collection {target_collection} exists and indexed.")

    # 3. Query chunks from MySQL
    t0 = time.monotonic()
    async with session_factory() as db:
        query = (
            select(ArticleChunk)
            .join(Article, ArticleChunk.article_id == Article.id)
            .options(
                selectinload(ArticleChunk.article).selectinload(Article.article_pages),
                selectinload(ArticleChunk.article).selectinload(Article.photos),
                selectinload(ArticleChunk.article).selectinload(Article.tables),
                selectinload(ArticleChunk.article).selectinload(Article.issue).selectinload(Issue.newspaper),
            )
            .order_by(ArticleChunk.id)
        )
        if issue_id is not None:
            query = query.where(Article.issue_id == issue_id)
        if limit is not None:
            query = query.limit(limit)

        result = await db.execute(query)
        chunks: list[ArticleChunk] = list(result.scalars().all())

    total_chunks = len(chunks)
    print(f"Found {total_chunks} chunks to process.")
    if total_chunks == 0:
        print("No chunks found matching criteria. Done.")
        await qdrant.close()
        return

    # 4. Process in batches
    indexed_count = 0
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]

        b_t0 = time.monotonic()
        if dry_run:
            print(f"[{i + 1}-{i + len(batch)}/{total_chunks}] DRY RUN: would embed {len(batch)} chunks")
            continue

        # Embed with RETRIEVAL_DOCUMENT
        vectors = await embed_provider.embed(texts, task_type="RETRIEVAL_DOCUMENT")

        points: list[VectorPoint] = []
        for chunk, vector in zip(batch, vectors, strict=False):
            art = chunk.article
            iss = art.issue if art else None
            np = iss.newspaper if iss else None

            point_id = chunk.embedding_vector_id or f"chunk-{chunk.id}"
            pages = [ap.page_number for ap in art.article_pages] if art and art.article_pages else []

            payload = {
                "article_id": chunk.article_id,
                "issue_id": iss.id if iss else None,
                "newspaper_name": np.name if np else "Daily",
                "issue_date": str(iss.issue_date) if iss else "",
                "headline": art.headline if art else "",
                "section": (art.section if art else None) or "General",
                "article_type": (art.article_type if art else None) or "unknown",
                "prominence_score": art.prominence_score if art else 0.0,
                "has_photo": bool(art.photos) if art else False,
                "has_table": bool(art.tables) if art else False,
                "chunk_type": chunk.chunk_type,
                "chunk_index": chunk.chunk_index,
                "page_numbers": pages,
                "chunk_text": chunk.text,
            }

            points.append(
                VectorPoint(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Upsert into article_chunks_v2
        await qdrant.upsert(points, collection_name=target_collection)
        indexed_count += len(points)
        b_time = round((time.monotonic() - b_t0) * 1000)
        print(f"[{indexed_count}/{total_chunks}] Indexed {len(points)} chunks into {target_collection} ({b_time}ms)")

    elapsed = round(time.monotonic() - t0, 2)
    print(f"\n Successfully indexed {indexed_count} chunks into '{target_collection}' in {elapsed}s.")

    # Show info
    if not dry_run:
        info = await qdrant.collection_info(target_collection)
        print(f"Collection status: {info}")

    await qdrant.close()
    await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-index NewsLens-AI chunks into article_chunks_v2")
    parser.add_argument("--issue-id", type=int, help="Filter by specific issue ID")
    parser.add_argument("--limit", type=int, help="Maximum number of chunks to re-index")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding API (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing to Qdrant")
    args = parser.parse_args()

    asyncio.run(
        reindex(
            issue_id=args.issue_id,
            limit=args.limit,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
