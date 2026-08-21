"""Initial schema — all 16 tables for NewsLens-AI.

Creates the complete database schema including:
- Core entities: newspapers, issues, pages
- Articles: articles, article_pages, article_chunks, photos, tables
- Metadata: entities, article_entities, topics, article_topics, events, article_events
- System: ingestion_jobs, query_log

Post-table DDL:
- MySQL FULLTEXT index on articles(headline, full_text) for lexical search
- B-tree indexes for structured query performance

Revision ID: 001
Revises: None
Create Date: 2026-08-21
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -------------------------------------------------------------------------
    # 1. newspapers
    # -------------------------------------------------------------------------
    op.create_table(
        "newspapers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("publisher", sa.String(255), nullable=True),
        sa.Column("default_language", sa.String(10), nullable=True),
        sa.Column("country", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_newspapers"),
    )

    # -------------------------------------------------------------------------
    # 2. ingestion_jobs (referenced by issues.source_zip_id)
    # -------------------------------------------------------------------------
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "source_type",
            sa.Enum("single_pdf", "multi_pdf", "folder", "zip", name="source_type_enum"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "completed", "failed", "partial", name="job_status_enum"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("total_files", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_files", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_files", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_log", mysql.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_ingestion_jobs"),
    )

    # -------------------------------------------------------------------------
    # 3. issues
    # -------------------------------------------------------------------------
    op.create_table(
        "issues",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("newspaper_id", sa.Integer(), nullable=False),
        sa.Column("issue_date", sa.Date(), nullable=False),
        sa.Column("edition", sa.String(100), nullable=True),
        sa.Column("language", sa.String(10), nullable=True),
        sa.Column("total_pages", sa.Integer(), nullable=True),
        sa.Column("source_zip_id", sa.Integer(), nullable=True),
        sa.Column("ingestion_status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["newspaper_id"], ["newspapers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_zip_id"], ["ingestion_jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_issues"),
        sa.UniqueConstraint("newspaper_id", "issue_date", "edition", name="uq_issue_newspaper_date_edition"),
    )
    op.create_index("ix_issues_newspaper_id", "issues", ["newspaper_id"])
    op.create_index("ix_issues_issue_date", "issues", ["issue_date"])
    op.create_index("ix_issues_newspaper_date", "issues", ["newspaper_id", "issue_date"])

    # -------------------------------------------------------------------------
    # 4. pages
    # -------------------------------------------------------------------------
    op.create_table(
        "pages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("issue_id", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("raster_object_key", sa.String(512), nullable=True),
        sa.Column("width_px", sa.Integer(), nullable=True),
        sa.Column("height_px", sa.Integer(), nullable=True),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("ingestion_status", sa.String(50), nullable=False, server_default="pending"),
        sa.ForeignKeyConstraint(["issue_id"], ["issues.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_pages"),
        sa.UniqueConstraint("issue_id", "page_number", name="uq_page_issue_number"),
    )
    op.create_index("ix_pages_issue_id", "pages", ["issue_id"])

    # -------------------------------------------------------------------------
    # 5. articles
    # -------------------------------------------------------------------------
    op.create_table(
        "articles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("issue_id", sa.Integer(), nullable=False),
        sa.Column("primary_page_id", sa.Integer(), nullable=True),
        sa.Column("headline", sa.String(1024), nullable=True),
        sa.Column("subheadline", sa.String(1024), nullable=True),
        sa.Column("byline_author", sa.String(512), nullable=True),
        sa.Column("section", sa.String(255), nullable=True),
        sa.Column(
            "article_type",
            sa.Enum(
                "news", "editorial", "sidebar", "advertisement",
                "photo_caption", "table_content", "continuation", "unknown",
                name="article_type_enum",
            ),
            nullable=False,
            server_default="unknown",
        ),
        sa.Column("language", sa.String(10), nullable=True),
        sa.Column("prominence_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("word_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("full_text", mysql.LONGTEXT(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["issue_id"], ["issues.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["primary_page_id"], ["pages.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_articles"),
    )
    op.create_index("ix_articles_issue_id", "articles", ["issue_id"])
    op.create_index("ix_articles_primary_page_id", "articles", ["primary_page_id"])
    op.create_index("ix_articles_section", "articles", ["section"])
    op.create_index("ix_articles_language", "articles", ["language"])
    op.create_index("ix_articles_prominence", "articles", ["prominence_score"])

    # -------------------------------------------------------------------------
    # 6. article_pages
    # -------------------------------------------------------------------------
    op.create_table(
        "article_pages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("page_id", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("bbox_json", mysql.JSON(), nullable=True),
        sa.Column("block_order", sa.Integer(), nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["page_id"], ["pages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_article_pages"),
    )
    op.create_index("ix_article_pages_article_id", "article_pages", ["article_id"])
    op.create_index("ix_article_pages_page_id", "article_pages", ["page_id"])
    op.create_index("ix_article_pages_article_page", "article_pages", ["article_id", "page_number"])

    # -------------------------------------------------------------------------
    # 7. article_chunks
    # -------------------------------------------------------------------------
    op.create_table(
        "article_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding_vector_id", sa.String(255), nullable=True),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_article_chunks"),
    )
    op.create_index("ix_article_chunks_article_id", "article_chunks", ["article_id"])

    # -------------------------------------------------------------------------
    # 8. entities (self-referential FK added after table exists)
    # -------------------------------------------------------------------------
    op.create_table(
        "entities",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column(
            "type",
            sa.Enum("person", "org", "location", "misc", name="entity_type_enum"),
            nullable=False,
        ),
        sa.Column("canonical_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["canonical_id"], ["entities.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_entities"),
        sa.UniqueConstraint("name", "type", name="uq_entity_name_type"),
    )

    # -------------------------------------------------------------------------
    # 9. article_entities
    # -------------------------------------------------------------------------
    op.create_table(
        "article_entities",
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("mention_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("salience_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["entity_id"], ["entities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("article_id", "entity_id", name="pk_article_entities"),
    )
    op.create_index("ix_article_entities_entity_id", "article_entities", ["entity_id"])

    # -------------------------------------------------------------------------
    # 10. topics
    # -------------------------------------------------------------------------
    op.create_table(
        "topics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("taxonomy_path", sa.String(1024), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_topics"),
        sa.UniqueConstraint("name", name="uq_topics_name"),
    )

    # -------------------------------------------------------------------------
    # 11. article_topics
    # -------------------------------------------------------------------------
    op.create_table(
        "article_topics",
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("topic_id", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("article_id", "topic_id", name="pk_article_topics"),
    )
    op.create_index("ix_article_topics_topic_id", "article_topics", ["topic_id"])

    # -------------------------------------------------------------------------
    # 12. events (self-referential FK)
    # -------------------------------------------------------------------------
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(1024), nullable=False),
        sa.Column("canonical_date", sa.Date(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("event_cluster_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["event_cluster_id"], ["events.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_events"),
    )

    # -------------------------------------------------------------------------
    # 13. article_events
    # -------------------------------------------------------------------------
    op.create_table(
        "article_events",
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("article_id", "event_id", name="pk_article_events"),
    )
    op.create_index("ix_article_events_event_id", "article_events", ["event_id"])

    # -------------------------------------------------------------------------
    # 14. photos
    # -------------------------------------------------------------------------
    op.create_table(
        "photos",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=True),
        sa.Column("page_id", sa.Integer(), nullable=False),
        sa.Column("bbox_json", mysql.JSON(), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("object_key", sa.String(512), nullable=True),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["page_id"], ["pages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_photos"),
    )
    op.create_index("ix_photos_page_id", "photos", ["page_id"])

    # -------------------------------------------------------------------------
    # 15. tables
    # -------------------------------------------------------------------------
    op.create_table(
        "tables",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=True),
        sa.Column("page_id", sa.Integer(), nullable=False),
        sa.Column("bbox_json", mysql.JSON(), nullable=True),
        sa.Column("extracted_json", mysql.JSON(), nullable=True),
        sa.Column("object_key", sa.String(512), nullable=True),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["page_id"], ["pages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_tables"),
    )
    op.create_index("ix_tables_page_id", "tables", ["page_id"])

    # -------------------------------------------------------------------------
    # 16. query_log
    # -------------------------------------------------------------------------
    op.create_table(
        "query_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(255), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_type", sa.String(100), nullable=True),
        sa.Column("plan_json", mysql.JSON(), nullable=True),
        sa.Column("tool_calls_json", mysql.JSON(), nullable=True),
        sa.Column("answer_text", mysql.LONGTEXT(), nullable=True),
        sa.Column("citations_json", mysql.JSON(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("model_provider", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_query_log"),
    )
    op.create_index("ix_query_log_created_at", "query_log", ["created_at"])

    # =========================================================================
    # Post-table: FULLTEXT index (MySQL-specific — must use raw SQL)
    # Note: op.create_index() does not support FULLTEXT in Alembic reliably.
    # =========================================================================
    op.execute(
        "ALTER TABLE articles ADD FULLTEXT INDEX ft_articles_headline_text (headline, full_text)"
    )


def downgrade() -> None:
    # Drop in reverse dependency order
    op.drop_table("query_log")
    op.drop_table("tables")
    op.drop_table("photos")
    op.drop_table("article_events")
    op.drop_table("events")
    op.drop_table("article_topics")
    op.drop_table("topics")
    op.drop_table("article_entities")
    op.drop_table("entities")
    op.drop_table("article_chunks")
    op.drop_table("article_pages")
    op.drop_table("articles")
    op.drop_table("pages")
    op.drop_table("issues")
    op.drop_table("ingestion_jobs")
    op.drop_table("newspapers")

    # Drop MySQL ENUMs (they remain as types if not cleaned up)
    op.execute("DROP TYPE IF EXISTS article_type_enum")
    op.execute("DROP TYPE IF EXISTS entity_type_enum")
    op.execute("DROP TYPE IF EXISTS source_type_enum")
    op.execute("DROP TYPE IF EXISTS job_status_enum")
