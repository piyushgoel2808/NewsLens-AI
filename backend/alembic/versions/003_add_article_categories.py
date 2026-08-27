"""Add article_categories table and category fields to articles.

Revision ID: 003
Revises: 002
Create Date: 2026-08-28
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SEED_CATEGORIES = [
    "Politics",
    "Business & Markets",
    "Sports",
    "Entertainment",
    "World/International",
    "Technology",
    "Health",
    "Crime & Law",
    "Opinion/Editorial",
    "Lifestyle",
    "Science & Environment",
    "Local/Metro",
    "Other",
]


def upgrade() -> None:
    # 1. Create article_categories table
    op.create_table(
        "article_categories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["parent_id"], ["article_categories.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # 2. Seed canonical categories
    categories_table = sa.table(
        "article_categories",
        sa.column("name", sa.String),
    )
    op.bulk_insert(
        categories_table,
        [{"name": cat} for cat in SEED_CATEGORIES],
    )

    # 3. Add category columns to articles
    op.add_column(
        "articles",
        sa.Column("category_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "articles",
        sa.Column("category_confidence", sa.Float(), nullable=True),
    )
    op.add_column(
        "articles",
        sa.Column("printed_section", sa.String(length=128), nullable=True),
    )
    op.create_foreign_key(
        "fk_articles_category_id_article_categories",
        "articles",
        "article_categories",
        ["category_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_articles_category_id",
        "articles",
        ["category_id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_articles_category_id_article_categories", "articles", type_="foreignkey")
    op.drop_index("ix_articles_category_id", table_name="articles")
    op.drop_column("articles", "printed_section")
    op.drop_column("articles", "category_confidence")
    op.drop_column("articles", "category_id")
    op.drop_table("article_categories")
