"""Add printed_page_number and ad flags to pages and article_pages.

Revision ID: 002
Revises: 001
Create Date: 2026-08-22
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "pages",
        sa.Column("printed_page_number", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "pages",
        sa.Column(
            "is_advertisement_page",
            sa.Boolean(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.add_column(
        "article_pages",
        sa.Column("printed_page_number", sa.String(length=50), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("article_pages", "printed_page_number")
    op.drop_column("pages", "is_advertisement_page")
    op.drop_column("pages", "printed_page_number")
