"""Add visual intelligence fields to photos and article_chunks.

Revision ID: 004
Revises: 003
Create Date: 2026-08-28
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add visual intelligence columns to photos
    op.add_column(
        "photos",
        sa.Column("vlm_description", sa.Text(), nullable=True),
    )
    op.add_column(
        "photos",
        sa.Column("visual_type", sa.String(length=50), nullable=True),
    )

    # 2. Add chunk_type to article_chunks
    op.add_column(
        "article_chunks",
        sa.Column(
            "chunk_type",
            sa.String(length=20),
            server_default="text",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("article_chunks", "chunk_type")
    op.drop_column("photos", "visual_type")
    op.drop_column("photos", "vlm_description")
