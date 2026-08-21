"""SQLAlchemy ORM model: QueryLog."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.mysql import JSON, LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class QueryLog(Base):
    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String(255))
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str | None] = mapped_column(String(100))  # archetype name
    # Planner output: ordered list of tool calls with arguments
    plan_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    # Actual tool call inputs + outputs
    tool_calls_json: Mapped[list[Any] | None] = mapped_column(JSON)
    answer_text: Mapped[str | None] = mapped_column(LONGTEXT)
    # List of {newspaper, date, edition, page, article_id} citation dicts
    citations_json: Mapped[list[Any] | None] = mapped_column(JSON)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    model_provider: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, index=True
    )
