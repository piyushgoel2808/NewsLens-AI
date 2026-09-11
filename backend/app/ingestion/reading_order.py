"""Re-export shim for reading_order backward compatibility.

Canonical implementation has moved to `app.ingestion.layout.analyzer`.
"""

from __future__ import annotations

from app.ingestion.layout.analyzer import (
    BlockType,
    LayoutElement,
    OrderedReadingBlock,
    ReadingOrderResolver,
)

__all__ = [
    "BlockType",
    "LayoutElement",
    "OrderedReadingBlock",
    "ReadingOrderResolver",
]
