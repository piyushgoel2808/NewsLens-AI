"""Re-export shim for ocr_service backward compatibility.

Canonical implementation has moved to `app.ingestion.parsers.ocr`.
"""

from __future__ import annotations

from app.ingestion.parsers.ocr import OCRService

__all__ = [
    "OCRService",
]
