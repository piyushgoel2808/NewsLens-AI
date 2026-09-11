"""Re-export shim for compressor backward compatibility.

Canonical implementation has moved to `app.ingestion.storage`.
"""

from __future__ import annotations

from app.ingestion.storage import compress_pdf, compress_pdf_bytes

__all__ = [
    "compress_pdf",
    "compress_pdf_bytes",
]
