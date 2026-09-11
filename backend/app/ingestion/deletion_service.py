"""Re-export shim for deletion_service backward compatibility.

Canonical implementation has moved to `app.ingestion.storage`.
"""

from __future__ import annotations

from app.ingestion.storage import DeletionService

__all__ = [
    "DeletionService",
]
