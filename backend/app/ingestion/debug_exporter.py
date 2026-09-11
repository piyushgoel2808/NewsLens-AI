"""Re-export shim for debug_exporter backward compatibility.

Canonical implementation has moved to `app.ingestion.storage`.
"""

from __future__ import annotations

from app.ingestion.storage import DebugArtifactsExporter, _slugify

__all__ = [
    "DebugArtifactsExporter",
    "_slugify",
]
