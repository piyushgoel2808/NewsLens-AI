"""Debug Artifacts Exporter for NewsLens-AI.

Exports structured JSON artifacts during newspaper ingestion to enable
detailed inspection, testing, and debugging of:
1. Raw OCR extractions (from MinerU Neural OCR / PDF Page Detector).
2. Hierarchical RAG chunks generated for vector retrieval.
3. Discrete segmented editorial articles manifest.
4. Identified advertisements and commercial notices.
5. Comprehensive ingestion summary metrics.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


def _slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    clean = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", clean)


class DebugArtifactsExporter:
    """Exports structured debug JSON files for an ingested newspaper issue."""

    def __init__(self, base_output_dir: str | Path = "debug_output") -> None:
        self.base_output_dir = Path(base_output_dir)

    def get_issue_debug_dir(
        self,
        issue_id: int,
        newspaper_name: str,
        issue_date: str,
        edition: str = "morning",
    ) -> Path:
        """Construct the output directory path for an issue."""
        np_slug = _slugify(newspaper_name) or "daily"
        date_slug = _slugify(str(issue_date)) or "unknown_date"
        edition_slug = _slugify(edition) or "default"
        folder_name = f"{np_slug}_{date_slug}_{edition_slug}_issue_{issue_id}"
        target_dir = self.base_output_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def export_issue_artifacts(
        self,
        issue_id: int,
        newspaper_name: str,
        issue_date: str,
        edition: str = "morning",
        page_extractions: list[dict[str, Any]] | None = None,
        rag_chunks: list[dict[str, Any]] | None = None,
        articles: list[dict[str, Any]] | None = None,
        advertisements: list[dict[str, Any]] | None = None,
        summary_metrics: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Export all 5 structured debug JSON files for an issue.

        Returns a dictionary mapping artifact names to their absolute file paths.
        """
        target_dir = self.get_issue_debug_dir(
            issue_id=issue_id,
            newspaper_name=newspaper_name,
            issue_date=issue_date,
            edition=edition,
        )

        exported_files: dict[str, str] = {}

        # 1. OCR Extracted Text
        ocr_payload = {
            "issue_id": issue_id,
            "newspaper_name": newspaper_name,
            "issue_date": str(issue_date),
            "edition": edition,
            "exported_at": datetime.now(UTC).isoformat(),
            "total_pages": len(page_extractions or []),
            "pages": page_extractions or [],
        }
        ocr_path = target_dir / "ocr_extracted_text.json"
        with ocr_path.open("w", encoding="utf-8") as f:
            json.dump(self._sanitize(ocr_payload), f, indent=2, ensure_ascii=False)
        exported_files["ocr_extracted_text"] = str(ocr_path.resolve())

        # 2. RAG Chunks
        chunks_payload = {
            "issue_id": issue_id,
            "newspaper_name": newspaper_name,
            "issue_date": str(issue_date),
            "edition": edition,
            "exported_at": datetime.now(UTC).isoformat(),
            "total_chunks": len(rag_chunks or []),
            "chunks": rag_chunks or [],
        }
        chunks_path = target_dir / "rag_chunks.json"
        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump(self._sanitize(chunks_payload), f, indent=2, ensure_ascii=False)
        exported_files["rag_chunks"] = str(chunks_path.resolve())

        # 3. Articles Manifest
        articles_payload = {
            "issue_id": issue_id,
            "newspaper_name": newspaper_name,
            "issue_date": str(issue_date),
            "edition": edition,
            "exported_at": datetime.now(UTC).isoformat(),
            "total_articles": len(articles or []),
            "articles": articles or [],
        }
        articles_path = target_dir / "articles_manifest.json"
        with articles_path.open("w", encoding="utf-8") as f:
            json.dump(self._sanitize(articles_payload), f, indent=2, ensure_ascii=False)
        exported_files["articles_manifest"] = str(articles_path.resolve())

        # 4. Identified Advertisements
        ads_payload = {
            "issue_id": issue_id,
            "newspaper_name": newspaper_name,
            "issue_date": str(issue_date),
            "edition": edition,
            "exported_at": datetime.now(UTC).isoformat(),
            "total_advertisements": len(advertisements or []),
            "advertisements": advertisements or [],
        }
        ads_path = target_dir / "identified_advertisements.json"
        with ads_path.open("w", encoding="utf-8") as f:
            json.dump(self._sanitize(ads_payload), f, indent=2, ensure_ascii=False)
        exported_files["identified_advertisements"] = str(ads_path.resolve())

        # 5. Ingestion Summary
        summary_payload = {
            "issue_id": issue_id,
            "newspaper_name": newspaper_name,
            "issue_date": str(issue_date),
            "edition": edition,
            "exported_at": datetime.now(UTC).isoformat(),
            "output_directory": str(target_dir.resolve()),
            "total_pages": len(page_extractions or []),
            "total_articles": len(articles or []),
            "total_chunks": len(rag_chunks or []),
            "total_advertisements": len(advertisements or []),
            "metrics": summary_metrics or {},
            "generated_files": exported_files,
        }
        summary_path = target_dir / "ingestion_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(self._sanitize(summary_payload), f, indent=2, ensure_ascii=False)
        exported_files["ingestion_summary"] = str(summary_path.resolve())

        logger.info(
            "Debug artifacts exported successfully",
            extra={
                "issue_id": issue_id,
                "output_dir": str(target_dir),
                "total_articles": len(articles or []),
                "total_chunks": len(rag_chunks or []),
            },
        )

        return exported_files

    def _sanitize(self, data: Any) -> Any:
        """Recursively convert non-serializable objects into JSON-compatible types."""
        from collections.abc import Mapping, Sequence
        from enum import Enum

        if data is None or isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, bytes):
            return f"<bytes len={len(data)}>"
        if isinstance(data, Enum):
            return data.value
        if is_dataclass(data) and not isinstance(data, type):
            return self._sanitize(asdict(data))
        if isinstance(data, Mapping):
            return {str(k): self._sanitize(v) for k, v in data.items()}
        if isinstance(data, (list, tuple, set, Sequence)) and not isinstance(
            data, (str, bytes, bytearray)
        ):
            return [self._sanitize(item) for item in data]
        if hasattr(data, "isoformat") and callable(data.isoformat):
            return data.isoformat()
        if hasattr(data, "__dict__"):
            return self._sanitize(dict(data.__dict__))
        return str(data)
