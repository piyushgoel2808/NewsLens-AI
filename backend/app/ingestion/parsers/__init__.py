"""Document Parsing Engines Subpackage for NewsLens-AI Ingestion.

Unifies broadsheet layout understanding and transcription engines:
1. DoclingLayoutParser: Deep vision-based document layout parsing (DocLayNet + RapidOCR/PaddleOCR).
2. UnifiedExtractor: Multimodal VLM extraction (Gemini Vision / Gemma 4).
3. OCRService: RapidOCR scanned page orchestration.
4. schemas: Pydantic structured output models for layout and enrichment.
"""

from __future__ import annotations

from app.ingestion.parsers.docling import (
    CorruptedPdfTextLayerError,
    DoclingLayoutParser,
    DoclingParsedItem,
    ExtractedPhotoData,
)
from app.ingestion.parsers.ocr import OCRService
from app.ingestion.parsers.schemas import (
    ArticleEnrichment,
    ArticleGenre,
    ArticleSkeleton,
    ExtractedEntity,
    ExtractedTable,
    PageLayoutExtraction,
    ProminenceTier,
    SectionType,
)
from app.ingestion.parsers.vlm import (
    PHASE1_LAYOUT_PROMPT,
    PHASE2_ENRICH_PROMPT,
    UnifiedExtractor,
)

__all__ = [
    "ArticleEnrichment",
    "ArticleGenre",
    "ArticleSkeleton",
    "CorruptedPdfTextLayerError",
    "DoclingLayoutParser",
    "DoclingParsedItem",
    "ExtractedEntity",
    "ExtractedPhotoData",
    "ExtractedTable",
    "OCRService",
    "PHASE1_LAYOUT_PROMPT",
    "PHASE2_ENRICH_PROMPT",
    "PageLayoutExtraction",
    "ProminenceTier",
    "SectionType",
    "UnifiedExtractor",
]
