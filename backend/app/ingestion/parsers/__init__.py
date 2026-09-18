"""Document Parsing Engines Subpackage for NewsLens-AI Ingestion.

Unifies broadsheet layout understanding and transcription engines:
1. DoclingLayoutParser: Deep vision-based document layout parsing (DocLayNet + RapidOCR/PaddleOCR).
2. UnifiedExtractor: Multimodal VLM extraction (Gemini Vision / Gemma 4).
3. GFVSPageSegmenter: Gemini-First Vision Segmentation for scanned pages.
4. OCRService: RapidOCR scanned page orchestration.
5. schemas: Pydantic structured output models for layout and enrichment.
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
from app.ingestion.parsers.gfvs import (
    GFVS_LAYOUT_PROMPT,
    GFVSArticle,
    GFVSPageResult,
    GFVSPageSegmenter,
    resolve_gemini_provider,
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
    "GFVS_LAYOUT_PROMPT",
    "GFVSArticle",
    "GFVSPageResult",
    "GFVSPageSegmenter",
    "OCRService",
    "PHASE1_LAYOUT_PROMPT",
    "PHASE2_ENRICH_PROMPT",
    "PageLayoutExtraction",
    "ProminenceTier",
    "SectionType",
    "UnifiedExtractor",
    "resolve_gemini_provider",
]
