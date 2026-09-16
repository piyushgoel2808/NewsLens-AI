"""NewsLens-AI Ingestion Subsystem.

Provides end-to-end processing pipeline for printed broadsheet newspapers:
- Intake and pre-compression of PDF/ZIP archives (storage.py)
- PyMuPDF 150 DPI page rasterization (rasterizer.py)
- Digital text layer detection & gibberish classification (detector.py)
- Multi-column layout analysis, reading order resolution & ad envelopes (layout/)
- RapidOCR masthead verification, consensus extraction & printed folio detection (metadata.py)
- Article boundary segmentation and cross-page jump continuation (layout/)
- Multimodal document parsing and visual extraction (parsers/)
- Topic classification and named entity extraction (classifier.py)
- Newspaper-aware chunking and dense vector indexing (chunker.py, embedder.py)
- Atomic 3-tier hard deletion (storage.py)
"""

from __future__ import annotations

from app.ingestion.chunker import NewspaperChunker
from app.ingestion.classifier import ArticleClassifier
from app.ingestion.detector import PDFPageDetector
from app.ingestion.embedder import ArticleEmbedder
from app.ingestion.intake import IntakeService
from app.ingestion.layout import (
    ArticleSegmenter,
    CrossPageAssembler,
    LayoutAnalyzer,
    ReadingOrderResolver,
)
from app.ingestion.media_extractor import MediaExtractor
from app.ingestion.metadata import (
    ConsensusExtractor,
    FolioDetector,
    MastheadVerifier,
    extract_newspaper_and_date_consensus,
)
from app.ingestion.metadata_extractor import MetadataExtractor
from app.ingestion.parsers import (
    DoclingLayoutParser,
    OCRService,
    UnifiedExtractor,
)
from app.ingestion.rasterizer import (
    PDFRasterizer,
    RasterizedPage,
)
from app.ingestion.single_pass_extractor import (
    PageVisualAnalysis,
    RegionAnalysis,
    SinglePassVisualExtractor,
    VisualRegion,
    VisualRegionResult,
)
from app.ingestion.storage import (
    DebugArtifactsExporter,
    DeletionService,
    compress_pdf,
    compress_pdf_bytes,
)
from app.ingestion.tasks import run_ingestion_pipeline
from app.ingestion.visual_extractor import VisualDataExtractor

__all__ = [
    "ArticleClassifier",
    "ArticleEmbedder",
    "ArticleSegmenter",
    "ConsensusExtractor",
    "CrossPageAssembler",
    "DebugArtifactsExporter",
    "DeletionService",
    "DoclingLayoutParser",
    "FolioDetector",
    "IntakeService",
    "LayoutAnalyzer",
    "MastheadVerifier",
    "MediaExtractor",
    "MetadataExtractor",
    "NewspaperChunker",
    "OCRService",
    "PDFPageDetector",
    "PDFRasterizer",
    "PageVisualAnalysis",
    "RasterizedPage",
    "ReadingOrderResolver",
    "RegionAnalysis",
    "SinglePassVisualExtractor",
    "UnifiedExtractor",
    "VisualDataExtractor",
    "VisualRegion",
    "VisualRegionResult",
    "compress_pdf",
    "compress_pdf_bytes",
    "extract_newspaper_and_date_consensus",
    "run_ingestion_pipeline",
]
