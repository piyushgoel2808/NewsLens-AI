"""NewsLens-AI Ingestion Subsystem.

Provides end-to-end processing pipeline for printed broadsheet newspapers:
- Intake and pre-compression of PDF/ZIP archives
- PyMuPDF 300 DPI page rasterization
- Digital text layer detection & gibberish classification
- Multi-column layout analysis, reading order resolution & ad envelopes
- RapidOCR masthead verification and printed folio detection
- Article boundary segmentation and cross-page jump continuation
- Multimodal visual extraction (photos, tables, charts)
- Topic classification and named entity extraction
- Newspaper-aware chunking and dense vector indexing
- Atomic 3-tier hard deletion (Qdrant, MinIO, MySQL)
"""

from __future__ import annotations

from app.ingestion.chunker import NewspaperChunker
from app.ingestion.classifier import ArticleClassifier
from app.ingestion.cross_page_assembler import CrossPageAssembler
from app.ingestion.deletion_service import DeletionService
from app.ingestion.detector import PDFPageDetector
from app.ingestion.docling_parser import DoclingLayoutParser
from app.ingestion.embedder import ArticleEmbedder
from app.ingestion.folio_detector import FolioDetector
from app.ingestion.intake import IntakeService
from app.ingestion.layout_analyzer import LayoutAnalyzer
from app.ingestion.masthead_verifier import MastheadVerifier
from app.ingestion.rasterizer import PDFRasterizer, RasterizedPage
from app.ingestion.segmenter import ArticleSegmenter
from app.ingestion.tasks import run_ingestion_pipeline
from app.ingestion.unified_extractor import UnifiedExtractor

__all__ = [
    "ArticleClassifier",
    "ArticleEmbedder",
    "ArticleSegmenter",
    "CrossPageAssembler",
    "DeletionService",
    "DoclingLayoutParser",
    "FolioDetector",
    "IntakeService",
    "LayoutAnalyzer",
    "MastheadVerifier",
    "NewspaperChunker",
    "PDFPageDetector",
    "PDFRasterizer",
    "RasterizedPage",
    "UnifiedExtractor",
    "run_ingestion_pipeline",
]
