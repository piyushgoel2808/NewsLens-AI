"""PDF Pre-Ingestion Compression Layer.

Provides fast, lossless stream and structural compression for incoming PDF broadsheets
prior to storage in MinIO and downstream processing (Docling, MinerU, PyMuPDF, OCR).

Uses PyMuPDF (fitz) with garbage collection and stream deflation.
"""

from __future__ import annotations

from typing import Any

import pymupdf

from app.core.logging import get_logger

logger = get_logger(__name__)


def compress_pdf_bytes(pdf_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    """Compress PDF bytes using PyMuPDF lossless stream deflation and object deduplication.

    Args:
        pdf_bytes: Raw input PDF bytes.

    Returns:
        tuple of (processed_bytes, compression_metadata_dict).
        If compression fails, is not beneficial, or input is invalid, returns the original
        bytes with status explanation in metadata.
    """
    original_size = len(pdf_bytes) if pdf_bytes else 0
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        return pdf_bytes, {
            "original_bytes": original_size,
            "compressed_bytes": original_size,
            "saved_bytes": 0,
            "reduction_pct": 0.0,
            "status": "skipped_not_pdf",
        }

    doc: pymupdf.Document | None = None
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        if doc.is_encrypted:
            return pdf_bytes, {
                "original_bytes": original_size,
                "compressed_bytes": original_size,
                "saved_bytes": 0,
                "reduction_pct": 0.0,
                "status": "skipped_encrypted",
            }

        # Lossless optimization:
        # - garbage=4: merge duplicate objects, unreferenced stream cleaning
        # - clean=True: sanitize content streams
        # - deflate=True: deflate uncompressed streams
        # - deflate_images=True: compress uncompressed image streams
        # - deflate_fonts=True: compress uncompressed font streams
        compressed = doc.tobytes(
            garbage=4,
            deflate=True,
            clean=True,
            deflate_images=True,
            deflate_fonts=True,
        )

        comp_size = len(compressed)
        if comp_size == 0 or comp_size >= original_size:
            return pdf_bytes, {
                "original_bytes": original_size,
                "compressed_bytes": original_size,
                "saved_bytes": 0,
                "reduction_pct": 0.0,
                "status": "skipped_no_reduction",
            }

        saved_bytes = original_size - comp_size
        reduction_pct = round((saved_bytes / original_size) * 100.0, 2)

        return compressed, {
            "original_bytes": original_size,
            "compressed_bytes": comp_size,
            "saved_bytes": saved_bytes,
            "reduction_pct": reduction_pct,
            "status": "compressed",
        }
    except Exception as exc:
        logger.warning(
            "PDF compression failed; falling back to original bytes",
            extra={"error": str(exc), "original_bytes": original_size},
        )
        return pdf_bytes, {
            "original_bytes": original_size,
            "compressed_bytes": original_size,
            "saved_bytes": 0,
            "reduction_pct": 0.0,
            "status": "failed",
            "error": str(exc),
        }
    finally:
        if doc is not None:
            doc.close()


def compress_pdf(pdf_bytes: bytes) -> bytes:
    """Convenience wrapper returning only the compressed PDF bytes."""
    compressed_bytes, _ = compress_pdf_bytes(pdf_bytes)
    return compressed_bytes
