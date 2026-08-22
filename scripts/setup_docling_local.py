"""Setup and Warmup script for Docling local neural models and OCR engines.

Downloads, caches, and verifies all required Docling neural layout parsing,
table structure recognition, and OCR models on the local machine.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def setup_docling() -> None:
    print("=" * 70)
    print(" NewsLens-AI — Docling Local Setup & Warmup")
    print("=" * 70)

    print("\n[1/4] Verifying Docling core libraries...")
    import importlib.metadata

    try:
        import docling
        import pypdfium2
        import rapidocr
        import torch

        print(f"  ✓ docling version: {importlib.metadata.version('docling')}")
        print(f"  ✓ docling-core version: {importlib.metadata.version('docling-core')}")
        print(f"  ✓ pypdfium2 version: {importlib.metadata.version('pypdfium2')}")
        print(f"  ✓ rapidocr version: {importlib.metadata.version('rapidocr')}")

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps (Apple Silicon GPU)"
        print(f"  ✓ Hardware Acceleration: {device}")
    except Exception as e:
        print(f"  ✗ Import Error: {e}")
        print("  Please run 'make install' or 'cd backend && uv sync --all-extras'")
        sys.exit(1)

    print("\n[2/4] Initializing DocumentConverter & Pipeline Options...")
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    start_time = time.time()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    print(f"  ✓ DocumentConverter initialized in {time.time() - start_time:.2f}s")

    print("\n[3/4] Warming up layout, table, and OCR neural models...")
    sample_pdf = backend_dir / "tests" / "fixtures" / "sample_digital_frontpage.pdf"
    if sample_pdf.exists():
        conv_start = time.time()
        res = converter.convert(sample_pdf)
        doc = res.document
        item_count = len(list(doc.iterate_items()))
        print(f"  ✓ Model warmup complete in {time.time() - conv_start:.2f}s")
        print(f"  ✓ Document parsed: {len(doc.pages)} page(s), {item_count} structured element(s)")
    else:
        print("  ! Sample fixture not found; skipped test conversion.")

    print("\n[4/4] Verifying DoclingProvider adapter in NewsLens-AI...")
    from app.providers.docling_provider import DoclingProvider

    provider = DoclingProvider(lang="en", do_ocr=True, do_table_structure=True)
    print(f"  ✓ DoclingProvider loaded: provider_name='{provider.provider_name}'")
    print(f"  ✓ Capability: layout={provider.capability.supports_layout}, vision={provider.capability.supports_vision}")

    print("\n" + "=" * 70)
    print(" ✅ Docling Local Setup & Warmup Complete! All models ready offline.")
    print("=" * 70)


if __name__ == "__main__":
    setup_docling()
