"""Unit tests for Docling Document Layout and Reading Order Provider."""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from app.core.config import Settings
from app.ingestion.layout_analyzer import LayoutAnalyzer
from app.providers.base import (
    DocumentLayoutProvider,
    ExtractedDocumentNode,
    MinerUParseResult,
    OCREngine,
    OCRResult,
)
from app.providers.docling_provider import (
    DoclingProvider,
    detect_device_mode,
    parse_html_table_to_matrix,
    parse_markdown_table_to_matrix,
)
from app.providers.registry import ModelRegistry


def test_docling_provider_protocols() -> None:
    """Verify DoclingProvider satisfies both DocumentLayoutProvider and OCREngine protocols."""
    provider = DoclingProvider(lang="en")
    assert isinstance(provider, DocumentLayoutProvider)
    assert isinstance(provider, OCREngine)
    assert provider.provider_name == "docling"
    assert provider.capability.supports_layout is True
    assert provider.capability.supports_vision is True


def test_detect_device_mode() -> None:
    """Verify hardware accelerator detection returns a valid device string."""
    device = detect_device_mode()
    assert device in ("cuda", "mps", "cpu")


def test_parse_markdown_table_to_matrix() -> None:
    """Verify parsing Markdown table strings into headers and rows matrix."""
    md_table = """
    | Company | Revenue (Cr) | Profit (Cr) |
    | :--- | :--- | :--- |
    | Reliance Industries | 2,45,000 | 19,500 |
    | Tata Consultancy Services | 62,500 | 12,100 |
    | Infosys Technologies | 38,200 | 6,500 |
    """
    headers, rows = parse_markdown_table_to_matrix(md_table)
    assert headers == ["Company", "Revenue (Cr)", "Profit (Cr)"]
    assert len(rows) == 3
    assert rows[0] == ["Reliance Industries", "2,45,000", "19,500"]
    assert rows[1] == ["Tata Consultancy Services", "62,500", "12,100"]
    assert rows[2] == ["Infosys Technologies", "38,200", "6,500"]


def test_parse_html_table_to_matrix() -> None:
    """Verify parsing HTML table strings into headers and rows matrix."""
    html_table = """
    <table>
      <thead>
        <tr><th>Index</th><th>Value</th><th>Change</th></tr>
      </thead>
      <tbody>
        <tr><td>NIFTY 50</td><td>24,850.20</td><td>+125.40</td></tr>
        <tr><td>BSE SENSEX</td><td>81,420.50</td><td>+410.80</td></tr>
      </tbody>
    </table>
    """
    headers, rows = parse_html_table_to_matrix(html_table)
    assert headers == ["Index", "Value", "Change"]
    assert len(rows) == 2
    assert rows[0] == ["NIFTY 50", "24,850.20", "+125.40"]
    assert rows[1] == ["BSE SENSEX", "81,420.50", "+410.80"]


@pytest.mark.asyncio
async def test_docling_parse_pdf_document_live_fixture() -> None:
    """Test full document conversion and structured node extraction on fixture PDF."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_digital_frontpage.pdf"
    if not fixture_path.exists():
        pytest.skip("Fixture PDF not found")

    pdf_bytes = fixture_path.read_bytes()
    provider = DoclingProvider(lang="en", do_ocr=False, do_table_structure=True)

    results = await provider.parse_pdf_document(pdf_bytes=pdf_bytes, lang="en")
    assert len(results) >= 1
    page1 = results[0]
    assert isinstance(page1, MinerUParseResult)
    assert page1.page_number == 1
    assert len(page1.nodes) > 0

    # Verify structured nodes contain text and valid bounding boxes
    text_nodes = [n for n in page1.nodes if n.text]
    assert len(text_nodes) > 0
    for node in text_nodes:
        assert isinstance(node, ExtractedDocumentNode)
        assert node.bbox is not None
        assert len(node.bbox) == 4
        x0, y0, x1, y1 = node.bbox
        assert x1 >= x0
        assert y1 >= y0


@pytest.mark.asyncio
async def test_docling_parse_page_image_and_ocr() -> None:
    """Test single page raster image parsing and OCR extraction."""
    # Generate a simple test image with text
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    provider = DoclingProvider(lang="en", do_ocr=False)
    res = await provider.parse_page_image(image_bytes=img_bytes, page_number=2)
    assert isinstance(res, MinerUParseResult)
    assert res.page_number == 2

    ocr_res = await provider.ocr(image_bytes=img_bytes)
    assert isinstance(ocr_res, OCRResult)


@pytest.mark.asyncio
async def test_docling_model_registry_resolution() -> None:
    """Verify ModelRegistry correctly instantiates DoclingProvider for layout and OCR tasks."""
    settings = Settings()
    registry = ModelRegistry(settings=settings)
    # Explicitly verify docling provider instantiation
    provider = registry.get_provider_by_id("docling_parser")
    assert isinstance(provider, DoclingProvider)
    assert provider.provider_name == "docling"


@pytest.mark.asyncio
async def test_layout_analyzer_with_docling() -> None:
    """Verify LayoutAnalyzer cleanly consumes Docling pre-linearized reading order."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_digital_frontpage.pdf"
    if not fixture_path.exists():
        pytest.skip("Fixture PDF not found")

    pdf_doc = pymupdf.open(fixture_path)
    page = pdf_doc[0]
    pix = page.get_pixmap(dpi=150)
    image_bytes = pix.tobytes("png")
    width_px = pix.width
    height_px = pix.height
    pdf_doc.close()

    analyzer = LayoutAnalyzer(vision_provider=DoclingProvider())
    layout_res = await analyzer.analyze_page(
        page_number=1,
        width_px=width_px,
        height_px=height_px,
        image_bytes=image_bytes,
    )

    assert layout_res.page_number == 1
    assert layout_res.source in ("docling", "vlm", "heuristic")
    assert len(layout_res.elements) > 0
    assert len(layout_res.reading_order) > 0
    # Verify reading order indices are strictly sequential 1..N
    indices = [b.reading_order_index for b in layout_res.reading_order]
    assert indices == list(range(1, len(indices) + 1))
