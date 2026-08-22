"""Unit tests for MinerU (magic-pdf) document layout and OCR engine provider."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from app.providers.base import (
    DocumentLayoutProvider,
    OCREngine,
)
from app.providers.mineru_provider import (
    MinerUProvider,
    detect_device_mode,
    parse_html_table_to_matrix,
    parse_markdown_table_to_matrix,
)


def _create_sample_pdf_bytes() -> bytes:
    """Create an in-memory 2-page PDF for testing."""
    doc = pymupdf.open()
    # Page 1: Multi-column article with title and table
    page1 = doc.new_page(width=600, height=800)
    page1.insert_text((50, 50), "MAJOR BREAKTHROUGH IN CLEAN ENERGY", fontsize=18)
    page1.insert_text(
        (50, 100),
        "By Staff Reporter\n\nSolar power production surged across the region.",
        fontsize=10,
    )
    table_md = (
        "| Quarter | Solar (MW) | Wind (MW) |\n"
        "|---|---|---|\n"
        "| Q1 | 1200 | 850 |\n"
        "| Q2 | 1450 | 920 |"
    )
    page1.insert_text((50, 200), table_md, fontsize=9)

    # Page 2: Second story
    page2 = doc.new_page(width=600, height=800)
    page2.insert_text((50, 50), "MARKET COMMODITY UPDATE", fontsize=16)
    page2.insert_text((50, 100), "Crude oil stabilized at lower trading bands.", fontsize=10)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestMinerUProvider:
    """Tests for MinerU Provider implementation and protocol conformance."""

    def test_protocol_conformance(self) -> None:
        """Verify MinerUProvider satisfies DocumentLayoutProvider and OCREngine."""
        provider = MinerUProvider()
        assert isinstance(provider, DocumentLayoutProvider)
        assert isinstance(provider, OCREngine)
        assert provider.provider_name == "mineru"
        assert provider.capability.supports_layout is True
        assert provider.capability.supports_vision is True

    def test_detect_device_mode(self) -> None:
        """Ensure device mode detects one of valid platforms."""
        device = detect_device_mode()
        assert device in ("cuda", "mps", "cpu")

    def test_ensure_magic_pdf_config(self, tmp_path: Path) -> None:
        """Ensure magic-pdf.json is initialized with models-dir and device-mode."""
        cfg_file = tmp_path / "magic-pdf.json"
        provider = MinerUProvider(
            models_dir=str(tmp_path / "models"),
            device_mode="cpu",
            config_path=str(cfg_file),
        )
        assert provider.provider_name == "mineru"
        assert cfg_file.exists()
        cfg_content = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert cfg_content["models-dir"] == str(tmp_path / "models")
        assert cfg_content["device-mode"] == "cpu"
        assert "table-config" in cfg_content

    def test_parse_markdown_table_to_matrix(self) -> None:
        """Verify Markdown table text parsing to 2D JSON matrix."""
        md = (
            "| Quarter | Solar (MW) | Wind (MW) |\n"
            "|---|---|---|\n"
            "| Q1 | 1200 | 850 |\n"
            "| Q2 | 1450 | 920 |"
        )
        headers, rows = parse_markdown_table_to_matrix(md)
        assert headers == ["Quarter", "Solar (MW)", "Wind (MW)"]
        assert len(rows) == 2
        assert rows[0] == ["Q1", "1200", "850"]
        assert rows[1] == ["Q2", "1450", "920"]

    def test_parse_html_table_to_matrix(self) -> None:
        """Verify HTML table parsing to 2D JSON matrix."""
        html = (
            "<table><tr><th>Sector</th><th>Growth</th></tr>"
            "<tr><td>Energy</td><td>15%</td></tr></table>"
        )
        headers, rows = parse_html_table_to_matrix(html)
        assert headers == ["Sector", "Growth"]
        assert rows == [["Energy", "15%"]]

    @pytest.mark.asyncio
    async def test_parse_pdf_document(self) -> None:
        """Verify end-to-end PDF parsing into MinerUParseResult with reading order nodes."""
        pdf_bytes = _create_sample_pdf_bytes()
        provider = MinerUProvider()
        results = await provider.parse_pdf_document(pdf_bytes=pdf_bytes)

        assert len(results) == 2
        page1 = results[0]
        assert page1.page_number == 1
        assert len(page1.nodes) > 0

        # Verify title node exists
        titles = [n for n in page1.nodes if n.node_type == "title"]
        assert len(titles) >= 1
        assert "MAJOR BREAKTHROUGH" in titles[0].text

        # Verify table node exists and contains structured matrix
        tables = [n for n in page1.nodes if n.node_type == "table"]
        assert len(tables) >= 1
        assert tables[0].table_data is not None
        assert "Quarter" in tables[0].table_data.headers

        # Verify 1D reading order indexing
        assert [n.reading_order for n in page1.nodes] == list(range(len(page1.nodes)))

    @pytest.mark.asyncio
    async def test_ocr_engine_protocol(self) -> None:
        """Verify OCR interface on image raster bytes."""
        doc = pymupdf.open()
        p = doc.new_page(width=300, height=300)
        p.insert_text((50, 50), "OCR TEST HEADING", fontsize=14)
        pix = p.get_pixmap()
        img_bytes = pix.tobytes("png")
        doc.close()

        provider = MinerUProvider()
        ocr_res = await provider.ocr(img_bytes)
        assert ocr_res.mean_confidence > 0.0
        assert "OCR TEST HEADING" in ocr_res.full_text
