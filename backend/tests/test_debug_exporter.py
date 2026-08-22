"""Tests for DebugArtifactsExporter and debug artifact REST endpoints."""

import json
from pathlib import Path

from app.ingestion.debug_exporter import DebugArtifactsExporter


class TestDebugArtifactsExporter:
    def test_export_all_debug_artifacts(self, tmp_path: Path) -> None:
        exporter = DebugArtifactsExporter(base_output_dir=tmp_path)

        page_extractions = [
            {
                "page_number": 1,
                "printed_page_number": "1",
                "page_type": "scanned",
                "requires_ocr": True,
                "ocr_confidence": 0.95,
                "dimensions": {"width_px": 1200, "height_px": 1600},
                "total_blocks": 2,
                "blocks": [
                    {
                        "block_index": 0,
                        "text": "COGNIZANT BEATS IT PEERS",
                        "bbox": [100.0, 200.0, 500.0, 250.0],
                        "confidence": 0.98,
                        "font_size": 24.0,
                        "block_type": "headline",
                    },
                    {
                        "block_index": 1,
                        "text": "Cognizant posted strong first-quarter numbers.",
                        "bbox": [100.0, 260.0, 500.0, 600.0],
                        "confidence": 0.95,
                        "font_size": 10.0,
                        "block_type": "body",
                    },
                ],
                "full_page_text": (
                    "COGNIZANT BEATS IT PEERS\n"
                    "Cognizant posted strong first-quarter numbers."
                ),
            }
        ]

        rag_chunks = [
            {
                "chunk_index": 0,
                "chunk_id": "chunk-101-0",
                "article_id": 101,
                "headline": "Cognizant beats IT peers, cuts outlook",
                "section": "Corporate",
                "article_type": "news",
                "pages_spanned": [1],
                "printed_pages_spanned": ["1"],
                "word_count": 50,
                "character_count": 300,
                "text": "Cognizant posted strong first-quarter numbers...",
                "is_indexed_in_vector_db": True,
            }
        ]

        articles = [
            {
                "article_id": 101,
                "headline": "Cognizant beats IT peers, cuts outlook",
                "subheadline": "Strong Q1 results offset by macro headwinds",
                "byline_author": "Staff Reporter",
                "section": "Corporate",
                "article_type": "news",
                "prominence_score": 0.85,
                "word_count": 250,
                "pages_spanned": [1],
                "printed_pages_spanned": ["1"],
                "summary": "Cognizant beat Street estimates in Q1.",
                "entities": ["Cognizant", "India"],
                "topics": ["IT Services", "Earnings"],
                "full_text": "Cognizant posted strong first-quarter numbers...",
            }
        ]

        advertisements = [
            {
                "ad_id": 102,
                "headline_or_banner": "JUNIPER GREEN ENERGY IPO",
                "article_type": "advertisement",
                "pages_spanned": [1],
                "printed_pages_spanned": ["1"],
                "word_count": 35,
                "text_content": "Initial Public Offer of Equity Shares...",
                "bboxes": [{"page": 1, "bboxes": [[0.0, 0.0, 1200.0, 1600.0]]}],
            }
        ]

        summary_metrics = {
            "total_rendered_pages": 1,
            "total_articles": 1,
            "total_rag_chunks": 1,
            "total_advertisements": 1,
        }

        exported = exporter.export_issue_artifacts(
            issue_id=42,
            newspaper_name="Mint",
            issue_date="2026-07-30",
            edition="morning",
            page_extractions=page_extractions,
            rag_chunks=rag_chunks,
            articles=articles,
            advertisements=advertisements,
            summary_metrics=summary_metrics,
        )

        assert "ocr_extracted_text" in exported
        assert "rag_chunks" in exported
        assert "articles_manifest" in exported
        assert "identified_advertisements" in exported
        assert "ingestion_summary" in exported

        # Verify OCR Extracted Text JSON file
        ocr_file = Path(exported["ocr_extracted_text"])
        assert ocr_file.exists()
        ocr_data = json.loads(ocr_file.read_text(encoding="utf-8"))
        assert ocr_data["issue_id"] == 42
        assert ocr_data["total_pages"] == 1
        assert len(ocr_data["pages"][0]["blocks"]) == 2

        # Verify RAG Chunks JSON file
        chunks_file = Path(exported["rag_chunks"])
        assert chunks_file.exists()
        chunks_data = json.loads(chunks_file.read_text(encoding="utf-8"))
        assert chunks_data["total_chunks"] == 1
        assert chunks_data["chunks"][0]["chunk_id"] == "chunk-101-0"

        # Verify Articles Manifest JSON file
        articles_file = Path(exported["articles_manifest"])
        assert articles_file.exists()
        art_data = json.loads(articles_file.read_text(encoding="utf-8"))
        assert art_data["total_articles"] == 1
        assert art_data["articles"][0]["headline"] == "Cognizant beats IT peers, cuts outlook"

        # Verify Identified Advertisements JSON file
        ads_file = Path(exported["identified_advertisements"])
        assert ads_file.exists()
        ads_data = json.loads(ads_file.read_text(encoding="utf-8"))
        assert ads_data["total_advertisements"] == 1
        assert ads_data["advertisements"][0]["headline_or_banner"] == "JUNIPER GREEN ENERGY IPO"

        # Verify Ingestion Summary JSON file
        summary_file = Path(exported["ingestion_summary"])
        assert summary_file.exists()
        sum_data = json.loads(summary_file.read_text(encoding="utf-8"))
        assert sum_data["total_articles"] == 1
        assert sum_data["total_chunks"] == 1
