"""Unit tests for Docling 2D Neural Layout Parser."""

from __future__ import annotations

import pytest

from app.ingestion.docling_parser import (
    _AUTHOR_NAME_PATTERN,
    _DATELINE_PATTERN,
    DoclingLayoutParser,
    DoclingParsedItem,
)


class TestDoclingLayoutParser:
    """Test suite for DoclingLayoutParser."""

    def test_parser_initialization(self) -> None:
        parser = DoclingLayoutParser()
        assert parser is not None
        assert hasattr(parser, "parse_page")
        assert hasattr(parser, "assemble_articles")

    def test_author_byline_patterns(self) -> None:
        assert bool(_AUTHOR_NAME_PATTERN.match("Manu Pubby")) is True
        assert bool(_AUTHOR_NAME_PATTERN.match("By Krishna Kumar")) is True
        assert bool(_AUTHOR_NAME_PATTERN.match("Dipanjan Roy Chaudhury")) is True
        assert bool(_AUTHOR_NAME_PATTERN.match("NEW DELHI: Hindustan Aeronautics Ltd")) is False

    def test_dateline_patterns(self) -> None:
        assert bool(_DATELINE_PATTERN.match("New Delhi: Hindustan Aeronautics Ltd")) is True
        assert bool(_DATELINE_PATTERN.match("Mumbai: The Bombay High Court")) is True
        assert bool(_DATELINE_PATTERN.match("HAL Safran deal announced")) is False

    def test_assemble_articles_separates_stories(self) -> None:
        parser = DoclingLayoutParser()
        items = [
            DoclingParsedItem(
                label="title",
                text="HAL, Safran to Develop Next-gen Helicopter Engine",
                bbox=(50.0, 100.0, 400.0, 130.0),
                page_number=7,
                level=1,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="Manu Pubby",
                bbox=(50.0, 135.0, 200.0, 150.0),
                page_number=7,
                level=2,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="New Delhi: Hindustan Aeronautics Ltd and Safran have signed a major agreement to co-develop helicopter engines.",
                bbox=(50.0, 155.0, 400.0, 250.0),
                page_number=7,
                level=2,
            ),
            DoclingParsedItem(
                label="title",
                text="MPSC Admits Drug Inspector Paper Leak",
                bbox=(450.0, 100.0, 800.0, 130.0),
                page_number=7,
                level=1,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="Krishna Kumar",
                bbox=(450.0, 135.0, 600.0, 150.0),
                page_number=7,
                level=2,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="Mumbai: The Maharashtra Public Service Commission admitted to a question paper leak.",
                bbox=(450.0, 155.0, 800.0, 250.0),
                page_number=7,
                level=2,
            ),
        ]

        articles = parser.assemble_articles(
            page_number=7,
            items=items,
            width_px=1000,
            height_px=1400,
        )

        assert len(articles) == 2
        assert articles[0].headline == "HAL, Safran to Develop Next-gen Helicopter Engine"
        assert articles[0].byline_author == "Manu Pubby"
        assert "Safran" in articles[0].body_text

        assert articles[1].headline == "MPSC Admits Drug Inspector Paper Leak"
        assert articles[1].byline_author == "Krishna Kumar"
        assert "Maharashtra" in articles[1].body_text

    def test_assemble_articles_coalesces_title_and_deck(self) -> None:
        parser = DoclingLayoutParser()
        items = [
            DoclingParsedItem(
                label="title",
                text="U.S. farmers struggle to get basic services",
                bbox=(50.0, 50.0, 500.0, 90.0),
                page_number=5,
                level=1,
            ),
            DoclingParsedItem(
                label="section_header",
                text="Lower government staffing has led to problems with loans and infrastructure BY LINDA QIU",
                bbox=(50.0, 95.0, 500.0, 140.0),
                page_number=5,
                level=2,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="WASHINGTON — Mary and Zachariah Box, farmers in New Mexico, believed they were on the verge of buying property.",
                bbox=(50.0, 150.0, 500.0, 300.0),
                page_number=5,
                level=3,
            ),
        ]

        articles = parser.assemble_articles(
            page_number=5,
            items=items,
            width_px=1000,
            height_px=1400,
        )

        assert len(articles) == 1
        assert articles[0].headline == "U.S. farmers struggle to get basic services"
        assert articles[0].subheadline == "Lower government staffing has led to problems with loans and infrastructure"
        assert articles[0].byline_author == "Linda Qiu"
        assert "Mary and Zachariah Box" in articles[0].body_text

    def test_extract_page_media_items(self) -> None:
        parser = DoclingLayoutParser()
        items = [
            DoclingParsedItem(
                label="picture",
                text="",
                bbox=(200.0, 200.0, 800.0, 600.0),
                page_number=5,
            ),
            DoclingParsedItem(
                label="caption",
                text="Mary and Zachariah Box turned to private lenders when the loan they sought from the administration stalled.",
                bbox=(200.0, 605.0, 800.0, 640.0),
                page_number=5,
            ),
        ]

        photos = parser.extract_page_media_items(items)
        assert len(photos) == 1
        assert photos[0].bbox == (200.0, 200.0, 800.0, 600.0)
        assert "Mary and Zachariah Box" in (photos[0].caption or "")
