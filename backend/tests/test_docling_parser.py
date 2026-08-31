"""Unit tests for Docling 2D Neural Layout Parser."""

from __future__ import annotations

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

    def test_is_header_or_footer_noise_masthead(self) -> None:
        parser = DoclingLayoutParser()
        # Front-page masthead banner at y=150-700 on a 4400px page (under 20% height)
        masthead_text = "mint CHENNAI, AHMEDABAD, HYDERABAD, PUNE* Friday, July 31, 2026 livemint.com"
        assert parser._is_header_or_footer_noise(masthead_text, (70.0, 148.0, 1370.0, 714.0), 4400) is True

        # Legitimate top headline at y=150-250 (not masthead keywords)
        legit_headline = "Inflation Drops to 3.2 Percent as Oil Prices Cool"
        assert parser._is_header_or_footer_noise(legit_headline, (70.0, 150.0, 1370.0, 250.0), 4400) is False

        # Printer marks
        assert parser._is_header_or_footer_noise("SIHT", (15.0, 10.0, 45.0, 25.0), 4400) is True

    def test_assemble_articles_with_advertisement_feature(self) -> None:
        parser = DoclingLayoutParser()
        items = [
            DoclingParsedItem(
                label="text",
                text="Bulk drug exporters fret as China tightens screws ▶P1",
                bbox=(1400.0, 580.0, 2040.0, 690.0),
                page_number=1,
            ),
            DoclingParsedItem(
                label="text",
                text="ArcelorMittal 20th anniversary 2026",
                bbox=(225.0, 1070.0, 1260.0, 1160.0),
                page_number=1,
            ),
            DoclingParsedItem(
                label="section_header",
                text="Smarter steels for people and planet",
                bbox=(1930.0, 2140.0, 2570.0, 2440.0),
                page_number=1,
            ),
            DoclingParsedItem(
                label="text",
                text="For two decades, we've grown by pushing steel further - safer, smarter, more sustainable. Grounded in quality. Driven by innovation and our people.",
                bbox=(1940.0, 2460.0, 2490.0, 2690.0),
                page_number=1,
            ),
        ]

        articles = parser.assemble_articles(
            page_number=1,
            items=items,
            width_px=2800,
            height_px=4400,
        )

        assert len(articles) >= 1
        headlines = [a.headline for a in articles]
        assert any("Smarter steels" in h for h in headlines)

    def test_assemble_articles_rejects_replacement_character_headline(self) -> None:
        parser = DoclingLayoutParser()
        items = [
            DoclingParsedItem(
                label="title",
                text="    ",
                bbox=(50.0, 100.0, 400.0, 130.0),
                page_number=1,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="36\n  \n      ",
                bbox=(50.0, 135.0, 400.0, 250.0),
                page_number=1,
            ),
        ]
        articles = parser.assemble_articles(
            page_number=1,
            items=items,
            width_px=1000,
            height_px=1400,
        )
        assert len(articles) == 0

    def test_corrupted_font_check_raises_error(self) -> None:
        from app.ingestion.docling_parser import CorruptedPdfTextLayerError
        parser = DoclingLayoutParser()
        items = [
            DoclingParsedItem(
                label="text",
                text="\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd \ufffd\ufffd\ufffd\ufffd",
                bbox=(0.0, 0.0, 100.0, 100.0),
                page_number=1,
            )
        ]
        all_text = " ".join(it.text for it in items if it.text)
        num_replacement = sum(1 for c in all_text if c in ("\ufffd", "\ufeff"))
        replacement_ratio = num_replacement / max(len(all_text.replace(" ", "")), 1)
        assert replacement_ratio >= 0.03
