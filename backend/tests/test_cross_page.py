"""Unit tests for Cross-Page Story Continuation and Jump Assembler."""

from __future__ import annotations

from app.ingestion.cross_page_assembler import CrossPageAssembler
from app.ingestion.segmenter import SegmentedArticle


class TestCrossPageAssembler:
    """Test suite for CrossPageAssembler."""

    def test_empty_input_returns_empty(self) -> None:
        assembler = CrossPageAssembler()
        assert assembler.assemble_issue_articles({}) == []

    def test_assemble_single_page_articles(self) -> None:
        assembler = CrossPageAssembler()
        art1 = SegmentedArticle(
            article_temp_id="p1_art_1",
            headline="DISCOVERY OF NEW SPECIES",
            body_text="Biologists have discovered a new marine species in the deep ocean.",
            word_count=11,
            bbox_list=[(50.0, 50.0, 200.0, 200.0)],
        )
        pages_articles = {1: [art1]}
        assembled = assembler.assemble_issue_articles(pages_articles)

        assert len(assembled) == 1
        assert assembled[0].headline == "DISCOVERY OF NEW SPECIES"
        assert assembled[0].primary_page_number == 1
        assert len(assembled[0].pages_mapping) == 1
        assert assembled[0].pages_mapping[0].page_number == 1

    def test_assemble_cross_page_jump_story(self) -> None:
        assembler = CrossPageAssembler()

        # Page 1 portion jumping to Page 3
        page1_art = SegmentedArticle(
            article_temp_id="p1_art_1",
            headline="GLOBAL CLIMATE ACCORD SIGNED",
            byline_author="By Jane Smith",
            body_text=(
                "World leaders gathered in Geneva today to sign the historic treaty.\n"
                "Continued on Page 3"
            ),
            jump_to_page=3,
            word_count=14,
            bbox_list=[(30.0, 30.0, 250.0, 200.0)],
        )

        # Page 3 continuation portion
        page3_art = SegmentedArticle(
            article_temp_id="p3_art_1",
            headline="CLIMATE ACCORD (Continued from Page 1)",
            body_text=(
                "The terms mandate significant emission cuts by the year 2035 "
                "across all industrial sectors."
            ),
            jump_from_page=1,
            word_count=15,
            bbox_list=[(30.0, 50.0, 250.0, 300.0)],
        )

        # Independent Page 3 article
        page3_other = SegmentedArticle(
            article_temp_id="p3_art_2",
            headline="LOCAL SPORTS UPDATE",
            body_text="The home team secured a decisive victory in last night's championship game.",
            word_count=12,
            bbox_list=[(300.0, 50.0, 500.0, 300.0)],
        )

        pages_articles = {
            1: [page1_art],
            3: [page3_art, page3_other],
        }

        assembled = assembler.assemble_issue_articles(pages_articles)
        assert len(assembled) == 2

        # Verify stitched climate story
        climate_story = next(a for a in assembled if "CLIMATE" in a.headline)
        assert climate_story.primary_page_number == 1
        assert climate_story.byline_author == "By Jane Smith"
        assert "Geneva" in climate_story.full_text
        assert "emission cuts" in climate_story.full_text
        assert len(climate_story.pages_mapping) == 2
        assert climate_story.pages_mapping[0].page_number == 1
        assert climate_story.pages_mapping[1].page_number == 3
        assert climate_story.pages_mapping[1].block_order == 1

        # Verify independent sports story
        sports_story = next(a for a in assembled if "SPORTS" in a.headline)
        assert sports_story.primary_page_number == 3
        assert len(sports_story.pages_mapping) == 1

    def test_assemble_cross_page_shortened_jump_headline_with_containment(self) -> None:
        """Verify shortened jump headline stitches into lead story via containment."""
        assembler = CrossPageAssembler()
        p1_lead = SegmentedArticle(
            article_temp_id="p1_lead",
            headline="Cognizant beats IT peers with 5.6% jump in Q2 constant currency revenues",
            byline_author="By John Doe",
            body_text=(
                "Cognizant reported stellar growth in its financial services portfolio.\n"
                "Continued on Page 11"
            ),
            jump_to_page=11,
            word_count=18,
            bbox_list=[(30.0, 30.0, 400.0, 300.0)],
        )
        p11_cont = SegmentedArticle(
            article_temp_id="p11_cont",
            headline="COGNIZANT BEATS PEERS",
            byline_author="By John Doe",
            body_text=(
                "The company's digital transformation bookings surged over "
                "thirty percent year on year."
            ),
            jump_from_page=1,
            word_count=14,
            bbox_list=[(50.0, 50.0, 350.0, 250.0)],
        )

        assembled = assembler.assemble_issue_articles({1: [p1_lead], 11: [p11_cont]})
        assert len(assembled) == 1
        lead = assembled[0]
        assert "Cognizant beats IT peers" in lead.headline
        assert "stellar growth" in lead.full_text
        assert "digital transformation bookings surged" in lead.full_text
        assert len(lead.pages_mapping) == 2
        assert lead.pages_mapping[0].page_number == 1
        assert lead.pages_mapping[1].page_number == 11

    def test_anti_collision_rejects_unrelated_short_headline_match(self) -> None:
        """Verify unrelated 2-word headline without jump markers is not mistakenly merged."""
        assembler = CrossPageAssembler()
        p1_art = SegmentedArticle(
            article_temp_id="p1_tech",
            headline="Global technology firms expand AI research hubs across Asia",
            byline_author="By Alice Walker",
            body_text=(
                "New investments in computing infrastructure have doubled in "
                "the current quarter."
            ),
            word_count=16,
            bbox_list=[(30.0, 30.0, 400.0, 200.0)],
        )
        p4_art = SegmentedArticle(
            article_temp_id="p4_asia",
            headline="Asia Hubs",  # Very short headline, different author, NO continuation marker
            byline_author="By Bob Smith",
            body_text="Singapore and Tokyo reported record tourist arrivals in the spring season.",
            word_count=15,
            bbox_list=[(50.0, 50.0, 350.0, 200.0)],
        )

        assembled = assembler.assemble_issue_articles({1: [p1_art], 4: [p4_art]})
        # Must stay as 2 distinct independent articles
        assert len(assembled) == 2
