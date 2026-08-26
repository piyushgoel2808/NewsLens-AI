"""Unit tests for the UnifiedExtractor and Extraction Schemas."""

from app.ingestion.extraction_schemas import (
    ArticleEnrichment,
    ArticleSkeleton,
    PageLayoutExtraction,
)
from app.ingestion.unified_extractor import UnifiedExtractor


def test_article_skeleton_schema_validation():
    """Verify ArticleSkeleton Pydantic validation."""
    data = {
        "headline": "RBI Holds Repo Rate Steady at 6.5%",
        "subheadline": "Monetary Policy Committee votes unanimously",
        "byline": "Special Correspondent",
        "article_type": "news",
        "section": "Economy & Policy",
        "prominence": "lead",
        "bbox": [10.0, 10.0, 450.0, 950.0],
        "continues_to_page": 4,
        "continued_from_page": None,
        "has_table": True,
        "has_photo": True,
    }
    skeleton = ArticleSkeleton.model_validate(data)
    assert skeleton.headline == "RBI Holds Repo Rate Steady at 6.5%"
    assert skeleton.article_type == "news"
    assert skeleton.section == "Economy & Policy"
    assert skeleton.prominence == "lead"
    assert skeleton.continues_to_page == 4


def test_page_layout_extraction_schema():
    """Verify PageLayoutExtraction schema and defaults."""
    data = {
        "page_number": 1,
        "newspaper_brand": "Mint",
        "issue_date": "2026-08-25",
        "printed_page_number": "1",
        "is_advertisement_page": False,
        "articles": [
            {
                "headline": "Markets Hit New All-Time High",
                "subheadline": "Sensex crosses 82,000 mark",
                "byline": "Market Bureau",
                "article_type": "news",
                "section": "Markets & Data",
                "prominence": "major",
                "bbox": [50.0, 100.0, 300.0, 800.0],
            }
        ],
    }
    layout = PageLayoutExtraction.model_validate(data)
    assert layout.page_number == 1
    assert layout.newspaper_brand == "Mint"
    assert len(layout.articles) == 1
    assert layout.articles[0].section == "Markets & Data"


def test_article_enrichment_schema():
    """Verify ArticleEnrichment model validation."""
    data = {
        "body_text": "The Reserve Bank of India on Friday kept the repo rate unchanged at 6.5 per cent.\n\nGovernor Shaktikanta Das announced the MPC decision.",
        "summary": "RBI maintained the policy repo rate at 6.5% during its latest MPC meeting.",
        "entities": [
            {"name": "Reserve Bank of India", "type": "org", "mention_count": 2},
            {"name": "Shaktikanta Das", "type": "person", "mention_count": 1},
        ],
        "topics": ["Economy > Monetary Policy", "Economy > Banking & Finance"],
        "tables": [],
    }
    enrichment = ArticleEnrichment.model_validate(data)
    assert len(enrichment.entities) == 2
    assert enrichment.entities[0].name == "Reserve Bank of India"
    assert "Economy > Monetary Policy" in enrichment.topics


def test_json_repair_and_parse():
    """Verify UnifiedExtractor._repair_and_parse_json handles malformed and truncated JSON."""
    # Test 1: JSON inside markdown fences
    fenced = '```json\n{"page_number": 1, "articles": []}\n```'
    parsed = UnifiedExtractor._repair_and_parse_json(fenced)
    assert parsed == {"page_number": 1, "articles": []}

    # Test 2: Unclosed braces
    unclosed = '{"page_number": 1, "newspaper_brand": "Mint", "articles": [{"headline": "Test"'
    parsed2 = UnifiedExtractor._repair_and_parse_json(unclosed)
    assert parsed2 is not None
    assert parsed2.get("newspaper_brand") == "Mint"

    # Test 3: Trailing commas
    trailing = '{"page_number": 1, "articles": [{"headline": "Test",},],}'
    parsed3 = UnifiedExtractor._repair_and_parse_json(trailing)
    assert parsed3 is not None
    assert parsed3.get("page_number") == 1


def test_json_repair_with_thought_tags_and_fences():
    """Verify reasoning/thinking tags (<thought>, <think>) are cleanly stripped."""
    thought_output = """
    <thought>
    Let's analyze the image layout.
    I can see two articles on page 1.
    </thought>
    ```json
    {
        "page_number": 1,
        "newspaper_brand": "Business Standard",
        "articles": [
            {
                "headline": "Markets Surge on GDP Optimism",
                "bbox": ["10.5", "20.5", "300.0", "400.0"],
                "prominence": "lead"
            }
        ]
    }
    ```
    """
    parsed = UnifiedExtractor._repair_and_parse_json(thought_output)
    assert parsed is not None
    assert parsed.get("newspaper_brand") == "Business Standard"

    # Test normalization and validation
    layout = UnifiedExtractor._normalize_and_validate_layout(parsed, page_number=1)
    assert layout.page_number == 1
    assert len(layout.articles) == 1
    assert layout.articles[0].headline == "Markets Surge on GDP Optimism"
    assert layout.articles[0].bbox == [10.5, 20.5, 300.0, 400.0]
def test_repair_truncated_json_mid_sentence():
    """Verify recovery when LLM abruptly stops generating mid-sentence/mid-array."""
    truncated_raw = (
        '{"page_number": 2, "newspaper_brand": "Business Standard", "issue_date": "2024-07-29", '
        '"printed_page_number": "2", "is_advertisement_page": true, '
        '"articles": ['
        '{"headline": "First Valid Article", "bbox": [10.0, 10.0, 200.0, 300.0], "prominence": "lead"}, '
        '{"headline": "[Advertisement] ODISHA FOOD'
    )
    parsed = UnifiedExtractor._repair_and_parse_json(truncated_raw)
    assert parsed is not None
    assert parsed.get("page_number") == 2
    assert parsed.get("newspaper_brand") == "Business Standard"
    assert len(parsed.get("articles", [])) == 1
    assert parsed["articles"][0]["headline"] == "First Valid Article"

    # Test normalization on recovered JSON
    layout = UnifiedExtractor._normalize_and_validate_layout(parsed, page_number=2)
    assert layout.page_number == 2
    assert len(layout.articles) == 1
    assert layout.articles[0].headline == "First Valid Article"


def test_unified_extractor_engine_resolution():
    """Verify UnifiedExtractor resolves google_cloud_vision and gemma engines correctly."""
    # Engine name passed explicitly
    extractor_gcv = UnifiedExtractor(engine_name="google_cloud_vision")
    prov_gcv = extractor_gcv._get_provider()
    assert prov_gcv.provider_name == "google_cloud_vision"

    extractor_gemma = UnifiedExtractor(engine_name="gemma4:26b")
    prov_gemma = extractor_gemma._get_provider()
    assert prov_gemma.provider_name == "ollama"

    extractor_auto = UnifiedExtractor(engine_name="auto")
    prov_auto = extractor_auto._get_provider()
    assert prov_auto.provider_name == "google_cloud_vision"


