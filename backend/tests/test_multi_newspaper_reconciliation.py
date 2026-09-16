"""Unit tests for multi-newspaper reconciliation, date drift safety, taxonomy filtering, and date anchoring."""

from app.agent.extractor import extract_parameters_from_query
from app.agent.prompt_context import build_synthesizer_user_prompt
from app.agent.tool_factory import reconcile_and_sanitize_arguments
from app.ingestion.classifier import ArticleClassifier, _load_taxonomy_config


def test_reconcile_preserves_multiple_valid_newspapers():
    """Verify that reconcile_and_sanitize_arguments does NOT overwrite secondary newspapers."""
    query = "what are the common topics of health related news in the morning standard and the Goan dated 2/8/2026"
    extracted = extract_parameters_from_query(query)

    assert "The Morning Standard" in extracted.get("target_newspapers", [])
    assert "The Goan" in extracted.get("target_newspapers", [])

    # Morning Standard tool call must be preserved
    args_ms = reconcile_and_sanitize_arguments(
        tool_name="sql_analytics",
        args={"newspaper_name": "The Morning Standard", "issue_date": "2026-08-02"},
        extracted=extracted,
        query=query,
    )
    assert args_ms["newspaper_name"] == "The Morning Standard"
    assert args_ms["issue_date"] == "2026-08-02"

    # The Goan tool call MUST be preserved, NOT overwritten with The Morning Standard!
    args_goan = reconcile_and_sanitize_arguments(
        tool_name="sql_analytics",
        args={"newspaper_name": "The Goan", "issue_date": "2026-08-02"},
        extracted=extracted,
        query=query,
    )
    assert args_goan["newspaper_name"] == "The Goan"
    assert args_goan["issue_date"] == "2026-08-02"

    # Case-insensitive variation of The Goan must normalize to canonical brand
    args_goan_lower = reconcile_and_sanitize_arguments(
        tool_name="hybrid_search",
        args={"newspaper_name": "the goan", "issue_date": "2026-08-02"},
        extracted=extracted,
        query=query,
    )
    assert args_goan_lower["newspaper_name"] == "The Goan"

    # Completely foreign brand should be sanitized to primary extracted brand
    args_foreign = reconcile_and_sanitize_arguments(
        tool_name="sql_analytics",
        args={"newspaper_name": "The Hindu", "issue_date": "2026-08-02"},
        extracted=extracted,
        query=query,
    )
    assert args_foreign["newspaper_name"] == "The Morning Standard"


def test_reconcile_preserves_comparison_newspaper():
    """Verify that comparison_newspaper argument is properly reconciled."""
    query = "compare coverage between The Morning Standard and The Goan dated 2/8/2026"
    extracted = extract_parameters_from_query(query)

    args = reconcile_and_sanitize_arguments(
        tool_name="coverage_analysis",
        args={
            "source_newspaper": "The Morning Standard",
            "comparison_newspaper": "the goan",
            "issue_date": "2026-08-02",
        },
        extracted=extracted,
        query=query,
    )
    assert args["source_newspaper"] == "The Morning Standard"
    assert args["comparison_newspaper"] == "The Goan"


def test_taxonomy_who_pronoun_does_not_trigger_health():
    """Verify that English pronoun 'who' does NOT cause false-positive classification as Health."""
    _load_taxonomy_config()
    clf = ArticleClassifier()

    # 1. Fishermen story with pronoun "who"
    hl1 = "Fishermen await official update"
    body1 = (
        "All we know is that work has not started at the mouth of the River Sal. "
        "The Fisheries Department is in a better position to explain who delayed the project."
    )
    cat1, _, _ = clf.infer_category_from_content(headline=hl1, body=body1)
    assert cat1 != "Health"

    # 2. Constitutional law story with question "Who"
    hl2 = "Who owns the Constitution?"
    body2 = "The Supreme Court delivered a historic verdict on basic structure and democracy."
    cat2, _, _ = clf.infer_category_from_content(headline=hl2, body=body2)
    assert cat2 != "Health"


def test_prompt_context_anchors_verified_dates():
    """Verify that build_synthesizer_user_prompt extracts and strictly anchors verified dates."""
    evidence = [
        {
            "headline": "Diagnostic, lifestyle, and management plans",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-02",
            "snippet": "Health report on lifestyle and gut health.",
        },
        {
            "headline": "Stay one step ahead",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-02",
            "snippet": "Brain fog management techniques.",
        },
    ]

    prompt = build_synthesizer_user_prompt(
        query="health news in the goan dated 2/8/2026",
        archetype="single_issue_deep_dive",
        evidence_items=evidence,
        context="Mock Context",
    )

    assert "Verified Target Issue Date(s): 2026-08-02" in prompt
    assert "strictly reflect the verified date(s): 2026-08-02" in prompt
    assert "The Goan" in prompt


def test_extract_citations_excludes_manifest_and_aggregate_tools():
    """Verify that extract_citations NEVER returns aggregate tool manifests (article_id == 0)."""
    from app.agent.synthesizer import AnswerSynthesizer

    synth = AnswerSynthesizer()
    evidence = [
        {
            "article_id": 0,
            "headline": "Issue Manifest: The Goan (2026-08-02)",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-02",
            "pages": [1],
            "snippet": "Manifest of 20 articles in The Goan.",
            "source_tool": "sql_analytics",
        },
        {
            "article_id": 142,
            "headline": "Understanding your gut health and brain connection",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-02",
            "pages": [5],
            "snippet": "Researchers explain the gut-brain axis and probiotics.",
            "source_tool": "hybrid_search",
        },
    ]

    answer_text = (
        "### ⚡ Executive Summary\n"
        "Recent research highlights gut health.\n\n"
        "### 📌 Key Verified Facts & Highlights\n"
        "- Studies demonstrate the gut-brain axis [The Goan, 2026-08-02, Page 5, \"Understanding your gut health and brain connection\"]."
    )

    citations = synth.extract_citations(answer_text, evidence)
    assert len(citations) == 1
    assert citations[0]["article_id"] == 142
    assert citations[0]["headline"] == "Understanding your gut health and brain connection"
    assert citations[0]["page_number"] == 5
    assert not any(c["article_id"] == 0 for c in citations)


def test_extract_citations_fallback_strictly_picks_real_articles():
    """Verify that when inline citation matching misses, fallback only picks real articles."""
    from app.agent.synthesizer import AnswerSynthesizer

    synth = AnswerSynthesizer()
    evidence = [
        {
            "article_id": 0,
            "headline": "Issue Manifest: The Goan (2026-08-02)",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-02",
            "pages": [1],
            "snippet": "Manifest text.",
            "source_tool": "sql_analytics",
        },
        {
            "article_id": 99,
            "headline": "New Pediatric Ward Launched in South Goa",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-02",
            "pages": [3],
            "snippet": "Pediatric ward opened.",
            "source_tool": "hybrid_search",
        },
    ]

    # Vague text that doesn't mention the headline
    answer_text = "Medical infrastructure saw new developments in healthcare facilities."
    citations = synth.extract_citations(answer_text, evidence)

    assert len(citations) == 1
    assert citations[0]["article_id"] == 99
    assert citations[0]["headline"] == "New Pediatric Ward Launched in South Goa"
    assert not any(c["article_id"] == 0 for c in citations)


def test_folio_elimination_clean_page_format():
    """Verify that chunker and executor formats do NOT produce dual folio notations."""
    from app.agent.executor import format_coverage_difference_snippet
    from app.agent.synthesizer import _DEFAULT_STRUCTURE
    from app.ingestion.chunker import NewspaperChunker

    # 1. Synthesizer prompt structure has no PDF_Page
    assert "{PDF_Page}" not in _DEFAULT_STRUCTURE
    assert "Page {Page_Number}" in _DEFAULT_STRUCTURE
    assert "SINGLE PAGE FORMAT MANDATE" in _DEFAULT_STRUCTURE

    # 2. Coverage difference snippet uses clean Page <N>
    diff_res = {
        "source_newspaper": "The Goan",
        "comparison_newspaper": "The Morning Standard",
        "issue_date": "2026-08-02",
        "total_source_articles": 1,
        "total_comparison_articles": 0,
        "exclusive_count": 1,
        "shared_count": 0,
        "exclusive_articles": [
            {
                "headline": "Exclusive Health Investigation",
                "page_number": 4,
                "section": "Health",
            }
        ],
    }
    snippet = format_coverage_difference_snippet(diff_res)
    assert "[Page 4]" in snippet
    assert "PDF Page" not in snippet

    # 3. Chunker format uses clean Page(s)
    chunker = NewspaperChunker()
    hdr = chunker.create_header_context(
        headline="Hospital Upgrades",
        newspaper_name="The Goan",
        issue_date="2026-08-02",
        pages=[4],
        printed_pages=["4"],
    )
    assert "Page(s): 4" in hdr
    assert "(PDF p." not in hdr

