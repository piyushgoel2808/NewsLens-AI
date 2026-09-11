"""Unit tests for multi-newspaper reconciliation, date drift safety, taxonomy filtering, and date anchoring."""

import pytest
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
