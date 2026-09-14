"""Unit tests for ToolCritic evaluation scorecard and closed-loop self-refinement."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from app.agent.sandbox import ASTSafetyScanner, SandboxedExecutor
from app.agent.tool_critic import EvaluationScorecard, ToolCritic
from app.agent.tool_maker import ToolMaker
from app.providers.base import ModelResponse


def test_tool_critic_ast_sql_extraction_handles_variables_and_text():
    """Verify AST-based SQL extraction resolves variable assignments, text() calls, and plain SQL strings."""
    critic = ToolCritic()
    code = """
from sqlalchemy import text

query_var = '''
SELECT a.id, a.headline, a.section
FROM articles a
WHERE a.word_count > 100
'''

async def analyze(db, query, context):
    stmt = text(query_var)
    res = await db.execute(stmt)
    return {"summary": "done", "data": []}
"""
    extracted = critic.extract_sql_queries_ast(code)
    assert len(extracted) >= 1
    assert "SELECT a.id, a.headline, a.section" in extracted[0]


def test_tool_critic_srf_detects_column_hallucinations():
    """Verify Metric 2 (SRF) flags non-existent columns like published_at and suggests issues.issue_date."""
    critic = ToolCritic()
    code = """
from sqlalchemy import text

async def analyze(db, query, context):
    stmt = text("SELECT id, headline, published_at FROM articles WHERE published_at = :dt")
    return {"summary": "done", "data": []}
"""
    sql_queries = critic.extract_sql_queries_ast(code)
    score, issues, fixes = critic.audit_sql_schema(sql_queries)

    assert score < 1.0
    assert any("published_at" in issue for issue in issues)
    assert any("issues.issue_date" in fix for fix in fixes)


def test_tool_critic_srf_detects_cartesian_multiplier_risk():
    """Verify Metric 2 (SRF) flags COUNT(a.id) without DISTINCT when joining articles and pages."""
    critic = ToolCritic()
    code = """
from sqlalchemy import text

async def analyze(db, query, context):
    stmt = text(\"\"\"
        SELECT n.name, COUNT(a.id) as total_articles
        FROM articles a
        JOIN issues i ON a.issue_id = i.id
        JOIN pages p ON p.issue_id = i.id
        GROUP BY n.name
    \"\"\")
    return {"summary": "done", "data": []}
"""
    sql_queries = critic.extract_sql_queries_ast(code)
    score, issues, fixes = critic.audit_sql_schema(sql_queries)

    assert score < 1.0
    assert any("Cartesian" in issue for issue in issues)
    assert any("DISTINCT" in fix for fix in fixes)


def test_tool_critic_dsf_distinguishes_legitimate_absence_from_hallucination():
    """Guardrail 1: When data is [] but summary truthfully acknowledges absence, DSF must be 1.0."""
    critic = ToolCritic()

    # Case A: Legitimate absence truthfully reported
    truthful_summary = "No articles found matching quantum computing in The Goan for the specified date."
    score_truthful, issues_truthful, _ = critic.audit_data_to_summary(
        summary=truthful_summary,
        data=[],
        metadata={},
        query="quantum computing",
        context={"available_newspapers": ["The Goan"], "available_dates": ["2026-08-01"]},
    )
    assert score_truthful == 1.0
    assert len(issues_truthful) == 0

    # Case B: Severe Hallucination - empty data but claiming positive articles
    hallucinated_summary = "The Goan extensively covered the topic with 14 articles across front-page editions."
    score_hallucinated, issues_hallucinated, fixes = critic.audit_data_to_summary(
        summary=hallucinated_summary,
        data=[],
        metadata={},
        query="quantum computing",
        context={"available_newspapers": ["The Goan"], "available_dates": ["2026-08-01"]},
    )
    assert score_hallucinated <= 0.2
    assert any("Hallucination detected" in issue for issue in issues_hallucinated)


def test_tool_critic_dsf_detects_metadata_and_summary_number_mismatch():
    """Verify Metric 4 (DSF) flags discrepancies between computed metadata and numbers cited in summary."""
    critic = ToolCritic()
    mismatch_summary = "The issue of The Goan contains 24 pages."
    score, issues, fixes = critic.audit_data_to_summary(
        summary=mismatch_summary,
        data=[{"newspaper_name": "The Goan", "pages": [1]}],
        metadata={"total_pages": 12},  # actual is 12, summary claimed 24!
        query="pages in the goan",
        context={},
    )
    assert score < 1.0
    assert any("Number mismatch" in issue for issue in issues)
    assert any("12" in fix for fix in fixes)


def test_tool_critic_rps_detects_date_formatting_defect_when_data_empty():
    """Verify Metric 5 (RPS) diagnoses un-normalized date format (2/8/2026) when query returns 0 rows."""
    critic = ToolCritic()
    code_with_defect = """
async def analyze(db, query, context):
    stmt = text("SELECT * FROM issues WHERE issue_date = '2/8/2026'")
    return {"summary": "no records", "data": []}
"""
    score, issues, fixes = critic.audit_filter_plausibility(
        code=code_with_defect,
        query="find issue on 2/8/2026",
        context={"available_dates": ["2026-08-02"]},
        is_empty_result=True,
    )
    assert score < 1.0
    assert any("2026-08-02" in fix for fix in fixes)


@pytest.mark.asyncio
async def test_tool_maker_closed_loop_refinement_fixes_hallucination_and_bounds_trace():
    """Test closed-loop refinement where Attempt 1 has hallucination/defects, and Attempt 2 self-repairs.

    Also tests Guardrail 3: ensures retry messages list is bounded to 4 messages.
    """
    scanner = ASTSafetyScanner()
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=5)
    critic = ToolCritic()

    # Attempt 1: Code executes cleanly, but returns empty data while summary claims 10 articles (hallucination)
    attempt_1_code = """
async def analyze(db, query, context):
    return {
        "summary": "Found 10 articles on the topic.",
        "data": [],
        "metadata": {}
    }
"""
    # Attempt 2: Code truthfully reports legitimate absence
    attempt_2_code = """
async def analyze(db, query, context):
    return {
        "summary": "No articles found matching the requested query.",
        "data": [],
        "metadata": {}
    }
"""
    mock_provider = MagicMock()
    # Track messages passed to provider on complete() calls
    recorded_message_histories = []

    async def mock_complete(messages, **kwargs):
        recorded_message_histories.append(list(messages))
        if len(recorded_message_histories) == 1:
            return ModelResponse(text=f"```python\n{attempt_1_code}\n```")
        return ModelResponse(text=f"```python\n{attempt_2_code}\n```")

    mock_provider.complete = AsyncMock(side_effect=mock_complete)

    tool_maker = ToolMaker(scanner=scanner, sandbox=sandbox, provider=mock_provider, critic=critic)
    res = await tool_maker.generate_and_execute(
        query="Research quantum computing in Goan",
        context={"available_newspapers": ["The Goan"], "available_dates": ["2026-08-01"]},
        max_retries=2,
    )

    # 1. Closed loop succeeded on attempt 2!
    assert res.success is True
    assert res.retries_used == 1
    assert len(res.critique_history) >= 1
    assert "Hallucination detected" in res.critique_history[0]

    # 2. Guardrail 3: Check token-budgeted message history on retry attempt 2
    assert len(recorded_message_histories) == 2
    retry_messages = recorded_message_histories[1]
    assert len(retry_messages) == 4  # System, User initial, Assistant code, User critique
    assert retry_messages[0].role == "system"
    assert retry_messages[1].role == "user"
    assert "Hallucination detected" in retry_messages[3].content


def test_tool_critic_srf_allows_category_alias_and_joins():
    """Verify Metric 2 (SRF) does NOT penalize valid 'AS category' aliases or article_categories joins."""
    critic = ToolCritic()
    valid_sql = [
        """
        SELECT c.name AS category, a.word_count
        FROM articles a
        JOIN article_categories c ON a.category_id = c.id
        WHERE a.word_count > 0
        GROUP BY category
        """
    ]
    score, issues, fixes = critic.audit_sql_schema(valid_sql)
    assert score == 1.0
    assert len(issues) == 0


def test_tool_critic_srf_catches_direct_articles_category():
    """Verify Metric 2 (SRF) correctly flags non-existent column when directly queried on articles."""
    critic = ToolCritic()
    bad_sql = [
        """
        SELECT a.category, a.headline
        FROM articles a
        """
    ]
    score, issues, fixes = critic.audit_sql_schema(bad_sql)
    assert score < 1.0
    assert any("SQL queries non-existent column 'category'" in issue for issue in issues)


def test_extract_sql_queries_ast_deduplicates_whitespace_variations():
    """Verify AST extractor deduplicates identical queries even with whitespace or newline differences."""
    critic = ToolCritic()
    code = """
import pandas as pd
from sqlalchemy import text

async def analyze(db, query, context):
    stmt = text(\"\"\"
        SELECT id, headline
        FROM articles
    \"\"\")
    res = await db.execute(stmt)
    return {"summary": "done", "data": []}
"""
    queries = critic.extract_sql_queries_ast(code)
    # Must only extract 1 query, not 2!
    assert len(queries) == 1


def test_tool_critic_dsf_allows_aggregate_markdown_table_with_empty_article_data():
    """Verify ToolCritic accepts aggregate analytical queries where data is [] but summary has a markdown table."""
    critic = ToolCritic()
    table_summary = (
        "### Photo Count per Section for The Goan (2026-08-05)\n\n"
        "| Section | Photo Count |\n"
        "| :--- | :--- |\n"
        "| Front Page | 11 |\n"
        "| National | 19 |\n"
        "| Health | 12 |\n"
        "| New Delhi | 9 |\n"
        "| Crime & Law | 6 |\n"
        "| Corporate & Industry | 5 |\n"
        "| Entertainment | 5 |\n"
        "| Technology | 2 |\n"
        "| Lifestyle | 1 |\n"
        "| Sports | 1 |\n"
        "| **Total** | **71** |\n\n"
        "Total 71 photos appear across 10 sections."
    )
    raw_output = {
        "summary": table_summary,
        "data": [],
        "metadata": {"total_photos": 71, "section_count": 10},
    }
    scorecard = critic.evaluate(
        code="async def analyze(db, query, context): return {}",
        raw_output=raw_output,
        query="How many photos appear in each section of The Goan on August 5, 2026?",
        context={"available_newspapers": ["The Goan"], "available_dates": ["2026-08-05"]},
    )
    assert scorecard.dsf_score == 1.0
    assert scorecard.rps_score == 1.0
    assert scorecard.is_acceptable is True
    assert len(scorecard.suggested_fixes) == 0
    assert scorecard.critique == ""


def test_tool_critic_rejects_nan_metric_and_demands_repair():
    """Verify ToolCritic detects NaN/nan in summary or metadata, rejects output, and provides actionable repair advice."""
    critic = ToolCritic()
    nan_output = {
        "summary": "The average article length in The Goan on 2026-08-01 is nan words.",
        "data": [],
        "metadata": {"avg_word_count": float("nan")},
    }
    scorecard = critic.evaluate(
        code="async def analyze(db, query, context): return {}",
        raw_output=nan_output,
        query="WHAT IS THE AVG LENGTH OF ARTICLES IN NEWSPAPER THE GOAN DATED 1/8/2026",
        context={"available_newspapers": ["The Goan"], "available_dates": ["2026-08-01"]},
    )
    assert scorecard.is_acceptable is False
    assert scorecard.dsf_score == 0.0
    assert any("NaN" in issue for issue in scorecard.critique.split("\n"))
    assert any("pre-normalized context" in fix for fix in scorecard.suggested_fixes)


