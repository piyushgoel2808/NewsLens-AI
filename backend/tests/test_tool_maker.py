"""Unit tests for ToolMaker code generation, self-correction, and evidence normalization."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.sandbox import ASTSafetyScanner, SandboxedExecutor
from app.agent.tool_maker import ToolMaker, normalize_dynamic_output
from app.providers.base import ModelResponse


def test_normalize_dynamic_output_creates_evidence_items():
    output = {
        "summary": "### Statistical Analysis\nPearson correlation is 0.8542.",
        "data": [
            {
                "article_id": 101,
                "issue_id": 12,
                "headline": "Economic Surge Reported in Goa",
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
                "snippet": "Strong economic growth indicated across sectors.",
                "page_number": 1,
            }
        ],
        "metadata": {"pearson_correlation": 0.8542},
    }

    items = normalize_dynamic_output(output, query="Calculate correlation")
    assert len(items) == 2

    # 1. Primary summary evidence item
    summary_item = items[0]
    assert summary_item["article_id"] == 0
    assert summary_item["source_tool"] == "dynamic_analysis"
    assert "Pearson correlation is 0.8542" in summary_item["snippet"]
    assert summary_item["prominence_score"] == 0.95
    assert summary_item["metadata"]["pearson_correlation"] == 0.8542

    # 2. Supporting article item
    art_item = items[1]
    assert art_item["article_id"] == 101
    assert art_item["newspaper_name"] == "The Goan"
    assert art_item["source_tool"] == "dynamic_analysis"
    assert art_item["prominence_score"] == 0.80


def test_tool_maker_extract_code_handles_fences_and_think_tags():
    scanner = ASTSafetyScanner()
    sandbox = SandboxedExecutor(db_url_readonly="")
    tool_maker = ToolMaker(scanner=scanner, sandbox=sandbox)

    raw = """
<think>
Need to compute word count variance across articles.
</think>
Here is the code to calculate the stats:
```python
import statistics

async def analyze(db, query, context):
    return {"summary": "done", "data": []}
```
Hope this helps!
"""
    extracted = tool_maker._extract_code(raw)
    assert extracted.startswith("import statistics")
    assert "async def analyze" in extracted
    assert "<think>" not in extracted
    assert "```" not in extracted


@pytest.mark.asyncio
async def test_tool_maker_generates_and_executes_successfully():
    scanner = ASTSafetyScanner()
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=5)

    valid_code = """
import statistics

async def analyze(db, query, context):
    vals = [10, 20, 30, 40, 50]
    return {
        "summary": f"Average is {statistics.mean(vals)}",
        "data": [{"id": 1, "val": 10}],
        "metadata": {"mean": 30}
    }
"""
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        return_value=ModelResponse(text=f"```python\n{valid_code}\n```")
    )

    tool_maker = ToolMaker(scanner=scanner, sandbox=sandbox, provider=mock_provider)
    res = await tool_maker.generate_and_execute("Calculate the average of values")

    assert res.success is True
    assert len(res.evidence_items) >= 1
    assert "Average is 30" in res.evidence_items[0]["snippet"]
    assert res.retries_used == 0


@pytest.mark.asyncio
async def test_tool_maker_self_corrects_on_ast_violation():
    scanner = ASTSafetyScanner()
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=5)

    # Attempt 1 has forbidden 'import os'
    bad_code = """
import os

async def analyze(db, query, context):
    return {"summary": "bad", "data": []}
"""
    # Attempt 2 is safe
    good_code = """
import math

async def analyze(db, query, context):
    return {"summary": f"Sqrt 16 is {math.sqrt(16)}", "data": []}
"""
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        side_effect=[
            ModelResponse(text=f"```python\n{bad_code}\n```"),
            ModelResponse(text=f"```python\n{good_code}\n```"),
        ]
    )

    tool_maker = ToolMaker(scanner=scanner, sandbox=sandbox, provider=mock_provider)
    res = await tool_maker.generate_and_execute("Calculate square root", max_retries=2)

    assert res.success is True
    assert res.retries_used == 1  # 1 retry was needed
    assert "Sqrt 16 is 4.0" in res.evidence_items[0]["snippet"]


@pytest.mark.asyncio
async def test_tool_maker_self_corrects_on_runtime_error():
    scanner = ASTSafetyScanner()
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=5)

    # Attempt 1 causes ZeroDivisionError
    bad_runtime_code = """
async def analyze(db, query, context):
    val = 100 / 0
    return {"summary": str(val), "data": []}
"""
    # Attempt 2 fixes it
    fixed_runtime_code = """
async def analyze(db, query, context):
    val = 100 / 2
    return {"summary": f"Result is {val}", "data": []}
"""
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        side_effect=[
            ModelResponse(text=f"```python\n{bad_runtime_code}\n```"),
            ModelResponse(text=f"```python\n{fixed_runtime_code}\n```"),
        ]
    )

    tool_maker = ToolMaker(scanner=scanner, sandbox=sandbox, provider=mock_provider)
    res = await tool_maker.generate_and_execute("Calculate fraction", max_retries=2)

    assert res.success is True
    assert res.retries_used == 1
    assert "Result is 50.0" in res.evidence_items[0]["snippet"]


@pytest.mark.asyncio
async def test_tool_maker_stops_at_max_retries():
    scanner = ASTSafetyScanner()
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=5)

    # Persistent syntax error
    bad_code = """
async def analyze(db, query, context):
    invalid syntax here ???
"""
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        return_value=ModelResponse(text=f"```python\n{bad_code}\n```")
    )

    tool_maker = ToolMaker(scanner=scanner, sandbox=sandbox, provider=mock_provider)
    res = await tool_maker.generate_and_execute("Test failure", max_retries=1)

    assert res.success is False
    assert res.retries_used == 1
    assert "Syntax error" in res.error or "SyntaxError" in res.error


@pytest.mark.asyncio
async def test_unsupported_sql_analytics_analysis_type_delegates_to_dynamic_tool():
    """Verify that when sql_analytics receives an unsupported analysis_type (like count_pages), it delegates to dynamic tool."""
    from app.agent.executor import ToolExecutor
    from app.agent.state import AgentState
    from app.agent.tool_maker import ToolMakerResult

    mock_tool_maker = MagicMock()
    mock_tool_maker.generate_and_execute = AsyncMock(
        return_value=ToolMakerResult(
            success=True,
            evidence_items=[
                {
                    "article_id": 0,
                    "headline": "The Goan Page Count (2026-08-02)",
                    "newspaper_name": "The Goan",
                    "issue_date": "2026-08-02",
                    "snippet": "The issue of The Goan dated 2026-08-02 contains 12 pages.",
                    "prominence_score": 0.95,
                    "source_tool": "dynamic_analysis",
                }
            ],
        )
    )

    executor = ToolExecutor(
        session_factory=MagicMock(),
        hybrid_search=MagicMock(),
        entity_search=MagicMock(),
        timeline_builder=MagicMock(),
        sql_analytics=MagicMock(),
        coverage_analyzer=MagicMock(),
        web_search=MagicMock(),
        tool_maker=mock_tool_maker,
    )

    state: AgentState = {
        "query": "NO OF PAGES IN THE GOAN DATE 2/8/2026",
        "original_query": "NO OF PAGES IN THE GOAN DATE 2/8/2026",
        "chat_history": [],
        "archetype": "factual_lookup",
        "plan": [],
        "tool_executions": [],
        "evidence_items": [],
        "synthesized_answer": "",
        "citations": [],
        "cost_usd": 0.0,
        "latency_ms": 0,
        "user_id": None,
        "model_override": None,
        "enable_web_search": False,
        "web_search_results": [],
        "active_issue_id": None,
        "active_newspaper_name": None,
        "active_issue_date": None,
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
    }

    call = {
        "tool_name": "sql_analytics",
        "arguments": {
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-02",
            "analysis_type": "count_pages",
        },
    }

    evidence_items, record, updates = await executor.execute_single_tool(call, state)

    # Must invoke tool maker and return dynamic evidence items!
    mock_tool_maker.generate_and_execute.assert_called_once()
    assert len(evidence_items) == 1
    assert "contains 12 pages" in evidence_items[0]["snippet"]
    assert record["results_count"] == 1


@pytest.mark.asyncio
async def test_crag_evaluator_triggers_dynamic_tool_fallback_on_zero_evidence():
    """Verify Corrective RAG evaluator invokes ToolMaker when static tools return 0 evidence."""
    from app.agent.evaluator import EvidenceEvaluator
    from app.agent.state import AgentState
    from app.agent.tool_maker import ToolMakerResult

    mock_tool_maker = MagicMock()
    mock_tool_maker.generate_and_execute = AsyncMock(
        return_value=ToolMakerResult(
            success=True,
            evidence_items=[
                {
                    "article_id": 0,
                    "headline": "Dynamic Query Result",
                    "newspaper_name": "The Goan",
                    "issue_date": "2026-08-02",
                    "snippet": "CRAG Dynamic Fallback: Total pages is 12.",
                    "prominence_score": 0.95,
                    "source_tool": "dynamic_analysis",
                }
            ],
        )
    )

    evaluator = EvidenceEvaluator(
        entity_search=MagicMock(),
        web_search=MagicMock(),
        tool_maker=mock_tool_maker,
        sql_analytics=MagicMock(),
    )

    state: AgentState = {
        "query": "NO OF PAGES IN THE GOAN DATE 2/8/2026",
        "original_query": "NO OF PAGES IN THE GOAN DATE 2/8/2026",
        "chat_history": [],
        "archetype": "factual_lookup",
        "plan": [],
        "tool_executions": [],
        "evidence_items": [],
        "synthesized_answer": "",
        "citations": [],
        "cost_usd": 0.0,
        "latency_ms": 0,
        "user_id": None,
        "model_override": None,
        "enable_web_search": False,
        "web_search_results": [],
        "active_issue_id": None,
        "active_newspaper_name": None,
        "active_issue_date": None,
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
    }

    # Initial evidence is empty (static tools failed)
    filtered_evidence, tool_records = await evaluator.evaluate_and_fallback(
        evidence=[],
        state=state,
    )

    assert any(tr["tool_name"] == "crag_dynamic_tool_fallback" for tr in tool_records)


def test_ensure_standard_imports_auto_prepends_missing_modules():
    """Verify ensure_standard_imports injects only missing modules without duplicating existing ones."""
    from app.agent.tool_maker import ensure_standard_imports

    # Code without any imports
    code = """
async def analyze(db, query, context):
    match = re.search(r"\\d+", query)
    val = math.sqrt(25)
    mean_val = statistics.mean([1, 2, 3])
    df = pd.DataFrame([{"a": 1}])
    arr = np.array([1, 2])
    stmt = text("SELECT 1")
    return {"summary": "ok", "data": []}
"""
    result = ensure_standard_imports(code)
    assert "import re" in result
    assert "import math" in result
    assert "import statistics" in result
    assert "import pandas as pd" in result
    assert "import numpy as np" in result
    assert "from sqlalchemy import text" in result

    # Code that already has imports should NOT duplicate them
    already_imported = """
import re
import pandas as pd

async def analyze(db, query, context):
    return {"summary": str(re.search(r"\\d+", "123")), "data": []}
"""
    res2 = ensure_standard_imports(already_imported)
    assert res2.count("import re") == 1
    assert res2.count("import pandas as pd") == 1


@pytest.mark.asyncio
async def test_tool_maker_auto_injects_imports_and_runs_without_name_error():
    """Verify tool maker auto-injects missing 'import re' and executes cleanly without NameError."""
    scanner = ASTSafetyScanner()
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=5)

    # Generated code omits 'import re'
    code_missing_re = """
async def analyze(db, query, context):
    m = re.search(r"(\\d+)", query)
    num = m.group(1) if m else "0"
    return {
        "summary": f"Extracted number {num} from query.",
        "data": [],
        "metadata": {"extracted_num": int(num)}
    }
"""
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        return_value=ModelResponse(text=f"```python\n{code_missing_re}\n```")
    )

    tool_maker = ToolMaker(scanner=scanner, sandbox=sandbox, provider=mock_provider)
    res = await tool_maker.generate_and_execute("Find 42 articles")

    assert res.success is True
    assert len(res.evidence_items) >= 1
    assert "Extracted number 42" in res.evidence_items[0]["snippet"]


@pytest.mark.asyncio
async def test_tool_maker_populates_error_when_retries_exhausted():
    """Verify res.error is properly populated with the critic's critique when all retries fail."""
    scanner = ASTSafetyScanner()
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=5)

    # Always returns code with forbidden 'import os'
    bad_code = """
import os
async def analyze(db, query, context):
    return {"summary": "bad", "data": []}
"""
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        return_value=ModelResponse(text=f"```python\n{bad_code}\n```")
    )

    tool_maker = ToolMaker(scanner=scanner, sandbox=sandbox, provider=mock_provider)
    res = await tool_maker.generate_and_execute("Test failure", max_retries=1)

    assert res.success is False
    assert res.error is not None
    assert len(res.error) > 0
    assert "Security Violation" in res.error or "AST" in res.error


@pytest.mark.asyncio
async def test_executor_native_photo_count_per_section_bypasses_dynamic_synthesis():
    """Verify that photo_count_per_section runs via native SQLAnalyticsEngine without invoking ToolMaker."""
    from app.agent.executor import ToolExecutor
    from app.agent.state import AgentState

    mock_sql_analytics = MagicMock()
    mock_sql_analytics.get_photo_counts_by_section = AsyncMock(
        return_value={
            "total_photos": 71,
            "section_counts": [
                {"section": "National", "count": 19},
                {"section": "Health", "count": 12},
                {"section": "Front Page", "count": 11},
            ],
            "by_section": {"National": 19, "Health": 12, "Front Page": 11},
            "filters": {"newspaper_name": "The Goan", "issue_date": "2026-08-05"},
        }
    )
    mock_tool_maker = MagicMock()

    executor = ToolExecutor(
        session_factory=MagicMock(),
        hybrid_search=MagicMock(),
        entity_search=MagicMock(),
        timeline_builder=MagicMock(),
        sql_analytics=mock_sql_analytics,
        coverage_analyzer=MagicMock(),
        web_search=MagicMock(),
        tool_maker=mock_tool_maker,
    )

    state: AgentState = {
        "query": "How many photos appear in each section of The Goan on August 5, 2026?",
        "original_query": "How many photos appear in each section of The Goan on August 5, 2026?",
        "chat_history": [],
        "archetype": "quantitative_trend",
        "plan": [],
        "tool_executions": [],
        "evidence_items": [],
        "synthesized_answer": "",
        "citations": [],
        "cost_usd": 0.0,
        "latency_ms": 0,
        "user_id": None,
        "model_override": None,
        "enable_web_search": False,
        "web_search_results": [],
        "active_issue_id": None,
        "active_newspaper_name": None,
        "active_issue_date": None,
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
    }

    call = {
        "tool_name": "sql_analytics",
        "arguments": {
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-05",
            "analysis_type": "photo_count_per_section",
        },
    }

    evidence_items, record, updates = await executor.execute_single_tool(call, state)

    mock_tool_maker.generate_and_execute.assert_not_called()
    mock_sql_analytics.get_photo_counts_by_section.assert_called_once()
    assert len(evidence_items) == 1
    assert record["results_count"] == 71
    assert "Total Matching Photos: 71" in evidence_items[0]["snippet"]
    assert "| Front Page | 11 |" in evidence_items[0]["snippet"]


