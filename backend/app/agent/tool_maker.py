"""LLM-as-Tool-Maker dynamic tool generation and execution engine."""

from __future__ import annotations

import contextlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.agent.sandbox import ASTSafetyScanner, SandboxedExecutor, ScanResult
from app.core.logging import get_logger
from app.providers.base import Message
from app.providers.registry import get_registry

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool Maker System Prompt with Archive Schema & Few-Shot Templates
# ---------------------------------------------------------------------------

TOOL_MAKER_SYSTEM_PROMPT = """You are an expert Python data analyst and software engineer for NewsLens-AI, a broadsheet newspaper intelligence platform.
Your task is to write an asynchronous Python function `analyze(db, query, context)` that executes SQL queries against the newspaper archive and computes custom statistical or analytical findings to directly answer the user's question.

### DATABASE SCHEMA (MySQL 8.0)
The database contains the following tables and columns:
1. `newspapers`: id (INT PK), name (VARCHAR, e.g. 'The Goan', 'Hindustan Times', 'The Navhind Times', 'Herald')
2. `issues`: id (INT PK), newspaper_id (INT FK -> newspapers.id), issue_date (DATE, 'YYYY-MM-DD'), total_pages (INT)
3. `articles`:
   - id (INT PK)
   - issue_id (INT FK -> issues.id)
   - headline (TEXT)
   - subheadline (TEXT)
   - byline_author (VARCHAR)
   - section (VARCHAR)
   - article_type (VARCHAR: 'news', 'editorial', 'opinion', 'analysis', 'sports')
   - prominence_score (FLOAT 0.0-1.0)
   - word_count (INT)
   - summary (TEXT)
   - full_text (LONGTEXT)
   - category_id (INT FK -> article_categories.id)
4. `article_categories`: id (INT PK), name (VARCHAR, e.g. 'Politics', 'Business', 'Sports', 'Health', 'Crime')
5. `pages`: id (INT PK), issue_id (INT FK -> issues.id), page_number (INT)
6. `photos`: id (INT PK), article_id (INT FK), page_id (INT FK), caption (TEXT), visual_type (VARCHAR)
7. `entities`: id (INT PK), name (VARCHAR), type (VARCHAR: 'person', 'org', 'location')
8. `article_entities`: article_id (INT FK), entity_id (INT FK), mention_count (INT), salience_score (FLOAT)

### CONTRACT & FUNCTION SIGNATURE
You MUST write a function with this exact signature:

```python
async def analyze(db, query: str, context: dict) -> dict:
    \"\"\"
    Args:
        db: SQLAlchemy AsyncSession connected to the database (or None if mock/test).
        query: Raw question from the user.
        context: Dict with keys 'available_newspapers', 'available_dates', 'categories'.
    Returns:
        dict with keys:
            - 'summary': str (Detailed Markdown summary of findings, mathematical numbers, and insights)
            - 'data': list[dict] (Optional list of article records with keys: 'article_id', 'headline', 'newspaper_name', 'issue_date', 'snippet')
            - 'metadata': dict (Dictionary of computed metrics like correlation, mean, variance, counts)
    \"\"\"
```

### CRITICAL RULES & CONSTRAINTS
1. **Parameterized Queries**: ALWAYS use SQLAlchemy `text(\"SELECT ... WHERE col = :val\")` with dictionary parameters: `await db.execute(text(...), {\"val\": ...})`.
2. **NO String Interpolation in SQL**: NEVER use f-strings or `.format()` inside SQL query strings.
3. **Libraries Allowed**: `sqlalchemy` (`text`), `pandas` (as `pd`), `numpy` (as `np`), `scipy.stats`, `math`, `statistics`, `collections`, `datetime`, `json`, `re`.
4. **Security Restrictions**: NO file I/O (`open`), NO subprocess, NO network calls, NO `os`, NO `sys`, NO `eval`, NO `exec`.
5. **Robustness**: Always handle the case where `db` is None or returns 0 rows. Check for division by zero before calculating ratios or correlations.
6. **Return Format**: Return ONLY the Python code inside ```python ``` code fences. No conversational banter or explanations outside the code block.

### FEW-SHOT EXAMPLES

#### Example 1: Statistical Correlation Between Two Newspapers
```python
import numpy as np
import pandas as pd
from sqlalchemy import text


async def analyze(db, query: str, context: dict) -> dict:
    if db is None:
        return {
            "summary": "Mock database session provided.",
            "data": [],
            "metadata": {},
        }

    stmt = text(\"\"\"
        SELECT i.issue_date, n.name AS newspaper_name, COUNT(a.id) AS article_count
        FROM articles a
        JOIN issues i ON a.issue_id = i.id
        JOIN newspapers n ON i.newspaper_id = n.id
        WHERE n.name IN ('Hindustan Times', 'The Goan')
        GROUP BY i.issue_date, n.name
        ORDER BY i.issue_date ASC
    \"\"\")
    res = await db.execute(stmt)
    rows = res.fetchall()

    if not rows:
        return {
            "summary": "Insufficient data across the requested newspapers to compute correlation.",
            "data": [],
            "metadata": {},
        }

    df = pd.DataFrame(
        rows, columns=["issue_date", "newspaper_name", "article_count"]
    )
    pivot = df.pivot(
        index="issue_date", columns="newspaper_name", values="article_count"
    ).dropna()

    if len(pivot) < 2:
        return {
            "summary": f"Found only {len(pivot)} overlapping dates. Minimum 2 required for correlation.",
            "data": [],
            "metadata": {"sample_size": len(pivot)},
        }

    corr = float(pivot["Hindustan Times"].corr(pivot["The Goan"]))
    summary = f"### Pearson Correlation Analysis\\n\\n- **Correlation Coefficient**: {corr:.4f}\\n- **Overlapping Dates Evaluated**: {len(pivot)}\\n"

    return {
        "summary": summary,
        "data": [],
        "metadata": {"pearson_correlation": round(corr, 4), "dates": len(pivot)},
    }
```

#### Example 2: Category Word Count Variance & Distribution
```python
import pandas as pd
from sqlalchemy import text


async def analyze(db, query: str, context: dict) -> dict:
    if db is None:
        return {
            "summary": "Mock database session provided.",
            "data": [],
            "metadata": {},
        }

    stmt = text(\"\"\"
        SELECT c.name AS category, a.word_count, a.id AS article_id, a.headline, i.issue_date, n.name AS newspaper_name
        FROM articles a
        JOIN article_categories c ON a.category_id = c.id
        JOIN issues i ON a.issue_id = i.id
        JOIN newspapers n ON i.newspaper_id = n.id
        WHERE a.word_count > 0
    \"\"\")
    res = await db.execute(stmt)
    rows = res.fetchall()

    if not rows:
        return {
            "summary": "No articles with valid word counts found.",
            "data": [],
            "metadata": {},
        }

    df = pd.DataFrame(
        rows,
        columns=[
            "category",
            "word_count",
            "article_id",
            "headline",
            "issue_date",
            "newspaper_name",
        ],
    )
    grouped = (
        df.groupby("category")["word_count"]
        .agg(["count", "mean", "var", "std", "median"])
        .reset_index()
    )
    grouped = grouped.sort_values(by="var", ascending=False)

    lines = [
        "### Article Word Count Distribution & Variance by Category\\n",
        "| Category | Total Articles | Mean Words | Median | Variance | Std Dev |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for _, r in grouped.iterrows():
        var_str = f"{r['var']:.1f}" if pd.notnull(r["var"]) else "N/A"
        std_str = f"{r['std']:.1f}" if pd.notnull(r["std"]) else "N/A"
        lines.append(
            f"| {r['category']} | {int(r['count'])} | {r['mean']:.1f} | {r['median']:.1f} | {var_str} | {std_str} |"
        )

    return {
        "summary": "\\n".join(lines),
        "data": df.head(10).to_dict(orient="records"),
        "metadata": {
            "top_variance_category": (
                grouped.iloc[0]["category"] if len(grouped) > 0 else None
            )
        },
    }
```

#### Example 3: Issue Page Count or Newspaper Edition Metadata
```python
import re
from sqlalchemy import text


async def analyze(db, query: str, context: dict) -> dict:
    if db is None:
        return {
            "summary": "Mock database session provided.",
            "data": [],
            "metadata": {},
        }

    # Query issues and pages for matching newspaper and date
    stmt = text(\"\"\"
        SELECT n.name AS newspaper_name, i.issue_date, i.total_pages, COUNT(p.id) AS page_count
        FROM issues i
        JOIN newspapers n ON i.newspaper_id = n.id
        LEFT JOIN pages p ON p.issue_id = i.id
        WHERE (
            :np_name IS NULL 
            OR LOWER(n.name) LIKE LOWER(:np_pattern)
        )
        AND (
            :target_date IS NULL 
            OR i.issue_date = :target_date
        )
        GROUP BY n.name, i.issue_date, i.total_pages
        LIMIT 5
    \"\"\")

    # Extract date if present (e.g. 2/8/2026 -> 2026-08-02 or YYYY-MM-DD)
    date_match = re.search(r"(\\b\\d{4}-\\d{1,2}-\\d{1,2}\\b)", query)
    target_dt = None
    if date_match:
        target_dt = date_match.group(1)
    else:
        dmy_match = re.search(r"(\\b\\d{1,2})[/.-](\\d{1,2})[/.-](\\d{4})\\b", query)
        if dmy_match:
            d, m, y = dmy_match.groups()
            target_dt = f"{y}-{int(m):02d}-{int(d):02d}"

    # Extract newspaper pattern
    np_pattern = "%Goan%" if "goan" in query.lower() else (
        "%Hindustan Times%" if "hindustan" in query.lower() or "ht" in query.lower() else (
            "%Navhind%" if "navhind" in query.lower() else (
                "%Herald%" if "herald" in query.lower() else "%"
            )
        )
    )

    res = await db.execute(stmt, {
        "np_name": np_pattern if np_pattern != "%" else None,
        "np_pattern": np_pattern,
        "target_date": target_dt,
    })
    rows = res.fetchall()

    if not rows:
        return {
            "summary": f"No newspaper issue found matching query criteria (date: {target_dt}, newspaper: {np_pattern}).",
            "data": [],
            "metadata": {},
        }

    row = rows[0]
    pages_reported = row[2] or row[3] or 0
    summary = f"The issue of **{row[0]}** dated **{row[1]}** contains **{pages_reported} pages**."

    return {
        "summary": summary,
        "data": [
            {
                "newspaper_name": row[0],
                "issue_date": str(row[1]),
                "headline": f"{row[0]} Page Count ({row[1]})",
                "snippet": summary,
                "pages": list(range(1, pages_reported + 1)) if pages_reported else [1],
            }
        ],
        "metadata": {
            "newspaper_name": row[0],
            "issue_date": str(row[1]),
            "total_pages": pages_reported,
        },
    }
```
"""


# ---------------------------------------------------------------------------
# Tool Maker Result Container
# ---------------------------------------------------------------------------


@dataclass
class ToolMakerResult:
    """Result of LLM tool generation and sandboxed execution."""

    success: bool
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    code: str = ""
    raw_output: dict[str, Any] | None = None
    error: str | None = None
    retries_used: int = 0
    execution_time_ms: int = 0


# ---------------------------------------------------------------------------
# Evidence Normalization Helper
# ---------------------------------------------------------------------------


def normalize_dynamic_output(
    output: dict[str, Any],
    query: str,
) -> list[dict[str, Any]]:
    """Convert raw dynamic tool output dictionary into standard NewsLens evidence items."""
    evidence_items: list[dict[str, Any]] = []
    summary_text = output.get("summary") or ""
    metadata = output.get("metadata") or {}

    # 1. Primary Structural Evidence Item (Summary)
    if summary_text:
        evidence_items.append({
            "article_id": 0,
            "issue_id": 0,
            "headline": f"Dynamic Analytical Computation: {query[:60]}",
            "newspaper_name": "Statistical Engine",
            "issue_date": "",
            "page_number": 1,
            "pages": [1],
            "snippet": summary_text,
            "prominence_score": 0.95,
            "source_tool": "dynamic_analysis",
            "metadata": metadata,
        })

    # 2. Supporting Individual Article Records
    raw_data = output.get("data")
    if isinstance(raw_data, list):
        for idx, row in enumerate(raw_data[:20]):  # Cap at 20 supporting items
            if isinstance(row, dict):
                art_id = row.get("article_id") or row.get("id") or (idx + 1)
                with contextlib.suppress(ValueError, TypeError):
                    art_id = int(art_id)

                headline = str(row.get("headline") or row.get("title") or f"Record #{idx + 1}")
                np_name = str(row.get("newspaper_name") or row.get("newspaper") or "")
                dt_str = str(row.get("issue_date") or row.get("date") or "")
                snip = str(row.get("snippet") or row.get("summary") or row.get("full_text") or json.dumps(row))

                evidence_items.append({
                    "article_id": art_id,
                    "issue_id": row.get("issue_id", 0),
                    "headline": headline,
                    "newspaper_name": np_name,
                    "issue_date": dt_str,
                    "page_number": row.get("page_number", 1),
                    "pages": [row.get("page_number", 1)],
                    "snippet": snip[:500],
                    "prominence_score": 0.80,
                    "source_tool": "dynamic_analysis",
                })

    return evidence_items


# ---------------------------------------------------------------------------
# ToolMaker Engine with Self-Correction Loop
# ---------------------------------------------------------------------------


class ToolMaker:
    """Orchestrates dynamic Python code generation, AST safety scanning, and sandboxed execution."""

    def __init__(
        self,
        scanner: ASTSafetyScanner,
        sandbox: SandboxedExecutor,
        provider: Any = None,
    ) -> None:
        self._scanner = scanner
        self._sandbox = sandbox
        self._provider = provider

    def _extract_code(self, text: str) -> str:
        """Extract clean Python code from LLM response text."""
        # Strip <think> tags if present
        cleaned = re.sub(r"(?s)<think>.*?</think>", "", text).strip()

        # Match ```python ... ```
        m = re.search(r"```(?:python)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Fallback: scan for async def analyze or import
        if "async def analyze" in cleaned:
            idx = cleaned.find("async def analyze")
            # See if there are imports before async def analyze
            import_idx = cleaned.find("import ")
            start = import_idx if 0 <= import_idx < idx else idx
            return cleaned[start:].strip()

        return cleaned

    def _resolve_provider(self, model_override: str | None = None) -> Any:
        """Resolve LLM provider for tool code generation."""
        if self._provider is not None:
            return self._provider
        reg = get_registry()
        try:
            return reg.get_chat_provider(model_override or "query_planner")
        except Exception:
            return reg.get_chat_provider("query_condenser")

    async def generate_and_execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        model_override: str | None = None,
        max_retries: int = 2,
    ) -> ToolMakerResult:
        """Synthesize Python tool code, verify safety, execute in sandbox, with self-correction."""
        t_start = time.monotonic()
        provider = self._resolve_provider(model_override)
        ctx = context or {}

        # 1. Initial Prompt Construction
        user_prompt = (
            f"Write an asynchronous Python tool to answer the following broadsheet research query:\n"
            f"Query: \"{query}\"\n\n"
            f"ARCHIVE CONTEXT:\n"
            f"Available Newspapers: {ctx.get('available_newspapers', [])}\n"
            f"Available Dates: {ctx.get('available_dates', [])[:10]}\n"
            f"Available Categories: {ctx.get('categories', [])}\n\n"
            f"Write ONLY the `analyze(db, query, context)` function in valid Python."
        )

        messages = [
            Message(role="system", content=TOOL_MAKER_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        last_code = ""
        last_error = ""

        for attempt in range(max_retries + 1):
            try:
                # LLM Code Generation Step
                resp = await provider.complete(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2048,
                )
                raw_code = self._extract_code(resp.text)
                last_code = raw_code

                # 2. AST Static Security Verification
                scan_res: ScanResult = self._scanner.scan(raw_code)
                if not scan_res.is_safe:
                    err_details = "; ".join(scan_res.violations)
                    logger.warning(
                        f"Dynamic tool failed AST security scan on attempt {attempt}: {err_details}"
                    )
                    last_error = f"Security Violation: {err_details}"

                    # Add self-correction message
                    messages.append(Message(role="assistant", content=raw_code))
                    messages.append(Message(
                        role="user",
                        content=(
                            f"Your code was rejected by the AST Safety Scanner with the following violations:\n"
                            f"{err_details}\n\n"
                            f"Strictly avoid forbidden imports (os, sys, subprocess, open, requests) and destructive SQL. "
                            f"Fix the code and rewrite the complete `analyze` function."
                        ),
                    ))
                    continue

                # 3. Sandboxed Execution Step
                sandbox_res = await self._sandbox.execute(
                    code=raw_code,
                    query=query,
                    context=ctx,
                )

                if sandbox_res.success and sandbox_res.output:
                    dur_ms = round((time.monotonic() - t_start) * 1000)
                    evidence_items = normalize_dynamic_output(sandbox_res.output, query)
                    return ToolMakerResult(
                        success=True,
                        evidence_items=evidence_items,
                        code=raw_code,
                        raw_output=sandbox_res.output,
                        retries_used=attempt,
                        execution_time_ms=dur_ms,
                    )

                # Execution failed; formulate self-correction feedback
                last_error = sandbox_res.error or "Sandbox execution failed with unknown error."
                logger.warning(
                    f"Dynamic tool execution failed on attempt {attempt}: {last_error}"
                )

                messages.append(Message(role="assistant", content=raw_code))
                messages.append(Message(
                    role="user",
                    content=(
                        f"Executing your code in the sandbox raised the following runtime error:\n"
                        f"```\n{last_error}\n```\n\n"
                        f"Common issues: column name typos, wrong JOIN conditions, missing await, or divide by zero. "
                        f"Fix the bug and provide the corrected `analyze` function."
                    ),
                ))

            except Exception as e:
                last_error = str(e)
                logger.error(f"Error during ToolMaker iteration {attempt}: {e}")

        dur_ms = round((time.monotonic() - t_start) * 1000)
        return ToolMakerResult(
            success=False,
            code=last_code,
            error=last_error or "Failed to synthesize a working tool after maximum retries.",
            retries_used=max_retries,
            execution_time_ms=dur_ms,
        )


__all__ = [
    "TOOL_MAKER_SYSTEM_PROMPT",
    "ToolMaker",
    "ToolMakerResult",
    "normalize_dynamic_output",
]
