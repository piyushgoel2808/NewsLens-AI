"""LLM-as-Tool-Maker dynamic tool generation and execution engine."""

from __future__ import annotations

import contextlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.agent.archive_context import STATIC_BROADSHEET_SCHEMA
from app.agent.sandbox import ASTSafetyScanner, SandboxedExecutor, ScanResult
from app.agent.tool_critic import EvaluationScorecard, ToolCritic
from app.core.logging import get_logger
from app.providers.base import Message
from app.providers.registry import get_registry

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool Maker System Prompt with Archive Schema & Few-Shot Templates
# ---------------------------------------------------------------------------

_TOOL_MAKER_HEADER = """You are an expert Python data analyst and software engineer for NewsLens-AI, a broadsheet newspaper intelligence platform.
Your task is to write an asynchronous Python function `analyze(db, query, context)` that executes SQL queries against the newspaper archive and computes custom statistical or analytical findings to directly answer the user's question.
"""

_TOOL_MAKER_BODY = r"""### CONTRACT & FUNCTION SIGNATURE
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
7. **Date Normalization & Safety**: The archive stores dates in ISO `'YYYY-MM-DD'` format. Pre-normalized dates in `context` (e.g. `context.get('target_date')` or `context.get('issue_date')`) are ALREADY valid `'YYYY-MM-DD'` strings. Pass them directly as SQL bind parameters: `await db.execute(text("... WHERE i.issue_date = :dt"), {"dt": target_date})`. NEVER call `datetime.strptime()` without checking `if date_str:` first, because `datetime.strptime(None, ...)` raises `TypeError: strptime() argument 1 must be str, not None`.
8. **Newspaper Matching**: Use case-insensitive matching or `LIKE '%Name%'` (e.g. `LOWER(n.name) LIKE '%goan%'` or `n.name IN ('The Goan', 'Hindustan Times')`) to prevent prefix mismatches.
9. **Cartesian Product Prevention**: When joining `articles` with `pages` or `photos`, ALWAYS use `COUNT(DISTINCT a.id)` to avoid inflated duplicate counts.
10. **Strict Grounding & Truthful Absence**: Your `summary` MUST strictly report the actual counts and data found. If the query yields 0 rows, clearly state that no matching records were found in the archive—NEVER invent positive numbers or narrative facts.
11. **Category Relational Schema**: The `articles` table has NO `category` column; it only contains `category_id (INT FK -> article_categories.id)`. The category string name (e.g. 'Sports', 'Politics') is stored in `article_categories.name`. To filter or group by category, ALWAYS join: `JOIN article_categories c ON a.category_id = c.id` and filter on `c.name = :category` (or `LOWER(c.name) = LOWER(:category)`). NEVER write `articles.category`, `a.category`, or compare integer `a.category_id` directly to a category name string. Match SQL bind parameter names (`:category`) exactly to the keys in your parameters dict passed to `db.execute()`.
12. **Advertisements Schema**: Advertisements and commercial notices are stored as records in the `articles` table with `article_type = 'advertisement'` or in section `'Advertisements & Notices'`. Do NOT query `photos` for advertisements (`photos.visual_type` is `'photo'`, not `'ad'`). Full-page advertisement wraps are also flagged on `pages.is_advertisement_page = 1`.
13. **Variable Naming & SQLAlchemy text() Safety**: NEVER name a variable `text = ...`! Assigning to a variable named `text` shadows `from sqlalchemy import text` and causes a fatal Python `UnboundLocalError`. Always use `snippet`, `content`, `article_text`, or `row_text` for string content.
14. **Pre-Normalized Context Parameters**: The `context` dictionary provides pre-normalized keys: `context.get('target_date')` (or `context.get('issue_date')` / `context.get('date')`), `context.get('newspaper_name')`, `context.get('date_from')`, `context.get('date_to')`, and `context.get('category')`.
    - ALWAYS prioritize these values over manual regex parsing.
    - These values are ALREADY valid strings for SQL queries (e.g. `'2026-09-10'`). Do NOT parse them with `datetime.strptime()`.
    - If a date parameter is not provided in context, it will be `None`. Always check `if target_date:` before filtering on it in SQL.
15. **Zero-Division & Strict No-NaN Safety**: NEVER return `NaN`, `nan`, or `null` in your summary or metadata! Before calculating `.mean()`, `.std()`, correlations, or averages, verify that rows exist (`if not rows:` or `if df.empty:`). If 0 rows match, explicitly return `{'summary': 'No matching records found in the archive...', 'data': [], 'metadata': {'count': 0}}`. Never output `nan` or `NaN` for any metric.
16. **Pandas Filtering & Type Safety**: SQL `WHERE` clauses already filter by date and newspaper. DO NOT re-filter `df['issue_date'] == target_dt` in pandas if SQL already filtered on date, because MySQL dates can cause type mismatch with strings. If you must compare dates in pandas, convert first with `df['issue_date'].astype(str) == str(target_dt)`. NEVER do `df['newspaper_name'] == np_pattern` in pandas if `np_pattern` contains SQL wildcards (`%`) like `'%Goan%'`; use `.str.contains(clean_np, case=False, na=False)` without wildcards.
17. **DataFrame & Dict Key Consistency**: Always match column names in pandas EXACTLY to your SQL `SELECT` aliases. If you need to access `df['category']` or `row['category']`, ensure you wrote `SELECT c.name AS category` in your SQL! If you wrote `SELECT c.name AS category_name`, you must access `df['category_name']`. Never access dictionary or DataFrame keys that were not selected in your SQL query.
18. **Page Number Filtering & Junction Schema**: The `articles` table has NO `page_number` column! Articles link to pages via the `article_pages` junction table. To filter or calculate metrics (like average word count, article counts, or page manifests) for a specific page, ALWAYS join `article_pages`:
    ```sql
    SELECT a.id, a.headline, a.word_count, ap.page_number
    FROM articles a
    JOIN article_pages ap ON a.id = ap.article_id
    JOIN issues i ON a.issue_id = i.id
    JOIN newspapers n ON i.newspaper_id = n.id
    WHERE ap.page_number = :page_num AND i.issue_date = :issue_date
    ```
    NEVER write `articles.page_number`, `a.page_number`, `p.article_id`, or `p.primary_page_id`. Standardize 100% on sequential integer page numbers (`page_number` in `article_pages`), NEVER printed page strings or folios.

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

    # Prioritize pre-normalized context parameters over query regex parsing
    target_dt = context.get("target_date")
    if not target_dt:
        date_match = re.search(r"(\b\d{4}-\d{1,2}-\d{1,2}\b)", query)
        if date_match:
            target_dt = date_match.group(1)
        else:
            dmy_match = re.search(r"(\b\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", query)
            if dmy_match:
                d, m, y = dmy_match.groups()
                target_dt = f"{y}-{int(m):02d}-{int(d):02d}"

    newspaper_name = context.get("newspaper_name")
    if newspaper_name:
        clean_np = newspaper_name.replace("The ", "").strip()
        np_pattern = f"%{clean_np}%"
    else:
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

#### Example 4: Average Word Count and Metrics on a Specific Page
```python
import re
import pandas as pd
from sqlalchemy import text


async def analyze(db, query: str, context: dict) -> dict:
    if db is None:
        return {
            "summary": "Mock database session provided.",
            "data": [],
            "metadata": {},
        }

    # Extract target page number (e.g. page 6)
    page_match = re.search(r"page\s*(\d+)", query, re.IGNORECASE)
    target_page = int(page_match.group(1)) if page_match else 6

    # Target date and newspaper from pre-normalized context
    target_date = context.get("target_date") or context.get("issue_date")
    newspaper_name = context.get("newspaper_name") or "Hindustan Times"

    stmt = text(\"\"\"
        SELECT a.id AS article_id, a.headline, a.word_count, ap.page_number, n.name AS newspaper_name, i.issue_date
        FROM articles a
        JOIN article_pages ap ON a.id = ap.article_id
        JOIN issues i ON a.issue_id = i.id
        JOIN newspapers n ON i.newspaper_id = n.id
        WHERE ap.page_number = :page_number
          AND (:target_date IS NULL OR i.issue_date = :target_date)
          AND (:np_name IS NULL OR LOWER(n.name) LIKE LOWER(:np_name))
        ORDER BY a.prominence_score DESC
    \"\"\")

    res = await db.execute(stmt, {
        "page_number": target_page,
        "target_date": target_date,
        "np_name": f"%{newspaper_name}%" if newspaper_name else None,
    })
    rows = res.fetchall()

    if not rows:
        return {
            "summary": f"No articles found on Page {target_page} for {newspaper_name} on {target_date}.",
            "data": [],
            "metadata": {"count": 0, "page_number": target_page},
        }

    df = pd.DataFrame(rows, columns=["article_id", "headline", "word_count", "page_number", "newspaper_name", "issue_date"])
    valid_words = df[df["word_count"] > 0]["word_count"]
    avg_words = float(valid_words.mean()) if not valid_words.empty else 0.0
    total_words = int(valid_words.sum()) if not valid_words.empty else 0

    summary = (
        f"### Page {target_page} Article Word Count Analytics\\n\\n"
        f"- **Newspaper**: {df.iloc[0]['newspaper_name']}\\n"
        f"- **Date**: {df.iloc[0]['issue_date']}\\n"
        f"- **Page Number**: {target_page}\\n"
        f"- **Total Articles Found**: {len(df)}\\n"
        f"- **Average Word Count**: {avg_words:.1f} words\\n"
        f"- **Total Words on Page**: {total_words:,} words\\n"
    )

    return {
        "summary": summary,
        "data": df.to_dict(orient="records"),
        "metadata": {
            "page_number": target_page,
            "article_count": len(df),
            "average_word_count": round(avg_words, 2),
            "total_words": total_words,
        },
    }
```
"""

TOOL_MAKER_SYSTEM_PROMPT = f"{_TOOL_MAKER_HEADER}\n{STATIC_BROADSHEET_SCHEMA}\n\n{_TOOL_MAKER_BODY}"


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
    scorecard: EvaluationScorecard | None = None
    critique_history: list[str] = field(default_factory=list)
    error: str | None = None
    retries_used: int = 0
    execution_time_ms: int = 0


# ---------------------------------------------------------------------------
# Evidence Normalization Helper
# ---------------------------------------------------------------------------


def normalize_dynamic_output(
    output: dict[str, Any],
    query: str,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert raw dynamic tool output dictionary into standard NewsLens evidence items."""
    evidence_items: list[dict[str, Any]] = []
    summary_text = output.get("summary") or ""
    metadata = output.get("metadata") or {}

    # Sanitize NaN values from metadata
    clean_metadata: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, float) and (v != v or str(v).lower() == "nan"):
            clean_metadata[k] = 0.0
        elif isinstance(v, str) and v.lower() == "nan":
            clean_metadata[k] = "0"
        else:
            clean_metadata[k] = v

    # 1. Primary Structural Evidence Item (Summary)
    if summary_text:
        target_np = str(
            clean_metadata.get("newspaper_name")
            or (context.get("newspaper_name") if context else "")
            or "Archive Analytics"
        )
        target_dt = str(
            clean_metadata.get("issue_date")
            or (context.get("target_date") if context else "")
            or ""
        )
        evidence_items.append({
            "article_id": 0,
            "issue_id": 0,
            "headline": f"Analytical Computation: {query[:60]}",
            "newspaper_name": target_np,
            "issue_date": target_dt,
            "page_number": 0,
            "pages": [],
            "snippet": summary_text,
            "prominence_score": 0.95,
            "source_tool": "dynamic_analysis",
            "is_statistical_metric": True,
            "is_article": False,
            "metadata": clean_metadata,
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


def ensure_standard_imports(code: str) -> str:
    """Prepend essential standard library and data imports if referenced in code but omitted."""
    if not code:
        return code

    needed: list[str] = []
    # Regular expressions
    if re.search(r"\bre\.", code) and not re.search(r"^\s*(?:import\s+re\b|from\s+re\b)", code, re.MULTILINE):
        needed.append("import re")
    # Math
    if (re.search(r"\bmath\.", code) or "sqrt(" in code) and not re.search(r"^\s*(?:import\s+math\b|from\s+math\b)", code, re.MULTILINE):
        needed.append("import math")
    # Statistics
    if re.search(r"\bstatistics\.", code) and not re.search(r"^\s*(?:import\s+statistics\b|from\s+statistics\b)", code, re.MULTILINE):
        needed.append("import statistics")
    # JSON
    if re.search(r"\bjson\.", code) and not re.search(r"^\s*(?:import\s+json\b|from\s+json\b)", code, re.MULTILINE):
        needed.append("import json")
    # Datetime
    if (re.search(r"\bdatetime\.", code) or re.search(r"\bdate\.", code)) and not re.search(r"^\s*(?:import\s+datetime\b|from\s+datetime\b)", code, re.MULTILINE):
        needed.append("from datetime import date, datetime")
    # Pandas
    if (re.search(r"\bpd\.", code) or "DataFrame(" in code or "Series(" in code) and not re.search(r"^\s*(?:import\s+pandas\b|from\s+pandas\b)", code, re.MULTILINE):
        needed.append("import pandas as pd")
    # Numpy
    if re.search(r"\bnp\.", code) and not re.search(r"^\s*(?:import\s+numpy\b|from\s+numpy\b)", code, re.MULTILINE):
        needed.append("import numpy as np")
    # SQLAlchemy text
    if re.search(r"\btext\s*\(", code) and not re.search(r"^\s*(?:from\s+sqlalchemy\s+import\s+.*?\btext\b|import\s+sqlalchemy\b)", code, re.MULTILINE):
        needed.append("from sqlalchemy import text")

    # Guard against UnboundLocalError when local variable 'text' shadows sqlalchemy.text
    if re.search(r"\btext\s*\(", code) and (re.search(r"\btext\s*=\s*", code) or re.search(r"\bfor\s+text\s+in\b", code)):
        code = re.sub(r"\btext\s*=\s*", "article_content = ", code)
        code = re.sub(r"\bfor\s+text\s+in\b", "for article_content in", code)
        code = re.sub(r"\btext\.strip\(", "article_content.strip(", code)
        code = re.sub(r"\btext\.lower\(", "article_content.lower(", code)
        code = re.sub(r"\blen\(text\)", "len(article_content)", code)

    if needed:
        return "\n".join(needed) + "\n\n" + code
    return code


# ---------------------------------------------------------------------------
# ToolMaker Engine with Self-Correction Loop
# ---------------------------------------------------------------------------


class ToolMaker:
    """Orchestrates dynamic Python code generation, AST safety scanning, and sandboxed execution with closed-loop refinement."""

    def __init__(
        self,
        scanner: ASTSafetyScanner,
        sandbox: SandboxedExecutor,
        provider: Any = None,
        critic: ToolCritic | None = None,
    ) -> None:
        self._scanner = scanner
        self._sandbox = sandbox
        self._provider = provider
        self._critic = critic or ToolCritic()

    def _extract_code(self, text: str) -> str:
        """Extract clean Python code from LLM response text."""
        # Strip <think> tags if present
        cleaned = re.sub(r"(?s)<think>.*?</think>", "", text).strip()

        # Match ```python ... ``` (or unclosed trailing fence)
        m = re.search(r"```(?:python)?\s*\n(.*?)(?:\n```|$)", cleaned, re.DOTALL)
        if m:
            code = m.group(1)
        elif "async def analyze" in cleaned:
            idx = cleaned.find("async def analyze")
            # See if there are imports before async def analyze
            import_idx = cleaned.find("import ")
            start = import_idx if 0 <= import_idx < idx else idx
            code = cleaned[start:]
        else:
            code = cleaned

        # Clean trailing whitespace and line continuation characters (e.g. "\\   \\n" -> "\\\\n")
        cleaned_lines = []
        for line in code.splitlines():
            cl = re.sub(r"\\\s+$", "\\\\", line)
            if not cl.endswith("\\"):
                cl = cl.rstrip()
            cleaned_lines.append(cl)
        return "\n".join(cleaned_lines).strip()

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
        attempted_tools: list[dict[str, Any]] | None = None,
        gap_diagnosis: str | None = None,
        max_retries: int = 2,
    ) -> ToolMakerResult:
        """Synthesize Python tool code, verify safety, execute in sandbox, with closed-loop self-refinement."""
        t_start = time.monotonic()
        provider = self._resolve_provider(model_override)
        ctx = context or {}

        target_date = ctx.get("target_date") or ctx.get("issue_date") or ctx.get("date")
        if target_date:
            ctx["target_date"] = str(target_date)
            ctx["issue_date"] = str(target_date)

        # 1. Initial Prompt Construction
        prompt_parts = [
            "Write an asynchronous Python tool to answer the following broadsheet research query:\n",
            f"Query: \"{query}\"\n\n",
            "ARCHIVE CONTEXT:\n",
            f"Available Newspapers: {ctx.get('available_newspapers', [])}\n",
            f"Available Dates: {ctx.get('available_dates', [])[:10]}\n",
            f"Available Categories: {ctx.get('categories', [])}\n",
            "\nPRE-EXTRACTED PARAMETERS (Already validated in `context` dictionary):\n",
            f"- `context.get('target_date')`: {repr(ctx.get('target_date'))}\n",
            f"- `context.get('newspaper_name')`: {repr(ctx.get('newspaper_name'))}\n",
            f"- `context.get('category')`: {repr(ctx.get('category'))}\n",
            "\nCRITICAL CODE PATTERN: Use `target_date = context.get('target_date') or context.get('issue_date')` and `newspaper_name = context.get('newspaper_name')`. Do NOT write custom word.split() loops to parse dates.\n",
        ]

        if attempted_tools or gap_diagnosis:
            prompt_parts.append("\nPRIOR RETRIEVAL AUDIT & REASON FOR FALLBACK:")
            if attempted_tools:
                tool_summaries = []
                for t in attempted_tools[:4]:
                    t_name = t.get("tool_name") or "tool"
                    t_input = t.get("tool_input") or {}
                    t_count = t.get("results_count", 0)
                    tool_summaries.append(f"- Attempted `{t_name}` with arguments {t_input} (yielded {t_count} raw items)")
                prompt_parts.append("Prior tools executed:\n" + "\n".join(tool_summaries))
            if gap_diagnosis:
                prompt_parts.append(f"Evaluator Audit Diagnosis:\n{gap_diagnosis.strip()}\n")
            prompt_parts.append(
                "YOUR TASK: Static retrieval failed to adequately answer the query. "
                "Write custom Python and SQL queries to specifically overcome this shortfall and directly answer the query."
            )

        prompt_parts.append("\nWrite ONLY the `analyze(db, query, context)` function in valid Python.")
        user_prompt = "\n".join(prompt_parts)

        system_msg = Message(role="system", content=TOOL_MAKER_SYSTEM_PROMPT)
        initial_user_msg = Message(role="user", content=user_prompt)

        last_code = ""
        last_error = ""
        last_scorecard: EvaluationScorecard | None = None
        critique_history: list[str] = []

        messages = [system_msg, initial_user_msg]

        for attempt in range(max_retries + 1):
            try:
                # LLM Code Generation Step
                resp = await provider.complete(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    thinking_budget=0,
                )
                raw_code = self._extract_code(resp.text)
                raw_code = ensure_standard_imports(raw_code)
                last_code = raw_code

                # 2. AST Static Security Verification
                scan_res: ScanResult = self._scanner.scan(raw_code)
                if not scan_res.is_safe:
                    err_details = "; ".join(scan_res.violations)
                    logger.warning(
                        f"Dynamic tool failed AST security scan on attempt {attempt}: {err_details}"
                    )
                    last_error = f"Security Violation: {err_details}"
                    scorecard = self._critic.evaluate(
                        code=raw_code,
                        raw_output=None,
                        query=query,
                        context=ctx,
                        scan_violations=scan_res.violations,
                    )
                    last_scorecard = scorecard

                    critique_lines = [
                        "Your code was rejected by the AST Safety Scanner:",
                        scorecard.critique or err_details,
                    ]
                    if scorecard.suggested_fixes:
                        critique_lines.append("\nSUGGESTED REPAIR ACTIONS:")
                        critique_lines.extend(f"- {fix}" for fix in scorecard.suggested_fixes)
                    critique_lines.append("\nFix the code and rewrite the complete `analyze` function.")

                    critique_msg = "\n".join(critique_lines)
                    critique_history.append(critique_msg)

                    # Guardrail 3: Token-budgeted compact retry trace (strictly 4 messages max)
                    messages = [
                        system_msg,
                        initial_user_msg,
                        Message(role="assistant", content=f"```python\n{raw_code}\n```"),
                        Message(role="user", content=critique_msg),
                    ]
                    continue

                # 3. Sandboxed Execution Step
                sandbox_res = await self._sandbox.execute(
                    code=raw_code,
                    query=query,
                    context=ctx,
                )

                if not sandbox_res.success or not sandbox_res.output:
                    last_error = sandbox_res.error or "Sandbox execution failed with unknown error."
                    logger.warning(
                        f"Dynamic tool execution failed on attempt {attempt}: {last_error}"
                    )
                    scorecard = self._critic.evaluate(
                        code=raw_code,
                        raw_output=None,
                        query=query,
                        context=ctx,
                        error=last_error,
                    )
                    last_scorecard = scorecard

                    critique_lines = [
                        "Executing your code in the sandbox raised a runtime error:",
                        f"```\n{last_error}\n```",
                    ]
                    if scorecard.suggested_fixes:
                        critique_lines.append("\nSUGGESTED REPAIR ACTIONS:")
                        critique_lines.extend(f"- {fix}" for fix in scorecard.suggested_fixes)
                    critique_lines.append("\nFix the bug and provide the corrected `analyze` function.")

                    critique_msg = "\n".join(critique_lines)
                    critique_history.append(critique_msg)

                    # Guardrail 3: Token-budgeted compact retry trace
                    messages = [
                        system_msg,
                        initial_user_msg,
                        Message(role="assistant", content=f"```python\n{raw_code}\n```"),
                        Message(role="user", content=critique_msg),
                    ]
                    continue

                # 4. Post-Execution Diagnostic Critic Audit
                scorecard = self._critic.evaluate(
                    code=raw_code,
                    raw_output=sandbox_res.output,
                    query=query,
                    context=ctx,
                )
                last_scorecard = scorecard

                if not scorecard.is_acceptable and attempt < max_retries:
                    critique_lines = [
                        "Your code executed but FAILED quality, schema, or anti-hallucination audits:",
                        scorecard.critique,
                    ]
                    if scorecard.suggested_fixes:
                        critique_lines.append("\nSUGGESTED REPAIR ACTIONS:")
                        critique_lines.extend(f"- {fix}" for fix in scorecard.suggested_fixes)
                    critique_lines.append("\nPlease rewrite the complete `analyze` function resolving these issues.")

                    critique_msg = "\n".join(critique_lines)
                    critique_history.append(critique_msg)
                    last_error = scorecard.critique
                    logger.warning(f"Dynamic tool failed quality audit on attempt {attempt}: {scorecard.critique}")

                    # Guardrail 3: Token-budgeted compact retry trace
                    messages = [
                        system_msg,
                        initial_user_msg,
                        Message(role="assistant", content=f"```python\n{raw_code}\n```"),
                        Message(role="user", content=critique_msg),
                    ]
                    continue

                # Successful execution (or retries exhausted on acceptable output)
                dur_ms = round((time.monotonic() - t_start) * 1000)
                evidence_items = normalize_dynamic_output(sandbox_res.output, query, context=ctx)
                err_msg = None
                if not scorecard.is_acceptable:
                    err_msg = scorecard.critique or "Dynamic tool failed diagnostic quality audit."
                return ToolMakerResult(
                    success=scorecard.is_acceptable,
                    evidence_items=evidence_items if scorecard.is_acceptable else [],
                    code=raw_code,
                    raw_output=sandbox_res.output,
                    scorecard=scorecard,
                    critique_history=critique_history,
                    error=err_msg,
                    retries_used=attempt,
                    execution_time_ms=dur_ms,
                )

            except Exception as e:
                last_error = str(e)
                logger.error(f"Error during ToolMaker iteration {attempt}: {e}")
                if attempt == 0:
                    try:
                        reg = get_registry()
                        curr_name = getattr(provider, "provider_name", "").lower()
                        fallback_id = "ollama_llama3" if "gemini" in curr_name else "gemini_flash"
                        provider = reg.get_chat_provider(fallback_id)
                        logger.info(f"Failing over ToolMaker to fallback provider: {fallback_id}")
                    except Exception as fb_err:
                        logger.warning(f"Could not switch provider on failure: {fb_err}")

        dur_ms = round((time.monotonic() - t_start) * 1000)
        return ToolMakerResult(
            success=False,
            code=last_code,
            scorecard=last_scorecard,
            critique_history=critique_history,
            error=last_error or (last_scorecard.critique if last_scorecard else None) or "Failed to synthesize an acceptable tool after maximum retries.",
            retries_used=max_retries,
            execution_time_ms=dur_ms,
        )


__all__ = [
    "TOOL_MAKER_SYSTEM_PROMPT",
    "EvaluationScorecard",
    "ToolCritic",
    "ToolMaker",
    "ToolMakerResult",
    "ensure_standard_imports",
    "normalize_dynamic_output",
]
