"""Diagnostic evaluation and scorecard critic for LLM-generated dynamic tools.

Audits generated tools across 5 dimensions:
1. SASC: Syntactic & AST Security Compliance
2. SRF: SQL Relational & Schema Fidelity
3. REH: Runtime Execution & Subprocess Health
4. DSF: Internal Data-to-Summary Faithfulness (with Legitimate Absence detection)
5. RPS: Intent Alignment & Filter Plausibility
"""

from __future__ import annotations

import ast
import contextlib
import re
from dataclasses import dataclass, field
from typing import Any

from app.agent.archive_context import (
    ARCHIVE_SCHEMA,
    KNOWN_COLUMN_HALLUCINATIONS,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Evaluation Scorecard Dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvaluationScorecard:
    """Quantitative scorecard and diagnostic critique for a generated dynamic tool."""

    sasc_score: float = 1.0  # Syntactic & AST Security Compliance (1.0 or 0.0)
    srf_score: float = 1.0   # SQL Relational & Schema Fidelity (0.0 to 1.0)
    reh_score: float = 1.0   # Runtime Execution & Subprocess Health (1.0 or 0.0)
    dsf_score: float = 1.0   # Data-to-Summary Faithfulness (0.0 to 1.0)
    rps_score: float = 1.0   # Intent Alignment & Filter Plausibility (0.0 to 1.0)
    is_acceptable: bool = True
    critique: str = ""
    suggested_fixes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ToolCritic Class
# ---------------------------------------------------------------------------


class ToolCritic:
    """Evaluates dynamic tools and produces structured diagnostic feedback for closed-loop refinement."""

    def __init__(self, schema: dict[str, frozenset[str]] | None = None) -> None:
        self._schema = schema or ARCHIVE_SCHEMA

    def extract_sql_queries_ast(self, code: str) -> list[str]:
        """Resiliently extract SQL query strings from Python AST without assuming exact text() formatting."""
        raw_queries: list[str] = []
        try:
            tree = ast.parse(code)
        except Exception:
            return raw_queries

        # 1. Track variable assignments to string literals
        var_strings: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            var_strings[target.id] = node.value.value
                        elif isinstance(node.value, ast.JoinedStr):
                            # Approximate f-string parts
                            parts = [
                                part.value for part in node.value.values
                                if isinstance(part, ast.Constant) and isinstance(part.value, str)
                            ]
                            var_strings[target.id] = " ".join(parts)

        # 2. Find calls to text(...)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name == "text" and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        raw_queries.append(arg.value)
                    elif isinstance(arg, ast.Name) and arg.id in var_strings:
                        raw_queries.append(var_strings[arg.id])

        # 3. Fallback scan for any string constant containing SQL syntax
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value.strip()
                if re.search(r"\bSELECT\b.*?\bFROM\b", s, re.IGNORECASE | re.DOTALL):
                    raw_queries.append(s)

        # Normalize and deduplicate statements
        deduped: list[str] = []
        seen: set[str] = set()
        for q in raw_queries:
            norm = " ".join(q.split()).strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(q.strip())

        return deduped

    def audit_sql_schema(self, sql_queries: list[str]) -> tuple[float, list[str], list[str]]:
        """Audit SQL queries against archive schema, detecting non-existent columns and Cartesian risks."""
        if not sql_queries:
            return 1.0, [], []

        issues: list[str] = []
        fixes: list[str] = []
        penalties = 0.0

        for sql in sql_queries:
            # Check for known column hallucinations
            for bad_col, fix_hint in KNOWN_COLUMN_HALLUCINATIONS.items():
                # 1. Ignore if bad_col is merely an output alias: AS bad_col
                sql_check = re.sub(rf"\bAS\s+[`\"']?{re.escape(bad_col)}[`\"']?\b", "", sql, flags=re.IGNORECASE)

                # 2. Ignore if bad_col is DATE() function call
                if bad_col == "date":
                    sql_check = re.sub(r"\bDATE\s*\(", "", sql_check, flags=re.IGNORECASE)

                # 3. If bad_col was used as an alias in SELECT, also allow referencing it in GROUP BY, ORDER BY, or HAVING
                if re.search(rf"\bAS\s+[`\"']?{re.escape(bad_col)}[`\"']?\b", sql, re.IGNORECASE):
                    sql_check = re.sub(rf"\b(?:GROUP\s+BY|ORDER\s+BY|HAVING)\b[^;]*\b{re.escape(bad_col)}\b", "", sql_check, flags=re.IGNORECASE)

                if bad_col == "category":
                    # Check if 'category' is erroneously referenced directly on articles table
                    # or if category is used as a column without joining article_categories
                    has_direct_col = bool(re.search(r"\b(?:articles|a)\.category\b(?!\s*_id)", sql, re.IGNORECASE))
                    has_unjoined_category = (
                        bool(re.search(r"\bSELECT\b.*?\bcategory\b(?!\s*(?:id|_id|_categories))\b.*?\bFROM\b", sql_check, re.IGNORECASE | re.DOTALL))
                        and not bool(re.search(r"\b(?:JOIN|FROM)\s+article_categories\b", sql, re.IGNORECASE))
                    )
                    if has_direct_col or has_unjoined_category:
                        issues.append(f"SQL queries non-existent column '{bad_col}'.")
                        fixes.append(f"Replace '{bad_col}' with {fix_hint}.")
                        penalties += 0.3
                    continue

                if bad_col in ("page_number", "page", "pages"):
                    # Check if page_number is referenced directly on articles table,
                    # or if pages table is joined incorrectly on primary_page_id/article_id
                    has_direct_col = bool(re.search(r"\b(?:articles|a)\.(?:page_number|page|pages)\b", sql, re.IGNORECASE))
                    has_unjoined_page = (
                        bool(re.search(r"\b(?:WHERE|SELECT|GROUP\s+BY|ORDER\s+BY)\b.*?\bpage_number\b", sql_check, re.IGNORECASE | re.DOTALL))
                        and not bool(re.search(r"\b(?:JOIN|FROM)\s+(?:article_pages|pages)\b", sql, re.IGNORECASE))
                    )
                    has_invalid_page_join = bool(re.search(r"\b(?:pages|p)\.(?:article_id|primary_page_id)\b", sql, re.IGNORECASE))
                    if has_direct_col or has_unjoined_page or has_invalid_page_join:
                        issues.append("SQL references 'page_number' directly on 'articles' table or invalid page join column.")
                        fixes.append("The 'articles' table has NO 'page_number' column. Join `article_pages`: `JOIN article_pages ap ON a.id = ap.article_id WHERE ap.page_number = :page_num`.")
                        penalties += 0.3
                    continue

                if bad_col == "primary_page_id":
                    has_invalid_page_col = bool(re.search(r"\b(?:pages|p)\.primary_page_id\b", sql, re.IGNORECASE))
                    if has_invalid_page_col:
                        issues.append("SQL references 'primary_page_id' on 'pages' table.")
                        fixes.append("Join `pages` on `articles.primary_page_id = pages.id` or use `JOIN article_pages ap ON articles.id = ap.article_id`.")
                        penalties += 0.3
                    continue

                pattern = rf"\b{re.escape(bad_col)}\b"
                if re.search(pattern, sql_check, re.IGNORECASE):
                    issues.append(f"SQL queries non-existent column '{bad_col}'.")
                    fixes.append(f"Replace '{bad_col}' with {fix_hint}.")
                    penalties += 0.3

            # Check for string formatting in SQL (SQL injection / unparameterized risk)
            if re.search(r"f['\"].*?SELECT", sql, re.IGNORECASE) or "%s" in sql or "{" in sql:
                if "{" in sql and "}" in sql and not sql.strip().startswith("SELECT"):
                    pass  # might be dict
                elif re.search(r"\{[a-zA-Z_]\w*\}", sql):
                    issues.append("SQL query contains string interpolation (f-string/format) instead of :param binding.")
                    fixes.append("Always use SQLAlchemy parameterized binding: `text('... WHERE col = :val')`.")
                    penalties += 0.2

            # Check Cartesian multiplier risk
            # Joining articles and pages/photos without DISTINCT on COUNT
            has_articles = bool(re.search(r"\bFROM\s+articles\b|\bJOIN\s+articles\b", sql, re.IGNORECASE))
            has_pages = bool(re.search(r"\bJOIN\s+pages\b", sql, re.IGNORECASE))
            has_photos = bool(re.search(r"\bJOIN\s+photos\b", sql, re.IGNORECASE))
            if (
                has_articles
                and (has_pages or has_photos)
                and re.search(r"COUNT\s*\(\s*(?:a\.)?id\s*\)", sql, re.IGNORECASE)
                and not re.search(r"COUNT\s*\(\s*DISTINCT", sql, re.IGNORECASE)
            ):
                issues.append("Potential Cartesian product: Counting articles across 1:N join without DISTINCT.")
                fixes.append("Use `COUNT(DISTINCT a.id)` when joining `pages` or `photos`.")
                penalties += 0.2

        score = max(0.0, 1.0 - penalties)
        return score, issues, fixes

    def audit_data_to_summary(
        self,
        summary: str,
        data: list[dict[str, Any]],
        metadata: dict[str, Any],
        query: str,
        context: dict[str, Any],
    ) -> tuple[float, list[str], list[str]]:
        """Audit data-to-summary faithfulness with strict 'Legitimate Absence' discrimination."""
        issues: list[str] = []
        fixes: list[str] = []

        summary_lower = (summary or "").lower()
        has_data = bool(data)

        # Absence acknowledgments with word boundaries to avoid false positives (e.g. '10 articles' matching '0 articles')
        absence_markers = (
            r"\bno\s+articles\s+found\b",
            r"\bno\s+newspaper\s+issue\s+found\b",
            r"\bno\s+records\b",
            r"\binsufficient\s+data\b",
            r"\b0\s+articles\b",
            r"\bnot\s+found\b",
            r"\b0\s+matching\b",
            r"\bnone\s+found\b",
            r"\bno\s+matching\b",
            r"\bempty\b",
        )
        acknowledges_absence = any(re.search(marker, summary_lower) for marker in absence_markers)

        # Check whether result contains aggregate statistical computations
        # In aggregate queries (e.g. photo count per section, correlations, distributions),
        # individual article records in 'data' are omitted, but 'metadata' contains computed metrics
        # or 'summary' contains markdown tables / structured findings.
        has_metadata_metrics = bool(
            metadata
            and any(
                isinstance(v, (int, float, dict, list)) and bool(v)
                for v in metadata.values()
            )
        )
        has_markdown_table = bool(re.search(r"\|.*\|.*\|\s*\n\|[\s\-:]+\|", summary or ""))
        is_aggregate_computation = has_metadata_metrics or has_markdown_table

        # 1. Guardrail 1: If neither article data NOR aggregate computation results exist
        if not has_data and not is_aggregate_computation:
            if acknowledges_absence:
                # Legitimate absence truthfully reported!
                return 1.0, [], []

            # Data is empty but summary claims positive results -> Severe Hallucination!
            positive_claims = re.findall(r"\b(\d+)\s+(?:articles|pages|reports|issues|stories|photos)\b", summary_lower)
            if positive_claims:
                issues.append(
                    f"Hallucination detected: SQL query returned 0 rows, but summary asserts positive counts: {positive_claims}."
                )
                fixes.append("Align summary strictly with returned data. When rows are empty, state that no records were found.")
                return 0.1, issues, fixes

            if len(summary.strip()) > 150:
                issues.append("SQL query returned 0 records, but summary contains an extensive narrative.")
                fixes.append("Ensure summary accurately reflects that no matching records were returned by the archive.")
                return 0.4, issues, fixes

            return 0.9, [], []

        # 2. Check for NaN / null in summary or metadata (computational failure or empty dataset)
        summary_has_nan = bool(
            re.search(r"\b(?:nan|null)\s*(?:words?|articles?|issues?|pages?|%|\b)", summary_lower)
            or "is nan" in summary_lower
        )
        metadata_has_nan = any(
            (isinstance(v, float) and (v != v or str(v).lower() == "nan"))
            or (isinstance(v, str) and v.lower() in ("nan", "null"))
            for v in metadata.values()
        )
        if summary_has_nan or metadata_has_nan:
            row_count = metadata.get("row_count") if "row_count" in metadata else (metadata.get("total_rows") if "total_rows" in metadata else len(data))
            if row_count == 0 and acknowledges_absence:
                # Legitimate empty subset result where average/metric is mathematically undefined on 0 rows
                return 1.0, [], []
            issues.append("Calculation produced NaN (Not a Number) or null values in summary or metadata metrics.")
            fixes.append(
                "Verify SQL WHERE clauses and table joins against schema. Check that rows exist before computing statistics "
                "(e.g. df['col'].mean() on empty DataFrame yields NaN). Use pre-normalized context parameters "
                "(context.get('target_date'), context.get('newspaper_name')) and handle empty results gracefully."
            )
            return 0.0, issues, fixes

        # 3. When data or aggregate computation is present: check numerical consistency
        # Extract numbers in metadata vs summary
        penalties = 0.0
        for k, v in metadata.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if "page" in k.lower() and isinstance(v, int):
                    m_pages = re.findall(r"\b(\d+)\s+pages?\b", summary_lower)
                    for p in m_pages:
                        with contextlib.suppress(ValueError):
                            if int(p) != v and v > 0:
                                issues.append(f"Number mismatch: metadata['{k}'] is {v}, but summary states {p} pages.")
                                fixes.append(f"Report the exact value of {k} ({v}) in summary.")
                                penalties += 0.2
                elif "photo" in k.lower() and isinstance(v, int):
                    m_photos = re.findall(r"\b(\d+)\s+photos?\b", summary_lower)
                    for p in m_photos:
                        with contextlib.suppress(ValueError):
                            if int(p) != v and v > 0:
                                issues.append(f"Number mismatch: metadata['{k}'] is {v}, but summary states {p} photos.")
                                fixes.append(f"Report the exact value of {k} ({v}) in summary.")
                                penalties += 0.2

        score = max(0.0, 1.0 - penalties)
        return score, issues, fixes

    def audit_filter_plausibility(
        self,
        code: str,
        query: str,
        context: dict[str, Any],
        is_empty_result: bool,
    ) -> tuple[float, list[str], list[str]]:
        """Detect formatting defects (non-ISO dates, alias mismatches) when query returns 0 rows."""
        issues: list[str] = []
        fixes: list[str] = []

        if not is_empty_result:
            return 1.0, [], []

        available_dates = context.get("available_dates", [])
        available_newspapers = context.get("available_newspapers", [])

        # Check for un-normalized date format passed into SQL
        # e.g. '2/8/2026' or '02-08-2026' in query or code
        dmy_match = re.search(r"(\b\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", query)
        if dmy_match:
            d, m, y = dmy_match.groups()
            expected_iso = context.get("target_date") or f"{y}-{int(m):02d}-{int(d):02d}"
            normalizes_date = (
                expected_iso in code
                or (("target_date" in code or "date_from" in code) and "context" in code)
                or ("int(m)" in code or "strftime" in code or "strptime" in code)
            )
            # Check if code didn't convert to ISO or use normalized date
            if (dmy_match.group(0) in code and not normalizes_date) or (not normalizes_date and expected_iso not in code):
                issues.append(f"Date formatting mismatch: Query mentions date '{dmy_match.group(0)}' which was not normalized to ISO '{expected_iso}'.")
                fixes.append(f"Convert input date '{dmy_match.group(0)}' to MySQL DATE format '{expected_iso}' or use `context.get('target_date')` directly.")
                return 0.3, issues, fixes

        # Check for pandas comparison against SQL wildcards
        if re.search(r'==\s*np_pattern|\.str\.lower\(\)\s*==\s*.*pattern', code):
            issues.append("Pandas wildcard comparison defect: Code compares a pandas DataFrame column directly against a pattern containing SQL wildcards '%', resulting in 0 matching rows.")
            fixes.append("Remove redundant pandas equality filtering on wildcard patterns, or use `df[col].str.contains(clean_name, case=False, na=False)` without '%' signs.")
            return 0.3, issues, fixes

        # Check newspaper name matching
        # e.g. user says 'goan', code searches '= 'Goan'' instead of 'The Goan' or LIKE '%Goan%'
        uses_context_np = "newspaper_name" in code and "context" in code
        if not uses_context_np:
            for np in available_newspapers:
                clean_np = np.lower().replace("the ", "").strip()
                if clean_np in query.lower() and (f"'{clean_np}'" in code or f'"{clean_np}"' in code):
                    issues.append(f"Newspaper naming mismatch: Searched for '{clean_np}' but database record is '{np}'.")
                    fixes.append(f"Use `LOWER(n.name) LIKE '%{clean_np}%'` or exact match '{np}'.")
                    return 0.4, issues, fixes

        # If available dates exist and user asked for date out of bounds
        if available_dates:
            iso_dates_in_query = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
            for idt in iso_dates_in_query:
                if idt not in available_dates:
                    # Legitimate out-of-range date!
                    return 1.0, [], []

        return 1.0, [], []

    def evaluate(
        self,
        code: str,
        raw_output: dict[str, Any] | None,
        query: str,
        context: dict[str, Any],
        error: str | None = None,
        scan_violations: list[str] | None = None,
    ) -> EvaluationScorecard:
        """Execute full 5-metric evaluation and build structured diagnostic critique."""
        all_issues: list[str] = []
        all_fixes: list[str] = []

        # 1. Metric 1: Syntactic & AST Security Compliance (SASC)
        sasc_score = 1.0
        if scan_violations:
            sasc_score = 0.0
            all_issues.append(f"Security/AST violations: {'; '.join(scan_violations)}")
            all_fixes.append("Remove all forbidden modules, built-ins, and unsafe attributes.")

        # 2. Metric 2: SQL Relational & Schema Fidelity (SRF)
        sql_queries = self.extract_sql_queries_ast(code)
        srf_score, srf_issues, srf_fixes = self.audit_sql_schema(sql_queries)
        all_issues.extend(srf_issues)
        all_fixes.extend(srf_fixes)

        # 3. Metric 3: Runtime Execution Health (REH)
        reh_score = 1.0
        if error:
            reh_score = 0.0
            all_issues.append(f"Subprocess runtime error: {error}")
            if "UnboundLocalError" in error and "text" in error:
                all_fixes.append("UnboundLocalError on 'text': You assigned to a variable named 'text = ...' which shadows 'from sqlalchemy import text'. Rename your variable to 'snippet', 'content', or 'article_text'.")
            elif "strptime" in error and ("None" in error or "str" in error):
                all_fixes.append(
                    "TypeError in strptime(): strptime() was called with a None value. "
                    "Dates from `context` (e.g. `context.get('target_date')` or `context.get('issue_date')`) are ALREADY ISO strings ('YYYY-MM-DD'). "
                    "Pass them directly into SQL parameters without calling datetime.strptime(). "
                    "If parsing optional dates from text, always guard with `if date_str:`."
                )
            elif "KeyError" in error:
                m_key = re.search(r"KeyError:\s*['\"]?([^'\"]+)['\"]?", error)
                bad_k = m_key.group(1) if m_key else "the key"
                all_fixes.append(
                    f"KeyError on '{bad_k}': You accessed '{bad_k}' in a dictionary or DataFrame, but it was not present. "
                    f"Make sure you explicitly SELECT this column in your SQL query (e.g. `SELECT c.name AS {bad_k}...` or `SELECT a.{bad_k}...`), "
                    f"and verify that your pandas column names or dictionary accesses match your SQL SELECT aliases exactly."
                )
            elif "StatementError" in error or "bind parameter" in error:
                m_param = re.search(r"bind parameter '([^']+)'", error)
                p_name = m_param.group(1) if m_param else "parameter"
                all_fixes.append(
                    f"Missing bind parameter '{p_name}': Your SQL query contains `:{p_name}`, but '{p_name}' was not in the parameters dictionary passed to `db.execute()`. "
                    f"Ensure every `:{p_name}` placeholder in your `text(...)` query has an exact matching key in the dictionary: `db.execute(stmt, {{'{p_name}': ...}})`."
                )
            else:
                all_fixes.append("Fix uncaught exception, type error, or missing await.")

        # 4. Metric 4: Internal Data-to-Summary Faithfulness (DSF)
        # 5. Metric 5: Intent Alignment & Filter Plausibility (RPS)
        dsf_score = 1.0
        rps_score = 1.0

        if raw_output and isinstance(raw_output, dict):
            summary = raw_output.get("summary") or ""
            data = raw_output.get("data") or []
            metadata = raw_output.get("metadata") or {}
            has_metadata_metrics = bool(
                metadata
                and any(
                    isinstance(v, (int, float, dict, list)) and bool(v)
                    for v in metadata.values()
                )
            )
            has_markdown_table = bool(re.search(r"\|.*\|.*\|\s*\n\|[\s\-:]+\|", summary or ""))
            is_empty = not bool(data) and not has_metadata_metrics and not has_markdown_table

            dsf_score, dsf_issues, dsf_fixes = self.audit_data_to_summary(
                summary=summary,
                data=data,
                metadata=metadata,
                query=query,
                context=context,
            )
            all_issues.extend(dsf_issues)
            all_fixes.extend(dsf_fixes)

            rps_score, rps_issues, rps_fixes = self.audit_filter_plausibility(
                code=code,
                query=query,
                context=context,
                is_empty_result=is_empty,
            )
            all_issues.extend(rps_issues)
            all_fixes.extend(rps_fixes)
        elif not error and not scan_violations:
            reh_score = 0.0
            all_issues.append("Sandbox returned empty or malformed output dictionary.")
            all_fixes.append("Ensure `analyze` returns a dictionary with 'summary', 'data', and 'metadata'.")

        # Determine Acceptability
        # SASC and REH must be 1.0 (hard requirements)
        # SRF, DSF, RPS must meet quality thresholds
        is_acceptable = (
            sasc_score == 1.0
            and reh_score == 1.0
            and srf_score >= 0.70
            and dsf_score >= 0.70
            and rps_score >= 0.70
        )

        critique_text = ""
        if not is_acceptable:
            critique_lines = ["Audit diagnostics detected the following defects:"]
            for idx, issue in enumerate(all_issues, 1):
                critique_lines.append(f"{idx}. {issue}")
            critique_text = "\n".join(critique_lines)

        return EvaluationScorecard(
            sasc_score=sasc_score,
            srf_score=srf_score,
            reh_score=reh_score,
            dsf_score=dsf_score,
            rps_score=rps_score,
            is_acceptable=is_acceptable,
            critique=critique_text,
            suggested_fixes=all_fixes,
        )


__all__ = [
    "ARCHIVE_SCHEMA",
    "EvaluationScorecard",
    "KNOWN_COLUMN_HALLUCINATIONS",
    "ToolCritic",
]
