"""LLM-as-Tool-Maker dynamic tool generation and execution engine."""

from __future__ import annotations

import contextlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.agent.sandbox import ASTSafetyScanner, SandboxedExecutor, ScanResult
from app.agent.tool_critic import EvaluationScorecard, ToolCritic
from app.agent.tool_maker_prompts import (
    TOOL_MAKER_SYSTEM_PROMPT,
)
from app.core.logging import get_logger
from app.providers.base import Message
from app.providers.registry import get_registry

logger = get_logger(__name__)


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
        """Resolve LLM provider for tool code generation via shared failover candidates."""
        if self._provider is not None:
            return self._provider
        from app.agent.llm_helpers import resolve_provider_candidates

        candidates = resolve_provider_candidates(
            task_name="query_planner",
            model_override=model_override,
            fallback_task_keys=("query_condenser", "answerer"),
            prefer_local=False,
        )
        if candidates:
            return candidates[0]
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
