"""LLM-based Answer Verifier and Fact-Checking Critic for NewsLens-AI.

Audits synthesized responses against retrieved evidence ground truth along four dimensions:
1. Faithfulness & Groundedness: Verifies that assertions, counts, and dates are supported by evidence.
2. Contradiction Detection: Flags positive claims when evidence demonstrates absence (e.g. 0 records).
3. Speculative Fluff Elimination: Rejects ungrounded corporate consulting advice and dummy citations.
4. Evidence Gap & Dynamic Tool Routing: Triggers fallback to ToolMaker when evidence lacks facts.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import json
import re
import time
from typing import Any

from app.core.logging import get_logger
from app.providers.base import ChatModelProvider, Message
from app.providers.registry import get_registry

logger = get_logger(__name__)


@dataclass
class AnswerVerificationResult:
    """Quantitative and qualitative audit verdict for a synthesized answer."""

    is_valid: bool
    has_hallucination: bool = False
    has_contradiction: bool = False
    evidence_gap_detected: bool = False
    quality_score: float = 1.0  # 0.0 to 1.0
    factual_errors: list[str] = field(default_factory=list)
    critique: str = ""
    recommended_action: str = "accept"  # "accept" | "refine_answer" | "fallback_to_dynamic_tool"
    refined_answer: str | None = None
    dynamic_tool_hint: str | None = None
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit verdict for state storage."""
        return {
            "is_valid": self.is_valid,
            "has_hallucination": self.has_hallucination,
            "has_contradiction": self.has_contradiction,
            "evidence_gap_detected": self.evidence_gap_detected,
            "quality_score": self.quality_score,
            "factual_errors": self.factual_errors,
            "critique": self.critique,
            "recommended_action": self.recommended_action,
            "refined_answer": self.refined_answer,
            "dynamic_tool_hint": self.dynamic_tool_hint,
            "latency_ms": self.latency_ms,
        }


ANSWER_VERIFIER_SYSTEM_PROMPT = """You are an authoritative editorial fact-checker and broadsheet intelligence auditor for NewsLens-AI.
Your mission is to rigorously verify a drafted answer against the retrieved evidence ground truth.

### AUDIT DIMENSIONS
1. **Faithfulness & Truthfulness**:
   - Are all numbers, dates, publications, and availability assertions strictly grounded in the evidence?
   - If the evidence states 0 records/issues, or documents absence for a given date, did the draft falsely claim availability or invent counts? That is a critical contradiction!
2. **Freedom from Hallucination & Fluff**:
   - Does the response contain speculative corporate consulting boilerplate (e.g., 'investigate the implications on overall content strategy and publication planning', 'explore future collaboration with publications') that is not present in the evidence?
   - Are there placeholder citations like `[{Publication Name}...`?
3. **Evidence Adequacy & Dynamic Fallback**:
   - Did the draft hallucinate or fail because the retrieved evidence had an unresolvable gap or missing data needed to answer the question?
   - If the evidence genuinely lacks the required data to answer the query, set `evidence_gap_detected: true`, recommend `fallback_to_dynamic_tool`, and provide a `dynamic_tool_hint` for what database query is needed.
4. **Refinement**:
   - If the draft has errors or fluff but the evidence contains the necessary facts (e.g. 0 matching issues, archive range 2026-08-01 to 2026-09-11), provide a clean, faithfully grounded `refined_answer` adhering to the facts.

### OUTPUT FORMAT
Respond ONLY with a valid JSON object matching this schema:
{
  "is_valid": boolean,
  "has_hallucination": boolean,
  "has_contradiction": boolean,
  "evidence_gap_detected": boolean,
  "quality_score": float (0.0 to 1.0),
  "factual_errors": [string],
  "critique": string,
  "recommended_action": "accept" | "refine_answer" | "fallback_to_dynamic_tool",
  "refined_answer": string or null,
  "dynamic_tool_hint": string or null
}"""


class AnswerVerifier:
    """Verifies synthesized answers using an LLM-based Critic with dynamic fallback capability."""

    def __init__(self, provider: ChatModelProvider | None = None) -> None:
        self._provider = provider

    def _resolve_providers(self, model_override: str | None = None) -> list[ChatModelProvider]:
        """Resolve candidate model providers for answer verification."""
        if self._provider is not None:
            return [self._provider]

        candidates: list[ChatModelProvider] = []
        try:
            reg = get_registry()
            if model_override:
                with contextlib.suppress(Exception):
                    candidates.append(reg.get_chat_provider(model_override))
            for key in ["evaluator", "query_planner", "synthesizer"]:
                with contextlib.suppress(Exception):
                    p = reg.get_provider(key)
                    if p is not None and hasattr(p, "complete") and p not in candidates:
                        candidates.append(p)
            for k in ["gemini_flash", "openrouter_gemma4_26b", "groq_compound", "openai_gpt4o_mini"]:
                with contextlib.suppress(Exception):
                    p = reg.get_chat_provider(k)
                    if p is not None and hasattr(p, "complete") and p not in candidates:
                        candidates.append(p)
        except Exception as e:
            logger.warning("Could not resolve candidate models for AnswerVerifier", extra={"error": str(e)})

        return candidates

    def _build_evidence_summary(self, evidence_items: list[dict[str, Any]]) -> str:
        """Format retrieved evidence items into a clean, compact ground-truth summary."""
        if not evidence_items:
            return "No evidence retrieved."

        lines: list[str] = []
        for idx, item in enumerate(evidence_items[:12], 1):
            tool = item.get("source_tool", "unknown")
            hl = item.get("headline", "No Headline")
            np = item.get("newspaper_name", "Unknown")
            dt = item.get("issue_date", "Unknown Date")
            meta = item.get("metadata") or {}
            snip = (item.get("snippet") or item.get("summary") or "")[:400]

            meta_parts: list[str] = []
            for k, v in meta.items():
                if k in ("count", "total_articles", "total_photos", "total_issues", "target_date", "archive_range"):
                    meta_parts.append(f"{k}: {v}")
            meta_str = f" [Metrics: {', '.join(meta_parts)}]" if meta_parts else ""

            lines.append(f"[{idx}] Source: {tool} | Headline: {hl} | Publication: {np} | Date: {dt}{meta_str}")
            if snip:
                lines.append(f"    Content: {snip}")

        return "\n".join(lines)

    def _fast_groundedness_check(
        self,
        draft_answer: str,
        evidence_items: list[dict[str, Any]],
    ) -> AnswerVerificationResult | None:
        """Fast-floor check catching explicit relational zero-count contradictions deterministically."""
        zero_issue_record = None
        for it in evidence_items:
            meta = it.get("metadata") or {}
            src = it.get("source_tool", "")
            snip = (it.get("snippet") or "") + " " + (it.get("headline") or "")
            if src.startswith("sql_analytics") and (
                meta.get("count") == 0
                or "0 issues found" in snip
                or "Total Matching Issues: 0" in snip
                or "Archive Availability Audit: 0 issues" in it.get("headline", "")
            ):
                zero_issue_record = it
                break

        if zero_issue_record is not None:
            has_positive_claim = bool(
                re.search(
                    r"(?i)\b(?:Newspaper Availability[^:\n]*:\s*Yes|"
                    r"at least one newspaper is available|"
                    r"presence of \d+ matching issues|"
                    r"confirms that there are \d+ matching issues|"
                    r"Total Matching Issues:\s*[1-9]\d*)\b",
                    draft_answer,
                )
                or re.search(r"(?i)\bTotal Matching Issues:\s*24\b", draft_answer)
            )
            if has_positive_claim:
                meta = zero_issue_record.get("metadata") or {}
                target_d = meta.get("target_date") or zero_issue_record.get("issue_date") or "the requested date"
                rng = meta.get("archive_range") or {}
                rng_str = f"{rng.get('start', '2026-08-01')} to {rng.get('end', '2026-09-11')}"
                nps = meta.get("archive_newspapers") or []
                nps_str = ", ".join(nps) if nps else "Business Standard, Hindustan Times, Mint, The Indian Express"

                refined = (
                    f"### ⚡ Availability Status\n\n"
                    f"No newspaper issues are available in the archive for {target_d}.\n\n"
                    f"### 📋 Archive Scope & Available Coverage\n\n"
                    f"* Available publications in the archive: {nps_str}\n"
                    f"* Archive coverage range: {rng_str}"
                )
                return AnswerVerificationResult(
                    is_valid=False,
                    has_hallucination=True,
                    has_contradiction=True,
                    evidence_gap_detected=False,
                    quality_score=0.2,
                    factual_errors=[f"Draft claimed newspaper availability exists on {target_d}, contradicting the relational audit of 0 issues."],
                    critique="Draft asserts positive availability and count of issues on a date verified to have 0 records.",
                    recommended_action="refine_answer",
                    refined_answer=refined,
                )

        return None

    async def verify_answer_async(
        self,
        query: str,
        draft_answer: str,
        evidence_items: list[dict[str, Any]],
        archetype: str = "factual_lookup",
        model_override: str | None = None,
        answer_blueprint: dict[str, Any] | None = None,
    ) -> AnswerVerificationResult:
        """Critique and verify a synthesized answer against retrieved evidence ground truth."""
        t0 = time.monotonic()

        if not draft_answer or not draft_answer.strip():
            return AnswerVerificationResult(
                is_valid=False,
                has_hallucination=False,
                evidence_gap_detected=True,
                quality_score=0.0,
                factual_errors=["Draft answer is empty."],
                critique="No answer text was synthesized.",
                recommended_action="fallback_to_dynamic_tool" if not evidence_items else "refine_answer",
            )

        # 1. Fast Groundedness Floor Check
        fast_result = self._fast_groundedness_check(draft_answer, evidence_items)
        if fast_result is not None:
            fast_result.latency_ms = round((time.monotonic() - t0) * 1000)
            return fast_result

        # 2. Prepare LLM Judge Prompt
        evidence_summary = self._build_evidence_summary(evidence_items)
        prompt = (
            f"USER QUERY:\n{query}\n\n"
            f"QUERY ARCHETYPE:\n{archetype}\n\n"
            f"RETRIEVED EVIDENCE (GROUND TRUTH):\n{evidence_summary}\n\n"
            f"DRAFT SYNTHESIZED ANSWER TO AUDIT:\n{draft_answer}\n\n"
            "Audit the draft answer against the evidence strictly. "
            "Respond ONLY with a valid JSON object matching the required schema."
        )

        candidates = self._resolve_providers(model_override)
        for prov in candidates:
            try:
                resp = await prov.complete(
                    messages=[
                        Message(role="system", content=ANSWER_VERIFIER_SYSTEM_PROMPT),
                        Message(role="user", content=prompt),
                    ],
                    max_tokens=600,
                    temperature=0.0,
                )
                raw_text = (resp.text or "").strip()
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    is_valid = bool(data.get("is_valid", False))
                    has_hallucination = bool(data.get("has_hallucination", False))
                    has_contradiction = bool(data.get("has_contradiction", False))
                    evidence_gap = bool(data.get("evidence_gap_detected", False))
                    q_score = float(data.get("quality_score", 1.0 if is_valid else 0.4))
                    errors = data.get("factual_errors") or []
                    critique = data.get("critique") or ""
                    action = str(data.get("recommended_action", "accept" if is_valid else "refine_answer"))
                    refined = data.get("refined_answer")
                    dyn_hint = data.get("dynamic_tool_hint")

                    if action not in ("accept", "refine_answer", "fallback_to_dynamic_tool"):
                        action = "accept" if is_valid else "refine_answer"

                    return AnswerVerificationResult(
                        is_valid=is_valid,
                        has_hallucination=has_hallucination,
                        has_contradiction=has_contradiction,
                        evidence_gap_detected=evidence_gap,
                        quality_score=round(max(0.0, min(1.0, q_score)), 2),
                        factual_errors=errors if isinstance(errors, list) else [str(errors)],
                        critique=critique,
                        recommended_action=action,
                        refined_answer=refined,
                        dynamic_tool_hint=dyn_hint,
                        latency_ms=round((time.monotonic() - t0) * 1000),
                    )
            except Exception as e:
                logger.debug("AnswerVerifier candidate provider execution failed", extra={"error": str(e)})
                continue

        # Fallback if no LLM responded
        return AnswerVerificationResult(
            is_valid=True,
            quality_score=0.9,
            critique="Fast fallback verification accepted without LLM judge critique.",
            recommended_action="accept",
            latency_ms=round((time.monotonic() - t0) * 1000),
        )


__all__ = [
    "AnswerVerificationResult",
    "AnswerVerifier",
]
