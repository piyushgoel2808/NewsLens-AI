"""Answer Synthesizer: Grounded LLM narrative generation with strict source citations."""

from __future__ import annotations

import contextlib
import re
from collections.abc import AsyncIterator
from typing import Any

from app.agent.answer_verifier import AnswerVerifier
from app.agent.citation_resolver import (
    extract_citations_from_text,
    make_citation,
    resolve_authoritative_citations,
)
from app.agent.fallback_presenter import (
    EMPTY_EVIDENCE_RESPONSE,
    generate_deterministic_summary,
    has_valid_evidence,
    render_broadsheet_perspectives,
    render_comparison_matrix,
    render_explore_further,
    render_front_page_comparison,
)
from app.agent.llm_helpers import resolve_provider_candidates
from app.agent.models import AnswerBlueprint
from app.agent.prompt_context import (
    build_evidence_context,
    build_synthesizer_user_prompt,
    clean_snippet,
)
from app.agent.state import AgentCitation
from app.agent.synthesizer_prompts import (
    _DEFAULT_STRUCTURE,
    COMMON_ANALYTICAL_GUIDELINES,
    COMMON_MEMORY_AND_CONSTRAINTS,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_synthesizer_system_prompt,
    compile_structure_from_blueprint,
)
from app.agent.taxonomy import (
    DOMAIN_TAXONOMY,
    detect_domain_from_query,
    is_domain_match,
    score_evidence_item,
)
from app.core.cost_tracker import record_usage_and_cost
from app.core.logging import get_logger
from app.providers.base import ChatModelProvider, Message, ProviderCapability
from app.providers.registry import get_registry

logger = get_logger(__name__)

# Re-exports and backward-compatibility aliases
_clean_snippet = clean_snippet
_detect_domain_from_query = detect_domain_from_query
_is_domain_match = is_domain_match

DEFAULT_CLOUD_FAILOVER: tuple[str, ...] = ()
DEFAULT_LOCAL_FAILOVER: tuple[str, ...] = ()


def parse_thought_and_answer(text: str) -> tuple[str, str]:
    """Extract thought/reasoning trace and clean response text from model output."""
    if not text:
        return "", ""

    split_pattern = (
        r"\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|"
        r"In summary|Summary:|Answer:|Draft:\s*\n|Executive Summary)"
    )

    if "<think>" in text:
        if "</think>" in text:
            match = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
            if match:
                thought = match.group(1).strip()
                ans = match.group(2).strip()
                if ans:
                    return thought, ans
                split_match = re.search(split_pattern, thought, flags=re.IGNORECASE)
                if split_match:
                    s_idx = split_match.start()
                    return thought[:s_idx].strip(), thought[s_idx:].strip()
                return thought, ""
        else:
            after_think = text.split("<think>", 1)[1]
            split_match = re.search(split_pattern, after_think, flags=re.IGNORECASE)
            if split_match:
                split_idx = split_match.start()
                return after_think[:split_idx].strip(), after_think[split_idx:].strip()
            return after_think.strip(), ""

    reasoning_prefix_match = re.match(
        r"^(?:Here'?s a thinking process:?|Thinking Process:?|Thought:?)\s*",
        text,
        flags=re.IGNORECASE,
    )
    if reasoning_prefix_match:
        split_match = re.search(split_pattern, text, flags=re.IGNORECASE)
        if split_match:
            s_idx = split_match.start()
            thought_part = text[:s_idx].strip()
            ans_candidate = text[s_idx:].strip()
            ans_candidate = re.sub(r"^Draft:\s*\n*", "", ans_candidate, flags=re.IGNORECASE).strip()
            return thought_part, ans_candidate
        return text.strip(), ""

    ans_text = re.sub(
        r"(?i)^(?:\*{0,2}EXECUTIVE\s+INTELLIGENCE\s+BRIEFING\*{0,2}\s*\n+)?Date:\s*[A-Za-z]+\s+\d{1,2},?\s+20\d{2}\s*\([^\)]*Current Analysis[^\)]*\)\s*Subject:[^\n]+\n*",
        "",
        text.strip(),
    ).strip()

    ans_text = deduplicate_repetitive_lines(ans_text)
    return "", ans_text


def deduplicate_repetitive_lines(text: str) -> str:
    """Deduplicate cyclical or repeating bullet points/lines from local LLM generation."""
    if not text:
        return ""
    lines = text.split("\n")
    if len(lines) < 4:
        return text

    deduped: list[str] = []
    seen_bullets: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("*", "-", "•")):
            norm = re.sub(r"\[.*?\]", "", stripped).strip().lower()
            if len(norm) > 15:
                if norm in seen_bullets:
                    continue
                seen_bullets.add(norm)
        deduped.append(line)

    return "\n".join(deduped)


_CATALOG_BLOCK_REGEX = re.compile(
    r"(?:^|\n)#{1,4}\s*📊\s*(?:Key Findings and Statistics|Article Breakdown by Section|Article Details|Key Visual Elements|Conclusion)[\s\S]*?(?=(?:\n#{1,4}\s+|\Z))",
    re.IGNORECASE,
)

_CORPORATE_FILLER_REGEX = re.compile(
    r"(?im)^[*-•]?\s*(?:investigate|analyze|examine|explore)\s+(?:the\s+)?(?:implications\s+of\s+the\s+findings\s+on\s+(?:the\s+)?overall\s+content\s+strategy|potential\s+for\s+future\s+collaboration\s+with\s+the\s+publications|relevance\s+of\s+the\s+findings\s+in\s+the\s+context\s+of\s+(?:the\s+)?publications)[^\n]*\n?",
)

_EMPTY_SECTION_REGEX = re.compile(
    r"(?:^|\n)#{1,4}\s*(?:🔍\s*)?Explore Further\s*\n+(?=\n#{1,4}|\Z)",
    re.IGNORECASE,
)


def clean_synthesized_answer(
    text: str,
    query: str,
    archetype: str,
    evidence_items: list[dict[str, Any]] | None = None,
) -> str:
    """Clean and post-process synthesized answer to eliminate robotic catalog blocks, speculative consulting filler, and ungrounded hallucinations."""
    if not text:
        return ""

    text = _CORPORATE_FILLER_REGEX.sub("", text)
    text = _EMPTY_SECTION_REGEX.sub("", text)
    text = re.sub(
        r"\[\{(?:Publication|Newspaper)\s+Name\}[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    q_lower = query.lower()
    is_single = bool(
        re.search(
            r"\b(?:tell me about|explain|summ[ae]ri[sz]e|find|read|what does|describe|details? of|takeaways? from)\s+(?:about|on|for)?\s*(?:this|the|that|ir)?\s*(?:article|story|piece|it)\b"
            r"|\b(?:in\s+\d+\s+words?|brief\s+summary|key\s+takeaways?)\b",
            q_lower,
        )
        or bool(re.search(r"[\"“][^\"”]{8,150}[\"”]", query))
    ) and not any(w in q_lower for w in ["how many", "count", "list all", "catalog", "compare all"])

    if is_single and _CATALOG_BLOCK_REGEX.search(text):
        cleaned = _CATALOG_BLOCK_REGEX.sub("\n", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if len(cleaned) >= 80:
            text = cleaned

    return text.strip()


def verify_and_correct_answer_groundedness(
    answer_text: str,
    query: str,
    evidence_items: list[dict[str, Any]],
    archetype: str | None = None,
) -> tuple[str, bool, str | None]:
    """Fact-check synthesized answer against evidence ground truth."""
    if not answer_text:
        return answer_text, False, None

    cleaned_ans = clean_synthesized_answer(answer_text, query, archetype or "", evidence_items)
    verifier = AnswerVerifier()
    res = verifier._fast_groundedness_check(cleaned_ans, evidence_items, query=query)
    if res is not None and not res.is_valid and res.refined_answer:
        return res.refined_answer, True, res.critique

    was_modified = cleaned_ans != answer_text
    return cleaned_ans, was_modified, "Scrubbed formatting and filler from answer." if was_modified else None


def resolve_dynamic_token_budget(
    provider: Any,
    archetype: str | None = None,
    answer_blueprint: AnswerBlueprint | dict[str, Any] | None = None,
    query: str | None = None,
) -> int:
    """Dynamically allocate the maximum output token budget for LLM response generation."""
    capability: ProviderCapability | None = getattr(provider, "capability", None)
    if isinstance(capability, ProviderCapability):
        max_out = capability.max_output_tokens or 4096
        is_reasoning = capability.is_reasoning_model
        headroom = capability.reasoning_headroom
    else:
        model_name = str(getattr(provider, "_model", getattr(provider, "model_name", "")) or "").lower()
        is_reasoning = any(k in model_name for k in ("2.5", "thinking", "r1", "nemotron", "qwq", "o1", "o3", "reasoning"))
        max_out = 8192 if is_reasoning else 4096
        headroom = 2048 if is_reasoning else 0

    target_words: int | None = None
    if isinstance(answer_blueprint, AnswerBlueprint):
        target_words = answer_blueprint.target_word_count
    elif isinstance(answer_blueprint, dict):
        target_words = answer_blueprint.get("target_word_count")

    if target_words is None and query:
        wc_match = re.search(r"\b(?:in|under|around|within|max(?:imum)?)\s+(\d+)\s+words?\b", query.lower())
        if wc_match:
            with contextlib.suppress(ValueError):
                target_words = int(wc_match.group(1))

    if target_words and target_words > 0:
        needed_tokens = max(1024, int(target_words * 4))
        if is_reasoning:
            needed_tokens += headroom
        return max(1024, min(max_out, needed_tokens))

    if is_reasoning:
        return max_out

    if archetype in {"scalar_count", "conversational_meta_query", "archive_availability"}:
        return min(max_out, 1024)

    return max_out


class AnswerSynthesizer:
    """Synthesizes grounded narrative answers from retrieved evidence."""

    _render_comparison_matrix = staticmethod(render_comparison_matrix)
    _render_front_page_comparison = staticmethod(render_front_page_comparison)
    _render_broadsheet_perspectives = staticmethod(render_broadsheet_perspectives)
    _render_explore_further = staticmethod(render_explore_further)
    _generate_deterministic_summary = staticmethod(generate_deterministic_summary)
    _compile_structure_from_blueprint = staticmethod(compile_structure_from_blueprint)
    _make_citation = staticmethod(make_citation)
    clean_synthesized_answer = staticmethod(clean_synthesized_answer)
    resolve_token_budget = staticmethod(resolve_dynamic_token_budget)

    ARCHETYPE_TOKEN_CAPS: dict[str, int] = {
        "factual_lookup": 1024,
        "archive_availability": 768,
        "conversational_meta_query": 512,
        "scalar_count": 512,
        "article_deep_dive": 2048,
        "thematic_timeline": 3072,
        "cross_newspaper_comparison": 3072,
        "quantitative_trend": 2048,
        "article_catalog": 2048,
        "_default": 3072,
    }

    def __init__(
        self,
        provider: ChatModelProvider | None = None,
        session_factory: Any | None = None,
    ) -> None:
        self._provider = provider
        self._session_factory = session_factory

    def _has_valid_evidence(self, evidence_items: list[dict[str, Any]]) -> bool:
        return has_valid_evidence(evidence_items)

    def _get_provider(self, model_override: str | None = None) -> ChatModelProvider | None:
        if self._provider and not model_override:
            return self._provider
        try:
            return get_registry().get_chat_provider(model_override)
        except Exception as e:
            logger.warning(
                "Could not load LLM answerer provider, using deterministic fallback",
                extra={"error": str(e), "override": model_override},
            )
        return None

    def _get_provider_candidates(self, model_override: str | None = None) -> list[ChatModelProvider]:
        """Return an ordered list of viable chat model providers for failover resilience."""
        return resolve_provider_candidates(
            task_name=None,
            model_override=model_override,
            injected_provider=self._provider,
            prefer_local=False,
        )

    def _build_evidence_context(self, evidence_items: list[dict[str, Any]], query: str = "") -> str:
        return build_evidence_context(evidence_items, query=query)

    def _build_synthesizer_user_prompt(
        self,
        query: str,
        archetype: str,
        evidence_items: list[dict[str, Any]],
        context: str,
    ) -> str:
        return build_synthesizer_user_prompt(
            query=query, archetype=archetype, evidence_items=evidence_items, context=context,
        )

    async def resolve_authoritative_citations(
        self,
        citations: list[AgentCitation],
        target_page: int | None = None,
    ) -> list[AgentCitation]:
        return await resolve_authoritative_citations(
            citations=citations,
            session_factory=self._session_factory,
            target_page=target_page,
        )

    def extract_citations(
        self,
        text: str,
        evidence_items: list[dict[str, Any]],
        target_page: int | None = None,
        archetype: str | None = None,
    ) -> list[AgentCitation]:
        return extract_citations_from_text(
            text=text,
            evidence_items=evidence_items,
            target_page=target_page,
            archetype=archetype,
        )

    def _build_synthesizer_system_prompt(
        self,
        archetype: str,
        query: str,
        evidence_items: list[dict[str, Any]] | None = None,
        answer_blueprint: AnswerBlueprint | dict[str, Any] | None = None,
    ) -> str:
        return build_synthesizer_system_prompt(
            archetype=archetype,
            query=query,
            evidence_items=evidence_items,
            answer_blueprint=answer_blueprint,
        )

    def _prepare_messages(
        self,
        query: str,
        archetype: str,
        evidence_items: list[dict[str, Any]],
        chat_history: list[dict[str, Any]] | None = None,
        answer_blueprint: AnswerBlueprint | dict[str, Any] | None = None,
    ) -> list[Message]:
        """Build system, history, and user messages for both sync and streaming synthesis."""
        context = self._build_evidence_context(evidence_items, query=query)
        user_prompt = self._build_synthesizer_user_prompt(
            query=query, archetype=archetype, evidence_items=evidence_items, context=context,
        )
        system_prompt = self._build_synthesizer_system_prompt(
            archetype=archetype,
            query=query,
            evidence_items=evidence_items,
            answer_blueprint=answer_blueprint,
        )
        messages = [Message(role="system", content=system_prompt)]
        if chat_history:
            for turn in chat_history[-6:]:
                r = turn.get("role", "user")
                c = str(turn.get("content", "")).strip()
                if c:
                    messages.append(Message(role=r, content=c))
        messages.append(Message(role="user", content=user_prompt))
        return messages

    async def synthesize(
        self,
        query: str,
        archetype: str,
        evidence_items: list[dict[str, Any]],
        model_override: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        answer_blueprint: AnswerBlueprint | dict[str, Any] | None = None,
        target_page: int | None = None,
    ) -> tuple[str, list[AgentCitation], float]:
        """Synthesize answer with citations from evidence and conversation context with failover."""
        has_evidence = self._has_valid_evidence(evidence_items)
        is_conversational = archetype in {"conversational_meta_query", "clarification_needed"}

        if not has_evidence and not is_conversational:
            return EMPTY_EVIDENCE_RESPONSE, [], 0.0

        messages = self._prepare_messages(
            query=query,
            archetype=archetype,
            evidence_items=evidence_items,
            chat_history=chat_history,
            answer_blueprint=answer_blueprint,
        )

        cost_usd = 0.0
        for provider in self._get_provider_candidates(model_override=model_override):
            try:
                effective_max_tokens = resolve_dynamic_token_budget(
                    provider=provider,
                    archetype=archetype,
                    answer_blueprint=answer_blueprint,
                    query=query,
                )
                response = await provider.complete(
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=0.1,
                    thinking_budget=0,
                )
                th_trace, cleaned_answer = parse_thought_and_answer(response.text)
                answer_text = cleaned_answer if cleaned_answer else ("" if th_trace else response.text)
                p_name = getattr(provider, "provider_name", "llm")
                m_name = getattr(provider, "_model", getattr(provider, "model_name", "default"))
                calc_cost = record_usage_and_cost(
                    provider=p_name,
                    model=m_name,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
                cost_usd = max(calc_cost, response.cost_usd)
                if answer_text.strip():
                    answer_text = clean_synthesized_answer(answer_text, query, archetype, evidence_items)
                    answer_text, was_corrected, diag = verify_and_correct_answer_groundedness(
                        answer_text, query, evidence_items, archetype=archetype
                    )
                    if was_corrected:
                        logger.warning("Synthesizer fact-checker intercepted ungrounded answer", extra={"diagnosis": diag})
                    citations = self.extract_citations(answer_text, evidence_items, target_page=target_page, archetype=archetype)
                    citations = await self.resolve_authoritative_citations(citations, target_page=target_page)
                    return answer_text, citations, cost_usd
                logger.warning(
                    "Provider outputted reasoning without answer section, attempting failover candidate",
                    extra={"provider": p_name, "model": m_name},
                )
            except Exception as e:
                logger.warning(
                    "LLM synthesis failed on provider, trying next candidate",
                    extra={"provider": getattr(provider, "provider_name", ""), "error": str(e)},
                )

        answer_text = self._generate_deterministic_summary(query, evidence_items, archetype=archetype)
        answer_text = clean_synthesized_answer(answer_text, query, archetype, evidence_items)
        answer_text, was_corrected, diag = verify_and_correct_answer_groundedness(
            answer_text, query, evidence_items, archetype=archetype
        )
        if was_corrected:
            logger.warning("Synthesizer fact-checker intercepted ungrounded answer in deterministic summary", extra={"diagnosis": diag})
        citations = self.extract_citations(answer_text, evidence_items, target_page=target_page, archetype=archetype)
        citations = await self.resolve_authoritative_citations(citations, target_page=target_page)
        return answer_text, citations, cost_usd

    async def synthesize_stream(
        self,
        query: str,
        archetype: str,
        evidence_items: list[dict[str, Any]],
        model_override: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        answer_blueprint: AnswerBlueprint | dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream synthesized answer token by token with conversational context and failover."""
        has_evidence = self._has_valid_evidence(evidence_items)
        is_conversational = archetype in {"conversational_meta_query", "clarification_needed"}

        if not has_evidence and not is_conversational:
            yield EMPTY_EVIDENCE_RESPONSE
            return

        messages = self._prepare_messages(
            query=query,
            archetype=archetype,
            evidence_items=evidence_items,
            chat_history=chat_history,
            answer_blueprint=answer_blueprint,
        )

        for provider in self._get_provider_candidates(model_override=model_override):
            try:
                effective_max_tokens = resolve_dynamic_token_budget(
                    provider=provider,
                    archetype=archetype,
                    answer_blueprint=answer_blueprint,
                    query=query,
                )
                stream_gen = provider.complete_stream(
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=0.1,
                    thinking_budget=0,
                )
                streamed_any = False
                async for chunk in stream_gen:
                    streamed_any = True
                    yield chunk
                if streamed_any:
                    return
            except Exception as e:
                logger.warning(
                    "Streaming LLM synthesis failed on provider, trying next in failover",
                    extra={"provider": getattr(provider, "provider_name", ""), "error": str(e)},
                )

        summary = self._generate_deterministic_summary(query, evidence_items, archetype=archetype)
        summary = clean_synthesized_answer(summary, query, archetype, evidence_items)
        summary, _, _ = verify_and_correct_answer_groundedness(summary, query, evidence_items, archetype=archetype)
        for word in summary.split(" "):
            yield word + " "


__all__ = [
    "AnswerSynthesizer",
    "COMMON_ANALYTICAL_GUIDELINES",
    "COMMON_MEMORY_AND_CONSTRAINTS",
    "DEFAULT_CLOUD_FAILOVER",
    "DEFAULT_LOCAL_FAILOVER",
    "DOMAIN_TAXONOMY",
    "EMPTY_EVIDENCE_RESPONSE",
    "SYNTHESIZER_SYSTEM_PROMPT",
    "_DEFAULT_STRUCTURE",
    "clean_synthesized_answer",
    "compile_structure_from_blueprint",
    "deduplicate_repetitive_lines",
    "detect_domain_from_query",
    "is_domain_match",
    "parse_thought_and_answer",
    "resolve_dynamic_token_budget",
    "score_evidence_item",
    "verify_and_correct_answer_groundedness",
]
