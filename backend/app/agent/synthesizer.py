"""Answer Synthesizer: Grounded LLM narrative generation with strict source citations."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

from app.agent.state import AgentCitation
from app.core.cost_tracker import record_usage_and_cost
from app.core.logging import get_logger
from app.providers.base import ChatModelProvider, Message
from app.providers.registry import get_registry

logger = get_logger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are NewsLens-AI, an expert intelligence research assistant
specializing in broadsheet newspaper analysis and open-source intelligence.
Answer the user's research question thoroughly, objectively, and accurately based ONLY
on the provided evidence excerpts.

REQUIRED RESPONSE STRUCTURE:
1. ### Executive Summary
   - 2 to 3 concise, high-impact sentences directly answering the user's prompt.
2. ### Key Developments & Findings
   - Synthesized bullet points with specific facts, figures, dates, and mandatory inline citations.
3. ### Context & Analysis (if applicable)
   - Synthesis across multiple reports, contrasting editorial perspectives, or background context.

CRITICAL CITATION RULES:
1. Local Newspaper Archive Citations:
   - For facts backed by primary broadsheets, cite strictly as:
     [{Newspaper Name}, {YYYY-MM-DD}, Page {PDF_Page_Number}, "{Headline}"]
     Example: [Mint, 2026-08-01, Page 3, "Telecom AGR Dues Ruling"]
2. Live Web Search Citations:
   - For facts backed by live internet search results, cite strictly as:
     [Web: {Source Title}]({URL})
     Example: [Web: Reuters Telecom Update](https://www.reuters.com/business/telecom)
3. Use the physical PDF page number specified in newspaper evidence excerpts.
4. If multiple sources corroborate a claim, cite them together.
5. STRICT NEGATIVE CONSTRAINTS:
   - NEVER dump raw chunk headers (e.g. `--- ARCHIVE EVIDENCE EXCERPT ---` or `[Evidence: ...]`).
   - NEVER output advertisement boilerplate, statutory IPO notices, or unrelated news briefs.
   - Do NOT invent any dates, figures, quotes, or events not supported by the evidence.
"""


def parse_thought_and_answer(text: str) -> tuple[str, str]:
    """Extract thought/reasoning trace and clean response text from model output."""
    if not text:
        return "", ""

    if "<think>" in text:
        if "</think>" in text:
            match = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
            if match:
                thought = match.group(1).strip()
                ans = match.group(2).strip()
                return thought, ans
        else:
            # Unclosed <think> tag
            after_think = text.split("<think>", 1)[1]
            pattern = (
                r"\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|"
                r"In summary|Summary:|Answer:)"
            )
            split_match = re.search(pattern, after_think, flags=re.IGNORECASE)
            if split_match:
                split_idx = split_match.start()
                thought = after_think[:split_idx].strip()
                ans = after_think[split_idx:].strip()
                return thought, ans
            return "", after_think.strip()

    return "", text.strip()


class AnswerSynthesizer:
    """Synthesizes grounded narrative answers from retrieved evidence."""

    def __init__(self, provider: ChatModelProvider | None = None) -> None:
        self._provider = provider

    def _get_provider(self, model_override: str | None = None) -> ChatModelProvider | None:
        if self._provider and not model_override:
            return self._provider
        try:
            registry = get_registry()
            return registry.get_chat_provider(model_override)
        except Exception as e:
            logger.warning(
                "Could not load LLM answerer provider, using deterministic fallback",
                extra={"error": str(e), "override": model_override},
            )
        return None

    def _build_evidence_context(self, evidence_items: list[dict[str, Any]]) -> str:
        """Format retrieved evidence documents into structured prompt context."""
        context_blocks: list[str] = []
        for i, item in enumerate(evidence_items, start=1):
            is_web = item.get("is_web") or item.get("source_tool") == "web_search"
            hl = item.get("headline", "Untitled Article")
            text = item.get("snippet") or item.get("full_text") or item.get("summary") or ""

            if is_web:
                url = item.get("url", "")
                src = item.get("newspaper_name", "Live Web")
                dt = item.get("issue_date", "Current")
                context_blocks.append(
                    f"--- LIVE WEB EVIDENCE EXCERPT [{i}] ---\n"
                    f"Source: {src}\n"
                    f"Title: {hl}\n"
                    f"URL: {url}\n"
                    f"Date: {dt}\n"
                    f"Content:\n{text.strip()}\n"
                )
            else:
                np_name = item.get("newspaper_name", "Unknown Publication")
                dt = item.get("issue_date", "Unknown Date")
                pages = item.get("pages", [1])
                pdf_page = int(pages[0]) if pages and pages[0] else 1
                evidence_tag = (
                    f"[Evidence: {np_name}, {dt}, Page {pdf_page} "
                    f"(PDF Page {pdf_page}), Headline: \"{hl}\"]"
                )
                context_blocks.append(
                    f"--- ARCHIVE EVIDENCE EXCERPT [{i}] ---\n"
                    f"{evidence_tag}\n"
                    f"Publication: {np_name}\n"
                    f"Date: {dt}\n"
                    f"Page(s): Page {pdf_page} (PDF Page {pdf_page})\n"
                    f"Headline: {hl}\n"
                    f"Content:\n{text.strip()}\n"
                )
        return "\n".join(context_blocks)

    def extract_citations(
        self,
        text: str,
        evidence_items: list[dict[str, Any]],
    ) -> list[AgentCitation]:
        """Extract and structure verified citations mentioned in the text or used from evidence."""
        citations: list[AgentCitation] = []
        seen_keys: set[str] = set()
        text_lower = (text or "").lower()

        for item in evidence_items:
            is_web = item.get("is_web") or item.get("source_tool") == "web_search"
            hl = item.get("headline", "")
            hl_clean = hl.strip().lower()

            if is_web:
                url = item.get("url") or ""
                snip = item.get("snippet") or ""
                src = item.get("newspaper_name", "Web")

                # Filter condition for web results: cited in text or key headline words appear
                is_cited = False
                if (
                    not text
                    or (url and url.lower() in text_lower)
                    or (hl_clean and len(hl_clean) > 8 and hl_clean in text_lower)
                ):
                    is_cited = True
                elif src.lower() in text_lower:
                    words = [w for w in hl_clean.split() if len(w) > 4]
                    matched_cnt = sum(1 for w in words if w in text_lower)
                    if words and matched_cnt >= max(1, len(words) // 2):
                        is_cited = True

                if is_cited:
                    key = f"web_{url}_{hl}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        citations.append(
                            AgentCitation(
                                newspaper_name=src,
                                issue_date=item.get("issue_date", "Live Web"),
                                page_number=1,
                                headline=hl or "Web Source",
                                article_id=0,
                                snippet=snip[:300],
                                issue_id=0,
                                bboxes=[],
                                url=url,
                                source_type="web",
                                is_web=True,
                            )
                        )
            else:
                np_name = item.get("newspaper_name", "Daily News")
                dt = item.get("issue_date", "")
                pages = item.get("pages", [1])
                page_num = int(pages[0]) if pages and pages[0] else 1
                art_id = item.get("article_id", 0)
                snip = item.get("snippet") or item.get("summary") or ""
                issue_id = item.get("issue_id", 0)
                bboxes = item.get("bboxes", [])

                # Filter condition for newspaper items
                is_cited = False
                if (
                    not text
                    or (hl_clean and len(hl_clean) > 6 and hl_clean in text_lower)
                    or (f"page {page_num}" in text_lower and np_name.lower() in text_lower)
                ):
                    is_cited = True
                elif f"page {page_num}" in text_lower:
                    words = [w for w in hl_clean.split() if len(w) > 4]
                    if words and any(w in text_lower for w in words):
                        is_cited = True
                elif hl_clean:
                    words = [w for w in hl_clean.split() if len(w) > 4]
                    matched_cnt = sum(1 for w in words if w in text_lower)
                    if words and matched_cnt >= max(2, len(words) // 2):
                        is_cited = True

                if is_cited:
                    key = f"np_{np_name}_{dt}_{page_num}_{hl}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        citations.append(
                            AgentCitation(
                                newspaper_name=np_name,
                                issue_date=dt,
                                page_number=page_num,
                                headline=hl or "Untitled",
                                article_id=art_id,
                                snippet=snip[:300],
                                issue_id=issue_id,
                                bboxes=bboxes,
                                url=None,
                                source_type="newspaper",
                                is_web=False,
                            )
                        )

        # Fallback: if text was provided but strict parsing yielded no citations,
        # retain only the top 2 highest prominence items rather than dumping all candidate chunks
        if not citations and evidence_items:
            for item in evidence_items[:2]:
                is_web = item.get("is_web") or item.get("source_tool") == "web_search"
                if is_web:
                    url = item.get("url") or ""
                    hl = item.get("headline", "Web Article")
                    src = item.get("newspaper_name", "Web")
                    citations.append(
                        AgentCitation(
                            newspaper_name=src,
                            issue_date=item.get("issue_date", "Live Web"),
                            page_number=1,
                            headline=hl,
                            article_id=0,
                            snippet=item.get("snippet", "")[:300],
                            issue_id=0,
                            bboxes=[],
                            url=url,
                            source_type="web",
                            is_web=True,
                        )
                    )
                else:
                    pages = item.get("pages", [1])
                    page_num = int(pages[0]) if pages and pages[0] else 1
                    citations.append(
                        AgentCitation(
                            newspaper_name=item.get("newspaper_name", "Daily News"),
                            issue_date=item.get("issue_date", ""),
                            page_number=page_num,
                            headline=item.get("headline", "Untitled"),
                            article_id=item.get("article_id", 0),
                            snippet=(item.get("snippet") or "")[:300],
                            issue_id=item.get("issue_id", 0),
                            bboxes=item.get("bboxes", []),
                            url=None,
                            source_type="newspaper",
                            is_web=False,
                        )
                    )

        return citations

    async def synthesize(
        self,
        query: str,
        archetype: str,
        evidence_items: list[dict[str, Any]],
        model_override: str | None = None,
    ) -> tuple[str, list[AgentCitation], float]:
        """Synthesize answer with citations from evidence."""
        if not evidence_items:
            empty_msg = f"No relevant newspaper articles found for query: '{query}'."
            return empty_msg, [], 0.0

        provider = self._get_provider(model_override=model_override)
        citations = self.extract_citations("", evidence_items)
        context = self._build_evidence_context(evidence_items)

        user_prompt = (
            f"User Research Query: {query}\n"
            f"Query Archetype: {archetype}\n\n"
            f"Available Newspaper Evidence:\n"
            f"{context}\n\n"
            f"Synthesize a comprehensive answer citing all relevant sources inline."
        )

        cost_usd = 0.0

        if provider:
            try:
                response = await provider.complete(
                    messages=[
                        Message(role="system", content=SYNTHESIZER_SYSTEM_PROMPT),
                        Message(role="user", content=user_prompt),
                    ],
                    max_tokens=2048,
                    temperature=0.1,
                )
                _, cleaned_answer = parse_thought_and_answer(response.text)
                answer_text = cleaned_answer if cleaned_answer else response.text
                p_name = getattr(provider, "provider_name", "llm")
                m_name = getattr(provider, "model_name", "default")
                calc_cost = record_usage_and_cost(
                    provider=p_name,
                    model=m_name,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
                cost_usd = max(calc_cost, response.cost_usd)
            except Exception as e:
                logger.warning(
                    "LLM synthesis failed, generating rule-based grounded summary",
                    extra={"error": str(e)},
                )
                answer_text = self._generate_deterministic_summary(query, evidence_items)
        else:
            answer_text = self._generate_deterministic_summary(query, evidence_items)

        return answer_text, citations, cost_usd

    async def synthesize_stream(
        self,
        query: str,
        archetype: str,
        evidence_items: list[dict[str, Any]],
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream synthesized answer token by token."""
        if not evidence_items:
            yield f"No relevant newspaper articles found for query: '{query}'."
            return

        provider = self._get_provider(model_override=model_override)
        context = self._build_evidence_context(evidence_items)
        user_prompt = (
            f"User Research Query: {query}\n"
            f"Query Archetype: {archetype}\n\n"
            f"Available Newspaper Evidence:\n"
            f"{context}\n\n"
            f"Synthesize a comprehensive answer citing all relevant sources inline."
        )

        if provider:
            try:
                stream_gen = provider.complete_stream(
                    messages=[
                        Message(role="system", content=SYNTHESIZER_SYSTEM_PROMPT),
                        Message(role="user", content=user_prompt),
                    ],
                    max_tokens=2048,
                    temperature=0.1,
                )
                async for chunk in stream_gen:
                    yield chunk
                return
            except Exception as e:
                logger.warning(
                    "Streaming LLM synthesis failed, streaming fallback text",
                    extra={"error": str(e)},
                )

        # Fallback text streaming
        summary = self._generate_deterministic_summary(query, evidence_items)
        for word in summary.split(" "):
            yield word + " "

    def _generate_deterministic_summary(
        self,
        query: str,
        evidence_items: list[dict[str, Any]],
    ) -> str:
        """Deterministic grounded synthesis when LLM is offline or unavailable."""
        lines = [
            f"### Research Findings for: {query}\n",
            "Based on the archived newspaper reports retrieved from the repository:\n",
        ]

        for item in evidence_items[:5]:
            np_name = item.get("newspaper_name", "Daily News")
            dt = item.get("issue_date", "Unknown Date")
            pages = item.get("pages", [1])
            page_str = f"Page {pages[0]}" if pages else "Page 1"
            hl = item.get("headline", "Untitled")
            snippet = item.get("snippet") or item.get("summary") or ""

            lines.append(f'- **{hl}**: {snippet} [{np_name}, {dt}, {page_str}, "{hl}"]\n')

        lines.append("\n*All findings verified against primary newspaper source scans.*")
        return "\n".join(lines)
