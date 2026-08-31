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

SYNTHESIZER_SYSTEM_PROMPT = """You are NewsLens-AI, an elite broadsheet intelligence assistant.
Your goal is to analyze, explain, and synthesize coverage into a structured, highly readable brief.

CRITICAL ANALYTICAL GUIDELINES:
1. DEEP EXPLANATION & CONTEXT:
   - Do NOT just copy raw snippets or state vague headlines.
   - Deeply analyze the news story: what happened, why it happened, and what it means.
   - Adapt to the domain of the query:
     * For Economic/Market news: Extract indices, FPI/FII flows, sectors, macro triggers.
     * For Geopolitical/Policy news: Extract agreements, policies, actors, implications.
2. NOISE & RELEVANCE FILTERING:
   - Discard irrelevant candidate excerpts (such as classified ads, unrelated briefs,
     or layout noise) that do not match the research topic. Focus strictly on the query subject.
3. ANTI-HALLUCINATION & STRICT GROUNDING:
   - ABSOLUTE PROHIBITION: You must NEVER invent, fabricate, or assume any dates,
     newspaper names, entities, or facts not explicitly present in the provided evidence.
   - Ground all dates, metrics, and metadata strictly in the provided broadsheet source tags.
   - NEVER generate introductory memo headers containing arbitrary dates (such as "Date: October 26, 2023 (Current Analysis)").
     The ONLY dates permitted in your response are the publication dates explicitly stated in the evidence chunks.
   - Do NOT introduce external companies, institutions, or individuals unless they
     directly appear in the retrieved evidence chunks corresponding to the user's topic.
   - If the provided excerpts do not contain verified evidence to answer the query, explicitly state:
     "The uploaded broadsheet archives do not contain verified reporting on this topic."
   - PUBLICATION & ISSUE FIDELITY: If the user explicitly asks for a specific publication (e.g. 'The Economic Times' or 'Mint')
     or issue number, and the evidence reports that the issue was not found or contains a different publication:
     * Explicitly inform the user that the requested publication/issue is not in the archive.
     * NEVER pretend that another newspaper's articles belong to the requested publication.
   - Do NOT synthesize from ungrounded pre-training memory; rely solely on the verified evidence.

REQUIRED RESPONSE STRUCTURE:
1. ### ⚡ Executive Summary
   - 1 to 2 crisp, authoritative sentences explaining the main development and takeaway.

2. ### 📌 Key Verified Facts & Highlights
   - Bullet points of specific numbers, figures, dates, and quotes.
   - STRICT CITATION RULE: Every bullet point MUST end with an inline citation.
     * For Local Broadsheets: [{Newspaper Name}, {YYYY-MM-DD}, Page {PDF_Page}, "{Headline}"]
     * For Charts & Infographics: [📊 Chart: {Newspaper Name}, {YYYY-MM-DD}, Page {PDF_Page}, "{Headline}"]
     * For Live Web Search (if provided): [Web: {Source Title}]({URL})

3. ### 📰 Broadsheet Perspectives & Focus Areas
   - For Cross-Newspaper Comparisons: Clearly delineate coverage differences by publication:
     * **{Publication A} Focus**: Specific angles, tone, numbers emphasized.
     * **{Publication B} Focus**: Contrasting viewpoints, unique quotes, counter-arguments.
   - Group reporting by publication (e.g. **Mint**, **Business Standard**, **The Hindu**).
   - 1 concise bullet point per paper on that paper's specific angle, bias, or unique data.

4. ### 🔍 Explore Further
   - 2 to 3 concise follow-up prompts formatted strictly as:
     > 💡 Explore: <Specific follow-up question or angle>
     (Example: > 💡 Explore: What was Mint's detailed financial breakdown?)

CONVERSATIONAL & CITATION MEMORY RULES:
1. If the user asks about previous messages, dates, newspapers, citations, or metadata
   (e.g. "which newspaper was this from?", "what was the date?"), directly and concisely
   answer using the conversation history and cited sources.
2. STRICT NEGATIVE CONSTRAINTS:
   - NEVER dump raw headers (e.g. `--- ARCHIVE EVIDENCE EXCERPT ---` or `[Evidence: ...]`).
   - NEVER output advertisement boilerplate, legal notices, or unrelated book/movie reviews.
"""


def parse_thought_and_answer(text: str) -> tuple[str, str]:
    """Extract thought/reasoning trace and clean response text from model output."""
    if not text:
        return "", ""

    # 1. Standard <think>...</think> tags
    if "<think>" in text:
        if "</think>" in text:
            match = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
            if match:
                thought = match.group(1).strip()
                ans = match.group(2).strip()
                if ans:
                    return thought, ans
                # If ans is empty but thought contains structured sections or draft
                pattern = (
                    r"\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|"
                    r"In summary|Summary:|Answer:|Draft:|Executive Summary)"
                )
                split_match = re.search(pattern, thought, flags=re.IGNORECASE)
                if split_match:
                    s_idx = split_match.start()
                    return thought[:s_idx].strip(), thought[s_idx:].strip()
                return "", thought
        else:
            # Unclosed <think> tag
            after_think = text.split("<think>", 1)[1]
            pattern = (
                r"\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|"
                r"In summary|Summary:|Answer:|Draft:|Executive Summary)"
            )
            split_match = re.search(pattern, after_think, flags=re.IGNORECASE)
            if split_match:
                split_idx = split_match.start()
                thought = after_think[:split_idx].strip()
                ans = after_think[split_idx:].strip()
                return thought, ans
            return "", after_think.strip()

    # 2. Heuristic for reasoning prefixes (e.g. "Thinking Process:" or "Here's a thinking process:")
    reasoning_prefix_match = re.match(
        r"^(?:Here'?s a thinking process:?|Thinking Process:?|Thought:?)\s*",
        text,
        flags=re.IGNORECASE,
    )
    ans_text = text.strip()
    if reasoning_prefix_match:
        pattern = (
            r"\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|"
            r"In summary|Summary:|Answer:|Draft:|Executive Summary)"
        )
        split_match = re.search(pattern, text, flags=re.IGNORECASE)
        if split_match:
            s_idx = split_match.start()
            ans_text = text[s_idx:].strip()
            return text[:s_idx].strip(), ans_text

    # Post-clean: Strip hallucinated memo headers with arbitrary pre-training dates like "Date: October 26, 2023 (Current Analysis)"
    ans_text = re.sub(
        r"(?i)^(?:\*{0,2}EXECUTIVE\s+INTELLIGENCE\s+BRIEFING\*{0,2}\s*\n+)?Date:\s*[A-Za-z]+\s+\d{1,2},?\s+20\d{2}\s*\([^\)]*Current Analysis[^\)]*\)\s*Subject:[^\n]+\n*",
        "",
        ans_text,
    ).strip()

    return "", ans_text

EMPTY_EVIDENCE_RESPONSE = (
    "I could not find any evidence or articles matching this query in the database. "
    "Please try adjusting your search terms."
)


class AnswerSynthesizer:
    """Synthesizes grounded narrative answers from retrieved evidence."""

    def __init__(self, provider: ChatModelProvider | None = None) -> None:
        self._provider = provider

    def _has_valid_evidence(self, evidence_items: list[dict[str, Any]]) -> bool:
        """Check whether evidence contains non-empty grounded content."""
        if not evidence_items:
            return False
        for item in evidence_items:
            snip = item.get("snippet") or item.get("full_text") or item.get("summary") or ""
            hl = item.get("headline") or ""
            if len(snip.strip()) >= 5 or len(hl.strip()) >= 5:
                return True
        return False

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

    def _get_provider_candidates(
        self, model_override: str | None = None
    ) -> list[ChatModelProvider]:
        """Return an ordered list of viable chat model providers for failover resilience."""
        if self._provider and not model_override:
            return [self._provider]

        candidates: list[ChatModelProvider] = []
        seen_keys: set[str] = set()

        # 1. Primary requested provider
        primary = self._get_provider(model_override=model_override)
        if primary:
            candidates.append(primary)
            p_ident = f"{getattr(primary, 'provider_name', '')}:{getattr(primary, '_model', '')}"
            seen_keys.add(p_ident)

        # 2. Resilient failover sequence from active registry
        failover_keys = [
            "ollama_gemma4_12b",
            "ollama_gemma4_26b",
            "groq_compound",
            "gemini_flash",
            "groq_qwen",
            "ollama_llama3",
            "ollama_deepseek",
        ]
        try:
            registry = get_registry()
            for key in failover_keys:
                try:
                    p = registry.get_chat_provider(key)
                    if p:
                        ident = f"{getattr(p, 'provider_name', '')}:{getattr(p, '_model', '')}"
                        if ident not in seen_keys:
                            seen_keys.add(ident)
                            candidates.append(p)
                except Exception:
                    continue
        except Exception:
            pass

        return candidates

    def _build_evidence_context(self, evidence_items: list[dict[str, Any]]) -> str:
        """Format retrieved evidence documents into structured prompt context with strict token budgeting."""
        context_blocks: list[str] = []
        # Cap at top 12 items to prevent prompt explosion (guaranteeing <= 3500 tokens)
        budgeted_items = evidence_items[:12] if evidence_items else []
        seen_keys: set[str] = set()

        for item in budgeted_items:
            is_web = item.get("is_web") or item.get("source_tool") == "web_search"
            hl = item.get("headline", "Untitled Article")
            text = (item.get("snippet") or item.get("full_text") or item.get("summary") or "").strip()

            # Deduplication key across items
            dedup_key = f"{hl.lower().strip()}_{text[:80].lower().strip()}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Truncate overly long text snippets to 1,200 chars to protect context size
            if len(text) > 1200:
                text = text[:1200].rstrip() + " ... [excerpt truncated for length]"

            idx = len(context_blocks) + 1
            if is_web:
                url = item.get("url", "")
                src = item.get("newspaper_name", "Live Web")
                dt = item.get("issue_date", "Current")
                context_blocks.append(
                    f"--- LIVE WEB EVIDENCE EXCERPT [{idx}] ---\n"
                    f"Source: {src}\n"
                    f"Title: {hl}\n"
                    f"URL: {url}\n"
                    f"Date: {dt}\n"
                    f"Content:\n{text}\n"
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
                    f"--- ARCHIVE EVIDENCE EXCERPT [{idx}] ---\n"
                    f"{evidence_tag}\n"
                    f"Publication: {np_name}\n"
                    f"Date: {dt}\n"
                    f"Page(s): Page {pdf_page} (PDF Page {pdf_page})\n"
                    f"Headline: {hl}\n"
                    f"Content:\n{text}\n"
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
            url = item.get("url") or ""

            is_referenced = False
            if is_web:
                if (url and url.lower() in text_lower) or (
                    hl_clean and len(hl_clean) > 4 and hl_clean in text_lower
                ):
                    is_referenced = True
            else:
                np_name = str(item.get("newspaper_name", "")).lower()
                pages = item.get("pages", [1])
                page_num = int(pages[0]) if pages and pages[0] else 1
                if (hl_clean and len(hl_clean) > 5 and hl_clean in text_lower) or (
                    f"page {page_num}" in text_lower and np_name in text_lower
                ):
                    is_referenced = True

            if is_referenced:
                dedup_key = f"{item.get('newspaper_name')}_{item.get('issue_date')}_{hl}"
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    if is_web:
                        citations.append(
                            AgentCitation(
                                newspaper_name=item.get("newspaper_name", "Live Web"),
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
                        page_digits = item.get("pages", [1])
                        page_val = int(page_digits[0]) if page_digits else 1
                        citations.append(
                            AgentCitation(
                                newspaper_name=item.get("newspaper_name", "Daily News"),
                                issue_date=item.get("issue_date", ""),
                                page_number=page_val,
                                headline=hl,
                                article_id=item.get("article_id", 0),
                                snippet=(item.get("snippet") or "")[:300],
                                issue_id=item.get("issue_id", 0),
                                bboxes=item.get("bboxes", []),
                                url=None,
                                source_type="newspaper",
                                is_web=False,
                            )
                        )

        # Fallback if no specific inline references matched
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
        chat_history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[AgentCitation], float]:
        """Synthesize answer with citations from evidence and conversation context with failover."""
        has_evidence = self._has_valid_evidence(evidence_items)
        has_meta_history = bool(chat_history and archetype == "conversational_meta_query")

        # Strict Empty Evidence Hard-Stop
        if not has_evidence and not has_meta_history:
            return EMPTY_EVIDENCE_RESPONSE, [], 0.0

        citations = self.extract_citations("", evidence_items)
        context = self._build_evidence_context(evidence_items)

        user_prompt = (
            f"User Research Query: {query}\n"
            f"Query Archetype: {archetype}\n\n"
            f"Available Newspaper Evidence:\n"
            f"{context or 'No new search results—refer to conversation history if applicable.'}\n\n"
            f"Synthesize an insightful, highly-structured executive intelligence response."
        )

        messages = [Message(role="system", content=SYNTHESIZER_SYSTEM_PROMPT)]
        if chat_history:
            for turn in chat_history[-6:]:
                r = turn.get("role", "user")
                c = str(turn.get("content", "")).strip()
                if c:
                    messages.append(Message(role=r, content=c))
        messages.append(Message(role="user", content=user_prompt))

        cost_usd = 0.0
        answer_text = ""
        providers = self._get_provider_candidates(model_override=model_override)

        for provider in providers:
            try:
                response = await provider.complete(
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.1,
                )
                _, cleaned_answer = parse_thought_and_answer(response.text)
                answer_text = cleaned_answer if cleaned_answer else response.text
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
                    return answer_text, citations, cost_usd
            except Exception as e:
                logger.warning(
                    "LLM synthesis failed on provider, trying next candidate",
                    extra={"provider": getattr(provider, "provider_name", ""), "error": str(e)},
                )

        # Fallback if all providers fail
        answer_text = self._generate_deterministic_summary(query, evidence_items)
        return answer_text, citations, cost_usd

    async def synthesize_stream(
        self,
        query: str,
        archetype: str,
        evidence_items: list[dict[str, Any]],
        model_override: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream synthesized answer token by token with conversational context and failover."""
        has_evidence = self._has_valid_evidence(evidence_items)
        has_meta_history = bool(chat_history and archetype == "conversational_meta_query")

        # Strict Empty Evidence Hard-Stop
        if not has_evidence and not has_meta_history:
            yield EMPTY_EVIDENCE_RESPONSE
            return

        context = self._build_evidence_context(evidence_items)
        user_prompt = (
            f"User Research Query: {query}\n"
            f"Query Archetype: {archetype}\n\n"
            f"Available Newspaper Evidence:\n"
            f"{context or 'No new search results—refer to conversation history if applicable.'}\n\n"
            f"Synthesize an insightful, highly-structured executive intelligence response."
        )

        messages = [Message(role="system", content=SYNTHESIZER_SYSTEM_PROMPT)]
        if chat_history:
            for turn in chat_history[-6:]:
                r = turn.get("role", "user")
                c = str(turn.get("content", "")).strip()
                if c:
                    messages.append(Message(role=r, content=c))
        messages.append(Message(role="user", content=user_prompt))

        providers = self._get_provider_candidates(model_override=model_override)

        for provider in providers:
            try:
                stream_gen = provider.complete_stream(
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.1,
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

        # Fallback text streaming
        summary = self._generate_deterministic_summary(query, evidence_items)
        for word in summary.split(" "):
            yield word + " "

    def _generate_deterministic_summary(
        self,
        query: str,
        evidence_items: list[dict[str, Any]],
    ) -> str:
        """Structured deterministic grounded synthesis when all LLMs are offline."""
        if not self._has_valid_evidence(evidence_items):
            return EMPTY_EVIDENCE_RESPONSE

        # Filter out obvious noise items if other items have keyword overlap
        query_words = {w.lower() for w in query.split() if len(w) > 3}
        relevant_items = []
        for item in evidence_items:
            text_corpus = (
                item.get("headline", "")
                + " "
                + item.get("snippet", "")
                + " "
                + item.get("summary", "")
            ).lower()
            if not query_words or any(qw in text_corpus for qw in query_words):
                relevant_items.append(item)

        filtered_evidence = relevant_items if relevant_items else evidence_items
        first = filtered_evidence[0]
        first_np = first.get("newspaper_name", "Daily News")
        first_dt = first.get("issue_date", "")

        lines = [
            "### ⚡ Executive Summary",
            (
                f"Key broadsheet reporting regarding **{query}** was documented across "
                f"regional archives, led by *{first_np}* ({first_dt}).\n"
            ),
            "### 📌 Key Verified Facts & Highlights",
        ]

        # Group by publication
        pub_groups: dict[str, list[dict[str, Any]]] = {}
        for item in filtered_evidence[:6]:
            np_name = item.get("newspaper_name", "Archive")
            pub_groups.setdefault(np_name, []).append(item)
            dt = item.get("issue_date", "")
            pages = item.get("pages", [1])
            page_str = f"Page {pages[0]}" if pages else "Page 1"
            hl = item.get("headline", "Untitled")
            snip = item.get("snippet") or item.get("summary") or ""
            # Clean indexing metadata tags
            clean_snip = re.sub(r"\[Newspaper:.*?\]", "", snip).strip()
            clean_snip = re.sub(r"\[Page\(s\):.*?\]", "", clean_snip).strip()
            lines.append(
                f'- **{hl}**: {clean_snip[:180]}... [{np_name}, {dt}, {page_str}, "{hl}"]'
            )

        lines.append("\n### 📰 Broadsheet Perspectives")
        for pub, items in pub_groups.items():
            top_hl = items[0].get("headline", "Reporting")
            lines.append(
                f"- **{pub}**: Emphasized '{top_hl}' across {len(items)} related report(s)."
            )

        lines.append("\n### 🔍 Explore Further")
        for pub in list(pub_groups.keys())[:2]:
            lines.append(f"> 💡 Explore: What was {pub}'s detailed coverage on this topic?")

        return "\n".join(lines)
