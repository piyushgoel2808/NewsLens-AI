"""Answer Synthesizer: Grounded LLM narrative generation with strict source citations."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

from app.agent.fallback_presenter import (
    EMPTY_EVIDENCE_RESPONSE,
    generate_deterministic_summary,
    has_valid_evidence,
    render_broadsheet_perspectives,
    render_comparison_matrix,
    render_explore_further,
    render_front_page_comparison,
)
from app.agent.prompt_context import (
    build_evidence_context,
    build_synthesizer_user_prompt,
    clean_snippet,
)
from app.agent.state import AgentCitation
from app.agent.taxonomy import (
    DOMAIN_TAXONOMY,
    detect_domain_from_query,
    is_domain_match,
    score_evidence_item,
)
from app.core.cost_tracker import record_usage_and_cost
from app.core.logging import get_logger
from app.providers.base import ChatModelProvider, Message
from app.providers.registry import get_registry
from app.retrieval.sql_analytics import sanitize_headline

logger = get_logger(__name__)

# Re-exports and backward-compatibility aliases
_clean_snippet = clean_snippet
_detect_domain_from_query = detect_domain_from_query
_is_domain_match = is_domain_match

# Provider failover sequences
DEFAULT_CLOUD_FAILOVER: tuple[str, ...] = (
    "nvidia_nemotron",
    "openrouter_nemotron",
    "openrouter_gemma4_26b",
    "gemini_flash",
    "groq_compound",
    "openai_gpt4o_mini",
    "groq_qwen",
    "ollama_llama3",
    "ollama_deepseek",
)

DEFAULT_LOCAL_FAILOVER: tuple[str, ...] = (
    "nvidia_nemotron",
    "ollama_llama3",
    "ollama_deepseek",
    "openrouter_nemotron",
    "openrouter_gemma4_26b",
    "gemini_flash",
    "groq_compound",
)


def parse_thought_and_answer(text: str) -> tuple[str, str]:
    """Extract thought/reasoning trace and clean response text from model output."""
    if not text:
        return "", ""

    split_pattern = (
        r"\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|"
        r"In summary|Summary:|Answer:|Draft:\s*\n|Executive Summary)"
    )

    # 1. Standard <think>...</think> tags
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

    # 2. Heuristic for reasoning prefixes
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

    # Post-clean: Strip hallucinated memo headers with arbitrary dates
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


# ---------------------------------------------------------------------------
# Modular System Prompt Sections
# ---------------------------------------------------------------------------

COMMON_ANALYTICAL_GUIDELINES = """CRITICAL ANALYTICAL GUIDELINES:
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
4. CONVERSATION HISTORY & PUBLICATION ISOLATION:
   - The conversation history is ONLY for understanding conversational context and follow-up intent.
   - Substantive facts, headlines, and analysis must be derived EXCLUSIVELY from the "Available Newspaper Evidence" for the CURRENT turn.
   - NEVER carry over news stories, events, or newspaper titles from previous conversation turns into a new date-specific or comparative query.
   - If the user asks to compare newspapers on a specific date, you must ONLY report on the publications explicitly present in the verified evidence for that date.
5. REASONING EFFICIENCY CONSTRAINT:
   - If you are a reasoning model, keep internal chain-of-thought concise (<150 words) focusing strictly on verifying facts against evidence.
   - Do NOT output repetitive drafts during reasoning. Immediately proceed to output the structured brief in the required markdown format."""

COMMON_MEMORY_AND_CONSTRAINTS = """CONVERSATIONAL & CITATION MEMORY RULES:
1. If the user asks about previous messages, dates, newspapers, citations, or metadata
   (e.g. "which newspaper was this from?", "what was the date?"), directly and concisely
   answer using the conversation history and cited sources.
2. VISUAL ELEMENTS, PHOTOS & INFOGRAPHICS:
   - When the user asks whether an article has a photo, image, graphic, chart, or infographic, ALWAYS inspect the "Attached Photos & Visual Elements" listed under that article in the evidence.
   - If visual elements are present, explicitly state YES, describe what the image/figure portrays using its caption and visual scene description, and cite the article.
   - If no attached visual elements are listed in the evidence for that article, state that none are attached.
3. STRICT NEGATIVE CONSTRAINTS:
   - NEVER dump raw headers (e.g. `--- ARCHIVE EVIDENCE EXCERPT ---` or `[Evidence: ...]`).
   - NEVER output advertisement boilerplate, legal notices, or unrelated book/movie reviews.
   - DIRECT RESPONSE & NO-SCRATCHPAD MANDATE:
     * CRITICAL: Do NOT output internal scratchpad notes, planning thoughts, chain-of-thought analysis, or preamble such as "Here's a thinking process:".
     * You MUST begin your response IMMEDIATELY with the first section header: "### ⚡ Executive Summary".
   - NATURAL EDITORIAL SYNTHESIS & NO QUERY ECHOING:
     * NEVER copy, quote, or concatenate the raw user query string into the Executive Summary or headers (e.g. do NOT write "Key broadsheet reporting regarding [USER QUERY] was documented...").
     * State directly and journalistically what developments transpired in the verified news reporting."""

_DEFAULT_STRUCTURE = """REQUIRED RESPONSE STRUCTURE:
1. ### ⚡ Executive Summary
   - 1 to 2 crisp, authoritative sentences explaining the main development and takeaway.

2. ### 📌 Key Verified Facts & Highlights
   - Bullet points of specific numbers, figures, dates, and quotes.
   - STRICT CITATION RULE: Every bullet point MUST end with an inline citation:
     [{Newspaper Name}, {YYYY-MM-DD}, Page {Page_Number}, "{Headline}"]
     * For Charts & Infographics: [📊 Chart: {Newspaper Name}, {YYYY-MM-DD}, Page {Page_Number}, "{Headline}"]
     * For Live Web Search (if provided): [Web: {Source Title}]({URL})
   - SINGLE PAGE FORMAT MANDATE:
     * Use the single standard page number (e.g. Page 1, Page 5). NEVER output dual-page or folio notation such as "Page X (PDF p.Y)".

3. ### 📰 Broadsheet Perspectives & Focus Areas
   - Group reporting by publication (e.g. **Mint**, **Business Standard**, **The Hindu**).
   - 1 concise bullet point per paper on that paper's specific angle, bias, or unique data.

4. ### 🔍 Explore Further
   - 2 to 3 concise follow-up prompts formatted strictly as:
     > 💡 Explore: <Specific follow-up question or angle>

ANTI-REPETITION CONSTRAINT:
- Do NOT repeat the same sentences, statistics, or phrasing across the Executive Summary and Key Highlights. Each section must provide distinct, complementary value."""

SYNTHESIZER_SYSTEM_PROMPT = (
    f"You are NewsLens-AI, an elite broadsheet intelligence assistant.\n"
    f"Your goal is to analyze, explain, and synthesize coverage into a structured, highly readable brief.\n\n"
    f"{COMMON_ANALYTICAL_GUIDELINES}\n\n"
    f"{_DEFAULT_STRUCTURE}\n\n"
    f"{COMMON_MEMORY_AND_CONSTRAINTS}"
)


# ---------------------------------------------------------------------------
# AnswerSynthesizer Coordinator
# ---------------------------------------------------------------------------

class AnswerSynthesizer:
    """Synthesizes grounded narrative answers from retrieved evidence."""

    _render_comparison_matrix = staticmethod(render_comparison_matrix)
    _render_front_page_comparison = staticmethod(render_front_page_comparison)
    _render_broadsheet_perspectives = staticmethod(render_broadsheet_perspectives)
    _render_explore_further = staticmethod(render_explore_further)

    def __init__(self, provider: ChatModelProvider | None = None) -> None:
        self._provider = provider

    def _has_valid_evidence(self, evidence_items: list[dict[str, Any]]) -> bool:
        """Check whether evidence contains non-empty grounded content."""
        return has_valid_evidence(evidence_items)

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

        primary = self._get_provider(model_override=model_override)
        if primary:
            candidates.append(primary)
            seen_keys.add(f"{getattr(primary, 'provider_name', '')}:{getattr(primary, '_model', '')}")

        is_cloud_request = bool(
            primary
            and getattr(primary, "provider_name", "") in {"openrouter", "gemini", "groq", "openai", "nvidia"}
        ) or (model_override and any(p in model_override for p in ["openrouter", "gemini", "groq", "openai", "nvidia"]))

        failover_keys = DEFAULT_CLOUD_FAILOVER if is_cloud_request else DEFAULT_LOCAL_FAILOVER
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

    def _build_evidence_context(self, evidence_items: list[dict[str, Any]], query: str = "") -> str:
        """Format retrieved evidence documents into structured prompt context with strict token budgeting."""
        return build_evidence_context(evidence_items, query=query)

    def _build_synthesizer_user_prompt(
        self,
        query: str,
        archetype: str,
        evidence_items: list[dict[str, Any]],
        context: str,
    ) -> str:
        """Construct grounded synthesizer prompt with explicit publication boundaries."""
        return build_synthesizer_user_prompt(
            query=query, archetype=archetype, evidence_items=evidence_items, context=context,
        )

    @staticmethod
    def _make_citation(item: dict[str, Any], headline: str | None = None) -> AgentCitation:
        """Create an AgentCitation object from an evidence item."""
        is_web = bool(item.get("is_web") or item.get("source_tool") == "web_search")
        hl = headline or item.get("headline", "Untitled")
        if is_web:
            return AgentCitation(
                newspaper_name=item.get("newspaper_name", "Live Web"),
                issue_date=item.get("issue_date", "Live Web"),
                page_number=1,
                headline=hl,
                article_id=0,
                snippet=(item.get("snippet") or "")[:300],
                issue_id=0,
                bboxes=[],
                url=item.get("url") or "",
                source_type="web",
                is_web=True,
            )
        pages = item.get("pages", [1])
        page_val = int(pages[0]) if pages and pages[0] else 1
        return AgentCitation(
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

    def extract_citations(
        self,
        text: str,
        evidence_items: list[dict[str, Any]],
    ) -> list[AgentCitation]:
        """Extract and structure verified citations mentioned in the text or used from evidence."""
        citations: list[AgentCitation] = []
        seen_keys: set[str] = set()
        text_lower = (text or "").lower()

        # Step 1: Filter candidates to genuine articles (article_id > 0) or web articles
        # Strictly exclude aggregate tool artifacts (article_id == 0) like manifests and coverage matrices
        candidate_items = [
            item for item in evidence_items
            if (bool(item.get("is_web") or item.get("source_tool") == "web_search") or int(item.get("article_id") or 0) > 0)
        ]
        # In case tests or mock fixtures omit article_id, keep non-manifest items
        if not candidate_items and evidence_items:
            candidate_items = [
                item for item in evidence_items
                if not str(item.get("headline", "")).startswith(
                    ("Issue Manifest:", "Coverage Audit:", "Article Count Analysis:", "Archive Topic", "Frontpage Prominence")
                )
            ]

        # Step 2: Check for references in synthesized text
        for item in candidate_items:
            is_web = bool(item.get("is_web") or item.get("source_tool") == "web_search")
            raw_hl = item.get("headline", "")
            sub_hl = item.get("subheadline")
            byline = item.get("byline_author")
            snip = item.get("snippet") or item.get("summary") or ""
            hl, _ = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
            hl_clean = hl.strip().lower()
            url = item.get("url") or ""

            is_referenced = False
            if is_web:
                if (url and url.lower() in text_lower) or (hl_clean and len(hl_clean) > 4 and hl_clean in text_lower):
                    is_referenced = True
            else:
                # Check exact or substring headline match
                if hl_clean and len(hl_clean) > 5 and hl_clean in text_lower:
                    is_referenced = True
                elif hl_clean:
                    # Match significant headline tokens (>= 4 chars)
                    tokens = [w for w in re.findall(r"\w+", hl_clean) if len(w) > 3]
                    if len(tokens) >= 3:
                        matched_tokens = sum(1 for tok in tokens if tok in text_lower)
                        if matched_tokens / len(tokens) >= 0.65:
                            is_referenced = True

                # Check punctuation-stripped headline
                if not is_referenced and hl_clean:
                    hl_simple = re.sub(r"[^\w\s]", "", hl_clean).strip()
                    if len(hl_simple) > 8 and hl_simple in re.sub(r"[^\w\s]", "", text_lower):
                        is_referenced = True

            if is_referenced:
                dedup_key = f"{item.get('newspaper_name')}_{item.get('issue_date')}_{hl}"
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    citations.append(self._make_citation(item, headline=hl))

        # Fallback if no specific inline references matched: strictly pick top genuine candidate articles
        if not citations and candidate_items:
            for item in candidate_items[:2]:
                raw_hl = item.get("headline", "")
                sub_hl = item.get("subheadline")
                byline = item.get("byline_author")
                snip = item.get("snippet") or item.get("summary") or ""
                hl, _ = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
                citations.append(self._make_citation(item, headline=hl))

        return citations

    def _build_synthesizer_system_prompt(
        self,
        archetype: str,
        query: str,
        evidence_items: list[dict[str, Any]] | None = None,
    ) -> str:
        """Construct intent-aware dynamic system prompt tailored to query archetype and domain."""
        domain = detect_domain_from_query(query, evidence_items)

        if archetype == "cross_newspaper_comparison" and domain:
            metric_col = DOMAIN_TAXONOMY.get(domain, {}).get("metric_col", "Key Takeaways & Core Findings")
            structure = f"""REQUIRED RESPONSE STRUCTURE:
1. ### ⚡ Executive Summary: {domain} Intelligence
   - 2 to 3 crisp, authoritative sentences synthesizing the primary {domain} developments, market triggers, policy actions, and key broadsheet takeaways.
   - Introduce developments directly; never copy the user's prompt or write robotic meta-text.

2. ### 📊 Cross-Newspaper {domain} Comparison Matrix
   - A structured Markdown comparison table comparing the publications:
     | Publication | Issue Date | Top {domain} Headline | {metric_col} | Editorial Angle / Focus |
   - Populate actual details, figures, and focus areas from the verified evidence for each newspaper.
   - ZERO-COVERAGE RULE: If a publication lacks standalone reporting in {domain}, explicitly write:
     | **[Publication]** | [Date] | No standalone {domain} reporting | Carried no standalone {domain} reporting in this edition | Minimal / No coverage |

3. ### 📌 Key Verified Sector Highlights & Policies
   - Bullet points detailing specific sector decisions, fiscal moves, corporate actions, or metrics.
   - STRICT CITATION RULE: Every bullet point MUST end with an inline citation:
     [{{Newspaper Name}}, {{YYYY-MM-DD}}, Page {{Page_Number}}, "{{Headline}}"]
   - CITATION FORMAT TEMPLATE:
     * [Specific factual finding derived exclusively from verified article] [Publication Name, YYYY-MM-DD, Page X, "Exact Headline"].
     (Do NOT copy placeholder text; cite ONLY real articles from the provided evidence!)
   - SINGLE PAGE FORMAT MANDATE:
     * Use the clean single page number (e.g. Page 1, Page 5). NEVER output dual-page or folio notation such as "Page X (PDF p.Y)".

4. ### 📰 Broadsheet Editorial Framing & Divergence
   - Delineate differences in editorial tone, priorities, and depth between publications:
     * **{{Publication A}} Framing**: Specific emphasis, tone, unique figures.
     * **{{Publication B}} Framing**: Contrasting angle, counter-points, exclusive focus (or state that the edition carried no dedicated coverage in this domain).

5. ### 🔍 Explore Further
   - 2 to 3 concise follow-up prompts formatted strictly as:
     > 💡 Explore: <Specific follow-up question in {domain}>

ANTI-REPETITION CONSTRAINT:
- Do NOT repeat identical sentences, quotes, or phrasing across the Executive Summary, Comparison Matrix, and Key Highlights. Each section must provide distinct, complementary value.

STRICT DOMAIN PURITY MANDATE & NEGATIVE CONSTRAINTS:
- The user has specifically requested analysis of: {domain}.
- STRICT NEGATIVE EXCLUSIONS: When the query targets {domain}, NEVER include articles from Tax, Legal/Court trials, Crime, Sports, or Events/Entertainment (such as venue schedules, concert listings, court hearings, or tax compliance reports).
- ZERO-COVERAGE REPORTING: If a publication lacks standalone reporting in {domain}, explicitly state in the matrix and highlights:
  "[Publication] carried no standalone {domain} reporting in this edition."
  Do NOT substitute stories from other sections (such as events, court cases, or taxes) to fill out coverage for that newspaper!
- Concentrate exclusively on verified {domain} reporting!"""

        elif archetype == "article_catalog":
            display_domain = domain or "Archive"
            structure = f"""REQUIRED RESPONSE STRUCTURE:
1. ### ⚡ Executive Summary: {display_domain} Article Catalog
   - 1 to 2 crisp, authoritative sentences synthesizing the catalog scope: total articles identified, publication editions, active date, and primary topical coverage.

2. ### 📋 Comprehensive {display_domain} Articles Catalog
   - A comprehensive Markdown table listing all matching articles from the archive manifest:
     | # | Publication | Issue Date | Page | Section | Headline | Author / Byline |
   - Use the sanitized, clean headlines from the evidence. Never display author/doctor names as headlines!
   - Every row must reflect an actual article from the manifest.

3. ### 📌 Key Sector Highlights & Featured Stories
   - 3 to 5 bullet points highlighting the most notable stories, findings, policy decisions, or medical/business developments.
   - STRICT CITATION RULE: Every bullet point MUST end with an inline citation:
     [{{Newspaper Name}}, {{YYYY-MM-DD}}, Page {{Page_Number}}, "{{Headline}}"]
   - CITATION FORMAT TEMPLATE:
     * [Specific factual finding derived exclusively from verified article] [Publication Name, YYYY-MM-DD, Page X, "Exact Headline"].
     (Do NOT copy placeholder text; cite ONLY real articles from the provided evidence!)
   - SINGLE PAGE FORMAT MANDATE:
     * Use the clean single page number (e.g. Page 1, Page 5). NEVER output dual-page or folio notation such as "Page X (PDF p.Y)".

4. ### 🔍 Explore Further
   - 2 to 3 concise follow-up prompts formatted strictly as:
     > 💡 Explore: <Specific follow-up question or deep-dive into one of the listed articles>

ANTI-REPETITION CONSTRAINT:
- Do NOT repeat the same sentences, statistics, or phrasing across the Executive Summary, Catalog Table, and Key Highlights. Each section must provide distinct, complementary value."""

        elif archetype == "cross_newspaper_comparison":
            structure = """REQUIRED RESPONSE STRUCTURE:
1. ### ⚡ Executive Summary: Broadsheet Edition Overview
   - 2 to 3 authoritative sentences comparing the editions: overall news agendas, scale, and top themes of the day across publications.

2. ### 📰 Front-Page (Page 1) Lead Stories Comparison
   - Compare the banner headlines and Page 1 lead stories chosen by each editorial board.
   - Explain what each publication prioritized as its primary top-of-the-fold news and why.
   - Strict inline citations: [{Newspaper Name}, {YYYY-MM-DD}, Page 1, "{Headline}"]

3. ### 📊 Section Distribution & Coverage Scale
   - Breakdown and comparison of edition size (total pages, total articles) and editorial composition (National vs Regional vs Business vs Sports vs Opinion).

4. ### 🎯 Exclusive Stories & Distinct Agendas
   - Highlight stories covered exclusively by one paper and omitted by the other(s).
   - Characterize the unique editorial voice and regional focus of each newspaper.

5. ### 🔍 Explore Further
   - 2 to 3 concise follow-up prompts formatted strictly as:
     > 💡 Explore: <Specific comparative follow-up question>"""

        elif archetype == "thematic_timeline":
            structure = """REQUIRED RESPONSE STRUCTURE:
1. ### ⚡ Executive Summary: Chronological Progression
   - Crisp synthesis of the overarching arc, beginning, inflection points, and current status of the event.

2. ### 📅 Milestone Timeline & Event Progression
   - Chronological sequence of dated milestones and developments:
     * **{Date / Phase}**: Specific development, key figures, actions taken. [{Newspaper Name}, {YYYY-MM-DD}, Page {Page_Number}, "{Headline}"]
   - SINGLE PAGE FORMAT MANDATE:
     * Use the clean single page number (e.g. Page 1, Page 5). NEVER output dual-page or folio notation such as "Page X (PDF p.Y)".

3. ### 📈 Thematic Trajectory & Broadsheet Evolution
   - Analysis of how broadsheet coverage, sentiment, and editorial stance shifted across the timeline.

4. ### 🔍 Explore Further
   - 2 to 3 concise follow-up prompts formatted strictly as:
     > 💡 Explore: <Specific timeline follow-up question>"""

        else:
            structure = _DEFAULT_STRUCTURE

        return (
            f"You are NewsLens-AI, an elite broadsheet intelligence assistant.\n"
            f"Your goal is to analyze, explain, and synthesize coverage into a structured, highly readable brief.\n\n"
            f"{COMMON_ANALYTICAL_GUIDELINES}\n\n"
            f"{structure}\n\n"
            f"{COMMON_MEMORY_AND_CONSTRAINTS}"
        )

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

        if not has_evidence and not has_meta_history:
            return EMPTY_EVIDENCE_RESPONSE, [], 0.0

        context = self._build_evidence_context(evidence_items, query=query)
        user_prompt = self._build_synthesizer_user_prompt(
            query=query, archetype=archetype, evidence_items=evidence_items, context=context,
        )
        system_prompt = self._build_synthesizer_system_prompt(
            archetype=archetype, query=query, evidence_items=evidence_items,
        )

        messages = [Message(role="system", content=system_prompt)]
        if chat_history:
            for turn in chat_history[-6:]:
                r = turn.get("role", "user")
                c = str(turn.get("content", "")).strip()
                if c:
                    messages.append(Message(role=r, content=c))
        messages.append(Message(role="user", content=user_prompt))

        cost_usd = 0.0
        providers = self._get_provider_candidates(model_override=model_override)

        for provider in providers:
            try:
                response = await provider.complete(messages=messages, max_tokens=6144, temperature=0.1)
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
                    citations = self.extract_citations(answer_text, evidence_items)
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
        citations = self.extract_citations(answer_text, evidence_items)
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

        if not has_evidence and not has_meta_history:
            yield EMPTY_EVIDENCE_RESPONSE
            return

        context = self._build_evidence_context(evidence_items, query=query)
        user_prompt = self._build_synthesizer_user_prompt(
            query=query, archetype=archetype, evidence_items=evidence_items, context=context,
        )
        system_prompt = self._build_synthesizer_system_prompt(
            archetype=archetype, query=query, evidence_items=evidence_items,
        )

        messages = [Message(role="system", content=system_prompt)]
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
                stream_gen = provider.complete_stream(messages=messages, max_tokens=6144, temperature=0.1)
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
        for word in summary.split(" "):
            yield word + " "

    def _generate_deterministic_summary(
        self,
        query: str,
        evidence_items: list[dict[str, Any]],
        archetype: str = "factual_lookup",
    ) -> str:
        """Structured deterministic grounded synthesis when all LLMs are offline."""
        return generate_deterministic_summary(query, evidence_items, archetype=archetype)


__all__ = [
    "AnswerSynthesizer",
    "COMMON_ANALYTICAL_GUIDELINES",
    "COMMON_MEMORY_AND_CONSTRAINTS",
    "DEFAULT_CLOUD_FAILOVER",
    "DEFAULT_LOCAL_FAILOVER",
    "DOMAIN_TAXONOMY",
    "EMPTY_EVIDENCE_RESPONSE",
    "SYNTHESIZER_SYSTEM_PROMPT",
    "detect_domain_from_query",
    "is_domain_match",
    "parse_thought_and_answer",
    "score_evidence_item",
]
