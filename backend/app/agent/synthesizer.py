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
from app.retrieval.sanitizer import repair_text_ligatures
from app.retrieval.sql_analytics import sanitize_headline

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Centralized Domain Taxonomy
# ---------------------------------------------------------------------------

DOMAIN_TAXONOMY: dict[str, dict[str, Any]] = {
    "Economics & Finance": {
        "regex": r"\b(econom(?:y|ic|ics)?|financ(?:e|ial)?|business|markets?|trade|tax(?:es|ation)?|budget|fiscal|monetary|bank(?:s|ing)?|corporate|revenue|gdp|stocks?|shares?)\b",
        "stems": [
            "econom", "financ", "business", "market", "trade", "tax", "solar",
            "power", "seabed", "fund", "money", "bank", "stock", "rupee", "dollar",
            "gdp", "rbi", "corp", "profit",
        ],
        "metric_col": "Key Figures & Metrics",
        "negative_hl": [],
        "required_override": [],
    },
    "Health & Medicine": {
        "regex": r"\b(health|hospitals?|pharma(?:ceutical)?|medicines?|vaccines?|diseases?|doctors?)\b",
        "stems": [
            "health", "hospital", "pharma", "medicine", "doctor", "patient",
            "disease", "vaccin", "virus", "treatment", "care", "clinic", "surgery",
            "drug", "medical", "heart", "infect", "liver", "blood", "cancer",
            "illness", "symptom", "organ", "diet", "nutrition", "wellness", "therapy",
        ],
        "metric_col": "Key Findings & Medical Focus",
        "negative_hl": [
            "when: ", "where: ", "studio xo", "cases still pending", "tax collections",
            "excise duty", "deductions", "cricket", "bjp", "congress",
        ],
        "required_override": [
            "health", "doctor", "hospital", "medicine", "disease", "patient", "heart",
        ],
    },
    "Sports": {
        "regex": r"\b(sports?|cricket|football|tennis|olympics?|tournaments?|match(?:es)?|boxing)\b",
        "stems": [
            "sport", "cricket", "football", "tennis", "olympic", "tournament",
            "match", "boxing", "player", "game",
        ],
        "metric_col": "Key Match Results & Scores",
        "negative_hl": [],
        "required_override": [],
    },
    "Politics & Governance": {
        "regex": r"\b(politic(?:s|al)?|elections?|parliament|assembly|ministers?|cabinet|governance|policy|bills?)\b",
        "stems": [
            "politic", "election", "parliament", "assembly", "minister", "cabinet",
            "governance", "policy", "bill", "party", "vote",
        ],
        "metric_col": "Key Policy Decisions & Statements",
        "negative_hl": [],
        "required_override": [],
    },
    "Crime & Law": {
        "regex": r"\b(crimes?|courts?|legal|law|police|arrest(?:s|ed)?|investigations?|verdicts?|bail)\b",
        "stems": [
            "crime", "court", "legal", "law", "police", "arrest", "investigation",
            "verdict", "bail", "judge", "jail",
        ],
        "metric_col": "Key Legal Proceedings & Verdicts",
        "negative_hl": [],
        "required_override": [],
    },
    "Technology & AI": {
        "regex": r"\b(tech|technology|ai|artificial\s+intelligence|cyber|software)\b",
        "stems": [
            "tech", "technology", "ai", "artificial intelligence", "cyber",
            "software", "digital", "chip",
        ],
        "metric_col": "Key Technical Innovations & Specs",
        "negative_hl": [],
        "required_override": [],
    },
}

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

EMPTY_EVIDENCE_RESPONSE = (
    "I could not find any evidence or articles matching this query in the database. "
    "Please try adjusting your search terms."
)

_CHUNK_TAG_REGEX = re.compile(
    r"\[(?:Newspaper|Page\(s\)|Exact Chunk Match|Visual Data Asset|Article Parent Context|📷\s*Attached Image/Photo):?.*?\]",
    re.IGNORECASE,
)


def _clean_snippet(snip: str) -> str:
    """Strip retrieval artifacts and chunk tags from snippet text."""
    cleaned = _CHUNK_TAG_REGEX.sub("", snip).strip()
    cleaned = re.sub(r"^[\<\#\s\.\,\-]+", "", cleaned).strip()
    return repair_text_ligatures(cleaned)


def _detect_domain_from_query(query: str, evidence_items: list[dict[str, Any]] | None = None) -> str | None:
    """Detect specific domain/sector from user query or evidence manifests."""
    if not query:
        return None
    q_lower = query.lower()
    for domain, spec in DOMAIN_TAXONOMY.items():
        if re.search(spec["regex"], q_lower):
            return domain

    # Inspect evidence items if manifests contain explicit CATEGORY
    if evidence_items:
        for item in evidence_items:
            snip = item.get("snippet", "")
            m = re.search(r"CATEGORY:\s*([A-Za-z &]+)", snip)
            if m:
                cat = m.group(1).strip()
                if any(w in cat.lower() for w in ["econom", "financ", "business", "market"]):
                    return "Economics & Finance"
                return cat
    return None


def _is_domain_match(item: dict[str, Any], domain: str | None) -> bool:
    """Check if an evidence item strictly matches the target domain."""
    if not domain:
        return True
    spec = DOMAIN_TAXONOMY.get(domain)
    domain_terms = (
        spec["stems"] if spec else
        [w.lower() for w in re.findall(r"\b\w{3,}\b", domain.lower()) if w.lower() not in {"and", "the", "for"}]
    )
    text_corpus = (
        (item.get("headline") or "")
        + " "
        + (item.get("snippet") or "")
        + " "
        + (item.get("summary") or "")
        + " "
        + (item.get("section") or "")
    ).lower()
    hl = (item.get("headline") or "").lower()

    if spec:
        negative_patterns = spec.get("negative_hl", [])
        required_override = spec.get("required_override", [])
        if (
            negative_patterns
            and any(x in hl for x in negative_patterns)
            and (not required_override or not any(h in hl for h in required_override))
        ):
            return False

    return any(dt in text_corpus for dt in domain_terms)


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

    return "", ans_text


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
     [{Newspaper Name}, {YYYY-MM-DD}, Page {PDF_Page}, "{Headline}"]
     * For Charts & Infographics: [📊 Chart: {Newspaper Name}, {YYYY-MM-DD}, Page {PDF_Page}, "{Headline}"]
     * For Live Web Search (if provided): [Web: {Source Title}]({URL})

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
# AnswerSynthesizer Implementation
# ---------------------------------------------------------------------------

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
        domain_name = _detect_domain_from_query(query, evidence_items) if query else None
        if domain_name and evidence_items:
            spec = DOMAIN_TAXONOMY.get(domain_name)
            domain_terms = (
                spec["stems"] if spec else
                [w.lower() for w in re.findall(r"\b\w{3,}\b", domain_name.lower()) if w.lower() not in {"and", "the", "for"}]
            )

            def _domain_score(it: dict[str, Any]) -> int:
                t = (it.get("headline", "") + " " + it.get("snippet", "") + " " + it.get("summary", "")).lower()
                hl = (it.get("headline", "")).lower()
                if it.get("source_tool") in ("sql_analytics", "coverage_analysis") or "MANIFEST" in t or "RECONCILIATION" in t:
                    return 100
                if (
                    spec
                    and spec.get("negative_hl")
                    and any(x in hl for x in spec["negative_hl"])
                    and not any(h in hl for h in spec.get("required_override", []))
                ):
                    return -50
                if any(dt in t for dt in domain_terms):
                    return 50
                return 0

            sorted_evidence = sorted(evidence_items, key=_domain_score, reverse=True)
            budgeted_items = sorted_evidence[:12]
        else:
            budgeted_items = evidence_items[:12] if evidence_items else []

        seen_keys: set[str] = set()
        context_blocks: list[str] = []

        for item in budgeted_items:
            is_web = item.get("is_web") or item.get("source_tool") == "web_search"
            raw_hl = item.get("headline", "Untitled Article")
            sub_hl = item.get("subheadline")
            byline = item.get("byline_author")
            snip = item.get("snippet") or item.get("summary") or item.get("full_text") or ""
            hl, eff_byline = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
            hl = repair_text_ligatures(hl)
            item["headline"] = hl
            if eff_byline:
                item["byline_author"] = eff_byline

            raw_text = (item.get("snippet") or item.get("full_text") or item.get("summary") or "").strip()
            text = repair_text_ligatures(raw_text)
            item["snippet"] = text

            dedup_key = f"{hl.lower().strip()}_{text[:80].lower().strip()}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            is_manifest_or_matrix = (
                item.get("source_tool") in ("sql_analytics", "coverage_analysis")
                or "RELATIONAL ARCHIVE MANIFEST" in text
                or "COVERAGE RECONCILIATION MATRIX" in text
                or "VERIFIED EXCLUSIVE COVERAGE" in text
            )
            max_chars = 4000 if is_manifest_or_matrix else 1200
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + " ... [excerpt truncated for length]"

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
                evidence_tag = f'[Evidence: {np_name}, {dt}, Page {pdf_page} (PDF Page {pdf_page}), Headline: "{hl}"]'

                photos = item.get("photos") or []
                photos_text = ""
                if photos:
                    photo_lines = [
                        f"  * [{p.get('visual_type') or 'Photo'} {p_idx}] Caption: \"{p.get('caption') or 'No printed caption'}\""
                        + (f" | Visual Scene: {p['vlm_description']}" if p.get("vlm_description") else "")
                        for p_idx, p in enumerate(photos, 1)
                    ]
                    photos_text = "\nAttached Photos & Visual Elements:\n" + "\n".join(photo_lines) + "\n"

                context_blocks.append(
                    f"--- ARCHIVE EVIDENCE EXCERPT [{idx}] ---\n"
                    f"{evidence_tag}\n"
                    f"Publication: {np_name}\n"
                    f"Date: {dt}\n"
                    f"Page(s): Page {pdf_page} (PDF Page {pdf_page})\n"
                    f"Headline: {hl}\n"
                    f"Content:\n{text}\n"
                    f"{photos_text}"
                )
        return "\n".join(context_blocks)

    def _build_synthesizer_user_prompt(
        self,
        query: str,
        archetype: str,
        evidence_items: list[dict[str, Any]],
        context: str,
    ) -> str:
        """Construct grounded synthesizer prompt with explicit publication boundaries."""
        verified_pubs = sorted(list({
            str(item.get("newspaper_name", "")).strip() for item in evidence_items
            if item.get("newspaper_name") and item.get("newspaper_name") not in (
                "Multi-Newspaper Audit", "Aggregated Archive Analytics", "Archive", "Unknown Publication", "Live Web"
            )
        }))
        pubs_note = f"Verified Available Publications for this Query: {', '.join(verified_pubs)}\n" if verified_pubs else ""
        isolation_rule = (
            f"STRICT PUBLICATION & DATE ISOLATION:\n"
            f"- You must ONLY report on and analyze the verified publications present in the current evidence ({', '.join(verified_pubs)}).\n"
            f"- NEVER mention, summarize, or cite articles from other publications or dates discussed in earlier conversation turns.\n\n"
            if verified_pubs else ""
        )
        domain = _detect_domain_from_query(query, evidence_items)
        domain_note = (
            f"TARGET DOMAIN FOCUS: {domain}\n"
            f"- Strictly filter your synthesis to {domain} topics, markets, figures, and developments.\n"
            f"- Discard any unrelated general news (sports, crime, local repairs) from the final response.\n\n"
            if domain else ""
        )

        return (
            f"User Research Query: {query}\n"
            f"Query Archetype: {archetype}\n"
            f"{pubs_note}"
            f"{isolation_rule}"
            f"{domain_note}"
            f"Available Newspaper Evidence:\n"
            f"{context or 'No new search results—refer to conversation history if applicable.'}\n\n"
            f"Synthesize an insightful, highly-structured executive intelligence response."
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

        for item in evidence_items:
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
                    citations.append(self._make_citation(item, headline=hl))

        # Fallback if no specific inline references matched
        if not citations and evidence_items:
            for item in evidence_items[:2]:
                citations.append(self._make_citation(item))

        return citations

    def _build_synthesizer_system_prompt(
        self,
        archetype: str,
        query: str,
        evidence_items: list[dict[str, Any]] | None = None,
    ) -> str:
        """Construct intent-aware dynamic system prompt tailored to query archetype and domain."""
        domain = _detect_domain_from_query(query, evidence_items)

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
     [{{Newspaper Name}}, {{YYYY-MM-DD}}, Page {{PDF_Page}}, "{{Headline}}"]
   - CITATION FORMAT TEMPLATE:
     * [Specific factual finding derived exclusively from verified article] [Publication Name, YYYY-MM-DD, Page X, "Exact Headline"].
     (Do NOT copy placeholder text; cite ONLY real articles from the provided evidence!)

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
     [{{Newspaper Name}}, {{YYYY-MM-DD}}, Page {{PDF_Page}}, "{{Headline}}"]
   - CITATION FORMAT TEMPLATE:
     * [Specific factual finding derived exclusively from verified article] [Publication Name, YYYY-MM-DD, Page X, "Exact Headline"].
     (Do NOT copy placeholder text; cite ONLY real articles from the provided evidence!)

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
     * **{Date / Phase}**: Specific development, key figures, actions taken. [{Newspaper Name}, {YYYY-MM-DD}, Page {PDF_Page}, "{Headline}"]

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

        citations = self.extract_citations("", evidence_items)
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
        if not self._has_valid_evidence(evidence_items):
            return EMPTY_EVIDENCE_RESPONSE

        # 1. Sanitize all evidence headlines and repair font ligatures
        for item in evidence_items:
            raw_hl = item.get("headline", "Untitled Article")
            sub_hl = item.get("subheadline")
            byline = item.get("byline_author")
            snip = item.get("snippet") or item.get("summary") or item.get("full_text") or ""
            hl, eff_byline = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
            item["headline"] = repair_text_ligatures(hl)
            if eff_byline:
                item["byline_author"] = eff_byline
            if item.get("snippet"):
                item["snippet"] = repair_text_ligatures(item["snippet"])

        # 2. Domain & keyword filtering
        domain = _detect_domain_from_query(query, evidence_items)
        query_words = {w.lower() for w in query.split() if len(w) > 3}
        if domain and domain in DOMAIN_TAXONOMY:
            query_words.update(DOMAIN_TAXONOMY[domain]["stems"])

        if archetype == "cross_newspaper_comparison":
            relevant_items = evidence_items
        else:
            relevant_items = [
                item for item in evidence_items
                if not query_words or any(
                    qw in (
                        (item.get("headline") or "")
                        + " "
                        + (item.get("snippet") or "")
                        + " "
                        + (item.get("summary") or "")
                    ).lower()
                    for qw in query_words
                )
            ]

        filtered_evidence = relevant_items if relevant_items else evidence_items

        # 3. Partition real news articles vs manifests
        real_articles = [
            item for item in filtered_evidence
            if not (item.get("headline") or "").startswith("Issue Manifest:")
            and "exact chunk match" not in (item.get("headline") or "").lower()
            and item.get("newspaper_name") not in ("Multi-Newspaper Audit", "Aggregated Archive Analytics")
        ]

        first = (real_articles or filtered_evidence)[0]
        first_np = first.get("newspaper_name", "Daily News")
        first_dt = first.get("issue_date", "")

        pub_groups: dict[str, list[dict[str, Any]]] = {}
        for item in filtered_evidence:
            np_name = item.get("newspaper_name", "Archive")
            if np_name not in ("Multi-Newspaper Audit", "Aggregated Archive Analytics", "Archive"):
                pub_groups.setdefault(np_name, []).append(item)

        if not pub_groups:
            for item in filtered_evidence[:6]:
                np_name = item.get("newspaper_name", "Archive")
                pub_groups.setdefault(np_name, []).append(item)

        lines: list[str] = []

        # 4. Render sections by archetype
        if archetype == "cross_newspaper_comparison":
            summary_desc = (
                f"Comparative analysis across verified broadsheet archives covering {domain or 'regional news'} developments."
            )
            lines.append(
                f"### ⚡ Executive Summary: {domain} Intelligence" if domain else "### ⚡ Executive Summary: Broadsheet Edition Comparison"
            )
            lines.append(f"{summary_desc}\n")

            if domain:
                lines.extend(self._render_comparison_matrix(pub_groups, domain))
            else:
                lines.extend(self._render_front_page_comparison(pub_groups, domain))
        else:
            lines.append("### ⚡ Executive Summary")
            lines.append(
                f"Archival broadsheet reporting covering {domain or 'regional news'} was documented across "
                f"regional publications, led by *{first_np}* ({first_dt}).\n"
            )
            lines.append("### 📌 Key Verified Facts & Highlights")

        # Prioritize real news articles for facts and highlights
        domain_reals = [a for a in real_articles if _is_domain_match(a, domain)]
        display_facts = domain_reals[:6] if domain_reals else real_articles[:6] if real_articles else filtered_evidence[:6]
        for item in display_facts:
            np_name = item.get("newspaper_name", "Archive")
            dt = item.get("issue_date", "")
            pages = item.get("pages", [1])
            page_str = f"Page {pages[0]}" if pages else "Page 1"
            hl = item.get("headline", "Untitled")
            clean_snip = _clean_snippet(item.get("snippet") or item.get("summary") or "")
            lines.append(f'- **{hl}**: {clean_snip[:180]}... [{np_name}, {dt}, {page_str}, "{hl}"]')

        lines.extend(self._render_broadsheet_perspectives(pub_groups, domain))
        lines.extend(self._render_explore_further(pub_groups))

        return "\n".join(lines)

    @staticmethod
    def _render_comparison_matrix(pub_groups: dict[str, list[dict[str, Any]]], domain: str) -> list[str]:
        """Render cross-newspaper comparison matrix table with zero-coverage handling."""
        lines = [
            f"### 📊 Cross-Newspaper {domain} Comparison Matrix\n",
            "| Publication | Issue Date | Coverage Focus | Key Verified Highlights |",
            "|---|---|---|---|",
        ]
        for pub, items in pub_groups.items():
            pub_reals = [
                it for it in items
                if not (it.get("headline") or "").startswith("Issue Manifest:")
                and "exact chunk match" not in (it.get("headline") or "").lower()
                and _is_domain_match(it, domain)
            ]
            if pub_reals:
                lead_item = pub_reals[0]
                hl = lead_item.get("headline") or "General Reporting"
                if hl.startswith("Issue Manifest:"):
                    hl = f"General {domain or 'News'} Archive Coverage"
                dt_val = lead_item.get("issue_date") or items[0].get("issue_date", "")
                lines.append(f"| **{pub}** | {dt_val} | {len(pub_reals)} verified item(s) | {hl} |")
            else:
                dt_val = items[0].get("issue_date", "")
                lines.append(
                    f"| **{pub}** | {dt_val} | No standalone {domain} reporting | "
                    f"Carried no standalone {domain} reporting in this edition |"
                )
        lines.append("\n### 📌 Key Verified Sector Highlights & Policies")
        return lines

    @staticmethod
    def _render_front_page_comparison(
        pub_groups: dict[str, list[dict[str, Any]]], domain: str | None
    ) -> list[str]:
        """Render front page headline comparisons across publications."""
        lines = ["### 📰 Front-Page (Page 1) Lead Stories Comparison"]
        for pub, items in pub_groups.items():
            pub_reals = [
                it for it in items
                if not (it.get("headline") or "").startswith("Issue Manifest:")
                and "exact chunk match" not in (it.get("headline") or "").lower()
            ]
            lead_item = pub_reals[0] if pub_reals else items[0]
            hl = lead_item.get("headline") or "Front-page report"
            if hl.startswith("Issue Manifest:"):
                hl = f"General {domain or 'News'} Archive Coverage"
            p_num = lead_item.get("pages", [1])[0]
            dt_val = lead_item.get("issue_date") or items[0].get("issue_date", "")
            lines.append(f"- **{pub}**: Front-page lead report '{hl}' [{pub}, {dt_val}, Page {p_num}, \"{hl}\"]")
        lines.append("\n### 📊 Section Distribution & Coverage Scale")
        return lines

    @staticmethod
    def _render_broadsheet_perspectives(
        pub_groups: dict[str, list[dict[str, Any]]], domain: str | None
    ) -> list[str]:
        """Render publication perspectives and zero-coverage notes."""
        lines = ["\n### 📰 Broadsheet Perspectives"]
        for pub, items in pub_groups.items():
            pub_reals = [
                it for it in items
                if not (it.get("headline") or "").startswith("Issue Manifest:")
                and "exact chunk match" not in (it.get("headline") or "").lower()
                and _is_domain_match(it, domain)
            ]
            if pub_reals:
                top_hl = pub_reals[0].get("headline") or "Reporting"
                if top_hl.startswith("Issue Manifest:"):
                    top_hl = f"General {domain or 'News'} Coverage"
                lines.append(f"- **{pub}**: Emphasized '{top_hl}' across {len(pub_reals)} related report(s).")
            else:
                lines.append(f"- **{pub}**: Carried no dedicated {domain} reports in this issue.")
        return lines

    @staticmethod
    def _render_explore_further(pub_groups: dict[str, list[dict[str, Any]]]) -> list[str]:
        """Render explore further follow-up hints."""
        lines = ["\n### 🔍 Explore Further"]
        for pub in list(pub_groups.keys())[:2]:
            lines.append(f"> 💡 Explore: What was {pub}'s detailed coverage on this topic?")
        return lines
