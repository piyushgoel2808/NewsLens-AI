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
4. CONVERSATION HISTORY & PUBLICATION ISOLATION:
   - The conversation history is ONLY for understanding conversational context and follow-up intent.
   - Substantive facts, headlines, and analysis must be derived EXCLUSIVELY from the "Available Newspaper Evidence" for the CURRENT turn.
   - NEVER carry over news stories, events, or newspaper titles from previous conversation turns into a new date-specific or comparative query.
   - If the user asks to compare newspapers on a specific date, you must ONLY report on the publications explicitly present in the verified evidence for that date.

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
"""


def _detect_domain_from_query(query: str, evidence_items: list[dict[str, Any]] | None = None) -> str | None:
    """Detect specific domain/sector from user query or evidence manifests."""
    if not query:
        return None
    q_lower = query.lower()
    if re.search(r"\b(econom(?:y|ic|ics)?|financ(?:e|ial)?|business|markets?|trade|tax(?:es|ation)?|budget|fiscal|monetary|bank(?:s|ing)?|corporate|revenue|gdp|stocks?|shares?)\b", q_lower):
        return "Economics & Finance"
    if re.search(r"\b(sports?|cricket|football|tennis|olympics?|tournaments?|match(?:es)?|boxing)\b", q_lower):
        return "Sports"
    if re.search(r"\b(politic(?:s|al)?|elections?|parliament|assembly|ministers?|cabinet|governance|policy|bills?)\b", q_lower):
        return "Politics & Governance"
    if re.search(r"\b(crimes?|courts?|legal|law|police|arrest(?:s|ed)?|investigations?|verdicts?|bail)\b", q_lower):
        return "Crime & Law"
    if re.search(r"\b(health|hospitals?|pharma(?:ceutical)?|medicines?|vaccines?|diseases?|doctors?)\b", q_lower):
        return "Health & Medicine"
    if re.search(r"\b(tech|technology|ai|artificial\s+intelligence|cyber|software)\b", q_lower):
        return "Technology & AI"

    # Also inspect evidence items if manifests contain explicit CATEGORY
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
                    r"In summary|Summary:|Answer:|Draft:\s*\n|Executive Summary)"
                )
                split_match = re.search(pattern, thought, flags=re.IGNORECASE)
                if split_match:
                    s_idx = split_match.start()
                    return thought[:s_idx].strip(), thought[s_idx:].strip()
                return thought, ""
        else:
            # Unclosed <think> tag
            after_think = text.split("<think>", 1)[1]
            pattern = (
                r"\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|"
                r"In summary|Summary:|Answer:|Draft:\s*\n|Executive Summary)"
            )
            split_match = re.search(pattern, after_think, flags=re.IGNORECASE)
            if split_match:
                split_idx = split_match.start()
                thought = after_think[:split_idx].strip()
                ans = after_think[split_idx:].strip()
                return thought, ans
            return after_think.strip(), ""

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
            r"In summary|Summary:|Answer:|Draft:\s*\n|Executive Summary)"
        )
        split_match = re.search(pattern, text, flags=re.IGNORECASE)
        if split_match:
            s_idx = split_match.start()
            thought_part = text[:s_idx].strip()
            ans_candidate = text[s_idx:].strip()
            ans_candidate = re.sub(r"^Draft:\s*\n*", "", ans_candidate, flags=re.IGNORECASE).strip()
            return thought_part, ans_candidate
        # If no answer section was emitted, the whole output was reasoning
        return text.strip(), ""

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
        is_cloud_request = bool(
            primary
            and getattr(primary, "provider_name", "") in {"openrouter", "gemini", "groq", "openai", "nvidia"}
        ) or (model_override and any(p in model_override for p in ["openrouter", "gemini", "groq", "openai", "nvidia"]))

        if is_cloud_request:
            failover_keys = [
                "nvidia_nemotron",
                "openrouter_nemotron",
                "openrouter_gemma4_26b",
                "gemini_flash",
                "groq_compound",
                "openai_gpt4o_mini",
                "groq_qwen",
                "ollama_llama3",
                "ollama_deepseek",
            ]
        else:
            failover_keys = [
                "nvidia_nemotron",
                "ollama_llama3",
                "ollama_deepseek",
                "openrouter_nemotron",
                "openrouter_gemma4_26b",
                "gemini_flash",
                "groq_compound",
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

    def _build_evidence_context(self, evidence_items: list[dict[str, Any]], query: str = "") -> str:
        """Format retrieved evidence documents into structured prompt context with strict token budgeting."""
        context_blocks: list[str] = []
        domain_name = _detect_domain_from_query(query, evidence_items) if query else None
        if domain_name and evidence_items:
            domain_stem_map: dict[str, list[str]] = {
                "Economics & Finance": [
                    "econom", "financ", "business", "market", "trade", "tax", "solar",
                    "power", "seabed", "fund", "money", "bank", "stock", "rupee", "dollar", "gdp"
                ],
                "Health & Medicine": [
                    "health", "hospital", "pharma", "medicine", "doctor", "patient",
                    "disease", "vaccin", "virus", "treatment", "care", "clinic", "surgery", "drug", "medical"
                ],
                "Sports": [
                    "sport", "cricket", "football", "tennis", "olympic", "tournament", "match", "boxing", "player", "game"
                ],
                "Politics & Governance": [
                    "politic", "election", "parliament", "assembly", "minister", "cabinet", "governance", "policy", "bill", "party", "vote"
                ],
                "Crime & Law": [
                    "crime", "court", "legal", "law", "police", "arrest", "investigation", "verdict", "bail", "judge", "jail"
                ],
                "Technology & AI": [
                    "tech", "technology", "ai", "artificial intelligence", "cyber", "software", "digital", "chip"
                ],
            }
            domain_terms = domain_stem_map.get(
                domain_name,
                [w.lower() for w in re.findall(r"\b\w{3,}\b", domain_name.lower()) if w.lower() not in {"and", "the", "for"}]
            )

            def _domain_score(it: dict[str, Any]) -> int:
                t = (it.get("headline", "") + " " + it.get("snippet", "") + " " + it.get("summary", "")).lower()
                if it.get("source_tool") in ("sql_analytics", "coverage_analysis") or "MANIFEST" in t or "RECONCILIATION" in t:
                    return 100
                if any(dt in t for dt in domain_terms):
                    return 50
                return 0

            sorted_evidence = sorted(evidence_items, key=_domain_score, reverse=True)
            budgeted_items = sorted_evidence[:12]
        else:
            budgeted_items = evidence_items[:12] if evidence_items else []

        seen_keys: set[str] = set()

        from app.retrieval.sanitizer import repair_text_ligatures
        from app.retrieval.sql_analytics import sanitize_headline

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

            # Deduplication key across items
            dedup_key = f"{hl.lower().strip()}_{text[:80].lower().strip()}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Allow up to 4,000 chars for issue manifests, coverage audits, and exclusive coverage differences so full headline listings are preserved
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
                evidence_tag = (
                    f"[Evidence: {np_name}, {dt}, Page {pdf_page} "
                    f"(PDF Page {pdf_page}), Headline: \"{hl}\"]"
                )
                photos = item.get("photos") or []
                photos_text = ""
                if photos:
                    photo_lines = []
                    for p_idx, p in enumerate(photos, 1):
                        p_type = p.get("visual_type") or "Photo"
                        p_cap = p.get("caption") or "No printed caption"
                        p_desc = p.get("vlm_description") or ""
                        desc_str = f" | Visual Scene: {p_desc}" if p_desc else ""
                        photo_lines.append(f"  * [{p_type} {p_idx}] Caption: \"{p_cap}\"{desc_str}")
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
        domain_note = ""
        domain = _detect_domain_from_query(query, evidence_items)
        if domain:
            domain_note = (
                f"TARGET DOMAIN FOCUS: {domain}\n"
                f"- Strictly filter your synthesis to {domain} topics, markets, figures, and developments.\n"
                f"- Discard any unrelated general news (sports, crime, local repairs) from the final response.\n\n"
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

    def extract_citations(
        self,
        text: str,
        evidence_items: list[dict[str, Any]],
    ) -> list[AgentCitation]:
        """Extract and structure verified citations mentioned in the text or used from evidence."""
        citations: list[AgentCitation] = []
        seen_keys: set[str] = set()
        text_lower = (text or "").lower()

        from app.retrieval.sql_analytics import sanitize_headline

        for item in evidence_items:
            is_web = item.get("is_web") or item.get("source_tool") == "web_search"
            raw_hl = item.get("headline", "")
            sub_hl = item.get("subheadline")
            byline = item.get("byline_author")
            snip = item.get("snippet") or item.get("summary") or ""
            hl, _ = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
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

    def _build_synthesizer_system_prompt(
        self,
        archetype: str,
        query: str,
        evidence_items: list[dict[str, Any]] | None = None,
    ) -> str:
        """Construct intent-aware dynamic system prompt tailored to the query archetype and domain."""
        domain = _detect_domain_from_query(query, evidence_items)

        common_guidelines = """CRITICAL ANALYTICAL GUIDELINES:
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
   - If the user asks to compare newspapers on a specific date, you must ONLY report on the publications explicitly present in the verified evidence for that date."""

        common_memory_and_constraints = """CONVERSATIONAL & CITATION MEMORY RULES:
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
     * You MUST begin your response IMMEDIATELY with the first section header: "### ⚡ Executive Summary"."""

        if archetype == "cross_newspaper_comparison" and domain:
            is_finance = any(w in domain.lower() for w in ["econom", "financ", "business", "market", "trade"])
            is_health = "health" in domain.lower()
            is_politics = any(w in domain.lower() for w in ["politic", "law", "crime"])

            if is_finance:
                metric_col = "Key Figures & Metrics"
            elif is_health:
                metric_col = "Key Findings & Medical Focus"
            elif is_politics:
                metric_col = "Key Policy Decisions & Statements"
            else:
                metric_col = "Key Takeaways & Core Findings"

            structure = f"""REQUIRED RESPONSE STRUCTURE:
1. ### ⚡ Executive Summary: {domain} Intelligence
   - 2 to 3 crisp, authoritative sentences synthesizing the primary {domain} developments, market triggers, policy actions, and key broadsheet takeaways.

2. ### 📊 Cross-Newspaper {domain} Comparison Matrix
   - A structured Markdown comparison table comparing the publications:
     | Publication | Issue Date | Top {domain} Headline | {metric_col} | Editorial Angle / Focus |
   - Populate actual details, figures, and focus areas from the verified evidence for each newspaper.

3. ### 📌 Key Verified Sector Highlights & Policies
   - Bullet points detailing specific sector decisions, fiscal moves, corporate actions, or metrics.
   - STRICT CITATION RULE: Every bullet point MUST end with an inline citation:
     [{{Newspaper Name}}, {{YYYY-MM-DD}}, Page {{PDF_Page}}, "{{Headline}}"]
   - CONCRETE CITATION EXAMPLE:
     * Over 4.5 lakh taxpayers claimed deductions under Section 80C according to official CBDT data [The Morning Standard, 2026-08-01, Page 7, "Direct Tax Compliance Surges in Q1"].

4. ### 📰 Broadsheet Editorial Framing & Divergence
   - Delineate differences in editorial tone, priorities, and depth between publications:
     * **{{Publication A}} Framing**: Specific emphasis, tone, unique figures.
     * **{{Publication B}} Framing**: Contrasting angle, counter-points, exclusive focus.

5. ### 🔍 Explore Further
   - 2 to 3 concise follow-up prompts formatted strictly as:
     > 💡 Explore: <Specific follow-up question in {domain}>

ANTI-REPETITION CONSTRAINT:
- Do NOT repeat identical sentences, quotes, or phrasing across the Executive Summary, Comparison Matrix, and Key Highlights. Each section must provide distinct, complementary value.

STRICT DOMAIN PURITY MANDATE:
- The user has specifically requested analysis of: {domain}.
- You MUST STRICTLY DISCARD and IGNORE any news stories or manifest entries that belong to other domains (such as local road repairs, boxing matches, migrant crossings, or unrelated crime) even if they appear in the archive manifest or evidence snippets.
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
   - CONCRETE CITATION EXAMPLE:
     * Over 4.5 lakh taxpayers claimed deductions under Section 80C according to official CBDT data [The Morning Standard, 2026-08-01, Page 7, "Direct Tax Compliance Surges in Q1"].

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
            structure = """REQUIRED RESPONSE STRUCTURE:
1. ### ⚡ Executive Summary
   - 1 to 2 crisp, authoritative sentences explaining the main development and takeaway.

2. ### 📌 Key Verified Facts & Highlights
   - Bullet points of specific numbers, figures, dates, and quotes.
   - STRICT CITATION RULE: Every bullet point MUST end with an inline citation:
     [{Newspaper Name}, {YYYY-MM-DD}, Page {PDF_Page}, "{Headline}"]
     * For Charts & Infographics: [📊 Chart: {Newspaper Name}, {YYYY-MM-DD}, Page {PDF_Page}, "{Headline}"]
     * For Live Web Search (if provided): [Web: {Source Title}]({URL})
   - CONCRETE CITATION EXAMPLE:
     * Over 4.5 lakh taxpayers claimed deductions under Section 80C according to official CBDT data [The Morning Standard, 2026-08-01, Page 7, "Direct Tax Compliance Surges in Q1"].

3. ### 📰 Broadsheet Perspectives & Focus Areas
   - Group reporting by publication (e.g. **Mint**, **Business Standard**, **The Hindu**).
   - 1 concise bullet point per paper on that paper's specific angle, bias, or unique data.

4. ### 🔍 Explore Further
   - 2 to 3 concise follow-up prompts formatted strictly as:
     > 💡 Explore: <Specific follow-up question or angle>

ANTI-REPETITION CONSTRAINT:
- Do NOT repeat the same sentences, statistics, or phrasing across the Executive Summary and Key Highlights. Each section must provide distinct, complementary value."""

        return (
            f"You are NewsLens-AI, an elite broadsheet intelligence assistant.\n"
            f"Your goal is to analyze, explain, and synthesize coverage into a structured, highly readable brief.\n\n"
            f"{common_guidelines}\n\n"
            f"{structure}\n\n"
            f"{common_memory_and_constraints}"
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

        # Strict Empty Evidence Hard-Stop
        if not has_evidence and not has_meta_history:
            return EMPTY_EVIDENCE_RESPONSE, [], 0.0

        citations = self.extract_citations("", evidence_items)
        context = self._build_evidence_context(evidence_items, query=query)

        user_prompt = self._build_synthesizer_user_prompt(
            query=query,
            archetype=archetype,
            evidence_items=evidence_items,
            context=context,
        )

        system_prompt = self._build_synthesizer_system_prompt(
            archetype=archetype,
            query=query,
            evidence_items=evidence_items,
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
        answer_text = ""
        providers = self._get_provider_candidates(model_override=model_override)

        for provider in providers:
            try:
                response = await provider.complete(
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.1,
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

        # Fallback if all providers fail
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

        # Strict Empty Evidence Hard-Stop
        if not has_evidence and not has_meta_history:
            yield EMPTY_EVIDENCE_RESPONSE
            return

        context = self._build_evidence_context(evidence_items, query=query)
        user_prompt = self._build_synthesizer_user_prompt(
            query=query,
            archetype=archetype,
            evidence_items=evidence_items,
            context=context,
        )

        system_prompt = self._build_synthesizer_system_prompt(
            archetype=archetype,
            query=query,
            evidence_items=evidence_items,
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

        # Sanitize all evidence headlines and repair font ligatures
        from app.retrieval.sanitizer import repair_text_ligatures
        from app.retrieval.sql_analytics import sanitize_headline

        for item in evidence_items:
            raw_hl = item.get("headline", "Untitled Article")
            sub_hl = item.get("subheadline")
            byline = item.get("byline_author")
            snip = item.get("snippet") or item.get("summary") or item.get("full_text") or ""
            hl, eff_byline = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
            hl = repair_text_ligatures(hl)
            item["headline"] = hl
            if eff_byline:
                item["byline_author"] = eff_byline
            if item.get("snippet"):
                item["snippet"] = repair_text_ligatures(item["snippet"])

        domain = _detect_domain_from_query(query, evidence_items)
        query_words = {w.lower() for w in query.split() if len(w) > 3}
        if domain:
            domain_stem_map = {
                "Economics & Finance": ["econom", "financ", "business", "market", "trade", "tax", "solar", "fund", "money", "bank", "stock", "gdp", "rupee"],
                "Health & Medicine": ["health", "hospital", "pharma", "medicine", "doctor", "patient", "disease", "vaccin", "drug", "care", "clinic", "surgery", "medical"],
                "Sports": ["sport", "cricket", "football", "tennis", "olympic", "match", "boxing"],
                "Politics & Governance": ["politic", "election", "parliament", "minister", "cabinet", "policy", "bill", "party"],
                "Crime & Law": ["crime", "court", "legal", "law", "police", "arrest", "verdict", "bail", "judge"],
                "Technology & AI": ["tech", "ai", "cyber", "software", "digital", "chip"],
            }
            query_words.update(domain_stem_map.get(domain, []))

        if archetype == "cross_newspaper_comparison":
            relevant_items = evidence_items
        else:
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

        pub_groups: dict[str, list[dict[str, Any]]] = {}
        for item in filtered_evidence:
            np_name = item.get("newspaper_name", "Archive")
            if np_name not in ("Multi-Newspaper Audit", "Aggregated Archive Analytics", "Archive"):
                pub_groups.setdefault(np_name, []).append(item)

        if not pub_groups:
            for item in filtered_evidence[:6]:
                np_name = item.get("newspaper_name", "Archive")
                pub_groups.setdefault(np_name, []).append(item)

        domain = _detect_domain_from_query(query, evidence_items)

        if archetype == "cross_newspaper_comparison":
            lines = [
                f"### ⚡ Executive Summary: {domain} Intelligence" if domain else "### ⚡ Executive Summary: Broadsheet Edition Comparison",
                (
                    f"Comprehensive comparative analysis covering **{query}** across "
                    f"verified broadsheet archives.\n"
                ),
            ]
            if domain:
                lines.append(f"### 📊 Cross-Newspaper {domain} Comparison Matrix\n")
                lines.append("| Publication | Issue Date | Coverage Focus | Key Verified Highlights |")
                lines.append("|---|---|---|---|")
                for pub, items in pub_groups.items():
                    first_item = items[0]
                    lines.append(f"| **{pub}** | {first_item.get('issue_date', '')} | {len(items)} verified item(s) | {first_item.get('headline', '')} |")
                lines.append("\n### 📌 Key Verified Sector Highlights & Policies")
            else:
                lines.append("### 📰 Front-Page (Page 1) Lead Stories Comparison")
                for pub, items in pub_groups.items():
                    first_item = items[0]
                    p_num = first_item.get("pages", [1])[0]
                    lines.append(f"- **{pub}**: Front-page lead report '{first_item.get('headline')}' [{pub}, {first_item.get('issue_date')}, Page {p_num}, \"{first_item.get('headline')}\"]")
                lines.append("\n### 📊 Section Distribution & Coverage Scale")

        else:
            lines = [
                "### ⚡ Executive Summary",
                (
                    f"Key broadsheet reporting regarding **{query}** was documented across "
                    f"regional archives, led by *{first_np}* ({first_dt}).\n"
                ),
                "### 📌 Key Verified Facts & Highlights",
            ]

        for item in filtered_evidence[:6]:
            np_name = item.get("newspaper_name", "Archive")
            dt = item.get("issue_date", "")
            pages = item.get("pages", [1])
            page_str = f"Page {pages[0]}" if pages else "Page 1"
            hl = item.get("headline", "Untitled")
            snip = item.get("snippet") or item.get("summary") or ""
            clean_snip = re.sub(r"\[Newspaper:.*?\]", "", snip).strip()
            clean_snip = re.sub(r"\[Page\(s\):.*?\]", "", clean_snip).strip()
            clean_snip = repair_text_ligatures(clean_snip)
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
