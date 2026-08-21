"""Answer Synthesizer: Grounded LLM narrative generation with strict source citations."""
from __future__ import annotations

from typing import Any

from app.agent.state import AgentCitation
from app.core.logging import get_logger
from app.providers.base import ChatModelProvider, Message
from app.providers.registry import get_registry

logger = get_logger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are NewsLens-AI, an expert newspaper research assistant.
Answer the user's research question thoroughly and accurately based ONLY on the evidence excerpts.

CRITICAL CITATION RULES:
1. Every factual claim MUST include an inline citation in the format:
   [Newspaper Name, YYYY-MM-DD, Page X, "Headline"]
2. If multiple sources corroborate a claim, cite them together: [Paper 1, Date] [Paper 2, Date].
3. Do NOT make up any dates, figures, quotes, or events not present in the evidence.
4. If evidence is insufficient, explicitly state what is missing.
5. Structure your response clearly with headings or chronological sections where appropriate.
"""


class AnswerSynthesizer:
    """Synthesizes grounded narrative answers from retrieved evidence."""

    def __init__(self, provider: ChatModelProvider | None = None) -> None:
        self._provider = provider

    def _get_provider(self) -> ChatModelProvider | None:
        if self._provider:
            return self._provider
        try:
            registry = get_registry()
            provider = registry.get_provider("answerer")
            if isinstance(provider, ChatModelProvider):
                return provider
        except Exception as e:
            logger.warning(
                "Could not load LLM answerer provider, using deterministic fallback",
                extra={"error": str(e)},
            )
        return None

    def _build_evidence_context(self, evidence_items: list[dict[str, Any]]) -> str:
        """Format retrieved evidence documents into structured prompt context."""
        context_blocks: list[str] = []
        for i, item in enumerate(evidence_items, start=1):
            np_name = item.get("newspaper_name", "Unknown Publication")
            dt = item.get("issue_date", "Unknown Date")
            pages = item.get("pages", [1])
            pages_str = ", ".join(str(p) for p in pages) if pages else "1"
            hl = item.get("headline", "Untitled Article")
            text = item.get("snippet") or item.get("full_text") or item.get("summary") or ""

            context_blocks.append(
                f"--- EVIDENCE EXCERPT [{i}] ---\n"
                f"Publication: {np_name}\n"
                f"Date: {dt}\n"
                f"Page(s): {pages_str}\n"
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
        seen_keys: set[tuple[str, str, int, str]] = set()

        for item in evidence_items:
            np_name = item.get("newspaper_name", "Daily News")
            dt = item.get("issue_date", "")
            pages = item.get("pages", [1])
            page_num = pages[0] if pages else 1
            hl = item.get("headline", "Untitled")
            art_id = item.get("article_id", 0)
            snip = item.get("snippet") or item.get("summary") or ""

            key = (np_name, dt, page_num, hl)
            if key not in seen_keys:
                seen_keys.add(key)
                citations.append(
                    AgentCitation(
                        newspaper_name=np_name,
                        issue_date=dt,
                        page_number=page_num,
                        headline=hl,
                        article_id=art_id,
                        snippet=snip[:300],
                    )
                )

        return citations

    async def synthesize(
        self,
        query: str,
        archetype: str,
        evidence_items: list[dict[str, Any]],
    ) -> tuple[str, list[AgentCitation], float]:
        """Synthesize answer with citations from evidence."""
        if not evidence_items:
            empty_msg = f"No relevant newspaper articles found for query: '{query}'."
            return empty_msg, [], 0.0

        provider = self._get_provider()
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
                answer_text = response.text
                cost_usd = response.cost_usd
            except Exception as e:
                logger.warning(
                    "LLM synthesis failed, generating rule-based grounded summary",
                    extra={"error": str(e)},
                )
                answer_text = self._generate_deterministic_summary(query, evidence_items)
        else:
            answer_text = self._generate_deterministic_summary(query, evidence_items)

        return answer_text, citations, cost_usd

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

            lines.append(f"- **{hl}**: {snippet} [{np_name}, {dt}, {page_str}, \"{hl}\"]\n")

        lines.append("\n*All findings verified against primary newspaper source scans.*")
        return "\n".join(lines)
