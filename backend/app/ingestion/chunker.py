"""Newspaper-aware hierarchical chunking engine.

Splits article full text into chunks while preserving paragraph boundaries and
prepending global newspaper context headers:
[Newspaper: {newspaper} | Date: {date} | Section: {section} | Headline: {headline} | Pages: {pages}]

Ensures each chunk retains global document context for vector retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logging import get_logger
from app.ingestion.detector import is_noise_or_promo_text, sanitize_block_text

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """A single contextualized article chunk."""

    chunk_index: int
    text: str
    token_count: int
    header_context: str
    raw_text: str


class NewspaperChunker:
    """Hierarchical chunker for newspaper articles."""

    def __init__(
        self,
        target_chunk_tokens: int = 350,
        chunk_overlap_tokens: int = 50,
        max_chunk_tokens: int = 500,
    ) -> None:
        self.target_chunk_tokens = target_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.max_chunk_tokens = max_chunk_tokens

    def create_header_context(
        self,
        newspaper_name: str,
        issue_date: str,
        headline: str,
        section: str | None = None,
        pages: list[int] | None = None,
        printed_pages: list[str] | None = None,
    ) -> str:
        """Construct the standardized contextual metadata header."""
        parts = [
            f"Newspaper: {newspaper_name}",
            f"Date: {issue_date}",
        ]
        if section:
            parts.append(f"Section: {section}")
        parts.append(f"Headline: {headline}")
        if printed_pages:
            p_str = ", ".join(printed_pages)
            if pages:
                pdf_str = ", ".join(str(p) for p in sorted(pages))
                parts.append(f"Page(s): {p_str} (PDF p.{pdf_str})")
            else:
                parts.append(f"Page(s): {p_str}")
        elif pages:
            pages_str = ", ".join(str(p) for p in sorted(pages))
            parts.append(f"Page(s): {pages_str}")

        return "[" + " | ".join(parts) + "]"

    def chunk_article(
        self,
        full_text: str,
        newspaper_name: str = "",
        issue_date: str = "",
        headline: str = "",
        section: str | None = None,
        pages: list[int] | None = None,
        printed_pages: list[str] | None = None,
    ) -> list[DocumentChunk]:
        """Split article text into overlapping, header-contextualized chunks."""
        text = sanitize_block_text(full_text).strip()
        if not text:
            return []

        header = self.create_header_context(
            newspaper_name=newspaper_name,
            issue_date=issue_date,
            headline=headline,
            section=section,
            pages=pages,
            printed_pages=printed_pages,
        )

        paragraphs = [
            p.strip() for p in re.split(r"\n\s*\n", text)
            if p.strip() and not is_noise_or_promo_text(p.strip())
        ]
        if not paragraphs:
            return []

        chunks: list[DocumentChunk] = []
        current_paragraphs: list[str] = []
        current_word_count = 0
        chunk_idx = 0

        for para in paragraphs:
            para_words = len(para.split())

            # If adding this paragraph exceeds target size and we already have content
            if current_paragraphs and (current_word_count + para_words > self.target_chunk_tokens):
                raw_chunk_text = "\n\n".join(current_paragraphs)
                combined_chunk_text = f"{header}\n\n{raw_chunk_text}"
                approx_tokens = len(combined_chunk_text.split())

                chunks.append(
                    DocumentChunk(
                        chunk_index=chunk_idx,
                        text=combined_chunk_text,
                        token_count=approx_tokens,
                        header_context=header,
                        raw_text=raw_chunk_text,
                    )
                )
                chunk_idx += 1

                # Overlap: keep the last paragraph if possible
                has_multi_paras = len(current_paragraphs) > 1
                last_para_short = (
                    has_multi_paras
                    and len(current_paragraphs[-1].split()) <= self.chunk_overlap_tokens
                )
                if last_para_short:
                    current_paragraphs = [current_paragraphs[-1], para]
                    current_word_count = len(current_paragraphs[0].split()) + para_words
                else:
                    current_paragraphs = [para]
                    current_word_count = para_words
            else:
                current_paragraphs.append(para)
                current_word_count += para_words

        # Final chunk
        if current_paragraphs:
            raw_chunk_text = "\n\n".join(current_paragraphs)
            combined_chunk_text = f"{header}\n\n{raw_chunk_text}"
            approx_tokens = len(combined_chunk_text.split())

            chunks.append(
                DocumentChunk(
                    chunk_index=chunk_idx,
                    text=combined_chunk_text,
                    token_count=approx_tokens,
                    header_context=header,
                    raw_text=raw_chunk_text,
                )
            )

        logger.debug(
            "Article chunked",
            extra={"headline": headline[:40], "chunks_created": len(chunks)},
        )

        return chunks
