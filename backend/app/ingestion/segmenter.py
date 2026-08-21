"""Article Boundary Segmenter for Newspaper Pages.

Analyzes 1D reading order blocks and 2D spatial layouts to partition a page
into distinct, cohesive article units.

Extracts:
- Headline and subheadline
- Byline author
- Body text blocks and unified text
- Bounding box envelopes on the page
- Jump lines (continuation references)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.ingestion.reading_order import BlockType, OrderedReadingBlock

logger = get_logger(__name__)

# Regular expressions to detect jump lines / continuation markers
JUMP_OUT_REGEX = re.compile(
    r"(?:continued\s+on\s+page\s+(\d+)|see\s+page\s+(\d+)|cont['’]d\s+p\.?\s*(\d+)|turn\s+to\s+page\s+(\d+))",
    re.IGNORECASE,
)

JUMP_IN_REGEX = re.compile(
    r"(?:continued\s+from\s+page\s+(\d+)|from\s+page\s+(\d+)|cont['’]d\s+from\s+p\.?\s*(\d+))",
    re.IGNORECASE,
)

BYLINE_REGEX = re.compile(
    r"^(?:by\s+([A-Z][a-zA-Z\s\.\,\-]+)|special\s+to\s+([A-Z][a-zA-Z\s\.\,\-]+)|by\s+our\s+special\s+correspondent|from\s+our\s+bureau)",
    re.IGNORECASE,
)


@dataclass
class SegmentedArticle:
    """A single coherent article unit on a page."""

    article_temp_id: str
    headline: str
    subheadline: str | None = None
    byline_author: str | None = None
    body_text: str = ""
    word_count: int = 0
    bbox_list: list[tuple[float, float, float, float]] = field(default_factory=list)
    jump_to_page: int | None = None
    jump_from_page: int | None = None
    raw_blocks: list[OrderedReadingBlock] = field(default_factory=list)


class ArticleSegmenter:
    """Partitions reading blocks on a newspaper page into discrete article units."""

    def segment_page(
        self,
        page_number: int,
        ordered_blocks: list[OrderedReadingBlock],
    ) -> list[SegmentedArticle]:
        """Group ordered reading blocks into segmented article units."""
        if not ordered_blocks:
            return []

        articles: list[SegmentedArticle] = []
        current_article: SegmentedArticle | None = None
        article_counter = 1

        for block in ordered_blocks:
            text = block.text.strip()
            if not text:
                continue

            is_headline = block.block_type in (
                BlockType.BANNER_HEADLINE,
                BlockType.HEADLINE,
            )

            if is_headline or current_article is None:
                # Close previous article if exists and has text
                if current_article and (current_article.body_text or current_article.headline):
                    if not current_article.body_text:
                        current_article.body_text = current_article.headline
                    full_content = (
                        f"{current_article.headline}\n\n{current_article.body_text}".strip()
                        if current_article.headline != current_article.body_text
                        else current_article.body_text
                    )
                    current_article.word_count = len(full_content.split())
                    articles.append(current_article)

                # Check if text contains jump-in reference
                jump_from = None
                jump_in_match = JUMP_IN_REGEX.search(text)
                if jump_in_match:
                    pages = [p for p in jump_in_match.groups() if p is not None]
                    if pages:
                        jump_from = int(pages[0])

                # Extract headline vs initial body if multi-line or very long
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                headline_text = lines[0][:300] if lines else text[:300]
                initial_body = "\n".join(lines[1:]) if len(lines) > 1 else text

                current_article = SegmentedArticle(
                    article_temp_id=f"p{page_number}_art_{article_counter}",
                    headline=headline_text,
                    body_text=initial_body,
                    bbox_list=[block.bbox],
                    jump_from_page=jump_from,
                    raw_blocks=[block],
                )
                article_counter += 1
                continue

            # Check if this block is a byline
            byline_match = BYLINE_REGEX.match(text)
            if byline_match and not current_article.byline_author and len(text) < 100:
                current_article.byline_author = text
                current_article.bbox_list.append(block.bbox)
                current_article.raw_blocks.append(block)
                continue

            # Check if this block contains a jump-to destination
            jump_out_match = JUMP_OUT_REGEX.search(text)
            if jump_out_match:
                pages = [p for p in jump_out_match.groups() if p is not None]
                if pages:
                    current_article.jump_to_page = int(pages[0])

            # Append to body text
            if current_article.body_text:
                current_article.body_text += "\n\n" + text
            else:
                current_article.body_text = text

            current_article.bbox_list.append(block.bbox)
            current_article.raw_blocks.append(block)

        # Finalize the last article
        if current_article and (current_article.body_text or current_article.headline):
            if not current_article.body_text:
                current_article.body_text = current_article.headline
            full_content = (
                f"{current_article.headline}\n\n{current_article.body_text}".strip()
                if current_article.headline != current_article.body_text
                else current_article.body_text
            )
            current_article.word_count = len(full_content.split())
            articles.append(current_article)

        # Fallback for OCR/scanned pages with blocks but 0 articles detected
        if not articles:
            text_blocks = [b for b in ordered_blocks if b.text and b.text.strip()]
            if text_blocks:
                combined_text = "\n\n".join(b.text.strip() for b in text_blocks)
                first_line = combined_text.split("\n")[0][:200].strip()
                fallback_hl = first_line if first_line else f"Page {page_number} News"
                fallback_art = SegmentedArticle(
                    article_temp_id=f"p{page_number}_art_fallback_1",
                    headline=fallback_hl,
                    body_text=combined_text,
                    word_count=len(combined_text.split()),
                    bbox_list=[b.bbox for b in text_blocks],
                    raw_blocks=text_blocks,
                )
                articles.append(fallback_art)

        logger.info(
            "Page segmented into articles",
            extra={"page_number": page_number, "article_count": len(articles)},
        )

        return articles
