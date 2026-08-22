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


BOILERPLATE_TOKENS = {
    "limited", "ltd", "corp", "corporation", "pvt", "private", "equity", "issue",
    "issue,", "shares", "company", "notice", "promoters", "price", "band", "page",
    "continued", "from", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "advertisement", "public", "statutory", "tender", "bid", "face", "value",
}

MIN_ARTICLE_WORD_COUNT = 30


def is_valid_headline_candidate(text: str) -> bool:
    """Ensure a block text is substantial enough to define an article headline."""
    cleaned = re.sub(r"[^\w\s]", "", text).strip()
    words = cleaned.split()
    if not words:
        return False
    # Single words (e.g. "LIMITED", "ISSUE", "EQUITY") are never valid article headlines
    if len(words) == 1:
        return False
    # Filter out pure boilerplate token combinations
    if all(w.lower() in BOILERPLATE_TOKENS for w in words):
        return False
    # Require at least 2 words and substantial character length
    return not (len(words) < 2 or len(cleaned) < 12)


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
        is_advertisement_page: bool = False,
    ) -> list[SegmentedArticle]:
        """Group ordered reading blocks into segmented article units.

        If is_advertisement_page is True, the entire page is grouped into a single
        cohesive [Advertisement] article without internal fragmentation.
        """
        if not ordered_blocks:
            return []

        # Pillar 2: Full-Page Advertisement / Notice Single-Unit Enveloping
        if is_advertisement_page:
            text_blocks = [b for b in ordered_blocks if b.text and b.text.strip()]
            if text_blocks:
                combined_text = "\n\n".join(b.text.strip() for b in text_blocks)
                first_line = combined_text.split("\n")[0][:150].strip()
                ad_hl = (
                    f"[Advertisement] {first_line}"
                    if not first_line.upper().startswith("[ADVERTISEMENT]")
                    else first_line
                )
                ad_art = SegmentedArticle(
                    article_temp_id=f"p{page_number}_art_ad_1",
                    headline=ad_hl,
                    body_text=combined_text,
                    word_count=len(combined_text.split()),
                    bbox_list=[b.bbox for b in text_blocks],
                    raw_blocks=text_blocks,
                )
                logger.info(
                    "Page grouped as single advertisement article",
                    extra={"page_number": page_number, "word_count": ad_art.word_count},
                )
                return [ad_art]

        articles: list[SegmentedArticle] = []
        current_article: SegmentedArticle | None = None
        article_counter = 1

        for block in ordered_blocks:
            text = block.text.strip()
            if not text:
                continue

            is_headline = (
                block.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
                and is_valid_headline_candidate(text)
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

        # Pillar 3: Minimum Structural Thresholds & Orphan Snippet Absorption Pass
        consolidated: list[SegmentedArticle] = []
        for art in articles:
            full_c = (
                f"{art.headline}\n\n{art.body_text}".strip()
                if art.headline != art.body_text
                else art.body_text
            )
            w_count = len(full_c.split())
            art.word_count = w_count

            has_valid_hl = bool(art.headline and is_valid_headline_candidate(art.headline))
            has_distinct_body = bool(
                art.body_text
                and art.body_text != art.headline
                and len(art.body_text.split()) >= 5
            )
            is_valid_structured_article = (
                has_valid_hl and has_distinct_body and w_count >= 12
            ) or (w_count >= MIN_ARTICLE_WORD_COUNT)

            if not is_valid_structured_article and consolidated:
                prev = consolidated[-1]
                prev.body_text += f"\n\n{full_c}"
                prev.bbox_list.extend(art.bbox_list)
                prev.raw_blocks.extend(art.raw_blocks)
                prev_c = (
                    f"{prev.headline}\n\n{prev.body_text}".strip()
                    if prev.headline != prev.body_text
                    else prev.body_text
                )
                prev.word_count = len(prev_c.split())
            else:
                consolidated.append(art)

        if consolidated:
            # If first article is an orphan snippet, merge forward into next
            if len(consolidated) > 1:
                first = consolidated[0]
                first_has_valid_hl = bool(
                    first.headline and is_valid_headline_candidate(first.headline)
                )
                first_has_body = bool(
                    first.body_text
                    and first.body_text != first.headline
                    and len(first.body_text.split()) >= 5
                )
                first_is_valid = (
                    first_has_valid_hl and first_has_body and first.word_count >= 12
                ) or (first.word_count >= MIN_ARTICLE_WORD_COUNT)

                if not first_is_valid:
                    first = consolidated.pop(0)
                    second = consolidated[0]
                    first_c = (
                        f"{first.headline}\n\n{first.body_text}".strip()
                        if first.headline != first.body_text
                        else first.body_text
                    )
                    second.body_text = f"{first_c}\n\n{second.body_text}"
                    second.bbox_list = first.bbox_list + second.bbox_list
                    second.raw_blocks = first.raw_blocks + second.raw_blocks
                    second_c = (
                        f"{second.headline}\n\n{second.body_text}".strip()
                        if second.headline != second.body_text
                        else second.body_text
                    )
                    second.word_count = len(second_c.split())
            articles = consolidated

        logger.info(
            "Page segmented into articles",
            extra={"page_number": page_number, "article_count": len(articles)},
        )

        return articles
