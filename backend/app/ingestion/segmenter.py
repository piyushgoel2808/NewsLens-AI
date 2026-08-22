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
    r"(?:continued\s+on\s+page\s+(\d+)|see\s+page\s+(\d+)|cont['’]d\s+p\.?\s*(\d+)|turn\s+to\s+page\s+(\d+)|details\s+on\s+page\s+(\d+)|report\s+on\s+page\s+(\d+))",
    re.IGNORECASE,
)

JUMP_IN_REGEX = re.compile(
    r"(?:continued\s+from\s+page\s+(\d+)|from\s+page\s+(\d+)|cont['’]d\s+from\s+p\.?\s*(\d+))",
    re.IGNORECASE,
)

TEASER_REGEX = re.compile(
    r"(?i)\b(?:turn\s+to\s+page\s*(\d+)|see\s+page\s*(\d+)|details\s+on\s+page\s*(\d+)|"
    r"report\s+on\s+page\s*(\d+)|full\s+report\s+on\s+page\s*(\d+)|page\s*(\d+)|"
    r"continued\s+on\s+page\s*(\d+)|\.{3}|…)\b"
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

MIN_ARTICLE_WORD_COUNT = 40

SECTION_HEADER_BLACKLIST = frozenset(
    {
        "tech & startups",
        "tech and startups",
        "deals, tech & startups",
        "deals tech startups",
        "mark to market",
        "news wrap",
        "in brief",
        "news in brief",
        "corporate",
        "global",
        "views",
        "views & opinions",
        "views and opinions",
        "long story",
        "mint money",
        "economy & policy",
        "economy and policy",
        "business of life",
        "plain facts",
        "mint primer",
        "companies",
        "markets",
        "smart way",
        "heprice",
        "nitial",
        "initial",
        "su",
        "myths and mantras",
        "mint curator",
        "ask mint",
    }
)


NUMERIC_STAT_PATTERN = re.compile(
    r"\b(?:\d+[\d,\.]*\s*(?:cr|crore|mn|million|bn|billion|lakh|%|pts|bps|usd|inr)?|"
    r"[\$₹€£]\s*\d+[\d,\.]*)\b",
    re.IGNORECASE,
)

KICKER_REGEX = re.compile(
    r"(?i)^(?:OUR\s+VIEW|MY\s+VIEW|THEIR\s+VIEW|QUICK\s+EDIT|PLAIN\s+FACTS|MINT\s+PRIMER|"
    r"MARK\s+TO\s+MARKET|MINT\s+CURATOR|COLUMN|ASK\s+MINT|POWER\s+POINT|"
    r"ECONOMY\s+&\s+POLICY|DEALS,\s+TECH\s+&\s+STARTUPS|MYTHS\s+AND\s+MANTRAS|"
    r"IN\s+BRIEF|ROUNDUP|NEWS\s+IN\s+BRIEF|LONG\s+STORY|MINT\s+MONEY|"
    r"VIEWS\s+&\s+OPINIONS|BUSINESS\s+OF\s+LIFE|CORPORATE|GLOBAL|COMPANIES)"
    r"[:\s\|\-]+(.*)$"
)


def extract_kicker_and_clean_headline(raw_text: str) -> tuple[str, str | None]:
    """Extract kicker/category prefix and return (clean_headline, kicker_text)."""
    clean_lines = " ".join(line.strip() for line in raw_text.split("\n") if line.strip())
    match = KICKER_REGEX.match(clean_lines.strip())
    if match:
        kicker_part = clean_lines[:match.start(1)].strip(" :|-")
        clean_hl = match.group(1).strip()
        if is_valid_headline_candidate(clean_hl):
            return clean_hl, kicker_part
    return clean_lines.strip(), None


def is_valid_headline_candidate(text: str) -> bool:
    """Ensure a block text is substantial enough to define an article headline."""
    cleaned = re.sub(r"[^\w\s]", "", text).strip()
    words = cleaned.split()
    if not words:
        return False
    # Single words (e.g. "LIMITED", "ISSUE", "EQUITY") are never valid article headlines
    if len(words) == 1:
        return False
    # Filter out section headers and recurring layout tags
    if (
        cleaned.lower() in SECTION_HEADER_BLACKLIST
        or text.strip().lower() in SECTION_HEADER_BLACKLIST
    ):
        return False
    # Filter out pure boilerplate token combinations
    if all(w.lower() in BOILERPLATE_TOKENS for w in words):
        return False
    # Numeric stat boxes are never headlines
    stat_matches = NUMERIC_STAT_PATTERN.findall(text)
    if len(stat_matches) >= 3 or (len(stat_matches) >= 2 and len(words) <= 6):
        return False
    # Require at least 2 words and substantial character length
    return not (len(words) < 2 or len(cleaned) < 8)


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
    is_teaser: bool = False
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
            from app.ingestion.detector import check_is_advertisement_text

            ad_blocks: list[OrderedReadingBlock] = []
            editorial_blocks: list[OrderedReadingBlock] = []

            for b in ordered_blocks:
                if not b.text or not b.text.strip():
                    continue
                # On Page 1, top region (< 850px) without ad keywords is editorial teaser/lead
                is_top_editorial = (
                    page_number == 1
                    and b.bbox[1] < 850
                    and not check_is_advertisement_text(b.text)
                )
                if is_top_editorial:
                    editorial_blocks.append(b)
                else:
                    ad_blocks.append(b)

            results: list[SegmentedArticle] = []
            if ad_blocks:
                combined_text = "\n\n".join(b.text.strip() for b in ad_blocks)
                # Find first line that represents the primary ad title
                first_line = ""
                for b in ad_blocks:
                    t = b.text.strip()
                    if check_is_advertisement_text(t) or len(t.split()) >= 2:
                        first_line = t.split("\n")[0][:120].strip()
                        break
                if not first_line:
                    first_line = combined_text.split("\n")[0][:120].strip()

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
                    bbox_list=[b.bbox for b in ad_blocks],
                    raw_blocks=ad_blocks,
                )
                logger.info(
                    "Page grouped advertisement unit",
                    extra={
                        "page_number": page_number,
                        "word_count": ad_art.word_count,
                        "headline": ad_hl,
                    },
                )
                results.append(ad_art)

            if editorial_blocks:
                ed_articles = self.segment_page(
                    page_number, editorial_blocks, is_advertisement_page=False
                )
                results.extend(ed_articles)

            if results:
                return results

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

                # Extract headline and initial body
                subhead_text: str | None = None
                if is_headline:
                    # Preserve entire headline string (do not truncate multi-line headlines)
                    clean_hl, kicker = extract_kicker_and_clean_headline(text)
                    headline_text = clean_hl
                    subhead_text = kicker
                    initial_body = ""
                else:
                    # First block on page without detected headline
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    clean_hl, kicker = extract_kicker_and_clean_headline(lines[0] if lines else "")
                    headline_text = clean_hl if clean_hl else f"Page {page_number} News"
                    subhead_text = kicker
                    initial_body = text

                is_teaser_candidate = bool(
                    block.block_type == BlockType.TEASER
                    or (page_number in (1, 2) and TEASER_REGEX.search(text))
                )

                current_article = SegmentedArticle(
                    article_temp_id=f"p{page_number}_art_{article_counter}",
                    headline=headline_text,
                    subheadline=subhead_text,
                    body_text=initial_body,
                    bbox_list=[block.bbox],
                    jump_from_page=jump_from,
                    is_teaser=is_teaser_candidate,
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
                pages = [p for p in jump_out_match.groups() if p is not None and p.isdigit()]
                if pages:
                    current_article.jump_to_page = int(pages[0])

            # Check teaser regex if jump not yet set
            if current_article.jump_to_page is None:
                teaser_match = TEASER_REGEX.search(text)
                if teaser_match:
                    pages = [p for p in teaser_match.groups() if p is not None and p.isdigit()]
                    if pages:
                        current_article.jump_to_page = int(pages[0])
                        current_article.is_teaser = True

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
            if (
                page_number in (1, 2)
                and current_article.word_count < 45
                and (
                    current_article.jump_to_page is not None
                    or TEASER_REGEX.search(full_content)
                )
            ):
                current_article.is_teaser = True
                if current_article.jump_to_page is None:
                    t_match = TEASER_REGEX.search(full_content)
                    if t_match:
                        t_pages = [p for p in t_match.groups() if p is not None and p.isdigit()]
                        if t_pages:
                            current_article.jump_to_page = int(t_pages[0])
            articles.append(current_article)

        # Fallback for OCR/scanned pages with blocks but 0 articles detected
        if not articles:
            text_blocks = [b for b in ordered_blocks if b.text and b.text.strip()]
            if text_blocks:
                combined_text = "\n\n".join(b.text.strip() for b in text_blocks)
                first_line = combined_text.split("\n")[0][:200].strip()
                clean_hl, kicker = extract_kicker_and_clean_headline(first_line)
                fallback_hl = clean_hl if clean_hl else f"Page {page_number} News"
                fallback_art = SegmentedArticle(
                    article_temp_id=f"p{page_number}_art_fallback_1",
                    headline=fallback_hl,
                    subheadline=kicker,
                    body_text=combined_text,
                    word_count=len(combined_text.split()),
                    bbox_list=[b.bbox for b in text_blocks],
                    raw_blocks=text_blocks,
                )
                articles.append(fallback_art)

        # Debundle News Briefs / Shorts clusters before consolidation
        debundled_articles: list[SegmentedArticle] = []
        counter = 1
        for art in articles:
            briefs = self._debundle_shorts_cluster(art, page_number, counter)
            debundled_articles.extend(briefs)
            counter += len(briefs)
        articles = debundled_articles

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
                and len(art.body_text.split()) >= 4
            )
            is_shorts = art.headline.startswith("[Shorts]") and w_count >= 15
            is_ad = (
                art.headline.startswith("[Advertisement]")
                or art.headline.startswith("[Public Notice]")
            )
            is_valid_teaser = bool(art.is_teaser and art.jump_to_page is not None)
            is_valid_structured_article = (
                is_valid_teaser
                or is_shorts
                or is_ad
                or (has_valid_hl and has_distinct_body and w_count >= 10)
                or (has_valid_hl and w_count >= 15)
                or (w_count >= MIN_ARTICLE_WORD_COUNT)
            )

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
                    and len(first.body_text.split()) >= 4
                )
                first_is_valid = (
                    (first.is_teaser and first.jump_to_page is not None)
                    or (first.headline.startswith("[Shorts]") and first.word_count >= 15)
                    or (first.headline.startswith("[Advertisement]"))
                    or (first_has_valid_hl and first_has_body and first.word_count >= 10)
                    or (first_has_valid_hl and first.word_count >= 15)
                    or (first.word_count >= MIN_ARTICLE_WORD_COUNT)
                )

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

        # Final Purge: Drop any remaining standalone fragment / boilerplate stub < 25 words
        valid_final_articles: list[SegmentedArticle] = []
        for art in articles:
            is_ad = art.headline.startswith(
                ("[Advertisement]", "[Public Notice]")
            )
            is_valid_teaser = bool(art.is_teaser and art.jump_to_page is not None)
            is_shorts = art.headline.startswith("[Shorts]") and art.word_count >= 15
            is_substantial = (
                is_valid_headline_candidate(art.headline)
                and art.word_count >= 6
            )
            is_keep = (
                is_ad
                or is_valid_teaser
                or is_shorts
                or is_substantial
                or art.word_count >= MIN_ARTICLE_WORD_COUNT
            )
            if is_keep:
                valid_final_articles.append(art)
            else:
                logger.info(
                    "Dropping standalone sub-threshold article fragment",
                    extra={
                        "page_number": page_number,
                        "headline": art.headline[:60],
                        "word_count": art.word_count,
                    },
                )
        articles = valid_final_articles

        logger.info(
            "Page segmented into articles",
            extra={"page_number": page_number, "article_count": len(articles)},
        )

        return articles

    def _debundle_shorts_cluster(
        self,
        art: SegmentedArticle,
        page_number: int,
        start_idx: int,
    ) -> list[SegmentedArticle]:
        """Debundle a 'Shorts' or 'News in Brief' column cluster into distinct child articles.

        Performs vertical bounding box slicing based on line/character span so that
        each debundled short gets a dedicated, non-overlapping bounding box.
        """
        full_text = (
            f"{art.headline}\n\n{art.body_text}".strip()
            if art.headline != art.body_text
            else art.body_text
        ).strip()
        if not full_text:
            return [art]

        is_shorts_hl = bool(
            re.search(
                r"(?i)\b(?:SHORTS|IN BRIEF|BRIEFS|ROUNDUP|NEWS IN BRIEF|"
                r"PLAIN FACTS|MARKET BRIEFS|QUICK READS|CHART OF THE DAY)\b",
                art.headline,
            )
        )

        # Look for bullet/slug split markers: •, ▪, ►, ■, bold slugs (e.g. SLUG:), or numbered items
        marker_pattern = re.compile(
            r"(?:^|\n\s*)(?:[•▪►■\*\–]|\d+\.|\b[A-Z\s]{3,30}:)\s*"
        )
        matches = list(marker_pattern.finditer(full_text))

        if not is_shorts_hl and len(matches) < 2:
            return [art]

        # Extract split chunks
        spans: list[tuple[int, int]] = []
        if matches:
            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
                spans.append((start, end))
        else:
            paras = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            if is_shorts_hl and len(paras) >= 2:
                cur_pos = 0
                for p in paras:
                    pos = full_text.find(p, cur_pos)
                    if pos >= 0:
                        spans.append((pos, pos + len(p)))
                        cur_pos = pos + len(p)

        if len(spans) < 2:
            return [art]

        total_len = max(len(full_text), 1)
        if len(art.bbox_list) > 1:
            body_boxes = art.bbox_list[1:]
            bx0 = min(b[0] for b in body_boxes)
            by0 = min(b[1] for b in body_boxes)
            bx1 = max(b[2] for b in body_boxes)
            by1 = max(b[3] for b in body_boxes)
            base_bbox = (bx0, by0, bx1, by1)
        elif art.bbox_list:
            base_bbox = art.bbox_list[0]
        else:
            base_bbox = (0.0, 0.0, 100.0, 100.0)

        x0, y0, x1, y1 = base_bbox
        h_span = max(y1 - y0, 1.0)

        debundled: list[SegmentedArticle] = []
        counter = 0

        for start, end in spans:
            chunk_raw = full_text[start:end].strip()
            clean_chunk = re.sub(r"^[•▪►■\*\–\d\.\:\s]+", "", chunk_raw).strip()
            if not clean_chunk:
                continue

            words = clean_chunk.split()
            if len(words) < 15:
                if debundled:
                    debundled[-1].body_text += f"\n\n{clean_chunk}"
                    debundled[-1].word_count = len(debundled[-1].body_text.split())
                    slice_y1 = y0 + (end / total_len) * h_span
                    prev_b = debundled[-1].bbox_list[0]
                    debundled[-1].bbox_list = [
                        (prev_b[0], prev_b[1], prev_b[2], max(prev_b[3], slice_y1))
                    ]
                continue

            colon_match = re.search(r"^([A-Z\s]{3,35})\s*[:–-]\s*(.*)", clean_chunk)
            if colon_match:
                slug_hl = f"[Shorts] {colon_match.group(1).strip()}"
                brief_body = clean_chunk
            else:
                first_sent = clean_chunk.split(".")[0][:90].strip()
                slug_hl = (
                    f"[Shorts] {first_sent}"
                    if first_sent
                    else f"[Shorts] News Brief {counter + 1}"
                )
                brief_body = clean_chunk

            slice_y0 = y0 + (start / total_len) * h_span
            slice_y1 = y0 + (end / total_len) * h_span
            sliced_bbox = (x0, round(slice_y0, 2), x1, round(slice_y1, 2))

            short_art = SegmentedArticle(
                article_temp_id=f"p{page_number}_short_{start_idx + counter}",
                headline=slug_hl,
                body_text=brief_body,
                word_count=len(words),
                bbox_list=[sliced_bbox],
            )
            debundled.append(short_art)
            counter += 1

        if len(debundled) >= 2:
            logger.info(
                "Debundled shorts cluster into distinct articles",
                extra={"page_number": page_number, "briefs_count": len(debundled)},
            )
            return debundled

        return [art]
