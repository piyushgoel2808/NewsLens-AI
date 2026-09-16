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
from app.ingestion.detector import is_noise_or_promo_text
from app.ingestion.layout.analyzer import BlockType, OrderedReadingBlock
from app.ingestion.layout.slugs import (
    MARKETING_SLOGAN_REGEX,
    STANDALONE_FEATURE_KICKER_REGEX,
    extract_kicker_and_clean_headline,
    is_numbered_feature_subhead,
    is_pullquote_author_block,
    is_syndication_or_agency_slug,
    is_toc_index_block,
    is_valid_headline_candidate,
)

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

MIN_ARTICLE_WORD_COUNT = 40


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
    section: str | None = None
    printed_section: str | None = None



class ArticleSegmenter:
    """Partitions reading blocks on a newspaper page into discrete article units."""

    def _finalize_article(
        self,
        art: SegmentedArticle,
        articles: list[SegmentedArticle],
        page_number: int,
    ) -> None:
        """Compute word count, check teasers, and append article to results."""
        if not art.body_text:
            art.body_text = art.headline
        full_content = (
            f"{art.headline}\n\n{art.body_text}".strip()
            if art.headline != art.body_text
            else art.body_text
        )
        art.word_count = len(full_content.split())
        if (
            page_number in (1, 2)
            and art.word_count < 45
            and (art.jump_to_page is not None or TEASER_REGEX.search(full_content))
        ):
            art.is_teaser = True
            if art.jump_to_page is None:
                t_match = TEASER_REGEX.search(full_content)
                if t_match:
                    t_pages = [p for p in t_match.groups() if p is not None and p.isdigit()]
                    if t_pages:
                        art.jump_to_page = int(t_pages[0])
        articles.append(art)

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
        pending_byline: str | None = None
        pending_kicker: str | None = None
        in_multi_part_feature = False

        for idx, block in enumerate(ordered_blocks):
            text = block.text.strip()
            if not text or is_noise_or_promo_text(text):
                continue

            # Table of Contents (ToC) Index isolation (Step 2 & Step 4)
            if block.block_type == BlockType.TOC_INDEX or is_toc_index_block(text):
                logger.info(
                    "Isolated Table of Contents / Index block from article formation",
                    extra={"page_number": page_number, "text": text[:60]},
                )
                # Sever reading order chain: never merge into article or form hallucinated story
                continue

            # Pullquote author / speaker attribution isolation (Step 3)
            if (
                block.block_type in (BlockType.PULLQUOTE_AUTHOR, BlockType.METADATA)
                or is_pullquote_author_block(text)
            ):
                logger.info(
                    "Isolated Pullquote Author / Attribution from headline formation",
                    extra={"page_number": page_number, "text": text[:60]},
                )
                if current_article and len(current_article.body_text.split()) > 0:
                    current_article.body_text = f"{current_article.body_text}\n\n{text}".strip()
                    current_article.bbox_list.append(block.bbox)
                    current_article.raw_blocks.append(block)
                continue

            # 1. Multi-part feature kicker check (e.g. standalone MINT PRIMER, PLAIN FACTS) (Task C)
            if STANDALONE_FEATURE_KICKER_REGEX.match(text):
                in_multi_part_feature = True
                # Scan ahead for the immediate sub-banner master headline
                master_hl: str | None = None
                for future_blk in ordered_blocks[idx + 1: idx + 8]:
                    f_text = future_blk.text.strip()
                    if (
                        is_valid_headline_candidate(f_text)
                        and not is_numbered_feature_subhead(f_text)
                        and not is_syndication_or_agency_slug(f_text)
                    ):
                        master_hl = f_text
                        break
                if master_hl:
                    if current_article and (current_article.body_text or current_article.headline):
                        self._finalize_article(current_article, articles, page_number)
                    clean_hl, _ = extract_kicker_and_clean_headline(master_hl)
                    current_article = SegmentedArticle(
                        article_temp_id=f"p{page_number}_art_{article_counter}",
                        headline=clean_hl,
                        subheadline=text,
                        body_text="",
                        bbox_list=[block.bbox],
                        raw_blocks=[block],
                        byline_author=pending_byline,
                    )
                    pending_byline = None
                    pending_kicker = None
                    article_counter += 1
                    continue
                else:
                    pending_kicker = text
                    continue

            # 2. Syndication slug / agency stamp check (Task A)
            if is_syndication_or_agency_slug(text):
                if current_article and not current_article.byline_author:
                    current_article.byline_author = text
                else:
                    pending_byline = text
                continue

            # 3. Numbered feature subhead / question check (Task C)
            if is_numbered_feature_subhead(text):
                if current_article:
                    current_article.body_text = (
                        f"{current_article.body_text}\n\n{text}".strip()
                        if current_article.body_text
                        else text
                    )
                    current_article.bbox_list.append(block.bbox)
                    current_article.raw_blocks.append(block)
                else:
                    # Look ahead for master headline on page
                    master_hl = None
                    for future_blk in ordered_blocks[idx + 1: idx + 8]:
                        f_text = future_blk.text.strip()
                        if (
                            is_valid_headline_candidate(f_text)
                            and not is_numbered_feature_subhead(f_text)
                            and not is_syndication_or_agency_slug(f_text)
                        ):
                            master_hl = f_text
                            break
                    clean_hl = master_hl or f"Feature: {text[:60]}"
                    current_article = SegmentedArticle(
                        article_temp_id=f"p{page_number}_art_{article_counter}",
                        headline=clean_hl,
                        body_text=text,
                        bbox_list=[block.bbox],
                        raw_blocks=[block],
                    )
                    article_counter += 1
                continue

            # 4. Valid headline check
            is_headline = (
                block.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
                and is_valid_headline_candidate(text)
            )

            # If in multi-part feature (e.g. Mint Primer), do not split on internal headings
            if (
                in_multi_part_feature
                and is_headline
                and current_article
                and current_article.headline
            ):
                if text == current_article.headline or current_article.headline in text:
                    continue
                if block.block_type != BlockType.BANNER_HEADLINE and len(text.split()) <= 15:
                    current_article.body_text = (
                        f"{current_article.body_text}\n\n{text}".strip()
                        if current_article.body_text
                        else text
                    )
                    current_article.bbox_list.append(block.bbox)
                    current_article.raw_blocks.append(block)
                    continue
                else:
                    in_multi_part_feature = False

            if is_headline:
                if current_article and (current_article.body_text or current_article.headline):
                    self._finalize_article(current_article, articles, page_number)

                clean_hl, kicker = extract_kicker_and_clean_headline(text)
                jump_from = None
                jump_in_match = JUMP_IN_REGEX.search(text)
                if jump_in_match:
                    pages = [p for p in jump_in_match.groups() if p is not None]
                    if pages:
                        jump_from = int(pages[0])

                is_teaser_candidate = bool(
                    block.block_type == BlockType.TEASER
                    or (page_number in (1, 2) and TEASER_REGEX.search(text))
                )

                current_article = SegmentedArticle(
                    article_temp_id=f"p{page_number}_art_{article_counter}",
                    headline=clean_hl,
                    subheadline=kicker or pending_kicker,
                    body_text="",
                    bbox_list=[block.bbox],
                    jump_from_page=jump_from,
                    is_teaser=is_teaser_candidate,
                    raw_blocks=[block],
                    byline_author=pending_byline,
                )
                pending_byline = None
                pending_kicker = None
                article_counter += 1
                continue

            # 5. Non-headline block when current_article is None (initial blocks on page) (Task B)
            if current_article is None:
                # Scan ahead for the nearest headline on the page
                nearest_hl = None
                for future_blk in ordered_blocks[idx:]:
                    f_text = future_blk.text.strip()
                    if (
                        future_blk.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
                        and is_valid_headline_candidate(f_text)
                    ):
                        nearest_hl = f_text
                        break

                if nearest_hl:
                    clean_hl, kicker = extract_kicker_and_clean_headline(nearest_hl)
                    current_article = SegmentedArticle(
                        article_temp_id=f"p{page_number}_art_{article_counter}",
                        headline=clean_hl,
                        subheadline=kicker or pending_kicker,
                        body_text=text,
                        bbox_list=[block.bbox],
                        raw_blocks=[block],
                        byline_author=pending_byline,
                    )
                    pending_byline = None
                    pending_kicker = None
                    article_counter += 1
                else:
                    first_line = text.split("\n")[0][:60].strip()
                    clean_hl, kicker = extract_kicker_and_clean_headline(first_line)
                    current_article = SegmentedArticle(
                        article_temp_id=f"p{page_number}_art_{article_counter}",
                        headline=clean_hl if clean_hl else f"Page {page_number} News",
                        subheadline=kicker,
                        body_text=text,
                        bbox_list=[block.bbox],
                        raw_blocks=[block],
                    )
                    article_counter += 1
                continue

            # 6. Check byline match
            byline_match = BYLINE_REGEX.match(text)
            if byline_match and not current_article.byline_author and len(text) < 100:
                cand = byline_match.group(1) or text
                if not MARKETING_SLOGAN_REGEX.search(cand):
                    current_article.byline_author = text
                    current_article.bbox_list.append(block.bbox)
                    current_article.raw_blocks.append(block)
                    continue

            # 7. Check jump lines
            jump_out_match = JUMP_OUT_REGEX.search(text)
            if jump_out_match:
                pages = [p for p in jump_out_match.groups() if p is not None and p.isdigit()]
                if pages:
                    current_article.jump_to_page = int(pages[0])

            if current_article.jump_to_page is None:
                teaser_match = TEASER_REGEX.search(text)
                if teaser_match:
                    pages = [p for p in teaser_match.groups() if p is not None and p.isdigit()]
                    if pages:
                        current_article.jump_to_page = int(pages[0])
                        current_article.is_teaser = True

            # 8. Append to current article body
            if current_article.body_text:
                current_article.body_text += "\n\n" + text
            else:
                current_article.body_text = text

            current_article.bbox_list.append(block.bbox)
            current_article.raw_blocks.append(block)

        # Finalize the last article
        if current_article and (current_article.body_text or current_article.headline):
            self._finalize_article(current_article, articles, page_number)

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

        # Pillar 3: Minimum Structural Thresholds & Non-Destructive Orphan Snippet Absorption
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
            is_shorts = art.headline.startswith("[Shorts]") and w_count >= 8
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

            # Restrict merging: Absorb if not a valid structured article and below minimum standalone threshold (< 15 words)
            if not is_valid_structured_article and w_count < 15 and consolidated:
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
            # If first article is a tiny non-sentence fragment (< 6 words), merge forward
            if len(consolidated) > 1:
                first = consolidated[0]
                first_has_valid_hl = bool(
                    first.headline and is_valid_headline_candidate(first.headline)
                )
                first_is_valid = (
                    (first.is_teaser and first.jump_to_page is not None)
                    or (first.headline.startswith("[Shorts]") and first.word_count >= 8)
                    or (first.headline.startswith("[Advertisement]"))
                    or (first_has_valid_hl and first.word_count >= 8)
                    or (first.word_count >= MIN_ARTICLE_WORD_COUNT)
                )

                if not first_is_valid and first.word_count < 6:
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

        # Final Purge: Keep valid structured articles, shorts, ads, and substantial items
        valid_final_articles: list[SegmentedArticle] = []
        for art in articles:
            is_ad = art.headline.startswith(
                ("[Advertisement]", "[Public Notice]")
            )
            is_valid_teaser = bool(art.is_teaser and art.jump_to_page is not None)
            is_shorts = art.headline.startswith("[Shorts]") and art.word_count >= 8
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
        """Debundle a 'Shorts', datelined capsules, or 'News in Brief' column cluster into distinct child articles."""
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
                r"PLAIN FACTS|MARKET BRIEFS|QUICK READS|CHART OF THE DAY|IN FOCUS|CITY BRIEFS)\b",
                art.headline,
            )
        )

        # Look for bullet/slug split markers or Datelines: NEW DELHI:, MUMBAI —, PTI, etc.
        marker_pattern = re.compile(
            r"(?:^|\n\s*)(?:[•▪►■\*\–—]|\d+\.|\b(?:NEW\s+DELHI|MUMBAI|BENGALURU|CHENNAI|KOLKATA|HYDERABAD|WASHINGTON|LONDON|BEIJING|[A-Z\s]{3,25})\s*[:–—\-]\s*|\b[A-Z][A-Z\s]{2,20}\s*[-–—]\s*)\s*"
        )
        matches = list(marker_pattern.finditer(full_text))

        is_feature_explainer = bool(
            re.search(
                r"(?i)\b(?:MINT\s+PRIMER|PLAIN\s+FACTS|LONG\s+STORY|EXPLAINER|PRIMER|Q&A)\b",
                art.headline,
            )
            or (art.subheadline and re.search(r"(?i)\b(?:MINT\s+PRIMER|PLAIN\s+FACTS|EXPLAINER|PRIMER)\b", art.subheadline))
        )

        # Do NOT debundle unified feature explainers or regular articles with bylines
        if is_feature_explainer or (art.byline_author and not is_shorts_hl):
            return [art]

        if not is_shorts_hl and len(matches) < 2:
            return [art]

        # Extract split chunks
        paras = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        spans: list[tuple[int, int]] = []
        if matches and len(matches) >= 2:
            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
                spans.append((start, end))
        elif is_shorts_hl and len(paras) >= 2:
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
            if len(words) < 8:
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


# =============================================================================
# Cross-Page Story Continuation & Jump Assembly (absorbed from cross_page_assembler)
# =============================================================================


@dataclass
class PageBBoxMapping:
    """Represents an article's spatial presence on a specific page."""

    page_number: int
    bbox_list: list[tuple[float, float, float, float]]
    block_order: int = 0


@dataclass
class AssembledArticle:
    """A fully stitched article that may span one or more pages."""

    headline: str
    subheadline: str | None = None
    byline_author: str | None = None
    full_text: str = ""
    primary_page_number: int = 1
    word_count: int = 0
    pages_mapping: list[PageBBoxMapping] = field(default_factory=list)
    section: str | None = None
    printed_section: str | None = None
    merged_article_temp_ids: list[str] = field(default_factory=list)


def _clean_continuation_slug(headline: str) -> str:
    """Strip cross-page suffixes like '... FROM PAGE 1' before token comparison."""
    cleaned = re.sub(
        r"[\.\…]+\s*(?:from\s+page\s+\d+|cont['’]?d?|continued\s+from\s+p(?:age)?\s*\d*).*$",
        "",
        headline,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or headline


def _significant_tokens(text: str, ignore: set[str]) -> set[str]:
    """Extract significant tokens (>= 4 chars, not stopwords)."""
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) >= 4 and w not in ignore}


def _headline_overlap_metrics(h1: str, h2: str) -> tuple[float, float, int, int]:
    """Calculate token overlap metrics (Jaccard, asymmetric containment, min token length, entity overlap)."""
    h1_clean = _clean_continuation_slug(h1)
    h2_clean = _clean_continuation_slug(h2)
    tokens1 = set(re.findall(r"\w+", h1_clean.lower()))
    tokens2 = set(re.findall(r"\w+", h2_clean.lower()))
    ignore = {
        "continued", "from", "page", "cont", "see", "to", "the", "a", "an",
        "in", "on", "of", "and", "or", "for", "with", "at", "by", "as", "is",
    }
    t1 = tokens1 - ignore
    t2 = tokens2 - ignore
    if not t1 or not t2:
        return 0.0, 0.0, 0, 0
    intersection = len(t1 & t2)
    jaccard = intersection / len(t1 | t2)
    min_len = min(len(t1), len(t2))
    containment = intersection / min_len if min_len > 0 else 0.0

    sig1 = _significant_tokens(h1_clean, ignore)
    sig2 = _significant_tokens(h2_clean, ignore)
    entity_overlap = len(sig1 & sig2) if (sig1 and sig2) else 0

    return jaccard, containment, min_len, entity_overlap


def _has_continuation_marker(cand: SegmentedArticle) -> bool:
    """Check if candidate segment has physical continuation markers (jump-in or ellipsis)."""
    if cand.jump_from_page is not None:
        return True
    text_sample = f"{cand.headline} {cand.body_text[:200]}".lower()
    return bool(
        re.search(r"\b(?:continued\s+from|from\s+page|cont['’]d|turn\s+to|\.{3}|…)\b", text_sample)
    )


class CrossPageAssembler:
    """Assembles articles across pages using jump links and headline matching."""

    def assemble_issue_articles(
        self,
        pages_articles: dict[int, list[SegmentedArticle]],
    ) -> list[AssembledArticle]:
        """Merge cross-page article fragments across all pages of an issue."""
        if not pages_articles:
            return []

        sorted_pages = sorted(pages_articles.keys())
        assembled: list[AssembledArticle] = []
        absorbed_ids: set[str] = set()

        for page_num in sorted_pages:
            for art in pages_articles[page_num]:
                if art.article_temp_id in absorbed_ids:
                    continue

                initial_full_text = art.body_text.strip() if art.body_text else ""
                if art.headline and art.headline not in initial_full_text:
                    initial_full_text = (
                        f"{art.headline}\n\n{initial_full_text}".strip()
                        if initial_full_text
                        else art.headline
                    )
                if not initial_full_text:
                    initial_full_text = art.headline or "Untitled Article"

                current = AssembledArticle(
                    headline=art.headline or "Untitled Article",
                    subheadline=art.subheadline,
                    byline_author=art.byline_author,
                    full_text=initial_full_text,
                    primary_page_number=page_num,
                    pages_mapping=[
                        PageBBoxMapping(
                            page_number=page_num,
                            bbox_list=art.bbox_list,
                            block_order=0,
                        )
                    ],
                    section=art.section,
                    printed_section=art.printed_section,
                    merged_article_temp_ids=[art.article_temp_id],
                )

                target_page = art.jump_to_page
                current_source_page = page_num
                block_order_counter = 1

                for next_page in [p for p in sorted_pages if p > page_num]:
                    candidate_articles = pages_articles.get(next_page, [])
                    matched_continuation: SegmentedArticle | None = None

                    for cand in candidate_articles:
                        if cand.article_temp_id in absorbed_ids:
                            continue

                        jaccard, containment, min_len, entity_overlap = _headline_overlap_metrics(
                            current.headline, cand.headline
                        )

                        is_teaser_target = bool(art.is_teaser and target_page == next_page)
                        explicit_page_link = cand.jump_from_page == current_source_page

                        if (target_page == next_page or is_teaser_target) and (
                            explicit_page_link
                            or containment >= 0.30
                            or jaccard >= 0.20
                            or (entity_overlap >= 1 and _has_continuation_marker(cand))
                        ):
                            matched_continuation = cand
                            break

                        if (
                            containment >= 0.80
                            or jaccard >= 0.60
                            or (entity_overlap >= 2 and _has_continuation_marker(cand))
                        ):
                            if min_len < 4:
                                byline_match = bool(
                                    current.byline_author
                                    and cand.byline_author
                                    and current.byline_author.lower() in cand.byline_author.lower()
                                )
                                cont_marker = _has_continuation_marker(cand)
                                explicit_jump = cand.jump_from_page == current_source_page
                                safe_short_match = (
                                    byline_match
                                    or cont_marker
                                    or explicit_jump
                                    or jaccard >= 0.60
                                )
                                if not safe_short_match:
                                    continue

                            matched_continuation = cand
                            break

                    if matched_continuation:
                        absorbed_ids.add(matched_continuation.article_temp_id)
                        current.merged_article_temp_ids.append(matched_continuation.article_temp_id)
                        if len(matched_continuation.headline) > len(current.headline):
                            current.headline = matched_continuation.headline
                        if matched_continuation.byline_author and not current.byline_author:
                            current.byline_author = matched_continuation.byline_author
                        if matched_continuation.subheadline and not current.subheadline:
                            current.subheadline = matched_continuation.subheadline

                        current.full_text += "\n\n" + matched_continuation.body_text
                        current.pages_mapping.append(
                            PageBBoxMapping(
                                page_number=next_page,
                                bbox_list=matched_continuation.bbox_list,
                                block_order=block_order_counter,
                            )
                        )
                        block_order_counter += 1
                        current_source_page = next_page
                        target_page = matched_continuation.jump_to_page
                        if not target_page:
                            break

                current.word_count = len(current.full_text.split())

                if art.is_teaser and block_order_counter == 1 and current.word_count < 35:
                    if assembled:
                        prev_art = assembled[-1]
                        prev_art.full_text += f"\n\n{current.full_text}"
                        prev_art.word_count = len(prev_art.full_text.split())
                        prev_art.pages_mapping[0].bbox_list.extend(art.bbox_list)
                        prev_art.merged_article_temp_ids.extend(current.merged_article_temp_ids)
                    continue

                assembled.append(current)

        logger.info(
            "Issue cross-page assembly complete",
            extra={"total_assembled": len(assembled), "pages": len(sorted_pages)},
        )

        return assembled


__all__ = [
    "ArticleSegmenter",
    "AssembledArticle",
    "BYLINE_REGEX",
    "CrossPageAssembler",
    "JUMP_IN_REGEX",
    "JUMP_OUT_REGEX",
    "MIN_ARTICLE_WORD_COUNT",
    "PageBBoxMapping",
    "SegmentedArticle",
    "TEASER_REGEX",
    "_has_continuation_marker",
    "_headline_overlap_metrics",
]
