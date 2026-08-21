"""Cross-Page Story Continuation and Jump Assembler.

Identifies and links articles that span multiple pages (e.g. stories that
start on Page 1 and continue on Page 4 via jump lines).

Constructs unified AssembledArticle objects and tracks exact per-page spatial
bounding box sequences for the `article_pages` junction table.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.ingestion.segmenter import SegmentedArticle

logger = get_logger(__name__)


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


def _headline_similarity(h1: str, h2: str) -> float:
    """Calculate token overlap Jaccard similarity between two headlines."""
    tokens1 = set(re.findall(r"\w+", h1.lower()))
    tokens2 = set(re.findall(r"\w+", h2.lower()))
    # Remove common stop words and continuation markers
    ignore = {"continued", "from", "page", "cont", "see", "to", "the", "a", "an", "in", "on", "of"}
    t1 = tokens1 - ignore
    t2 = tokens2 - ignore
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


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

        # Keep track of continuation segments that get absorbed
        absorbed_ids: set[str] = set()

        for page_num in sorted_pages:
            for art in pages_articles[page_num]:
                if art.article_temp_id in absorbed_ids:
                    continue

                # Initialize assembled article starting on this page
                current = AssembledArticle(
                    headline=art.headline,
                    subheadline=art.subheadline,
                    byline_author=art.byline_author,
                    full_text=art.body_text,
                    primary_page_number=page_num,
                    pages_mapping=[
                        PageBBoxMapping(
                            page_number=page_num,
                            bbox_list=art.bbox_list,
                            block_order=0,
                        )
                    ],
                )

                # Check if this article jumps to a subsequent page
                target_page = art.jump_to_page
                current_source_page = page_num
                block_order_counter = 1

                # Search subsequent pages for matching continuation
                for next_page in [p for p in sorted_pages if p > page_num]:
                    candidate_articles = pages_articles.get(next_page, [])
                    matched_continuation: SegmentedArticle | None = None

                    for cand in candidate_articles:
                        if cand.article_temp_id in absorbed_ids:
                            continue

                        # Case 1: Explicit jump target match
                        if target_page == next_page and (
                            cand.jump_from_page == current_source_page
                            or _headline_similarity(current.headline, cand.headline) >= 0.3
                        ):
                            matched_continuation = cand
                            break

                        # Case 2: Headline continuation without explicit target
                        if _headline_similarity(current.headline, cand.headline) >= 0.6:
                            matched_continuation = cand
                            break

                    if matched_continuation:
                        absorbed_ids.add(matched_continuation.article_temp_id)
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
                assembled.append(current)

        logger.info(
            "Issue cross-page assembly complete",
            extra={"total_assembled": len(assembled), "pages": len(sorted_pages)},
        )

        return assembled
