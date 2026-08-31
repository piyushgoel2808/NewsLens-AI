"""Docling-based Neural Document Layout & Article Parser for NewsLens-AI.

Replaces heuristic 1D reading-order segmenter with Docling's deep layout vision
model (DocLayNet + RapidOCR/PaddleOCR). Operates directly on broadsheet pages,
preserving column boundaries and generating isolated, cleanly segmented articles.
"""

from __future__ import annotations

import asyncio
import io
import re
from dataclasses import dataclass
from typing import Any

import pymupdf
from docling.datamodel.base_models import DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter
from docling_core.types.doc import CoordOrigin

from app.core.logging import get_logger
from app.ingestion.detector import is_text_gibberish
from app.ingestion.layout_analyzer import (
    clean_ocr_text_artifacts,
    is_syndication_or_agency_slug,
)
from app.ingestion.segmenter import (
    SECTION_HEADER_BLACKLIST,
    SegmentedArticle,
    extract_kicker_and_clean_headline,
    is_valid_headline_candidate,
)
from app.providers.base import (
    DocumentLayoutProvider,
    ExtractedDocumentNode,
    ExtractedPhotoData,
    ExtractedTableData,
    MinerUParseResult,
    ProviderCapability,
)

logger = get_logger(__name__)


class CorruptedPdfTextLayerError(Exception):
    """Raised when Docling output contains corrupted font CMaps or replacement characters (\ufffd)."""
    pass

# Boilerplate tokens to reject as article headlines
_PAGE_HEADER_KEYWORDS = {
    "PAGE", "VOL", "NO", "EDITION", "NEW DELHI", "MUMBAI", "BENGALURU",
    "THE ECONOMIC TIMES", "ECONOMIC TIMES", "THE TRIBUNE", "MINT",
    "THE HINDU", "THE TIMES OF INDIA", "BUSINESS STANDARD",
}

# Regex matching author names (e.g. "Manu Pubby", "Dipanjan Roy Chaudhury", "Anubhuti Vishnoi")
_AUTHOR_NAME_PATTERN = re.compile(
    r"^(?:By\s+)?[A-Z][a-z]+(?:\s+[A-Z]\.?)*(?:\s+[A-Z][a-z]+){1,2}$"
)

# Standard Indian newspaper dateline prefixes (e.g. "New Delhi:", "Mumbai:", "Bengaluru:")
_DATELINE_PATTERN = re.compile(
    r"^(?:New Delhi|Mumbai|Bengaluru|Kolkata|Chennai|Hyderabad|Ahmedabad|Pune|Jaipur|Lucknow|"
    r"Chandigarh|Patna|Bhopal|Srinagar|Jammu|Washington|London|Beijing|Moscow|Tokyo)\s*:",
    re.IGNORECASE,
)


@dataclass
class DoclingParsedItem:
    """Intermediate parsed item from Docling document."""

    label: str  # 'title', 'section_header', 'text', 'table', 'picture', 'caption', 'list_item'
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in top-left pixel space
    page_number: int = 1
    level: int = 1
    table_data: ExtractedTableData | None = None
    photo_data: ExtractedPhotoData | None = None


class DoclingLayoutParser(DocumentLayoutProvider):
    """Deep Layout Parser powered by Docling for broadsheet newspaper pages."""

    def __init__(self, do_ocr: bool = True, do_table_structure: bool = True) -> None:
        self._options = PdfPipelineOptions()
        self._options.do_ocr = do_ocr
        self._options.do_table_structure = do_table_structure
        self._converter = DocumentConverter()

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            supports_vision=True,
            supports_tool_use=False,
            supports_streaming=False,
            supports_structured_output=True,
            context_window=32768,
        )

    @property
    def provider_name(self) -> str:
        return "docling"

    def _convert_bbox_to_pixels(
        self,
        bbox: Any,
        page_w_pts: float,
        page_h_pts: float,
        width_px: int,
        height_px: int,
    ) -> tuple[float, float, float, float]:
        """Convert a Docling BoundingBox to top-left pixel space (x0, y0, x1, y1)."""
        if not bbox:
            return (0.0, 0.0, float(width_px), float(height_px))

        orig = getattr(bbox, "coord_origin", CoordOrigin.BOTTOMLEFT)
        l, t, r, b = bbox.l, bbox.t, bbox.r, bbox.b

        if orig == CoordOrigin.BOTTOMLEFT:
            y0_pts = page_h_pts - t
            y1_pts = page_h_pts - b
            x0_pts = l
            x1_pts = r
        else:
            x0_pts = l
            y0_pts = t
            x1_pts = r
            y1_pts = b

        scale_x = width_px / page_w_pts if page_w_pts > 0 else 1.0
        scale_y = height_px / page_h_pts if page_h_pts > 0 else 1.0

        x0 = max(0.0, min(float(width_px), x0_pts * scale_x))
        y0 = max(0.0, min(float(height_px), y0_pts * scale_y))
        x1 = max(0.0, min(float(width_px), x1_pts * scale_x))
        y1 = max(0.0, min(float(height_px), y1_pts * scale_y))

        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def _is_header_or_footer_noise(
        self,
        text: str,
        bbox: tuple[float, float, float, float],
        height_px: int,
    ) -> bool:
        """Check if an element is running header/masthead or footer folio noise."""
        y0, y1 = bbox[1], bbox[3]
        clean = text.strip().upper()

        if y1 <= height_px * 0.07:
            if any(kw in clean for kw in _PAGE_HEADER_KEYWORDS) or len(clean.split()) <= 6:
                return True

        # Enhanced masthead banner detection on front/wrap pages up to 20% page height
        if y1 <= height_px * 0.20:
            has_masthead_brand = any(kw in clean for kw in _PAGE_HEADER_KEYWORDS)
            has_epaper_or_url = any(u in clean for u in ("LIVEMINT", "EPAPER", ".COM", "WWW."))
            has_date_marker = bool(
                re.search(r"\b(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\b", clean)
                and re.search(r"\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|\d{4})\b", clean)
            )
            is_printer_mark = clean in ("SIHT", "CMYK", "A ND-NDE")
            if (has_masthead_brand and (has_epaper_or_url or has_date_marker)) or is_printer_mark:
                return True
            if clean in _PAGE_HEADER_KEYWORDS:
                return True

        if y0 >= height_px * 0.95:
            if re.search(r"\b(?:PAGE\s*\d+|\d+\s*\|\s*[A-Z]+)\b", clean) or len(clean.split()) <= 4:
                return True

        return False

    def parse_docling_document(
        self,
        pdf_bytes: bytes,
        page_number: int,
        width_px: int,
        height_px: int,
    ) -> list[DoclingParsedItem]:
        """Convert a single-page PDF slice using Docling and extract parsed items."""
        stream = DocumentStream(name=f"page_{page_number}.pdf", stream=io.BytesIO(pdf_bytes))
        conv_res = self._converter.convert(stream)
        docling_doc = conv_res.document

        try:
            pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            p_rect = pdf_doc[0].rect
            page_w_pts = float(p_rect.width)
            page_h_pts = float(p_rect.height)
            pdf_doc.close()
        except Exception:
            page_w_pts = 595.0
            page_h_pts = 842.0

        items: list[DoclingParsedItem] = []
        visited_text_ids: set[int] = set()

        def _process_text_item(t_item: Any, lvl: int) -> DoclingParsedItem | None:
            t_id = id(t_item)
            if t_id in visited_text_ids:
                return None
            visited_text_ids.add(t_id)

            t_text = clean_ocr_text_artifacts(getattr(t_item, "text", "") or "")
            if not t_text:
                return None

            t_bbox_obj = t_item.prov[0].bbox if getattr(t_item, "prov", None) and len(t_item.prov) > 0 else None
            t_bbox_px = self._convert_bbox_to_pixels(
                bbox=t_bbox_obj,
                page_w_pts=page_w_pts,
                page_h_pts=page_h_pts,
                width_px=width_px,
                height_px=height_px,
            )

            if self._is_header_or_footer_noise(t_text, t_bbox_px, height_px):
                return None

            t_label = t_item.label.value if hasattr(t_item.label, "value") else str(t_item.label)
            return DoclingParsedItem(
                label=t_label,
                text=t_text,
                bbox=t_bbox_px,
                page_number=page_number,
                level=lvl,
            )

        for item, level in docling_doc.iterate_items():
            raw_label = item.label.value if hasattr(item.label, "value") else str(item.label)
            text = clean_ocr_text_artifacts(getattr(item, "text", "") or "")

            bbox_obj = item.prov[0].bbox if item.prov and len(item.prov) > 0 else None
            bbox_px = self._convert_bbox_to_pixels(
                bbox=bbox_obj,
                page_w_pts=page_w_pts,
                page_h_pts=page_h_pts,
                width_px=width_px,
                height_px=height_px,
            )

            if self._is_header_or_footer_noise(text, bbox_px, height_px):
                continue

            # If this is a picture, include the picture node and resolve any nested child texts
            if raw_label == "picture":
                items.append(
                    DoclingParsedItem(
                        label=raw_label,
                        text="",
                        bbox=bbox_px,
                        page_number=page_number,
                        level=level,
                    )
                )
                for child_ref in getattr(item, "children", []):
                    try:
                        cref_str = getattr(child_ref, "cref", "")
                        if cref_str.startswith("#/texts/") and hasattr(docling_doc, "texts"):
                            idx = int(cref_str.split("/")[-1])
                            child_text_item = docling_doc.texts[idx]
                            parsed_child = _process_text_item(child_text_item, lvl=level + 1)
                            if parsed_child:
                                items.append(parsed_child)
                    except Exception:
                        pass
                continue

            if not text and raw_label != "table":
                continue

            visited_text_ids.add(id(item))

            tbl_data: ExtractedTableData | None = None
            if raw_label == "table":
                tbl_text = getattr(item, "text", "") or ""
                tbl_data = ExtractedTableData(
                    bbox=bbox_px,
                    headers=[],
                    rows=[],
                    raw_markdown=tbl_text,
                )

            items.append(
                DoclingParsedItem(
                    label=raw_label,
                    text=text,
                    bbox=bbox_px,
                    page_number=page_number,
                    level=level,
                    table_data=tbl_data,
                )
            )

        # Secondary safety net: check if any texts in docling_doc.texts were unvisited
        for t_item in getattr(docling_doc, "texts", []):
            parsed_t = _process_text_item(t_item, lvl=1)
            if parsed_t:
                items.append(parsed_t)

        # Integrity Validation: Detect corrupted embedded font CMaps / \ufffd dominance
        all_text = " ".join(it.text for it in items if it.text)
        if all_text.strip():
            num_replacement = sum(1 for c in all_text if c in ("\ufffd", "\ufeff"))
            replacement_ratio = num_replacement / max(len(all_text.replace(" ", "")), 1)
            if replacement_ratio >= 0.03 or is_text_gibberish(all_text):
                logger.warning(
                    "Docling parsed text contains corrupted font artifacts / replacement characters",
                    extra={
                        "page_number": page_number,
                        "replacement_chars": num_replacement,
                        "replacement_ratio": round(replacement_ratio, 4),
                    },
                )
                raise CorruptedPdfTextLayerError(
                    f"Page {page_number} text layer is corrupted with {num_replacement} replacement characters "
                    f"({replacement_ratio:.1%}). Requires pure Image OCR fallback."
                )

        return items

    def assemble_articles(
        self,
        page_number: int,
        items: list[DoclingParsedItem],
        width_px: int,
        height_px: int,
        is_advertisement_page: bool = False,
    ) -> list[SegmentedArticle]:
        """Assemble Docling parsed items into isolated, non-jumbled SegmentedArticle units."""
        if not items:
            return []

        if is_advertisement_page:
            all_text = "\n\n".join(it.text for it in items if it.text)
            return [
                SegmentedArticle(
                    article_temp_id=f"p{page_number}_ad_full",
                    headline="[Advertisement] Full Page Notice",
                    body_text=all_text,
                    word_count=len(all_text.split()),
                    bbox_list=[(0.0, 0.0, float(width_px), float(height_px))],
                )
            ]

        articles: list[SegmentedArticle] = []
        active_page_section: str | None = None
        current_headline: str = ""
        current_subheadline: str | None = None
        current_byline: str | None = None
        current_body_parts: list[str] = []
        current_bboxes: list[tuple[float, float, float, float]] = []
        art_counter = 0

        def _flush_current_article() -> None:
            nonlocal current_headline, current_subheadline, current_byline
            nonlocal current_body_parts, current_bboxes, art_counter, active_page_section

            if not current_headline and not current_body_parts:
                return

            body_str = "\n\n".join(current_body_parts).strip()
            if current_subheadline:
                body_str = f"{current_subheadline}\n\n{body_str}".strip() if body_str else current_subheadline

            h_text = current_headline
            if not h_text and body_str:
                sentences = [s.strip() for s in body_str.split("\n") if s.strip()]
                if sentences:
                    # If first line is a very short brand/kicker and second line is a substantive headline
                    if len(sentences[0].split()) <= 2 and len(sentences) > 1 and len(sentences[1].split()) >= 3:
                        h_text = sentences[1]
                        remaining = [sentences[0]] + sentences[2:]
                        body_str = "\n".join(remaining).strip()
                    elif len(sentences[0].split()) <= 15:
                        h_text = sentences[0]
                        body_str = "\n".join(sentences[1:]).strip()
                    else:
                        h_text = sentences[0][:80] + "..."

            if not h_text:
                return

            kicker, clean_h = extract_kicker_and_clean_headline(h_text)
            final_h = clean_h or h_text

            # Discard headlines that are predominantly replacement characters (\ufffd) or gibberish
            if "\ufffd" in final_h or is_text_gibberish(final_h):
                return

            full_text = f"{final_h}\n\n{body_str}".strip() if final_h != body_str else body_str
            w_count = len(full_text.split())

            if w_count >= 10:
                art_counter += 1
                articles.append(
                    SegmentedArticle(
                        article_temp_id=f"page_{page_number}_art_{art_counter}",
                        headline=final_h,
                        subheadline=current_subheadline,
                        byline_author=current_byline,
                        body_text=body_str or final_h,
                        word_count=w_count,
                        bbox_list=list(current_bboxes),
                        section=active_page_section,
                        printed_section=active_page_section,
                    )
                )

            current_headline = ""
            current_subheadline = None
            current_byline = None
            current_body_parts = []
            current_bboxes = []

        total_items = len(items)
        last_picture_bbox: tuple[float, float, float, float] | None = None

        for idx, item in enumerate(items):
            lbl = item.label
            txt = item.text.strip()
            bbox = item.bbox

            if not txt and lbl not in ("picture", "table"):
                continue

            # Teaser strip boundary check:
            # If previous items are from a top teaser strip (containing page pointers) and there is a large vertical gap, flush them.
            if current_bboxes and (bbox[1] - current_bboxes[-1][3] > height_px * 0.07):
                if any(re.search(r"(?:▶|►|>|->)?\s*P\d{1,2}\b", p) for p in current_body_parts):
                    _flush_current_article()

            next_txt = items[idx + 1].text.strip() if idx + 1 < total_items else ""

            # Check if this element is an author byline (e.g. "BY LINDA QIU", "BY REIS THEBAULT", "Manu Pubby")
            byline_inline_match = re.search(r"(?i)\bBY\s+([A-Z\s\.\-]{3,50})$", txt)
            is_author_byline = bool(
                is_syndication_or_agency_slug(txt)
                or _AUTHOR_NAME_PATTERN.match(txt)
                or (len(txt.split()) <= 5 and any(b_tag in txt for b_tag in ["Bureau", "Reporter", "Correspondent", "Special", "By"]))
            )

            # Check if this element or next element has a dateline (e.g. "New Delhi:", "WASHINGTON -")
            has_following_dateline = bool(_DATELINE_PATTERN.match(next_txt) or re.match(r"^[A-Z\s]{3,20}\s*[-–—:]", next_txt))

            # 1. Direct Author Byline handling
            if is_author_byline or (has_following_dateline and len(txt.split()) <= 4 and not txt.isupper()):
                clean_byline = re.sub(r"(?i)^by\s+", "", txt).strip()
                current_byline = clean_byline
                current_bboxes.append(bbox)
                continue

            # 2. Section Header or Title
            if lbl in ("section_header", "title"):
                # Check for running page section header banner (e.g. "SPORTS WORLD PLAY", "ET MARKETS", "OPINION")
                is_section_banner = bool(
                    txt.upper() in SECTION_HEADER_BLACKLIST
                    or txt.upper() in _PAGE_HEADER_KEYWORDS
                    or (bbox[1] < height_px * 0.18 and len(txt.split()) <= 5 and any(
                        s in txt.lower() for s in ["sport", "play", "market", "corporate", "economy", "opinion", "national", "world", "editorial", "science", "tech", "entertainment", "life"]
                    ))
                )
                if is_section_banner:
                    if not active_page_section and len(txt.split()) <= 6:
                        active_page_section = txt.title()
                    continue

                # If byline is embedded at the end of the text (e.g. "... BY LINDA QIU")
                if byline_inline_match:
                    found_byline = byline_inline_match.group(1).strip().title()
                    current_byline = found_byline
                    txt = txt[:byline_inline_match.start()].strip()

                # Short internal all-caps subheads (e.g. "STRONG FOUNDATION", "DIPLOMATIC CHANNELS")
                # When an article is already active and has body content, coalesce as an internal sub-heading
                is_internal_subhead = bool(
                    current_headline
                    and len(current_body_parts) > 0
                    and (txt.isupper() or len(txt.split()) <= 3)
                    and len(txt.split()) <= 4
                    and not has_following_dateline
                )

                if is_internal_subhead:
                    current_body_parts.append(f"\n### {txt}\n")
                    current_bboxes.append(bbox)
                    continue

                # Headline & Subheadline / Deck Coalescence:
                # If current_headline is already active BUT has NO body paragraphs yet:
                if current_headline and not current_body_parts:
                    # Check if this new text is a deck/subheadline (a 1-3 line summary sentence expanding on the headline)
                    is_deck = bool(
                        (not current_subheadline and len(txt.split()) >= 4 and not txt.isupper())
                        or (len(txt.split()) > len(current_headline.split()) and not txt.isupper())
                        or byline_inline_match
                    )
                    if is_deck:
                        current_subheadline = txt
                        current_bboxes.append(bbox)
                        continue
                    else:
                        # If previous headline was a very short kicker (1-3 words), swap it
                        if len(current_headline.split()) <= 3:
                            current_subheadline = current_headline
                            current_headline = txt
                            current_bboxes.append(bbox)
                            continue
                        else:
                            # Genuine separate headline on the page
                            _flush_current_article()
                            current_headline = txt
                            current_bboxes.append(bbox)
                            continue

                # If current_headline already has body paragraphs, encountering a valid headline candidate starts a new article!
                if is_valid_headline_candidate(txt):
                    _flush_current_article()
                    current_headline = txt
                    current_bboxes.append(bbox)
                    continue
                else:
                    # Subheadline or leading summary sentence
                    if current_headline and not current_subheadline and len(txt.split()) <= 30:
                        current_subheadline = txt
                    else:
                        current_body_parts.append(txt)
                    current_bboxes.append(bbox)

            # 3. Body paragraphs & text
            elif lbl in ("text", "paragraph", "list_item"):
                # Check for byline at start or end of paragraph
                if byline_inline_match:
                    current_byline = byline_inline_match.group(1).strip().title()
                    txt = txt[:byline_inline_match.start()].strip()

                lines = [l.strip() for l in txt.split("\n") if l.strip()]
                if lines and (is_syndication_or_agency_slug(lines[0]) or _AUTHOR_NAME_PATTERN.match(lines[0]) or re.match(r"(?i)^by\s+[A-Z]", lines[0])):
                    current_byline = re.sub(r"(?i)^by\s+", "", lines[0]).strip()
                    lines = lines[1:]
                    txt = "\n".join(lines)

                if txt:
                    # If we don't have a subheadline yet, have no body yet, and this paragraph is short/sentence-like without dateline
                    if current_headline and not current_subheadline and not current_body_parts and len(txt.split()) <= 30 and not _DATELINE_PATTERN.match(txt):
                        current_subheadline = txt
                    else:
                        current_body_parts.append(txt)
                    current_bboxes.append(bbox)

            # 4. Tables & Charts
            elif lbl in ("table", "chart"):
                if txt:
                    current_body_parts.append(f"[Table / Data]:\n{txt}")
                    current_bboxes.append(bbox)

            # 5. Captions & Pictures
            elif lbl == "picture":
                last_picture_bbox = bbox
                current_bboxes.append(bbox)

            elif lbl == "caption":
                if txt:
                    current_body_parts.append(f"[Photo Caption]: {txt}")
                    current_bboxes.append(bbox)

        _flush_current_article()
        return articles

    def extract_page_media_items(
        self,
        items: list[DoclingParsedItem],
    ) -> list[ExtractedPhotoData]:
        """Extract all pictures and their associated captions from parsed Docling items using 2D spatial matching."""
        pictures: list[tuple[float, float, float, float]] = []
        captions: list[tuple[tuple[float, float, float, float], str]] = []

        for item in items:
            lbl = item.label.lower()
            if lbl in ("picture", "figure", "image", "chart", "diagram"):
                if item.bbox and (item.bbox[2] - item.bbox[0] >= 30) and (item.bbox[3] - item.bbox[1] >= 30):
                    pictures.append(item.bbox)
            elif lbl == "caption" and item.text.strip():
                captions.append((item.bbox, item.text.strip()))

        photos: list[ExtractedPhotoData] = []
        for p_box in pictures:
            px0, py0, px1, py1 = p_box
            pw = max(px1 - px0, 1.0)

            # Find best matching caption for this picture
            best_caption = ""
            best_score = float("inf")

            for c_box, c_text in captions:
                cx0, cy0, cx1, cy1 = c_box
                # Check horizontal overlap with caption
                h_overlap = max(0.0, min(px1, cx1) - max(px0, cx0))
                # Check vertical distance
                if cy0 >= py1:
                    v_dist = cy0 - py1  # caption is below photo
                elif cy1 <= py0:
                    v_dist = py0 - cy1  # caption is above photo
                else:
                    v_dist = 0.0  # overlapping vertically

                # Proximity penalty: allow up to 600px vertical distance for large hero images
                if v_dist <= 600.0:
                    score = v_dist - (h_overlap / pw * 100.0)
                    if score < best_score:
                        best_score = score
                        best_caption = c_text

            photos.append(ExtractedPhotoData(bbox=p_box, caption=best_caption))

        return photos

    async def parse_page(
        self,
        pdf_bytes: bytes,
        page_number: int,
        width_px: int,
        height_px: int,
        is_advertisement_page: bool = False,
    ) -> list[SegmentedArticle]:
        """Asynchronously extract and segment articles for a single page."""
        loop = asyncio.get_running_loop()
        parsed_items = await loop.run_in_executor(
            None,
            self.parse_docling_document,
            pdf_bytes,
            page_number,
            width_px,
            height_px,
        )
        return self.assemble_articles(
            page_number=page_number,
            items=parsed_items,
            width_px=width_px,
            height_px=height_px,
            is_advertisement_page=is_advertisement_page,
        )

    async def parse_pdf_document(
        self,
        pdf_bytes: bytes,
        lang: str = "en",
        extract_tables: bool = True,
    ) -> list[MinerUParseResult]:
        """Protocol implementation: Parse complete PDF document."""
        loop = asyncio.get_running_loop()

        def _do_parse() -> list[MinerUParseResult]:
            src_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            results: list[MinerUParseResult] = []

            for p_idx in range(len(src_doc)):
                page = src_doc[p_idx]
                p_rect = page.rect
                single_doc = pymupdf.open()
                single_doc.insert_pdf(src_doc, from_page=p_idx, to_page=p_idx)
                p_bytes = single_doc.tobytes()
                single_doc.close()

                items = self.parse_docling_document(
                    pdf_bytes=p_bytes,
                    page_number=p_idx + 1,
                    width_px=int(p_rect.width),
                    height_px=int(p_rect.height),
                )
                nodes = [
                    ExtractedDocumentNode(
                        node_type=it.label,
                        text=it.text,
                        bbox=it.bbox,
                        reading_order=i + 1,
                        level=it.level,
                        table_data=it.table_data,
                    )
                    for i, it in enumerate(items)
                ]
                results.append(
                    MinerUParseResult(
                        page_number=p_idx + 1,
                        nodes=nodes,
                        markdown_content="\n\n".join(it.text for it in items),
                    )
                )
            src_doc.close()
            return results

        return await loop.run_in_executor(None, _do_parse)

    async def parse_page_image(
        self,
        image_bytes: bytes,
        page_number: int = 1,
        lang: str = "en",
    ) -> MinerUParseResult:
        """Protocol implementation: Parse single page image by wrapping into a PDF."""
        img_doc = pymupdf.open()
        img_pdf_bytes = img_doc.convert_to_pdf(image_bytes)
        img_doc.close()

        pdf_doc = pymupdf.open("pdf", img_pdf_bytes)
        rect = pdf_doc[0].rect
        width_px = int(rect.width)
        height_px = int(rect.height)
        pdf_doc.close()

        items = await asyncio.get_running_loop().run_in_executor(
            None,
            self.parse_docling_document,
            img_pdf_bytes,
            page_number,
            width_px,
            height_px,
        )
        nodes = [
            ExtractedDocumentNode(
                node_type=it.label,
                text=it.text,
                bbox=it.bbox,
                reading_order=i + 1,
                level=it.level,
                table_data=it.table_data,
            )
            for i, it in enumerate(items)
        ]
        return MinerUParseResult(
            page_number=page_number,
            nodes=nodes,
            markdown_content="\n\n".join(it.text for it in items),
        )
