"""Docling Document Layout Analysis, Reading Order, Table Structure, and OCR Provider.

Provides deep neural multi-column layout detection, reading order linearization,
table matrix extraction, and document graph parsing natively from PDF bytes
and page images using IBM Docling.

Features:
- Standardized DocumentLayoutProvider & OCREngine protocol conformance.
- Dynamic hardware acceleration detection (Apple Silicon MPS -> NVIDIA CUDA -> CPU).
- Coordinate transformation from Docling points to top-left origin pixel / canvas coordinates.
- Table matrix JSON serialization (headers, rows, raw_markdown, raw_html) for SQL Analytics.
- Pre-linearized reading order preservation with Zero-Drop layout guarantees.
- Seamless fallback adapter for lightweight/unit-testing environments.
"""

from __future__ import annotations

import asyncio
import io
import re
from typing import Any

import pymupdf
from PIL import Image

from app.core.logging import get_logger
from app.providers.base import (
    DocumentLayoutProvider,
    ExtractedDocumentNode,
    ExtractedPhotoData,
    ExtractedTableData,
    MinerUParseResult,
    OCRBlock,
    OCREngine,
    OCRResult,
    ProviderCapability,
)

logger = get_logger(__name__)


def detect_device_mode() -> str:
    """Detect available hardware accelerator: cuda, mps, or cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def parse_markdown_table_to_matrix(
    md_text: str,
) -> tuple[list[str], list[list[str]]]:
    """Parse Markdown table text into headers and row matrix."""
    lines = [line.strip() for line in md_text.strip().split("\n") if line.strip()]
    if not lines:
        return [], []

    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if not table_lines:
        return [], []

    # First line is headers
    raw_headers = [c.strip() for c in table_lines[0].strip("|").split("|")]
    headers = [h for h in raw_headers if h]

    # Filter out divider lines (e.g. |---|---|)
    rows: list[list[str]] = []
    for line in table_lines[1:]:
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        row_cells = [c.strip() for c in line.strip("|").split("|")]
        if row_cells:
            rows.append(row_cells)

    return headers, rows


def parse_html_table_to_matrix(
    html_text: str,
) -> tuple[list[str], list[list[str]]]:
    """Parse basic HTML table text into headers and row matrix."""
    headers: list[str] = []
    rows: list[list[str]] = []

    th_matches = re.findall(r"<th[^>]*>(.*?)</th>", html_text, re.IGNORECASE | re.DOTALL)
    if th_matches:
        headers = [re.sub(r"<[^>]+>", "", th).strip() for th in th_matches]

    tr_matches = re.findall(r"<tr[^>]*>(.*?)</tr>", html_text, re.IGNORECASE | re.DOTALL)
    for tr in tr_matches:
        td_matches = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.IGNORECASE | re.DOTALL)
        if td_matches:
            row = [re.sub(r"<[^>]+>", "", td).strip() for td in td_matches]
            if any(cell for cell in row):
                rows.append(row)

    return headers, rows


class DoclingProvider(DocumentLayoutProvider, OCREngine):
    """Deep neural layout analysis, reading order, and table parsing provider using Docling."""

    def __init__(
        self,
        lang: str = "en",
        do_ocr: bool = False,
        do_table_structure: bool = True,
    ) -> None:
        self.lang = lang
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
        self.device = detect_device_mode()
        self._digital_converter: Any = None
        self._ocr_converter: Any = None
        self._converter_lock = asyncio.Lock()

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            supports_vision=True,
            supports_layout=True,
            supports_structured_output=True,
        )

    @property
    def provider_name(self) -> str:
        return "docling"

    def _get_converter(self, need_ocr: bool = False) -> Any:
        """Initialize or return the cached Docling DocumentConverter instance."""
        if need_ocr and self._ocr_converter is not None:
            return self._ocr_converter
        if not need_ocr and self._digital_converter is not None:
            return self._digital_converter

        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = need_ocr
            pipeline_options.do_table_structure = self.do_table_structure

            # Configure OCR language hints if OCR is requested
            ocr_opt = getattr(pipeline_options, "ocr_options", None)
            if need_ocr and ocr_opt:
                if self.lang == "hi":
                    ocr_opt.lang = ["hi", "en"]
                else:
                    ocr_opt.lang = ["en"]

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
            logger.info(
                "Docling DocumentConverter initialized successfully",
                extra={
                    "device": self.device,
                    "need_ocr": need_ocr,
                    "do_table_structure": self.do_table_structure,
                },
            )
            if need_ocr:
                self._ocr_converter = converter
            else:
                self._digital_converter = converter
            return converter
        except Exception as e:
            logger.warning(
                "Could not initialize neural Docling DocumentConverter (will use fallback)",
                extra={"error": str(e)},
            )
            return None

    def _convert_docling_document(
        self,
        pdf_bytes: bytes,
        need_ocr: bool = False,
    ) -> Any:
        """Synchronously convert PDF bytes to DoclingDocument."""
        converter = self._get_converter(need_ocr=need_ocr)
        if converter is None:
            return None

        from docling.datamodel.base_models import DocumentStream

        stream = DocumentStream(name="document.pdf", stream=io.BytesIO(pdf_bytes))
        conv_res = converter.convert(stream)
        return conv_res.document

    def _extract_nodes_from_docling_doc(
        self,
        doc: Any,
        page_no: int,
    ) -> tuple[list[ExtractedDocumentNode], str]:
        """Extract structured nodes and markdown from a single DoclingDocument page."""
        nodes: list[ExtractedDocumentNode] = []
        md_lines: list[str] = []

        page_item = doc.pages.get(page_no)
        page_height = (
            float(page_item.size.height)
            if page_item and hasattr(page_item, "size")
            else 1000.0
        )

        for idx, (item, _level) in enumerate(doc.iterate_items(page_no=page_no), start=1):
            label = str(getattr(item, "label", "")).lower()
            text = str(getattr(item, "text", "")).strip()

            # Determine node type and level
            node_type = "text"
            heading_level: int | None = None
            table_data: ExtractedTableData | None = None
            photo_data: ExtractedPhotoData | None = None

            if "title" in label or "header" in label:
                node_type = "title"
                heading_level = getattr(item, "level", 1)
            elif "table" in label:
                node_type = "table"
                raw_md = ""
                raw_html = ""
                if hasattr(item, "export_to_markdown"):
                    try:
                        raw_md = item.export_to_markdown(doc=doc)
                    except TypeError:
                        raw_md = item.export_to_markdown()
                    except Exception:
                        raw_md = ""
                if hasattr(item, "export_to_html"):
                    try:
                        raw_html = item.export_to_html(doc=doc)
                    except TypeError:
                        raw_html = item.export_to_html()
                    except Exception:
                        raw_html = ""

                headers, rows = parse_markdown_table_to_matrix(raw_md)
                if not headers and raw_html:
                    headers, rows = parse_html_table_to_matrix(raw_html)

                item_bbox = (0.0, 0.0, 0.0, 0.0)
                prov = getattr(item, "prov", [])
                if prov:
                    raw_b = prov[0].bbox
                    if hasattr(raw_b, "to_top_left_origin"):
                        tl = raw_b.to_top_left_origin(page_height)
                        item_bbox = (float(tl.l), float(tl.t), float(tl.r), float(tl.b))
                    else:
                        item_bbox = (float(raw_b.l), float(raw_b.t), float(raw_b.r), float(raw_b.b))

                table_data = ExtractedTableData(
                    bbox=item_bbox,
                    headers=headers,
                    rows=rows,
                    raw_markdown=raw_md or text,
                    raw_html=raw_html or None,
                )
            elif "picture" in label or "image" in label or "figure" in label:
                node_type = "image"
                caption_text: str | None = None
                captions = getattr(item, "captions", [])
                if captions:
                    try:
                        resolved_cap = captions[0].resolve(doc)
                        caption_text = getattr(resolved_cap, "text", None)
                    except Exception:
                        caption_text = None

                item_bbox = (0.0, 0.0, 0.0, 0.0)
                prov = getattr(item, "prov", [])
                if prov:
                    raw_b = prov[0].bbox
                    if hasattr(raw_b, "to_top_left_origin"):
                        tl = raw_b.to_top_left_origin(page_height)
                        item_bbox = (float(tl.l), float(tl.t), float(tl.r), float(tl.b))
                    else:
                        item_bbox = (float(raw_b.l), float(raw_b.t), float(raw_b.r), float(raw_b.b))

                photo_data = ExtractedPhotoData(
                    bbox=item_bbox,
                    caption=caption_text,
                )
            elif "caption" in label:
                node_type = "caption"

            # Extract bounding box
            node_bbox = (0.0, 0.0, 0.0, 0.0)
            prov = getattr(item, "prov", [])
            if prov:
                raw_b = prov[0].bbox
                if hasattr(raw_b, "to_top_left_origin"):
                    tl = raw_b.to_top_left_origin(page_height)
                    node_bbox = (float(tl.l), float(tl.t), float(tl.r), float(tl.b))
                elif hasattr(raw_b, "as_tuple"):
                    node_bbox = tuple(float(x) for x in raw_b.as_tuple()[:4])  # type: ignore[assignment]
                else:
                    node_bbox = (
                        float(getattr(raw_b, "l", 0.0)),
                        float(getattr(raw_b, "t", 0.0)),
                        float(getattr(raw_b, "r", 0.0)),
                        float(getattr(raw_b, "b", 0.0)),
                    )

            if text or table_data or photo_data:
                nodes.append(
                    ExtractedDocumentNode(
                        node_type=node_type,
                        text=text,
                        bbox=node_bbox,
                        reading_order=idx,
                        level=heading_level,
                        table_data=table_data,
                        photo_data=photo_data,
                    )
                )
                if text:
                    md_lines.append(text)

        page_md = "\n\n".join(md_lines)
        return nodes, page_md

    def _fallback_parse_pdf_bytes(
        self,
        pdf_bytes: bytes,
    ) -> list[MinerUParseResult]:
        """Fast, robust PyMuPDF fallback when Docling is offline or in testing."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        results: list[MinerUParseResult] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            text_page = page.get_text("dict")

            nodes: list[ExtractedDocumentNode] = []
            order_idx = 1
            md_lines: list[str] = []

            # Compute dominant font size
            font_sizes: list[float] = []
            for b in text_page.get("blocks", []):
                if b.get("type") == 0:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("text", "").strip():
                                font_sizes.append(float(span.get("size", 10.0)))

            dom_size = 10.0
            if font_sizes:
                rounded = [round(s, 1) for s in font_sizes]
                from collections import Counter

                dom_size = Counter(rounded).most_common(1)[0][0]

            for b in text_page.get("blocks", []):
                if b.get("type") == 0:  # Text
                    b_lines: list[str] = []
                    b_sizes: list[float] = []
                    for line in b.get("lines", []):
                        line_parts = [
                            s.get("text", "")
                            for s in line.get("spans", [])
                            if s.get("text", "").strip()
                        ]
                        if line_parts:
                            b_lines.append(" ".join(line_parts))
                            for s in line.get("spans", []):
                                b_sizes.append(float(s.get("size", 10.0)))

                    b_text = "\n".join(b_lines).strip()
                    if b_text:
                        mean_sz = sum(b_sizes) / len(b_sizes) if b_sizes else 10.0
                        is_title = mean_sz >= dom_size * 1.25
                        bbox = (
                            float(b["bbox"][0]),
                            float(b["bbox"][1]),
                            float(b["bbox"][2]),
                            float(b["bbox"][3]),
                        )
                        nodes.append(
                            ExtractedDocumentNode(
                                node_type="title" if is_title else "text",
                                text=b_text,
                                bbox=bbox,
                                reading_order=order_idx,
                                level=1 if is_title else None,
                            )
                        )
                        order_idx += 1
                        md_lines.append(b_text)
                elif b.get("type") == 1:  # Image
                    bbox = (
                        float(b["bbox"][0]),
                        float(b["bbox"][1]),
                        float(b["bbox"][2]),
                        float(b["bbox"][3]),
                    )
                    nodes.append(
                        ExtractedDocumentNode(
                            node_type="image",
                            text="",
                            bbox=bbox,
                            reading_order=order_idx,
                            photo_data=ExtractedPhotoData(bbox=bbox),
                        )
                    )
                    order_idx += 1

            results.append(
                MinerUParseResult(
                    page_number=page_num,
                    nodes=nodes,
                    markdown_content="\n\n".join(md_lines),
                    is_ocr_fallback=False,
                )
            )

        doc.close()
        return results

    async def parse_pdf_document(
        self,
        pdf_bytes: bytes,
        lang: str = "en",
        extract_tables: bool = True,
        need_ocr: bool = False,
    ) -> list[MinerUParseResult]:
        """Parse complete PDF document using Docling neural layout and reading order."""
        try:
            docling_doc = await asyncio.to_thread(
                self._convert_docling_document, pdf_bytes, need_ocr=need_ocr
            )
            if docling_doc is None or not getattr(docling_doc, "pages", {}):
                return self._fallback_parse_pdf_bytes(pdf_bytes)

            results: list[MinerUParseResult] = []
            for p_num in range(1, len(docling_doc.pages) + 1):
                nodes, page_md = self._extract_nodes_from_docling_doc(docling_doc, page_no=p_num)
                results.append(
                    MinerUParseResult(
                        page_number=p_num,
                        nodes=nodes,
                        markdown_content=page_md,
                        is_ocr_fallback=need_ocr,
                        ocr_confidence=1.0,
                    )
                )
            return results
        except Exception as e:
            logger.warning(
                "Docling parse_pdf_document failed, using spatial fallback",
                extra={"error": str(e)},
            )
            return self._fallback_parse_pdf_bytes(pdf_bytes)

    async def parse_page_image(
        self,
        image_bytes: bytes,
        page_number: int = 1,
        lang: str = "en",
    ) -> MinerUParseResult:
        """Parse a single page raster image by wrapping it in a temporary PDF."""
        try:
            # Wrap raster image into a 1-page PDF
            pil_img = Image.open(io.BytesIO(image_bytes))
            width_pt, height_pt = pil_img.size

            pdf_doc = pymupdf.open()
            rect = pymupdf.Rect(0, 0, width_pt, height_pt)
            pdf_page = pdf_doc.new_page(width=width_pt, height=height_pt)
            pdf_page.insert_image(rect, stream=image_bytes)

            single_pdf_bytes = pdf_doc.tobytes()
            pdf_doc.close()

            doc_results = await self.parse_pdf_document(single_pdf_bytes, lang=lang, need_ocr=True)
            if doc_results:
                res = doc_results[0]
                res.page_number = page_number
                return res
        except Exception as e:
            logger.warning(
                "Docling parse_page_image failed",
                extra={"page_number": page_number, "error": str(e)},
            )

        return MinerUParseResult(
            page_number=page_number,
            nodes=[],
            markdown_content="",
            is_ocr_fallback=True,
            ocr_confidence=0.0,
        )

    async def ocr(
        self,
        image_bytes: bytes,
        lang_hint: str | None = None,
    ) -> OCRResult:
        """Run OCR on a page image using Docling."""
        parse_res = await self.parse_page_image(
            image_bytes=image_bytes,
            page_number=1,
            lang=lang_hint or self.lang,
        )

        blocks: list[OCRBlock] = []
        for n in parse_res.nodes:
            if n.text and n.text.strip():
                blocks.append(
                    OCRBlock(
                        text=n.text,
                        bbox=n.bbox,
                        confidence=1.0,
                        language=lang_hint,
                    )
                )

        full_t = "\n\n".join(b.text for b in blocks)
        return OCRResult(
            blocks=blocks,
            full_text=full_t,
            mean_confidence=1.0 if blocks else 0.0,
            language=lang_hint,
        )
