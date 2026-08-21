"""MinerU (magic-pdf) Document Layout Analysis and OCR Provider.

Provides neural multi-column layout detection, reading order linear sequence
resolution, table matrix extraction, and document parsing natively from PDF bytes
and page images.

Features:
- Automatic `magic-pdf.json` configuration generation on startup.
- Dynamic hardware acceleration detection (CUDA -> MPS -> CPU).
- Table matrix JSON serialization (headers, rows, raw_markdown) for SQL Analytics.
- Unified support for both DocumentLayoutProvider and OCREngine protocols.
- Seamless fallback adapter for lightweight/unit-testing environments.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from app.core.logging import get_logger
from app.providers.base import (
    DocumentLayoutProvider,
    ExtractedDocumentNode,
    ExtractedPhotoData,
    ExtractedTableData,
    MinerUParseResult,
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
            rows.append(row)

    return headers, rows


class MinerUProvider(DocumentLayoutProvider, OCREngine):
    """Document layout, reading order, and neural OCR engine powered by MinerU / magic-pdf."""

    def __init__(
        self,
        models_dir: str | None = None,
        device_mode: str | None = None,
        method: str = "auto",
        lang: str = "en",
        config_path: str | None = None,
    ) -> None:
        self._method = method
        self._lang = lang
        self._device = device_mode or detect_device_mode()
        self._models_dir = (
            models_dir
            or os.getenv("MINERU_MODELS_DIR")
            or str(Path.home() / ".cache" / "mineru" / "models")
        )
        self._config_path = (
            config_path
            or os.getenv("MINERU_CONFIG_PATH")
            or str(Path.home() / "magic-pdf.json")
        )
        self._magic_pdf_available: bool | None = None
        self._ensure_magic_pdf_config()

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            supports_vision=True,
            supports_layout=True,
            supports_tool_use=False,
            supports_streaming=False,
            supports_structured_output=True,
            context_window=32768,
        )

    @property
    def provider_name(self) -> str:
        return "mineru"

    def _ensure_magic_pdf_config(self) -> None:
        """Initialize and validate the magic-pdf.json configuration file."""
        cfg_file = Path(self._config_path)
        if not cfg_file.exists():
            cfg_data = {
                "models-dir": self._models_dir,
                "device-mode": self._device,
                "table-config": {
                    "model": "TableMaster",
                    "is_table_recog_enable": True,
                    "max_time": 400,
                },
            }
            try:
                cfg_file.parent.mkdir(parents=True, exist_ok=True)
                cfg_file.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
                logger.info(
                    "Initialized MinerU magic-pdf.json configuration",
                    extra={"path": str(cfg_file), "device": self._device},
                )
            except Exception as e:
                logger.warning(
                    "Could not write magic-pdf.json config file",
                    extra={"error": str(e), "path": str(cfg_file)},
                )

    def _check_magic_pdf(self) -> bool:
        """Verify if magic-pdf library is available in environment."""
        if self._magic_pdf_available is None:
            try:
                import magic_pdf  # noqa: F401

                self._magic_pdf_available = True
            except ImportError:
                self._magic_pdf_available = False
                logger.info("magic-pdf not in environment; utilizing native MinerU adapter")
        return self._magic_pdf_available

    async def parse_pdf_document(
        self,
        pdf_bytes: bytes,
        lang: str = "en",
        extract_tables: bool = True,
    ) -> list[MinerUParseResult]:
        """Parse complete PDF document into structured reading-order nodes."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._parse_pdf_sync(pdf_bytes, lang, extract_tables),
        )

    def _parse_pdf_sync(
        self,
        pdf_bytes: bytes,
        lang: str = "en",
        extract_tables: bool = True,
    ) -> list[MinerUParseResult]:
        """Synchronous implementation of PDF parsing with MinerU / Adapter."""
        if self._check_magic_pdf():
            try:
                return self._parse_with_magic_pdf(
                    pdf_bytes,
                    lang=lang,
                    extract_tables=extract_tables,
                )
            except Exception as e:
                logger.warning(
                    "magic-pdf execution raised error; falling back to resilient adapter",
                    extra={"error": str(e)},
                )

        return self._parse_with_native_adapter(pdf_bytes, lang=lang, extract_tables=extract_tables)

    def _parse_with_magic_pdf(
        self,
        pdf_bytes: bytes,
        lang: str = "en",
        extract_tables: bool = True,
    ) -> list[MinerUParseResult]:
        """Execute parsing using official magic-pdf UNIPipe."""
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.pipe.UNIPipe import UNIPipe

        temp_image_dir = "/tmp/mineru_images"
        os.makedirs(temp_image_dir, exist_ok=True)
        image_writer = FileBasedDataWriter(temp_image_dir)
        jso_useful_key = {"_pdf_type": "", "model_list": []}

        pipe = UNIPipe(
            pdf_bytes=pdf_bytes,
            jso_useful_key=jso_useful_key,
            image_writer=image_writer,
            is_debug=False,
            lang=lang or self._lang,
            table_enable=extract_tables,
        )
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()

        content_list = pipe.pipe_mk_uni_format("images", drop_mode="none")
        md_content = pipe.pipe_mk_markdown("images", drop_mode="none")

        results: list[MinerUParseResult] = []
        pages_dict: dict[int, list[ExtractedDocumentNode]] = {}

        for item in content_list:
            page_idx = int(item.get("page_idx", 0))
            category = item.get("type", "text")
            raw_text = item.get("text", "") or ""
            bbox_raw = item.get("bbox", [0, 0, 0, 0])
            bbox = (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )

            table_data: ExtractedTableData | None = None
            if category == "table" or "table" in item:
                html = item.get("html") or ""
                headers, rows = parse_html_table_to_matrix(html)
                if not headers and raw_text:
                    headers, rows = parse_markdown_table_to_matrix(raw_text)
                table_data = ExtractedTableData(
                    bbox=bbox,
                    headers=headers,
                    rows=rows,
                    raw_markdown=raw_text,
                    raw_html=html if html else None,
                )

            photo_data: ExtractedPhotoData | None = None
            if category in ("image", "photo"):
                caption = item.get("caption") or item.get("img_caption")
                photo_data = ExtractedPhotoData(bbox=bbox, caption=caption)

            node = ExtractedDocumentNode(
                node_type=category,
                text=raw_text,
                bbox=bbox,
                reading_order=len(pages_dict.get(page_idx, [])),
                level=item.get("level"),
                table_data=table_data,
                photo_data=photo_data,
            )

            pages_dict.setdefault(page_idx, []).append(node)

        for p_idx, nodes in pages_dict.items():
            results.append(
                MinerUParseResult(
                    page_number=p_idx + 1,
                    nodes=nodes,
                    markdown_content=md_content,
                    is_ocr_fallback=False,
                )
            )

        return results

    def _parse_with_native_adapter(
        self,
        pdf_bytes: bytes,
        lang: str = "en",
        extract_tables: bool = True,
    ) -> list[MinerUParseResult]:
        """Resilient native adapter producing full MinerU-conforming structured nodes."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        results: list[MinerUParseResult] = []

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            nodes: list[ExtractedDocumentNode] = []
            rect = page.rect
            width, height = rect.width, rect.height

            # Extract raw blocks
            blocks = page.get_text("blocks")
            order_idx = 0

            for b in blocks:
                x0, y0, x1, y1, text, block_no, b_type = b
                clean_text = text.strip()
                if not clean_text:
                    continue

                bbox = (float(x0), float(y0), float(x1), float(y1))

                # Check for table structure in text
                is_table = "|" in clean_text or (
                    "\t" in clean_text and len(clean_text.split("\n")) > 2
                )
                if is_table and extract_tables:
                    headers, rows = parse_markdown_table_to_matrix(clean_text)
                    if headers or len(rows) > 1:
                        table_data = ExtractedTableData(
                            bbox=bbox,
                            headers=headers,
                            rows=rows,
                            raw_markdown=clean_text,
                        )
                        nodes.append(
                            ExtractedDocumentNode(
                                node_type="table",
                                text=clean_text,
                                bbox=bbox,
                                reading_order=order_idx,
                                table_data=table_data,
                            )
                        )
                        order_idx += 1
                        continue

                # Check for headings / banner titles
                lines = clean_text.split("\n")
                is_title = False
                level: int | None = None
                if (
                    len(lines) <= 2
                    and len(clean_text) < 150
                    and ((y0 < height * 0.25 and (x1 - x0) > width * 0.5) or clean_text.isupper())
                ):
                    is_title = True
                    level = 1 if (x1 - x0) > width * 0.6 else 2

                if is_title:
                    nodes.append(
                        ExtractedDocumentNode(
                            node_type="title",
                            text=clean_text,
                            bbox=bbox,
                            reading_order=order_idx,
                            level=level or 2,
                        )
                    )
                else:
                    nodes.append(
                        ExtractedDocumentNode(
                            node_type="text",
                            text=clean_text,
                            bbox=bbox,
                            reading_order=order_idx,
                        )
                    )
                order_idx += 1

            # Extract photos / images on page
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    img_rects = page.get_image_rects(xref)
                    for r in img_rects:
                        img_bbox = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
                        photo_data = ExtractedPhotoData(bbox=img_bbox, caption=None)
                        nodes.append(
                            ExtractedDocumentNode(
                                node_type="image",
                                text="",
                                bbox=img_bbox,
                                reading_order=order_idx,
                                photo_data=photo_data,
                            )
                        )
                        order_idx += 1
                except Exception:
                    pass

            results.append(
                MinerUParseResult(
                    page_number=page_num,
                    nodes=nodes,
                    markdown_content="\n\n".join(n.text for n in nodes if n.text),
                    is_ocr_fallback=False,
                )
            )

        return results

    async def parse_page_image(
        self,
        image_bytes: bytes,
        page_number: int = 1,
        lang: str = "en",
    ) -> MinerUParseResult:
        """Parse a single page raster image."""
        loop = asyncio.get_event_loop()

        def _img_to_res() -> MinerUParseResult:
            img = Image.open(io.BytesIO(image_bytes))
            pdf_bytes_io = io.BytesIO()
            img.convert("RGB").save(pdf_bytes_io, format="PDF")
            parsed_doc = self._parse_pdf_sync(pdf_bytes_io.getvalue(), lang=lang)
            if parsed_doc and parsed_doc[0].nodes:
                res = parsed_doc[0]
                res.page_number = page_number
                return res
            return MinerUParseResult(page_number=page_number, nodes=[])

        parsed = await loop.run_in_executor(None, _img_to_res)
        has_text = any(bool(n.text and n.text.strip()) for n in parsed.nodes)
        if not parsed.nodes or not has_text:
            # Fallback to OCR for pure raster / scanned images
            ocr_res = await self.ocr(image_bytes, lang_hint=lang)
            nodes: list[ExtractedDocumentNode] = []
            for idx, blk in enumerate(ocr_res.blocks):
                nodes.append(
                    ExtractedDocumentNode(
                        node_type="text",
                        text=blk.text,
                        bbox=blk.bbox,
                        reading_order=idx,
                    )
                )
            parsed = MinerUParseResult(
                page_number=page_number,
                nodes=nodes,
                markdown_content=ocr_res.full_text,
                is_ocr_fallback=True,
                ocr_confidence=ocr_res.mean_confidence,
            )

        return parsed

    async def ocr(
        self,
        image_bytes: bytes,
        lang_hint: str | None = None,
    ) -> OCRResult:
        """Execute OCR on an image conforming to OCREngine protocol."""
        from app.providers.tesseract_ocr import TesseractOCR

        lang = lang_hint or self._lang
        tess_lang = "eng" if "en" in lang else ("hin" if "hi" in lang else lang)
        tesseract = TesseractOCR(lang=tess_lang)
        return await tesseract.ocr(image_bytes, lang_hint=lang_hint)
