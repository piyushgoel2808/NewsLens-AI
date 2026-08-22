"""Tesseract OCR engine via pytesseract.

Implements OCREngine. Runs synchronous pytesseract calls in a thread pool
executor to avoid blocking the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.providers.base import OCRBlock, OCRResult, ProviderError

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ISO 639-1 code → Tesseract language code mapping
_LANG_MAP: dict[str, str] = {
    "en": "eng",
    "hi": "hin",
    "bn": "ben",
    "ta": "tam",
    "te": "tel",
    "mr": "mar",
    "gu": "guj",
    "kn": "kan",
    "ml": "mal",
    "pa": "pan",
    "ur": "urd",
    "ar": "ara",
    "zh": "chi_sim",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
}


def _iso_to_tesseract(lang_hint: str | None) -> str:
    """Convert ISO 639-1 hint to Tesseract language string."""
    if not lang_hint:
        return "eng"
    codes = [c.strip() for c in lang_hint.split("+")]
    tess_codes = [_LANG_MAP.get(c, c) for c in codes]
    return "+".join(tess_codes)


def _run_ocr(image_bytes: bytes, lang: str) -> OCRResult:
    """Synchronous OCR — run via run_in_executor."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise ProviderError(
            "pytesseract and Pillow are required for Tesseract OCR. "
            "Run: pip install pytesseract Pillow"
        ) from e

    image = Image.open(io.BytesIO(image_bytes))
    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as e:
        raise ProviderError(
            "Tesseract is not installed or not on PATH. "
            "Install via: brew install tesseract (macOS) or "
            "apt-get install tesseract-ocr (Ubuntu)"
        ) from e
    except Exception as e:
        raise ProviderError(f"Tesseract OCR failed: {e}") from e

    from collections import defaultdict

    lines_by_par: dict[
        tuple[int, int], dict[int, list[tuple[str, float, float, float, float, float]]]
    ] = defaultdict(lambda: defaultdict(list))

    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf_raw = data["conf"][i]
        if not text or conf_raw < 0:
            continue

        b_num = int(data.get("block_num", [0] * n)[i])
        p_num = int(data.get("par_num", [0] * n)[i])
        l_num = int(data.get("line_num", [0] * n)[i])

        confidence = float(conf_raw) / 100.0
        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])

        lines_by_par[(b_num, p_num)][l_num].append((text, confidence, x, y, x + w, y + h))

    blocks: list[OCRBlock] = []
    confidences: list[float] = []
    full_text_parts: list[str] = []

    for (_b_num, _p_num), lines_dict in lines_by_par.items():
        par_lines: list[str] = []
        par_x0 = float("inf")
        par_y0 = float("inf")
        par_x1 = float("-inf")
        par_y1 = float("-inf")
        par_confs: list[float] = []

        for l_num in sorted(lines_dict.keys()):
            word_tuples = lines_dict[l_num]
            line_text = " ".join(wt[0] for wt in word_tuples)
            par_lines.append(line_text)
            for _, c, x0, y0, x1, y1 in word_tuples:
                par_x0 = min(par_x0, x0)
                par_y0 = min(par_y0, y0)
                par_x1 = max(par_x1, x1)
                par_y1 = max(par_y1, y1)
                par_confs.append(c)

        if par_lines:
            block_text = "\n".join(par_lines)
            block_conf = sum(par_confs) / len(par_confs) if par_confs else 1.0
            block_bbox = (par_x0, par_y0, par_x1, par_y1)
            blocks.append(OCRBlock(text=block_text, bbox=block_bbox, confidence=block_conf))
            confidences.extend(par_confs)
            full_text_parts.append(block_text)

    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    full_text = "\n\n".join(full_text_parts)

    return OCRResult(
        blocks=blocks,
        full_text=full_text,
        mean_confidence=mean_confidence,
    )


class TesseractOCR:
    """OCR engine backed by Tesseract via pytesseract."""

    def __init__(self, lang: str = "eng") -> None:
        self._default_lang = lang

    @property
    def provider_name(self) -> str:
        return "tesseract"

    async def ocr(
        self,
        image_bytes: bytes,
        lang_hint: str | None = None,
    ) -> OCRResult:
        """Run OCR on image bytes asynchronously (via thread pool executor)."""
        lang = _iso_to_tesseract(lang_hint) if lang_hint else self._default_lang
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_ocr, image_bytes, lang)
        logger.info(
            "Tesseract OCR complete",
            extra={
                "lang": lang,
                "blocks": len(result.blocks),
                "mean_confidence": round(result.mean_confidence, 3),
            },
        )
        return result
