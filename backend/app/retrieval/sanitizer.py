"""Text sanitization and OCR font ligature repair utilities for broadsheet text and headlines."""

from __future__ import annotations

import re

# Direct Unicode typographic ligatures to ASCII decomposition
_UNICODE_LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}

# Regex patterns for common broadsheet OCR dropouts where ligatures (ff, fi, fl, ffi, ffl)
# collapsed into replacement characters (\ufffd), whitespace gaps, or dropouts.
_COMMON_LIGATURE_REPAIRS: list[tuple[re.Pattern[str], str]] = [
    # ff ligatures
    (re.compile(r"\be[\s\ufffd]+ort(s)?\b", re.IGNORECASE), r"effort\1"),
    (re.compile(r"\bdi[\s\ufffd]+erent(ly)?\b", re.IGNORECASE), r"different\1"),
    (re.compile(r"\bdi[\s\ufffd]+icult(y|ies)?\b", re.IGNORECASE), r"difficult\1"),
    (re.compile(r"\bo[\s\ufffd]+cer(s)?\b", re.IGNORECASE), r"officer\1"),
    (re.compile(r"\bo[\s\ufffd]+cial(s|ly)?\b", re.IGNORECASE), r"official\1"),
    (re.compile(r"\bo[\s\ufffd]+ce(s)?\b", re.IGNORECASE), r"office\1"),
    (re.compile(r"\be[\s\ufffd]+ect(s|ive|ively)?\b", re.IGNORECASE), r"effect\1"),
    (re.compile(r"\ba[\s\ufffd]+ect(s|ed|ing)?\b", re.IGNORECASE), r"affect\1"),
    (re.compile(r"\btra[\s\ufffd]+c\b", re.IGNORECASE), "traffic"),
    (re.compile(r"\bsta(?:[\ufffd]+|\s{2,})(?=\s|[.,;:!?]|$)", re.IGNORECASE), "staff"),
    (re.compile(r"\bsu[\s\ufffd]+er(s|ed|ing)?\b", re.IGNORECASE), r"suffer\1"),
    (re.compile(r"\bo[\s\ufffd]+er(s|ed|ing)?\b", re.IGNORECASE), r"offer\1"),
    (re.compile(r"\btari(?:[\ufffd]+|\s{2,})(?=\s|[.,;:!?]|$)", re.IGNORECASE), "tariff"),
    (re.compile(r"\ba[\s\ufffd]+ord(able|ability|ed|ing)?\b", re.IGNORECASE), r"afford\1"),
    (re.compile(r"\be[\s\ufffd]+cien(t|cy|tly)?\b", re.IGNORECASE), r"efficien\1"),
    (re.compile(r"\bsu[\s\ufffd]+cien(t|cy|tly)?\b", re.IGNORECASE), r"sufficien\1"),
    # fi ligatures
    (re.compile(r"\bpro[\s\ufffd]+t(s|able|ability)?\b", re.IGNORECASE), r"profit\1"),
    (re.compile(r"\bcon[\s\ufffd]+dence\b", re.IGNORECASE), "confidence"),
    (re.compile(r"\bcon[\s\ufffd]+dent\b", re.IGNORECASE), "confident"),
    (re.compile(r"\bde[\s\ufffd]+ne(s|d)?\b", re.IGNORECASE), r"define\1"),
    (re.compile(r"\bde[\s\ufffd]+nite(ly)?\b", re.IGNORECASE), r"definite\1"),
    (re.compile(r"\bquali[\s\ufffd]+ed\b", re.IGNORECASE), "qualified"),
    (re.compile(r"\bquali[\s\ufffd]+y\b", re.IGNORECASE), "qualify"),
    (re.compile(r"\bsigni[\s\ufffd]+cant(ly)?\b", re.IGNORECASE), r"significant\1"),
    (re.compile(r"\bbene[\s\ufffd]+t(s)?\b", re.IGNORECASE), r"benefit\1"),
    (re.compile(r"\bde[\s\ufffd]+cit(s)?\b", re.IGNORECASE), r"deficit\1"),
    (re.compile(r"\bcon[\s\ufffd]+rm(s|ed|ation)?\b", re.IGNORECASE), r"confirm\1"),
    (re.compile(r"\bmo[\s\ufffd]+ed\b", re.IGNORECASE), "modified"),
    (re.compile(r"\bspe[\s\ufffd]+c(s)?\b", re.IGNORECASE), r"specific\1"),
    (re.compile(r"\bno[\s\ufffd]+ed\b", re.IGNORECASE), "notified"),
    (re.compile(r"\bve[\s\ufffd]+ed\b", re.IGNORECASE), "verified"),
    (re.compile(r"\bcer[\s\ufffd]+ed\b", re.IGNORECASE), "certified"),
    (re.compile(r"\biden[\s\ufffd]+ed\b", re.IGNORECASE), "identified"),
    # fl ligatures
    (re.compile(r"\bin[\s\ufffd]+ation(ary)?\b", re.IGNORECASE), r"inflation\1"),
    (re.compile(r"\bcon[\s\ufffd]+ict(s)?\b", re.IGNORECASE), r"conflict\1"),
    (re.compile(r"\bin[\s\ufffd]+uence(s|d)?\b", re.IGNORECASE), r"influence\1"),
    (re.compile(r"\bre[\s\ufffd]+ect(s|ed|ing|ion)?\b", re.IGNORECASE), r"reflect\1"),
    (re.compile(r"\b[\s\ufffd]+oat(ing)?\b", re.IGNORECASE), r"float\1"),
    (re.compile(r"\b[\s\ufffd]+ow(s|ing|ed)?\b", re.IGNORECASE), r"flow\1"),
]


def repair_text_ligatures(text: str | None) -> str:
    """Repair broken font ligatures and OCR dropouts in broadsheet text and headlines."""
    if not text:
        return ""

    result = text

    # 1. Normalize unicode ligature symbols to plain ASCII equivalents
    for u_lig, ascii_rep in _UNICODE_LIGATURE_MAP.items():
        if u_lig in result:
            result = result.replace(u_lig, ascii_rep)

    # 2. Apply targeted dictionary regex repairs for common broadsheet words
    for pattern, replacement in _COMMON_LIGATURE_REPAIRS:
        result = pattern.sub(replacement, result)

    # 3. Strip any residual unprintable replacement characters between letters
    result = re.sub(r"([a-zA-Z])[\ufffd]+([a-zA-Z])", r"\1\2", result)
    result = result.replace("\ufffd", "")

    return result
