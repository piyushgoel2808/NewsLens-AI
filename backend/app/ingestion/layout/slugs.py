"""Unified Source of Truth for Newspaper Slugs, Wire Agencies & Datelines.

Centralizes:
1. Wire agency stamps, dateline cities, and syndication slugs.
2. Section header blacklists and Table of Contents (ToC) patterns.
3. Kicker extraction and headline validity predicates.
4. Pullquote attribution and OCR text repair dictionaries.
"""

from __future__ import annotations

import re

from app.ingestion.detector import (
    is_noise_or_promo_text,
    is_title_case_or_uppercase,
    sanitize_block_text,
)

# =============================================================================
# Wire Agencies, Syndication & Dateline Vocabularies
# =============================================================================

WIRE_AGENCIES: frozenset[str] = frozenset(
    {
        "PTI",
        "AGENCIES",
        "THE GOAN I NETWORK",
        "THE GOAN NETWORK",
        "GOAN NETWORK",
        "REUTERS",
        "AFP",
        "IANS",
        "UNI",
        "AP",
        "ANI",
        "BLOOMBERG",
        "TNN",
        "EXPRESS NEWS SERVICE",
        "SPECIAL CORRESPONDENT",
        "STAFF REPORTER",
        "BUREAU",
        "NEWS DESK",
        "PRESS TRUST OF INDIA",
        "UNITED NEWS OF INDIA",
    }
)

DATELINE_CITIES: frozenset[str] = frozenset(
    {
        "NEW YORK", "WASHINGTON", "LONDON", "BEIJING", "MOSCOW", "TOKYO", "PARIS",
        "BERLIN", "CANBERRA", "ISLAMABAD", "COLOMBO", "DHAKA", "KATHMANDU", "GENEVA",
        "DUBAI", "DOHA", "RIYADH", "SINGAPORE", "BANGKOK", "TASHKENT", "SYDNEY",
        "MELBOURNE", "TORONTO", "OTTAWA", "LOS ANGELES", "SAN FRANCISCO", "CHICAGO",
        "PANAJI", "MARGAO", "VASCO", "MAPUSA", "PONDA", "VALPOI", "BICHOLIM",
        "CURCHORHEM", "QUEPEM", "CANACONA", "PERNEM", "SANGUEM", "PORVORIM", "CALANGUTE",
        "CANDOLIM", "SANQUELIM", "CORTALIM", "TIVIM", "CUNCOLIM", "BENAUDIM",
        "NEW DELHI", "MUMBAI", "BENGALURU", "KOLKATA", "CHENNAI", "HYDERABAD",
        "AHMEDABAD", "PUNE", "JAIPUR", "LUCKNOW", "CHANDIGARH", "PATNA", "BHOPAL",
        "SRINAGAR", "JAMMU", "GUWAHATI", "KOCHI", "THIRUVANANTHAPURAM", "RANCHI",
        "RAIPUR", "DEHRADUN", "SHIMLA", "AMRITSAR", "VARANASI", "AGRA", "INDORE",
        "NAGPUR", "VISAKHAPATNAM", "BHUBANESWAR", "SURAT", "VADODARA", "MANGALORE",
    }
)

SECTION_HEADER_BLACKLIST: frozenset[str] = frozenset(
    {
        "tech & startups",
        "tech and startups",
        "deals, tech & startups",
        "deals tech startups",
        "mark to market",
        "news wrap",
        "in brief",
        "news in brief",
        "brief update",
        "brief updates",
        "quick update",
        "quick updates",
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

SYNDICATION_SLUGS: frozenset[str] = frozenset(
    {
        "the wall street journal",
        "wall street journal",
        "reuters",
        "bloomberg",
        "bloomberg news",
        "pti",
        "press trust of india",
        "afp",
        "agence france-presse",
        "ap",
        "associated press",
        "financial times",
        "new york times",
        "business wire",
        "pr newswire",
        "news in numbers",
        "columns",
        "inside",
        "quote of the day",
        "data bites",
        "plain facts",
        "mint primer",
        "mint curator",
        "ask mint",
        "mark to market",
        "wsj",
    }
)

SYNDICATION_REGEX = re.compile(
    r"^(?:THE\s+)?(?:WALL\s+STREET\s+JOURNAL|REUTERS|BLOOMBERG(?:\s+NEWS)?|PTI|AFP|AP|"
    r"FINANCIAL\s+TIMES|NEW\s+YORK\s+TIMES|PRESS\s+TRUST\s+OF\s+INDIA|ASSOCIATED\s+PRESS|"
    r"QUOTE\s+OF\s+THE\s+DAY|DATA\s+BITES|PLAIN\s+FACTS|NEWS\s+IN\s+NUMBERS|COLUMNS|INSIDE|WSJ|"
    r"MARK\s+TO\s+MARKET|MINT\s+PRIMER|MINT\s+CURATOR|ASK\s+MINT)(?:\s*[\/\-–—|]\s*.*)?$",
    re.IGNORECASE,
)

NUMBERED_QUESTION_REGEX = re.compile(
    r"^(?:(?:Q\.?\s*)?\d{1,2}[\.\/\)]|\b(?:Q\d{1,2}|Part\s+\d+|Step\s+\d+)\b|"
    r"\b\d{1,2}\s+(?:How|Why|What|When|Where|Who|Which|Can|Will|Is|Are|Do|Does|Did|Should|Could|Would|Has|Have|Had))\s+",
    re.IGNORECASE,
)

STANDALONE_FEATURE_KICKER_REGEX = re.compile(
    r"^(?:MINT\s+PRIMER|PLAIN\s+FACTS|LONG\s+STORY|MARK\s+TO\s+MARKET|MINT\s+CURATOR|"
    r"ASK\s+MINT|POWER\s+POINT|MYTHS\s+AND\s+MANTRAS|DEALS,\s+TECH\s+&\s+STARTUPS|INSIDE)$",
    re.IGNORECASE,
)

TOC_SECTION_SLUGS_REGEX = re.compile(
    r"(?i)\b(?:Global|World|National|International|Business|Money|Economy|Views|"
    r"Editorial|Sport|Sports|Life|Metro|City|News|Focus|State|States|Showcase|"
    r"Inside|Features)\s*\|"
)

TOC_PAGE_POINTER_REGEX = re.compile(
    r"(?i)(?:>\s*P\s*\d+|>\s*Page\s*\d+|->\s*P\s*\d+|\bP\d{1,2}\b)"
)

KICKER_REGEX = re.compile(
    r"(?i)^(?:OUR\s+VIEW|MY\s+VIEW|THEIR\s+VIEW|QUICK\s+EDIT|PLAIN\s+FACTS|MINT\s+PRIMER|"
    r"MARK\s+TO\s+MARKET|MINT\s+CURATOR|COLUMN|ASK\s+MINT|POWER\s+POINT|"
    r"ECONOMY\s+&\s+POLICY|DEALS,\s+TECH\s+&\s+STARTUPS|MYTHS\s+AND\s+MANTRAS|"
    r"IN\s+BRIEF|ROUNDUP|NEWS\s+IN\s+BRIEF|LONG\s+STORY|MINT\s+MONEY|"
    r"VIEWS\s+&\s+OPINIONS|BUSINESS\s+OF\s+LIFE|CORPORATE|GLOBAL|COMPANIES)"
    r"[:\s\|\-]+(.*)$"
)

MARKETING_SLOGAN_REGEX = re.compile(
    r"(?i)\b(?:innovation|future|tomorrow|trusted|trust|quality|solutions|technology|technologies|"
    r"milestone|placement|equity|shares|corporate|leading|growth|global|investors|advisors|built for|backed by|driven by|designed for)\b"
)

BOILERPLATE_TOKENS: frozenset[str] = frozenset(
    {
        "limited", "ltd", "corp", "corporation", "pvt", "private", "equity", "issue",
        "issue,", "shares", "company", "notice", "promoters", "price", "band", "page",
        "continued", "from", "and", "or", "of", "in", "on", "at", "to", "for", "with",
        "advertisement", "public", "statutory", "tender", "bid", "face", "value",
    }
)

NUMERIC_STAT_PATTERN = re.compile(
    r"\b(?:\d+[\d,\.]*\s*(?:cr|crore|mn|million|bn|billion|lakh|%|pts|bps|usd|inr)?|"
    r"[\$₹€£]\s*\d+[\d,\.]*)\b",
    re.IGNORECASE,
)

PULLQUOTE_TITLE_REGEX = re.compile(
    r"(?i)\b(?:FOREIGN\s*MINISTER|PRIME\s*MINISTER|CHIEF\s*MINISTER|FINANCE\s*MINISTER|"
    r"HOME\s*MINISTER|DEFENCE\s*MINISTER|EXTERNAL\s*AFFAIRS\s*MINISTER|SECRETARY\s*GENERAL|"
    r"SPOKESPERSON|MANAGING\s*DIRECTOR|CHIEF\s*EXECUTIVE\s*OFFICER|EXECUTIVE\s*DIRECTOR|"
    r"FED\s*CHAIR(?:MAN)?|CENTRAL\s*BANK\s*GOVERNOR|CHIEF\s*JUSTICE|"
    r"AUSTRALIAN\s*FOREIGN\s*MINISTER|AUSTRALIANFOREIGN\s*MINISTER|PENNY\s*WONG|"
    r"PENNYWONG)\b"
)

HEADLINE_ACTION_VERBS: frozenset[str] = frozenset(
    {
        "holds", "cuts", "hikes", "raises", "drops", "rises", "falls", "soars",
        "plans", "buys", "sells", "warns", "sees", "eyes", "urges", "tells",
        "signs", "clears", "approves", "rejects", "posts", "hits", "leads",
        "mops", "caps", "curbs", "eases", "nods",
    }
)

CONTINUATION_END_TOKENS: frozenset[str] = frozenset(
    {
        "says", "warns", "sees", "eyes", "seeks", "aims", "cuts", "beats",
        "posts", "plans", "gets", "hits", "leads", "backs", "signs", "finds",
        "moots", "targets", "buys", "hires", "urges", "tells", "drops", "rises",
        "to", "in", "on", "at", "by", "for", "with", "from", "about", "into",
        "over", "after", "and", "or", "of", "as", "against", "despite", "under",
        "near", "up", "down", "out", "a", "an", "the",
    }
)

_TERMINAL_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "inc.", "corp.", "co.", "ltd.", "pvt.", "llc.", "u.s.", "u.k.", "d.c.",
        "govt.", "no.", "vs.", "v.", "dr.", "prof.", "st.", "jr.", "sr.",
    }
)

_ACRONYM_END_RE = re.compile(r"\b(?:[A-Z]\.){1,4}$", re.IGNORECASE)

_DATELINE_PREFIX_RE = re.compile(
    r"^([A-Za-z\s]{2,25})\s*[:–—\-]\s*",
)

_DATELINE_WITH_AGENCY_RE = re.compile(
    r"^[A-Za-z\s]{2,25}\s*\([A-Za-z\s\.\/]+\)\s*[:–—\-]\s*",
)

_LEGAL_NOTICE_RE = re.compile(
    r"(?i)\b(?:public notice is hereby given|notice inviting tender|"
    r"before the hon'?ble|in the matter of|whereas it has been|"
    r"notice is hereby given|this is to inform that|corrigendum to)\b"
)

OCR_HEADLINE_REPAIRS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bOl\s+estimates\b", re.IGNORECASE), "Q1 estimates"),
    (re.compile(r"\bQl\s+estimates\b", re.IGNORECASE), "Q1 estimates"),
    (re.compile(r"\bofGlasgow\b"), "of Glasgow"),
    (re.compile(r"\bcourtcases\b"), "court cases"),
    (re.compile(r"\btheworld\b"), "the world"),
    (re.compile(r"\beconomicdocudramahastheworld\b"), "economic docudrama has the world"),
    (re.compile(r"\bdocudramahastheworld\b"), "docudrama has the world"),
    (re.compile(r"\bartificialintelligence\b", re.IGNORECASE), "artificial intelligence"),
    (re.compile(r"\bcaseofoTT\b"), "case of OTT"),
    (re.compile(r"\bofaiding\b"), "of aiding"),
    (re.compile(r"\byourfamilybe\b"), "your family be"),
    (re.compile(r"\babletoaccess\b"), "able to access"),
    (re.compile(r"\briskforpharma\b"), "risk for pharma"),
    (re.compile(r"\btoChatGPT\b"), "to ChatGPT"),
    (re.compile(r"\bmutual fund-onlyPMS\b", re.IGNORECASE), "mutual fund-only PMS"),
    (re.compile(r"\b25lakh\b", re.IGNORECASE), "25 lakh"),
    (re.compile(r"\bageneric\b", re.IGNORECASE), "a generic"),
    (re.compile(r"\bIUNE\b"), "JUNE"),
    (re.compile(r"\bQl\b"), "Q1"),
]

# =============================================================================
# Layout Text Predicates & Cleaning Helpers
# =============================================================================


def is_syndication_or_agency_slug(text: str) -> bool:
    """Check if text is a syndication slug, wire agency stamp, or recurring column header."""
    t = text.strip()
    if not t:
        return False
    t_clean = t.lower().strip(" .:;,/–—-")
    if t_clean in SYNDICATION_SLUGS or t_clean in SECTION_HEADER_BLACKLIST:
        return True
    if SYNDICATION_REGEX.match(t):
        return True
    return bool(re.match(r"^(?:reuters|pti|bloomberg|afp|ap|ians|ani|uni)\s*[\/|\-–—]", t_clean))


def is_numbered_feature_subhead(text: str) -> bool:
    """Detect numbered subheadings / questions in feature explainers."""
    t = text.strip()
    if not t:
        return False
    return bool(NUMBERED_QUESTION_REGEX.match(t))


def is_toc_index_block(text: str) -> bool:
    """Check for high densities of delimiter patterns common in front-page indexes."""
    if not text or not text.strip():
        return False
    t = text.strip()

    has_slug = bool(TOC_SECTION_SLUGS_REGEX.search(t))
    has_pointer = bool(TOC_PAGE_POINTER_REGEX.search(t))
    pipe_count = t.count("|")

    if has_slug and has_pointer:
        return True
    if has_slug and pipe_count >= 1 and (">" in t or "P" in t):
        return True
    if pipe_count >= 2 and has_pointer:
        return True

    lines = [line_item.strip() for line_item in t.split("\n") if line_item.strip()]
    if len(lines) >= 2:
        toc_line_matches = sum(
            1
            for line_item in lines
            if (
                TOC_SECTION_SLUGS_REGEX.search(line_item)
                or TOC_PAGE_POINTER_REGEX.search(line_item)
                or "|" in line_item
            )
        )
        if toc_line_matches >= 2:
            return True

    return False


def is_garbled_ocr_noise(text: str) -> bool:
    """Detect garbled OCR noise strings with high ratio of non-words."""
    if not text or len(text.strip()) < 4:
        return False
    clean = re.sub(r"\s+", "", text)
    if not clean:
        return False
    symbol_count = sum(1 for c in clean if not (c.isalnum() or c in ".,!?'\"-–—$₹%()"))
    if symbol_count / len(clean) > 0.28:
        return True
    return bool(re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", clean.lower()))


def is_pullquote_author_block(text: str, surrounding_text: str = "") -> bool:
    """The Attribution Rule: Detect speaker/author attribution in pull quotes and sidebars."""
    t = text.strip()
    if not t:
        return False
    words = t.split()
    if len(words) > 8:
        return False

    words_clean = [w.lower().strip(".:;,!?'\"-–—") for w in words]
    if any(w in HEADLINE_ACTION_VERBS for w in words_clean):
        return False

    # 1. Matches specific author/minister/diplomat attribution names or titles
    if PULLQUOTE_TITLE_REGEX.search(t):
        return True

    # 2. Text is <= 6 words and surrounding/preceding block contains quotation marks
    has_quotes = bool(
        re.search(r'["“”‘’\']', surrounding_text) or re.search(r'["“”‘’\']', t)
    )
    clean = re.sub(r"[^\w\s]", "", t).strip()
    is_cased = clean.isupper() or is_title_case_or_uppercase(t)
    return bool(len(words) <= 6 and has_quotes and is_cased)


def clean_ocr_text_artifacts(text: str) -> str:
    """Repair unspaced tokens, font ligature bugs, and purge UUID/promo noise."""
    res = text
    for pattern, repl in OCR_HEADLINE_REPAIRS:
        res = pattern.sub(repl, res)
    return sanitize_block_text(res).strip()


def is_numeric_stat_box(text: str) -> bool:
    """Detect if text is an infographic/stat/table box rather than a textual headline."""
    tokens = text.strip().split()
    if not tokens:
        return False
    stat_matches = NUMERIC_STAT_PATTERN.findall(text)
    if len(stat_matches) >= 3 or (len(stat_matches) >= 2 and len(tokens) <= 6):
        return True
    numeric_tokens = sum(
        1
        for tok in tokens
        if any(c.isdigit() for c in tok)
        or tok.lower() in ("cr", "crore", "mn", "million", "bn", "billion", "lakh", "%", "pts", "bps")
    )
    return bool(len(tokens) > 0 and (numeric_tokens / len(tokens)) >= 0.40)


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
    if not words or len(words) < 2:
        return False
    # Use original text length as fallback when punctuation was stripped
    if len(cleaned) < 6 and len(text.strip()) < 8:
        return False
    # Single words are never valid article headlines
    if len(words) == 1:
        return False
    # Standalone dateline cities or wire agencies are never article headlines
    cleaned_upper = cleaned.upper()
    if cleaned_upper in DATELINE_CITIES or cleaned_upper in WIRE_AGENCIES:
        return False
    # Filter out syndication slugs, wire stamps, and numbered subheadings
    if is_syndication_or_agency_slug(text):
        return False
    if is_numbered_feature_subhead(text):
        return False
    # Filter out Table of Contents (ToC) / index teasers and pullquote author attributions
    if is_toc_index_block(text):
        return False
    if is_pullquote_author_block(text):
        return False
    if is_garbled_ocr_noise(text) or is_noise_or_promo_text(text):
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
    # Filter out legal / statutory boilerplate notices
    if _LEGAL_NOTICE_RE.search(text):
        return False
    # Reject dateline-starting text (e.g. "NEW DELHI: The finance ministry...")
    stripped_text = text.strip()
    if _DATELINE_WITH_AGENCY_RE.match(stripped_text):
        return False
    d_match = _DATELINE_PREFIX_RE.match(stripped_text)
    if d_match:
        prefix = re.sub(r"[^\w\s]", "", d_match.group(1)).strip().upper()
        if prefix in DATELINE_CITIES or prefix in WIRE_AGENCIES:
            return False
    # Numeric stat boxes are never headlines
    stat_matches = NUMERIC_STAT_PATTERN.findall(text)
    if len(stat_matches) >= 3 or (len(stat_matches) >= 2 and len(words) <= 6):
        return False

    r_text = text.rstrip()

    # Reject declarative sentences >= 5 words ending in a period or semicolon, with abbreviation protection
    if len(words) >= 5 and r_text.endswith((".", ";")):
        tokens = r_text.split()
        last_tok = tokens[-1].lower() if tokens else ""
        is_abbrev = (
            last_tok in _TERMINAL_ABBREVIATIONS
            or bool(_ACRONYM_END_RE.search(tokens[-1]))
        )
        if not is_abbrev:
            return False

    # Multi-sentence paragraphs ending in period, semicolon, or exclamation (>20 words) are not headlines
    return not (len(words) > 20 and r_text.endswith((".", ";", "!")))


__all__ = [
    "BOILERPLATE_TOKENS",
    "CONTINUATION_END_TOKENS",
    "DATELINE_CITIES",
    "HEADLINE_ACTION_VERBS",
    "KICKER_REGEX",
    "MARKETING_SLOGAN_REGEX",
    "NUMBERED_QUESTION_REGEX",
    "NUMERIC_STAT_PATTERN",
    "OCR_HEADLINE_REPAIRS",
    "PULLQUOTE_TITLE_REGEX",
    "SECTION_HEADER_BLACKLIST",
    "STANDALONE_FEATURE_KICKER_REGEX",
    "SYNDICATION_REGEX",
    "SYNDICATION_SLUGS",
    "TOC_PAGE_POINTER_REGEX",
    "TOC_SECTION_SLUGS_REGEX",
    "WIRE_AGENCIES",
    "clean_ocr_text_artifacts",
    "extract_kicker_and_clean_headline",
    "is_garbled_ocr_noise",
    "is_numbered_feature_subhead",
    "is_numeric_stat_box",
    "is_pullquote_author_block",
    "is_syndication_or_agency_slug",
    "is_toc_index_block",
    "is_valid_headline_candidate",
]
