"""Query Entity Recognition and Parameter Extraction for NewsLens-AI.

Extracts explicit broadsheet parameters (publication brands, publication dates,
issue IDs, page numbers, categories, differential indicators) and cleans conversational queries.
"""

from __future__ import annotations

import calendar
import contextlib
import re
from typing import Any

# ---------------------------------------------------------------------------
# Deterministic Brand & Section Patterns
# ---------------------------------------------------------------------------

_KNOWN_BRANDS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:the\s+)?goan(?:\s+everyday)?\b", re.I), "The Goan"),
    (re.compile(r"\b(?:the\s+)?economic\s+times\b|\bthe\s+economics\s+times\b|\beconomics\s+times\b", re.I), "The Economic Times"),
    (re.compile(r"\b(?:the\s+)?new\s+york\s+times\b|\bNYT\b", re.I), "The New York Times"),
    (re.compile(r"\b(?:the\s+)?times\s+of\s+india\b|\bTOI\b", re.I), "The Times of India"),
    (re.compile(r"\b(?:the\s+)?wall\s+street\s+journal\b|\bWSJ\b", re.I), "The Wall Street Journal"),
    (re.compile(r"\b(?:the\s+)?washington\s+post\b|\bWAPO\b", re.I), "The Washington Post"),
    (re.compile(r"\b(?:the\s+)?financial\s+times\b|\bFT\b", re.I), "Financial Times"),
    (re.compile(r"\b(?:the\s+)?guardian\b", re.I), "The Guardian"),
    (re.compile(r"\bbusiness\s+standard\b|\bBS\b", re.I), "Business Standard"),
    (re.compile(r"\b(?:the\s+)?indian\s+express\b|\bIE\b", re.I), "The Indian Express"),
    (re.compile(r"\b(?:the\s+)?hindu\b", re.I), "The Hindu"),
    (re.compile(r"\bhindustan\s+times\b|\bHT\b", re.I), "Hindustan Times"),
    (re.compile(r"\bmint\b|\blivemint\b", re.I), "Mint"),
    (re.compile(r"\b(?:the\s+)?daily\s+tribune\b|\btribune\b", re.I), "The Daily Tribune"),
    (re.compile(r"\b(?:the\s+)?daily\s+chronicle\b|\bchronicle\b", re.I), "The Daily Chronicle"),
    (re.compile(r"\bdaily\s+broadsheet\b", re.I), "Daily Broadsheet"),
    (re.compile(r"\b(?:(?:the|he)\s+)?morning\s+standard\b|\bmorning\s+standard\b", re.I), "The Morning Standard"),
    (re.compile(r"\b(?:the\s+)?financial\s+chronicle\b", re.I), "Financial Chronicle"),
    (re.compile(r"\b(?:the\s+)?daily\s+record\b", re.I), "The Daily Record"),
]

_SECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(sports?|athletics?|cricket|tennis|football|soccer|olympics?|tournaments?|ipl|bcci|icc|grand\s+slam|badminton|golf)\b", re.I), "Sports"),
    (re.compile(r"\b(entertainment|cinema|movies?|films?|bollywood|hollywood|arts?|music|concerts?|theatre|ott|actors?|actress(?:es)?)\b", re.I), "Entertainment"),
    (re.compile(r"\b(science|space|isro|nasa|satellites?|rockets?|astronomy|climate|environment|wildlife|forest|ocean|ecology)\b", re.I), "Science & Environment"),
    (re.compile(r"\b(tech|technology|ai|startups?|semiconductors?|chips?|software|fintech|gadgets?|cybersecurity|cloud|algorithms?)\b", re.I), "Technology"),
    (re.compile(r"\b(business|markets?|stocks?|shares|sensex|nifty|equities|ipo|corporate|companies|banking|finance|financial)\b", re.I), "Business & Markets"),
    (re.compile(r"\b(economy|economic|macroeconomics?|inflation|gdp|cpi|wpi|taxation|gst|tariffs?|budget|fiscal)\b", re.I), "Economy & Policy"),
    (re.compile(r"\b(politics|political|elections?|voters?|parliament|lok\s+sabha|rajya\s+sabha|ministers?|cabinet|bjp|congress)\b", re.I), "Politics"),
    (re.compile(r"\b(health|healthcare|hospitals?|doctors?|pharma|pharmaceuticals?|medicine|medicines?|vaccines?|diseases?)\b", re.I), "Health"),
    (re.compile(r"\b(crime|courts?|legal|law|supreme\s+court|high\s+court|judges?|verdicts?|bail|cbi|ed|police|arrests?|scams?|fraud)\b", re.I), "Crime & Law"),
    (re.compile(r"\b(opinion|editorials?|columns?|op-ed|commentary|viewpoints?|perspectives?)\b", re.I), "Opinion/Editorial"),
    (re.compile(r"\b(world|international|global|diplomacy|foreign\s+affairs|treaty|summit|united\s+nations|g20|brics)\b", re.I), "World/International"),
    (re.compile(r"\b(lifestyle|travel|tourism|food|dining|wellness|fitness|luxury|automotive)\b", re.I), "Lifestyle"),
]

_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_CONVERSATIONAL_PREFIX_PATTERNS = [
    r"^(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|please\s+)+",
    r"^(?:tell\s+me\s+(?:about|more\s+about)?|summarize|explain|what\s+is|what\s+are|what\s+happened\s+(?:to|with)?|who\s+is|who\s+was|search\s+for|find\s+(?:news\s+about|information\s+on)?|give\s+me\s+(?:a\s+summary\s+of|details\s+about)?|overview\s+of)\s+(?:the\s+|a\s+|an\s+)?",
]


# ---------------------------------------------------------------------------
# Extraction Functions
# ---------------------------------------------------------------------------

def extract_parameters_from_query(query: str) -> dict[str, Any]:
    """Extract explicit entity parameters (newspaper, issue ID, date, section/category, page) from query."""
    params: dict[str, Any] = {}
    if not query:
        return params

    # 1. Multi-Newspaper Brand Extraction
    matched_brands: list[tuple[int, str]] = []
    for pat, brand in _KNOWN_BRANDS_PATTERNS:
        for brand_m in pat.finditer(query):
            matched_brands.append((brand_m.start(), brand))

    if matched_brands:
        matched_brands.sort(key=lambda x: x[0])
        ordered_brands: list[str] = []
        for _, b in matched_brands:
            if b not in ordered_brands:
                ordered_brands.append(b)

        params["target_newspapers"] = ordered_brands
        params["newspaper_name"] = ordered_brands[0]
        if len(ordered_brands) >= 2:
            params["comparison_newspaper"] = ordered_brands[1]

        q_lower = query.lower()
        if any(w in q_lower for w in ["but not in", "not in", "absent", "exclusive", "omitted", "missing from"]):
            params["is_differential"] = True
            params["source_newspaper"] = ordered_brands[0]
            if len(ordered_brands) >= 2:
                params["comparison_newspaper"] = ordered_brands[1]

    # Shared / Similar wire coverage detection across publications
    q_lower_all = query.lower()
    _SHARED_TRIGGERS = (
        "similar",
        "shared",
        "common",
        "same article",
        "same articles",
        "same story",
        "same stories",
        "both newspaper",
        "both newspapers",
        "both paper",
        "both papers",
        "in both",
        "covered by both",
        "both carried",
        "syndicated",
        "wire stories",
        "wire story",
        "wire report",
        "wire reports",
    )
    if any(trig in q_lower_all for trig in _SHARED_TRIGGERS):
        params["is_shared"] = True
        if params.get("target_newspapers") and len(params["target_newspapers"]) >= 2:
            params["source_newspaper"] = params["target_newspapers"][0]
            params["comparison_newspaper"] = params["target_newspapers"][1]

    # 2. Issue ID Extraction
    iss_match = re.search(r"\bissue\s*(?:id\s*[:=]?\s*|\#\s*|no\.?\s*|number\s*)?(\d+)\b", query, re.I)
    if iss_match:
        with contextlib.suppress(ValueError):
            params["issue_id"] = int(iss_match.group(1))

    # 3. Date Extraction
    found_dates: list[str] = []
    # YYYY-MM-DD
    for iso_m in re.finditer(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", query):
        year_val, month_val, day_val = int(iso_m.group(1)), int(iso_m.group(2)), int(iso_m.group(3))
        iso_str = f"{year_val:04d}-{month_val:02d}-{day_val:02d}"
        if iso_str not in found_dates:
            found_dates.append(iso_str)

    # DD/MM/YYYY or DD-MM-YYYY
    for dmy_m in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", query):
        day_val, month_val, year_val = int(dmy_m.group(1)), int(dmy_m.group(2)), int(dmy_m.group(3))
        iso_str = f"{year_val:04d}-{month_val:02d}-{day_val:02d}"
        if iso_str not in found_dates:
            found_dates.append(iso_str)

    # Named months (e.g. 1st August 2026 or August 1, 2026)
    for m1 in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]+)(?:,)?\s+(\d{4})\b", query):
        if m1.group(2).lower() in _MONTH_MAP:
            day_val, month_val, year_val = int(m1.group(1)), _MONTH_MAP[m1.group(2).lower()], int(m1.group(3))
            iso_str = f"{year_val:04d}-{month_val:02d}-{day_val:02d}"
            if iso_str not in found_dates:
                found_dates.append(iso_str)

    for m2 in re.finditer(r"\b([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})\b", query):
        if m2.group(1).lower() in _MONTH_MAP:
            month_val, day_val, year_val = _MONTH_MAP[m2.group(1).lower()], int(m2.group(2)), int(m2.group(3))
            iso_str = f"{year_val:04d}-{month_val:02d}-{day_val:02d}"
            if iso_str not in found_dates:
                found_dates.append(iso_str)

    if found_dates:
        params["target_dates"] = found_dates
        params["issue_date"] = found_dates[0]
        if len(found_dates) >= 2:
            sorted_dates = sorted(found_dates)
            params["date_from"] = sorted_dates[0]
            params["date_to"] = sorted_dates[-1]
    else:
        # Named month + year without explicit day (e.g. "August 2026", "during August 2026")
        matched_month = False
        for my_m in re.finditer(r"\b([a-zA-Z]+)\s+(\d{4})\b", query):
            m_name = my_m.group(1).lower()
            if m_name in _MONTH_MAP:
                month_val = _MONTH_MAP[m_name]
                year_val = int(my_m.group(2))
                _, last_day = calendar.monthrange(year_val, month_val)
                d_from = f"{year_val:04d}-{month_val:02d}-01"
                d_to = f"{year_val:04d}-{month_val:02d}-{last_day:02d}"
                params["date_from"] = d_from
                params["date_to"] = d_to
                params["target_dates"] = [d_from, d_to]
                params["issue_date"] = None
                matched_month = True
                break

        if not matched_month:
            # Bare named month without explicit year (e.g. "in August", "during September") -> default to archive year 2026
            for bare_m in re.finditer(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", query, re.I):
                m_name = bare_m.group(1).lower()
                if m_name in _MONTH_MAP:
                    month_val = _MONTH_MAP[m_name]
                    year_val = 2026
                    _, last_day = calendar.monthrange(year_val, month_val)
                    d_from = f"{year_val:04d}-{month_val:02d}-01"
                    d_to = f"{year_val:04d}-{month_val:02d}-{last_day:02d}"
                    params["date_from"] = d_from
                    params["date_to"] = d_to
                    params["target_dates"] = [d_from, d_to]
                    params["issue_date"] = None
                    break

    # 4. Section / Category Extraction (mask matched brands to avoid false bleed like 'The Economic Times' matching 'Economy')
    query_for_sections = query
    for pat, _ in _KNOWN_BRANDS_PATTERNS:
        query_for_sections = pat.sub(" ", query_for_sections)

    for pat, cat_name in _SECTION_PATTERNS:
        if pat.search(query_for_sections):
            params["category_filter"] = cat_name
            break

    # 5. Explicit Headline Extraction from quotes or titles (e.g. article "Headline" or quoted string >= 8 chars)
    hl_match = re.search(r"(?:article|story|headline|titled|report)?\s*[\"“]([^\"”]{8,150})[\"”]", query, re.I)
    if hl_match:
        cand_hl = hl_match.group(1).strip()
        if not any(pat.fullmatch(cand_hl) for pat, _ in _KNOWN_BRANDS_PATTERNS):
            params["headline"] = cand_hl

    return params


def build_targeted_web_query(query: str) -> str:
    """Transform conversational prompts into high-precision search queries."""
    if not query:
        return ""
    cleaned = query.strip()
    for _ in range(3):
        prev = cleaned
        for pattern in _CONVERSATIONAL_PREFIX_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned == prev:
            break
    cleaned = re.sub(r"[\?\.\!]+$", "", cleaned).strip()
    return cleaned if len(cleaned) >= 3 else query.strip()


# Backward compatibility alias
_build_targeted_web_query = build_targeted_web_query

__all__ = [
    "_CONVERSATIONAL_PREFIX_PATTERNS",
    "_KNOWN_BRANDS_PATTERNS",
    "_MONTH_MAP",
    "_SECTION_PATTERNS",
    "_build_targeted_web_query",
    "build_targeted_web_query",
    "extract_parameters_from_query",
]
