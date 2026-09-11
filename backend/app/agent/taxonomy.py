"""Centralized Domain Taxonomy and Classification Engine for NewsLens-AI."""

from __future__ import annotations

import re
from typing import Any

DOMAIN_TAXONOMY: dict[str, dict[str, Any]] = {
    "Economics & Finance": {
        "regex": r"\b(econom(?:y|ic|ics)?|financ(?:e|ial)?|business|markets?|trade|tax(?:es|ation)?|budget|fiscal|monetary|bank(?:s|ing)?|corporate|revenue|gdp|stocks?|shares?)\b",
        "stems": [
            "econom", "financ", "business", "market", "trade", "tax", "solar",
            "power", "seabed", "fund", "money", "bank", "stock", "rupee", "dollar",
            "gdp", "rbi", "corp", "profit",
        ],
        "metric_col": "Key Figures & Metrics",
        "negative_hl": [],
        "required_override": [],
    },
    "Health & Medicine": {
        "regex": r"\b(health|hospitals?|pharma(?:ceutical)?|medicines?|vaccines?|diseases?|doctors?)\b",
        "stems": [
            "health", "hospital", "pharma", "medicine", "doctor", "patient",
            "disease", "vaccin", "virus", "treatment", "care", "clinic", "surgery",
            "drug", "medical", "heart", "infect", "liver", "blood", "cancer",
            "illness", "symptom", "organ", "diet", "nutrition", "wellness", "therapy",
        ],
        "metric_col": "Key Findings & Medical Focus",
        "negative_hl": [
            "when: ", "where: ", "studio xo", "cases still pending", "tax collections",
            "excise duty", "deductions", "cricket", "bjp", "congress",
        ],
        "required_override": [
            "health", "doctor", "hospital", "medicine", "disease", "patient", "heart",
        ],
    },
    "Sports": {
        "regex": r"\b(sports?|cricket|football|tennis|olympics?|tournaments?|match(?:es)?|boxing)\b",
        "stems": [
            "sport", "cricket", "football", "tennis", "olympic", "tournament",
            "match", "boxing", "player", "game",
        ],
        "metric_col": "Key Match Results & Scores",
        "negative_hl": [],
        "required_override": [],
    },
    "Politics & Governance": {
        "regex": r"\b(politic(?:s|al)?|elections?|parliament|assembly|ministers?|cabinet|governance|policy|bills?)\b",
        "stems": [
            "politic", "election", "parliament", "assembly", "minister", "cabinet",
            "governance", "policy", "bill", "party", "vote",
        ],
        "metric_col": "Key Policy Decisions & Statements",
        "negative_hl": [],
        "required_override": [],
    },
    "Crime & Law": {
        "regex": r"\b(crimes?|courts?|legal|law|police|arrest(?:s|ed)?|investigations?|verdicts?|bail)\b",
        "stems": [
            "crime", "court", "legal", "law", "police", "arrest", "investigation",
            "verdict", "bail", "judge", "jail",
        ],
        "metric_col": "Key Legal Proceedings & Verdicts",
        "negative_hl": [],
        "required_override": [],
    },
    "Technology & AI": {
        "regex": r"\b(tech|technology|ai|artificial\s+intelligence|cyber|software)\b",
        "stems": [
            "tech", "technology", "ai", "artificial intelligence", "cyber",
            "software", "digital", "chip",
        ],
        "metric_col": "Key Technical Innovations & Specs",
        "negative_hl": [],
        "required_override": [],
    },
}


def detect_domain_from_query(query: str, evidence_items: list[dict[str, Any]] | None = None) -> str | None:
    """Detect specific domain/sector from user query or evidence manifests."""
    if not query:
        return None
    q_lower = query.lower()
    for domain, spec in DOMAIN_TAXONOMY.items():
        if re.search(spec["regex"], q_lower):
            return domain

    # Inspect evidence items if manifests contain explicit CATEGORY
    if evidence_items:
        for item in evidence_items:
            snip = item.get("snippet", "")
            m = re.search(r"CATEGORY:\s*([A-Za-z &]+)", snip)
            if m:
                cat = m.group(1).strip()
                if any(w in cat.lower() for w in ["econom", "financ", "business", "market"]):
                    return "Economics & Finance"
                if cat in DOMAIN_TAXONOMY:
                    return cat
                for dom in DOMAIN_TAXONOMY:
                    if cat.lower() in dom.lower():
                        return dom
                return cat
    return None


def get_domain_terms(domain: str | None) -> list[str]:
    """Retrieve search token stems for a domain."""
    if not domain:
        return []
    spec = DOMAIN_TAXONOMY.get(domain)
    if spec:
        return list(spec["stems"])
    return [w.lower() for w in re.findall(r"\b\w{3,}\b", domain.lower()) if w.lower() not in {"and", "the", "for"}]


def is_domain_match(item: dict[str, Any], domain: str | None) -> bool:
    """Check if an evidence item strictly matches the target domain."""
    if not domain:
        return True
    spec = DOMAIN_TAXONOMY.get(domain)
    domain_terms = get_domain_terms(domain)

    text_corpus = (
        (item.get("headline") or "")
        + " "
        + (item.get("snippet") or "")
        + " "
        + (item.get("summary") or "")
        + " "
        + (item.get("section") or "")
    ).lower()
    hl = (item.get("headline") or "").lower()

    if spec:
        negative_patterns = spec.get("negative_hl", [])
        required_override = spec.get("required_override", [])
        if (
            negative_patterns
            and any(x in hl for x in negative_patterns)
            and (not required_override or not any(h in hl for h in required_override))
        ):
            return False

    return any(dt in text_corpus for dt in domain_terms)


def score_evidence_item(item: dict[str, Any], domain: str | None) -> int:
    """Score an evidence item for relevance budgeting under a target domain."""
    t = (item.get("headline", "") + " " + item.get("snippet", "") + " " + item.get("summary", "")).lower()
    hl = (item.get("headline", "")).lower()

    if item.get("source_tool") in ("sql_analytics", "coverage_analysis") or "MANIFEST" in t or "RECONCILIATION" in t:
        return 100

    if domain:
        spec = DOMAIN_TAXONOMY.get(domain)
        if (
            spec
            and spec.get("negative_hl")
            and any(x in hl for x in spec["negative_hl"])
            and not any(h in hl for h in spec.get("required_override", []))
        ):
            return -50

        domain_terms = get_domain_terms(domain)
        if any(dt in t for dt in domain_terms):
            return 50

    return 0


__all__ = [
    "DOMAIN_TAXONOMY",
    "detect_domain_from_query",
    "get_domain_terms",
    "is_domain_match",
    "score_evidence_item",
]
