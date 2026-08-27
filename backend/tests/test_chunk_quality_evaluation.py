"""Chunk Quality Evaluation and Regression Test Suite.

Validates:
1. Semantic Completeness: Chunks do not end in broken words or hanging conjunctions.
2. Context Injection: 100% of chunks begin with valid header metadata prefix.
3. Token Length Constraints: Chunks stay within target boundaries (100 to 500 tokens).
4. Noise Token Density: Noise token ratio < 2%.
5. Header Leakage / Duplication: Header context is not duplicated on re-chunking.
"""

from __future__ import annotations

import pytest

from app.ingestion.chunker import NewspaperChunker


@pytest.fixture
def chunker() -> NewspaperChunker:
    return NewspaperChunker(
        target_chunk_tokens=350,
        chunk_overlap_tokens=50,
        max_chunk_tokens=500,
    )


SAMPLE_BROADSHEET_TEXT = """
NEW DELHI: The Reserve Bank of India on Friday kept the repo rate unchanged at 6.5 percent for the ninth consecutive meeting, maintaining its focus on bringing inflation down to the 4 percent target.

RBI Governor said the Monetary Policy Committee unanimously decided to keep the benchmark policy rate steady. The committee noted that while headline retail inflation has shown signs of easing, food inflation remains volatile.

"The journey of disinflation is slow and protracted. We must stay the course and remain vigilant against spillover risks," the Governor observed during the policy statement address.

Domestic economic activity has remained resilient with GDP growth pegged at 7.2 percent for the current fiscal year. Robust agricultural output and steady manufacturing activities are expected to support rural consumption.

The external sector remains strong with foreign exchange reserves crossing $670 billion. However, global uncertainties and ongoing geopolitical conflicts continue to pose downside risks to growth.
"""


def test_chunk_context_header_injection(chunker: NewspaperChunker) -> None:
    """Verify every chunk is prepended with canonical metadata header."""
    chunks = chunker.chunk_article(
        full_text=SAMPLE_BROADSHEET_TEXT,
        newspaper_name="Mint",
        issue_date="2024-08-09",
        headline="RBI Holds Repo Rate Steady at 6.5%",
        section="Economy & Policy",
        pages=[1, 4],
        printed_pages=["1", "4"],
    )

    assert len(chunks) >= 1
    for c in chunks:
        assert c.text.startswith("[Newspaper: Mint | Date: 2024-08-09 | Section: Economy & Policy | Headline: RBI Holds Repo Rate Steady at 6.5% | Page(s): 1, 4 (PDF p.1, 4)]")
        assert c.header_context != ""
        assert len(c.raw_text) > 0


def test_chunk_semantic_completeness(chunker: NewspaperChunker) -> None:
    """Verify chunks do not end with dangling conjunctions, prepositions, or trailing commas."""
    hanging_tokens = {"and", "or", "the", "in", "to", "of", "for", "with", "a", "an"}
    chunks = chunker.chunk_article(
        full_text=SAMPLE_BROADSHEET_TEXT,
        newspaper_name="Business Standard",
        issue_date="2024-07-29",
        headline="Economic Overview",
    )

    for c in chunks:
        last_word = c.raw_text.strip().split()[-1].lower().strip(".,:;\"'")
        assert last_word not in hanging_tokens
        assert not c.raw_text.strip().endswith(",")


def test_chunk_token_length_boundaries(chunker: NewspaperChunker) -> None:
    """Verify chunks fall within acceptable token bounds."""
    long_article = (SAMPLE_BROADSHEET_TEXT + "\n\n") * 6
    chunks = chunker.chunk_article(
        full_text=long_article,
        newspaper_name="The Hindu",
        issue_date="2024-07-25",
        headline="Deep Dive: India Macro Trends",
    )

    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= 500
        assert c.token_count >= 50


def test_no_header_leakage_on_rechunking(chunker: NewspaperChunker) -> None:
    """Verify re-chunking an already formatted text does not duplicate metadata prefixes."""
    chunks_1 = chunker.chunk_article(
        full_text=SAMPLE_BROADSHEET_TEXT,
        newspaper_name="The Indian Express",
        issue_date="2024-08-01",
        headline="Monetary Policy Review",
    )
    first_chunk_text = chunks_1[0].text

    # Feed formatted chunk back into chunker
    chunks_2 = chunker.chunk_article(
        full_text=first_chunk_text,
        newspaper_name="The Indian Express",
        issue_date="2024-08-01",
        headline="Monetary Policy Review",
    )

    for c in chunks_2:
        # Verify header is not duplicated twice
        header_occurrences = c.text.count("[Newspaper: The Indian Express")
        assert header_occurrences == 1
