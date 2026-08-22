"""Unit tests for Newspaper-Aware Hierarchical Chunker."""

from __future__ import annotations

from app.ingestion.chunker import NewspaperChunker


class TestNewspaperChunker:
    """Test suite for NewspaperChunker."""

    def test_create_header_context(self) -> None:
        chunker = NewspaperChunker()
        header = chunker.create_header_context(
            newspaper_name="The Metropolis Chronicle",
            issue_date="1930-04-15",
            headline="WAR DECLARED IN THE CARIBBEAN",
            section="Front Page",
            pages=[1, 4],
        )

        assert "[Newspaper: The Metropolis Chronicle" in header
        assert "Date: 1930-04-15" in header
        assert "Headline: WAR DECLARED IN THE CARIBBEAN" in header
        assert "Section: Front Page" in header
        assert "Page(s): 1, 4" in header

    def test_chunk_empty_text_returns_empty(self) -> None:
        chunker = NewspaperChunker()
        assert chunker.chunk_article("") == []

    def test_chunk_short_article_single_chunk(self) -> None:
        chunker = NewspaperChunker(target_chunk_tokens=300)
        text = "This is a short single-paragraph article explaining a local event in detail."
        chunks = chunker.chunk_article(
            full_text=text,
            newspaper_name="Daily News",
            issue_date="2026-08-21",
            headline="LOCAL EVENT REPORT",
        )

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert "[Newspaper: Daily News" in chunks[0].text
        assert "LOCAL EVENT REPORT" in chunks[0].text
        assert text in chunks[0].text
        assert chunks[0].token_count > 10

    def test_chunk_multi_paragraph_splitting(self) -> None:
        chunker = NewspaperChunker(target_chunk_tokens=50, chunk_overlap_tokens=20)
        paras = [f"Paragraph {i}: " + " ".join(["word" for _ in range(35)]) for i in range(5)]
        full_text = "\n\n".join(paras)

        chunks = chunker.chunk_article(
            full_text=full_text,
            newspaper_name="Financial Times",
            issue_date="2026-08-21",
            headline="MARKET EXPANSION",
        )

        assert len(chunks) >= 3
        for i, chk in enumerate(chunks):
            assert chk.chunk_index == i
            assert chk.header_context in chk.text
            assert chk.token_count > 0
