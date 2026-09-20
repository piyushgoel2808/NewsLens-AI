"""Unit tests for Metadata Extractor (NER, Topics, and Summarization)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.ingestion.metadata_extractor import MetadataExtractor


class TestMetadataExtractor:
    """Test suite for MetadataExtractor."""

    def test_extract_entities_from_text(self) -> None:
        extractor = MetadataExtractor(db=MagicMock())
        headline = "PRIME MINISTER MEETS BANK CHIEFS IN NEW DELHI"
        text = (
            "Prime Minister John Smith met with the Reserve Bank Governor and "
            "leaders of Apex Corp yesterday in New Delhi to discuss market liquidity."
        )

        entities = extractor.extract_entities_from_text(text, headline=headline)
        assert len(entities) >= 3

        names = [e.name for e in entities]
        assert any("New Delhi" in n or "Delhi" in n for n in names)
        assert any(e.type == "location" for e in entities)
        assert any(e.type == "org" for e in entities)

        # Check salience ranking
        assert entities[0].salience_score >= 0.50

    def test_extract_topics_from_text(self) -> None:
        extractor = MetadataExtractor(db=MagicMock())
        headline = "STOCK MARKETS SURGE AFTER CENTRAL BANK RATE CUT"
        text = "Equity indices rose sharply as banking shares rallied on interest rate adjustments."

        topics = extractor.extract_topics_from_text(text, headline=headline)
        assert len(topics) >= 2

        topic_names = [t.name for t in topics]
        assert "Markets" in topic_names or "Banking" in topic_names or "Stocks" in topic_names

    def test_generate_summary(self) -> None:
        extractor = MetadataExtractor(db=MagicMock())
        headline = "MAJOR TRADE DEAL RATIFIED"
        text = (
            "Lawmakers approved the historic trade pact following intense debate. "
            "The agreement eliminates tariffs across key manufacturing categories.\n\n"
            "Business groups praised the decision as a milestone for commerce."
        )

        summary = extractor.generate_summary(text, headline=headline)
        assert "historic trade pact" in summary
        assert len(summary) > 20
        assert len(summary) <= 500

    @pytest.mark.asyncio
    async def test_process_and_persist_bulk_metadata(self) -> None:
        from unittest.mock import AsyncMock

        mock_db = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.add = MagicMock()

        mock_res = MagicMock()
        mock_res.scalars.return_value.all.return_value = []
        mock_res.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_res)

        extractor = MetadataExtractor(db=mock_db)
        articles_data = [
            (1, "PM VISITS MUMBAI", "Prime Minister arrived in Mumbai yesterday for infrastructure summit."),
            (2, "ELECTION COMMISSION MEETS", "Chief Election Commissioner reviewed security preparations in New Delhi."),
        ]

        results = await extractor.process_and_persist_bulk_metadata(articles_data)
        assert len(results) == 2
        assert 1 in results and 2 in results
        assert len(results[1].entities) > 0
        assert len(results[2].entities) > 0
        assert mock_db.flush.called
