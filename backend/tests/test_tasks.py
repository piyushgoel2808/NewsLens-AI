"""Unit and integration tests for tasks.py, masthead detection, and pipeline execution."""

from __future__ import annotations

from datetime import date
import pytest

from app.ingestion.detector import DigitalTextBlock
from app.ingestion.tasks import detect_masthead_and_date
from app.providers.base import OCRBlock


class TestMastheadAndDateDetection:
    """Test dynamic masthead and publication date extraction from Page 1 blocks."""

    def test_detect_mint_masthead_and_date_digital(self) -> None:
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="MINT | THURSDAY, 30 JULY 2026 | NEW DELHI",
                bbox=(50.0, 10.0, 950.0, 40.0),
            ),
            DigitalTextBlock(
                block_id=1,
                text="Cognizant beats IT peers, cuts outlook",
                bbox=(50.0, 100.0, 950.0, 150.0),
            ),
        ]
        brand, pub_date = detect_masthead_and_date(blocks, height_px=1400.0)
        assert brand == "Mint"
        assert pub_date == date(2026, 7, 30)

    def test_detect_business_standard_and_date_ocr(self) -> None:
        blocks = [
            OCRBlock(
                text="BUSINESS STANDARD | MUMBAI | AUGUST 21, 2026",
                bbox=(50.0, 10.0, 950.0, 40.0),
                confidence=0.98,
            ),
            OCRBlock(
                text="Market rallies to fresh lifetime highs",
                bbox=(50.0, 100.0, 950.0, 150.0),
                confidence=0.95,
            ),
        ]
        brand, pub_date = detect_masthead_and_date(blocks, height_px=1400.0)
        assert brand == "Business Standard"
        assert pub_date == date(2026, 8, 21)

    def test_ignore_blocks_lower_in_page(self) -> None:
        """Blocks far down the page should not override masthead."""
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="Some article mentioning 15 August 1947",
                bbox=(50.0, 600.0, 950.0, 650.0),
            )
        ]
        brand, pub_date = detect_masthead_and_date(blocks, height_px=1400.0)
        assert brand is None
        assert pub_date is None


class TestTaskIngestionStorageKey:
    """Test Celery task execution with object storage streaming references."""

    def test_celery_task_dispatches_with_storage_key(self) -> None:
        from unittest.mock import AsyncMock, patch
        from app.ingestion.tasks import process_issue_ingestion_task

        with patch("app.ingestion.tasks.run_ingestion_pipeline", new_callable=AsyncMock) as mock_pipe:
            mock_pipe.return_value = {"issue_id": 999, "status": "completed"}
            # Call underlying task function directly
            result = process_issue_ingestion_task.apply(
                args=(999,),
                kwargs={"storage_key": "originals/1/mint.pdf", "dpi": 150},
            ).get()

        assert result["status"] == "completed"
        assert result["issue_id"] == 999
        mock_pipe.assert_called_once_with(
            issue_id=999,
            pdf_bytes=None,
            storage_key="originals/1/mint.pdf",
            dpi=150,
            parser_engine="auto",
        )

    @pytest.mark.asyncio
    async def test_execute_ingestion_pipeline_streams_from_storage(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch
        from app.ingestion.tasks import _execute_ingestion_pipeline

        mock_store = AsyncMock()
        mock_store.get.return_value = b"%PDF-mocked-stream"

        mock_session = AsyncMock()
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_res

        class MockSessionContext:
            async def __aenter__(self) -> AsyncMock:
                return mock_session

            async def __aexit__(self, *args: object) -> None:
                pass

        mock_session_factory = MagicMock(side_effect=MockSessionContext)

        with (
            patch("app.ingestion.tasks.PDFRasterizer") as mock_raster_cls,
            patch("app.ingestion.tasks.PDFPageDetector") as mock_detector_cls,
            patch("app.ingestion.tasks.UnifiedExtractor") as mock_extractor_cls,
            patch("app.ingestion.tasks.ConsensusExtractor") as mock_consensus_cls,
            patch("app.ingestion.tasks.get_settings") as mock_settings,
        ):
            mock_settings.return_value.minio.bucket_originals = "test-originals"
            mock_raster = AsyncMock()
            mock_raster.rasterize_pdf_bytes.return_value = []
            mock_raster_cls.return_value = mock_raster

            mock_detector = MagicMock()
            mock_detector.analyze_document_bytes.return_value = []
            mock_detector_cls.return_value = mock_detector

            mock_consensus = MagicMock()
            mock_consensus.extract_consensus.return_value = ("Daily News", date(2026, 8, 1), {})
            mock_consensus_cls.return_value = mock_consensus

            await _execute_ingestion_pipeline(
                issue_id=55,
                pdf_bytes=None,
                storage_key="originals/10/daily.pdf",
                minio=mock_store,
                session_factory=mock_session_factory,
            )

            mock_store.get.assert_called_once_with(
                bucket="test-originals",
                key="originals/10/daily.pdf",
            )
            mock_raster.rasterize_pdf_bytes.assert_called_once_with(
                pdf_bytes=b"%PDF-mocked-stream",
                issue_id=55,
                dpi=150,
            )

    def test_cross_encoder_process_wide_cache(self) -> None:
        from unittest.mock import MagicMock, patch
        from app.retrieval.reranker import CrossEncoderReranker, _SHARED_CROSS_ENCODERS

        _SHARED_CROSS_ENCODERS.clear()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95]

        with patch("sentence_transformers.CrossEncoder", return_value=mock_model) as mock_cls:
            r1 = CrossEncoderReranker(model_name="test-cross-encoder-model")
            scores1 = r1.predict([("query", "doc")])
            assert scores1 == [0.95]

            # Second instance must reuse cached model without calling CrossEncoder again
            r2 = CrossEncoderReranker(model_name="test-cross-encoder-model")
            scores2 = r2.predict([("query2", "doc2")])
            assert scores2 == [0.95]

            assert mock_cls.call_count == 1



