"""Unit tests for PDFPageDetector digital vs scanned classification and gibberish detection."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.detector import PageType, PDFPageDetector, is_text_gibberish

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestPDFPageDetector:
    """Test digital text layer detection, gibberish heuristics, and classification."""

    def test_detect_digital_page(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"
        detector = PDFPageDetector()
        results = detector.analyze_document_bytes(pdf_path.read_bytes())

        assert len(results) == 1
        res = results[0]
        assert res.page_type in (PageType.DIGITAL, PageType.HYBRID)
        assert res.requires_ocr is False
        assert res.character_count > 100
        assert res.word_count > 20
        assert len(res.blocks) >= 3
        # Ensure headline candidate detected
        assert any(b.is_heading_candidate for b in res.blocks)

    def test_detect_scanned_page(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_scanned_page.pdf"
        detector = PDFPageDetector()
        results = detector.analyze_document_bytes(pdf_path.read_bytes())

        assert len(results) == 1
        res = results[0]
        assert res.page_type == PageType.SCANNED
        assert res.requires_ocr is True
        assert res.character_count < 80

    def test_detect_multi_page_document(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_multi_page_issue.pdf"
        detector = PDFPageDetector()
        results = detector.analyze_document_bytes(pdf_path.read_bytes())

        assert len(results) == 3
        for i, res in enumerate(results, start=1):
            assert res.page_number == i
            assert res.requires_ocr is False
            assert len(res.blocks) >= 1

    def test_is_text_gibberish_valid_text(self) -> None:
        valid_text = (
            "The stock market experienced a major surge yesterday as investors reacted to "
            "positive economic indicators. Central bank officials stated that inflation "
            "remains under control while employment figures showed steady improvement."
        )
        assert is_text_gibberish(valid_text) is False

    def test_is_text_gibberish_replacement_characters(self) -> None:
        corrupted_text = (
            "The market was \ufffd\ufffd\ufffd " * 20
            + " and continued to \ufffd\ufffd fall sharply."
        )
        assert is_text_gibberish(corrupted_text, threshold=0.10) is True

    def test_is_text_gibberish_font_mapping_b_repetition(self) -> None:
        # Font unmapped glyph loop (as seen in Business Standard PDF)
        b_loop_text = "NEW DELHI | TUESDAY 7 JULY 2026\n" + "b" * 600
        assert is_text_gibberish(b_loop_text) is True

    def test_is_text_gibberish_control_characters(self) -> None:
        # Unmapped TrueType subset codes (as seen in Indian Express PDF)
        control_text = "".join(chr(i % 30 + 1) for i in range(500))
        assert is_text_gibberish(control_text) is True

    def test_detect_advertisement_heuristics(self) -> None:
        from unittest.mock import MagicMock

        detector = PDFPageDetector()
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.load_page.return_value = mock_page

        # Mock page with advertisement text and graphics
        mock_page.get_text.side_effect = lambda arg: (
            {"blocks": []}
            if arg == "dict"
            else "ADVERTISEMENT\nExclusive luxury apartments. Call now 1800-000. T&C Apply."
        )
        mock_page.get_images.return_value = [("img1", 0, 0, 0, 0, 0, 0, 0, 0)]

        res = detector.analyze_page(mock_doc, 0)
        assert res.page_type == PageType.ADVERTISEMENT
        assert res.is_advertisement is True
        assert res.requires_ocr is False

    def test_corporate_news_article_never_flagged_as_advertisement(self) -> None:
        """Verify long corporate M&A / finance news is not falsely flagged as an ad."""
        from app.ingestion.detector import check_is_advertisement_text, is_page_advertisement

        corporate_news = (
            "ChrysCapital buys controlling stake in Novartis India for Rs 1,800 crore. "
            "Private equity major ChrysCapital on Thursday announced it has agreed "
            "to acquire a controlling stake in the pharmaceutical company. "
            "Investment bankers from Goldman Sachs and Kotak Mahindra advised on the deal. "
            "Market analysts noted that equity shares rallied 4.5% on the stock exchange. "
        ) * 15  # ~600 words

        assert check_is_advertisement_text(corporate_news) is False
        assert is_page_advertisement(
            page_blocks_text=corporate_news,
            page_number=12,
            total_pages=16,
            word_count=len(corporate_news.split()),
        ) is False

    def test_retail_tech_launch_spread_detected_as_ad(self) -> None:
        """Verify smartphone / consumer tech launch spreads with pricing matrices are flagged."""
        from app.ingestion.detector import check_is_advertisement_text, is_page_advertisement

        retail_spread = (
            "Experience Next-Gen Innovation. Starting at Rs 79,999. "
            "Pre-order now and get an exchange bonus of up to Rs 10,000. "
            "No Cost EMI available starting at Rs 3,999/month. Zero down payment options. "
            "Visit us at www.brandstore.com or download the app. T&C Apply."
        )
        assert check_is_advertisement_text(retail_spread) is True
        assert is_page_advertisement(
            page_blocks_text=retail_spread,
            page_number=1,
            total_pages=16,
            word_count=len(retail_spread.split()),
        ) is True

    def test_real_estate_and_auto_ad_detected(self) -> None:
        """Verify real estate and automotive ads are accurately identified."""
        from app.ingestion.detector import check_is_advertisement_text, is_page_advertisement

        real_estate_ad = (
            "Grand Launch in Central City. Premium 3 BHK & 4 BHK Residences. "
            "Ready to move apartments with possession soon. RERA Registration No: PRM/KA/12345. "
            "Inaugural offer: Flat 10% off on all bookings this weekend. "
            "Book now at our authorized dealership or call toll free 1800-456-7890."
        )
        assert check_is_advertisement_text(real_estate_ad) is True
        assert is_page_advertisement(
            page_blocks_text=real_estate_ad,
            page_number=6,
            total_pages=16,
            word_count=len(real_estate_ad.split()),
        ) is True

    def test_statutory_and_ipo_financial_notices(self) -> None:
        """Verify legal disclosures, tender notices, and IPO capital market notices."""
        from app.ingestion.detector import check_is_advertisement_text, is_page_advertisement

        statutory_notice = (
            "PUBLIC NOTICE. Notice is hereby given before the Hon'ble National "
            "Company Law Tribunal, Mumbai Bench. In the matter of Insolvency and Bankruptcy Code. "
            "E-Auction sale notice for assets. For details visit www.bankauctions.in. "
            "Corrigendum issued."
        )
        assert check_is_advertisement_text(statutory_notice) is True
        assert is_page_advertisement(
            page_blocks_text=statutory_notice,
            page_number=14,
            total_pages=16,
            word_count=len(statutory_notice.split()),
        ) is True

    def test_editorial_contrast_safety_guardrail(self) -> None:
        """Verify heavy editorial news pages with >= 3 editorial markers are protected."""
        from app.ingestion.detector import is_page_advertisement

        editorial_page = (
            "Union Budget Analysis and Fiscal Deficit Projections. "
            "From our Special Correspondent in New Delhi. Express News Service reports. "
            "The finance ministry bureau stated that capital expenditure will rise. "
            "Edited by Senior Political Editor. Continued on page 4 with full sector breakdown. "
            "Market commentators noted that several firms announced price bands and shares. "
        ) * 10  # ~400 words with 5 editorial markers

        assert is_page_advertisement(
            page_blocks_text=editorial_page,
            page_number=5,
            total_pages=16,
            word_count=len(editorial_page.split()),
        ) is False

    def test_tc_trapdoor_dense_ad_spread_flagged_regardless_of_word_count(self) -> None:
        """Verify cover wraps with dense terms and conditions (212 words) trigger T&C trapdoor."""
        from app.ingestion.detector import check_is_advertisement_text, is_page_advertisement

        dense_tech_wrap = (
            "Unfold the Future with the All-New Next-Gen Foldable Smartphone. "
            "Pre-order now at all authorized retail partner stores across India. "
            "Experience revolutionary dual-screen multitasking and ultra-vivid clarity "
            "with reinforced aerospace-grade aluminum chassis and titanium hinges. "
            "Equipped with next-generation neural processing unit and all-day battery life. "
            "Get up to Rs 15,000 exchange bonus on select older smartphone devices. "
            "Zero down payment options available across all leading retail partners. "
            "No Cost EMI options available from top financial partners starting at Rs 4,999/mo. "
            "Complimentary 1-year damage protection plan included during the inaugural period. "
            "Terms and conditions apply. Prices inclusive of all taxes. Offers valid at the "
            "sole discretion of the manufacturer and authorized distribution partners. "
            "Images simulated for illustrative purposes. Screen simulated for display preview. "
            "Optional accessories and protective covers sold separately. Colors subject to stock. "
            "Visit us at www.brandmobile.com or call toll free 1800-200-8899 for product specs. "
            "Annual percentage rate may vary by banking partner. Cashback credited in 90 days. "
            "Extended warranty valid only upon online registration within 15 days of invoice. "
            "Prices subject to change without prior notice. Standard retailer terms apply. "
            "Product warranty governed by regional service center policy guidelines. T&C Apply."
        )  # ~200 words with 6+ distinct T&C / legal phrases

        assert len(dense_tech_wrap.split()) > 180
        assert check_is_advertisement_text(dense_tech_wrap) is True
        assert is_page_advertisement(
            page_blocks_text=dense_tech_wrap,
            page_number=1,
            total_pages=16,
            word_count=len(dense_tech_wrap.split()),
        ) is True
