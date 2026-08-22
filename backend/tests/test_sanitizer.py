"""Unit tests for Pre-Chunking Regex Sanitizer (UUIDs, social promos, and printer marks)."""

from __future__ import annotations

from app.ingestion.detector import (
    is_noise_or_promo_text,
    sanitize_block_text,
)


class TestRegexSanitizer:
    """Test regex sanitizer noise purge heuristics."""

    def test_detect_uuid_strings(self) -> None:
        uuid_samples = [
            "3c2f1b0a-7d4e-4f8a-9a6b-123456789abc",
            "98765432-abcd-ef01-2345-6789abcdef01",
            "Page text with UUID 3c2f1b0a-7d4e-4f8a-9a6b-123456789abc embedded",
        ]
        for s in uuid_samples:
            assert is_noise_or_promo_text(s) is True

    def test_detect_whatsapp_and_telegram_promos(self) -> None:
        promo_samples = [
            "Join FREE Whatsapp Channel for daily epaper updates",
            "Join FREE Telegram Channel https://t.me/business_news",
            "https://t.me/joinchat/AAAAAF...",
            "https://chat.whatsapp.com/invite/XYZ123",
            "t.me/mint_epaper_daily",
            "Click here for WhatsApp channel link",
            "Join our Telegram group for free PDFs",
        ]
        for s in promo_samples:
            assert is_noise_or_promo_text(s) is True

    def test_detect_printer_marks(self) -> None:
        printer_samples = [
            "A ND-NDE C M Y K",
            "A ND-NDE C M Y K 22-08-2026",
            "CMYK",
            "cyan magenta yellow black",
            "epaper.livemint.com",
            "PDF version generated on 2026-08-22",
        ]
        for s in printer_samples:
            assert is_noise_or_promo_text(s) is True

    def test_valid_news_text_not_flagged(self) -> None:
        clean_samples = [
            "ISRO successfully launched the SSLV-D3 rocket carrying EOS-08 satellite.",
            "India's cotton imports are estimated to jump 25% following tariff exemptions.",
            "Cognizant beats revenue estimates in Q2 as AI adoption accelerates.",
            "The Reserve Bank of India kept repo rates unchanged at 6.5%.",
        ]
        for s in clean_samples:
            assert is_noise_or_promo_text(s) is False

    def test_sanitize_block_text_strips_noise_lines(self) -> None:
        dirty_block = (
            "ISRO successfully launched the SSLV-D3 rocket.\n"
            "Join FREE Whatsapp Channel for daily epaper\n"
            "The mission marked the third developmental flight.\n"
            "A ND-NDE C M Y K\n"
            "3c2f1b0a-7d4e-4f8a-9a6b-123456789abc"
        )
        cleaned = sanitize_block_text(dirty_block)
        assert "ISRO successfully launched the SSLV-D3 rocket." in cleaned
        assert "The mission marked the third developmental flight." in cleaned
        assert "Whatsapp" not in cleaned
        assert "A ND-NDE C M Y K" not in cleaned
        assert "3c2f1b0a" not in cleaned
