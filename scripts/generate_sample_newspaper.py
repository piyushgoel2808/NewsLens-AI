"""Synthetic newspaper PDF and archive generator for NewsLens-AI testing.

Generates realistic test fixtures:
1. Digital newspaper frontpage with multi-column layout, headlines, and captions.
2. Scanned-style newspaper page (bitmap image with no embedded font text layer).
3. Multi-page newspaper issue (3 pages: Frontpage, Editorial, Business).
4. ZIP archive bundling multiple issues for intake testing.

Usage:
    python scripts/generate_sample_newspaper.py
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw


def create_digital_page(
    headline: str,
    subheadline: str,
    articles: list[tuple[str, str]],
    date_str: str = "OCTOBER 24, 1929",
    newspaper_title: str = "THE METROPOLIS CHRONICLE",
) -> fitz.Document:
    """Create a digital vector PDF with realistic multi-column newspaper layout."""
    doc = fitz.open()
    # Standard BroadSheet page size in points: 595 x 842 (A4) or 612 x 792 (Letter)
    page = doc.new_page(width=595, height=842)

    # 1. Header & Masthead
    page.draw_rect(fitz.Rect(30, 25, 565, 27), color=(0, 0, 0), fill=(0, 0, 0))
    page.insert_text(
        fitz.Point(35, 20),
        f"VOL. LXVIII NO. 22,415 • {date_str} • THREE CENTS",
        fontsize=8,
        fontname="helv",
    )
    # Title Masthead
    page.insert_text(
        fitz.Point(80, 65),
        newspaper_title,
        fontsize=28,
        fontname="times-bold",
    )
    page.draw_line(fitz.Point(30, 75), fitz.Point(565, 75), color=(0, 0, 0), width=1.5)
    page.draw_line(fitz.Point(30, 78), fitz.Point(565, 78), color=(0, 0, 0), width=0.5)

    # 2. Main Banner Headline
    page.insert_text(
        fitz.Point(35, 105),
        headline,
        fontsize=20,
        fontname="times-bold",
    )
    page.insert_text(
        fitz.Point(35, 122),
        subheadline,
        fontsize=12,
        fontname="times-italic",
    )
    page.draw_line(fitz.Point(30, 132), fitz.Point(565, 132), color=(0.5, 0.5, 0.5), width=0.5)

    # 3. 3-Column Article Layout
    col_width = 165
    gutter = 15
    start_x = 35
    start_y = 145

    for i, (art_title, art_body) in enumerate(articles[:3]):
        cx0 = start_x + i * (col_width + gutter)
        cy0 = start_y

        # Column rule
        if i > 0:
            page.draw_line(
                fitz.Point(cx0 - (gutter / 2), start_y),
                fitz.Point(cx0 - (gutter / 2), 800),
                color=(0.8, 0.8, 0.8),
                width=0.5,
            )

        # Article Headline
        page.insert_text(
            fitz.Point(cx0, cy0 + 12),
            art_title,
            fontsize=13,
            fontname="times-bold",
        )
        page.insert_text(
            fitz.Point(cx0, cy0 + 24),
            "By Staff Correspondent",
            fontsize=8,
            fontname="helv",
        )

        # Article Body Text Box
        rect = fitz.Rect(cx0, cy0 + 32, cx0 + col_width, 790)
        page.insert_textbox(
            rect,
            art_body,
            fontsize=9,
            fontname="times-roman",
            align=fitz.TEXT_ALIGN_JUSTIFY,
        )

    # Footer
    page.draw_line(fitz.Point(30, 810), fitz.Point(565, 810), color=(0, 0, 0), width=0.5)
    page.insert_text(fitz.Point(270, 825), "Page 1", fontsize=8, fontname="helv")

    return doc


def create_scanned_page_pdf(
    headline: str,
    body_text: str,
) -> fitz.Document:
    """Create a scanned-style PDF by rendering text into a bitmap image without font stream."""
    img = Image.new("RGB", (1200, 1600), color=(245, 243, 235))  # Aged paper color
    draw = ImageDraw.Draw(img)

    # Draw headline
    draw.text((80, 80), headline, fill=(20, 20, 20))
    draw.line([(80, 140), (1120, 140)], fill=(40, 40, 40), width=3)

    # Draw body paragraphs
    y = 160
    for line in body_text.split("\n"):
        draw.text((80, y), line, fill=(30, 30, 30))
        y += 24

    # Convert PIL Image to PDF with zero text layer
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=img_bytes)
    return doc


def create_multi_page_issue() -> fitz.Document:
    """Create a realistic 3-page newspaper issue."""
    doc = fitz.open()

    articles_p1 = [
        (
            "MARKET IN SEVERE TUMBLE",
            "Heavy selling swept across the financial district yesterday as trading volumes reached record numbers. Bankers and treasury officials convened late in the evening to review credit facilities and restore order to the exchange. Leading industrial shares declined sharply during the afternoon session before stabilizing.",
        ),
        (
            "MAYOR PLANS TRANSIT EXPANSION",
            "City officials announced a broad infrastructure program aimed at connecting eastern suburbs via rapid transit lines. Construction is expected to commence early next spring, providing thousands of municipal jobs.",
        ),
        (
            "NEW OCEAN LINER DOCKS",
            "The pride of the transatlantic fleet arrived in the harbor this morning, carrying over two thousand passengers and dignitaries from European capitals.",
        ),
    ]

    articles_p2 = [
        (
            "EDITORIAL: THE ECONOMIC CROSSROADS",
            "Prudence must guide fiscal policy in the weeks ahead. Industrial productivity remains at historically robust levels, yet speculative excesses require calm deliberation and disciplined stewardship.",
        ),
        (
            "LETTERS TO THE EDITOR",
            "Citizens voice opinions regarding public park preservation and harbor dredging initiatives.",
        ),
        (
            "WEATHER OUTLOOK",
            "Brisk autumn winds from the northwest with clear skies through the weekend.",
        ),
    ]

    articles_p3 = [
        (
            "COMMERCE & INDUSTRY REPORT",
            "Steel manufacturing reports steady output while grain futures showed modest gains in midwestern markets. Automobile production figures exceeded quarterly forecasts.",
        ),
        (
            "SHIPPING INTELLIGENCE",
            "Vessels cleared for departure at dawn included cargo freighters bound for South American ports.",
        ),
        (
            "COMMODITY PRICES",
            "Wheat, corn, and cotton prices held firm across regional trading floors.",
        ),
    ]

    for p_num, arts in enumerate([articles_p1, articles_p2, articles_p3], start=1):
        p_doc = create_digital_page(
            headline=f"SECTION {p_num}: " + arts[0][0],
            subheadline="Special Dispatch to The Chronicle",
            articles=arts,
            newspaper_title="THE METROPOLIS CHRONICLE",
        )
        doc.insert_pdf(p_doc)
        p_doc.close()

    return doc


def generate_all_samples(output_dir: Path) -> dict[str, Path]:
    """Generate all sample fixtures into output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    # 1. Single-page digital frontpage
    doc1 = create_digital_page(
        headline="STOCKS SLUMP IN RECORD TRADING SESSION",
        subheadline="Billions in Paper Values Erased as Volume Overwhelms Ticker",
        articles=[
            (
                "PANIC ON WALL STREET",
                "Wild fluctuations unsettled the trading floor throughout yesterday. A group of influential bankers pledged financial support to preserve market stability.",
            ),
            (
                "CITY HALL APPROVES BOND",
                "The Board of Estimate voted unanimously to authorize waterworks improvements spanning three municipal districts.",
            ),
            (
                "AVIATOR SETS SPEED RECORD",
                "Captain Reynolds completed the coast-to-coast flight in under sixteen hours, establishing a new national benchmark.",
            ),
        ],
    )
    p1_path = output_dir / "sample_digital_frontpage.pdf"
    doc1.save(str(p1_path))
    doc1.close()
    generated["digital_frontpage"] = p1_path

    # 2. Scanned image-only PDF
    doc2 = create_scanned_page_pdf(
        headline="HISTORIC SCAN: ARCHIVAL ISSUE 1898",
        body_text=(
            "WAR DECLARED IN THE CARIBBEAN\n\n"
            "Forces mobilized across southern naval bases following congressional declaration.\n"
            "Troop transports are standing by in key harbors awaiting final sailing orders.\n"
            "Public gatherings held in town squares across the nation in patriotic support."
        ),
    )
    p2_path = output_dir / "sample_scanned_page.pdf"
    doc2.save(str(p2_path))
    doc2.close()
    generated["scanned_page"] = p2_path

    # 3. 3-Page Issue
    doc3 = create_multi_page_issue()
    p3_path = output_dir / "sample_multi_page_issue.pdf"
    doc3.save(str(p3_path))
    doc3.close()
    generated["multi_page_issue"] = p3_path

    # 4. ZIP Archive of multiple issues
    zip_path = output_dir / "sample_newspaper_archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(p1_path, arcname="issues/1929-10-24_morning.pdf")
        zf.write(p3_path, arcname="issues/1929-10-25_morning.pdf")
    generated["zip_archive"] = zip_path

    print(f"✓ Generated {len(generated)} sample fixtures in {output_dir}")
    for k, v in generated.items():
        print(f"  - {k}: {v.name} ({v.stat().st_size:,} bytes)")

    return generated


if __name__ == "__main__":
    fixtures_dir = Path(__file__).resolve().parent.parent / "backend" / "tests" / "fixtures"
    generate_all_samples(fixtures_dir)
