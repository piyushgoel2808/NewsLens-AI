"""Pydantic schemas for Single-Pass Newspaper Layout Extraction and Enrichment.

Used with Google Gemini Vision and Ollama Gemma 4 structured outputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Section Taxonomy & Article Types
# ---------------------------------------------------------------------------

SectionType = Literal[
    "Front Page",
    "National",
    "International",
    "Economy & Policy",
    "Markets & Data",
    "Corporate & Industry",
    "Banking & Finance",
    "Deals, Tech & Startups",
    "Opinion & Editorial",
    "Sports",
    "Science & Environment",
    "Life & Culture",
    "Personal Finance",
    "Law & Justice",
    "Defense & Security",
    "Real Estate & Infrastructure",
    "Advertisements & Notices",
    "News Briefs",
]

ArticleGenre = Literal[
    "news",
    "editorial",
    "opinion",
    "analysis",
    "advertisement",
    "sidebar",
    "photo_caption",
    "table_data",
    "letter",
    "obituary",
    "review",
    "teaser",
    "index",
    "unknown",
]

ProminenceTier = Literal["lead", "major", "standard", "minor", "filler"]


# ---------------------------------------------------------------------------
# Phase 1 Schemas: Page Layout & Article Skeletons
# ---------------------------------------------------------------------------

class ArticleSkeleton(BaseModel):
    """Skeleton of a single article or item detected on a newspaper page."""

    headline: str = Field(
        description="Headline or title of the article verbatim. For ads, use '[Advertisement] <Company/Subject>'."
    )
    subheadline: str | None = Field(
        None, description="Subheadline, kicker, deck, or strapline if present."
    )
    byline: str | None = Field(
        None, description="Author, reporter name, agency stamp (e.g. Reuters, PTI, Bureau) if present."
    )
    body_text: str | None = Field(
        None, description="Extracted body text paragraphs belonging to this article."
    )
    article_type: ArticleGenre = Field(
        default="news",
        description="Journalism genre of the article."
    )
    section: SectionType = Field(
        default="National",
        description="Categorized newspaper section."
    )
    prominence: ProminenceTier = Field(
        default="standard",
        description="Visual editorial prominence on the page (lead=top banner/main story, major=multi-column prominent, standard=regular article, minor=brief/capsule, filler=tiny snippet)."
    )
    bbox: list[float] = Field(
        description="Normalized bounding box [ymin, xmin, ymax, xmax] scaled 0 to 1000, or absolute pixel coordinates [x0, y0, x1, y1]."
    )
    continues_to_page: int | None = Field(
        None, description="Page number where this story continues (e.g. from 'Continued on Page 4')."
    )
    continued_from_page: int | None = Field(
        None, description="Page number this story is continued from (e.g. from 'Continued from Page 1')."
    )
    has_table: bool = Field(
        default=False, description="True if this item contains structured tabular financial/statistical data."
    )
    has_photo: bool = Field(
        default=False, description="True if this item is accompanied by a photo or illustration."
    )


class PageLayoutExtraction(BaseModel):
    """Phase 1: Complete page layout analysis and all contained articles."""

    page_number: int = Field(
        default=1, description="Physical PDF page index (1-indexed)."
    )
    newspaper_brand: str | None = Field(
        None, description="Newspaper brand/title if visible in masthead or header/footer (e.g. 'Mint', 'Business Standard', 'The Hindu')."
    )
    issue_date: str | None = Field(
        None, description="Publication date in YYYY-MM-DD format if visible on the page."
    )
    printed_page_number: str | None = Field(
        None, description="Printed newspaper page number/folio string (e.g. '1', 'A-3', 'IV')."
    )
    is_advertisement_page: bool = Field(
        default=False, description="True if the entire page is an advertisement wrap or jacket."
    )
    articles: list[ArticleSkeleton] = Field(
        default_factory=list,
        description="List of all discrete articles, sidebars, tables, and notices on this page in reading order."
    )


# ---------------------------------------------------------------------------
# Phase 2 Schemas: Article Enrichment
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    """Named Entity extracted from article text."""

    name: str = Field(description="Entity name (e.g. 'Reserve Bank of India', 'Narendra Modi', 'Mumbai').")
    type: Literal["person", "org", "location", "misc"] = Field(
        description="Entity category."
    )
    mention_count: int = Field(default=1, description="Number of times mentioned in the article.")


class ExtractedTable(BaseModel):
    """Structured table extracted from article content."""

    caption: str | None = Field(None, description="Table title or heading.")
    headers: list[str] = Field(default_factory=list, description="Column header strings.")
    rows: list[list[str]] = Field(default_factory=list, description="Row values.")
    markdown: str = Field(default="", description="Table formatted in GitHub-flavored Markdown.")


class ArticleEnrichment(BaseModel):
    """Phase 2: Full text transcription and deep enrichment for a major article."""

    body_text: str = Field(
        description="Complete verbatim transcribed body text of the article. Clean paragraphs separated by double newlines."
    )
    summary: str = Field(
        description="Concise 2-3 sentence executive summary of the article.",
        max_length=800
    )
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Key Named Entities mentioned in the article."
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Hierarchical topic paths (e.g. 'Economy > Monetary Policy', 'Markets > Equities', 'Technology > AI')."
    )
    tables: list[ExtractedTable] = Field(
        default_factory=list,
        description="Structured data tables contained within the article if any."
    )
