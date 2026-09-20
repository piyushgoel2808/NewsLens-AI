"""Metadata Extractor: Named Entity Recognition (NER), Topic Classification, and Summarization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.article import Article
from app.models.entity import (
    ArticleEntity,
    ArticleTopic,
    Entity,
    Topic,
)

logger = get_logger(__name__)

# Heuristic regular expressions for entity extraction
ORG_SUFFIXES = re.compile(
    r"\b(?:Corp|Corporation|Inc|Ltd|Limited|Bank|Ministry|Parliament|Congress|Senate|Bureau|Agency|Group|Board|Association|Commission|University|Party|Committee)\b",
    re.IGNORECASE,
)

CAPITALIZED_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")

KNOWN_LOCATIONS = {
    "Delhi",
    "New Delhi",
    "Mumbai",
    "London",
    "Washington",
    "Geneva",
    "Paris",
    "Tokyo",
    "Beijing",
    "Kolkata",
    "Chennai",
    "Bengaluru",
    "Hyderabad",
    "India",
    "US",
    "USA",
    "United States",
    "China",
    "UK",
    "United Kingdom",
    "Russia",
    "Europe",
    "Asia",
}

TOPIC_TAXONOMY_MAP = {
    "politics": ("Politics", "Politics > Government & Policy"),
    "election": ("Elections", "Politics > Elections"),
    "market": ("Markets", "Economy > Financial Markets"),
    "stock": ("Stocks", "Economy > Equities"),
    "bank": ("Banking", "Economy > Banking & Finance"),
    "tax": ("Taxation", "Economy > Fiscal Policy"),
    "tech": ("Technology", "Technology & Innovation"),
    "science": ("Science", "Science & Research"),
    "climate": ("Environment", "Environment & Climate"),
    "energy": ("Energy", "Industry > Energy"),
    "sport": ("Sports", "Sports > General"),
    "cricket": ("Cricket", "Sports > Cricket"),
    "court": ("Judiciary", "Law & Justice"),
    "war": ("Defense", "International > Defense & Conflict"),
}


@dataclass
class ExtractedEntity:
    """An identified named entity."""

    name: str
    type: str  # 'person', 'org', 'location', 'misc'
    mention_count: int = 1
    salience_score: float = 0.5


@dataclass
class ExtractedTopic:
    """An assigned topic category."""

    name: str
    taxonomy_path: str
    confidence: float = 0.9


@dataclass
class ArticleMetadataResult:
    """Consolidated metadata output for an article."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    topics: list[ExtractedTopic] = field(default_factory=list)
    summary: str | None = None


class MetadataExtractor:
    """Extracts entities, topics, and summaries and synchronizes with MySQL."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._entity_cache: dict[tuple[str, str], int] = {}
        self._topic_cache: dict[str, int] = {}

    def extract_entities_from_text(self, text: str, headline: str = "") -> list[ExtractedEntity]:
        """Extract Named Entities using patterns and heuristic clustering."""
        entities_dict: dict[tuple[str, str], int] = {}
        combined = f"{headline}\n{text}"

        # 1. Match potential multi-word capitalized phrases
        matches = CAPITALIZED_NAME_PATTERN.findall(combined)
        for match in matches:
            match_clean = match.strip()
            if len(match_clean) < 3 or match_clean in ("The Daily", "Newspaper Intelligence"):
                continue

            # Classify entity type
            if match_clean in KNOWN_LOCATIONS:
                etype = "location"
            elif ORG_SUFFIXES.search(match_clean):
                etype = "org"
            elif re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+$", match_clean):
                etype = "person"
            else:
                etype = "misc"

            key = (match_clean, etype)
            entities_dict[key] = entities_dict.get(key, 0) + 1

        # Calculate salience scores based on mention frequency and headline presence
        results: list[ExtractedEntity] = []
        total_mentions = max(1, sum(entities_dict.values()))

        for (name, etype), count in entities_dict.items():
            in_headline = name.lower() in headline.lower()
            freq_score = (count / total_mentions) * 0.5
            headline_bonus = 0.5 if in_headline else 0.1
            salience = round(min(1.0, freq_score + headline_bonus), 2)
            results.append(
                ExtractedEntity(
                    name=name,
                    type=etype,
                    mention_count=count,
                    salience_score=salience,
                )
            )

        # Sort by salience
        results.sort(key=lambda e: e.salience_score, reverse=True)
        return results[:15]  # Top 15 entities per article

    def extract_topics_from_text(self, text: str, headline: str = "") -> list[ExtractedTopic]:
        """Map article text to hierarchical topic taxonomy."""
        combined_lower = f"{headline} {text}".lower()
        matched_topics: dict[str, ExtractedTopic] = {}

        for keyword, (topic_name, tax_path) in TOPIC_TAXONOMY_MAP.items():
            if keyword in combined_lower and topic_name not in matched_topics:
                matched_topics[topic_name] = ExtractedTopic(
                    name=topic_name,
                    taxonomy_path=tax_path,
                    confidence=0.95 if keyword in headline.lower() else 0.80,
                )

        if not matched_topics:
            matched_topics["General News"] = ExtractedTopic(
                name="General News",
                taxonomy_path="News > General",
                confidence=0.70,
            )

        return list(matched_topics.values())

    def generate_summary(self, text: str, headline: str = "") -> str:
        """Generate concise 2-sentence article summary."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return headline or "News article report."

        # Take first 1-2 sentences of lead paragraph
        lead_sentences = re.split(r"(?<=[.!?])\s+", paragraphs[0])
        summary = " ".join(lead_sentences[:2]).strip()
        if len(summary) < 30 and len(paragraphs) > 1:
            second_lead = re.split(r"(?<=[.!?])\s+", paragraphs[1])
            summary += " " + " ".join(second_lead[:1])

        return summary[:500]

    async def process_and_persist_metadata(
        self,
        article_id: int,
        headline: str,
        full_text: str,
        pre_extracted_entities: list[ExtractedEntity] | None = None,
        pre_extracted_topics: list[ExtractedTopic] | None = None,
        pre_extracted_summary: str | None = None,
    ) -> ArticleMetadataResult:
        """Extract or persist metadata and synchronize relational entity and topic tables."""
        entities = pre_extracted_entities or self.extract_entities_from_text(full_text, headline=headline)
        topics = pre_extracted_topics or self.extract_topics_from_text(full_text, headline=headline)
        summary = pre_extracted_summary or self.generate_summary(full_text, headline=headline)

        # 1. Update Article summary
        art_stmt = select(Article).where(Article.id == article_id)
        art_res = await self._db.execute(art_stmt)
        article = art_res.scalar_one_or_none()
        if article:
            article.summary = summary

        # 2. Persist Entities & ArticleEntity junction using in-memory cache
        for ent in entities:
            ent_key = (ent.name, ent.type)
            db_entity_id = self._entity_cache.get(ent_key)
            if not db_entity_id:
                stmt = select(Entity.id).where(Entity.name == ent.name, Entity.type == ent.type)
                res = await self._db.execute(stmt)
                db_entity_id = res.scalar_one_or_none()
                if not db_entity_id:
                    db_entity = Entity(name=ent.name, type=ent.type)
                    self._db.add(db_entity)
                    await self._db.flush()
                    db_entity_id = db_entity.id
                self._entity_cache[ent_key] = db_entity_id

            # Link article entity
            art_ent = ArticleEntity(
                article_id=article_id,
                entity_id=db_entity_id,
                mention_count=ent.mention_count,
                salience_score=ent.salience_score,
            )
            self._db.add(art_ent)

        # 3. Persist Topics & ArticleTopic junction using in-memory cache
        for top in topics:
            topic_id = self._topic_cache.get(top.name)
            if not topic_id:
                top_stmt = select(Topic.id).where(Topic.name == top.name)
                top_res = await self._db.execute(top_stmt)
                topic_id = top_res.scalar_one_or_none()
                if not topic_id:
                    topic_record = Topic(name=top.name, taxonomy_path=top.taxonomy_path)
                    self._db.add(topic_record)
                    await self._db.flush()
                    topic_id = topic_record.id
                self._topic_cache[top.name] = topic_id

            art_top = ArticleTopic(
                article_id=article_id,
                topic_id=topic_id,
                confidence=top.confidence,
            )
            self._db.add(art_top)

        await self._db.flush()

        logger.info(
            "Article metadata persisted",
            extra={
                "article_id": article_id,
                "entities_count": len(entities),
                "topics_count": len(topics),
            },
        )

        return ArticleMetadataResult(
            entities=entities,
            topics=topics,
            summary=summary,
        )

    async def process_and_persist_bulk_metadata(
        self,
        articles_data: list[tuple[int, str, str]],  # [(article_id, headline, full_text)]
    ) -> dict[int, ArticleMetadataResult]:
        """Extract and persist metadata for an entire issue in high-speed bulk batches."""
        if not articles_data:
            return {}

        results: dict[int, ArticleMetadataResult] = {}
        all_extracted: list[tuple[int, list[ExtractedEntity], list[ExtractedTopic], str]] = []

        # 1. In-memory extraction for all articles
        for article_id, headline, full_text in articles_data:
            ents = self.extract_entities_from_text(full_text, headline=headline)
            tops = self.extract_topics_from_text(full_text, headline=headline)
            summ = self.generate_summary(full_text, headline=headline)
            all_extracted.append((article_id, ents, tops, summ))
            results[article_id] = ArticleMetadataResult(entities=ents, topics=tops, summary=summ)

        # 2. Warm caches in single bulk queries
        distinct_entity_names = list({e.name for _, ents, _, _ in all_extracted for e in ents if (e.name, e.type) not in self._entity_cache})
        if distinct_entity_names:
            ents_stmt = select(Entity).where(Entity.name.in_(distinct_entity_names))
            ents_res = await self._db.execute(ents_stmt)
            for db_e in ents_res.scalars().all():
                self._entity_cache[(db_e.name, db_e.type)] = db_e.id

        distinct_topic_names = list({t.name for _, _, tops, _ in all_extracted for t in tops if t.name not in self._topic_cache})
        if distinct_topic_names:
            tops_stmt = select(Topic).where(Topic.name.in_(distinct_topic_names))
            tops_res = await self._db.execute(tops_stmt)
            for db_t in tops_res.scalars().all():
                self._topic_cache[db_t.name] = db_t.id

        # 3. Create missing entities/topics
        for _, ents, tops, _ in all_extracted:
            for ent in ents:
                key = (ent.name, ent.type)
                if key not in self._entity_cache:
                    new_ent = Entity(name=ent.name, type=ent.type)
                    self._db.add(new_ent)
                    await self._db.flush()
                    self._entity_cache[key] = new_ent.id

            for top in tops:
                if top.name not in self._topic_cache:
                    new_top = Topic(name=top.name, taxonomy_path=top.taxonomy_path)
                    self._db.add(new_top)
                    await self._db.flush()
                    self._topic_cache[top.name] = new_top.id

        # 4. Link all ArticleEntity and ArticleTopic records in batch
        for article_id, ents, tops, summ in all_extracted:
            art_stmt = select(Article).where(Article.id == article_id)
            art_res = await self._db.execute(art_stmt)
            article = art_res.scalar_one_or_none()
            if article:
                article.summary = summ

            for ent in ents:
                ent_id = self._entity_cache[(ent.name, ent.type)]
                art_ent = ArticleEntity(
                    article_id=article_id,
                    entity_id=ent_id,
                    mention_count=ent.mention_count,
                    salience_score=ent.salience_score,
                )
                self._db.add(art_ent)

            for top in tops:
                top_id = self._topic_cache[top.name]
                art_top = ArticleTopic(
                    article_id=article_id,
                    topic_id=top_id,
                    confidence=top.confidence,
                )
                self._db.add(art_top)

        await self._db.flush()
        return results
