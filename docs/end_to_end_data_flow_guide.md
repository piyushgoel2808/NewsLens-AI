# NewsLens-AI End-to-End Data Flow & Data Structure Guide

> **Document Version**: 1.0.0  
> **System Scope**: Ingestion, Layout Vision, Relational Storage, Vector Indexing, Query Planning, Multi-Tool Retrieval, Corrective RAG (CRAG), Answer Synthesis, Anti-Hallucination Guardrails, and IR Evaluation.  
> **Target Audience**: Core Engineers, AI Researchers, and System Architects.

---

## Table of Contents

1. [High-Level Architecture & Data Flow Sequence](#1-high-level-architecture--data-flow-sequence)
2. [Phase 1: Ingestion Pipeline (From Raw PDF to Relational & Vector Storage)](#2-phase-1-ingestion-pipeline-from-raw-pdf-to-relational--vector-storage)
   - [1.1 Document Ingestion & Page Rendering](#11-document-ingestion--page-rendering)
   - [1.2 Docling 2D Layout & Vision OCR Parser](#12-docling-2d-layout--vision-ocr-parser)
   - [1.3 Article Boundary Assembly & Coalescence](#13-article-boundary-assembly--coalescence)
   - [1.4 Cross-Page Continuation Assembly](#14-cross-page-continuation-assembly)
   - [1.5 Classification, Prominence Scoring & Visual Asset Extraction](#15-classification-prominence-scoring--visual-asset-extraction)
   - [1.6 SQL Database Persistence (All 12+ Tables Populated)](#16-sql-database-persistence-all-12-tables-populated)
   - [1.7 Hierarchical Newspaper Chunking & Qdrant Vector Indexing](#17-hierarchical-newspaper-chunking--qdrant-vector-indexing)
3. [Phase 2: Conversational Pre-Processing & Query Condensation](#3-phase-2-conversational-pre-processing--query-condensation)
   - [2.1 Chat History & Metadata Detection](#21-chat-history--metadata-detection)
   - [2.2 Active Context Extraction & Query-Aware Isolation](#22-active-context-extraction--query-aware-isolation)
   - [2.3 Coreference Resolution & Condensed Query Generation](#23-coreference-resolution--condensed-query-generation)
4. [Phase 3: Cognitive Query Planner & Structured Chain-of-Thought](#4-phase-3-cognitive-query-planner--structured-chain-of-thought)
   - [3.1 Query Intent Classification & Archetype Mapping](#31-query-intent-classification--archetype-mapping)
   - [3.2 The Pydantic `QueryPlan` Output](#32-the-pydantic-queryplan-output)
5. [Phase 4: Tool Execution Deep Dive (Inputs, SQL Queries & Returned JSON)](#5-phase-4-tool-execution-deep-dive-inputs-sql-queries--returned-json)
   - [Tool 1: `hybrid_search` (Dense + Sparse + RRF + Cross-Encoder Reranking)](#tool-1-hybrid_search-dense--sparse--rrf--cross-encoder-reranking)
   - [Tool 2: `sql_analytics` (Manifests, Trends & Coverage Differences)](#tool-2-sql_analytics-manifests-trends--coverage-differences)
   - [Tool 3: `entity_search` (Multi-Hop Entity Graph & Salience Scoring)](#tool-3-entity_search-multi-hop-entity-graph--salience-scoring)
   - [Tool 4: `timeline_builder` (Narrative Chronological Trajectory)](#tool-4-timeline_builder-narrative-chronological-trajectory)
   - [Tool 5: `coverage_analysis` (3-Tier Negative Coverage Audit)](#tool-5-coverage_analysis-3-tier-negative-coverage-audit)
   - [Tool 6: `web_search` (Live Web Verification Fallback)](#tool-6-web_search-live-web-verification-fallback)
6. [Phase 5: Corrective RAG (CRAG) Relevance Gate & Fallbacks](#6-phase-5-corrective-rag-crag-relevance-gate--fallbacks)
   - [5.1 Stemmed Query Matching & Relevance Scoring](#51-stemmed-query-matching--relevance-scoring)
   - [5.2 Macro Manifest Protection](#52-macro-manifest-protection)
   - [5.3 Corrective Fallback Activation](#53-corrective-fallback-activation)
7. [Phase 6: Answer Synthesis, Prompt Budgeting & SSE Streaming](#7-phase-6-answer-synthesis-prompt-budgeting--sse-streaming)
   - [6.1 Evidence Context Budgeting & Publication Scoping](#61-evidence-context-budgeting--publication-scoping)
   - [6.2 The Complete Synthesizer Prompt Structure](#62-the-complete-synthesizer-prompt-structure)
   - [6.3 LLM Generation, `<think>` Tag Separation & Inline Citations](#63-llm-generation-think-tag-separation--inline-citations)
   - [6.4 Server-Sent Events (SSE) Wire Protocol](#64-server-sent-events-sse-wire-protocol)
8. [Phase 7: Anti-Hallucination Guardrails & Context Isolation](#8-phase-7-anti-hallucination-guardrails--context-isolation)
9. [Phase 8: Comprehensive Top-K Lifecycle Reference](#9-phase-8-comprehensive-top-k-lifecycle-reference)
10. [Phase 9: Information Retrieval & Generation Evaluation Metrics](#10-phase-9-information-retrieval--generation-evaluation-metrics)

---

## 1. High-Level Architecture & Data Flow Sequence

The diagram below maps how an uploaded newspaper PDF flows through layout recognition, relational storage, and vector indexing, and how a user query traverses condensation, planning, multi-tool execution, CRAG evaluation, and streaming answer synthesis.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       INGESTION PIPELINE                                               │
│                                                                                                        │
│   PDF Broadsheet      PyMuPDF Render     Docling Layout Model     Segmenter & Assembler                │
│  ┌──────────────┐     ┌─────────────┐    ┌─────────────────────┐  ┌─────────────────────┐              │
│  │ 16-32 Pages  │ ──> │ 300 DPI PNG │ ─> │ 2D Bounding Boxes   │─>│ Coalesce Headlines, │              │
│  │ Broad-sheet  │     │ 2480x3508px │    │ Labels & OCR Text   │  │ Byline, Body, Decks │              │
│  └──────────────┘     └─────────────┘    └─────────────────────┘  └──────────┬──────────┘              │
│                                                                              │                         │
│                                            ┌─────────────────────────────────┴──────────────────┐      │
│                                            ▼                                                    ▼      │
│                                ┌───────────────────────┐                            ┌────────────────┐ │
│                                │ MySQL Relational DB   │                            │ Qdrant Vectors │ │
│                                │ 17 Normalized Tables  │                            │ 1024-dim BGE   │ │
│                                └───────────────────────┘                            └────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                                   │ User Query
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        AGENTIC QUERY PIPELINE                                          │
│                                                                                                        │
│  User Query + History        Query Condenser (LLM)      Query Planner (Structured CoT)                 │
│  ┌────────────────────┐      ┌─────────────────────┐    ┌─────────────────────────────┐                │
│  │ "List all its news │ ───> │ Resolves pronouns & │ ──>│ Determines Archetype &       │                │
│  │ on page 3"         │      │ active publication  │    │ schedules 1 to 4 tools      │                │
│  └────────────────────┘      └─────────────────────┘    └──────────────┬──────────────┘                │
│                                                                        │                               │
│                   ┌────────────────────────────────────────────────────┴───────────────┐               │
│                   ▼                                                                    ▼               │
│       ┌───────────────────────┐                                            ┌───────────────────────┐   │
│       │ Tool Execution:       │                                            │ Tool Execution:       │   │
│       │ hybrid_search         │                                            │ sql_analytics         │   │
│       │ (Dense + Sparse + RRF)│                                            │ (Manifest/Exclusion)  │   │
│       └───────────┬───────────┘                                            └───────────┬───────────┘   │
│                   │                                                                    │               │
│                   └─────────────────────────────────┬──────────────────────────────────┘               │
│                                                     ▼                                                  │
│                                     ┌───────────────────────────────┐                                  │
│                                     │ Corrective RAG (CRAG) Gate    │                                  │
│                                     │ Stemmed Relevance Scoring     │                                  │
│                                     │ Fallback if Grounding = 0     │                                  │
│                                     └───────────────┬───────────────┘                                  │
│                                                     ▼                                                  │
│                                     ┌───────────────────────────────┐                                  │
│                                     │ Synthesizer Prompt Budgeting  │                                  │
│                                     │ Top 12 Items / <= 3500 Tokens │                                  │
│                                     └───────────────┬───────────────┘                                  │
│                                                     ▼                                                  │
│                                     ┌───────────────────────────────┐                                  │
│                                     │ LLM Streaming Synthesis       │                                  │
│                                     │ Thought + Structured Brief    │                                  │
│                                     └───────────────────────────────┘                                  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: Ingestion Pipeline (From Raw PDF to Relational & Vector Storage)

Let us follow a concrete edition of **The Goan Everywhere**, published on **August 1, 2026** (File: `the_goan_2026-08-01.pdf`, 12 Pages).

### 1.1 Document Ingestion & Page Rendering

1. The PDF is submitted to `POST /api/ingest/upload` or via CLI `python -m app.ingestion.tasks`.
2. `PyMuPDF` (`fitz`) rasterizes each page at **300 DPI** using matrix scaling factor $4.166$:
   - Width: `2480 px`, Height: `3508 px`.
   - Generates PNG byte streams stored in `storage/renders/{issue_id}/page_{page_num}.png`.
3. The database transaction initializes the `issues` and `pages` records.

#### Exact SQL Executed
```sql
INSERT INTO newspapers (id, name, code, publisher, country, language)
VALUES (1, 'The Goan', 'the_goan', 'Fomento Media', 'India', 'English');

INSERT INTO issues (id, newspaper_id, issue_date, edition, total_pages, status)
VALUES (93, 1, '2026-08-01', 'Panaji', 12, 'processing');

INSERT INTO pages (id, issue_id, page_number, printed_page_number, width, height, image_path)
VALUES 
  (1001, 93, 1, '1', 2480, 3508, 'storage/renders/93/page_1.png'),
  (1002, 93, 2, '2', 2480, 3508, 'storage/renders/93/page_2.png');
```

---

### 1.2 Docling 2D Layout & Vision OCR Parser

[`backend/app/ingestion/docling_parser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/docling_parser.py) runs the page image and PDF text layer through DocLayNet:

1. **Bounding Box Normalization**: Converts native PDF coordinate system ($\text{Bottom-Left Origin}$) into standard web pixel space ($\text{Top-Left Origin: } [x_0, y_0, x_1, y_1]$).
2. **Layout Node Classification**: Emits discrete structural tokens: `title`, `section_header`, `text`, `table`, `picture`, `caption`.
3. **Corrupted Font CMap Guard**: Computes the ratio of Unicode replacement characters `\ufffd` or `\ufeff`. If the replacement ratio exceeds $3.0\%$, it raises `CorruptedPdfTextLayerError` and triggers pure Image OCR fallback via `GoogleCloudVisionOCR`.

#### Intermediate JSON Emitted by Docling (`DoclingParsedItem`)
```json
[
  {
    "label": "section_header",
    "text": "GOA NEWS",
    "bbox": [180.0, 140.0, 480.0, 195.0],
    "page_number": 1,
    "level": 1
  },
  {
    "label": "title",
    "text": "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS",
    "bbox": [180.0, 220.0, 1650.0, 340.0],
    "page_number": 1,
    "level": 1
  },
  {
    "label": "text",
    "text": "By Staff Reporter",
    "bbox": [180.0, 355.0, 520.0, 385.0],
    "page_number": 1,
    "level": 2
  },
  {
    "label": "text",
    "text": "Panaji: The Corporation of the City of Panaji (CCP) on Friday officially commenced automated AI-powered traffic surveillance across 14 major arterial junctions. Mayor Rohit Monserrate announced that over 450 contactless e-challans were issued on day one.",
    "bbox": [180.0, 410.0, 890.0, 820.0],
    "page_number": 1,
    "level": 2
  },
  {
    "label": "picture",
    "text": "",
    "bbox": [920.0, 410.0, 1650.0, 950.0],
    "page_number": 1,
    "level": 1
  },
  {
    "label": "caption",
    "text": "CCP surveillance control room monitoring live feeds from the Mandovi bridge junction on Friday.",
    "bbox": [920.0, 960.0, 1650.0, 1010.0],
    "page_number": 1,
    "level": 2
  }
]
```

---

### 1.3 Article Boundary Assembly & Coalescence

In [`docling_parser.py:assemble_articles()`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/docling_parser.py#L420-L520):

* **Byline Stripping**: `By Staff Reporter` matches `_AUTHOR_NAME_PATTERN` and is extracted to `byline_author = "Staff Reporter"`.
* **Dateline Extraction**: `Panaji:` matches `_DATELINE_PATTERN`.
* **Internal Subhead Coalescence**: If an all-caps short block appears (e.g. `### PENALTY STRUCTURE`) inside an active story, it is coalesced into `full_text` as markdown rather than splitting into a fake new article.
* **Deck Coalescence**: Subheadlines that expand on the title are bound to `subheadline`.

#### Assembled Article Output (`SegmentedArticle`)
```json
{
  "article_temp_id": "page_1_art_0",
  "headline": "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS",
  "subheadline": "Over 450 automated citations issued on Day 1 across 14 key junctions",
  "byline_author": "Staff Reporter",
  "body_text": "Panaji: The Corporation of the City of Panaji (CCP) on Friday officially commenced automated AI-powered traffic surveillance across 14 major arterial junctions. Mayor Rohit Monserrate announced that over 450 contactless e-challans were issued on day one. ### PENALTY STRUCTURE Violations include signal jumping, riding without helmets, and lane indiscipline with fines ranging from Rs 500 to Rs 2,000.",
  "word_count": 218,
  "bbox_list": [
    [180.0, 220.0, 1650.0, 340.0],
    [180.0, 410.0, 890.0, 820.0]
  ],
  "jump_to_page": null,
  "jump_from_page": null,
  "is_teaser": false
}
```

---

### 1.4 Cross-Page Continuation Assembly

When an article ends with `"Continued on Page 4 ▶"`, [`CrossPageAssembler`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/tasks.py#L422) locates matching jump markers on Page 4 (`"Continued from Page 1"`), links the body blocks into an `AssembledArticle`, and records multiple entries in `ArticlePage`.

---

### 1.5 Classification, Prominence Scoring & Visual Asset Extraction

1. **Article Classification** (`ArticleClassifier`):
   - Category: `Local Governance` / `Economy` (Confidence: `0.94`).
   - Section: `Goa News`.
   - Prominence Score: `0.88` (Calculated using Page 1 placement factor $1.0$, headline size factor $0.85$, and accompanying image bonus $+0.15$).
2. **Visual Asset Extraction** (`MediaExtractor`):
   - Bounding box `[920, 410, 1650, 950]` is cropped and saved to disk as `storage/photos/93/photo_1_1.jpg`.
   - Bounding-box spatial envelope binding binds the photo to `article_id = 5401`.
3. **Metadata & Entity Extraction** (`MetadataExtractor`):
   - Entities: `Corporation of the City of Panaji` (`org`), `Rohit Monserrate` (`person`), `Panaji` (`location`), `Mandovi Bridge` (`location`).
   - Topics: `Traffic Management`, `Municipal Administration`.

---

### 1.6 SQL Database Persistence (All 12+ Tables Populated)

Once assembled, [`tasks.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/tasks.py#L460-L650) persists the normalized relational records in a single database transaction.

```
                                RELATIONAL SCHEMA GRAPH
                                
  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
  │  newspapers  │ ──<1:N>─ │    issues    │ ──<1:N>─ │    pages     │
  └──────────────┘         └──────┬───────┘         └──────┬───────┘
                                  │                        │
                                 <1:N>                    <1:N>
                                  │                        │
                                  ▼                        ▼
                       ┌──────────────────────┐   ┌──────────────────┐
                       │       articles       │──<│  article_pages   │
                       └──┬──────┬──────────┬─┘   └──────────────────┘
                          │      │          │
                       <1:N>   <1:N>      <1:N>
                          │      │          │
                          ▼      ▼          ▼
             ┌──────────────┐ ┌─────────┐ ┌──────────────────┐
             │article_chunks│ │ photos  │ │ article_entities │
             └──────────────┘ └─────────┘ └────────┬─────────┘
                                                   │
                                                 <N:1>
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │     entities     │
                                          └──────────────────┘
```

#### Exact SQL Rows Created

##### Table: `articles`
```sql
INSERT INTO articles (
  id, issue_id, primary_page_id, category_id, headline, subheadline, 
  byline_author, section, printed_section, article_type, prominence_score, 
  word_count, full_text
) VALUES (
  5401, 93, 1001, 4,
  'PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS',
  'Over 450 automated citations issued on Day 1 across 14 key junctions',
  'Staff Reporter', 'Goa News', 'GOA NEWS', 'news', 0.88, 218,
  'Panaji: The Corporation of the City of Panaji (CCP) on Friday officially commenced automated AI-powered traffic surveillance across 14 major arterial junctions. Mayor Rohit Monserrate announced that over 450 contactless e-challans were issued on day one. ### PENALTY STRUCTURE Violations include signal jumping, riding without helmets, and lane indiscipline with fines ranging from Rs 500 to Rs 2,000.'
);
```

##### Table: `article_pages`
```sql
INSERT INTO article_pages (article_id, page_id, page_number, printed_page_number, bbox_json, block_order)
VALUES (
  5401, 1001, 1, '1',
  '{"bboxes": [[180.0, 220.0, 1650.0, 340.0], [180.0, 410.0, 890.0, 820.0]]}',
  1
);
```

##### Table: `photos`
```sql
INSERT INTO photos (id, page_id, article_id, photo_index, image_path, width, height, caption, bbox_json)
VALUES (
  812, 1001, 5401, 1,
  'storage/photos/93/photo_1_1.jpg', 730, 540,
  'CCP surveillance control room monitoring live feeds from the Mandovi bridge junction on Friday.',
  '{"bbox": [920.0, 410.0, 1650.0, 950.0]}'
);
```

##### Tables: `entities` and `article_entities`
```sql
INSERT INTO entities (id, name, type) VALUES (301, 'Corporation of the City of Panaji', 'org');
INSERT INTO entities (id, name, type) VALUES (302, 'Rohit Monserrate', 'person');

INSERT INTO article_entities (article_id, entity_id, mention_count, salience_score)
VALUES 
  (5401, 301, 3, 0.95),
  (5401, 302, 1, 0.82);
```

##### Tables: `topics` and `article_topics`
```sql
INSERT INTO topics (id, name, taxonomy_path) VALUES (45, 'Local Governance', 'Society > Governance');

INSERT INTO article_topics (article_id, topic_id, confidence)
VALUES (5401, 45, 0.94);
```

---

### 1.7 Hierarchical Newspaper Chunking & Qdrant Vector Indexing

[`NewspaperChunker`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/chunker.py) splits the article into overlapping chunks ($\approx 350$ tokens) while **prepending global newspaper context headers**:

#### Standardized Context Header
```text
[Newspaper: The Goan | Date: 2026-08-01 | Section: Goa News | Headline: PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS | Page(s): 1 (PDF p.1)]
```

#### Persisted in MySQL Table: `article_chunks`
```sql
INSERT INTO article_chunks (
  id, article_id, chunk_index, chunk_text, token_count, qdrant_point_id, has_visual_data
) VALUES (
  18901, 5401, 0,
  '[Newspaper: The Goan | Date: 2026-08-01 | Section: Goa News | Headline: PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS | Page(s): 1 (PDF p.1)] Panaji: The Corporation of the City of Panaji (CCP) on Friday officially commenced automated AI-powered traffic surveillance across 14 major arterial junctions. Mayor Rohit Monserrate announced that over 450 contactless e-challans were issued on day one...',
  112, '9b2c8a14-72de-4e3a-b851-4e78a631f290', 0
);
```

#### Vector Point Upserted to Qdrant Collection `newslens_articles`
```json
{
  "id": "9b2c8a14-72de-4e3a-b851-4e78a631f290",
  "vector": [0.0241, -0.0158, 0.0891, 0.0042, "... 1024 float dimensions from BAAI/bge-m3 ..."],
  "payload": {
    "article_id": 5401,
    "issue_id": 93,
    "newspaper_name": "The Goan",
    "issue_date": "2026-08-01",
    "headline": "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS",
    "section": "Goa News",
    "article_type": "news",
    "prominence_score": 0.88,
    "has_photo": true,
    "has_table": false,
    "has_visual_data": false,
    "chunk_type": "text",
    "chunk_index": 0,
    "page_numbers": [1],
    "printed_pages": ["1"],
    "entities": [
      "Corporation of the City of Panaji",
      "Rohit Monserrate",
      "Panaji"
    ],
    "topics": ["Local Governance"],
    "chunk_text": "[Newspaper: The Goan | Date: 2026-08-01 | Section: Goa News | Headline: PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS | Page(s): 1 (PDF p.1)] Panaji: The Corporation of the City of Panaji...",
    "raw_text": "Panaji: The Corporation of the City of Panaji..."
  }
}
```

---

## 3. Phase 2: Conversational Pre-Processing & Query Condensation

Consider a multi-turn conversation where the user previously asked about news in Goa and now submits:
`"List the news that are in the GOAN dated 1/8/2026 but not in he Morning Standard dated 1/8/2026"`
followed on the next turn by:
`"list all those 11 articles"`

```
                               QUERY CONDENSATION PIPELINE
                               
   Turn N User Input: "list all those 11 articles"
                       + Chat History
                             │
                             ▼
   ┌────────────────────────────────────────────────────────┐
   │ 1. in_context_meta_query check                         │
   │    • Regex matches: "which date?", "who is author?"    │
   │    • Returns False (substantive follow-up)             │
   └─────────────────────────┬──────────────────────────────┘
                             ▼
   ┌────────────────────────────────────────────────────────┐
   │ 2. extract_active_issue_from_history()                 │
   │    • Resolves Active Newspaper: The Goan               │
   │    • Resolves Comparison Newspaper: The Morning Standard│
   │    • Resolves Issue Date: 2026-08-01                   │
   │    • Resolves is_differential: True                    │
   └─────────────────────────┬──────────────────────────────┘
                             ▼
   ┌────────────────────────────────────────────────────────┐
   │ 3. Coreference Rewrite (condenser.py)                  │
   │    • "list all those 11 articles"                      │
   │      ==> "list all those 11 articles in The Goan       │
   │           but not in The Morning Standard              │
   │           dated 2026-08-01"                            │
   └────────────────────────────────────────────────────────┘
```

### 2.1 Chat History & Metadata Detection

If the user asks:
`"Which newspaper was that from?"`
`is_in_context_meta_query()` in [`condenser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/condenser.py#L44) evaluates to `True`. The query engine skips tool retrieval completely, routing directly to the synthesizer to answer from prior conversational context.

### 2.2 Active Context Extraction & Query-Aware Isolation

When evaluating `"list all those 11 articles"`:
1. `extract_active_issue_from_history()` scans backwards across the previous assistant messages.
2. It identifies:
   ```python
   {
       "newspaper_name": "The Goan",
       "comparison_newspaper": "The Morning Standard",
       "issue_date": "2026-08-01",
       "is_differential": True
   }
   ```
3. **Query-Aware Isolation**: If the user's new query mentions a *different* date (e.g., `2/8/2026`), the condenser purges the previous date and newspaper bindings to prevent cross-date memory leakage.

### 2.3 Condensed Query Output

The condensation engine rewrites the follow-up prompt into a fully qualified standalone query:
```text
"list all those 11 articles in The Goan but not in The Morning Standard dated 2026-08-01"
```

---

## 4. Phase 3: Cognitive Query Planner & Structured Chain-of-Thought

The condensed query enters [`QueryPlanner.plan_query_async()`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/planner.py):

```
                        COGNITIVE QUERY PLANNER ARCHITECTURE
                        
   Input: "list all those 11 articles in The Goan but not in The Morning Standard dated 2026-08-01"
                                 │
                                 ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 1. Parameter Extraction (extract_parameters_from_query)                │
   │    • target_newspapers: ['The Goan', 'The Morning Standard']           │
   │    • comparison_newspaper: 'The Morning Standard'                      │
   │    • issue_date: '2026-08-01'                                          │
   │    • is_differential: True ("but not in")                              │
   └─────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 2. Structured Chain-of-Thought Generation                              │
   │    • Archetype: cross_newspaper_comparison                             │
   │    • Primary Tool: sql_analytics (coverage_difference)                 │
   │    • Secondary Tool: hybrid_search (The Goan, 2026-08-01, top_k=12)    │
   └────────────────────────────────────────────────────────────────────────┘
```

### 3.1 The Pydantic `QueryPlan` JSON Output

The planner emits a validated Pydantic model:

```json
{
  "thought_process": "User is performing a differential comparison between two specific broadsheets on 2026-08-01: stories present in The Goan but absent from The Morning Standard. This requires computing the exact deterministic relational set difference of article manifests via sql_analytics(analysis_type='coverage_difference'). We also schedule secondary hybrid_search scoped to The Goan on 2026-08-01 to retrieve narrative excerpts for the top exclusive stories.",
  "archetype": "cross_newspaper_comparison",
  "primary_tool": "sql_analytics",
  "arguments": {
    "analysis_type": "coverage_difference",
    "newspaper_name": "The Goan",
    "comparison_newspaper": "The Morning Standard",
    "issue_date": "2026-08-01",
    "query": "list all those 11 articles in The Goan but not in The Morning Standard dated 2026-08-01"
  },
  "include_secondary_hybrid_search": true,
  "secondary_search_query": "exclusive articles in The Goan not in The Morning Standard",
  "tool_calls": [
    {
      "tool_name": "sql_analytics",
      "arguments": {
        "analysis_type": "coverage_difference",
        "newspaper_name": "The Goan",
        "comparison_newspaper": "The Morning Standard",
        "issue_date": "2026-08-01",
        "query": "list all those 11 articles in The Goan but not in The Morning Standard dated 2026-08-01"
      },
      "purpose": "Compute verified article difference: stories in The Goan absent from The Morning Standard on 2026-08-01"
    },
    {
      "tool_name": "hybrid_search",
      "arguments": {
        "query": "exclusive articles in The Goan not in The Morning Standard",
        "newspaper_name": "The Goan",
        "date_from": "2026-08-01",
        "date_to": "2026-08-01",
        "top_k": 12
      },
      "purpose": "Retrieve representative article excerpts from The Goan on 2026-08-01"
    }
  ]
}
```

---

## 5. Phase 4: Tool Execution Deep Dive (Inputs, SQL Queries & Returned JSON)

NewsLens-AI supports 6 discrete agentic tools. Below is the exact operational lifecycle, SQL query, and returned JSON for every tool.

---

### Tool 1: `hybrid_search` (Dense + Sparse + RRF + Cross-Encoder Reranking)

[`backend/app/retrieval/hybrid_search.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/hybrid_search.py) executes a two-stage retrieval cascade:

```
                            TWO-STAGE HYBRID RETRIEVAL CASCADE
                            
                       Query: "smart traffic challans Panaji"
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      Stage 1A: Dense Vector                         Stage 1B: Sparse Keyword
      Qdrant HNSW Cosine Search                      MySQL MATCH ... AGAINST
      Fetch top_k * 3 = 18 points                    Fetch top_k * 3 = 18 rows
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                        Reciprocal Rank Fusion (RRF)
                        RRF = 0.5/(60 + r_dense) + 0.5/(60 + r_sparse)
                                         │
                                         ▼
                        Candidate Pool: Top 75 Candidates
                        Load Articles, Pages & Bounding Boxes
                                         │
                                         ▼
                        Stage 2: Cross-Encoder Reranking
                        bge-reranker-v2-m3 Cross-Attention
                                         │
                                         ▼
                        Return Top K (e.g. top_k = 6)
```

#### Step 1: Tool Inputs
```json
{
  "query": "smart traffic challans Panaji",
  "newspaper_name": "The Goan",
  "date_from": "2026-08-01",
  "date_to": "2026-08-01",
  "page_filter": "1",
  "top_k": 6
}
```

#### Step 2: Database Operations Executed
1. **Qdrant Vector Search (`top_k * 3 = 18`)**:
   ```python
   # Dense vector embedding via BAAI/bge-m3
   query_vector = await embed_provider.embed_one("smart traffic challans Panaji")
   # Filter: newspaper_id == 1 AND issue_date == "2026-08-01" AND page_numbers contains 1
   ```
2. **MySQL Fulltext Search (`top_k * 3 = 18`)**:
   ```sql
   SELECT a.id, a.headline, a.full_text,
          MATCH(a.headline, a.full_text) AGAINST('smart traffic challans Panaji' IN NATURAL LANGUAGE MODE) AS score
   FROM articles a
   JOIN issues i ON a.issue_id = i.id
   JOIN article_pages ap ON a.id = ap.article_id
   WHERE i.newspaper_id = 1 
     AND i.issue_date = '2026-08-01'
     AND ap.page_number = 1
     AND MATCH(a.headline, a.full_text) AGAINST('smart traffic challans Panaji' IN NATURAL LANGUAGE MODE) > 0
   ORDER BY score DESC
   LIMIT 18;
   ```
3. **Reciprocal Rank Fusion (RRF) with $k = 60$**:
   For Article ID 5401 (Vector Rank 1, Sparse Rank 1):
   $$\text{RRF Score} = 0.5 \times \frac{1}{60 + 1} + 0.5 \times \frac{1}{60 + 1} = 0.008196 + 0.008196 = 0.01639$$
4. **Second-Stage Cross-Encoder Reranking**:
   Pairs query with candidate text:
   `("smart traffic challans Panaji", "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS\nPanaji: The Corporation of the City of Panaji...")`
   Cross-Encoder output score: `0.9421`.

#### Step 3: Tool Output Returned to Agent
```json
[
  {
    "article_id": 5401,
    "issue_id": 93,
    "headline": "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS",
    "newspaper_name": "The Goan",
    "issue_date": "2026-08-01",
    "pages": [1],
    "printed_pages": ["1"],
    "bboxes": [[[180.0, 220.0, 1650.0, 340.0], [180.0, 410.0, 890.0, 820.0]]],
    "snippet": "Panaji: The Corporation of the City of Panaji (CCP) on Friday officially commenced automated AI-powered traffic surveillance across 14 major arterial junctions. Mayor Rohit Monserrate announced that over 450 contactless e-challans were issued on day one.",
    "prominence_score": 0.88,
    "rerank_score": 0.9421,
    "source_tool": "hybrid_search"
  }
]
```

---

### Tool 2: `sql_analytics` (Manifests, Trends & Coverage Differences)

[`backend/app/retrieval/sql_analytics.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/sql_analytics.py) performs relational aggregation and deterministic set differences.

#### Operation A: `issue_summary` (Whole Broadsheet Manifest)

##### Input Arguments
```json
{
  "analysis_type": "issue_summary",
  "newspaper_name": "The Goan",
  "issue_date": "2026-08-01"
}
```

##### Executed SQL Query
```sql
SELECT 
    a.id,
    a.headline,
    a.byline_author,
    a.section,
    a.word_count,
    a.prominence_score,
    COALESCE(MIN(ap.page_number), a.primary_page_id) AS page_number,
    COALESCE(MIN(ap.printed_page_number), CAST(MIN(ap.page_number) AS CHAR)) AS printed_page
FROM articles a
JOIN issues i ON a.issue_id = i.id
JOIN newspapers n ON i.newspaper_id = n.id
LEFT JOIN article_pages ap ON a.id = ap.article_id
WHERE n.name LIKE '%The Goan%' AND i.issue_date = '2026-08-01'
GROUP BY a.id, a.headline, a.byline_author, a.section, a.word_count, a.prominence_score, a.primary_page_id
ORDER BY page_number ASC, a.prominence_score DESC;
```

##### Output Returned to Agent
```json
{
  "newspaper_name": "The Goan",
  "issue_date": "2026-08-01",
  "total_articles": 174,
  "pages_count": 12,
  "articles": [
    {
      "article_id": 5401,
      "page_number": 1,
      "printed_page": "1",
      "section": "Goa News",
      "headline": "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS",
      "author": "Staff Reporter",
      "word_count": 218
    },
    {
      "article_id": 5402,
      "page_number": 1,
      "printed_page": "1",
      "section": "Goa News",
      "headline": "CHICALIM PANCHAYAT SEALS 3 COMMERCIAL UNITS OVER TRADE LICENCE DUES",
      "author": "Our Bureau",
      "word_count": 164
    }
  ]
}
```

---

#### Operation B: `coverage_difference` ("In X but not in Y")

##### Input Arguments
```json
{
  "analysis_type": "coverage_difference",
  "newspaper_name": "The Goan",
  "comparison_newspaper": "The Morning Standard",
  "issue_date": "2026-08-01"
}
```

##### Deterministic Difference Engine Logic (`get_newspaper_coverage_difference`)
1. Loads full manifests: `The Goan` (Issue #93, 174 articles) and `The Morning Standard` (Issue #98, 144 articles).
2. Normalizes headlines into non-stopword token sets ($\text{Token length} \ge 4$).
3. Computes Jaccard Token Overlap:
   $$J(A, B) = \frac{|T_A \cap T_B|}{|T_A \cup T_B|}$$
4. If $J(A, B) < 0.50$ across all articles in the comparison newspaper, the article is classified as **Exclusive**.
5. Filters out running page headers (`"SATURDAY"`, `"PANAJI"`, `"IN SHORT >>"`).

##### Output Returned to Agent
```json
{
  "source_newspaper": "The Goan",
  "comparison_newspaper": "The Morning Standard",
  "issue_date": "2026-08-01",
  "source_total_articles": 174,
  "comparison_total_articles": 144,
  "exclusive_count": 89,
  "exclusive_articles": [
    {
      "article_id": 5401,
      "page_number": 1,
      "printed_page": "1",
      "section": "Goa News",
      "headline": "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS"
    },
    {
      "article_id": 5402,
      "page_number": 1,
      "printed_page": "1",
      "section": "Goa News",
      "headline": "CHICALIM PANCHAYAT SEALS 3 COMMERCIAL UNITS OVER TRADE LICENCE DUES"
    },
    {
      "article_id": 5415,
      "page_number": 2,
      "printed_page": "2",
      "section": "Regional",
      "headline": "MARGAO CITIZENS FORUM OPPOSES POWER TARIFF REVISION PETITION"
    }
  ],
  "summary": "Verified 89 exclusive articles in The Goan (mostly regional Goa news, local municipal decisions, and state transport alerts) that were not reported in The Morning Standard on 2026-08-01."
}
```

---

### Tool 3: `entity_search` (Multi-Hop Entity Graph & Salience Scoring)

[`backend/app/retrieval/entity_filter.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/entity_filter.py) queries the entity co-occurrence graph.

#### Input Arguments
```json
{
  "entity_name": "Corporation of the City of Panaji",
  "top_k": 10
}
```

#### Executed SQL Query
```sql
SELECT 
    ae.salience_score,
    ae.mention_count,
    e.name AS entity_name,
    e.type AS entity_type,
    a.id AS article_id,
    a.issue_id,
    a.headline,
    a.summary,
    a.prominence_score,
    n.name AS newspaper_name,
    i.issue_date
FROM article_entities ae
JOIN entities e ON ae.entity_id = e.id
JOIN articles a ON ae.article_id = a.id
JOIN issues i ON a.issue_id = i.id
JOIN newspapers n ON i.newspaper_id = n.id
WHERE e.name LIKE '%Corporation of the City of Panaji%'
ORDER BY ae.salience_score DESC, ae.mention_count DESC
LIMIT 10;
```

#### Output Returned to Agent
```json
[
  {
    "article_id": 5401,
    "issue_id": 93,
    "entity_name": "Corporation of the City of Panaji",
    "entity_type": "org",
    "salience_score": 0.95,
    "mention_count": 3,
    "headline": "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS",
    "newspaper_name": "The Goan",
    "issue_date": "2026-08-01",
    "pages": [1],
    "bboxes": [[[180.0, 220.0, 1650.0, 340.0]]],
    "summary": "CCP commences automated AI traffic surveillance issuing 450 e-challans on day one.",
    "prominence_score": 0.88,
    "source_tool": "entity_search"
  }
]
```

---

### Tool 4: `timeline_builder` (Narrative Chronological Trajectory)

[`backend/app/retrieval/timeline_builder.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/timeline_builder.py) groups articles chronologically across dates.

#### Input Arguments
```json
{
  "query": "Panaji Smart City Infrastructure",
  "limit": 30
}
```

#### Executed SQL Query
```sql
SELECT a.id, a.headline, a.summary, a.full_text, a.prominence_score,
       i.issue_date, n.name AS newspaper_name
FROM articles a
JOIN issues i ON a.issue_id = i.id
JOIN newspapers n ON i.newspaper_id = n.id
WHERE (a.headline LIKE '%Panaji Smart City%' OR a.full_text LIKE '%Panaji Smart City%')
ORDER BY i.issue_date ASC, a.primary_page_id ASC
LIMIT 30;
```

#### Output Returned to Agent
```json
{
  "query": "Panaji Smart City Infrastructure",
  "total_dates": 2,
  "total_articles": 3,
  "date_groups": [
    {
      "date": "2026-08-01",
      "newspaper_name": "The Goan",
      "articles_count": 2,
      "milestones": [
        {
          "article_id": 5401,
          "headline": "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS",
          "summary": "Automated surveillance operational at 14 junctions.",
          "section": "Goa News",
          "pages": [1]
        }
      ]
    },
    {
      "date": "2026-08-02",
      "newspaper_name": "The Goan",
      "articles_count": 1,
      "milestones": [
        {
          "article_id": 5512,
          "headline": "SMART CITY ROAD DIGGING DEADLINE EXTENDED TO AUGUST 15",
          "summary": "PWD cites monsoon delays for incomplete drainage works.",
          "section": "Goa News",
          "pages": [3]
        }
      ]
    }
  ]
}
```

---

### Tool 5: `coverage_analysis` (3-Tier Negative Coverage Audit)

[`backend/app/retrieval/coverage_analyzer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/coverage_analyzer.py) determines if an event was covered across multiple publications.

#### 3-Tier Audit Process
* **Tier 1 (Relational Check)**: Checks if issues exist for each newspaper on the target date.
* **Tier 2 (Hybrid Search)**: Runs `_hybrid_search.search(query, top_k=5, newspaper_id=N)`.
* **Tier 3 (Confidence Classification)**:
  - If Cross-Encoder rerank score $\ge 0.85 \implies$ `COVERED`
  - If $0.40 \le \text{score} < 0.85 \implies$ `PARTIAL`
  - If score $< 0.40 \implies$ `OMITTED` (Negative Coverage Confirmed)

#### Output Returned to Agent
```json
{
  "query": "Panaji smart traffic challans",
  "target_date": "2026-08-01",
  "reconciliation_matrix": [
    {
      "newspaper_name": "The Goan",
      "status": "COVERED",
      "confidence": 0.94,
      "top_score": 0.9421,
      "matched_article_ids": [5401],
      "matched_headlines": ["PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS"],
      "evidence_snippet": "Panaji: The Corporation of the City of Panaji (CCP) on Friday officially commenced..."
    },
    {
      "newspaper_name": "The Morning Standard",
      "status": "OMITTED",
      "confidence": 0.96,
      "top_score": 0.1215,
      "matched_article_ids": [],
      "matched_headlines": [],
      "evidence_snippet": "Zero relevant coverage found. Top retrieved story was an unrelated New Delhi MCD report."
    }
  ]
}
```

---

### Tool 6: `web_search` (Live Web Verification Fallback)

Used when live search is toggled or when broadsheet archives lack coverage.

#### Input Arguments
```json
{
  "query": "Panaji municipal corporation traffic challans August 2026",
  "num_results": 5
}
```

#### Output Returned to Agent
```json
[
  {
    "title": "Panaji CCP deploys AI cameras for traffic fines",
    "url": "https://www.heraldgoa.in/news/goa/ccp-deploys-ai-cameras/210452",
    "snippet": "The Corporation of the City of Panaji has activated 14 camera corridors to catch helmetless riders and red light jumpers.",
    "newspaper_name": "Live Web",
    "issue_date": "2026-08-01",
    "source_tool": "web_search",
    "is_web": true
  }
]
```

---

## 6. Phase 5: Corrective RAG (CRAG) Relevance Gate & Fallbacks

[`backend/app/agent/graph.py:_evaluate_evidence_node()`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/graph.py#L770-L860) grades the raw evidence pool before the LLM prompt is assembled.

```
                           CORRECTIVE RAG (CRAG) PIPELINE
                           
                   [ Raw Evidence Items from Executed Tools ]
                                        │
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 1. Macro Structural Manifest Bypass                                    │
   │    • Is source_tool in ('sql_analytics', 'coverage_analysis')?         │
   │    • Yes ==> PRESERVE UNCONDITIONALLY (Relevance = 1.0)                │
   └────────────────────────────────────┬───────────────────────────────────┘
                                        │ (No)
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 2. Stemmed Query Relevance Scoring                                     │
   │    • Query: "traffic challans" ==> stems: {'traff', 'challan'}         │
   │    • Headline match: +2.0                                              │
   │    • Snippet / Body match: +1.0                                        │
   │    • Score = matches / len(query_tokens)                               │
   │    • If Score == 0.0 ==> PRUNE FROM CONTEXT                            │
   └────────────────────────────────────┬───────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 3. Grounding Verification & Corrective Fallback                        │
   │    • Are there any grounded items with length >= 20 chars?             │
   │    • If NO (0 hits):                                                   │
   │      - Step A: Fallback to entity_search(top_k=5)                      │
   │      - Step B: If still 0 and web enabled ==> web_search(num_results=4)│
   └────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Macro Manifest Protection

In earlier implementations, relational issue manifests (e.g. lists of all 174 articles) were pruned because their combined summaries did not match single search words. CRAG now guarantees that:
```python
if _is_structural_or_macro_evidence(item):
    item_relevance = 1.0  # Retain manifest unconditionally
```

---

## 7. Phase 6: Answer Synthesis, Prompt Budgeting & SSE Streaming

[`backend/app/agent/synthesizer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/synthesizer.py) prepares the final LLM prompt.

### 6.1 Evidence Context Budgeting & Publication Scoping

1. **Top 12 Item Cap**: Slices `evidence_items[:12]` to ensure prompt length never exceeds $\le 3,500$ tokens.
2. **Selective Length Allocations**:
   - Manifests, Reconciliation Matrices, and Coverage Differences: **up to 4,000 characters**.
   - Standard Article Excerpts: **up to 1,200 characters**.
3. **Publication Scoping Header**: Dynamically scans the evidence pool and injects an active publication barrier:
   ```text
   === CRITICAL CONVERSATION HISTORY GUARD ===
   VERIFIED AVAILABLE PUBLICATIONS FOR THIS QUERY:
     - The Goan (Issue Date: 2026-08-01)
     - The Morning Standard (Issue Date: 2026-08-01)
   Any newspaper not in this list does NOT exist for this query.
   You must NEVER mention, cite, or invent coverage from any other publication.
   ===========================================
   ```

---

### 6.2 The Complete Synthesizer Prompt Structure

```text
[SYSTEM PROMPT: SYNTHESIZER_SYSTEM_PROMPT]
- 4-Tier Broadsheet Format Required:
  ### ⚡ Executive Summary
  ### 📌 Key Verified Facts & Highlights
  ### 📰 Broadsheet Perspectives & Focus Areas
  ### 🔍 Explore Further
- Strict Inline Citation Rule: [{Newspaper}, {YYYY-MM-DD}, Page {P}, "{Headline}"]
- Anti-Hallucination Hard Stop if 0 Evidence.

[USER PROMPT]
User Query: "list all those 11 articles in The Goan but not in The Morning Standard dated 2026-08-01"

=== CRITICAL CONVERSATION HISTORY GUARD ===
VERIFIED AVAILABLE PUBLICATIONS FOR THIS QUERY:
  - The Goan (Issue Date: 2026-08-01)
  - The Morning Standard (Issue Date: 2026-08-01)
===========================================

Available Newspaper Evidence (Grounded Broadsheet Archive):

--- ARCHIVE EVIDENCE EXCERPT [1] ---
[Evidence: The Goan, 2026-08-01, Page 1 (PDF Page 1), Headline: "COVERAGE DIFFERENCE: THE GOAN vs THE MORNING STANDARD"]
Source: The Goan
Date: 2026-08-01
Content:
VERIFIED EXCLUSIVE COVERAGE: THE GOAN (Absent in The Morning Standard)
Total Exclusives Found: 89

1. Page 1 (PDF Page 1) [Goa News]: "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS"
2. Page 1 (PDF Page 1) [Goa News]: "CHICALIM PANCHAYAT SEALS 3 COMMERCIAL UNITS OVER TRADE LICENCE DUES"
3. Page 2 (PDF Page 2) [Regional]: "MARGAO CITIZENS FORUM OPPOSES POWER TARIFF REVISION PETITION"
...
```

---

### 6.3 LLM Generation, `<think>` Tag Separation & Inline Citations

The LLM outputs reasoning traces inside `<think>...</think>`, followed by the structured response. [`parse_thought_and_answer()`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/synthesizer.py#L84) splits these into live UI components.

#### Raw LLM Response Stream
```markdown
<think>
User asked to list exclusive articles in The Goan that were omitted from The Morning Standard on 2026-08-01.
Evidence excerpt 1 provides the exact deterministic difference from sql_analytics.
I will structure the response with an Executive Summary, bulleted list of the requested verified articles with citations, and perspective comparison.
</think>

### ⚡ Executive Summary
On August 1, 2026, *The Goan* published extensive regional and hyperlocal coverage that was omitted from *The Morning Standard*, led by Panaji's municipal traffic automation and civic enforcement in South Goa.

### 📌 Key Verified Facts & Highlights
* **Automated Civic Surveillance**: The Corporation of the City of Panaji rolled out AI surveillance across 14 arterial junctions, issuing 450 contactless e-challans on day one [*The Goan*, 2026-08-01, Page 1, "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS"].
* **Commercial Licencing Enforcement**: Chicalim Panchayat authorities sealed three major commercial complexes for long-pending trade licence dues [*The Goan*, 2026-08-01, Page 1, "CHICALIM PANCHAYAT SEALS 3 COMMERCIAL UNITS OVER TRADE LICENCE DUES"].
* **Public Utility Pushback**: The Margao Citizens Forum formally submitted an objection against the state electricity department's tariff hike petition [*The Goan*, 2026-08-01, Page 2, "MARGAO CITIZENS FORUM OPPOSES POWER TARIFF REVISION PETITION"].

### 📰 Broadsheet Perspectives & Focus Areas
* **The Goan Focus**: Heavy editorial emphasis on municipal accountability, panchayat administration, and local Goan public utility rates.
* **The Morning Standard Focus**: Completely omitted these state and civic developments, prioritizing national capital policy and federal announcements.

### 🔍 Explore Further
> 💡 Explore: What penalty amounts were specified for traffic violations in Panaji?
> 💡 Explore: What was the revenue impact of the Chicalim panchayat sealing drive?
```

---

### 6.4 Server-Sent Events (SSE) Wire Protocol

[`backend/app/api/routers/query.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/routers/query.py) streams the synthesis to the client:

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

event: stage
data: {"stage": "planning", "message": "Planning query strategy..."}

event: stage
data: {"stage": "tools", "message": "Executing sql_analytics and hybrid_search..."}

event: thought
data: {"token": "User asked to list exclusive articles in The Goan..."}

event: token
data: {"token": "### ⚡ Executive Summary\n"}

event: token
data: {"token": "On August 1, 2026, *The Goan* published..."}

event: citations
data: [
  {
    "article_id": 5401,
    "newspaper_name": "The Goan",
    "issue_date": "2026-08-01",
    "page_number": 1,
    "headline": "PANAJI MUNICIPAL CORPORATION ROLLS OUT SMART TRAFFIC CHALLANS",
    "bbox": [[180.0, 220.0, 1650.0, 340.0], [180.0, 410.0, 890.0, 820.0]]
  }
]

event: done
data: {"status": "completed", "latency_ms": 1420, "cost_usd": 0.0031}
```

---

## 8. Phase 7: Anti-Hallucination Guardrails & Context Isolation

NewsLens-AI enforces four strict anti-hallucination layers:

```
                            ANTI-HALLUCINATION SHIELD
                            
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. Empty Evidence Hard-Stop                                            │
  │    • If CRAG yields 0 articles, Synthesizer short-circuits:            │
  │      "The uploaded broadsheet archives do not contain verified         │
  │       reporting on this topic."                                        │
  └────────────────────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 2. Pre-Training Memo Header Stripper                                   │
  │    • Strip memorized pre-training boilerplate:                         │
  │      e.g. "Date: October 26, 2023 (Current Analysis)"                  │
  │    • Regex enforces that ONLY dates from evidence chunks appear.       │
  └────────────────────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 3. Cross-Turn Publication Scoping Barrier                              │
  │    • Explicitly injects the list of allowed publications.              │
  │    • Prevents topics from Turn 1 (e.g. LIV Golf) from leaking into     │
  │      Turn 2 queries for a new date.                                    │
  └────────────────────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 4. Deterministic Set Differences (sql_analytics)                       │
  │    • Computes exclusion counts directly in MySQL and Python.           │
  │    • The LLM is never allowed to guess article exclusion counts.       │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Phase 8: Comprehensive Top-K Lifecycle Reference

The table below traces how `top_k` transforms across every stage of the pipeline:

| Stage | Component | Default Value | Concrete Example | Reason & Effect |
|---|---|---|---|---|
| **Archetype Default** | `planner.py` | `top_k = 6` | `factual_lookup` | High precision, low distraction for specific factual questions. |
| **Archetype Comparison** | `planner.py` | `top_k = 12` | `cross_newspaper_comparison` | Retrieves 2–3 articles from each of 4–6 publications. |
| **Dense Candidate Fetch** | `hybrid_search.py` | $top\_k \times 3$ | $6 \times 3 = 18$ | Oversamples dense candidates from Qdrant prior to rank fusion. |
| **Sparse Candidate Fetch** | `hybrid_search.py` | $top\_k \times 3$ | $6 \times 3 = 18$ | Oversamples exact keyword matches from MySQL FULLTEXT. |
| **RRF Rank Constant** | `hybrid_search.py` | $k = 60$ | $\frac{1}{60 + \text{rank}}$ | Standard Cormack constant balancing dense and sparse ranks. |
| **Reranker Candidate Pool**| `hybrid_search.py` | $\max(75, top\_k \times 3)$ | 75 candidates | Balances 99%+ recall with sub-50ms Cross-Encoder latency. |
| **Cross-Encoder Slicing** | `reranker.py` | `top_k` | Returns top 6 | Cuts off candidates by cross-attention interaction score. |
| **CRAG Fallback Depth** | `graph.py` | `top_k = 5` | Entity fallback | Fallback search depth when initial retrieval returns 0 hits. |
| **Evidence Context Cap** | `synthesizer.py` | Top 12 items | `evidence[:12]` | Caps evidence tokens at $\le 3,500$ to prevent local LLM overflow. |

---

## 10. Phase 9: Information Retrieval & Generation Evaluation Metrics

[`backend/app/evaluation/metrics.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/evaluation/metrics.py) provides quantitative benchmarks:

### 1. Recall@K
$$\text{Recall@K} = \frac{|\text{Retrieved}_{1..K} \cap \text{GroundTruth}|}{|\text{GroundTruth}|}$$
*Example*: If a query has 4 ground truth articles and 3 are in the top 6 retrieved:
$$\text{Recall@6} = \frac{3}{4} = 0.75 \quad (75\%)$$

### 2. Precision@K
$$\text{Precision@K} = \frac{|\text{Retrieved}_{1..K} \cap \text{GroundTruth}|}{K}$$
*Example*:
$$\text{Precision@6} = \frac{3}{6} = 0.50 \quad (50\%)$$

### 3. Mean Reciprocal Rank (MRR)
$$\text{MRR} = \frac{1}{\text{Rank of First Relevant Item}}$$
*Example*: If the first relevant article is ranked at position 2:
$$\text{MRR} = \frac{1}{2} = 0.50$$

### 4. Normalized Discounted Cumulative Gain (NDCG@K)
$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

### 5. Citation Precision & Recall
$$\text{Citation Recall} = \frac{\text{Number of Claims with Verified Bracketed Citations}}{\text{Total Factual Claims Made}}$$
$$\text{Citation Precision} = \frac{\text{Number of Accurate Citations Grounded in Bounding Boxes}}{\text{Total Citations Output}}$$
NewsLens-AI requires **$100\%$ Citation Precision**: every bracketed citation `[Newspaper, YYYY-MM-DD, Page P, "Headline"]` must map to an existing row in `articles` and `article_pages`.

---

*End of End-to-End Data Flow & Data Structure Guide.*
