# NewsLens-AI Database Schema Reference Guide

This document is the definitive guide to the **NewsLens-AI** relational database schema (MySQL 8 with InnoDB engine). It details the architectural role of each table in the newspaper processing lifecycle, relationships, and provides a comprehensive breakdown of every single column: what it stores, concrete real-world examples, and the architectural rationale behind its design.

---

## 1. Why a Relational System of Record is Needed

A broadsheet newspaper presents unique data modeling challenges compared to standard web documents:
- **Spatial Fragmentation**: A front-page story routinely begins on Page 1, spans across multiple column tracks, breaks, and jumps to an interior continuation page (e.g., "Continued on Page 14, Col 2").
- **Multi-Modal Complexity**: Pages interleave editorial kickers, headlines, sub-decks, drop caps, photo crops with captions, bylines, weather matrices, and financial tables.
- **Relational Integrity**: Vector databases (like Qdrant) only store high-dimensional embeddings and coarse text payloads for approximate nearest-neighbor search. They cannot enforce foreign keys, execute cascading cleanups, maintain ACID transactional guarantees during multi-stage ingestion, or run fast relational filters (such as date ranges, editions, and categories).
- **Auditability & Observability**: Long-running asynchronous worker pipelines (rasterization $\to$ OCR $\to$ segmentation $\to$ classification $\to$ indexing) require explicit state machines, progress tracking, and error logging.

NewsLens-AI cleanly separates concerns across storage tiers:
1. **MinIO**: High-resolution raster images, cropped photo assets, and original PDFs.
2. **Qdrant**: High-dimensional vector embeddings (`article_chunks`) for semantic similarity search.
3. **MySQL (The 17 Tables)**: Relational system of record enforcing structural integrity, canonical taxonomies, named entity graphs, FULLTEXT search indexes, and provenance lineage.

---

## 2. Entity Relationship Overview

```
[ingestion_jobs] ──(source_zip_id)──┐
                                     │
[newspapers] 1 ──< [issues] 1 ──< [pages]
                      │                │
                      │ 1              │ 1
                      │                │
                      ├──< [articles]  ├──< [photos]
                      │       │   │    └──< [tables]
                      │       │   │
                      │       │   └──< [article_chunks] ──> (Qdrant Point UUID)
                      │       │
                      │       ├──< [article_pages] (Junction to [pages] + BBoxes)
                      │       ├──< [article_entities] >── [entities] (Canonical NER)
                      │       ├──< [article_topics]   >── [topics]
                      │       ├──< [article_events]   >── [events]
                      │       └──> [article_categories] (Canonical 13-class Taxonomy)
                      │
[query_log] (Independent audit log for Agentic RAG questions, tool plans & citations)
```

---

## 3. Detailed Table & Column Specifications

---

### Table 1: `newspapers`
* **Why Needed**: Top-level catalog of publication brands. A single installation may ingest multiple distinct newspaper brands (e.g. *The Daily Chronicle*, *Financial Express*), each with its own language conventions, regional focus, and publication frequencies.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `1` | Primary surrogate key uniquely identifying the publication. |
| `name` | `VARCHAR(255)` | NO | `"The Daily Chronicle"` | Official masthead brand name. Displayed in search results and UI citation badges. |
| `publisher` | `VARCHAR(255)` | YES | `"Chronicle Media Group Ltd."` | The publishing entity or syndicate. |
| `default_language` | `VARCHAR(10)` | YES | `"en"` | ISO 639-1 language code (e.g. `en`, `hi`). Guides default OCR models and VLM prompt bias. |
| `country` | `VARCHAR(100)` | YES | `"United States"` | Geographical origin, used for geopolitical query scoping and filtering. |
| `created_at` | `DATETIME` | NO | `"2026-08-21 08:30:00"` | Timestamp when this newspaper brand profile was created. |

---

### Table 2: `ingestion_jobs`
* **Why Needed**: Newspaper intake runs asynchronously across Celery background workers. Ingestion jobs track multi-file uploads (ZIP archives, batches of PDFs), monitoring progress, errors, and lifecycle status to prevent partial or orphaned data ingestion.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `101` | Unique surrogate key for the ingestion run. |
| `source_type` | `ENUM` | NO | `"zip"` | Format of the incoming payload: `single_pdf`, `multi_pdf`, `folder`, `zip`. Dictates unpacking and processing logic. |
| `status` | `ENUM` | NO | `"completed"` | Job lifecycle: `pending`, `running`, `completed`, `failed`, `partial`. Enables UI progress tracking and worker coordination. |
| `total_files` | `INT` | NO | `12` | Total number of individual edition files or PDFs detected in the batch. |
| `processed_files` | `INT` | NO | `12` | Number of files successfully processed through all pipeline stages. |
| `failed_files` | `INT` | NO | `0` | Count of files that failed pipeline execution. |
| `error_log` | `JSON` | YES | `[{"file": "p4.pdf", "stage": "ocr", "error": "Timeout"}]` | Detailed error stack traces per file and pipeline stage for operational debugging. |
| `started_at` | `DATETIME` | YES | `"2026-08-21 09:00:15"` | Timestamp when worker claimed and initiated the job. Used for SLA and throughput metrics. |
| `completed_at` | `DATETIME` | YES | `"2026-08-21 09:04:45"` | Timestamp when worker finalized all stages. |
| `created_at` | `DATETIME` | NO | `"2026-08-21 09:00:00"` | Timestamp when upload was registered via API. |

---

### Table 3: `issues`
* **Why Needed**: Represents a specific daily or weekly edition of a newspaper. Articles, pages, and citations must trace back to a specific printed issue on a specific date.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `501` | Unique surrogate key for the issue. |
| `newspaper_id` | `INT` (FK) | NO | `1` | Foreign key referencing `newspapers.id` (`ON DELETE CASCADE`). |
| `issue_date` | `DATE` | NO | `"2026-08-20"` | Physical date of print extracted from the consensus masthead header. |
| `edition` | `VARCHAR(100)` | YES | `"Late City Final"` | Edition identifier (e.g. "Morning", "Evening", "Delhi Late"). Accommodates regional print differences. |
| `language` | `VARCHAR(10)` | YES | `"en"` | Primary language of this specific edition. |
| `total_pages` | `INT` | YES | `24` | Total page count detected after PDF splitting or unpacking. |
| `source_zip_id` | `INT` (FK) | YES | `101` | Foreign key referencing `ingestion_jobs.id` (`ON DELETE SET NULL`) for provenance tracking. |
| `ingestion_status`| `VARCHAR(50)` | NO | `"completed"` | Issue-level status state machine (`pending`, `rasterized`, `indexed`, etc.). |
| `created_at` | `DATETIME` | NO | `"2026-08-21 09:00:20"` | When this issue record was inserted into the database. |

*Constraints*: `UNIQUE(newspaper_id, issue_date, edition)` prevents duplicate ingestion of the exact same printed edition.

---

### Table 4: `pages`
* **Why Needed**: Broadsheet layout analysis, OCR, visual asset harvesting, and raster preview rendering operate on a per-page basis. Articles and photo coordinates map directly to a physical page's pixel dimensions.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `2401` | Unique surrogate key for the page. |
| `issue_id` | `INT` (FK) | NO | `501` | Foreign key referencing `issues.id` (`ON DELETE CASCADE`). |
| `page_number` | `INT` | NO | `1` | Sequential physical order in the PDF (1-indexed). |
| `raster_object_key`| `VARCHAR(512)`| YES | `"pages/501/page_1.png"` | S3/MinIO object path storing the lossless 300 DPI page rendering. |
| `width_px` | `INT` | YES | `2480` | Image pixel width, critical for normalizing bounding box coordinates (`0.0` to `1.0`). |
| `height_px` | `INT` | YES | `3508` | Image pixel height, critical for bounding box normalization. |
| `ocr_confidence` | `FLOAT` | YES | `0.942` | Mean OCR confidence score across all words on this page. Used to detect degraded scans. |
| `printed_page_number`| `VARCHAR(50)`| YES | `"A1"` or `"Page 1"` | The folio page label printed on the header/footer (may differ from sequential index). |
| `is_advertisement_page`| `BOOLEAN` | NO | `0` (false) | Flag indicating a full-page commercial advert. Skips heavy article segmentation. |
| `ingestion_status`| `VARCHAR(50)` | NO | `"indexed"` | Page lifecycle status (`pending` $\to$ `rasterized` $\to$ `layout_done` $\to$ `ocr_done` $\to$ `segmented` $\to$ `classified` $\to$ `embedded` $\to$ `indexed`). |

*Constraints*: `UNIQUE(issue_id, page_number)` guarantees exactly one entry per physical sheet per issue.

---

### Table 5: `article_categories`
* **Why Needed**: Canonical newsroom taxonomy (13 standard categories such as Politics, Business, Sports, Technology). Standardizes classification, powering structured faceted filtering.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `2` | Unique surrogate key for the category. |
| `name` | `VARCHAR(64)` | NO | `"Business & Markets"` | Canonical name of the news category. Unique constraint applied. |
| `parent_id` | `INT` (FK) | YES | `NULL` (or `2` for sub-topic) | Self-referential foreign key for hierarchical nesting (e.g. Markets $\to$ Equities). |

---

### Table 6: `articles`
* **Why Needed**: The foundational narrative unit in NewsLens-AI. Represents a complete, assembled journalistic story across column tracks and jump pages, with headlines, bylines, editorial prominence, and lexical search capabilities.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `9012` | Unique surrogate key for the article. |
| `issue_id` | `INT` (FK) | NO | `501` | Foreign key referencing `issues.id` (`ON DELETE CASCADE`). |
| `primary_page_id` | `INT` (FK) | YES | `2401` | Foreign key referencing `pages.id` where the story begins. |
| `category_id` | `INT` (FK) | YES | `2` | Foreign key referencing canonical `article_categories.id`. |
| `category_confidence`| `FLOAT` | YES | `0.96` | Confidence score from the probabilistic 12-domain classifier. |
| `headline` | `VARCHAR(1024)` | YES | `"RBI Holds Benchmark Repo Rate at 6.5%"` | Primary title extracted across multi-column headline blocks. |
| `subheadline` | `VARCHAR(1024)` | YES | `"Inflation trajectory remains within target band"` | Secondary sub-deck or dek explaining the headline. |
| `byline_author` | `VARCHAR(512)` | YES | `"By Priya Sharma, Special Correspondent"` | Author or news agency (e.g. Reuters, PTI) attribution. |
| `section` | `VARCHAR(255)` | YES | `"Economy"` | Broad section name identified by layout or header ribbons. |
| `printed_section` | `VARCHAR(128)` | YES | `"Business Page"` | Verbatim section header text as printed in the page masthead. |
| `article_type` | `ENUM` | NO | `"news"` | Classification archetype: `news`, `editorial`, `opinion`, `analysis`, `advertisement`, `sidebar`, `photo_caption`, `table_data`, `letter`, `obituary`, `review`, etc. |
| `language` | `VARCHAR(10)` | YES | `"en"` | Language code detected for this specific story. |
| `prominence_score`| `FLOAT` | NO | `0.875` | Editorial weight calculated from page placement, headline font size, column span, and photos. Crucial for ranking search results. |
| `word_count` | `INT` | NO | `645` | Total word count of the assembled full text. |
| `summary` | `TEXT` | YES | `"The central bank decided to hold interest rates steady..."` | Short executive abstract generated by LLM for rapid UI preview cards. |
| `full_text` | `LONGTEXT` | YES | `"MUMBAI — The Reserve Bank of India on Thursday kept..."` | Complete reconstructed text of the article across all columns and jump pages. Indexed via MySQL FULLTEXT. |
| `created_at` | `DATETIME` | NO | `"2026-08-21 09:02:15"` | Timestamp of article assembly. |

*Indexes*: `FULLTEXT(headline, full_text)` powers fast BM25 lexical keyword retrieval and hybrid search.

---

### Table 7: `article_pages`
* **Why Needed**: Junction table modeling the many-to-many relationship between articles and pages. When an article jumps from Page 1 to Page 14, this table stores the bounding box coordinates on each page so the frontend can visually highlight the exact columns clicked by a user.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `14102` | Unique surrogate key for the junction row. |
| `article_id` | `INT` (FK) | NO | `9012` | Foreign key referencing `articles.id` (`ON DELETE CASCADE`). |
| `page_id` | `INT` (FK) | NO | `2401` | Foreign key referencing `pages.id` (`ON DELETE CASCADE`). |
| `page_number` | `INT` | NO | `1` | Physical page number where this segment of the story appears. |
| `printed_page_number`| `VARCHAR(50)`| YES | `"A1"` | Folio page designation for display. |
| `bbox_json` | `JSON` | YES | `{"bboxes": [[120, 450, 680, 1100], [700, 450, 1260, 900]]}` | Array of column bounding boxes `[ymin, xmin, ymax, xmax]` occupied by this article on this page. |
| `block_order` | `INT` | NO | `0` | Sequence index (`0` for lead segment, `1` for continuation segment) to preserve reading order. |

---

### Table 8: `article_chunks`
* **Why Needed**: Deconstructs lengthy articles into overlapping semantic chunks for embedding and vector similarity search, while keeping MySQL and Qdrant tightly coupled through deterministic point IDs.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `45001` | Unique surrogate key for the chunk. |
| `article_id` | `INT` (FK) | NO | `9012` | Foreign key referencing `articles.id` (`ON DELETE CASCADE`). |
| `chunk_index` | `INT` | NO | `0` | Positional index of the chunk within the article (`0`, `1`, `2`, ...). |
| `chunk_type` | `VARCHAR(20)` | NO | `"text"` | Content type: `text`, `table`, `visual_caption`. Adjusts embedding strategies. |
| `text` | `TEXT` | NO | `"The monetary policy committee voted unanimously to maintain..."` | The exact semantic text passage used to produce the dense vector embedding. |
| `token_count` | `INT` | NO | `348` | Number of BPE tokens in this chunk, ensuring chunking stays within embedding limits (e.g. 512 or 8192 tokens). |
| `embedding_vector_id`| `VARCHAR(255)`| YES | `"f47ac10b-58cc-4372-a567-0e02b2c3d479"` | UUID pointer to the vector payload in Qdrant. High-dimensional float vectors are kept out of MySQL for performance. |

---

### Table 9: `photos`
* **Why Needed**: Newspapers rely heavily on imagery. Photos, political cartoons, diagrams, and maps are cropped from the page, described by Vision-Language Models (VLMs), and linked to articles for multimodal retrieval.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `304` | Unique surrogate key for the visual asset. |
| `article_id` | `INT` (FK) | YES | `9012` | Foreign key referencing `articles.id` (`ON DELETE SET NULL`). Spatially resolved to the story it illustrates. |
| `page_id` | `INT` (FK) | NO | `2401` | Foreign key referencing `pages.id` (`ON DELETE CASCADE`) where the photo was printed. |
| `bbox_json` | `JSON` | YES | `{"ymin": 0.12, "xmin": 0.05, "ymax": 0.35, "xmax": 0.45}` | Normalized crop coordinates of the photo box on the page. |
| `caption` | `TEXT` | YES | `"Governor Shaktikanta Das addressing the press conference on Friday."` | Physical caption text printed underneath or beside the photograph. |
| `vlm_description` | `TEXT` | YES | `"A man in a navy suit speaking into microphones with Reserve Bank logo backdrop."` | Rich visual description synthesized by Qwen2.5-VL / Gemini Vision for multimodal search. |
| `visual_type` | `VARCHAR(50)` | YES | `"photo"` | Classification: `photo`, `illustration`, `chart`, `map`, `cartoon`. |
| `object_key` | `VARCHAR(512)`| YES | `"photos/501/p1_photo_01.jpg"` | MinIO/S3 object key to the cropped high-res JPEG asset. |

---

### Table 10: `tables`
* **Why Needed**: Tabular structures (stock market summaries, sports boxes, economic indices) cannot be understood via simple linear OCR text. Storing structured JSON matrices alongside visual crops enables quantitative Q&A.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `88` | Unique surrogate key for the table. |
| `article_id` | `INT` (FK) | YES | `9012` | Foreign key referencing `articles.id` (`ON DELETE SET NULL`). |
| `page_id` | `INT` (FK) | NO | `2401` | Foreign key referencing `pages.id` (`ON DELETE CASCADE`). |
| `bbox_json` | `JSON` | YES | `{"ymin": 0.65, "xmin": 0.10, "ymax": 0.85, "xmax": 0.48}` | Normalized bounding box coordinates of the table region. |
| `extracted_json` | `JSON` | YES | `{"headers": ["Policy", "Rate"], "rows": [["Repo", "6.5%"], ["Reverse Repo", "3.35%"]]}` | Matrix JSON structure produced by the Dual VLM + Spatial OCR parser. |
| `object_key` | `VARCHAR(512)`| YES | `"tables/501/p1_table_01.png"` | MinIO object key storing the cropped visual snippet of the table. |

---

### Table 11: `entities`
* **Why Needed**: Canonical Named Entity Recognition (NER) master registry. Unifies multiple name variants (e.g. "Joe Biden", "President Biden") into a single canonical entity, enabling cross-edition entity graphs.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `801` | Unique surrogate key for the entity. |
| `name` | `VARCHAR(512)` | NO | `"Shaktikanta Das"` | The standard resolved string name of the entity. |
| `type` | `ENUM` | NO | `"person"` | Entity classification: `person`, `org`, `location`, `misc`. |
| `canonical_id` | `INT` (FK) | YES | `NULL` (or `801` if merged) | Self-referential foreign key referencing `entities.id` (`ON DELETE SET NULL`) for entity de-duplication and aliases. |

*Constraints*: `UNIQUE(name, type)` ensures deterministic deduplication during NER extraction.

---

### Table 12: `article_entities`
* **Why Needed**: Join table associating articles with entities, capturing both mention frequency and narrative salience.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `article_id` | `INT` (FK, PK)| NO | `9012` | Composite primary key referencing `articles.id` (`ON DELETE CASCADE`). |
| `entity_id` | `INT` (FK, PK)| NO | `801` | Composite primary key referencing `entities.id` (`ON DELETE CASCADE`). |
| `mention_count` | `INT` | NO | `6` | Total count of occurrences of this entity within the article full text. |
| `salience_score` | `FLOAT` | NO | `0.85` | Importance weight (`0.0` to `1.0`) indicating how central the entity is to the narrative. |

---

### Table 13: `topics`
* **Why Needed**: Thematic subjects (e.g. "Monetary Policy", "Renewable Energy") that transcend newspaper page sections.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `42` | Unique surrogate key for the topic. |
| `name` | `VARCHAR(512)` | NO | `"Monetary Policy"` | Unique name of the topic. |
| `taxonomy_path` | `VARCHAR(1024)`| YES | `"Economy > Central Banking > Interest Rates"` | Hierarchical taxonomy breadcrumb for faceted filtering and navigation. |

---

### Table 14: `article_topics`
* **Why Needed**: Junction table mapping articles to topics, preserving model assignment confidence.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `article_id` | `INT` (FK, PK)| NO | `9012` | Composite primary key referencing `articles.id` (`ON DELETE CASCADE`). |
| `topic_id` | `INT` (FK, PK)| NO | `42` | Composite primary key referencing `topics.id` (`ON DELETE CASCADE`). |
| `confidence` | `FLOAT` | NO | `0.92` | Model confidence score (`0.0` to `1.0`) for topic association. |

---

### Table 15: `events`
* **Why Needed**: Models real-world temporal events (e.g. "G20 Summit 2026", "Lok Sabha Elections 2024") that unfold over multiple days and editions, powering chronological timeline generation.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `77` | Unique surrogate key for the event. |
| `name` | `VARCHAR(1024)`| NO | `"August 2026 RBI Monetary Policy Committee Meeting"` | Formal name or headline event title. |
| `canonical_date` | `DATE` | YES | `"2026-08-20"` | Primary calendar date on which the event occurred. |
| `description` | `TEXT` | YES | `"Bi-monthly rate decision meeting held by the Reserve Bank of India."` | Contextual narrative explaining the event. |
| `event_cluster_id`| `INT` (FK) | YES | `NULL` | Self-referential foreign key referencing `events.id` (`ON DELETE SET NULL`) for clustering ongoing stories. |

---

### Table 16: `article_events`
* **Why Needed**: Connects multiple news reports across dates and newspapers to their common underlying event.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `article_id` | `INT` (FK, PK)| NO | `9012` | Composite primary key referencing `articles.id` (`ON DELETE CASCADE`). |
| `event_id` | `INT` (FK, PK)| NO | `77` | Composite primary key referencing `events.id` (`ON DELETE CASCADE`). |
| `confidence` | `FLOAT` | NO | `0.98` | Match confidence between the reporting article and the canonical event. |

---

### Table 17: `query_log`
* **Why Needed**: Operational audit trail and latency/cost observability log for every natural language question answered by the Agentic RAG system. Enables inspection of LangGraph agent tool calls, citation validity, and query classification.

| Column | Type | Nullable | Example | Why / Description |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `INT` (PK, AI) | NO | `1204` | Unique surrogate key for the query execution record. |
| `user_id` | `VARCHAR(255)` | YES | `"usr_9981a"` | Identity or session token of the requesting user. |
| `query_text` | `TEXT` | NO | `"What was the RBI repo rate decision in August 2026?"` | The exact user query prompt submitted. |
| `query_type` | `VARCHAR(100)` | YES | `"factual_lookup"` | Archetype assigned by query router: `factual_lookup`, `comparative_timeline`, `entity_deepdive`, `visual_query`. |
| `plan_json` | `JSON` | YES | `{"steps": ["lookup_articles", "check_tables"]}` | Agentic plan generated by the LLM orchestrator detailing tool execution strategy. |
| `tool_calls_json` | `JSON` | YES | `[{"tool": "sql_search", "args": {"keyword": "repo rate"}, "results_count": 4}]` | Complete trace of tools executed, input parameters, and returned payloads. |
| `answer_text` | `LONGTEXT` | YES | `"On August 20, 2026, the RBI MPC held the benchmark repo rate at 6.5%..."` | Synthesized natural language response delivered to the user. |
| `citations_json` | `JSON` | YES | `[{"newspaper": "The Daily Chronicle", "date": "2026-08-20", "page": 1, "article_id": 9012}]` | Structured citations backing every factual claim with exact page and article IDs. |
| `latency_ms` | `INT` | YES | `850` | Total wall-clock time in milliseconds to plan, retrieve, and synthesize the answer. |
| `cost_usd` | `FLOAT` | YES | `0.0034` | Estimated USD cost incurred across LLM API tokens for the query. |
| `model_provider` | `VARCHAR(100)` | YES | `"gemini-2.5-flash"` | Provider and model that handled query planning and response generation. |
| `created_at` | `DATETIME` | NO | `"2026-08-31 01:25:00"` | Timestamp when the query was executed (indexed for telemetry dashboards). |

---

## 4. Key Architectural Patterns & Guarantees

1. **Foreign Key Deletion Integrity**:
   - `ON DELETE CASCADE` ensures that deleting a newspaper or issue cleanly prunes all associated pages, articles, bounding boxes, chunks, and junction records without leaving orphaned records.
   - `ON DELETE SET NULL` protects visual media assets (`photos`, `tables`) and canonical entities from inadvertent loss if an article or canonical record is reassigned.
2. **Decoupling Vector Payloads from Relational Records**:
   - MySQL stores structured attributes, bounding boxes, and full text for BM25 search. High-dimensional 1024-d float embeddings are stored in Qdrant and linked using `article_chunks.embedding_vector_id` (UUID), maintaining MySQL buffer pool efficiency.
3. **Multi-Page Article Stitching**:
   - Articles spanning multiple pages are stored as a single `articles` row linked via `article_pages` with ordered bounding boxes, ensuring that full stories can be retrieved as a single coherent text block while still linking back to exact page coordinates.

---

## 5. Agentic Query JSON Request Schemas & Tool Payloads

This section details the JSON structures passed into the Agentic RAG engine and generated by the Query Planner (`app/agent/planner.py`) as it routes questions to specialized retrieval backends.

### 5.1 Client Query Request (`POST /api/query` and `POST /api/query/stream`)
The frontend or client sends this JSON payload to initiate a research query:

```json
{
  "query": "What did Mint and Business Standard report regarding the RBI repo rate on August 20, 2026?",
  "user_id": "usr_9981a",
  "chat_history": [
    {
      "role": "user",
      "content": "Tell me about recent monetary policy news."
    },
    {
      "role": "assistant",
      "content": "In August 2026, the RBI MPC held rates steady..."
    }
  ],
  "model": "gemini_flash",
  "model_override": null,
  "enable_web_search": false
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | The user's natural language question (minimum 2 characters). |
| `user_id` | `string \| null` | Optional user or session identifier for query logging and tracking. |
| `chat_history` | `array<object>` | Prior conversational turns for coreference resolution and follow-up query condensation. |
| `model` / `model_override`| `string \| null` | Model provider selection (e.g. `gemini_flash`, `groq_llama`, `ollama_chat`). |
| `enable_web_search` | `boolean` | Flag enabling external internet search fallback when broadsheet archives do not have answers. |

---

### 5.2 Planner Output (`QueryPlan`)
The LangGraph planner parses the question into a structured JSON plan with an assigned archetype and tool routing arguments:

```json
{
  "thought_process": "User is asking for specific economic reporting across two newspapers for a specific date. Intent requires hybrid lexical and semantic search filtered to the RBI repo rate topic.",
  "archetype": "cross_newspaper_comparison",
  "primary_tool": "hybrid_search",
  "arguments": {
    "query": "RBI repo rate benchmark monetary policy committee",
    "newspaper_name": null,
    "issue_date": "2026-08-20",
    "category_filter": "Business & Markets",
    "page_filter": null,
    "top_k": 8
  }
}
```

---

### 5.3 Retrieval Tool Invocation & Argument Payloads

The planner routes the query to one of several specialized engines. Below are the exact argument JSON payloads for each tool:

#### 1. Hybrid Search (`hybrid_search`)
Combines dense vector search in Qdrant with sparse BM25 keyword matching in MySQL via Reciprocal Rank Fusion (RRF):

```json
{
  "query": "RBI benchmark repo rate inflation target",
  "newspaper_id": 1,
  "date_from": "2026-08-01",
  "date_to": "2026-08-31",
  "page_number": 1,
  "printed_page": "A1",
  "category_name": "Business & Markets",
  "top_k": 6
}
```

#### 2. SQL Analytics Engine (`sql_analytics`)
Executes parameterized aggregation queries across structured relational tables for macro trends, manifests, or distribution metrics:

**a) Issue Manifest / Page Summary (`analysis_type: "issue_summary"`):**
```json
{
  "analysis_type": "issue_summary",
  "newspaper_name": "The Daily Chronicle",
  "issue_date": "2026-08-20",
  "page_filter": "1",
  "exclude_page_filter": null,
  "category_filter": null
}
```

**b) Entity Trends Over Time (`analysis_type: "entity_trends"`):**
```json
{
  "analysis_type": "entity_trends",
  "term": "Shaktikanta Das",
  "group_by_period": "month"
}
```

**c) Relational Article Count (`analysis_type: "count_articles"`):**
```json
{
  "analysis_type": "count_articles",
  "newspaper_name": "Mint",
  "issue_date": "2026-08-20",
  "section": "Economy",
  "article_type": "news"
}
```

**d) Cross-Newspaper Coverage Matrix (`analysis_type: "coverage_comparison"`):**
```json
{
  "analysis_type": "coverage_comparison",
  "query": "RBI repo rate decision"
}
```

#### 3. Entity & Taxonomy Search (`entity_search`)
Queries articles connected through the canonical knowledge graph and taxonomy:

```json
{
  "entity_name": "Reserve Bank of India",
  "entity_type": "org",
  "topic_name": "Monetary Policy",
  "min_salience": 0.5,
  "top_k": 10
}
```

#### 4. Timeline Builder (`timeline_builder`)
Assembles a chronological progression of events across multiple print dates and tracks differing newspaper perspectives:

```json
{
  "query": "Adani Group Hindenburg allegations and Supreme Court hearings",
  "limit": 20
}
```

---

### 5.4 Corrective RAG (CRAG) Fallback Payloads (`evaluate_and_fallback`)
When initial retrieval yields zero high-confidence grounded articles or relevance falls below threshold, the CRAG node automatically triggers corrective fallback branches:

**CRAG Entity Fallback Payload:**
```json
{
  "tool_name": "crag_entity_fallback",
  "tool_input": {
    "entity_name": "Nvidia",
    "top_k": 5
  },
  "results_count": 3,
  "execution_time_ms": 142
}
```

**CRAG Live Web Fallback Payload (if enabled):**
```json
{
  "tool_name": "crag_web_fallback",
  "tool_input": {
    "query": "Nvidia Blackwell GPU shipments Q3 2026",
    "num_results": 4
  },
  "results_count": 4,
  "execution_time_ms": 480
}
```

---

## 6. Output JSON Schemas: Synthesis, Telemetry & Citations

### 6.1 Complete Query Response (`POST /api/query`)
When invoked as a standard synchronous endpoint, the engine returns this complete JSON response:

```json
{
  "query": "What did Mint and Business Standard report regarding the RBI repo rate on August 20, 2026?",
  "archetype": "cross_newspaper_comparison",
  "answer": "### ⚡ Executive Summary\nOn August 20, 2026, the RBI Monetary Policy Committee voted unanimously to keep the benchmark repo rate unchanged at 6.5%, citing persistent food inflation concerns.\n\n### 📌 Key Verified Facts & Highlights\n• The repo rate remains at 6.5% for the ninth consecutive meeting [Mint, 2026-08-20, Page 1, \"RBI Holds Repo Rate at 6.5%\"].\n• Retail CPI inflation was projected at 4.5% for FY27 [Business Standard, 2026-08-20, Page 3, \"MPC Retains Inflation Forecast\"].\n\n### 📰 Broadsheet Perspectives & Focus Areas\n• **Mint Focus**: Emphasized foreign institutional investor (FII) sentiment and bond yields.\n• **Business Standard Focus**: Highlighted industrial credit growth and private capex outlook.",
  "citations": [
    {
      "newspaper": "Mint",
      "date": "2026-08-20",
      "page": 1,
      "article_id": 9012,
      "headline": "RBI Holds Repo Rate at 6.5%",
      "bboxes": [
        {
          "page_number": 1,
          "ymin": 0.12,
          "xmin": 0.05,
          "ymax": 0.48,
          "xmax": 0.62
        }
      ]
    },
    {
      "newspaper": "Business Standard",
      "date": "2026-08-20",
      "page": 3,
      "article_id": 9045,
      "headline": "MPC Retains Inflation Forecast",
      "bboxes": [
        {
          "page_number": 3,
          "ymin": 0.25,
          "xmin": 0.40,
          "ymax": 0.70,
          "xmax": 0.85
        }
      ]
    }
  ],
  "plan": [
    {
      "tool_name": "hybrid_search",
      "arguments": {
        "query": "RBI repo rate benchmark decision",
        "top_k": 8
      },
      "purpose": "Retrieve primary news reports covering the MPC announcement."
    }
  ],
  "tool_executions": [
    {
      "tool_name": "hybrid_search",
      "tool_input": {
        "query": "RBI repo rate benchmark decision",
        "top_k": 8
      },
      "results_count": 6,
      "execution_time_ms": 320
    }
  ],
  "evidence_count": 6,
  "latency_ms": 1145,
  "cost_usd": 0.0028
}
```

---

### 6.2 Server-Sent Events (SSE) Streaming JSON Protocol (`POST /api/query/stream`)
For real-time UI interactions, the engine streams execution steps and tokens using SSE:

#### Event 1: `stage` (State Transition)
```text
event: stage
data: {"stage": "planning"}
```
*(Possible stages: `condensing_query`, `planning`, `tool_execution`, `web_search`, `synthesizing`, `completed`)*

#### Event 2: `plan` (Planned Tool Actions)
```text
event: plan
data: {"archetype": "factual_lookup", "plan": [{"tool_name": "hybrid_search", "arguments": {"query": "GDP growth Q1 2026", "top_k": 6}, "purpose": "Find GDP articles"}]}
```

#### Event 3: `tool_results` (Retrieval Execution Summary)
```text
event: tool_results
data: {"evidence_count": 4, "tools": [{"tool_name": "hybrid_search", "tool_input": {"query": "GDP growth Q1 2026"}, "results_count": 4, "execution_time_ms": 280}]}
```

#### Event 4: `think` (Streaming Chain-of-Thought Reasoning)
```text
event: think
data: {"delta": "Analyzing reported GDP figures across Mint and Business Standard..."}
```

#### Event 5: `token` (Streaming Synthesized Answer Tokens)
```text
event: token
data: {"delta": "India's Q1 GDP expanded by 6.8%"}
```

#### Event 6: `citations` (Structured Bounding-Box Grounding)
```text
event: citations
data: {"citations": [{"newspaper": "The Daily Chronicle", "date": "2026-08-20", "page": 1, "article_id": 9012, "headline": "Economy Grows 6.8%", "bboxes": [{"page_number": 1, "ymin": 0.1, "xmin": 0.2, "ymax": 0.5, "xmax": 0.6}]}]}
```

#### Event 7: `done` (Execution Telemetry)
```text
event: done
data: {"latency_ms": 940, "cost_usd": 0.0019, "evidence_count": 4}
```

