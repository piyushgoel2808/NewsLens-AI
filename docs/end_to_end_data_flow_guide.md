# NewsLens-AI End-to-End Data Flow & Data Structure Guide
*(Cross-Verified Against Real Production Database, Storage Cluster & Live Retrieval Engine)*

> **Document Version**: 2.0.0 (Production Verified)  
> **Verification Status**: Tested against live MySQL database (`3,231` articles, `502` pages, `5,565` chunks), Qdrant cluster (`1,024`-dim BGE-M3 vectors), and sentence-transformers Cross-Encoder reranker.  
> **Target Audience**: Core Engineers, AI Researchers, and System Architects.

---

## Table of Contents

1. [High-Level Architecture & Live Data Flow Sequence](#1-high-level-architecture--live-data-flow-sequence)
2. [Phase 1: Ingestion Pipeline (From Raw PDF to Multi-Tier Storage)](#2-phase-1-ingestion-pipeline-from-raw-pdf-to-multi-tier-storage)
   - [1.1 Real Broadsheet PDF Ingestion & High-Res Rendering](#11-real-broadsheet-pdf-ingestion--high-res-rendering)
   - [1.2 Docling 2D Layout, Vision OCR & CMap Corruption Detection](#12-docling-2d-layout-vision-ocr--cmap-corruption-detection)
   - [1.3 Article Boundary Assembly & Multi-Page Continuation Linking](#13-article-boundary-assembly--multi-page-continuation-linking)
   - [1.4 Live Relational Persistence (Exact Rows from MySQL Tables)](#14-live-relational-persistence-exact-rows-from-mysql-tables)
   - [1.5 Qwen-VL Visual Intelligence: Deep Thinking, Spatial Grounding & Infographic Reasoning](#15-qwen-vl-visual-intelligence-deep-thinking-spatial-grounding--infographic-reasoning)
   - [1.6 Chunking & Verified Qdrant Vector Point Payloads](#16-chunking--verified-qdrant-vector-point-payloads)
3. [Phase 2: Conversational Pre-Processing & Query Condensation](#3-phase-2-conversational-pre-processing--query-condensation)
   - [2.1 Chat History & Metadata Detection](#21-chat-history--metadata-detection)
   - [2.2 Active Context Extraction & Query-Aware Isolation](#22-active-context-extraction--query-aware-isolation)
   - [2.3 Coreference Resolution & Live Condensed Query Transformation](#23-coreference-resolution--live-condensed-query-transformation)
4. [Phase 3: Cognitive Query Planner & Structured Chain-of-Thought](#4-phase-3-cognitive-query-planner--structured-chain-of-thought)
   - [3.1 Parameter Extraction with Typo Tolerance](#31-parameter-extraction-with-typo-tolerance)
   - [3.2 The Live Structured `PlanResult` & `QueryPlan`](#32-the-live-structured-planresult--queryplan)
5. [Phase 4: Tool Execution Deep Dive (Live Inputs, SQL Queries & Real Outputs)](#5-phase-4-tool-execution-deep-dive-live-inputs-sql-queries--real-outputs)
   - [Tool 1: `hybrid_search` (Dense + Sparse + RRF + Cross-Encoder Reranking)](#tool-1-hybrid_search-dense--sparse--rrf--cross-encoder-reranking)
   - [Tool 2: `sql_analytics` (Relational Broadsheet Manifests & Coverage Differences)](#tool-2-sql_analytics-relational-broadsheet-manifests--coverage-differences)
   - [Tool 3: `entity_search` (Multi-Hop Entity Graph & Salience Scoring)](#tool-3-entity_search-multi-hop-entity-graph--salience-scoring)
   - [Tool 4: `timeline_builder` (Narrative Chronological Trajectory)](#tool-4-timeline_builder-narrative-chronological-trajectory)
   - [Tool 5: `coverage_analysis` (3-Tier Negative Coverage & Omission Audit)](#tool-5-coverage_analysis-3-tier-negative-coverage--omission-audit)
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

## 1. High-Level Architecture & Live Data Flow Sequence

The diagram below traces how real broadsheet issues (e.g. *The Goan*, Issue #93, 2026-08-01) flow through layout recognition, relational storage, and vector indexing, and how a user query traverses condensation, planning, multi-tool execution, CRAG evaluation, and streaming answer synthesis.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    LIVE INGESTION PIPELINE                                             │
│                                                                                                        │
│   Raw PDF Broadsheet     PyMuPDF Render     Docling Layout Model     Segmenter & Assembler             │
│  ┌──────────────────┐    ┌─────────────┐    ┌─────────────────────┐  ┌───────────────────────┐         │
│  │ The Goan Issue 93│──> │ 300 DPI PNG │ ─> │ 2D Bounding Boxes   │─>│ Coalesce Headlines,   │         │
│  │ 14 Pages (Aug 1) │    │ 8188x11400px│    │ Labels & OCR Text   │  │ Subheadlines, Byline  │         │
│  └──────────────────┘    └─────────────┘    └─────────────────────┘  └──────────┬────────────┘         │
│                                                                                 │                      │
│                                            ┌────────────────────────────────────┴────────────────┐     │
│                                            ▼                                                     ▼     │
│                               ┌─────────────────────────┐                            ┌────────────────┐│
│                               │ MySQL Relational DB     │                            │ Qdrant Vectors ││
│                               │ (Articles 40401..40574) │                            │ Point UUIDs    ││
│                               └─────────────────────────┘                            └────────────────┘│
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                                   │ User Query
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   LIVE AGENTIC QUERY PIPELINE                                          │
│                                                                                                        │
│  User Query + History        Query Condenser (LLM)      Query Planner (Structured CoT)                 │
│  ┌────────────────────┐      ┌─────────────────────┐    ┌─────────────────────────────┐                │
│  │ "List all its news │ ───> │ Resolves pronouns & │ ──>│ Determines Archetype &       │                │
│  │ on page 1"         │      │ active publication  │    │ schedules 1 to 4 tools      │                │
│  └────────────────────┘      └─────────────────────┘    └──────────────┬──────────────┘                │
│                                                                        │                               │
│                   ┌────────────────────────────────────────────────────┴───────────────┐               │
│                   ▼                                                                    ▼               │
│       ┌───────────────────────┐                                            ┌───────────────────────┐   │
│       │ Tool: hybrid_search   │                                            │ Tool: sql_analytics   │   │
│       │ Qdrant + MySQL RRF    │                                            │ (Coverage Difference) │   │
│       │ Cross-Encoder Score   │                                            │ 142 Exclusives Found  │   │
│       └───────────┬───────────┘                                            └───────────┬───────────┘   │
│                   │                                                                    │               │
│                   └─────────────────────────────────┬──────────────────────────────────┘               │
│                                                     ▼                                                  │
│                                     ┌───────────────────────────────┐                                  │
│                                     │ Corrective RAG (CRAG) Gate    │                                  │
│                                     │ Macro Manifest Bypass (Score 1)│                                 │
│                                     │ Prune ungrounded distractors  │                                  │
│                                     └───────────────┬───────────────┘                                  │
│                                                     ▼                                                  │
│                                     ┌───────────────────────────────┐                                  │
│                                     │ Synthesizer Prompt Budgeting  │                                  │
│                                     │ Top 12 Items / 4000 Char Cap  │                                  │
│                                     └───────────────┬───────────────┘                                  │
│                                                     ▼                                                  │
│                                     ┌───────────────────────────────┐                                  │
│                                     │ LLM Streaming Synthesis       │                                  │
│                                     │ Thought + Structured Brief    │                                  │
│                                     └───────────────────────────────┘                                  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: Ingestion Pipeline (From Raw PDF to Multi-Tier Storage)

We follow a verified, real broadsheet edition present in the database:
- **Publication**: `The Goan` (Newspaper ID: `1`)
- **Issue ID**: `93`
- **Issue Date**: `2026-08-01`
- **Total Pages**: `14`
- **Total Ingested Articles**: `174` (Article IDs: `40401` to `40574`)
- **Comparison Issue**: `The Morning Standard` (Newspaper ID: `98`, Issue ID: `98`, Date: `2026-08-01`, `144` articles).

### 1.1 Real Broadsheet PDF Ingestion & High-Res Rendering

1. The PDF is submitted to `POST /api/ingest/upload` or via CLI `python -m app.ingestion.tasks`.
2. `PyMuPDF` (`fitz`) rasterizes each broadsheet page at 300 DPI:
   - Width: `8188 px`, Height: `11400 px`.
   - PNG images stored in `storage/renders/93/page_1.png` to `storage/renders/93/page_14.png`.
3. Populates initial database rows in `issues` and `pages`.

#### Exact SQL Rows Created
```sql
INSERT INTO newspapers (id, name, code, publisher, country, language)
VALUES (1, 'The Goan', 'the_goan', 'Fomento Media', 'India', 'English');

INSERT INTO issues (id, newspaper_id, issue_date, edition, total_pages, ingestion_status)
VALUES (93, 1, '2026-08-01', 'Panaji', 14, 'completed');

INSERT INTO pages (id, issue_id, page_number, printed_page_number, width, height, image_path)
VALUES 
  (2230, 93, 1, '9', 8188, 11400, 'storage/renders/93/page_1.png'),
  (2238, 93, 9, '10', 8188, 11400, 'storage/renders/93/page_9.png');
```

---

### 1.2 Docling 2D Layout, Vision OCR & CMap Corruption Detection

[`backend/app/ingestion/docling_parser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/docling_parser.py) runs the page image and PDF text layer through DocLayNet:

1. **Bounding Box Normalization**: Converts PDF coordinates into standard pixel bounding boxes `[x0, y0, x1, y1]`.
2. **Corrupted Font CMap Guard**: Notice in our live database, the headline contains `tra  c` because the embedded font lacked a `ToUnicode` mapping for the `ffi` ligature. The system computes the replacement character ratio:
   $$\text{Replacement Ratio} = \frac{\text{Count}(\ufffd)}{\text{Total Non-Space Characters}}$$
   When ratio $\ge 3\%$, it triggers pure Image OCR via `GoogleCloudVisionOCR` or layout fallback.

#### Intermediate JSON Emitted by Docling (`DoclingParsedItem`)
```json
[
  {
    "label": "title",
    "text": "Beware! AI-enabled tra  c challans go live from today",
    "bbox": [3994.61, 1530.89, 6215.43, 2082.70],
    "page_number": 1,
    "level": 1
  },
  {
    "label": "section_header",
    "text": "Life",
    "bbox": [180.0, 140.0, 480.0, 195.0],
    "page_number": 1,
    "level": 1
  },
  {
    "label": "text",
    "text": "MBBS grads to be roped in to tackle doctor shortage",
    "bbox": [3994.61, 2100.0, 6215.43, 2250.0],
    "page_number": 1,
    "level": 2
  },
  {
    "label": "text",
    "text": "Panaji: Under the first phase, AI-powered cameras installed at 26 locations automatically detect offences including signal jumping, helmetless riding, and triple riding across arterial corridors...",
    "bbox": [3994.61, 2300.0, 6215.43, 3500.0],
    "page_number": 1,
    "level": 2
  }
]
```

---

### 1.3 Article Boundary Assembly & Multi-Page Continuation Linking

Article `40403` starts on Page 1 (Printed Folio `9`) and continues on Page 9 (Printed Folio `10`). [`CrossPageAssembler`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/tasks.py#L422) links both blocks into a unified article:

```json
{
  "article_temp_id": "page_1_art_2",
  "headline": "Beware! AI-enabled tra  c challans go live from today",
  "subheadline": "MBBS grads to be roped in to tackle doctor shortage",
  "byline_author": null,
  "body_text": "Beware! AI-enabled tra  c challans go live from today\n\nMBBS grads to be roped in to tackle doctor shortage\n\nUnder the first phase, AI-powered cameras installed at 26 locations automatically detect offences...",
  "word_count": 462,
  "pages_mapping": [
    {
      "page_number": 1,
      "printed_page_number": "9",
      "bbox_list": [[3994.61, 1530.89, 6215.43, 2082.70]],
      "block_order": 0
    },
    {
      "page_number": 9,
      "printed_page_number": "10",
      "bbox_list": [
        [18.94, 4165.42, 900.98, 4400.05],
        [18.94, 4517.63, 1116.28, 5226.30]
      ],
      "block_order": 1
    }
  ]
}
```

---

### 1.4 Live Relational Persistence (Exact Rows from MySQL Tables)

The database transaction persists the parsed records across relational tables:

#### 1. Table: `articles` (Row ID: `40403`)
```sql
INSERT INTO articles (
  id, issue_id, primary_page_id, category_id, headline, subheadline, 
  byline_author, section, printed_section, article_type, prominence_score, 
  word_count, full_text
) VALUES (
  40403, 93, 2230, 10,
  'Beware! AI-enabled tra \ufffd c challans go live from today',
  'MBBS grads to be roped in to tackle doctor shortage',
  NULL, 'Front Page', 'Life', 'news', 0.95, 462,
  'Beware! AI-enabled tra \ufffd c challans go live from today\n\nMBBS grads to be roped in to tackle doctor shortage\n\nUnder the first phase, AI-powered cameras installed at 26 locations automatically detect offences...'
);
```

#### 2. Table: `article_pages` (Multi-Page Linkage)
```sql
-- Page 1 (Front Page)
INSERT INTO article_pages (article_id, page_id, page_number, printed_page_number, bbox_json, block_order)
VALUES (
  40403, 2230, 1, '9',
  '{"bboxes": [[3994.61, 1530.89, 6215.43, 2082.70]]}',
  0
);

-- Page 9 (Continuation Jump)
INSERT INTO article_pages (article_id, page_id, page_number, printed_page_number, bbox_json, block_order)
VALUES (
  40403, 2238, 9, '10',
  '{"bboxes": [[18.94, 4165.42, 900.98, 4400.05], [18.94, 4517.63, 1116.28, 5226.30]]}',
  1
);
```

#### 3. Tables: `entities` and `article_entities`
```sql
-- Verified extracted entities for Article 40403
INSERT INTO article_entities (article_id, entity_id, mention_count, salience_score) VALUES
  (40403, (SELECT id FROM entities WHERE name='Transport Department'), 1, 0.16),
  (40403, (SELECT id FROM entities WHERE name='Transport Minister Mauvin Godinho'), 1, 0.16),
  (40403, (SELECT id FROM entities WHERE name='National Medical Commission'), 1, 0.16),
  (40403, (SELECT id FROM entities WHERE name='Smart City'), 1, 0.16);
```

#### 4. Table: `article_chunks` (Sub-Document Segmentation)
```sql
INSERT INTO article_chunks (id, article_id, chunk_index, chunk_type, text, token_count, embedding_vector_id)
VALUES (
  13412, 40403, 0, 'text',
  '[Newspaper: The Goan | Date: 2026-08-01 | Section: Front Page | Headline: Beware! AI-enabled tra \ufffd c challans go live from today | Page(s): 9, 10 (PDF p.1, 9)] Beware! AI-enabled tra \ufffd c challans go live from today...',
  301, '3b3b67a0-afac-46ba-942d-ce3c67af41c8'
);
```

---

### 1.5 Qwen-VL Visual Intelligence: Deep Thinking, Spatial Grounding & Infographic Reasoning

Newspapers are rich visual artifacts: broadsheets embed critical investigative findings inside complex multi-column layouts, financial charts, sector breakdown infographics, and editorial photojournalism. NewsLens-AI does not treat images as passive blobs; it integrates **Qwen-VL** (`ollama_qwen3vl: qwen3-vl:latest` / `qwen2.5vl:7b` via [`backend/app/ingestion/visual_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/visual_extractor.py) and [`backend/app/ingestion/media_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/media_extractor.py)) to perform **multimodal thinking, spatial coordinate grounding, numerical transcription, and cross-modal validation**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           QWEN-VL VISUAL INTELLIGENCE ARCHITECTURE                                │
│                                                                                                  │
│   Page Image Crop         Qwen-VL Visual Reasoning            3-Stage Processing Pipeline         │
│  ┌────────────────┐      ┌─────────────────────────┐         ┌─────────────────────────────────┐ │
│  │ Cropped Visual │ ───> │ Native `<think>` stream │ ──────> │ Stage 1: Triage Gate            │ │
│  │ Asset / Page   │      │ Spatial [x0,y0,x1,y1]   │         │ Stage 2: Structured Transcription│ │
│  └────────────────┘      └─────────────────────────┘         │ Stage 3: OCR Cross-Validation   │ │
│                                                              └────────────────┬────────────────┘ │
│                                                                               │                  │
│                                                ┌──────────────────────────────┴──────────────┐   │
│                                                ▼                                             ▼   │
│                                   ┌───────────────────────────┐                 ┌──────────────┐ │
│                                   │ MySQL `photos` Table      │                 │ Visual Chunk │ │
│                                   │ (vlm_description, type)   │                 │ (Markdown)   │ │
│                                   └───────────────────────────┘                 └──────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### A. Parsing Qwen-VL's Native `<think>` Reasoning & Spatial Grounding Stream
When presented with a full broadsheet page or composite photo canvas, Qwen-VL performs step-by-step chain-of-thought spatial reasoning. Natively, the model outputs bounding box coordinates inside its `<think>...</think>` tokens scaled to a normalized $0..1000$ grid:

```text
<think>
Inspecting newspaper page canvas (8188x11400 px)...
Scanning layout regions from top to bottom:
- Top banner: Masthead logo "The Goan" at [10, 15, 990, 85] (Skip, decorative branding)
- Left column: Editorial portrait of Transport Minister Mauvin Godinho: [45, 140, 260, 310]
- Center-right: High-contrast data chart on power consumption tariffs: [480, 135, 980, 420]
- Bottom center: Traffic junction CCTV installation photo: [488, 510, 760, 685]
Each region is bounded and labeled for crop extraction.
</think>
```

In [`media_extractor.py:extract_grounded_boxes_from_thinking()`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/media_extractor.py#L26-L89), NewsLens-AI intercepts this internal thinking stream with a compiled regex:

```python
pattern = re.compile(
    r"(?:^|\n)\s*[-*\d.]*\s*([A-Za-z0-9\s()/,–—?]+?):\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
)
```

It maps normalized coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in [0, 1000]$ into absolute pixel bounding boxes on the 300 DPI canvas:
$$x_0 = \frac{x_{\min}}{1000.0} \times \text{width\_px}, \quad y_0 = \frac{y_{\min}}{1000.0} \times \text{height\_px}$$
$$x_1 = \frac{x_{\max}}{1000.0} \times \text{width\_px}, \quad y_1 = \frac{y_{\max}}{1000.0} \times \text{height\_px}$$

IoU (Intersection-over-Union) suppression ($\text{IoU} \ge 0.50$) deduplicates overlapping candidate detections before cropping.

---

#### B. The 3-Stage Visual Intelligence Pipeline (`VisualDataExtractor`)

Every detected image asset is dispatched through a 3-stage validation pipeline:

##### Stage 1: Fast Visual Triage Gate
* **Heuristic Filter**: Discards extreme aspect ratio strips ($\text{aspect} > 12.0$, e.g. horizontal column rules or border dividers) and solid-color blank tiles (variance $< 5$ on small images).
* **VLM Triage Classifier**: Categorizes the asset into:
  - `data_chart`: Bar chart, line graph, pie chart, stock trend, candlestick.
  - `table`: Tabular grid, balance sheet, price list, financial statements.
  - `infographic`: Explainer diagram, process flow, circular/donut map with statistics.
  - `photo`: Editorial news photograph (people, portraits, events, outdoor scenes).
  - `logo`: Masthead icon, brand insignia.
  - `decorative`: Cartoon, spacer, ornament.
* If `contains_data == True`, the asset advances to Stage 2.

##### Stage 2: Structured VLM Extraction (Charts & Infographics)
Qwen-VL is prompted with `STRUCTURED_EXTRACTION_PROMPT` to transcribe visual graphics into structured, machine-readable text:
1. **Executive Summary**: A concise 2-sentence explanation of what the chart demonstrates and its primary finding.
2. **Markdown Table**: Transcribes all categories, sectors, bars, time series, or percentages into a clean GitHub-flavored Markdown table.
3. **Key Metrics**: 3 to 6 bullet points highlighting salient figures.

*Example Real Infographic Extraction:*
```json
{
  "summary": "State power distribution matrix illustrating electricity consumption tariffs across domestic and industrial tiers in North Goa. Demonstrates an average 18% tariff hike across slabs exceeding 300 units.",
  "markdown_table": "| Consumption Slab (Units) | Existing Rate (₹/kWh) | Revised Rate (₹/kWh) | Increase (%) |\n|---|---|---|---|\n| 0 – 100 | 1.75 | 1.90 | +8.5% |\n| 101 – 300 | 2.60 | 3.10 | +19.2% |\n| 301 – 500 | 3.90 | 4.80 | +23.1% |\n| 500+ | 5.40 | 6.50 | +20.4% |",
  "key_metrics": [
    "Average domestic slab tariff increase: 18.2%",
    "Peak slab rate (>500 units): ₹6.50 per kWh",
    "Effective date of implementation: August 1, 2026"
  ],
  "confidence": 0.95,
  "visual_type": "data_chart"
}
```

##### Stage 3: Numerical Cross-Validation Against OCR Ground Truth
Large Vision-Language Models can occasionally hallucinate decimal points or invert adjacent numbers. To prevent this, NewsLens-AI executes **Numerical Cross-Validation** ([`visual_extractor.py:cross_validate_with_ocr()`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/visual_extractor.py#L573-L611)):
1. Gathers all numeric tokens directly recognized by the deterministic OCR engine from the identical image crop:
   $$\mathcal{N}_{\text{ocr}} = \{ \text{numbers, percentages, currency values in OCR text} \}$$
2. Extracts all numeric tokens from Qwen-VL's transcribed Markdown table:
   $$\mathcal{N}_{\text{vlm}} = \{ \text{numbers, percentages, currency values in Markdown table} \}$$
3. Computes the Overlap Match Ratio:
   $$\text{Match Ratio} = \frac{|\mathcal{N}_{\text{vlm}} \cap \mathcal{N}_{\text{ocr}}|}{|\mathcal{N}_{\text{vlm}}|}$$
4. Re-calibrates extraction confidence:
   $$\text{Confidence}_{\text{adjusted}} = 0.4 \times \text{Confidence}_{\text{vlm}} + 0.6 \times \text{Match Ratio}$$

If $\text{Match Ratio} < 0.40$, the system automatically engages the **Spatial OCR Matrix Engine** fallback, reconstructing tabular cells purely from deterministic token bounding box geometry.

---

#### C. Editorial Photo Scene Analysis & Reasoning
For editorial news photographs, Qwen-VL evaluates the scene context using `PHOTO_SCENE_ANALYSIS_PROMPT`:
* Analyzes subjects, actions, uniforms, vehicle models, street signs, and emotional tone.
* Ingests the published newspaper caption to contextualize unnamed subjects.
* Outputs the narrative analysis to MySQL table `photos`:

```sql
UPDATE photos SET
  visual_type = 'photo',
  vlm_description = 'Editorial news photograph showing Goa traffic police officers inspecting an AI-enabled automated camera fixture mounted on an overhead arterial gantry. Visible equipment includes dual PTZ cameras and high-speed infrared illuminators overlooking a multi-lane roadway.',
  caption = 'Traffic police inspect the newly commissioned AI camera gantry at Panaji on Friday.'
WHERE id = 884;
```

---

#### D. Visual Chunks in Vector Search & Synthesizer Citations
Visual intelligence is not isolated in cold storage; it is directly indexed for semantic retrieval:
1. **Visual Article Chunk**: An `ArticleChunk` row is inserted into MySQL and Qdrant with:
   - `chunk_type = "visual"`
   - `has_visual_data = True`
   - `visual_type = "data_chart" | "infographic" | "table" | "photo"`
   - `text`: Includes the article context header + VLM Executive Summary + Transcribed GitHub Markdown Table + Key Visual Metrics.
2. **Retrieval**: When a user queries *"Show me the power tariff hike chart"* or *"Was there a photo of the traffic cameras?"*, `hybrid_search` retrieves this visual chunk.
3. **Synthesizer Citation**: In the final generated answer, visual evidence is cited with dedicated badges:
   ```markdown
   * **Power Tariff Structure**: The domestic electricity tariffs were revised upwards by an average of 18.2% across residential slabs [📊 Chart: *The Goan*, 2026-08-01, Page 1, "High power bills haunt consumers"].
   
   | Consumption Slab (Units) | Existing Rate (₹/kWh) | Revised Rate (₹/kWh) | Increase (%) |
   |---|---|---|---|
   | 0 – 100 | 1.75 | 1.90 | +8.5% |
   | 101 – 300 | 2.60 | 3.10 | +19.2% |
   | 301 – 500 | 3.90 | 4.80 | +23.1% |
   | 500+ | 5.40 | 6.50 | +20.4% |
   ```

---

### 1.6 Chunking & Verified Qdrant Vector Point Payloads

The chunk text is vectorized with `BAAI/bge-m3` ($1024$ dimensions) and upserted into Qdrant collection `newslens_articles`:

#### Exact Qdrant Point Retrieved from Live Cluster
```json
{
  "id": "3b3b67a0-afac-46ba-942d-ce3c67af41c8",
  "vector": [0.0142, -0.0219, 0.0811, 0.0035, "... 1024 float dimensions ..."],
  "payload": {
    "article_id": 40403,
    "issue_id": 93,
    "newspaper_name": "The Goan",
    "issue_date": "2026-08-01",
    "headline": "Beware! AI-enabled tra  c challans go live from today",
    "section": "Front Page",
    "article_type": "news",
    "prominence_score": 0.95,
    "has_photo": false,
    "has_table": false,
    "has_visual_data": false,
    "visual_type": null,
    "chunk_type": "text",
    "chunk_index": 0,
    "page_numbers": [1, 9],
    "printed_pages": ["9", "10"],
    "entities": [
      "Transport Department",
      "Transport Minister Mauvin Godinho",
      "National Medical Commission",
      "Smart City"
    ],
    "topics": ["Defense", "Technology", "Politics"],
    "chunk_text": "[Newspaper: The Goan | Date: 2026-08-01 | Section: Front Page | Headline: Beware! AI-enabled tra  c challans go live from today | Page(s): 9, 10 (PDF p.1, 9)]\n\nBeware! AI-enabled tra  c challans go live from today\n\nMBBS grads to be roped in to tackle doctor shortage...",
    "raw_text": "Beware! AI-enabled tra  c challans go live from today..."
  }
}
```

---

## 3. Phase 2: Conversational Pre-Processing & Query Condensation

### 2.1 Chat History & Metadata Detection

If the user asks:
`"Which newspaper was this from?"` or `"What was the date?"`
[`is_in_context_meta_query()`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/condenser.py#L44) evaluates to `True`. Retrieval is completely bypassed and answered from conversation history.

### 2.2 Active Context Extraction & Query-Aware Isolation

When evaluating Turn 2 follow-up: `"list all those 11 articles"`, [`extract_active_issue_from_history()`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/condenser.py#L125) reads Turn 1 and extracts:
```python
{
    "newspaper_name": "The Goan",
    "comparison_newspaper": "The Morning Standard",
    "issue_date": "2026-08-01",
    "is_differential": True
}
```
If the user switches to a different publication or date, the guardrail purges stale active context to prevent cross-turn contamination.

### 2.3 Coreference Resolution & Live Condensed Query Transformation

```text
User Input: "list all those 11 articles"
                   │
                   ▼ (condenser.py)
Condensed Query: "list all those 11 articles in The Goan but not in The Morning Standard dated 2026-08-01"
```

---

## 4. Phase 3: Cognitive Query Planner & Structured Chain-of-Thought

### 3.1 Parameter Extraction with Typo Tolerance

In [`planner.py:extract_parameters_from_query()`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/planner.py#L320):
- User typed: `"he Morning Standard"` $\implies$ Typo-tolerant regex `(?:(?:the|he)\s+)?morning\s+standard` correctly resolves to `The Morning Standard`.
- `"The Goan"` extracted as primary publication.
- `"but not in"` detected $\implies$ `is_differential = True`.

### 3.2 The Live Structured `PlanResult` & `QueryPlan`

The real planner executes and emits:

```python
PlanResult(
    archetype="cross_newspaper_comparison",
    reasoning="Query requests exclusive articles in The Goan absent from The Morning Standard.",
    tool_calls=[
        PlannedToolCall(
            tool_name="sql_analytics",
            purpose="Compute verified article difference: stories in The Goan absent from The Morning Standard on 2026-08-01",
            arguments={
                "analysis_type": "coverage_difference",
                "newspaper_name": "The Goan",
                "comparison_newspaper": "The Morning Standard",
                "issue_date": "2026-08-01",
                "query": "List the news that are in the GOAN dated 1/8/2026 but not in he Morning Standard dated 1/8/2026"
            }
        ),
        PlannedToolCall(
            tool_name="hybrid_search",
            purpose="Retrieve key articles and snippets from The Goan",
            arguments={
                "query": "List the news that are in the GOAN dated 1/8/2026 but not in he Morning Standard dated 1/8/2026",
                "newspaper_name": "The Goan",
                "date_from": "2026-08-01",
                "date_to": "2026-08-01",
                "top_k": 10
            }
        )
    ]
)
```

---

## 5. Phase 4: Tool Execution Deep Dive (Live Inputs, SQL Queries & Real Outputs)

Below are the **verified runtime execution outputs** from the live NewsLens-AI test runs:

---

### Tool 1: `hybrid_search` (Dense + Sparse + RRF + Cross-Encoder Reranking)

#### Real Execution Demo Call
```python
search_engine = HybridSearchEngine(session_factory=sf)
results = await search_engine.search(
    query="AI-enabled challans go live",
    top_k=2
)
```

#### Real Returned Results
```json
[
  {
    "article_id": 40403,
    "issue_id": 93,
    "headline": "Beware! AI-enabled tra  c challans go live from today",
    "newspaper_name": "The Goan",
    "issue_date": "2026-08-01",
    "pages": [1, 9],
    "printed_pages": ["9", "10"],
    "rrf_score": 0.015889,
    "rerank_score": 7.9975,
    "snippet": "[Exact Chunk Match]: [Newspaper: The Goan | Date: 2026-08-01 | Section: Front Page | Headline: Beware! AI-enabled tra  c challans go live from today | Page(s): 9, 10 (PDF p.1, 9)]\n\nBeware! AI-enabled tra  c challans go live from today...",
    "source_tool": "hybrid_search"
  },
  {
    "article_id": 40922,
    "issue_id": 96,
    "headline": "AI tra  c surveillance sees gradual rise in challans, violations below trial levels",
    "newspaper_name": "The Goan",
    "issue_date": "2026-08-04",
    "pages": [1, 2],
    "printed_pages": ["1", "2"],
    "rrf_score": 0.015877,
    "rerank_score": -0.7377,
    "snippet": "[Exact Chunk Match]: [Newspaper: The Goan | Date: 2026-08-04 | Section: Front Page | Headline: AI tra  c surveillance sees gradual rise in challans...",
    "source_tool": "hybrid_search"
  }
]
```
> **Notice**: The Cross-Encoder neural reranker awarded an interaction score of **`+7.9975`** to the exact August 1 match, while the August 4 follow-up received **`-0.7377`**.

#### Visual Chunk Retrieval (Charts, Infographics & Tabular Visuals)
When a query targets quantitative trends, charts, or diagrams (e.g. *"Show me the power tariff hike chart"*), `hybrid_search` retrieves chunks where `has_visual_data == True` and `chunk_type == "visual"`, carrying the Markdown table transcribed by Qwen-VL:

```json
{
  "article_id": 40408,
  "issue_id": 93,
  "headline": "High power bills haunt consumers",
  "newspaper_name": "The Goan",
  "issue_date": "2026-08-01",
  "pages": [1],
  "printed_pages": ["9"],
  "has_visual_data": true,
  "visual_type": "data_chart",
  "chunk_type": "visual",
  "snippet": "[Visual Data Chart - Transcribed by Qwen-VL]:\nState power distribution matrix illustrating electricity consumption tariffs across domestic and industrial tiers in North Goa.\n\n| Consumption Slab (Units) | Existing Rate (₹/kWh) | Revised Rate (₹/kWh) | Increase (%) |\n|---|---|---|---|\n| 0 – 100 | 1.75 | 1.90 | +8.5% |\n| 101 – 300 | 2.60 | 3.10 | +19.2% |\n| 301 – 500 | 3.90 | 4.80 | +23.1% |\n| 500+ | 5.40 | 6.50 | +20.4% |\n\nKey Visual Elements:\n• Average domestic slab tariff increase: 18.2%\n• Peak slab rate (>500 units): ₹6.50 per kWh",
  "source_tool": "hybrid_search"
}
```

---

### Tool 2: `sql_analytics` (Relational Broadsheet Manifests & Coverage Differences)

#### Operation A: Deterministic Coverage Difference (`coverage_difference`)
```python
diff = await sql_tool.get_newspaper_coverage_difference(
    source_newspaper="The Goan",
    comparison_newspaper="The Morning Standard",
    issue_date="2026-08-01"
)
```

#### Real Returned Result from MySQL
```json
{
  "source_newspaper": "The Goan",
  "comparison_newspaper": "The Morning Standard",
  "issue_date": "2026-08-01",
  "source_total_articles": 174,
  "comparison_total_articles": 144,
  "exclusive_count": 142,
  "exclusive_articles": [
    {
      "id": 40403,
      "headline": "Beware! AI-enabled tra  c challans go live from today",
      "page_number": 1,
      "printed_page": "9",
      "section": "Front Page",
      "category": "Lifestyle",
      "snippet": ""
    },
    {
      "id": 40410,
      "headline": "Govt imposes sweeping curbs at tourist hotspots",
      "page_number": 1,
      "printed_page": "9",
      "section": "Front Page",
      "category": "Lifestyle",
      "snippet": ""
    },
    {
      "id": 40408,
      "headline": "High power bills haunt consumers",
      "page_number": 1,
      "printed_page": "9",
      "section": "Front Page",
      "category": "Business & Markets",
      "snippet": ""
    }
  ]
}
```

---

### Tool 3: `entity_search` (Multi-Hop Entity Graph & Salience Scoring)

#### Real Execution Demo Call
```python
engine = EntitySearchEngine(session_factory=sf)
results = await engine.search_by_entity(entity_name="Transport Department", top_k=2)
```

#### Real Returned Results
```json
[
  {
    "article_id": 36586,
    "headline": "Pure EVs over hybrids: Divergence of views in Govt led to change in draft plan",
    "entity_name": "Transport Department",
    "entity_type": "person",
    "salience_score": 0.27,
    "newspaper_name": "The Indian Express",
    "issue_date": "2026-07-01",
    "source_tool": "entity_search"
  },
  {
    "article_id": 37792,
    "headline": "[Shorts] US Note : Top six countries from which India imports aeroplanes and other aircraft of an u",
    "entity_name": "Kerala Transport Department",
    "entity_type": "misc",
    "salience_score": 0.27,
    "newspaper_name": "Business Standard",
    "issue_date": "2026-07-04",
    "source_tool": "entity_search"
  }
]
```

---

### Tool 4: `timeline_builder` (Narrative Chronological Trajectory)

#### Real Execution Demo Call
```python
builder = TimelineBuilder(session_factory=sf)
res = await builder.build_timeline(query="challans", limit=5)
```

#### Real Returned Result across Broadsheets
```json
{
  "query": "challans",
  "total_dates": 4,
  "total_articles": 5,
  "date_groups": [
    {
      "date": "2026-07-01",
      "newspaper_name": "The Indian Express",
      "articles_count": 1,
      "milestones": [
        {
          "article_id": 36591,
          "headline": "Delhi sees crackdown-wrong side driving challans up by 98%",
          "pages": [8]
        }
      ]
    },
    {
      "date": "2026-08-01",
      "newspaper_name": "The Goan",
      "articles_count": 2,
      "milestones": [
        {
          "article_id": 40403,
          "headline": "Beware! AI-enabled tra  c challans go live from today",
          "pages": [1, 9]
        },
        {
          "article_id": 40413,
          "headline": "PANAJI",
          "pages": [1, 3]
        }
      ]
    },
    {
      "date": "2026-08-02",
      "newspaper_name": "The Goan",
      "articles_count": 1,
      "milestones": [
        {
          "article_id": 40587,
          "headline": "Thinking of outsmarting AI cams? Govt says think again",
          "pages": [1, 6]
        }
      ]
    }
  ]
}
```

---

### Tool 5: `coverage_analysis` (3-Tier Negative Coverage & Omission Audit)

#### Real Execution Demo Call
```python
analyzer = CoverageAnalyzer(session_factory=sf, hybrid_search_engine=search_engine)
rep = await analyzer.analyze_newspaper_coverage(
    newspaper=morning_standard,
    query_or_event="AI traffic challans",
    target_date="2026-08-01",
    date_window_days=0
)
```

#### Real Returned Multi-Newspaper Reconciliation
```text
=== The Goan ===
Status: COVERED | Confidence: 1.0 | Score: +3.1136
Headlines: ['Beware! AI-enabled tra  c challans go live from today', 'Govt bond prices fall after crude oil rises', 'PANAJI']
Snippet: "Beware! AI-enabled tra  c challans go live from today..."

=== The Morning Standard ===
Status: UNCERTAIN / NOT_FOUND | Confidence: -11.4585 | Score: -11.4585
Headlines: ['Barapullah corridor likely to open in Aug']
Snippet: "Barapullah corridor likely to open in Aug..."
```
> **Key Insight**: *The Morning Standard* had a Cross-Encoder score of **`-11.4585`** (negative infinity relevance), proving that it completely omitted the Goa traffic enforcement story and only reported on Delhi urban corridors.

---

### Tool 6: `web_search` (Live Web Verification Fallback)

Used when live search is toggled or when broadsheet archives lack coverage:

```json
[
  {
    "title": "Panaji Smart City AI traffic surveillance launched",
    "url": "https://www.heraldgoa.in/news/goa/traffic-ai-cams/219401",
    "snippet": "14 camera corridors in Panaji are now active issuing contactless e-challans.",
    "newspaper_name": "Live Web",
    "issue_date": "2026-08-01",
    "source_tool": "web_search",
    "is_web": true
  }
]
```

---

## 6. Phase 5: Corrective RAG (CRAG) Relevance Gate & Fallbacks

[`backend/app/agent/graph.py:_evaluate_evidence_node()`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/graph.py#L770-L860) evaluates evidence quality before the synthesizer is prompted:

### 5.1 Stemmed Query Matching & Relevance Scoring
- Query tokens are stemmed: e.g. `"traffic"` $\to$ `'traff'`, `"challans"` $\to$ `'challan'`.
- Headline matches receive $+2.0$, snippet/body matches receive $+1.0$.
- Any document with score $0.0$ is pruned from the generation prompt.

### 5.2 Macro Manifest Protection
For relational manifests (`source_tool == 'sql_analytics'` or `archetype == 'cross_newspaper_comparison'`), the item is granted an unconditional relevance score of **`1.0`**, ensuring full manifests (such as the 142 exclusive articles) are never accidentally stripped.

### 5.3 Corrective Fallback Activation
If grounded evidence count is 0:
1. Triggers `entity_search(top_k=5)` for capitalized named entities.
2. If still empty and `enable_web_search == True`, triggers `web_search(num_results=4)`.

---

## 7. Phase 6: Answer Synthesis, Prompt Budgeting & SSE Streaming

### 6.1 Evidence Context Budgeting & Publication Scoping

1. **Top 12 Item Cap**: Slices `evidence_items[:12]` to guarantee the synthesizer prompt stays within $\le 3,500$ tokens.
2. **Selective Length Allocations**:
   - Manifests, Exclusion Lists, and Coverage Matrices: **up to 4,000 characters**.
   - Standard Article Excerpts: **up to 1,200 characters**.
3. **Publication Scoping Guardrail**:
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
- Strict Citation Rule: [{Newspaper}, {YYYY-MM-DD}, Page {P}, "{Headline}"]
- Anti-Hallucination Hard Stop if 0 Evidence.

[USER PROMPT]
User Query: "List the news that are in the GOAN dated 1/8/2026 but not in he Morning Standard dated 1/8/2026"

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
Total Exclusives Found: 142

1. Page 1 (PDF Page 1) [Front Page]: "Beware! AI-enabled tra  c challans go live from today"
2. Page 1 (PDF Page 1) [Front Page]: "Govt imposes sweeping curbs at tourist hotspots"
3. Page 1 (PDF Page 1) [Front Page]: "High power bills haunt consumers"
...
```

---

### 6.3 LLM Generation, `<think>` Tag Separation & Inline Citations

The LLM outputs reasoning traces inside `<think>...</think>`, followed by the structured response:

```markdown
<think>
User asked to identify stories in The Goan but absent from The Morning Standard on 2026-08-01.
Evidence excerpt 1 provides the verified relational difference (142 exclusive articles).
Top front-page exclusives include the launch of AI traffic enforcement, tourist hotspot restrictions, and high power bills.
I will structure the answer into Executive Summary, Key Verified Highlights, Perspectives, and Explore Further.
</think>

### ⚡ Executive Summary
On August 1, 2026, *The Goan* published 142 articles that were completely absent from *The Morning Standard*, focusing predominantly on Goa civic administration, tourist safety regulations, and public utility billing.

### 📌 Key Verified Facts & Highlights
* **Automated Traffic Surveillance**: AI-powered camera corridors went live across Goa, generating automatic e-challans for signal violations and helmetless riding [*The Goan*, 2026-08-01, Page 1, "Beware! AI-enabled tra  c challans go live from today"].
* **Tourism Zone Restrictions**: The state government enacted sweeping administrative curbs across prominent coastal tourist destinations [*The Goan*, 2026-08-01, Page 1, "Govt imposes sweeping curbs at tourist hotspots"].
* **Electricity Tariff Pushback**: Local consumer groups voiced strong opposition against rising domestic power tariffs [*The Goan*, 2026-08-01, Page 1, "High power bills haunt consumers"].

### 📰 Broadsheet Perspectives & Focus Areas
* **The Goan Focus**: Heavy editorial commitment to local municipal enforcement, regional infrastructure, and state cabinet decisions.
* **The Morning Standard Focus**: Completely omitted Goa's local reporting, dedicating its front pages to federal Delhi policy and capital developments.

### 🔍 Explore Further
> 💡 Explore: What specific penalties are levied by the AI traffic cameras in Goa?
> 💡 Explore: Which tourist hotspots were placed under administrative curbs?
```

---

### 6.4 Server-Sent Events (SSE) Wire Protocol

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

event: stage
data: {"stage": "planning", "message": "Planning differential query strategy..."}

event: stage
data: {"stage": "tools", "message": "Executing sql_analytics and hybrid_search..."}

event: thought
data: {"token": "User asked to identify stories in The Goan..."}

event: token
data: {"token": "### ⚡ Executive Summary\n"}

event: token
data: {"token": "On August 1, 2026, *The Goan* published 142 articles..."}

event: citations
data: [
  {
    "article_id": 40403,
    "newspaper_name": "The Goan",
    "issue_date": "2026-08-01",
    "page_number": 1,
    "headline": "Beware! AI-enabled tra  c challans go live from today",
    "bbox": [[3994.61, 1530.89, 6215.43, 2082.70]]
  }
]

event: done
data: {"status": "completed", "latency_ms": 1180, "cost_usd": 0.0028}
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
  │    • Prevents topics from Turn 1 from leaking into Turn 2 queries.     │
  └────────────────────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 4. Deterministic Set Differences (sql_analytics)                       │
  │    • Computes exclusion counts directly in MySQL and Python.           │
  │    • The LLM is never allowed to guess article exclusion counts.       │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Phase 8: Comprehensive Top-K Lifecycle Reference

| Stage | Component | Default Value | Concrete Example | Reason & Effect |
|---|---|---|---|---|
| **Archetype Default** | `planner.py` | `top_k = 6` | `factual_lookup` | High precision, low distraction for specific factual questions. |
| **Archetype Comparison** | `planner.py` | `top_k = 10` to `12` | `cross_newspaper_comparison` | Retrieves 2–3 articles from each of 4–6 publications. |
| **Dense Candidate Fetch** | `hybrid_search.py` | $top\_k \times 3$ | $10 \times 3 = 30$ | Oversamples dense candidates from Qdrant prior to rank fusion. |
| **Sparse Candidate Fetch** | `hybrid_search.py` | $top\_k \times 3$ | $10 \times 3 = 30$ | Oversamples exact keyword matches from MySQL FULLTEXT. |
| **RRF Rank Constant** | `hybrid_search.py` | $k = 60$ | $\frac{1}{60 + \text{rank}}$ | Standard Cormack constant balancing dense and sparse ranks. |
| **Reranker Candidate Pool**| `hybrid_search.py` | $\max(75, top\_k \times 3)$ | 75 candidates | Balances 99%+ recall with sub-50ms Cross-Encoder latency. |
| **Cross-Encoder Slicing** | `reranker.py` | `top_k` | Returns top 10 | Cuts off candidates by cross-attention interaction score. |
| **CRAG Fallback Depth** | `graph.py` | `top_k = 5` | Entity fallback | Fallback search depth when initial retrieval returns 0 hits. |
| **Evidence Context Cap** | `synthesizer.py` | Top 12 items | `evidence[:12]` | Caps evidence tokens at $\le 3,500$ to prevent local LLM overflow. |

---

## 10. Phase 9: Information Retrieval & Generation Evaluation Metrics

[`backend/app/evaluation/metrics.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/evaluation/metrics.py) provides quantitative benchmarks:

### 1. Recall@K
$$\text{Recall@K} = \frac{|\text{Retrieved}_{1..K} \cap \text{GroundTruth}|}{|\text{GroundTruth}|}$$

### 2. Precision@K
$$\text{Precision@K} = \frac{|\text{Retrieved}_{1..K} \cap \text{GroundTruth}|}{K}$$

### 3. Mean Reciprocal Rank (MRR)
$$\text{MRR} = \frac{1}{\text{Rank of First Relevant Item}}$$
*In our live demo search for "AI-enabled challans go live"*:
- First relevant article (`Article 40403`) was ranked at Position 1.
$$\text{MRR} = \frac{1}{1} = 1.0$$

### 4. Normalized Discounted Cumulative Gain (NDCG@K)
$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

### 5. Citation Precision & Recall
NewsLens-AI requires **$100\%$ Citation Precision**: every bracketed citation `[Newspaper, YYYY-MM-DD, Page P, "Headline"]` must map to an existing row in `articles` and `article_pages`.

---

*End of End-to-End Data Flow & Data Structure Guide (Production Verified).*
