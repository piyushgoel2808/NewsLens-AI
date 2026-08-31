# NewsLens-AI Comprehensive Features Guide

NewsLens-AI delivers a full-stack, enterprise-grade newspaper intelligence system. This document catalogs every feature, user capability, analytical tool, and backend system feature in detail.

---

## 1. Broadsheet Reader & Spatial Overlay Explorer

* **High-Resolution 300 DPI Rendering**: View digitized broadsheet pages rendered at true archival resolution without loss of fidelity.
* **Interactive 2D Bounding Box Overlays**: Real-time canvas/SVG overlays highlighting article boundaries, column tracks, and headline decks directly over original newspaper scans.
* **Prominence Heatmap Color-Coding**:
  * 🟡 **Gold / Amber**: Front-page lead stories and major banner headlines (Prominence $\ge 0.70$).
  * 🔵 **Navy / Blue**: Standard body articles and inside editorial reports (Prominence $0.30 - 0.69$).
  * 🟢 **Teal / Cyan**: Secondary briefs, column shorts, and statutory notices (Prominence $< 0.30$).
* **Physical Page Folio vs PDF Index Tracking**: Accurately displays both the physical printed page folio (e.g. *"Page 5"*, *"Page B4"*, *"Foliated 14"*) and the zero-indexed PDF container index (`PDF p. 5`).
* **Zoom, Pan, and Responsive Viewport Controls**: Smooth zoom (50% to 300%), drag-to-pan, fit-to-width, fit-to-page, and keyboard navigation.
* **Interactive Single-Page Re-Ingestion (`POST /api/issues/{issue_id}/pages/{page_number}/reingest`)**:
  * Allows users to re-run layout parsing, OCR, and photo extraction for a single problematic page directly from the reader toolbar (`Re-ingest Page` button).
  * Atomically deletes previous page-exclusive articles, entities, topics, chunks, photos, and Qdrant vector points, replacing them with freshly parsed records without corrupting or restarting the entire multi-page issue.
* **VLM Spatial Grounding for Composite Photo Displays**:
  * Automatically applies VLM visual grounding (`detect_subphotos_via_vlm_grounding`) to identify, describe, and crop discrete portraits and photo insets on complex composite display pages.

---

## 2. Conversational Broadsheet AI Assistant

* **6 Specialized Broadsheet Query Archetypes**:
  1. `factual_lookup`: Direct extraction of names, statements, data points, or events with strict citations.
  2. `quantitative_trend`: Broad issue summaries, article manifests, category counts, and section distributions.
  3. `thematic_timeline`: Chronological progression and milestone evolution across dates.
  4. `cross_newspaper_comparison`: Comparative framing analysis across different publications.
  5. `entity_deep_dive`: Comprehensive profile of people, corporations, or geopolitical entities.
  6. `conversational_meta_query`: Conversational continuity, follow-ups, and greetings.
* **Conversational Context Condenser & Reference Disambiguation**:
  * Resolves pronouns (*"its"*, *"they"*, *"them"*, *"this newspaper"*).
  * Automatically binds active publication names (*"The Economic Times"*) and issue dates (*"2026-08-27"*) across multiple dialogue turns.
  * Short-circuits ambiguous opening questions with interactive suggestions.
  * **In-Context Meta-Query Detection (`is_in_context_meta_query`)**: Directly answers follow-up inquiries about prior turns (*"which newspaper was that?"*, *"what was the date?"*, *"who wrote this article?"*) directly from chat history without triggering wasteful retrieval cascades.
* **Intelligent Multi-Date Extraction & Single-Brand Comparative Routing**:
  * Automatically parses multi-date expressions across ISO, DD/MM/YYYY, and month-name formats (e.g., `1/8/2026 and 2/8/2026`).
  * If a query compares multiple editions or dates of the *same* newspaper, the planner intelligently routes to targeted SQL issue summaries and scoped hybrid search instead of invoking an all-newspaper `coverage_analysis` across the entire database.
* **Dynamic Brand-to-ID Filter Resolution**:
  * In `graph.py`, hybrid searches mentioning publication names (such as *"The Goan"*) dynamically query MySQL to resolve the exact `newspaper_id`, ensuring search results are strictly confined to the requested newspaper.
* **Context Budgeting & Anti-Hallucination Memo Protection**:
  * Caps evidence context to the top 12 items and truncates long excerpts to 1,200 characters, guaranteeing the synthesizer prompt stays within $\le 3,500$ tokens to prevent local LLM context overflow (`4096 tokens`).
  * Protects against pre-training knowledge cutoff date hallucinations (e.g., memorized 2023 dates) with negative prompt guards and regex post-cleaning.
* **Corrupted Font CMap Recovery & High-Precision Image OCR Fallback**:
  * Employs automated `\ufffd` replacement character and gibberish ratio detection (`CorruptedPdfTextLayerError`).
  * When a PDF's embedded fonts lack valid `ToUnicode` mapping tables (causing traditional text scrapers to output unmapped glyphs), automatically escalates to pure image OCR via `GoogleCloudVisionOCR` on the 300 DPI raster page.
  * Reconstructs 2D reading order with `LayoutAnalyzer` and segments clean articles without corrupt Unicode symbols.
* **4-Tier Structured Broadsheet Synthesis**:
  * **Executive Summary**: High-level macro context.
  * **Key Verified Facts & Highlights**: Bulleted list of verified claims with bracketed citations.
  * **Broadsheet Perspectives**: Editorial framing analysis comparing Page 1 front-page leads against inside reporting.
  * **Explore Further**: Contextual follow-up suggestions to continue exploration.
* **Strict Anti-Hallucination & Provenance Grounding**:
  * Every fact is tied to a verified citation in the format `[Newspaper, Issue Date, Page, "Headline"]`.
  * If no relevant facts exist in the database, the system executes an anti-hallucination hard stop rather than fabricating facts.
* **Server-Sent Events (SSE) Streaming**: Low-latency token streaming with live tool telemetry and reasoning traces.

---

## 3. Visual Infographic, Chart, Table & Photo Intelligence

* **Visual Asset Harvesting**: Automatically crops photos, corporate logos, data charts, circular/donut infographics, and tabular graphics from broadsheet pages.
* **Dual-Engine Visual Intelligence**:
  * **Multimodal VLM Analysis (Qwen-3VL & Vision LLMs)**:
    * Primary inference using local/hosted vision models (`qwen3-vl`, `qwen2.5-vl`, `gemini-1.5-pro`, `claude-3-5-sonnet`, `gpt-4o`) to transcribe financial bar charts, multi-year trend graphs, pie/donut charts, and tabular grids.
    * Generates 2-sentence executive summaries, extracts 3 to 6 key statistical metrics, and outputs clean GitHub-flavored Markdown tables.
    * **Anti-GBNF Deadlock & Token Starvation Protections**: Bypasses strict schema grammar locks on local vision models while utilizing multi-layer `repair_and_parse_json()` and recovering table transcriptions from reasoning thinking tokens when content buffers are starved.
  * **Deterministic Spatial OCR Matrix Reconstruction**: Zero-failure fallback engine that clusters OCR tokens into horizontal rows and column lanes, reconstructing GitHub-flavored Markdown tables and deriving statistical metrics (e.g. IPO subscription matrices) with confidence $\ge 0.85$.
* **Editorial Photograph Scene Intelligence**:
  * Automatically analyzes editorial photographs (people, events, vehicles, locations, protests, industry) to generate rich 2-3 sentence visual scene breakdowns, identifying visible subjects, context, and actions.
* **On-Demand Visual Intelligence API & Interactive Broadsheet Controls**:
  * **`POST /api/photos/{photo_id}/analyze`**: Triggers real-time on-demand VLM visual intelligence for any broadsheet photo or graphic, updating `vlm_description` and `visual_type` in MySQL.
  * **Interactive Reader Controls**: Broadsheet Reader photo cards include `⚡ Analyze with Qwen-VL` and `🔄 Re-Analyze with VLM` buttons with live loading animations and verified scene badges.
* **Spatial Polygon Media Binding**: Binds cropped photos and charts to their parent editorial article using horizontal overlap and vertical proximity algorithms.
* **Dedicated Visual RAG Chunks**: Generates unfragmented `[INFOGRAPHIC / DATA TABLE]` chunks embedded in Qdrant for dense semantic retrieval.
* **Interactive Media Inspector**: View high-resolution cropped assets in a side modal with full image zoom, caption, AI scene breakdown, and transcribed tabular data.

---

## 4. Universal 12-Domain Newsroom Taxonomy & Metaphor Disambiguation

* **12 Canonical Newsroom Desks**:
  * `Business & Markets`
  * `Economy & Policy`
  * `Politics & Governance`
  * `National`
  * `World & International`
  * `Corporate & Industry`
  * `Technology & Startups`
  * `Sports`
  * `Entertainment & Culture`
  * `Science & Environment`
  * `Health & Medicine`
  * `Opinion & Editorial`
* **Multi-Signal Probabilistic Classifier**:
  * Weighted token scoring: Headline ($3.0\times$), Subheadline/Deck ($2.0\times$), Body Text ($1.0\times$).
* **Domain Context Anchor Dampening**:
  * Dampens metaphorical keywords (e.g., sports/war idioms like *"Bulls hit for a six"*, *"political chess"*) by $0.15\times - 0.25\times$ when corporate or financial anchors are detected.
* **Multi-Topic Secondary Tagging**:
  * Identifies cross-domain articles (e.g. *"Rajasthan Royals Franchise Acquisition for ₹4,000 Cr"* as both *Business* and *Sports*) and stores secondary tags in relational junction tables.

---

## 5. Geometric Ad-Barrier Isolation & Slogan Suppression

* **Statutory Disclosure Detection**: Detects commercial envelopes matching regulatory notices (`QUALIFIED INSTITUTIONS PLACEMENT`, `BOOK RUNNING LEAD MANAGERS`, `ISSUE PRICE`, `REGISTRAR TO THE ISSUE`).
* **Convex Boundary Wall Isolation**: Encloses advertisements in geometric bounding boxes and injects synthetic delimiter headlines (`[Advertisement] <Ad Title>`), preventing ad copy from bleeding into adjacent news articles.
* **Marketing Slogan Byline Filter**: Uses `MARKETING_SLOGAN_REGEX` to prevent commercial taglines (*"By Innovation I Built For The Future"*, *"Backed by Trust"*) from being parsed as author bylines.

---

## 6. Relational Newspaper Analytics & Manifest Engine (`sql_analytics`)

* **Instant Broadsheet Manifests**: Generates structured table-of-contents listings showing all articles, pages, word counts, and authors for an entire issue.
* **Resilient Multi-Tier Issue Resolution**:
  * Automatically resolves queries even when user types an incorrect issue ID (e.g. querying `issue 84` when the database stores `Issue #88`) by falling back to `(newspaper_name, issue_date)`.
* **Cross-Sectional Filtering**: Filter articles by physical page number, printed folio, newspaper section, or canonical category.
* **Statistical Aggregations**: Compute article frequency trends, mention distributions, and front-page prominence ratios.

---

## 7. Thematic Storyline Trajectory & Timeline Canvas

* **Chronological Milestone Clustering**: Clusters related stories across multiple dates and issues to build comprehensive chronological trajectories.
* **Interactive Storyline Canvas**: Visual milestone cards with dates, summaries, and direct links to historical broadsheet pages.
* **Redis Caching**: Caches computationally intensive multi-week timeline trajectories for instant sub-10ms retrieval.

---

## 8. Interactive Multi-Hop Entity Knowledge Graph

* **Named Entity Extraction & Salience**: Identifies people, corporations, government bodies, and geopolitical locations with computed prominence salience scores ($0.0$ to $1.0$).
* **Entity Co-occurrence Graph**: Visualizes multi-hop connections and co-mention networks between entities.
* **Click-to-Inspect**: Select any entity node in the graph to filter all broadsheet articles referencing that entity.

---

## 9. Runtime Model Provider Binding & Hot-Swapping

* **Provider Agnostic Architecture**: Supports local inference (Ollama, Sentence-Transformers, Tesseract) and hosted providers (Anthropic, OpenAI, Google Gemini, Groq).
* **Hot-Swappable Task Bindings**: Configure distinct providers for `query_planner`, `answerer`, `layout_analysis`, `embedding`, and `ocr` in `model_config.yaml` or dynamically via the `/api/settings/model-bindings` API.
* **Task Capability Validation**: Validates that assigned providers satisfy required capabilities (e.g. vision support for layout analysis).

---

## 10. Cross-Publication Comparison & Perspective Analysis

* **Multi-Broadsheet Coverage Matrix**: Compares how multiple newspapers (e.g. *The Economic Times*, *Mint*, *The Hindu*, *The New York Times*) covered the same event on the same date.
* **Front-Page Lead Story Divergence**: Highlights differences in editorial priorities, tone, and lead headlines between publications.

---

## 11. Dual-Mode Archive & Live Internet Web Search Grounding

* **Live Web Search Fallback**: Toggle live internet search (via Serper, Tavily, or DuckDuckGo) alongside archived broadsheets.
* **Source Badge Differentiation**: Clear UI tags separating verified `[Broadsheet Archive]` citations from `[Live Web]` sources.
