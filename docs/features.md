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

* **7 Specialized Broadsheet Query Archetypes**:
  1. `factual_lookup`: Direct extraction of names, statements, data points, or events with strict citations.
  2. `quantitative_trend`: Broad issue summaries, article manifests, category counts, and section distributions.
  3. `thematic_timeline`: Chronological progression and milestone evolution across dates.
  4. `cross_newspaper_comparison`: Comparative framing analysis across different publications.
  5. `entity_deep_dive`: Comprehensive profile of people, corporations, or geopolitical entities.
  6. `negative_coverage_audit`: Verifying what a publication did NOT report or cover on a specific topic/date.
  7. `article_catalog`: Ultra-fast (<200ms) listing and catalog manifest generation for specific dates and categories.
* **Grounded Archive Intelligence & Environmental Planning**:
  * Injects live database issue dates, active publications, and canonical section categories directly into the planner LLM via `sql_analytics.get_archive_metadata()`.
  * Eliminates parameter hallucination and enables the agent to autonomously reason about what exists in the broadsheet archive before selecting tools.
* **Sub-200ms Instant Manifest Routing (`article_catalog`)**:
  * Routes listing requests (e.g. *"list all there health news"*, *"show all articles on politics"*) strictly to `sql_analytics`.
  * Completely bypasses vector embeddings, Qdrant hybrid search, and cross-encoder reranking, delivering complete manifests in sub-200ms.
* **Concurrent Multi-Tool Gathering (`asyncio.gather`)**:
  * Executes all planned tools (`sql_analytics`, `hybrid_search`, `coverage_analysis`) in parallel rather than sequentially, reducing comparative query latency by 40–60%.
* **Adaptive Zero-Hit Real-Time Fallback**:
  * If a category-filtered search returns 0 articles due to taxonomy mismatches, the execution engine dynamically catches the empty result and autonomously re-executes without the category filter.
* **Calibrated Negative Coverage Engine**:
  * Scopes cross-newspaper reconciliation queries strictly to publications with active issues on the target date.
  * Calibrates cross-encoder logit scoring (`>= -5.0`) and RRF thresholds (`>= 0.008`), eliminating false `PROCESSING_ERROR` classifications on genuine reporting.
* **Editorial Headline Cleansing & Author Box Sanitization (`sanitize_headline`)**:
  * Automatically detects author/doctor profile name boxes (e.g. `Dr. Smriti Naswa Singh`, `UTHAMA SANKARANARAYANAN`) mistakenly extracted as headlines and converts them into descriptive topical feature labels.
  * Employs common news vocabulary safeguards (`_COMMON_HEADLINE_VOCAB`) to protect genuine all-caps headlines (e.g. `TECH STOCKS RALLY`).
* **Broadsheet OCR Font Ligature Repair Engine (`repair_text_ligatures`)**:
  * Decomposes Unicode typographic ligatures (`\ufb00`–`\ufb06` $\to$ `ff`, `fi`, `fl`, `ffi`, `ffl`, `ft`, `st`).
  * Repairs broadsheet OCR dropout patterns where font ligatures collapsed into replacement glyphs or multi-space gaps (e.g. `e \ufffd orts` / `e   orts` $\to$ `efforts`, `in \ufffd ation` $\to$ `inflation`, `di \ufffd erent` $\to$ `different`, `sta\ufffd` $\to$ `staff`, `o\ufffd cial` $\to$ `official`).
  * Applied in-place across search result headlines, evidence snippets, and SQL manifest strings.
* **Autonomous Query Topic Preservation & Generic Filler Sanitization**:
  * Protects user domain queries from few-shot prompt contamination in the planner LLM.
  * In `_build_plan_from_structured_model()`, automatically detects generic filler phrases (e.g. `"newspaper coverage comparison"`) and restores substantive domain topics (e.g. `"health related news"`).
* **Minimal Sufficient Tool Scheduling & Conditional Coverage Analysis**:
  * Intelligently skips unconstrained 25-second `coverage_analysis` on domain comparison queries unless explicit negative audit/omission keywords (`omission`, `miss`, `gap`, `absent`) are specified.
  * For domain comparisons with category filters, relies on ultra-fast `sql_analytics` manifest (105ms) and targeted `hybrid_search` (1s), cutting overall execution latency by >90%.
* **Archetype Preservation in Deterministic Fallbacks**:
  * Ensures deterministic fallback generation in `synthesizer.py` respects the active `QueryArchetype`, preventing cross-newspaper comparative queries from downgrading into single-newspaper lookups.
  * Preserves all planned newspaper publications in the structured comparison tables.
* **Granular Domain Token Stem Budgeting**:
  * Maps composite domain labels (e.g. `Health & Medicine`) to granular keyword stems (`["health", "hospital", "pharma", "medicine", "doctor", ...]`), guaranteeing domain articles receive top priority during context token budgeting.
* **Domain-Adaptive Comparative Synthesis**:
  * Dynamically adapts comparison table headers to the query domain:
    * **Health & Medicine**: `Key Findings & Medical Focus`
    * **Finance & Markets**: `Key Figures & Metrics`
    * **Politics & Governance**: `Key Policy Decisions & Statements`
    * **General**: `Key Takeaways & Core Findings`
  * Includes dedicated markdown table schemas for `article_catalog` and enforces strict 1-shot citation patterns with anti-repetition constraints.
* **Conversational Context Condenser & Reference Disambiguation**:
  * Resolves pronouns (*"its"*, *"they"*, *"them"*, *"there"*, *"their"*, *"this newspaper"*).
  * Automatically binds active publication names (*"The Economic Times"*) and issue dates (*"2026-08-27"*) across multiple dialogue turns.
  * Short-circuits ambiguous opening questions with interactive suggestions.
  * **In-Context Meta-Query Detection (`is_in_context_meta_query`)**: Directly answers follow-up inquiries about prior turns (*"which newspaper was that?"*, *"what was the date?"*, *"who wrote this article?"*) directly from chat history without triggering wasteful retrieval cascades.
* **Intelligent Multi-Date Extraction & Single-Brand Comparative Routing**:
  * Automatically parses multi-date expressions across ISO, DD/MM/YYYY, and month-name formats (e.g., `1/8/2026 and 2/8/2026`).
  * If a query compares multiple editions or dates of the *same* newspaper, the planner intelligently routes to targeted SQL issue summaries and scoped hybrid search instead of invoking an all-newspaper `coverage_analysis` across the entire database.
* **Dynamic Brand-to-ID Filter Resolution**:
  * In `graph.py`, hybrid searches mentioning publication names (such as *"The Goan"*) dynamically query MySQL to resolve the exact `newspaper_id`, ensuring search results are strictly confined to the requested newspaper.
* **Dynamic Publication & Date Isolation**:
  * Employs query-aware guardrails in `extract_active_issue_from_history()` to drop stale publications or dates whenever a user switches to a different date or initiates a comparative query.
  * Injects `Verified Available Publications for this Query` and strict isolation constraints into synthesizer prompts, guaranteeing that past conversation topics (e.g. LIV Golf, Ram Temple) never bleed into new dates.
* **Multi-Brand Parameter Extraction & Typo Tolerance**:
  * Parses multiple newspaper brand names from complex queries in token order without breaking on the first match.
  * Detects differential exclusion syntax (`"but not in"`, `"not in"`, `"exclusive to"`, `"absent in"`).
  * Tolerates natural user typos (e.g., `"he Morning Standard"` correctly maps to `The Morning Standard`).
* **Conversational Follow-Up Enumeration**:
  * Coreference resolution engine preserves differential comparison contexts across subsequent turns.
  * Translates follow-up requests (e.g., *"list all those 11 articles"*) into fully specified standalone queries: *"list all those articles in The Goan but not in The Morning Standard dated 2026-08-01"*, eliminating count hallucinations and retrieving the full grounded manifest.
* **Context Budgeting & Anti-Hallucination Memo Protection**:
  * Caps evidence context items and selectively allocates up to 4,000 characters for relational issue manifests and coverage differences, guaranteeing the synthesizer receives complete article listings without token truncation.
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
* **NVIDIA NIM Hosted Acceleration (`nvidia_nemotron` & `nvidia_llama_vision`)**:
  * Seamless integration with NVIDIA API Catalog / NIM endpoints (`https://integrate.api.nvidia.com/v1`).
  * Delivers sub-second (~0.59s) reasoning completions with `nvidia/nemotron-3.5-lightning-30b-a3b`, streaming thinking deltas directly into the collapsible reasoning accordion.
  * Powers multimodal vision reading with `meta/llama-3.2-11b-vision-instruct` for high-resolution newspaper charts and photojournalism.
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
* **Deterministic Cross-Newspaper Differential Coverage (`get_newspaper_coverage_difference`)**:
  * Computes the exact set of articles present in publication $A$ but completely absent from publication $B$ on any shared publication date.
  * Employs headline tokenization, stop-word reduction, and Jaccard overlap scoring against the comparison newspaper's manifest.
  * Filters running page furniture ("SATURDAY", "IN SHORT >>", "PANAJI") to isolate genuine news reports.
  * Outputs categorized manifests classifying articles into Hyperlocal/Regional Exclusives (e.g. Goa local administration, traffic challans, municipal infrastructure) and Distinct Editorial Exclusives.

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
* **Differential Exclusives vs. Shared Wire Stories**:
  * Identifies stories that were exclusive to a single broadsheet versus syndicated national wire stories (PTI, ANI, Reuters) covered universally across all newspapers.
  * Provides granular page-by-page breakdown of regional news pages, editorial supplements, and local investigative scoops.

---

## 11. Dual-Mode Archive & Live Internet Web Search Grounding

* **Live Web Search Fallback**: Toggle live internet search (via Serper, Tavily, or DuckDuckGo) alongside archived broadsheets.
* **Source Badge Differentiation**: Clear UI tags separating verified `[Broadsheet Archive]` citations from `[Live Web]` sources.
