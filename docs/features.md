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
* **"Ask Agent About This Infographic / Photo" Direct Integration**:
  * Every cropped photo, diagram, and infographic card rendered in the broadsheet side inspector features a dedicated `"Ask Agent About This Infographic / Photo"` action button.
  * Clicking this button opens the conversational agent assistant with the asset pre-attached (`attachedAsset` state), showing an active blue banner with the headline, date, and page, immediately ready for multimodal reasoning.

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
* **Two-Stage Reranker Latency Optimization & Candidate Pool Capping**:
  * Employs CPU accelerator preference on macOS for `CrossEncoderReranker` (`ms-marco-MiniLM-L-6-v2`), avoiding the 10–15 second Metal shader compilation lag and GPU buffer synchronization stalls of MPS.
  * Caps candidate pairs passed to the neural cross-encoder at $K=20$ rather than unboundedly reranking all retrieved hits.
  * Drops warm hybrid search latency from **30,428 ms to 1,019 ms** (~30x speedup).
* **Strict Domain Purity & Zero-Coverage Matrix Reporting**:
  * Prevents off-domain article leakage (e.g. concert venue schedules, court trial delays, tax compliance reports) from polluting topical queries (e.g. Health & Medicine) through negative headline pattern filtering in SQL manifest builders.
  * Enforces explicit Zero-Coverage notices (`No standalone [Domain] reporting in this edition`) when publications lack reporting in the target sector, strictly preventing false substitutions.
  * Completely eliminates query-echoing in executive summaries, delivering objective journalistic syntheses.
* **Centralized Domain Taxonomy (`DOMAIN_TAXONOMY`) & Modular Synthesizer Architecture**:
  * Consolidates domain stem definitions, detection regexes, comparison table column headers, and negative exclusion patterns into a single module-level `DOMAIN_TAXONOMY` across all 6 core domains.
  * Employs top-level static imports and modular static renderers (`_render_comparison_matrix()`, `_render_front_page_comparison()`, `_render_broadsheet_perspectives()`, `_render_explore_further()`), eliminating Python module import locks and de-bloating `synthesizer.py`.
* **Archetype Preservation in Deterministic Fallbacks**:
  * Ensures deterministic fallback generation in `synthesizer.py` respects the active `QueryArchetype`, preventing cross-newspaper comparative queries from downgrading into single-newspaper lookups.
  * Preserves all planned newspaper publications in the structured comparison tables.
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
* **Inline Citation Parsing (`parse_inline_citation`)**:
  * Seamlessly extracts metadata from user follow-ups quoting previous answer citations in broadsheet bracketed formats (e.g. `[4] The Goan, 2026-08-01, Page 3, Headline: "..."` or `[{The Goan}, ...]`).
  * Resolves `headline`, `newspaper_name`, `issue_date`, and `page_number` for instant targeted inspection.
* **Active Reader Attached Asset Propagation**:
  * Supports direct asset routing from the Broadsheet Reader via `attached_article_id` and `attached_photo_id` fields in `QueryRequest`.
  * Renders an active blue attachment pill in `AgentAssistant.jsx` with headline, date, and page info, binding the asset to `PlanResult` for multimodal inquiry.
* **Strict Cross-Turn Invalidation Guardrails**:
  * In `extract_active_issue_from_history()`, strictly drops stale parameters (`article_id`, `photo_id`, `headline`, `page_number`, `target_newspapers`) across turns:
    * **Guardrail 1**: Purges parameters when user specifies a new or conflicting issue date.
    * **Guardrail 2**: Purges parameters when user switches to a different publication brand.
    * **Guardrail 3**: Purges parameters when user introduces an explicit headline or new topic.
  * Injects `Verified Available Publications for this Query` and strict isolation constraints into synthesizer prompts, guaranteeing that past conversation topics never bleed into new dates.
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
* **Server-Sent Events (SSE) Streaming with Visual Provenance**:
  * Low-latency token streaming with live tool telemetry (`event: stage` including `inspecting_visual_asset`).
  * Yields structured citations (`event: citations`) flagged with `is_visual_asset: true` and thumbnail endpoints (`/api/photos/{id}/image`), rendering interactive visual cards in the UI.

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
* **Agentic Visual Inspection Tool (`inspect_visual_asset`)**:
  * Equips the conversational agent with dedicated visual inspection capabilities across 5 execution strategies:
    * **Strategy A**: Direct `photo_id` lookup with companion chart discovery and defensive publication validation.
    * **Strategy B**: Target headline lookup mapped to visual assets on the matching date.
    * **Strategy C**: Direct `article_id` lookup retrieving all associated charts and diagrams.
    * **Strategy D**: Multi-criteria database search scoped by publication, date, page, and caption.
    * **Strategy E**: Scoped caption and VLM keyword search strictly joined with `Issue` to prevent photo leakage.
  * **On-Demand MinIO VLM Fallback**: If a targeted asset has a default placeholder description, streams the original high-resolution crop directly from MinIO `bucket_pages`, triggers `VisualDataExtractor.process_image_crop()`, transcribes the Markdown table/metrics, and persists the result to MySQL.
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

## 9. Dynamic Model Provider Studio & Hot-Swappable Runtime Bindings

* **Provider Agnostic Architecture**: Supports local inference (Ollama, Sentence-Transformers, RapidOCR, Tesseract) and hosted cloud providers (NVIDIA NIM, Anthropic Claude, OpenAI, Google Gemini, Groq).
* **Model Settings Studio (`ModelSettingsStudio.jsx`)**:
  * Dedicated interactive settings control panel in the web application for managing AI providers and active task bindings.
  * Real-time API key management, custom endpoint URLs, connection health testing, and temperature/context window controls.
  * Live visual capability indicators showing whether a selected model supports text chat, structured function calling, vision/multimodal reasoning, or embedding.
* **Hot-Swappable Task Bindings**:
  * Configure distinct models for each functional pipeline task independently:
    * `query_planner`: Fast structured tool scheduling and archetype classification.
    * `synthesizer` / `answerer`: High-capacity grounded synthesis and analytical reporting.
    * `vlm_extractor`: Multimodal vision models for charts, tables, and broadsheet photojournalism.
    * `embedding`: Dense vector representations (BAAI/bge-m3).
    * `ocr`: Local RapidOCR / Tesseract or cloud OCR engines.
  * Persisted in `model_config.yaml` and hot-reloaded dynamically via `/api/settings/model-bindings` without server restarts.
* **Task Capability Validation**: Validates that assigned providers satisfy required capabilities (e.g. vision support for layout analysis and chart extraction).

---

## 10. Cross-Publication Comparison & Perspective Analysis

* **Multi-Broadsheet Coverage Matrix**: Compares how multiple newspapers (e.g. *The Economic Times*, *Mint*, *The Hindu*, *The New York Times*) covered the same event on the same date.
* **Front-Page Lead Story Divergence**: Highlights differences in editorial priorities, tone, and lead headlines between publications.
* **Differential Exclusives vs. Shared Wire Stories**:
  * Identifies stories that were exclusive to a single broadsheet versus syndicated national wire stories (PTI, ANI, Reuters) covered universally across all newspapers.
  * Provides granular page-by-page breakdown of regional news pages, editorial supplements, and local investigative scoops.

---

## 11. Dual-Mode Archive & Live Internet Web Search Grounding

* **Journalism-First Multi-Tier Live Search Cascade**:
  * Seamlessly augments historical newspaper archives with real-time live internet grounding when `enable_web_search = True` is selected.
  * **Tier 1: NewsData.io API (Primary Journalism Engine)**:
    * Queries global journalistic press agencies, wire feeds, and accredited broadsheet publishers (Reuters, The Hindu, Mint, ANI, The Economic Times, Bloomberg, etc.).
    * Captures authoritative publisher attribution (`source_name`), original canonical URLs, publication timestamps (`pubDate`), and journalistic abstracts.
  * **Tier 2: Serper API (Google Search Engine)**: High-precision Google Web Search engine fallback for encyclopedic context and general web queries.
  * **Tier 3: Tavily Search API (AI Research Engine)**: Fact-dense search engine optimized for AI synthesis and contextual snippet extraction.
  * **Tier 4: DuckDuckGo HTML / Instant Search**: Zero-configuration, zero-API-key fallback ensuring 100% resilient live web grounding even without external credentials.
* **Dual-Mode Visual Citation Badging**:
  * Explicit provenance segregation between physical broadsheet print archives and live internet sources.
  * Printed broadsheet citations: `[Broadsheet Archive]` (`[{Newspaper}, YYYY-MM-DD, Page X, "Headline"]`).
  * Live web citations: `[Live Web]` (`[{Publisher/Domain}, YYYY-MM-DD, Live Web, "Headline"]`).
* **Zero-Failure Cascade & Graceful Degradation**:
  * Network timeouts, HTTP 5xx errors, and rate limits trigger immediate, non-blocking fallback to subsequent search tiers, maintaining sub-second response times without breaking the agent graph.

---

## 12. Dynamic Tool Synthesis & Subprocess AST Sandbox

* **On-Demand Dynamic Tool Synthesis (`tool_maker.py`)**:
  * Employs the **LLM-as-Tool-Maker** pattern to synthesize bespoke Python/SQL analysis functions when user inquiries exceed the scope of predefined static tools (e.g. newspaper page count distributions, cross-section statistics, custom multi-table aggregations).
  * Prompts the LLM with the complete 17-table MySQL schema, column definitions, and example analytical queries to generate self-contained, typed functions matching the signature:
    ```python
    def execute(connection, **kwargs) -> Dict[str, Any]: ...
    ```
* **Subprocess AST Sandbox Execution Engine (`sandbox.py`, `sandbox_runner.py`)**:
  * **Abstract Syntax Tree (AST) Safety Scanner**:
    * Pre-execution static analysis verifying AST node safety via `ASTSafetyScanner`.
    * **Whitelisted Safe Modules**: `math`, `datetime`, `re`, `json`, `collections`, `itertools`, `typing`, `sqlalchemy`, `decimal`.
    * **Blacklisted Forbidden Modules**: `os`, `sys`, `subprocess`, `socket`, `shutil`, `urllib`, `requests`, `pathlib`, `pickle`, `ctypes`, etc.
    * **Blacklisted Dangerous Builtins**: `open`, `eval`, `exec`, `compile`, `__import__`, `globals`, `locals`, `getattr`, `setattr`.
    * **Dunder Attribute Protection**: Prohibits access to `__subclasses__`, `__bases__`, `__globals__`, `__code__`, etc.
  * **Subprocess Process Isolation**:
    * Spawns an isolated subprocess (`sys.executable`) via non-blocking JSON IPC over stdin/stdout.
    * Enforces a hard **15-second execution timeout** and **512MB memory limit**.
  * **Read-Only Database Transactions**:
    * Executes all dynamic tool queries under an uncommitted, read-only transaction.
    * Forces `connection.rollback()` unconditionally in a `finally` block, completely preventing any accidental table updates, inserts, or deletions.
* **Two-Layer Reactive Dynamic Fallback Architecture**:
  * **Layer 1 (Unsupported Parameter Handoff in `executor.py`)**:
    * Intercepts tool execution when the Planner passes unsupported arguments (such as `analysis_type="page_count"` in `sql_analytics`).
    * Transparently hands off the request to `dynamic_analysis`, synthesizing a custom script on-the-fly and returning verified metrics without raising runtime errors.
  * **Layer 2 (CRAG Zero-Evidence Dynamic Toolmaker in `evaluator.py`)**:
    * When primary retrieval tools return zero hits or insufficient evidence ($< 0.4$) on quantitative or analytical queries, the Corrective RAG (CRAG) Evaluator intercepts the failure.
    * Invokes `ToolMaker` asynchronously to synthesize and run a focused analysis script, injecting high-confidence evidence ($1.0$) into the agent's context before final synthesis.

---

## 13. Cross-Date Context Isolation & Anti-Leakage Shield

* **Multi-Turn Date Drift Prevention**:
  * Eliminates cross-turn context contamination when users explore an attached visual asset on one date (e.g., Aug 5, 2026) and subsequently ask a question about another date (e.g., Aug 1, 2026).
* **4-Tier Anti-Leakage Shield**:
  1. **Query Condenser Isolation (`condenser.py`)**:
     * Detects explicit date and newspaper mentions in the incoming query.
     * Automatically evicts stale historical issue IDs, dates, and newspaper brands from conversation memory if a mismatch is found.
  2. **Attached Asset Eviction Gate (`query.py`, `graph.py`)**:
     * Inspects attached visual assets from the Broadsheet Reader.
     * Automatically prunes `attachedAsset` from the active query state if its publication date or newspaper conflicts with the explicit query intention.
  3. **Sanitizer Reconciliation (`tool_factory.py`)**:
     * Verifies planned tool arguments against explicit query dates, sanitizing any stale date filters carried over from earlier turns.
  4. **Executor Date Non-Overwriting Invariant (`executor.py`)**:
     * Enforces an immutable priority rule during tool execution: explicit user query dates strictly override attached asset metadata (`effective_date = explicit_query_date or asset_date or default_date`), guaranteeing that queries never target the wrong newspaper issue.


