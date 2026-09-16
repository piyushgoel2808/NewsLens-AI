# NewsLens-AI: Complete End-to-End Data Flow & System Architecture

This document serves as the master technical specification for the entire **NewsLens-AI** data pipeline. It unifies all data flows, state transitions, transformation matrices, subsystem boundaries, and execution sequences—tracking broadsheet newspaper PDFs from intake, computer vision, and spatial layout extraction, through multi-tier persistence, into hybrid vector/relational search, and finally through the LangGraph agentic RAG retrieval, dynamic code synthesis, and synthesis state machine.

---

## 1. High-Level System Architecture & Global Subsystem Topology

```mermaid
flowchart TD
    subgraph CLIENT ["1. Presentation Layer (React 18 + Vite SPA)"]
        UI_Reader["Broadsheet Reader<br/>(300 DPI Canvas + Overlay BBoxes)"]
        UI_Visual["Visual Asset Inspector<br/>(Charts, Photos, Tables)"]
        UI_Agent["Agentic Assistant<br/>(SSE Streaming Reasoning & Citations)"]
        UI_Studio["Model Settings Studio<br/>(Auto-Save, Hot-Swapping, Health Pings)"]
        UI_Graph["Entity Knowledge Graph<br/>(Multi-Hop Relational Network)"]
    end

    subgraph INTAKE ["2. Document Intake & Computer Vision Pipeline"]
        PDF["Broadsheet PDF / ZIP Archive"] --> Compressor["Pre-Ingestion Compressor<br/>(fitz.deflate / Ghostscript)"]
        Compressor --> SHA["SHA-256 Idempotency Check"]
        SHA --> MinIO_Orig[("MinIO: newslens-originals")]
        SHA --> Masthead["Visual Masthead Verifier<br/>(Top 22% Page 1 RapidOCR)"]
        Masthead --> Consensus["Multi-Page Folio Consensus<br/>(5x Header-Weighted Voting across P1-15)"]
        Consensus --> Rasterizer["PyMuPDF Rasterizer<br/>(300 DPI High-Res Rendering)"]
        Rasterizer --> MinIO_Pages[("MinIO: newslens-pages")]
        Rasterizer --> LayoutParser{"Layout Parser Engine<br/>(Registry Resolved)"}
        LayoutParser -->|Primary Local| Docling["Docling 2D Neural Parser<br/>(DocLayNet + RapidOCR)"]
        LayoutParser -->|Cloud VLM / Direct| CloudLayout["Google Cloud Vision / Gemini / Gemma"]
        Docling & CloudLayout --> LayoutEngine["5-Pass Spatial Layout Engine<br/>(Column De-bundling & Jump Stitching)"]
        LayoutEngine --> VisualExtractor["3-Stage Visual Extractor<br/>(Triage + VLM + Spatial OCR Matrix)"]
    end

    subgraph STORAGE ["3. Multi-Tier Persistence & Knowledge Layer"]
        LayoutEngine --> MySQL[("MySQL 8 (System of Record)<br/>Articles, FULLTEXT, Entities, Photos")]
        VisualExtractor --> MySQL
        LayoutEngine --> Chunker["Contextual Broadsheet Chunker<br/>(400-500 Tok + Context Headers)"]
        VisualExtractor --> Chunker
        Chunker --> Embedder["Embedding Provider<br/>(BAAI/bge-m3 1024-dim Dense)"]
        Embedder --> Qdrant[("Qdrant Vector DB<br/>article_chunks Collection")]
        MySQL -.-> Redis[("Redis 7 Cache<br/>Query Results, Celery Broker, Trajectories")]
    end

    subgraph AGENTIC ["4. Agentic RAG & LangGraph State Machine"]
        UserQuery["User Query + Attached Assets"] --> Condenser["Conversational Query Condenser<br/>(Inline Citations & Anti-Leakage Guardrails)"]
        Condenser --> Planner["Dynamic Query Planner<br/>(7 Archetypes & Live Archive Metadata)"]
        Planner --> Dispatcher["Concurrent Tool Execution Engine"]
        
        Dispatcher --> Tool_Hybrid["HybridSearchEngine<br/>(Dense Qdrant + MySQL FULLTEXT)"]
        Dispatcher --> Tool_Visual["InspectVisualAsset<br/>(5-Tier Cascade A-E & MinIO Streaming)"]
        Dispatcher --> Tool_SQL["SQLAnalyticsDispatcher<br/>(Strict 7-Enum Routines)"]
        Dispatcher --> Tool_Entity["EntityFilter<br/>(N-Hop Relational Search)"]
        Dispatcher --> Tool_Timeline["TimelineBuilder<br/>(Narrative Trajectories)"]
        Dispatcher --> Tool_Web["WebSearchEngine (4-Tier Grounding)<br/>NewsData.io ➔ Serper ➔ Tavily ➔ DDG"]
        Dispatcher --> Tool_Dynamic["DynamicAnalysis<br/>(Subprocess AST Sandbox)"]

        Tool_Hybrid --> RRF["Reciprocal Rank Fusion (RRF)<br/>+ Cross-Encoder Reranker (CPU)"]
        RRF & Tool_Visual & Tool_SQL & Tool_Entity & Tool_Timeline & Tool_Web & Tool_Dynamic --> CRAG{"Evidence Relevance Gate (CRAG)<br/>(Fast-Floor <5ms + LLM Judge)"}
        
        CRAG -->|Sufficient Grounding| Synthesizer["AnswerSynthesizer<br/>(Blueprint-Driven Grounded Brief)"]
        CRAG -->|Zero Evidence / Analytical Query| ToolMaker["LLM Tool Maker<br/>(Ad-Hoc Tool Synthesis)"]
        ToolMaker --> Tool_Dynamic
        Tool_Dynamic --> ToolCritic["ToolCritic (5-Metric Audit)<br/>(SASC, SRF, REH, DSF, RPS)"]
        ToolCritic -->|"Pass (Score >= 0.70)"| CRAG
        ToolCritic -->|Defect / NaN Detected| ToolMaker
        CRAG -->|Zero Evidence / Ambiguous| FallbackRouter["Fallback Web/Entity Search<br/>or Anti-Hallucination Notice"]
        FallbackRouter --> Synthesizer

        Synthesizer --> Verifier["Reflective Answer Verifier<br/>(4-Dimension Audit & Fast Scope Gates)"]
        Verifier -->|Evidence Gap Detected| ToolMaker
        Verifier -->|Verified / Refined| SSE["FastAPI SSE Streaming Router<br/>(Stages, Thoughts, Tokens, Visual Cards)"]
    end

    CLIENT <-->|HTTP REST & SSE Events| AGENTIC
```

---

## 2. Dynamic Model Provider Registry & Sovereign / Cloud Architecture

NewsLens-AI decouples application pipelines from hardcoded AI vendors using a **Hot-Swappable Provider Registry Architecture** ([`backend/app/providers/registry.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/registry.py)) managed by [`model_config.yaml`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/model_config.yaml) and the frontend **Model Settings Studio** ([`frontend/src/components/ModelSettingsStudio.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/ModelSettingsStudio.jsx)).

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DYNAMIC MODEL PROVIDER REGISTRY                                 │
├──────────────────────────────┬───────────────────────────────┬──────────────────────────────────┤
│ Tier 1: Local Sovereign      │ Tier 2: Google Gemini Cloud   │ Tier 3: Multi-Provider Gateways │
│ (100% On-Premise / Offline)  │ (Google AI Studio Primary)    │ (Commercial Gateways & Fallbacks)│
├──────────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
│ • Ollama Llama 3.1 8B        │ • Google Gemini 2.5 Flash     │ • OpenAI GPT-4o & GPT-4o-mini    │
│ • Ollama DeepSeek R1 14B     │   (Workhorse VLM, Plan, Synth)│ • OpenRouter (Gemma 4, Nemotron) │
│ • Ollama Qwen 2.5 VL / 3 VL  │ • Google Gemini 2.5 Pro       │ • NVIDIA NIM Catalog             │
│ • Docling Layout + RapidOCR  │   (Frontier Reasoning/Synthesis│ • Google Cloud Vision OCR        │
│ • BAAI/bge-m3 (1024d Dense)  │ • Google Gemini 2.5 Flash-Lite│ • Text-Embedding-3-Large         │
│                              │ • Transparent Model Failover  │ • HTTP 429 Cooldown Circuit Breaker
└──────────────────────────────┴───────────────────────────────┴──────────────────────────────────┘
```

### Granular Pipeline Task Bindings
1. **Stage 1 — Agentic Reasoning & Synthesis**:
   - `query_planner`: Autonomous tool sequence planner & sub-query generator (`gemini-3.8-flash` / `ollama_llama3`).
   - `answerer`: Multi-newspaper factual synthesizer & citation linker (`gemini-3.8-flash` / `ollama_llama3`).
   - `answer_verifier`: Reflective fact-checking critic and fluff eliminator (`gemini-3.8-flash` / `ollama_llama3`).
   - `query_condenser`: Coreference and pronoun resolution (`gemini-3.8-flash` / `ollama_llama3`).
2. **Stage 2 — Vision & Broadsheet Ingestion**:
   - `visual_extraction`: Multimodal chart, table, and scene extractor (`gemini-3.8-flash` / `ollama_qwen3vl`).
   - `layout_analysis`: 2D spatial layout and column parsing (`gemini-3.8-flash` / `docling_parser`).
   - `document_parser`: Broadsheet hierarchy structure extractor (`docling_parser`).
   - `ocr`: Character transcription engine (`rapidocr` / `docling`).
3. **Stage 3 — Classification & Indexing**:
   - `embedding`: 1024-dimensional dense vector generator (`local_embed_bge` - BAAI/bge-m3).
   - `article_segmentation`: Complex multi-column jump-line stitcher (`gemini-3.8-flash` / `ollama_deepseek`).
   - `classification`: 12-domain probabilistic categorization (`gemini-3.8-flash` / `ollama_llama3`).
   - `metadata_extraction`: Publication, edition, and date extractor (`gemini-3.8-flash` / `ollama_llama3`).

---

## 3. Document Intake & Broadsheet Ingestion Pipeline Data Flow

The ingestion pipeline processes complex 2D newspaper broadsheet scans through six sequential phases:

```
[ Broadsheet PDF / Archive Upload ]
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Intake, Compression & SHA-256 Idempotency                     │
│ • Compress raw PDF with Ghostscript / fitz.deflate                     │
│ • Calculate SHA-256 hash; verify against `issues` table                │
│ • Upload raw PDF to MinIO bucket `newslens-originals`                  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Visual Masthead Verifier & Multi-Page Folio Consensus         │
│ • Crop top 22% of Page 1; run RapidOCR ONNX (<0.6s)                    │
│ • Normalize Unicode superscripts (e.g. ²⁷⁰⁸²⁰²⁶ ➔ 27082026)            │
│ • Evaluate broadsheet brand patterns against dynamic registry          │
│ • Run multi-page folio consensus (5x header-zone weight over Pages 1-15)│
│ • Create/update `Issue` record in MySQL (newspaper_id, issue_date)     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: 300 DPI High-Res Rasterization & Digital Triage               │
│ • PyMuPDF renders 300 DPI high-resolution PNGs (fitz.Matrix(300/72))   │
│ • Upload page rasters to MinIO bucket `newslens-pages`                 │
│ • PDF Page Detector evaluates text density, vector lines, scanned print│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 4: 5-Pass Spatial Layout Analysis & 2D Article Segmentation      │
│ • Pass 0: Drop-cap initial re-attachment & font ligature repair        │
│ • Pass 1: Vertical paragraph stitching within column tracks            │
│ • Pass 2: Horizontal multi-column headline slice merging               │
│ • Pass 3: Statutory ad-envelope boundary wall detection                │
│ • Pass 4: Cross-page jump-line stitching ("Continued on Page 4")       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         │                                                   │
         ▼                                                   ▼
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│ Phase 5A: Text Assembly & Order      │    │ Phase 5B: Visual Asset Harvesting    │
│ • 2D Reading Order Graph (x, y, col) │    │ • Crop photos, logos, charts, tables │
│ • Column de-bundling (Shorts/Briefs) │    │ • Spatial containment binding        │
│ • Kicker extraction & bylines        │    │ • Visual triage (Photo vs Data Chart)│
│ • Cross-page jump-line stitching     │    │ • Dual VLM + Spatial OCR Matrix      │
└──────────────────┬───────────────────┘    └──────────────────┬───────────────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 6: Probabilistic 12-Domain Classification & Contextual Chunking  │
│ • Weighted scoring: Headline (3x), Subheadline (2x), Body text (1x)    │
│ • Domain Context Anchor Dampening for financial/political metaphors    │
│ • Secondary Topic Extraction; persist in `Topic` & `ArticleTopic`      │
│ • Insert `Article`, `ArticlePage`, `Photo` in MySQL 8                  │
│ • Contextual chunking: Prepend [Newspaper|Date|Sec|Headline|Pages]     │
│ • Embed via BAAI/bge-m3 (1024-dim dense); upsert into Qdrant           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Visual Asset Intelligence, Multimodal VLM & Failover Data Flow

Broadsheets embed crucial quantitative intelligence inside tables, stock charts, and infographics. NewsLens-AI handles visual data through a resilient 3-stage pipeline with active circuit breaking:

```mermaid
flowchart TD
    AssetCrop["Raw Visual Asset Crop<br/>(from 300 DPI Broadsheet Page)"] --> Stage1{"Stage 1: Fast Visual Triage Gate<br/>(PIL Heuristics + Number Density)"}
    
    Stage1 -->|"Dim < 80px or Aspect > 10:1"| Decorative["Filter as Decorative Divider / Icon"]
    Stage1 -->|Data Density / Chart Features| DataCandidate["Data-Bearing Candidate<br/>(data_chart, table, infographic)"]
    Stage1 -->|Photographic Texture| PhotoCandidate["Editorial Photo Candidate<br/>(scene, portrait, ceremony)"]

    DataCandidate --> VLM_Dispatch{"Resolve Vision Provider<br/>(_get_provider)"}
    PhotoCandidate --> VLM_Photo_Dispatch{"Resolve Vision Provider<br/>(_get_provider)"}

    VLM_Dispatch -->|Circuit Breaker Open| SecondaryVLM["Secondary Fallback VLM<br/>(Priority: Google Cloud Vision / Ollama Qwen 3 VL)"]
    VLM_Dispatch -->|Healthy Primary| PrimaryVLM["Primary VLM Provider<br/>(Google Gemini 2.5 Flash / Pro)"]

    PrimaryVLM -->|HTTP 429 / RateLimitExhausted| TripBreaker["Trip Circuit Breaker (60s Cooldown)<br/>Immediate Secondary Failover"]
    TripBreaker --> SecondaryVLM

    PrimaryVLM & SecondaryVLM -->|JSON Markdown Table Returned| Stage3["Stage 3: Numerical Cross-Validation<br/>(VLM Numbers vs OCR Spatial Tokens)"]
    PrimaryVLM & SecondaryVLM -->|Conversational Text| RegexRecovery["Regex Markdown Table Recovery"]
    RegexRecovery --> Stage3

    PrimaryVLM & SecondaryVLM -->|Failure / Timeout / Offline| SpatialOCR["Deterministic Spatial OCR Matrix Engine<br/>(PyTesseract Token BBoxes ➔ Markdown Table)"]
    SpatialOCR --> Stage3

    Stage3 --> VisualChunk["Create Dedicated Visual ArticleChunk<br/>chunk_type='visual', has_visual_data=True"]
    VisualChunk --> QdrantIndex[("Qdrant Vector DB<br/>Embedded via BAAI/bge-m3")]
    VisualChunk --> MySQLIndex[("MySQL 8 `photos` & `tables`<br/>vlm_description, markdown_table")]
```

### On-Demand Agent Trigger vs. Ingestion Extraction
- **Ingestion Time**: High-priority data charts and tables receive initial transcription.
- **On-Demand Query Time (`inspect_visual_asset`)**: When an agent query or attached asset (`attached_photo_id`) targets a photo where `vlm_description` is a placeholder, the system streams raw crop bytes from MinIO, executes on-demand VLM transcription, and caches the result back into MySQL.

---

## 5. Relational Knowledge Graph & Chronological Storyline Trajectories

```mermaid
graph LR
    subgraph KNOWLEDGE_GRAPH ["Entity Co-Occurrence Knowledge Graph"]
        E1(("Entity: HAL<br/>(Organization)"))
        E2(("Entity: Safran<br/>(Organization)"))
        E3(("Entity: SAFHAL Helicopter Engine<br/>(Product / Defense)"))
        E4(("Entity: Ministry of Defence<br/>(Government)"))
        
        E1 ---|"co-occurs (weight: 12)"| E2
        E2 ---|developed_product| E3
        E1 ---|manufactures| E3
        E1 ---|procurement_contract| E4
    end

    subgraph ARTICLE_NODES ["Article Evidence References"]
        A1["Article #42101<br/>'HAL, Safran ink engine pact'<br/>Mint (Page 1)"]
        A2["Article #42188<br/>'Defence procurement cleared'<br/>The Hindu (Page 5)"]
    end

    E1 -.->|mentioned in| A1
    E2 -.->|mentioned in| A1
    E3 -.->|subject of| A1
    E4 -.->|mentioned in| A2
```

### Storyline Trajectory Construction
1. Articles mentioning an entity or theme are clustered across publication dates.
2. The `TimelineBuilder` calculates narrative arcs, milestone events, and prominence scores.
3. Chronological milestone graphs are cached in Redis with a 1-hour TTL and rendered on the client as an interactive visual storyline canvas.

---

## 6. Multi-Tier Storage Layer Architecture & Data Lifecycle Matrix

| Layer | Component | Engine / Driver | Stored Data & Schema | Access Patterns & Indexing |
|---|---|---|---|---|
| **System of Record** | Relational Database | **MySQL 8** (`aiomysql` / SQLAlchemy 2) | • `newspapers`, `issues`, `pages`<br/>• `articles`, `article_pages`<br/>• `photos`, `tables`<br/>• `entities`, `article_entities`<br/>• `topics`, `article_topics`<br/>• `query_log`, `ingestion_jobs` | • Foreign keys & relational joins<br/>• `FULLTEXT(headline, full_text)`<br/>• B-tree indexes on `(newspaper_id, issue_date)`<br/>• Sub-5ms metadata queries |
| **Vector Store** | Dense Vector DB | **Qdrant** (`qdrant-client`) | • Collection: `article_chunks`<br/>• 1024-dim dense vectors (`BAAI/bge-m3`)<br/>• Payload: `article_id`, `issue_id`, `newspaper_name`, `issue_date`, `page_number`, `headline`, `section`, `has_visual_data`, `bboxes` | • Cosine similarity search (HNSW index)<br/>• Payload pre-filtering on `newspaper_name`, `issue_date`, `section`<br/>• Sub-15ms vector retrieval |
| **Object Store** | S3-Compatible Blob Store | **MinIO** (`minio-py`) | • Bucket `newslens-originals`: Raw source PDFs<br/>• Bucket `newslens-pages`: 300 DPI high-res page rasters<br/>• Cropped visual assets & chart PNGs | • High-throughput binary streaming<br/>• Public thumbnail HTTP endpoints (`/api/photos/{id}/image`)<br/>• Immutable asset storage |
| **In-Memory Cache** | Key-Value & Queue | **Redis 7** (`redis-py`) | • Celery background worker task queue<br/>• Query response cache (TTL: 1h)<br/>• Condensed query hash cache<br/>• Timeline trajectory cache<br/>• SSE Pub/Sub channels | • In-memory sub-millisecond lookups<br/>• Distributed task locks (`redis-lock`)<br/>• Automatic TTL expiration (1h to 24h) |

---

## 7. Conversational Agent Query Lifecycle & LangGraph Execution Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User as User Client (React SPA)
    participant Cache as Redis Cache
    participant API as FastAPI Router (/api/query/stream)
    participant Condenser as Conversational Query Condenser
    participant Graph as LangGraph State Machine
    participant Planner as Cognitive Query Planner
    participant Dispatcher as Tool Execution Engine
    participant DB as MySQL & Qdrant Vector DB
    participant ToolMaker as LLM Tool Maker
    participant Sandbox as Subprocess Sandbox
    participant Critic as ToolCritic (5-Metric Audit)
    participant CRAG as Corrective RAG (CRAG) Evaluator
    participant Synth as AnswerSynthesizer
    participant Verifier as Reflective Answer Verifier
    participant SSE as SSE Streaming Output

    User->>API: POST /api/query/stream (query, history, attached_asset_ids)
    API->>Cache: Check Query Cache Key
    alt Cache Hit
        Cache-->>User: Stream Cached Response (<5ms)
    else Cache Miss
        API->>Condenser: condense_conversational_query()
        Note over Condenser: Resolves pronouns & citations [Paper, Date, Page]<br/>Applies 3 Anti-Leakage Guardrails<br/>Preserves differential exclusion ("In X but not in Y")
        Condenser-->>API: Condensed Query + Active Filter State
        
        API->>Graph: AgentWorkflow.run(condensed_query)
        Graph->>Planner: plan_query_async() with Live Schema & Metadata
        Note over Planner: Enforces Strict Operational Contracts<br/>Resolves 1 of 7 Archetypes<br/>Compiles AnswerBlueprint (SectionSpecs)
        Planner-->>Graph: Execution Plan (archetype, tool_calls, parameters)

        Graph->>Dispatcher: Execute Planned Tools Concurrently
        par Concurrent Tool Invocations
            Dispatcher->>DB: hybrid_search (Qdrant Cosine + MySQL BM25 + RRF + Cross-Encoder)
            Dispatcher->>DB: inspect_visual_asset (5-Tier Cascade A-E & MinIO Crops)
            Dispatcher->>DB: sql_analytics (Strict 7-Enum Routines)
            Dispatcher->>DB: entity_search (Knowledge Graph & Co-occurrences)
            Dispatcher->>DB: timeline_builder (Chronological Milestone Clustering)
            Dispatcher->>DB: coverage_analysis (Multi-Broadsheet Negative Audit)
            Dispatcher->>DB: web_search (4-Tier Grounding: NewsData ➔ Serper ➔ Tavily ➔ DDG)
            Dispatcher->>ToolMaker: dynamic_analysis (Ad-hoc Python/SQL Synthesis)
        end

        opt Dynamic Tool Execution & Closed-Loop Critic
            ToolMaker->>Sandbox: Execute in Subprocess AST Sandbox (512MB RAM, 15s timeout)
            Sandbox->>DB: Read-Only DB Execution (Autocommit Disabled)
            DB-->>Sandbox: Raw Query Records & Aggregates
            Sandbox-->>Critic: Execution Result
            Critic->>Critic: Evaluate 5 Metrics (SASC, SRF, REH, DSF, RPS)
            alt Audit Fails (Score < 0.70)
                Critic-->>ToolMaker: Diagnostic Critique & Fix Hints (Up to 3 Retries)
                ToolMaker->>Sandbox: Re-execute Refined Script
            else Audit Passes (Score >= 0.70)
                Critic-->>Dispatcher: Verified High-Confidence Evidence
            end
        end

        Dispatcher-->>Graph: Aggregated Raw Evidence Items
        Graph->>CRAG: evaluate_evidence()
        Note over CRAG: Fast-Floor Check (<5ms for >=100 words editorial text)<br/>Reflexive LLM-as-Judge emits typed EvaluationVerdict
        alt CRAG Action: replan_static_tools
            CRAG->>Planner: replan_with_feedback_async()<br/>(Relax filters, expand top_k = max(8, k+4))
            Planner->>Dispatcher: Execute Reformed Tool Sequence
            Dispatcher-->>Graph: Injected Recovery Evidence
        else CRAG Action: synthesize_dynamic_tool
            CRAG->>ToolMaker: Synthesize Custom Analysis Tool
            ToolMaker->>Sandbox: Execute in AST Sandbox
            Sandbox-->>Critic: 5-Metric Scorecard Audit
            Critic-->>Graph: Injected High-Confidence Evidence
        end

        Graph->>Synth: synthesize_stream(query, evidence, archetype, blueprint)
        Note over Synth: Compiles prompt from AnswerBlueprint<br/>Allocates full parent text (up to 7,500 chars)<br/>Strips robotic catalog tables for narrative reading<br/>Enforces strict citation format: [Paper, Date, Page, Headline]
        Synth-->>Verifier: Draft Answer + Evidence Ground Truth
        
        Verifier->>Verifier: verify_answer_async()
        Note over Verifier: Fast Gates (<5ms): scope check, date alignment, absence<br/>Reflexive LLM Audit (Faithfulness, Fluff, Contradiction, Gaps)
        alt Verifier Verdict: fallback_to_dynamic_tool
            Verifier->>Graph: Loop back to execute_dynamic_code (1-cycle ceiling)
            Graph->>ToolMaker: Execute dynamic Python/SQL query for missing data
            ToolMaker-->>Synth: Injected Verified Dynamic Telemetry
            Synth->>Synth: Re-Synthesize Final Grounded Brief
        else Verifier Verdict: accept / refine_answer
            Verifier-->>Synth: Verified / Refined Response Brief
        end

        Synth->>SSE: Stream SSE Events (stage, thought, token, citations, done)
        SSE-->>User: Real-Time Markdown Stream + Provenance Citation Cards + Photo Thumbnails
        Synth->>DB: Persist Query Audit Log in `query_log`
        Synth->>Cache: Cache Result in Redis (TTL: 1 Hour)
    end
```

---

## 8. Query Planner & Dynamic Answer Blueprint Data Flow

```mermaid
flowchart TD
    subgraph INPUT_CONTEXT ["1. User Input & Conversational Context"]
        RawQuery["Raw User Query String"]
        ChatHistory["Multi-Turn Chat History (Redis)"]
        ActiveAssets["Active Attached Asset IDs<br/>(article_id, photo_id)"]
        RawQuery & ChatHistory & ActiveAssets --> Condenser["Conversational Query Condenser<br/>(backend/app/agent/condenser.py)"]
        Condenser --> CleanQuery["Condensed Query<br/>+ Sanitized Active Context"]
    end

    subgraph EXTRACTION ["2. Deterministic Semantic Extraction"]
        CleanQuery --> ParamExtractor["Semantic Parameter Extractor<br/>(backend/app/agent/extractor.py)"]
        ParamExtractor --> ExtractedEntities["• Publication Brand Patterns<br/>• ISO Dates & Date Ranges<br/>• Page Numbers & Sections<br/>• Differential / Shared Flags"]
    end

    subgraph SCHEMA_GROUNDING ["3. Static Schema & Decoupled Archive Bounds"]
        StaticSchema["STATIC_BROADSHEET_SCHEMA<br/>(Declarative Tables & Columns)"]
        ArchiveCache[("MySQL Archive Metadata<br/>5-Minute TTL In-Memory Cache")]
        OfflineFallback["Deterministic Fallback Bounds<br/>(Zero DB Network Latency)"]
        
        ArchiveCache -->|Healthy DB| GroundedBounds["Resolved Archive Bounds<br/>(min_date, max_date, publications)"]
        ArchiveCache -.->|DB Offline / CI| OfflineFallback --> GroundedBounds
        StaticSchema & GroundedBounds --> PlannerContext["Consolidated Planner System Prompt Context"]
    end

    subgraph PLANNER_ENGINE ["4. LLM Direct Tool Sequence Planner"]
        CleanQuery & ExtractedEntities & PlannerContext --> LLM_Planner["QueryPlanner.plan_query_async()<br/>(Gemini 2.5 Flash / Candidate Failover)"]
        
        LLM_Planner --> Contracts{"Enforce Strict Operational Contracts"}
        Contracts -->|sql_analytics| Contract_SQL["Strict 7-Enum Routine Contract<br/>(count_*, issue_summary, diff, shared)"]
        Contracts -->|dynamic_analysis| Contract_Dyn["Math, Aggregations, Ratios, Joins<br/>(Barred from narrative reading)"]
        Contracts -->|hybrid_search| Contract_Hybrid["Factual Excerpts & Dynamic Top-K (4-12)"]
        Contracts -->|inspect_visual| Contract_Visual["Multimodal Visual Crop Analysis"]
        
        Contracts --> RawPlan["Raw LLM Plan Candidate<br/>(Archetype, Tool Calls, Arguments)"]
    end

    subgraph BLUEPRINT_SELECTION ["5. Declarative Answer Blueprint Selection"]
        RawPlan --> BlueprintSelector{"Select Blueprint by Archetype"}
        BlueprintSelector --> B_Fact["factual_lookup Blueprint"]
        BlueprintSelector --> B_Comp["cross_newspaper_comparison Blueprint"]
        BlueprintSelector --> B_Cat["article_catalog Blueprint"]
        BlueprintSelector --> B_Time["thematic_timeline Blueprint"]
        BlueprintSelector --> B_Quant["scalar_count / trend Blueprint"]
        BlueprintSelector --> B_Dyn["analytical_computation Blueprint"]
        
        B_Fact & B_Comp & B_Cat & B_Time & B_Quant & B_Dyn --> CompiledBlueprint["Compiled AnswerBlueprint<br/>(Ordered SectionSpecs, Format Constraints, Word Count)"]
    end

    subgraph RECONCILIATION ["6. Tool Call Reconciliation & Pruning"]
        RawPlan & ExtractedEntities --> Reconciler["reconcile_and_sanitize_arguments()<br/>(backend/app/agent/tool_factory.py)"]
        Reconciler --> Clean1["Sanitize Generic Filler Queries"]
        Reconciler --> Clean2["Normalize Newspaper Brand Abbreviations"]
        Reconciler --> Clean3["Harmonize ISO Dates & Precedence"]
        Reconciler --> Clean4["Resolve Attached Asset vs Query Date Conflict"]
        
        Clean1 & Clean2 & Clean3 & Clean4 --> SanitizedPlan["Validated PlannedToolCall Sequence<br/>(1 to 3 Concurrent Calls)"]
    end

    subgraph DISPATCH ["7. LangGraph Execution Node"]
        SanitizedPlan & CompiledBlueprint --> ToolExecutorNode["ToolExecutor.execute_tools()<br/>(backend/app/agent/executor.py)"]
        ToolExecutorNode --> ConcurrentExec["Concurrent Async Execution<br/>(Qdrant, MySQL, MinIO, VLM, Sandbox)"]
    end

    subgraph REPLAN_LOOP ["8. Closed-Loop Adaptive Re-Planning"]
        CRAG_Gap["CRAG Evaluator Feedback<br/>(EvaluationVerdict.gap_diagnosis)"] -.->|Trigger Re-Plan| ReplanEngine["replan_with_feedback_async()<br/>• Relax Page / Category Constraints<br/>• Scale top_k = max(8, top_k + 4)<br/>• Anti-Repetition Guard<br/>• 1-Cycle Hard Ceiling"]
        ReplanEngine -.-> SanitizedPlan
    end
```

### The 7 Core Broadsheet Query Archetypes & Routing Matrix

| Archetype | Journalistic Intent | Primary Planned Tools | Typical Parameter Payload | Downstream Output Format |
|---|---|---|---|---|
| **`factual_lookup`** | Point-in-time facts, quotes, event details, or visual checks | `hybrid_search` (primary), `inspect_visual_asset` (if visual) | `query`, `newspaper_name`, `date_from`, `date_to`, `page_filter`, `top_k=6` | Direct narrative findings + bulleted operational details + citations |
| **`article_catalog`** | Whole-issue manifests, section catalogs, front-page listings | `sql_analytics` (`analysis_type="issue_summary"`) | `newspaper_name`, `issue_date`, `category_filter`, `page_filter` | Markdown manifest table (`#`, Headline, Page, Section, Byline) + featured highlights |
| **`cross_newspaper_comparison`** | Multi-broadsheet coverage diffs, exclusive stories, or shared wire news | `sql_analytics` (`coverage_difference` or `shared_coverage`) + `hybrid_search` | `newspaper_name`, `comparison_newspaper`, `issue_date`, `query`, `top_k=12` | Cross-newspaper comparison matrix table + editorial framing divergence bullets |
| **`thematic_timeline`** | Chronological evolution of developing storylines across editions | `timeline_builder` (primary), `hybrid_search` (corroborating) | `query`, `limit=25`, `top_k=8` | Chronological dated milestone timeline + trajectory narrative |
| **`quantitative_trend`** | Volume metrics, publication rosters, issue counts, ad counts | `sql_analytics` (`count_issues`, `count_articles`, `count_advertisements`, `count_photos`) | `analysis_type`, `newspaper_name`, `date_from`, `date_to`, `issue_date` | Direct authoritative findings + compact metric cards |
| **`entity_deep_dive`** | Multi-hop relational entity networks, corporate profiles, salience | `entity_search` (primary), `hybrid_search` (corroborating) | `entity_name`, `top_k=10` | Executive profile + corporate actions bullets + media scrutiny narrative |
| **`negative_coverage_audit`** | Verifying silence or absence of coverage across the broadsheet archive | `coverage_analysis` (primary), `sql_analytics` (secondary) | `query`, `target_date`, `newspaper_name` | 3-tier coverage matrix table (Primary, Corroborating, Omission verdict) |
| **`analytical_computation`** *(Special)* | Ad-hoc statistics, averages, length distributions, ratios, joins | `dynamic_analysis` (synthesized Python/SQL sandbox) | `query`, `analysis_description` | Computed mathematical metrics + verified tabular statistics |

### Strict Operational Contracts
- **`sql_analytics` Contract**: Bound strictly to 7 immutable routines. Cannot execute arbitrary SQL; cannot compute averages or ratios. Unsupported analytics hand off directly to `dynamic_analysis`.
- **`dynamic_analysis` Contract**: Bound to mathematical aggregations, averages, and multi-table joins. Barred from narrative reading where word count is an answer length constraint (`is_dynamic_analysis_permitted`).

---

## 9. 4-Tier Journalistic Web Search Grounding Data Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                  USER QUERY REQUIRING LIVE WEB GROUNDING               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 1: NewsData.io Accredited News API (Primary Journalistic Engine)  │
│ • Queries accredited global and national press agencies                │
│ • Parameters: `q={query}`, `country=in`, `language=en`, `category`     │
│ • Delivers structured publisher `source_name`, `pubDate`, and deep url │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │ (Success: >= 1 Article)   │ (Empty / Rate Limited / Error)
                     ▼                           ▼
       ┌───────────────────────────┐ ┌───────────────────────────────────┐
       │ Structured News Evidence  │ │ Tier 2: Serper API (Google Search)│
       └───────────────────────────┘ └─────────────────┬─────────────────┘
                                                       │
                                         ┌─────────────┴─────────────┐
                                         │ (Success)                 │ (Empty / Error)
                                         ▼                           ▼
                           ┌───────────────────────────┐ ┌───────────────────────┐
                           │ Google SERP Snippets      │ │ Tier 3: Tavily API    │
                           └───────────────────────────┘ └───────────┬───────────┘
                                                                     │
                                                       ┌─────────────┴───────────┐
                                                       │ (Success)               │ (Empty / Error)
                                                       ▼                         ▼
                                         ┌─────────────────────────┐ ┌───────────────────┐
                                         │ Tavily Research Context │ │ Tier 4: DuckDuckGo│
                                         └─────────────────────────┘ └───────────────────┘
```

| Search Tier | Provider | Primary Strength | Metadata Extracted |
|---|---|---|---|
| **Tier 1 (Primary)** | **NewsData.io** (`newsdataapi`) | Accredited broadsheets & news agencies (Reuters, Mint, The Hindu, PTI) | `title`, `source_name`, `pubDate`, `link`, `description`, `category` |
| **Tier 2 (Fallback)** | **Serper API** | High-index Google Web Search results | `title`, `snippet`, `link`, `date` |
| **Tier 3 (Fallback)** | **Tavily Search** | AI agent research engine optimized for RAG synthesis | `title`, `content`, `url`, `score` |
| **Tier 4 (Offline)** | **DuckDuckGo** | Zero-credential, zero-cost fallback HTML scraper | `title`, `snippet`, `link` |

---

## 10. Broadsheet Reader to Agent Visual Attachment Data Flow

When a user reads a newspaper in the interactive **Broadsheet Reader** and encounters a complex visual graphic, chart, or photo, they can route that specific asset directly into the **Agent Assistant** for deep multimodal inquiry.

```mermaid
sequenceDiagram
    autonumber
    actor Reader as User / Broadsheet Reader (React)
    participant State as Global Reader State
    participant Assistant as Agent Assistant Component
    participant API as FastAPI Backend (/api/query/stream)
    participant Condenser as Conversational Condenser
    participant Executor as Concurrent Tool Executor
    participant VLM as VLM Visual Data Extractor (MinIO)
    participant DB as MySQL Database

    Reader->>State: User clicks "Ask Agent About This Infographic / Photo"
    State->>Assistant: Open Assistant Panel with attachedAsset payload
    Note over Assistant: Renders attached asset banner<br/>with thumbnail, headline & date
    Reader->>Assistant: Submits query (e.g. "Explain the GDP projections in this chart")
    Assistant->>API: POST /api/query/stream<br/>{query, attached_article_id, attached_photo_id}
    API->>Condenser: Pass query, chat history, and attached IDs
    Condenser->>Condenser: Bind attached IDs to active turn -<br/>Purge on subsequent unrelated turns
    Condenser->>Executor: Plan tool execution with Strategy A/C (photo_id/article_id)
    Executor->>DB: Query Photo record by photo_id or article_id
    alt vlm_description is Placeholder
        Executor->>VLM: Stream raw image crop from MinIO bucket_pages
        VLM->>VLM: Run Multimodal VLM OCR & tabular synthesis
        VLM->>DB: Persist synthesized vlm_description to article_photos
    end
    DB-->>Executor: Complete visual description, caption, and metadata
    Executor-->>API: Synthesize evidence including visual asset
    API-->>Assistant: Stream response with is_visual_asset citation & /api/photos/{id}/image
    Assistant-->>Reader: Render rich response with interactive visual card thumbnail
```

### Lifecycle & Anti-Leakage Guardrails
1. **Interactive Attachment Banner**: `AgentAssistant.jsx` displays an active blue attachment chip above chat input with headline, thumbnail, date, and dismiss button.
2. **First-Turn Binding**: `attached_article_id` and `attached_photo_id` bind to tool arguments (`photo_id`, `article_id`).
3. **Cross-Turn Parameter Purge**: On subsequent turns, if the user asks an unrelated question or switches topics, Guardrails 1, 2, and 3 immediately purge visual parameters, preventing cross-turn context contamination.
4. **Cross-Date Asset Conflict Eviction**: If the user's query explicitly specifies a publication date that conflicts with the attached asset's date, the attached asset is evicted. In `executor.py`, `effective_date = explicit_query_date or asset_date` guarantees that attached assets can never overwrite explicit query dates.

---

## 11. Dynamic Tool Synthesis, Subprocess AST Sandbox & Closed-Loop ToolCritic Flow

When broadsheet analytical queries require bespoke aggregations, variances, or relational calculations that exceed predefined static tools, NewsLens-AI dynamically synthesizes, audits, and executes custom Python functions in a secure, sandboxed subprocess with automated self-refinement.

```mermaid
sequenceDiagram
    autonumber
    participant Graph as LangGraph Engine
    participant Maker as ToolMaker
    participant LLM as LLM Code Synthesizer
    participant Sandbox as Subprocess Sandbox
    participant Critic as ToolCritic

    Graph->>Maker: generate_and_execute(query, context, gap_diagnosis)
    Maker->>LLM: Prompt with STATIC_BROADSHEET_SCHEMA & Pre-Normalized Context
    LLM-->>Maker: Synthesized Python Script (analyze(db, query, context))
    Maker->>Maker: Auto-Import Pre-Injection (re, math, statistics, pd, np, text)
    Maker->>Critic: Pre-Execution AST & Schema Audit (SASC, SRF)
    
    alt Schema Flaw Detected (e.g. Cartesian join or bad column)
        Critic-->>Maker: Scorecard: Rejected (SRF < 0.70, suggested_fixes=[...])
        Maker->>LLM: Re-prompt with Diagnostic Critique & Fixes
        LLM-->>Maker: Corrected Python Script
    end

    Maker->>Sandbox: Execute in Isolated Subprocess (15s, 512MB RAM, Read-Only DB)
    Sandbox-->>Maker: Execution Result: {data: [...], metadata: {...}, summary: '...'}
    
    Maker->>Critic: Post-Execution Audit (REH, DSF, RPS)
    alt Calculation Error or NaN Detected
        Critic-->>Maker: Scorecard: Rejected (DSF=0.0, NaN detected)
        Maker->>LLM: Re-prompt with NaN Diagnostic Critique
        LLM-->>Maker: Refined Script handling empty rows gracefully
        Maker->>Sandbox: Re-execute in Subprocess
        Sandbox-->>Maker: Verified Grounded Result
    end

    Critic-->>Maker: Scorecard: Accepted (Score >= 0.70)
    Maker-->>Graph: Grounded Structured Evidence Items (Score: 1.0)
```

### AST Whitelist and Sandbox Isolation Parameters
- **AST Safety Scanner (`ASTSafetyScanner`)**: Blocks unauthorized modules (`os`, `sys`, `subprocess`, `shutil`, `socket`, `urllib`, `requests`, `pathlib`) and dangerous built-ins (`eval`, `exec`, `open`, `compile`, `__import__`, `globals`, `locals`).
- **Subprocess Sandbox (`sandbox_runner.py`)**: Spawns isolated worker via `subprocess.Popen([sys.executable])`, enforcing 15-second timeout, 512MB RAM ceiling, autocommit disabled, and unconditional `connection.rollback()`.
- **ToolCritic 5-Metric Scorecard**:
  1. **SASC**: Syntactic & AST Security Compliance.
  2. **SRF**: SQL Schema Fidelity (remaps hallucinated columns, requires `DISTINCT` on multi-table joins).
  3. **REH**: Runtime Subprocess Health (zero exit code, no exceptions).
  4. **DSF**: Data-to-Summary Faithfulness (legitimate absence vs. narrative hallucination; accepts aggregate tables).
  5. **RPS**: Intent Alignment & Filter Plausibility (ISO date validation).

---

## 12. Corrective RAG (CRAG), Answer Verification & Streaming Delivery Flow

The post-retrieval verification pipeline in NewsLens-AI ensures that every synthesized brief is factually accurate, citation-grounded, and free of hallucinations before reaching the user. It operates as a two-stage closed-loop quality gate:
1. **Corrective RAG (CRAG) Evidence Evaluation (`evaluator.py`)**: Intercepts retrieved tool evidence before synthesis. It evaluates qualitative and quantitative sufficiency, catches legitimate archival absences, and dynamically routes deficient queries into adaptive static re-planning or sandboxed ad-hoc tool synthesis.
2. **Reflective LLM Answer Verification (`answer_verifier.py`)**: Audits the synthesized draft response against raw database ground truth. It applies sub-5ms deterministic scope and math checks, followed by an LLM-as-judge fact-checker that strips fluff, refines ungrounded statements, or triggers a dynamic tool rollback loop.

### 12.1. CRAG Evidence Evaluation & Adaptive Fallback Routing Flowchart

```mermaid
flowchart TD
    subgraph INPUT ["1. Evidence Ingestion"]
        EvidenceIn["Retrieved Evidence Items from ToolExecutor<br/>(ToolExecutionRecord Collection)"]
    end

    subgraph FAST_FLOOR ["2. Deterministic Fast-Floor Gate (<5ms)"]
        FastFloor{"Fast-Floor Evaluation Check:<br/>• >= 1 Broadsheet Record?<br/>• Clean Editorial Text >= 100 Words?<br/>• Relevance Prominence >= 0.65?"}
        ImmediatePass["High-Confidence Immediate Bypass<br/>(is_sufficient=True, quality_score=1.0)<br/>Skip LLM Evaluator Latency Overhead"]
    end

    subgraph LLM_JUDGE ["3. Reflexive LLM-as-Judge Evaluation (evaluator.py)"]
        JudgePrompt["EvidenceEvaluator (LLM Judge)<br/>Audits Evidence Sufficiency & Gaps"]
        
        CheckAbsence{"Check 1: Legitimate Absence Invariant?<br/>(Availability query with 0 DB issues?)"}
        PassAbsence["Verdict: Sufficient (Score: 0.85)<br/>Proceed with Truthful Absence Grounding"]
        
        CheckBalance{"Check 2: Multi-Newspaper Balance?<br/>(Missing requested publication brand?)"}
        FailBalance["Verdict: Insufficient (Score: 0.35)<br/>Action: replan_static_tools<br/>Hint: missing_newspapers"]
        
        CheckDate{"Check 3: Temporal ISO Date Alignment?<br/>(Target date missing from evidence?)"}
        FailDate["Verdict: Insufficient (Score: 0.30)<br/>Action: replan_static_tools<br/>Hint: issue_date"]
        
        CheckQuant{"Check 4: Quantitative Payload Verification?<br/>(Math/volume query but 0 metrics/tables?)"}
        FailQuant["Verdict: Insufficient (Score: 0.40)<br/>Action: synthesize_dynamic_tool<br/>Hint: require_dynamic_tool"]
        
        CheckOverall{"Check 5: Overall Relevance Score >= 0.70?"}
        PassJudge["Verdict: Sufficient (Score >= 0.70)<br/>Action: proceed_to_synthesis"]
        FailGeneral["Verdict: Insufficient (Score < 0.70)<br/>Action: replan_static_tools"]
    end

    subgraph FALLBACK_ROUTING ["4. Corrective Fallback Routing Engine"]
        ActionRouter{"EvaluationVerdict Action Router"}
        
        subgraph PATHWAY_STATIC ["Pathway 1: Adaptive Static Re-Planning"]
            ReplanEngine["replan_with_feedback_async()<br/>• Relax narrow page & category filters<br/>• Broaden date ranges<br/>• Scale top_k = max(8, k + 4)<br/>• Anti-repetition guard prevents dups<br/>• 1-Cycle hard ceiling"]
            ReExecTools["ToolExecutor Re-Execution Node"]
        end
        
        subgraph PATHWAY_DYNAMIC ["Pathway 2: Dynamic ToolMaker Recovery"]
            ToolMakerEngine["DynamicToolMaker.generate_and_execute()<br/>• Synthesize bespoke Python script<br/>• Read-only MySQL connection with rollback<br/>• ToolCritic 5-metric scorecard (3 retries)<br/>• AST Subprocess Sandbox (15s, 512MB RAM)"]
        end
        
        subgraph PATHWAY_SILENCE ["Pathway 3: Authoritative Archival Absence"]
            SilenceNotice["Empty Evidence Hard-Stop Check<br/>Emit Authoritative Archival Silence:<br/>'The archived broadsheets contain no<br/>verifiable record of [Query]'<br/>Strict Anti-Hallucination Invariant"]
        end
    end

    subgraph SYNTHESIS ["5. Grounded Broadsheet Synthesis"]
        SynthesizerNode["AnswerSynthesizer<br/>(Blueprint-Driven Grounded Synthesis)"]
    end

    EvidenceIn --> FastFloor
    FastFloor -->|"Passes Fast Floor"| ImmediatePass
    FastFloor -->|"Below Fast Floor"| JudgePrompt
    
    JudgePrompt --> CheckAbsence
    CheckAbsence -->|"Yes (Legitimate Absence)"| PassAbsence
    CheckAbsence -->|"No"| CheckBalance
    
    CheckBalance -->|"Missing Requested Brand"| FailBalance
    CheckBalance -->|"Balanced Coverage"| CheckDate
    
    CheckDate -->|"Date Mismatch"| FailDate
    CheckDate -->|"Dates Aligned"| CheckQuant
    
    CheckQuant -->|"0 Metrics for Quant Query"| FailQuant
    CheckQuant -->|"Payload Valid"| CheckOverall
    
    CheckOverall -->|"Yes (Score >= 0.70)"| PassJudge
    CheckOverall -->|"No (Score < 0.70)"| FailGeneral
    
    ImmediatePass --> SynthesizerNode
    PassAbsence --> SynthesizerNode
    PassJudge --> SynthesizerNode
    
    FailBalance & FailDate & FailGeneral --> ActionRouter
    FailQuant --> ActionRouter
    
    ActionRouter -->|"replan_static_tools"| ReplanEngine
    ReplanEngine --> ReExecTools
    ReExecTools -->|"Recovery Evidence"| FastFloor
    ReExecTools -.->|"Evidence Still Empty"| SilenceNotice
    
    ActionRouter -->|"synthesize_dynamic_tool"| ToolMakerEngine
    ToolMakerEngine -->|"Verified Telemetry"| SynthesizerNode
    ToolMakerEngine -.->|"Execution Failed"| SilenceNotice
    
    SilenceNotice --> SynthesizerNode
```

### 12.2. Reflective LLM Answer Verifier & Editorial Critic Gate Flowchart

```mermaid
flowchart TD
    subgraph DRAFT_INPUT ["1. Synthesis Completion"]
        DraftAnswer["Synthesized Draft Answer (AnswerSynthesizer)<br/>+ Verified Ground Truth Evidence Payload"]
    end

    subgraph FAST_GATES ["2. Fast Groundedness Floor Gates (<5ms Deterministic)"]
        GateScope{"Gate 1: Scope Mismatch Interceptor<br/>(Archive-wide query restricted to 1 paper?)"}
        FailScope["Flagged: Scope Contradiction (Score: 0.2)<br/>Action: fallback_to_dynamic_tool<br/>Hint: Query across all broadsheets"]
        
        GateNaN{"Gate 2: Calculation NaN / Null Detector<br/>(Matches 'nan words', 'is nan', 'null count'?)"}
        FailNaN["Flagged: Math Defect (Score: 0.1)<br/>Action: fallback_to_dynamic_tool<br/>Hint: Recompute statistics in sandbox"]
        
        GateZero{"Gate 3: Zero-Issue Contradiction Refiner<br/>(0 DB records but draft claims 'Yes, available'?)"}
        RefineZero["Flagged: Absence Contradiction (Score: 0.2)<br/>Action: refine_answer<br/>Auto-replace with deterministic absence report"]
    end

    subgraph LLM_CRITIC ["3. Reflexive LLM-as-Judge Fact-Checking Critic"]
        CriticPrompt["Reflexive Answer Verifier (LLM Judge)<br/>Audits Draft against Ground Truth Evidence"]
        
        Dim1["Dimension 1: Groundedness & Citations<br/>Every claim backed by [Paper, Date, Page, Headline]"]
        Dim2["Dimension 2: Anti-Fluff & Precision<br/>Strip corporate consulting filler & speculative prose"]
        Dim3["Dimension 3: Archival Absence Fidelity<br/>No speculative inventing on unindexed dates"]
        Dim4["Dimension 4: Quantitative Accuracy<br/>Zero mathematical hallucination on averages/counts"]
        
        CriticPrompt --- Dim1 & Dim2 & Dim3 & Dim4
        CriticPrompt --> CriticDecision{"Evaluator Verdict Decision"}
    end

    subgraph VERIFIER_ACTIONS ["4. Verification Actions & State Machine Branching"]
        ActionAccept["Action: 'accept'<br/>Quality Score >= 0.70<br/>Grounded, factual, citation-compliant"]
        
        ActionRefine["Action: 'refine_answer'<br/>Quality Score 0.40 - 0.69<br/>Draft contains minor fluff or ungrounded claims<br/>Replace draft with verified refined_answer"]
        
        ActionFallback["Action: 'fallback_to_dynamic_tool'<br/>Quality Score < 0.40<br/>Diagnostic evidence gap or calculation defect"]
        
        subgraph ROLLBACK_LOOP ["5. LangGraph Dynamic Tool Rollback Loop"]
            RollbackNode["LangGraph Dynamic Tool Recovery<br/>(1-Cycle State Machine Ceiling)<br/>Branch directly to execute_dynamic_code"]
            ReCompute["Synthesize bespoke query in Subprocess Sandbox<br/>Inject verified metrics into evidence state"]
            ReSynth["Re-Synthesize Final Grounded Brief<br/>(AnswerSynthesizer)"]
        end
    end

    subgraph SSE_OUTPUT ["6. Client Delivery Channel"]
        SSEStream["FastAPI SSE Streaming Output (/api/query/stream)<br/>• Stage & Thought Events<br/>• Token-by-Token Markdown Stream<br/>• Interactive Broadsheet Citation Cards<br/>• High-Res Visual Asset Thumbnails (/api/photos/{id}/image)"]
    end

    DraftAnswer --> GateScope
    GateScope -->|"Violation Detected"| FailScope
    GateScope -->|"Pass"| GateNaN
    
    GateNaN -->|"NaN / Null Found"| FailNaN
    GateNaN -->|"Pass"| GateZero
    
    GateZero -->|"Contradiction Found"| RefineZero
    GateZero -->|"Pass"| CriticPrompt
    
    CriticDecision -->|"accept"| ActionAccept
    CriticDecision -->|"refine_answer"| ActionRefine
    CriticDecision -->|"fallback_to_dynamic_tool"| ActionFallback
    
    FailScope & FailNaN --> ActionFallback
    RefineZero --> ActionRefine
    
    ActionAccept --> SSEStream
    ActionRefine --> SSEStream
    
    ActionFallback --> RollbackNode
    RollbackNode --> ReCompute
    ReCompute --> ReSynth
    ReSynth --> SSEStream
```

### 12.3. Key Technical Invariants in Post-Retrieval Verification
1. **The Fast-Floor Optimization**: If retrieval already fetched `>= 1` broadsheet record with `>= 100` words of clean editorial body text and prominence `>= 0.65`, the system immediately passes to synthesis (`<5ms` latency), eliminating unnecessary LLM-as-judge calls for clean factual queries.
2. **Legitimate Archival Absence vs. Hallucination**: When a query asks for broadsheet availability on an unindexed date (e.g., Sunday) and the database returns 0 issues, the evaluator recognizes this as legitimate absence (`score = 0.85`), preventing false positive retries and instructing the synthesizer to state the absence authoritatively.
3. **Adaptive Static Re-Planning Safeguards**: When `replan_static_tools` triggers, narrow page or category constraints are purged, date ranges are widened, and `top_k` expands (`max(8, k+4)`). An anti-repetition filter guarantees the planner never re-runs identical tool configurations.
4. **LangGraph Dynamic Tool Rollback Ceiling**: If the `AnswerVerifier` detects a critical mathematical or comparative evidence gap, it triggers a 1-cycle rollback into `execute_dynamic_code`. The engine synthesizes, executes, and audits a bespoke Python tool in the AST sandbox before generating the final verified response. A strict 1-cycle counter prevents infinite fallback loops.

---

## 13. Failure Modes, Circuit Breakers & Resilience Matrix

| Failure Scenario | Trigger Detection | Immediate Mitigation | Ultimate Safety Net |
|---|---|---|---|
| **Cloud VLM Rate Limit** | Gemini / OpenRouter returns HTTP 429 (`RateLimitExhaustedError`) | `trip_circuit_breaker(60.0)` activates; subsequent requests bypass failing provider; immediate failover to **`google_cloud_vision`** / **`ollama_qwen3vl`** | **Deterministic Spatial OCR Matrix Engine** reconstructs tabular data from token coordinates; zero ingestion abort |
| **Malformed VLM Output** | VLM returns conversational text instead of structured JSON | Regex extraction of Markdown table blocks (`extract_markdown_table_from_raw_text`) | Deterministic OCR text density fallback |
| **Low-Confidence OCR** | Poor print quality, bleed-through, or broken text on old broadsheets | Consensus multi-page folio voting across Pages 1–15; RapidOCR ONNX with Unicode superscript normalization | Minimum confidence threshold filter; human verification flag in DB |
| **Zero Retrieval Hits** | Query mentions unindexed historical date or outside broadsheet scope | Evidence Relevance Gate detects 0 grounded chunks; triggers fallback web search via NewsData.io | Enforces strict Anti-Hallucination notice; explicitly states zero archival evidence found |
| **Unforeseen Analytics / Unsupported Parameters** | Query asks for custom metrics (page counts, size distribution) exceeding static tools | Layer 1 intercepts unsupported parameter and hands off to `dynamic_analysis`; Layer 2 CRAG invokes `ToolMaker` | Synthesizes ad-hoc tool; executes safely in subprocess AST Sandbox with 15s timeout, 512MB RAM cap, and read-only rollback |
| **Hallucinated Columns / Cartesian Joins in Dynamic Tools** | Generated tool queries non-existent columns (`published_at`) or joins without `DISTINCT` | `ToolCritic` SRF metric detects schema violations; re-prompts LLM with structured diagnostic critique | Bounded 3-retry loop with token-budgeted trace; enforces relational schema fidelity before evidence injection |
| **Statistical Metric Absence / Failed Analytics** | Query asks for variance, std dev, or complex ratios but tool fails or returns empty | Synthesizer detects absence of computed numbers in verified tool evidence | `QUANTITATIVE & STATISTICAL METRIC ABSENCE HARD-STOP` strictly forbids hallucinating numbers, truthfully reporting computation unavailability |
| **Cross-Date Asset / Stale Context Leakage** | User switches dates across multi-turn session with active attached asset | Query condenser and graph routers detect date conflict and prune attached asset | Executor strictly enforces `effective_date = explicit_query_date`, preventing queries against mismatched issues |
| **Network Outage / Cloud Down** | All external APIs (OpenRouter, Gemini, OpenAI) unreachable | Model Settings Studio switches to **Local Sovereign Preset** | 100% offline air-gapped execution via Ollama (Llama 3.1, DeepSeek R1, Qwen 3 VL), Docling, and local BGE-M3 |

---

*End of NewsLens-AI Complete End-to-End Data Flow & System Architecture Specification.*
