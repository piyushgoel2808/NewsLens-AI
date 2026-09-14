# NewsLens-AI Data Flow & System Architecture

> **Notice: Consolidated Specification**
> 
> This document has been merged and unified with `docs/data_flow_architecture.md` to establish a single, authoritative source of truth that is 100% accurate, up-to-date, and structured chronologically in the natural sequence of data flow:
>
> 👉 [**Complete End-to-End Data Flow & System Architecture (`docs/data_flow_architecture.md`)**](data_flow_architecture.md)
>
> ### What is Covered in the Unified Specification:
> 1. **High-Level System Architecture & Global Subsystem Topology** (Presentation, Intake, Storage, Agentic State Machine)
> 2. **Dynamic Model Provider Registry & Pipeline Task Bindings** (Hot-swapping, Sovereign Tier 1, Cloud Tier 2, Enterprise Tier 3)
> 3. **Document Intake & Broadsheet Ingestion Pipeline Data Flow** (Stages 1-6: Intake, Masthead RapidOCR, 300 DPI Rasterization, 5-Pass Spatial Layout, Reading Order Linearization, 12-Domain Classification, BGE-M3 Chunking)
> 4. **Visual Asset Intelligence, Multimodal VLM & Failover Data Flow** (3-stage visual pipeline, 60s cooldown circuit breaker, local Qwen failover, Spatial OCR Matrix)
> 5. **Relational Knowledge Graph & Chronological Storyline Trajectories** (Entity co-occurrence graph, Cytoscape JSON payload, Redis trajectory caching)
> 6. **Multi-Tier Storage Layer Architecture & Data Lifecycle Matrix** (MySQL 8, Qdrant, MinIO, Redis 7)
> 7. **Conversational Agent Query Lifecycle & LangGraph Execution Sequence** (Complete sequence diagram with Redis caching, query condensation, planner, tools, ToolMaker/Critic, CRAG, synthesizer, verifier, and SSE streaming)
> 8. **Query Planner & Dynamic Answer Blueprint Data Flow** (Complete Query Planner Architecture Flowchart, 7 verified archetypes, strict operational tool contracts)
> 9. **4-Tier Journalistic Web Search Grounding Data Flow** (NewsData.io ➔ Serper ➔ Tavily ➔ DuckDuckGo)
> 10. **Broadsheet Reader to Agent Visual Attachment Data Flow** (Interactive attachment sequence diagram and 4 anti-leakage guardrails)
> 11. **Dynamic Tool Synthesis, Subprocess AST Sandbox & Closed-Loop ToolCritic Flow** (6-step lifecycle, AST whitelist/blacklist, sandbox parameters, 5-metric scorecard with Legitimate Absence)
> 12. **Corrective RAG (CRAG), Answer Verification & Streaming Delivery Flow** (Fast-floor <5ms check, reflexive LLM-as-judge, blueprint synthesis, reflective LLM verifier)
> 13. **Failure Modes, Circuit Breakers & Resilience Matrix** (Comprehensive 9-scenario failure mode matrix)
