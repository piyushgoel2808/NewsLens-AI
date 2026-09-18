import React, { useState } from 'react';
import {
  Newspaper,
  Bot,
  Network,
  GitMerge,
  Layers,
  UploadCloud,
  Sliders,
  Sparkles,
  Cpu,
  Cloud,
  Database,
  Search,
  ArrowRight,
  ChevronRight,
  CheckCircle2,
  ExternalLink,
  ShieldCheck,
  Eye,
  BookOpen,
  Zap,
  Calendar,
  Palette,
  Activity,
  FileText,
  Layout,
  Scale,
  Compass,
  Code2,
  Terminal,
  BarChart3,
  Table,
  Image,
  Check,
  Maximize2,
  Radio,
} from 'lucide-react';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

// ---------------------------------------------------------------------------
// Architectural Pipeline Stages Configuration
// ---------------------------------------------------------------------------
const PIPELINE_STAGES = [
  {
    id: 'ingest_layout',
    stepNumber: '01',
    name: 'Spatial Ingestion & 2D Layout Analysis',
    shortName: 'Spatial 2D Layout',
    badge: '150 DPI Archival Scan',
    tagline: 'Preserves physical broadsheet geometry, multi-column tracks, and headline decks.',
    icon: Layout,
    color: 'emerald',
    description:
      'High-resolution broadsheet PDF pages are rasterized at 150 DPI (~1500–2000px width), cutting memory by 75% while preserving 100% typographic fidelity. Deep layout models segment articles, isolate advertisements with convex boundary walls, clean headline metadata, and decompose Unicode ligatures.',
    models: ['IBM Docling Cloud (SaaS)', 'DocLayNet (Local ONNX)', 'Google Cloud Vision OCR'],
    latency: '2.5s – 4.2s / broadsheet page',
    keyFiles: [
      'backend/app/ingestion/parsers/docling.py',
      'backend/app/ingestion/layout/analyzer.py',
      'backend/app/ingestion/tasks.py',
    ],
    technicalHighlights: [
      'Convex boundary walls isolate statutory advertisements and tender notices',
      'Editorial headline cleansing transforms author box names into topical decks',
      'Ligature repair engine restores broadsheet OCR dropouts (e.g. \\ufb00 → ff)',
      'Corrupted font CMap fallback escalates to pure image OCR on corrupted text layers',
    ],
  },
  {
    id: 'vision_extract',
    stepNumber: '02',
    name: 'Multimodal Visual Extraction & VLM Scene Intelligence',
    shortName: 'Multimodal VLM',
    badge: 'Single-Pass Vision',
    tagline: 'Harvests photos, diagrams, IPO matrices, and charts into structured Markdown.',
    icon: Eye,
    color: 'sky',
    description:
      'Harvests visual assets and transmits the 150 DPI canvas alongside normalized coordinate bounding boxes to multimodal vision models. Automatically reconstructs complex tabular data into clean GitHub-flavored Markdown tables and extracts 3–6 key statistical metrics in a single pass.',
    models: ['Google Gemini 3.8 Flash (VLM)', 'Qwen 3 VL / 2.5 VL (Local Ollama)', 'Spatial OCR Matrix Engine'],
    latency: '10s – 25s / entire page (all visual assets)',
    keyFiles: [
      'backend/app/ingestion/single_pass_extractor.py',
      'backend/app/ingestion/parsers/docling.py',
      'backend/app/api/routers/articles.py',
    ],
    technicalHighlights: [
      'SinglePassVisualExtractor processes all page visual items in 1 unified LLM request',
      'Transcribes financial charts, IPO subscription matrices, and trend graphics to Markdown tables',
      'Generates rich 2–3 sentence editorial photo scene descriptions for news photography',
      'Binds dedicated [INFOGRAPHIC / DATA TABLE] chunks for dense semantic retrieval',
    ],
  },
  {
    id: 'storage_dual',
    stepNumber: '03',
    name: 'Dual-Vector Hierarchy & Relational Storage',
    shortName: 'Dual-Vector Store',
    badge: 'Strict Vector Isolation',
    tagline: 'Dimension-aware auto-routing between GCP Cloud and Local Sovereign vector spaces.',
    icon: Database,
    color: 'indigo',
    description:
      'Separates metadata, relational article hierarchies, and high-dimensional dense embeddings across optimized storage tiers. Qdrant manages strict dual collections with automatic dimension routing, keeping Cloud Run serverless memory slim while preserving local sovereign vector spaces intact.',
    models: ['Gemini Embedding 001 (768d MRL)', 'BAAI BGE-M3 (1024d PyTorch)', 'MySQL 8 (17 Tables)'],
    latency: 'Sub-15ms vector retrieval; Sub-5ms relational queries',
    keyFiles: [
      'backend/app/storage/qdrant_store.py',
      'backend/app/providers/gemini_embedding_provider.py',
      'backend/app/storage/mysql_store.py',
    ],
    technicalHighlights: [
      'article_chunks_v2: 768d Matryoshka MRL for GCP Full Cloud with 0 container RAM overhead',
      'article_chunks: 1024d dense Cosine vectors for local BAAI/bge-m3 sentence-transformers',
      'MySQL 8 system of record storing 17 normalized tables with full-text BM25 indexes',
      'MinIO & Google Cloud Storage bucket tiers for original archival broadsheet scans',
    ],
  },
  {
    id: 'crag_agent',
    stepNumber: '04',
    name: 'Corrective RAG (CRAG) & Agentic Reasoning Cascade',
    shortName: 'CRAG & Neural Rerank',
    badge: 'Sub-Second Cascade',
    tagline: '7 specialized query archetypes, two-stage reranking, and dynamic AST sandbox toolmaker.',
    icon: Bot,
    color: 'amber',
    description:
      'An autonomous LangGraph newsroom planner classifies queries into 7 distinct broadsheet archetypes, executes parallel multi-tool gathering, verifies evidence with Corrective RAG (CRAG), and performs two-stage Cross-Encoder neural reranking with candidate pool capping before structured 4-tier synthesis.',
    models: ['Gemini 3.8 Flash / Nemotron 30B', 'Cross-Encoder MiniLM-L-6-v2', 'NewsData.io & Serper Web Cascade'],
    latency: '1.0s warm hybrid retrieval (30x speedup); sub-200ms catalog manifests',
    keyFiles: [
      'backend/app/retrieval/agent/graph.py',
      'backend/app/retrieval/reranker.py',
      'backend/app/retrieval/agent/tools/tool_maker.py',
    ],
    technicalHighlights: [
      '7 specialized query archetypes: factual, trend, timeline, comparison, deep dive, omission audit, catalog',
      'Two-stage neural retrieval: RRF Stage 1 (N=75) + Cross-Encoder Stage 2 (K=20 capped) in ~1.0s',
      'Dynamic Python/SQL tool synthesis in an AST-whitelisted, read-only subprocess sandbox',
      'Grounded visual citation streaming with direct bounding box overlay coordinates',
    ],
  },
];

// ---------------------------------------------------------------------------
// Core Capabilities Cards Configuration
// ---------------------------------------------------------------------------
const CAPABILITIES = [
  {
    id: 'reader_capability',
    title: 'Spatial Broadsheet Reader',
    tabTarget: 'reader',
    badge: '150 DPI Archival Rendering',
    icon: Newspaper,
    accent: 'emerald',
    tagline: 'Interactive 2D bounding boxes and prominence heatmaps directly on digitized broadsheet newsprint.',
    bullets: [
      'Gold (≥0.70), Blue (0.30–0.69), and Teal (<0.30) prominence heatmaps',
      'Displays both printed physical folio and zero-indexed PDF container page',
      'One-click single-page re-ingestion without corrupting multi-page issues',
      'Dedicated "Ask Agent About This Photo" button on cropped visual cards',
    ],
    cta: 'Open Broadsheet Reader',
  },
  {
    id: 'agent_capability',
    title: '7-Archetype AI Newsroom Assistant',
    tabTarget: 'agent',
    badge: 'Corrective RAG (CRAG)',
    icon: Bot,
    accent: 'sky',
    tagline: 'Autonomous journalistic query planning, contextual coreference resolution, and strict provenance.',
    bullets: [
      '7 archetypes: factual lookup, quantitative trend, timeline, comparison, deep dive, omission, catalog',
      'Sub-200ms instant manifest routing for category listings without heavy embeddings',
      'Concurrent multi-tool gathering (asyncio.gather) cutting latency by 40–60%',
      'Conversational pronoun coreference with strict anti-hallucination guardrails',
    ],
    cta: 'Consult AI Assistant',
  },
  {
    id: 'vlm_capability',
    title: 'Multimodal Chart & Table Intelligence',
    tabTarget: 'reader',
    badge: 'Automated Markdown Tables',
    icon: BarChart3,
    accent: 'amber',
    tagline: 'Transcribes complex financial charts, IPO subscription matrices, and news photography in a single pass.',
    bullets: [
      'Unified single-pass VLM extraction converting graphics into clean Markdown tables',
      'Extracts 3–6 verifiable statistical metrics per chart with high-confidence grounding',
      'Editorial photo scene intelligence describing subjects, locations, and actions',
      'Embedded as dedicated [INFOGRAPHIC / DATA TABLE] vectors in Qdrant',
    ],
    cta: 'Inspect Visual Intelligence',
  },
  {
    id: 'graph_capability',
    title: 'Multi-Hop Entity Knowledge Graph',
    tabTarget: 'graph',
    badge: 'Salience Co-occurrence Network',
    icon: Network,
    accent: 'purple',
    tagline: 'Interactive graph visualization of people, corporations, government bodies, and locations.',
    bullets: [
      'Computed entity salience scores (0.0–1.0) based on headline and body prominence',
      'Multi-hop co-mention networks revealing hidden corporate and political ties',
      'Click-to-inspect: filter all broadsheet articles referencing any selected entity node',
      'Cross-edition entity tracking across multiple dates and newspaper brands',
    ],
    cta: 'Explore Entity Graph',
  },
  {
    id: 'timeline_capability',
    title: 'Storyline Trajectory & Timeline Canvas',
    tabTarget: 'timeline',
    badge: 'Chronological Milestone Engine',
    icon: GitMerge,
    accent: 'teal',
    tagline: 'Clusters evolving stories across multiple publication dates into chronological narratives.',
    bullets: [
      'Multi-week event milestone clustering tracing developments over time',
      'Direct one-click deep links from milestone cards to original broadsheet scans',
      'Sub-10ms response times powered by high-speed Redis trajectory caching',
      'Temporal range filtering across custom calendar spans and issue manifests',
    ],
    cta: 'View Storyline Canvas',
  },
  {
    id: 'dual_track_capability',
    title: 'Dual-Track Execution & Model Studio',
    tabTarget: 'settings',
    badge: 'GCP Cloud & Local Sovereign',
    icon: Sliders,
    accent: 'rose',
    tagline: 'Hot-swappable task bindings with one-click presets for GCP Cloud Run and Local Sovereign setups.',
    bullets: [
      'Full Cloud (GCP): Gemini 3.8 Flash + Docling Cloud + Gemini Embedding (0-RAM)',
      'Local Sovereign: 100% on-premise Ollama + local Docling + BAAI/bge-m3 (No API Keys)',
      'High-Speed Hybrid: combines local layout geometry with cloud LLM reasoning',
      'Background vector re-indexing CLI (scripts/reindex_embeddings.py) for backfills',
    ],
    cta: 'Open Model Studio',
  },
];

// ---------------------------------------------------------------------------
// Interactive Sample Broadsheet Queries
// ---------------------------------------------------------------------------
const SAMPLE_QUERIES = [
  {
    archetype: 'cross_newspaper_comparison',
    label: 'Cross-Publication Framing',
    badge: 'Comparison',
    query: 'Compare the front-page lead stories between The Economic Times and Mint on 2026-08-01.',
    explanation: 'Plans parallel SQL issue summaries and targeted hybrid searches across both publications, compiling a side-by-side editorial matrix comparing headline tone, lead stories, and perspective divergence.',
    expectedTools: ['sql_analytics', 'hybrid_search'],
  },
  {
    archetype: 'article_catalog',
    label: 'Instant Manifest (<200ms)',
    badge: 'Fast Catalog',
    query: 'List all health and medical articles published in The Goan on 2026-08-01.',
    explanation: 'Routes directly to sql_analytics, bypassing heavy vector search and neural reranking to return a complete structured catalog manifest with page numbers, headlines, and word counts in sub-200ms.',
    expectedTools: ['sql_analytics'],
  },
  {
    archetype: 'negative_coverage_audit',
    label: 'Omission & Gap Audit',
    badge: 'Omission Audit',
    query: 'Audit what was reported in The Goan regarding municipal infrastructure that was omitted by The Morning Standard.',
    explanation: 'Executes calibrated Jaccard differential overlap analysis and cross-encoder logit scoring (≥ -5.0) to isolate genuine hyperlocal exclusives without false processing errors.',
    expectedTools: ['sql_analytics', 'coverage_analysis'],
  },
  {
    archetype: 'factual_lookup',
    label: 'Financial Table Transcription',
    badge: 'Multimodal RAG',
    query: 'What were the IPO subscription rates and key financial figures published in the Q3 Market Infographic on Page 4?',
    explanation: 'Dispatches inspect_visual_asset to locate the dedicated [INFOGRAPHIC / DATA TABLE] vector point, pulling the transcribed Markdown table and 3–6 extracted statistical metrics directly into the citation evidence.',
    expectedTools: ['inspect_visual_asset', 'hybrid_search'],
  },
];

// ---------------------------------------------------------------------------
// Themes Configuration
// ---------------------------------------------------------------------------
const THEMES = {
  midnight: {
    id: 'midnight',
    name: 'Midnight Newsroom',
    tag: 'Default AI Newsroom',
    bg: 'bg-slate-950',
    cardBg: 'bg-slate-900/80',
    cardBorder: 'border-slate-800',
    accentText: 'text-emerald-400',
    accentBg: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    glow: 'from-emerald-600/20 via-teal-600/10 to-transparent',
    heroBadge: 'bg-emerald-950/60 border-emerald-500/30 text-emerald-300',
    statBorder: 'border-slate-800 hover:border-emerald-500/40',
  },
  archival: {
    id: 'archival',
    name: 'Archival Broadsheet',
    tag: 'Vintage Newsprint',
    bg: 'bg-stone-950',
    cardBg: 'bg-stone-900/80',
    cardBorder: 'border-stone-800',
    accentText: 'text-amber-400',
    accentBg: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    glow: 'from-amber-600/20 via-yellow-600/10 to-transparent',
    heroBadge: 'bg-amber-950/60 border-amber-500/30 text-amber-300',
    statBorder: 'border-stone-800 hover:border-amber-500/40',
  },
  cyber: {
    id: 'cyber',
    name: 'Cybernetic Intelligence',
    tag: 'Quantum Vector Ink',
    bg: 'bg-zinc-950',
    cardBg: 'bg-zinc-900/80',
    cardBorder: 'border-zinc-800',
    accentText: 'text-cyan-400',
    accentBg: 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30',
    glow: 'from-cyan-600/20 via-indigo-600/10 to-transparent',
    heroBadge: 'bg-cyan-950/60 border-cyan-500/30 text-cyan-300',
    statBorder: 'border-zinc-800 hover:border-cyan-500/40',
  },
};

export default function HomePage() {
  const { setActiveTab, selectedModel, taskBindings } = useActiveHighlight();
  const [activeTheme, setActiveTheme] = useState('midnight');
  const [selectedStageId, setSelectedStageId] = useState('ingest_layout');
  const [activeQueryIndex, setActiveQueryIndex] = useState(0);

  const theme = THEMES[activeTheme] || THEMES.midnight;
  const currentStage = PIPELINE_STAGES.find((s) => s.id === selectedStageId) || PIPELINE_STAGES[0];
  const activeQuery = SAMPLE_QUERIES[activeQueryIndex];

  // Derive active execution tier
  const activeLlm = selectedModel || taskBindings?.answerer || taskBindings?.query_planner || 'gemini_flash';
  const isCloudGemini = activeLlm.startsWith('gemini');

  return (
    <div className={`min-h-full ${theme.bg} text-slate-100 transition-colors duration-300 pb-20`}>
      {/* -------------------------------------------------------------------- */}
      {/* 1. Masthead & Hero Section */}
      {/* -------------------------------------------------------------------- */}
      <header className="relative overflow-hidden border-b border-slate-800/80 pt-8 pb-14 px-4 sm:px-6 lg:px-8">
        {/* Subtle Ambient Radial Glow */}
        <div
          className={`absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-80 bg-gradient-to-b ${theme.glow} blur-3xl pointer-events-none`}
        />

        <div className="max-w-7xl mx-auto relative z-10">
          {/* Top Vintage Dateline Ribbon */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3 mb-8 font-mono text-[11px] text-slate-400 uppercase tracking-widest">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1.5 text-slate-300 font-semibold">
                <Calendar className="w-3.5 h-3.5 text-slate-400" />
                HISTORICAL BROADSHEET ARCHIVES
              </span>
              <span className="text-slate-600">|</span>
              <span>VOL. IV · NO. 182</span>
              <span className="text-slate-600">|</span>
              <span>AUTONOMOUS COGNITIVE RAG</span>
            </div>

            {/* Interactive Aesthetic Theme Switcher */}
            <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-lg border border-slate-800">
              <Palette className="w-3.5 h-3.5 text-slate-400 ml-1 mr-1" />
              {Object.values(THEMES).map((t) => (
                <button
                  key={t.id}
                  onClick={() => setActiveTheme(t.id)}
                  className={`px-2.5 py-1 rounded text-[10px] font-mono transition-all ${
                    activeTheme === t.id
                      ? 'bg-slate-800 text-white font-bold shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {t.name.split(' ')[0]}
                </button>
              ))}
            </div>
          </div>

          {/* Main Hero Headline & Introduction */}
          <div className="text-center max-w-4xl mx-auto">
            {/* Live Operational Status Pill */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border text-xs font-mono mb-6 shadow-sm">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-slate-300 font-medium">Production Architecture</span>
              <span className="text-slate-600">•</span>
              <span className={theme.accentText}>
                {isCloudGemini ? '☁️ GCP Full Cloud Active' : '🖥️ Sovereign Local Active'}
              </span>
              <span className="text-slate-600">•</span>
              <span className="text-slate-400 text-[11px]">Dual Collections (768d / 1024d)</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-serif font-bold text-slate-50 tracking-tight leading-tight mb-6">
              The Broadsheet Intelligence & Archival RAG Platform
            </h1>

            <p className="text-base sm:text-lg text-slate-300 font-normal leading-relaxed max-w-3xl mx-auto mb-10">
              Transforming physical newspaper broadsheets into an interconnected, multi-tier knowledge graph
              with <strong className="text-white font-semibold">2D spatial layout awareness</strong>,{' '}
              <strong className="text-white font-semibold">multimodal VLM extraction</strong>, and{' '}
              <strong className="text-white font-semibold">zero-hallucination agentic synthesis</strong>.
            </p>

            {/* Quick Action Navigation CTAs */}
            <div className="flex flex-wrap items-center justify-center gap-3.5">
              <button
                onClick={() => setActiveTab('reader')}
                className="flex items-center gap-2 px-6 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-sm transition-all shadow-lg shadow-emerald-950/50 hover:scale-[1.02]"
              >
                <Newspaper className="w-4 h-4" />
                <span>Explore Broadsheet Reader</span>
                <ArrowRight className="w-4 h-4 ml-1" />
              </button>

              <button
                onClick={() => setActiveTab('agent')}
                className="flex items-center gap-2 px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-100 border border-slate-700 font-semibold text-sm transition-all hover:scale-[1.02]"
              >
                <Bot className="w-4 h-4 text-sky-400" />
                <span>Ask AI Newsroom Agent</span>
              </button>

              <button
                onClick={() => setActiveTab('graph')}
                className="flex items-center gap-2 px-5 py-3 rounded-xl bg-slate-900/90 hover:bg-slate-800 text-slate-300 border border-slate-800 font-medium text-sm transition-all"
              >
                <Network className="w-4 h-4 text-purple-400" />
                <span>Entity Graph</span>
              </button>

              <button
                onClick={() => setActiveTab('settings')}
                className="flex items-center gap-2 px-5 py-3 rounded-xl bg-slate-900/90 hover:bg-slate-800 text-slate-300 border border-slate-800 font-medium text-sm transition-all"
              >
                <Sliders className="w-4 h-4 text-amber-400" />
                <span>Model Studio</span>
              </button>
            </div>
          </div>

          {/* Key Capabilities Quick Metrics Ribbon */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-14 pt-8 border-t border-slate-800/80">
            <div className={`p-4 rounded-xl ${theme.cardBg} border ${theme.statBorder} transition-all`}>
              <div className="text-2xl font-bold font-mono text-white mb-1">150 DPI</div>
              <div className="text-xs text-slate-400 font-medium">Archival Raster Fidelity</div>
              <div className="text-[10px] text-slate-500 font-mono mt-1">75% RAM savings vs 300 DPI</div>
            </div>

            <div className={`p-4 rounded-xl ${theme.cardBg} border ${theme.statBorder} transition-all`}>
              <div className="text-2xl font-bold font-mono text-white mb-1">~1.0s</div>
              <div className="text-xs text-slate-400 font-medium">Two-Stage Neural Rerank</div>
              <div className="text-[10px] text-slate-500 font-mono mt-1">30x speedup via MiniLM cascade</div>
            </div>

            <div className={`p-4 rounded-xl ${theme.cardBg} border ${theme.statBorder} transition-all`}>
              <div className="text-2xl font-bold font-mono text-white mb-1">7 Archetypes</div>
              <div className="text-xs text-slate-400 font-medium">Specialized Query Routing</div>
              <div className="text-[10px] text-slate-500 font-mono mt-1">&lt;200ms manifest catalogs</div>
            </div>

            <div className={`p-4 rounded-xl ${theme.cardBg} border ${theme.statBorder} transition-all`}>
              <div className="text-2xl font-bold font-mono text-white mb-1">0 MB RAM</div>
              <div className="text-xs text-slate-400 font-medium">GCP Cloud Run Footprint</div>
              <div className="text-[10px] text-slate-500 font-mono mt-1">Full Cloud Docling + Gemini</div>
            </div>
          </div>
        </div>
      </header>

      {/* -------------------------------------------------------------------- */}
      {/* 2. Interactive System Architecture Blueprint ("How We Implemented It") */}
      {/* -------------------------------------------------------------------- */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="mb-10 text-center sm:text-left flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-mono uppercase tracking-wider mb-2">
              <Code2 className="w-3.5 h-3.5" />
              Technical Implementation
            </div>
            <h2 className="text-2xl sm:text-3xl font-serif font-bold text-slate-100">
              Interactive System Architecture Blueprint
            </h2>
            <p className="text-sm text-slate-400 max-w-2xl mt-1">
              Click any stage of the 4-phase pipeline to inspect models, operational latencies, and underlying codebase modules.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500">
            Click stages to inspect details &rarr;
          </span>
        </div>

        {/* 4 Pipeline Stages Tabs */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          {PIPELINE_STAGES.map((stage) => {
            const Icon = stage.icon;
            const isSelected = stage.id === selectedStageId;

            return (
              <button
                key={stage.id}
                onClick={() => setSelectedStageId(stage.id)}
                className={`text-left p-4 rounded-xl border transition-all relative overflow-hidden ${
                  isSelected
                    ? 'bg-slate-900 border-emerald-500/80 shadow-lg shadow-emerald-950/40 ring-1 ring-emerald-500/30'
                    : 'bg-slate-900/50 border-slate-800 hover:border-slate-700 hover:bg-slate-900/80'
                }`}
              >
                {isSelected && (
                  <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 to-teal-400" />
                )}
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-xs font-bold text-slate-500">
                    STAGE {stage.stepNumber}
                  </span>
                  <div
                    className={`w-7 h-7 rounded-lg flex items-center justify-center ${
                      isSelected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                </div>
                <h3 className="text-sm font-semibold text-slate-100 mb-1 leading-snug">
                  {stage.shortName}
                </h3>
                <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                  {stage.tagline}
                </p>
              </button>
            );
          })}
        </div>

        {/* Selected Stage Deep-Dive Card */}
        <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 lg:p-8 shadow-xl">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left: Stage Overview & Description */}
            <div className="lg:col-span-7 space-y-5">
              <div className="flex flex-wrap items-center gap-2">
                <span className="px-2.5 py-0.5 rounded text-[11px] font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  STAGE {currentStage.stepNumber}
                </span>
                <span className="px-2.5 py-0.5 rounded text-[11px] font-mono bg-slate-800 text-slate-300 border border-slate-700">
                  {currentStage.badge}
                </span>
                <span className="text-xs text-slate-400 font-mono ml-auto">
                  ⚡ Benchmark: <strong className="text-slate-200">{currentStage.latency}</strong>
                </span>
              </div>

              <h3 className="text-xl sm:text-2xl font-serif font-bold text-slate-50">
                {currentStage.name}
              </h3>

              <p className="text-sm text-slate-300 leading-relaxed">
                {currentStage.description}
              </p>

              {/* Technical Highlights List */}
              <div className="space-y-2 pt-2">
                <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-400">
                  Key Architectural Implementations
                </h4>
                <div className="grid grid-cols-1 gap-2">
                  {currentStage.technicalHighlights.map((hl, i) => (
                    <div key={i} className="flex items-start gap-2.5 text-xs text-slate-300">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      <span>{hl}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: Models & Codebase References */}
            <div className="lg:col-span-5 bg-slate-950/80 rounded-xl border border-slate-800/90 p-5 space-y-5">
              {/* Models / Engines Used */}
              <div>
                <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-400 mb-2.5 flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-sky-400" />
                  Engines & Models Bound
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {currentStage.models.map((m, idx) => (
                    <span
                      key={idx}
                      className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-700 text-xs font-mono text-slate-200 shadow-sm"
                    >
                      {m}
                    </span>
                  ))}
                </div>
              </div>

              {/* Key Implementation Files */}
              <div>
                <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-400 mb-2.5 flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-amber-400" />
                  Codebase Source Files
                </h4>
                <div className="space-y-1.5">
                  {currentStage.keyFiles.map((f, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-[11px] font-mono text-slate-300"
                    >
                      <span className="truncate">{f}</span>
                      <span className="text-[10px] text-slate-500 shrink-0 ml-2">Source</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------------------- */}
      {/* 3. Core Capabilities Explorer ("What NewsLens-AI Can Do") */}
      {/* -------------------------------------------------------------------- */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center max-w-3xl mx-auto mb-12">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-sky-500/10 text-sky-400 border border-sky-500/20 text-xs font-mono uppercase tracking-wider mb-3">
            <Sparkles className="w-3.5 h-3.5" />
            Comprehensive Capabilities
          </div>
          <h2 className="text-3xl font-serif font-bold text-slate-100 mb-3">
            What NewsLens-AI Can Do
          </h2>
          <p className="text-sm text-slate-400">
            A purpose-built suite of analytical workflows designed for investigative journalists, researchers, and financial analysts navigating complex broadsheet newsprint.
          </p>
        </div>

        {/* 6 Capabilities Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {CAPABILITIES.map((cap) => {
            const Icon = cap.icon;

            return (
              <div
                key={cap.id}
                className={`p-6 rounded-2xl ${theme.cardBg} border ${theme.cardBorder} flex flex-col justify-between hover:border-slate-700 transition-all hover:shadow-xl group`}
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center text-emerald-400 border border-slate-700 group-hover:scale-105 transition-transform">
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700">
                      {cap.badge}
                    </span>
                  </div>

                  <h3 className="text-lg font-bold text-slate-100 mb-2 font-serif group-hover:text-emerald-400 transition-colors">
                    {cap.title}
                  </h3>

                  <p className="text-xs text-slate-400 mb-4 leading-relaxed">
                    {cap.tagline}
                  </p>

                  <ul className="space-y-2 mb-6 text-xs text-slate-300">
                    {cap.bullets.map((bullet, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-emerald-400 mt-0.5 font-bold">•</span>
                        <span>{bullet}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <button
                  onClick={() => setActiveTab(cap.tabTarget)}
                  className="w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-emerald-600 text-slate-200 hover:text-white font-medium text-xs flex items-center justify-center gap-1.5 transition-all shadow-sm"
                >
                  <span>{cap.cta}</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
        </div>
      </section>

      {/* -------------------------------------------------------------------- */}
      {/* 4. Interactive Broadsheet Query Simulator */}
      {/* -------------------------------------------------------------------- */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 mb-8">
            <div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-amber-500/10 text-amber-400 border border-amber-500/20 text-xs font-mono uppercase tracking-wider mb-2">
                <Terminal className="w-3.5 h-3.5" />
                Query Planning Simulator
              </div>
              <h3 className="text-2xl font-serif font-bold text-slate-100">
                Test Broadsheet Query Archetypes
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Select a sample broadsheet research prompt to see how the autonomous newsroom planner schedules specialized tools.
              </p>
            </div>

            {/* Archetype Selector Pills */}
            <div className="flex flex-wrap gap-2">
              {SAMPLE_QUERIES.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => setActiveQueryIndex(idx)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                    activeQueryIndex === idx
                      ? 'bg-amber-500 text-slate-950 font-bold shadow-md'
                      : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {q.badge}
                </button>
              ))}
            </div>
          </div>

          {/* Interactive Query Execution Card */}
          <div className="bg-slate-950/80 rounded-xl border border-slate-800 p-5 space-y-4 font-mono">
            <div>
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                User Inquiry
              </span>
              <div className="text-sm font-sans font-medium text-slate-100 bg-slate-900/90 p-3 rounded-lg border border-slate-800">
                "{activeQuery.query}"
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center">
              <div className="sm:col-span-8">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                  Cognitive Routing Rationale
                </span>
                <p className="text-xs text-slate-300 font-sans leading-relaxed">
                  {activeQuery.explanation}
                </p>
              </div>

              <div className="sm:col-span-4 bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                  Tools Scheduled
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {activeQuery.expectedTools.map((t, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[11px]"
                    >
                      {t}()
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setActiveTab('agent')}
                className="flex items-center gap-2 px-5 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-sans text-xs font-semibold transition-all shadow-md"
              >
                <Bot className="w-3.5 h-3.5" />
                <span>Run Live in Agent Assistant</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------------------- */}
      {/* 5. Production Ready Footer Banner */}
      {/* -------------------------------------------------------------------- */}
      <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-slate-800/80 pt-6 text-xs text-slate-500 font-mono">
          <div className="flex items-center gap-2">
            <Newspaper className="w-4 h-4 text-emerald-400" />
            <span>NewsLens-AI &mdash; Enterprise Broadsheet Intelligence Platform</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setActiveTab('reader')}
              className="hover:text-slate-300 transition-colors"
            >
              Reader
            </button>
            <button
              onClick={() => setActiveTab('agent')}
              className="hover:text-slate-300 transition-colors"
            >
              Agent
            </button>
            <button
              onClick={() => setActiveTab('graph')}
              className="hover:text-slate-300 transition-colors"
            >
              Entity Graph
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className="hover:text-slate-300 transition-colors"
            >
              Model Settings
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
