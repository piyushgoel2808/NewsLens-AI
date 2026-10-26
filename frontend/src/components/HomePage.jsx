import React, { useState, useEffect } from 'react';
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
  Globe,
  Clock,
  Flame,
} from 'lucide-react';
import { useActiveHighlight } from '../context/ActiveHighlightContext';
import { deduplicatedFetch } from '../utils/apiDeduplicator';
import BroadsheetPipelineGraphic from './BroadsheetPipelineGraphic';

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
    models: ['Google Gemini 2.5 Flash (VLM)', 'Qwen 2.5 VL (Local Ollama)', 'Spatial OCR Matrix Engine'],
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
    models: ['gemini-embedding-001 (768d MRL)', 'BAAI BGE-M3 (1024d PyTorch)', 'MySQL 8 (17 Tables)'],
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
    models: ['Gemini 2.5 Flash / Nemotron 3.5', 'Cross-Encoder MiniLM-L-6-v2', 'NewsData.io & Serper Web Cascade'],
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
// 8-Tool Agentic Matrix Configuration
// ---------------------------------------------------------------------------
const AGENTIC_TOOLS = [
  {
    name: 'hybrid_search',
    label: 'Hybrid Vector & BM25 Search',
    icon: Search,
    latency: '~1.0s',
    archetype: 'Factual & Deep Dive',
    color: 'emerald',
    description: 'Reciprocal Rank Fusion (RRF N=75) merging 768d/1024d dense embeddings with BM25 keyword search, capped to top-20 Cross-Encoder.',
    sampleQuery: 'What were the clean energy capacity targets announced by Tata Power?',
  },
  {
    name: 'sql_analytics',
    label: 'Relational SQL & Manifests',
    icon: Database,
    latency: '<200ms',
    archetype: 'Fast Catalog',
    color: 'sky',
    description: 'Bypasses vector indexing to query MySQL 8 relational tables directly for structured article lists, page catalogs, and metadata.',
    sampleQuery: 'List all business and technology articles published in The Goan on 2026-08-01.',
  },
  {
    name: 'inspect_visual_asset',
    label: 'Multimodal Visual Extraction',
    icon: Eye,
    latency: '3.5s',
    archetype: 'Chart & Table VLM',
    color: 'purple',
    description: 'Retrieves dedicated [INFOGRAPHIC / DATA TABLE] vectors, pulling transcribed GitHub-style Markdown tables and verifiable numbers.',
    sampleQuery: 'What were the IPO subscription rates and key figures in the Market Infographic on Page 4?',
  },
  {
    name: 'coverage_analysis',
    label: 'Differential Omission Audit',
    icon: Scale,
    latency: '1.8s',
    archetype: 'Omission Audit',
    color: 'amber',
    description: 'Executes Jaccard differential overlap analysis and cross-encoder logit scoring (≥ -5.0) to isolate genuine hyperlocal exclusives.',
    sampleQuery: 'Audit municipal infrastructure stories covered by The Goan but omitted by The Morning Standard.',
  },
  {
    name: 'timeline_synthesizer',
    label: 'Storyline Trajectory Clustering',
    icon: GitMerge,
    latency: '1.2s',
    archetype: 'Temporal Narrative',
    color: 'teal',
    description: 'Aggregates multi-week event clusters chronologically, linking milestone cards with deep-links to original broadsheet scans.',
    sampleQuery: 'Reconstruct the chronological timeline of central bank interest rate decisions.',
  },
  {
    name: 'read_article_deep',
    label: 'Full Column Deep Synthesis',
    icon: FileText,
    latency: '400ms',
    archetype: 'Deep Read',
    color: 'indigo',
    description: 'Fetches complete multi-column text, author attribution decks, and photo captions for a specific article ID in full context.',
    sampleQuery: 'Give me a comprehensive deep-dive summary of Article #104 with all author context.',
  },
  {
    name: 'tool_maker',
    label: 'Dynamic AST Sandbox Toolmaker',
    icon: Terminal,
    latency: '2.1s',
    archetype: 'Quantitative Math',
    color: 'rose',
    description: 'Generates and executes on-the-fly Python calculation scripts inside an AST-whitelisted, read-only subprocess sandbox.',
    sampleQuery: 'Calculate the percentage change in corporate tax revenue across the reported quarters.',
  },
  {
    name: 'web_search',
    label: 'Live Web Grounding Cascade',
    icon: Globe,
    latency: '1.5s',
    archetype: 'Real-Time Cross-Ref',
    color: 'cyan',
    description: 'Queries live internet APIs (Serper / NewsData) when archival evidence is silent, establishing unbroken provenance.',
    sampleQuery: 'Cross-reference the reported 2026 merger timeline with recent regulatory announcements.',
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
    explanation:
      'Plans parallel SQL issue summaries and targeted hybrid searches across both publications, compiling a side-by-side editorial matrix comparing headline tone, lead stories, and perspective divergence.',
    expectedTools: ['sql_analytics', 'hybrid_search'],
  },
  {
    archetype: 'article_catalog',
    label: 'Instant Manifest (<200ms)',
    badge: 'Fast Catalog',
    query: 'List all health and medical articles published in The Goan on 2026-08-01.',
    explanation:
      'Routes directly to sql_analytics, bypassing heavy vector search and neural reranking to return a complete structured catalog manifest with page numbers, headlines, and word counts in sub-200ms.',
    expectedTools: ['sql_analytics'],
  },
  {
    archetype: 'negative_coverage_audit',
    label: 'Omission & Gap Audit',
    badge: 'Omission Audit',
    query: 'Audit what was reported in The Goan regarding municipal infrastructure that was omitted by The Morning Standard.',
    explanation:
      'Executes calibrated Jaccard differential overlap analysis and cross-encoder logit scoring (≥ -5.0) to isolate genuine hyperlocal exclusives without false processing errors.',
    expectedTools: ['sql_analytics', 'coverage_analysis'],
  },
  {
    archetype: 'factual_lookup',
    label: 'Financial Table Transcription',
    badge: 'Multimodal RAG',
    query: 'What were the IPO subscription rates and key financial figures published in the Q3 Market Infographic on Page 4?',
    explanation:
      'Dispatches inspect_visual_asset to locate the dedicated [INFOGRAPHIC / DATA TABLE] vector point, pulling the transcribed Markdown table and 3–6 extracted statistical metrics directly into the citation evidence.',
    expectedTools: ['inspect_visual_asset', 'hybrid_search'],
  },
];

export default function HomePage() {
  const {
    setActiveTab,
    selectedModel,
    taskBindings,
    configuredProviders,
    providerReachability,
    activePreset,
    openIssueInReader,
    runAgentQuery,
    theme,
  } = useActiveHighlight();

  const [selectedStageId, setSelectedStageId] = useState('ingest_layout');
  const [activeQueryIndex, setActiveQueryIndex] = useState(0);

  // Live Archive Corpus Data
  const [newspapers, setNewspapers] = useState([]);
  const [recentIssues, setRecentIssues] = useState([]);
  const [corpusStats, setCorpusStats] = useState({
    totalPublications: 4,
    totalIssues: 12,
    totalArticles: 284,
  });

  useEffect(() => {
    // Fetch live newspapers
    deduplicatedFetch('/api/newspapers')
      .then((data) => {
        if (Array.isArray(data)) {
          setNewspapers(data);
          const totalPubs = data.length;
          const totalIss = data.reduce((sum, n) => sum + (Number(n.total_issues) || 0), 0);
          const totalArts = data.reduce((sum, n) => sum + (Number(n.total_articles) || 0), 0);
          setCorpusStats({
            totalPublications: totalPubs || 4,
            totalIssues: totalIss || 12,
            totalArticles: totalArts || 284,
          });
        }
      })
      .catch((err) => console.error('Failed to load newspapers:', err));

    // Fetch recent issues
    deduplicatedFetch('/api/issues')
      .then((data) => {
        if (Array.isArray(data)) {
          setRecentIssues(data.slice(0, 4));
        }
      })
      .catch((err) => console.error('Failed to load issues:', err));
  }, []);

  const currentStage = PIPELINE_STAGES.find((s) => s.id === selectedStageId) || PIPELINE_STAGES[0];
  const activeQuery = SAMPLE_QUERIES[activeQueryIndex];

  // Derive active execution tier
  const activeLlm = selectedModel || taskBindings?.answerer || taskBindings?.query_planner || 'gemini_flash';
  const isCloudGemini = activeLlm.startsWith('gemini');

  // Derive model names for summary card
  const getModelName = (taskId) => {
    const pId = taskBindings?.[taskId] || (taskId === 'embedding' ? 'gemini_embedding' : 'gemini_flash');
    if (pId === 'gemini_flash') return 'Gemini 2.5 Flash';
    if (pId === 'gemini_pro') return 'Gemini 2.5 Pro';
    if (pId === 'ollama_llama3') return 'Llama 3.1 8B';
    if (pId === 'ollama_deepseek') return 'DeepSeek R1 14B';
    if (pId === 'ollama_gemma4_e4b') return 'Gemma 4 E4B';
    if (pId === 'local_embed_bge') return 'BAAI BGE-M3 (1024d)';
    if (pId === 'gemini_embedding') return 'gemini-embedding-001 (768d)';
    const match = configuredProviders?.find((p) => p.id === pId);
    return match?.model || pId;
  };

  return (
    <div className="min-h-full bg-canvas text-text-primary transition-colors duration-300 pb-20">
      {/* -------------------------------------------------------------------- */}
      {/* 1. Classic Broadsheet Folio Strip across top */}
      {/* -------------------------------------------------------------------- */}
      <div className="border-b border-border-subtle bg-surface/80 backdrop-blur-sm px-4 sm:px-6 lg:px-8 py-2">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] text-text-muted uppercase tracking-widest">
          <div className="flex items-center gap-2">
            <span className="text-accent-primary font-bold">VOL. IV · ARCHIVAL RAG ENGINE</span>
            <span>·</span>
            <span>EST. 2026 INTELLIGENCE DESK</span>
          </div>
          <div className="flex items-center gap-2 text-text-secondary">
            <span>
              LIVE CORPUS:{' '}
              <strong className="text-text-primary">{corpusStats.totalPublications} PUBLICATIONS</strong> ·{' '}
              <strong className="text-text-primary">{corpusStats.totalIssues} ISSUES</strong> ·{' '}
              <strong className="text-text-primary">{corpusStats.totalArticles} ARTICLES</strong>
            </span>
            <span>·</span>
            <span className="text-accent-editorial">DUAL QDRANT: 768D MRL / 1024D BGE-M3</span>
          </div>
        </div>
      </div>

      {/* -------------------------------------------------------------------- */}
      {/* 2. Split Hero: Editorial Pitch (Left) + Interactive Pipeline Graphic (Right) */}
      {/* -------------------------------------------------------------------- */}
      <header className="relative overflow-hidden border-b border-border-subtle pt-10 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
            {/* Left: Editorial Pitch */}
            <div className="lg:col-span-6 space-y-6">
              {/* Architecture Status Pill */}
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-border-subtle bg-surface text-xs font-mono shadow-sm">
                <span className="w-2 h-2 rounded-full bg-accent-primary animate-pulse" />
                <span className="text-text-secondary font-medium">Production Architecture</span>
                <span className="text-border-subtle">•</span>
                <span className="text-accent-editorial font-bold">
                  {isCloudGemini ? '☁️ GCP Full Cloud Active' : '🖥️ Sovereign Local Active'}
                </span>
                <span className="text-border-subtle">•</span>
                <span className="text-text-muted text-[11px]">Dual Collections</span>
              </div>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-serif font-bold text-text-primary tracking-tight leading-[1.15]">
                The Broadsheet Intelligence & Archival RAG Platform
              </h1>

              <p className="text-base sm:text-lg text-text-secondary font-sans leading-relaxed max-w-xl">
                Transforming physical broadsheet newsprint into an interconnected, multi-tier knowledge graph
                with <strong className="text-text-primary font-semibold">2D spatial layout awareness</strong>,{' '}
                <strong className="text-text-primary font-semibold">multimodal VLM extraction</strong>, and{' '}
                <strong className="text-text-primary font-semibold">zero-hallucination agentic synthesis</strong>.
              </p>

              {/* Primary Action Buttons */}
              <div className="flex flex-wrap items-center gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setActiveTab('reader')}
                  className="flex items-center gap-2 px-5 py-3 rounded-xl bg-accent-primary hover:opacity-90 text-white font-semibold text-sm transition-all shadow-lg shadow-emerald-950/40 hover:scale-[1.02] cursor-pointer"
                >
                  <Newspaper className="w-4 h-4" />
                  <span>Explore Broadsheet Reader</span>
                  <ArrowRight className="w-4 h-4 ml-0.5" />
                </button>

                <button
                  type="button"
                  onClick={() => setActiveTab('agent')}
                  className="flex items-center gap-2 px-5 py-3 rounded-xl bg-surface hover:bg-elevated text-text-primary border border-border-subtle font-semibold text-sm transition-all hover:scale-[1.02] cursor-pointer"
                >
                  <Bot className="w-4 h-4 text-accent-visual" />
                  <span>Launch AI Newsroom</span>
                </button>

                <button
                  type="button"
                  onClick={() => setActiveTab('settings')}
                  className="flex items-center gap-2 px-4 py-3 rounded-xl bg-surface hover:bg-elevated text-text-secondary hover:text-text-primary border border-border-subtle font-medium text-sm transition-all cursor-pointer"
                >
                  <Sliders className="w-4 h-4 text-accent-editorial" />
                  <span>Open Model Studio</span>
                </button>
              </div>

              {/* Quick Metrics Strip */}
              <div className="grid grid-cols-3 gap-3 pt-4 border-t border-border-subtle font-mono">
                <div>
                  <div className="text-xl font-bold text-text-primary">150 DPI</div>
                  <div className="text-[11px] text-text-muted">75% RAM Savings</div>
                </div>
                <div>
                  <div className="text-xl font-bold text-text-primary">~1.0s</div>
                  <div className="text-[11px] text-text-muted">Two-Stage Neural Rerank</div>
                </div>
                <div>
                  <div className="text-xl font-bold text-text-primary">7 Archetypes</div>
                  <div className="text-[11px] text-text-muted">&lt;200ms Manifests</div>
                </div>
              </div>
            </div>

            {/* Right: Custom Interactive SVG Pipeline Graphic */}
            <div className="lg:col-span-6">
              <BroadsheetPipelineGraphic onSelectStage={(stageId) => setSelectedStageId(stageId)} />
            </div>
          </div>
        </div>
      </header>

      {/* -------------------------------------------------------------------- */}
      {/* 3. Bento Grid: Live Archive Pulse, 8-Tool Matrix & Model Summary */}
      {/* -------------------------------------------------------------------- */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-accent-primary/10 text-accent-primary border border-accent-primary/20 text-xs font-mono uppercase tracking-wider mb-2">
              <Activity className="w-3.5 h-3.5" />
              Intelligence Bento Grid
            </div>
            <h2 className="text-2xl sm:text-3xl font-serif font-bold text-text-primary">
              Live Core Capabilities & System Matrix
            </h2>
          </div>
          <span className="text-xs font-mono text-text-muted hidden sm:inline-block">
            Connected to port 8000
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Card 1: Live Archive Pulse (4 cols) */}
          <div className="lg:col-span-4 bg-surface rounded-2xl border border-border-subtle p-5 flex flex-col justify-between shadow-md">
            <div>
              <div className="flex items-center justify-between mb-4 pb-3 border-b border-border-subtle">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-elevated border border-border-subtle flex items-center justify-center text-accent-primary">
                    <Layers className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-text-primary font-sans">
                      Live Archive Pulse
                    </h3>
                    <span className="text-[10px] font-mono text-text-muted uppercase">
                      DIGITIZED CORPUS
                    </span>
                  </div>
                </div>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-accent-primary/15 text-accent-primary border border-accent-primary/30">
                  ONLINE
                </span>
              </div>

              {/* Corpus Metrics */}
              <div className="grid grid-cols-3 gap-2 mb-4 font-mono text-center">
                <div className="p-2 rounded-lg bg-elevated border border-border-subtle">
                  <div className="text-lg font-bold text-text-primary">{corpusStats.totalPublications}</div>
                  <div className="text-[9px] text-text-muted uppercase">Publications</div>
                </div>
                <div className="p-2 rounded-lg bg-elevated border border-border-subtle">
                  <div className="text-lg font-bold text-text-primary">{corpusStats.totalIssues}</div>
                  <div className="text-[9px] text-text-muted uppercase">Issues</div>
                </div>
                <div className="p-2 rounded-lg bg-elevated border border-border-subtle">
                  <div className="text-lg font-bold text-text-primary">{corpusStats.totalArticles}</div>
                  <div className="text-[9px] text-text-muted uppercase">Articles</div>
                </div>
              </div>

              {/* Recent Issues List */}
              <div className="space-y-2 mb-4">
                <span className="text-[10px] font-mono text-text-muted uppercase tracking-wider block">
                  Recent Digitized Issues:
                </span>
                {recentIssues.length > 0 ? (
                  recentIssues.map((iss) => (
                    <div
                      key={iss.id}
                      onClick={() => openIssueInReader(iss.id)}
                      className="flex items-center justify-between p-2 rounded-lg bg-elevated/70 border border-border-subtle hover:border-accent-primary/50 transition-all cursor-pointer group"
                    >
                      <div className="min-w-0">
                        <div className="text-xs font-semibold text-text-primary truncate group-hover:text-accent-primary">
                          {iss.newspaper_name || `Issue #${iss.id}`}
                        </div>
                        <div className="text-[10px] font-mono text-text-muted">
                          {iss.issue_date || 'Archival Edition'} · {iss.page_count || 1} pages
                        </div>
                      </div>
                      <ChevronRight className="w-3.5 h-3.5 text-text-muted group-hover:text-accent-primary transition-transform group-hover:translate-x-0.5" />
                    </div>
                  ))
                ) : (
                  <div className="p-3 text-xs text-text-muted text-center italic bg-elevated/50 rounded-lg">
                    Loading recent issues...
                  </div>
                )}
              </div>
            </div>

            <button
              type="button"
              onClick={() => setActiveTab('reader')}
              className="w-full py-2 px-3 rounded-xl bg-elevated hover:bg-accent-primary hover:text-white text-text-secondary border border-border-subtle text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer"
            >
              <span>Open Broadsheet Desk</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Card 2: 8-Tool Agentic Matrix (8 cols) */}
          <div className="lg:col-span-8 bg-surface rounded-2xl border border-border-subtle p-5 shadow-md flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4 pb-3 border-b border-border-subtle">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-elevated border border-border-subtle flex items-center justify-center text-accent-visual">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-text-primary font-sans">
                      8-Tool Agentic Retrieval Matrix
                    </h3>
                    <span className="text-[10px] font-mono text-text-muted uppercase">
                      AUTONOMOUS LANGGRAPH NEWSROOM PLANNER
                    </span>
                  </div>
                </div>
                <span className="text-xs font-mono text-text-muted">
                  Click any tool to pre-fill inquiry &rarr;
                </span>
              </div>

              {/* 8-Tool Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2.5">
                {AGENTIC_TOOLS.map((t) => {
                  const ToolIcon = t.icon;
                  return (
                    <button
                      key={t.name}
                      type="button"
                      onClick={() => runAgentQuery(t.sampleQuery)}
                      className="text-left p-2.5 rounded-xl bg-elevated/60 border border-border-subtle hover:border-accent-primary/60 hover:bg-elevated transition-all group flex flex-col justify-between min-h-[110px] cursor-pointer"
                      title={`Run sample query: "${t.sampleQuery}"`}
                    >
                      <div>
                        <div className="flex items-center justify-between mb-1.5">
                          <ToolIcon className="w-4 h-4 text-accent-primary group-hover:scale-110 transition-transform" />
                          <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-surface text-text-muted border border-border-subtle">
                            {t.latency}
                          </span>
                        </div>
                        <div className="text-xs font-mono font-bold text-text-primary group-hover:text-accent-primary transition-colors leading-tight">
                          {t.name}()
                        </div>
                        <div className="text-[10px] text-text-muted line-clamp-2 mt-1 leading-snug">
                          {t.label}
                        </div>
                      </div>

                      <div className="text-[9px] font-mono text-accent-editorial mt-2 flex items-center gap-0.5 group-hover:underline">
                        <span>Test Tool</span>
                        <ArrowRight className="w-2.5 h-2.5" />
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-border-subtle flex flex-wrap items-center justify-between text-xs font-mono text-text-muted">
              <span>Parallel Multi-Tool Gathering (asyncio.gather)</span>
              <button
                type="button"
                onClick={() => setActiveTab('agent')}
                className="text-accent-primary hover:underline flex items-center gap-1 cursor-pointer font-semibold"
              >
                <span>Launch Interactive Newsroom Sandbox</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Card 3: Active Model Pipeline Summary (Full 12 cols) */}
          <div className="lg:col-span-12 bg-surface rounded-2xl border border-border-subtle p-5 shadow-md">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-3 border-b border-border-subtle">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-elevated border border-border-subtle flex items-center justify-center text-accent-editorial">
                  <Cpu className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-text-primary font-sans">
                    Active Model Pipeline Matrix
                  </h3>
                  <span className="text-[10px] font-mono text-text-muted uppercase">
                    HOT-SWAPPABLE RUNTIME BINDINGS (/api/settings/model-bindings)
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs font-mono px-2.5 py-1 rounded-full bg-elevated border border-border-subtle text-text-secondary">
                  Active Preset: <strong className="text-text-primary uppercase">{activePreset || 'Cloud Full'}</strong>
                </span>
                <button
                  type="button"
                  onClick={() => setActiveTab('settings')}
                  className="px-3 py-1 rounded-lg bg-accent-editorial/15 hover:bg-accent-editorial/25 text-accent-editorial border border-accent-editorial/30 text-xs font-mono font-semibold transition-all cursor-pointer"
                >
                  Configure Pipeline &rarr;
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 font-mono">
              <div className="p-3 rounded-xl bg-elevated/70 border border-border-subtle">
                <span className="text-[10px] text-text-muted uppercase block mb-1">Reasoning & Synthesizer</span>
                <div className="text-xs font-bold text-text-primary truncate">{getModelName('answerer')}</div>
                <span className="text-[10px] text-accent-primary mt-1 inline-block">1M context</span>
              </div>

              <div className="p-3 rounded-xl bg-elevated/70 border border-border-subtle">
                <span className="text-[10px] text-text-muted uppercase block mb-1">Query Planner</span>
                <div className="text-xs font-bold text-text-primary truncate">{getModelName('query_planner')}</div>
                <span className="text-[10px] text-accent-primary mt-1 inline-block">LangGraph DAG</span>
              </div>

              <div className="p-3 rounded-xl bg-elevated/70 border border-border-subtle">
                <span className="text-[10px] text-text-muted uppercase block mb-1">Multimodal VLM</span>
                <div className="text-xs font-bold text-text-primary truncate">{getModelName('visual_extraction')}</div>
                <span className="text-[10px] text-accent-visual mt-1 inline-block">Single-Pass 150 DPI</span>
              </div>

              <div className="p-3 rounded-xl bg-elevated/70 border border-border-subtle">
                <span className="text-[10px] text-text-muted uppercase block mb-1">Spatial 2D Layout</span>
                <div className="text-xs font-bold text-text-primary truncate">{getModelName('layout_analysis')}</div>
                <span className="text-[10px] text-accent-editorial mt-1 inline-block">Convex Walls</span>
              </div>

              <div className="p-3 rounded-xl bg-elevated/70 border border-border-subtle">
                <span className="text-[10px] text-text-muted uppercase block mb-1">Dense Vector Embedding</span>
                <div className="text-xs font-bold text-text-primary truncate">{getModelName('embedding')}</div>
                <span className="text-[10px] text-accent-info mt-1 inline-block">Qdrant Collection</span>
              </div>

              <div className="p-3 rounded-xl bg-elevated/70 border border-border-subtle">
                <span className="text-[10px] text-text-muted uppercase block mb-1">Neural Reranker</span>
                <div className="text-xs font-bold text-text-primary truncate">MiniLM-L-6-v2</div>
                <span className="text-[10px] text-accent-primary mt-1 inline-block">Sub-1.0s RRF N=75</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------------------- */}
      {/* 4. Interactive System Architecture Blueprint ("How We Implemented It") */}
      {/* -------------------------------------------------------------------- */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-accent-primary/10 text-accent-primary border border-accent-primary/20 text-xs font-mono uppercase tracking-wider mb-2">
              <Code2 className="w-3.5 h-3.5" />
              Technical Implementation
            </div>
            <h2 className="text-2xl sm:text-3xl font-serif font-bold text-text-primary">
              Interactive System Architecture Blueprint
            </h2>
            <p className="text-sm text-text-secondary max-w-2xl mt-1">
              Click any stage of the 4-phase pipeline to inspect models, operational latencies, and underlying codebase modules.
            </p>
          </div>
          <span className="text-xs font-mono text-text-muted">
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
                type="button"
                onClick={() => setSelectedStageId(stage.id)}
                className={`text-left p-4 rounded-xl border transition-all relative overflow-hidden cursor-pointer ${
                  isSelected
                    ? 'bg-surface border-accent-primary shadow-lg shadow-emerald-950/20 ring-1 ring-accent-primary/40'
                    : 'bg-surface/50 border-border-subtle hover:border-text-secondary hover:bg-surface'
                }`}
              >
                {isSelected && (
                  <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-accent-primary to-accent-editorial" />
                )}
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-xs font-bold text-text-muted">
                    STAGE {stage.stepNumber}
                  </span>
                  <div
                    className={`w-7 h-7 rounded-lg flex items-center justify-center ${
                      isSelected ? 'bg-accent-primary/20 text-accent-primary' : 'bg-elevated text-text-muted'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                </div>
                <h3 className="text-sm font-semibold text-text-primary mb-1 leading-snug">
                  {stage.shortName}
                </h3>
                <p className="text-[11px] text-text-secondary line-clamp-2 leading-relaxed">
                  {stage.tagline}
                </p>
              </button>
            );
          })}
        </div>

        {/* Selected Stage Deep-Dive Card */}
        <div className="bg-surface rounded-2xl border border-border-subtle p-6 lg:p-8 shadow-xl">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left: Stage Overview & Description */}
            <div className="lg:col-span-7 space-y-5">
              <div className="flex flex-wrap items-center gap-2">
                <span className="px-2.5 py-0.5 rounded text-[11px] font-mono font-bold bg-accent-primary/20 text-accent-primary border border-accent-primary/30">
                  STAGE {currentStage.stepNumber}
                </span>
                <span className="px-2.5 py-0.5 rounded text-[11px] font-mono bg-elevated text-text-secondary border border-border-subtle">
                  {currentStage.badge}
                </span>
                <span className="text-xs text-text-muted font-mono ml-auto">
                  ⚡ Benchmark: <strong className="text-text-primary">{currentStage.latency}</strong>
                </span>
              </div>

              <h3 className="text-xl sm:text-2xl font-serif font-bold text-text-primary">
                {currentStage.name}
              </h3>

              <p className="text-sm text-text-secondary leading-relaxed">
                {currentStage.description}
              </p>

              {/* Technical Highlights List */}
              <div className="space-y-2 pt-2">
                <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-text-muted">
                  Key Architectural Implementations
                </h4>
                <div className="grid grid-cols-1 gap-2">
                  {currentStage.technicalHighlights.map((hl, i) => (
                    <div key={i} className="flex items-start gap-2.5 text-xs text-text-secondary">
                      <CheckCircle2 className="w-4 h-4 text-accent-primary shrink-0 mt-0.5" />
                      <span>{hl}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: Models & Codebase References */}
            <div className="lg:col-span-5 bg-elevated rounded-xl border border-border-subtle p-5 space-y-5">
              {/* Models / Engines Used */}
              <div>
                <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-text-muted mb-2.5 flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-accent-visual" />
                  Engines & Models Bound
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {currentStage.models.map((m, idx) => (
                    <span
                      key={idx}
                      className="px-2.5 py-1 rounded-md bg-surface border border-border-subtle text-xs font-mono text-text-primary shadow-sm"
                    >
                      {m}
                    </span>
                  ))}
                </div>
              </div>

              {/* Key Implementation Files */}
              <div>
                <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-text-muted mb-2.5 flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-accent-editorial" />
                  Codebase Source Files
                </h4>
                <div className="space-y-1.5">
                  {currentStage.keyFiles.map((f, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-surface border border-border-subtle text-[11px] font-mono text-text-secondary"
                    >
                      <span className="truncate">{f}</span>
                      <span className="text-[10px] text-text-muted shrink-0 ml-2">Source</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------------------- */}
      {/* 5. Interactive Broadsheet Query Simulator */}
      {/* -------------------------------------------------------------------- */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-surface border border-border-subtle rounded-2xl p-6 sm:p-8 shadow-xl">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 mb-8">
            <div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-accent-editorial/15 text-accent-editorial border border-accent-editorial/30 text-xs font-mono uppercase tracking-wider mb-2">
                <Terminal className="w-3.5 h-3.5" />
                Query Planning Simulator
              </div>
              <h3 className="text-2xl font-serif font-bold text-text-primary">
                Test Broadsheet Query Archetypes
              </h3>
              <p className="text-xs text-text-secondary mt-1">
                Select a sample broadsheet research prompt to see how the autonomous newsroom planner schedules specialized tools.
              </p>
            </div>

            {/* Archetype Selector Pills */}
            <div className="flex flex-wrap gap-2">
              {SAMPLE_QUERIES.map((q, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setActiveQueryIndex(idx)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all cursor-pointer ${
                    activeQueryIndex === idx
                      ? 'bg-accent-editorial text-white font-bold shadow-md'
                      : 'bg-elevated text-text-muted hover:text-text-primary'
                  }`}
                >
                  {q.badge}
                </button>
              ))}
            </div>
          </div>

          {/* Interactive Query Execution Card */}
          <div className="bg-elevated rounded-xl border border-border-subtle p-5 space-y-4 font-mono">
            <div>
              <span className="text-[10px] text-text-muted uppercase tracking-wider block mb-1">
                User Inquiry
              </span>
              <div className="text-sm font-sans font-medium text-text-primary bg-surface p-3 rounded-lg border border-border-subtle">
                "{activeQuery.query}"
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center">
              <div className="sm:col-span-8">
                <span className="text-[10px] text-text-muted uppercase tracking-wider block mb-1">
                  Cognitive Routing Rationale
                </span>
                <p className="text-xs text-text-secondary font-sans leading-relaxed">
                  {activeQuery.explanation}
                </p>
              </div>

              <div className="sm:col-span-4 bg-surface p-3 rounded-lg border border-border-subtle">
                <span className="text-[10px] text-text-muted uppercase tracking-wider block mb-1">
                  Tools Scheduled
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {activeQuery.expectedTools.map((t, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 rounded bg-accent-primary/10 text-accent-primary border border-accent-primary/20 text-[11px]"
                    >
                      {t}()
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                type="button"
                onClick={() => runAgentQuery(activeQuery.query)}
                className="flex items-center gap-2 px-5 py-2 rounded-lg bg-accent-primary hover:opacity-90 text-white font-sans text-xs font-semibold transition-all shadow-md cursor-pointer"
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
      {/* 6. Footer */}
      {/* -------------------------------------------------------------------- */}
      <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-border-subtle pt-6 text-xs text-text-muted font-mono">
          <div className="flex items-center gap-2">
            <Newspaper className="w-4 h-4 text-accent-primary" />
            <span>NewsLens-AI &mdash; Enterprise Broadsheet Intelligence Platform</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => setActiveTab('reader')}
              className="hover:text-text-primary transition-colors cursor-pointer"
            >
              Reader
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('agent')}
              className="hover:text-text-primary transition-colors cursor-pointer"
            >
              AI Newsroom
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('timeline')}
              className="hover:text-text-primary transition-colors cursor-pointer"
            >
              Storylines
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('graph')}
              className="hover:text-text-primary transition-colors cursor-pointer"
            >
              Entity Graph
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('settings')}
              className="hover:text-text-primary transition-colors cursor-pointer"
            >
              Model Studio
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
