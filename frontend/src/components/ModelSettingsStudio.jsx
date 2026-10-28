import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  Sliders,
  Cpu,
  Database,
  CheckCircle2,
  AlertCircle,
  Copy,
  RefreshCw,
  Code2,
  Cloud,
  Laptop,
  Zap,
  ShieldCheck,
  Sparkles,
  Server,
  Activity,
  ArrowRight,
  Eye,
  Wrench,
  Search,
  Download,
  Check,
  ExternalLink,
  Terminal,
  Filter,
  Layers,
  ChevronRight,
  ChevronDown,
  Play,
  RotateCcw,
  CheckSquare,
  Network,
  X,
  HelpCircle,
  Radio,
} from 'lucide-react';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

// ---------------------------------------------------------------------------
// Model Research & Capability Registry
// ---------------------------------------------------------------------------
export const MODEL_CATALOG = {
  // Reasoning & Planning
  gemini_pro: {
    id: 'gemini_pro',
    name: 'Google Gemini 2.5 Pro',
    provider: 'Google Vertex AI / AI Studio',
    category: 'reasoning',
    categoryLabel: 'Reasoning & Planning',
    context: '1M ctx',
    badges: ['CoT Reasoning', '1M Ctx', 'Tool Calling', 'Cloud Native'],
    supportsTasks: ['query_planner', 'answerer', 'article_segmentation', 'query_condenser'],
  },
  gemini_flash: {
    id: 'gemini_flash',
    name: 'Google Gemini 2.5 Flash',
    provider: 'Google AI Studio',
    category: 'reasoning',
    categoryLabel: 'Reasoning & Planning',
    context: '1M ctx',
    badges: ['Ultra Fast', '1M Ctx', 'Multimodal VLM', 'Tool Calling'],
    supportsTasks: [
      'query_planner',
      'answerer',
      'visual_extraction',
      'metadata_extraction',
      'classification',
      'query_condenser',
      'article_segmentation',
    ],
  },
  ollama_deepseek: {
    id: 'ollama_deepseek',
    name: 'DeepSeek-R1 14B',
    provider: 'Local Ollama Sovereign',
    category: 'reasoning',
    categoryLabel: 'Reasoning & Planning',
    context: '65K ctx',
    badges: ['Reasoning CoT', '65K Ctx', 'Local Sovereign', '100% Offline'],
    supportsTasks: ['query_planner', 'answerer', 'article_segmentation', 'query_condenser'],
  },
  ollama_nemotron: {
    id: 'ollama_nemotron',
    name: 'NVIDIA Nemotron 3.5 Lightning',
    provider: 'Local Ollama Sovereign',
    category: 'reasoning',
    categoryLabel: 'Reasoning & Planning',
    context: '128K ctx',
    badges: ['128K Ctx', 'Fast Agentic CoT', 'Tool Calling', 'Local Sovereign'],
    supportsTasks: ['query_planner', 'answerer', 'query_condenser'],
  },
  nvidia_nemotron: {
    id: 'nvidia_nemotron',
    name: 'NVIDIA Nemotron 3.5 Lightning 30B',
    provider: 'NVIDIA NIM Gateway',
    category: 'reasoning',
    categoryLabel: 'Reasoning & Planning',
    context: '128K ctx',
    badges: ['128K Ctx', 'High Speed', 'Hosted NIM'],
    supportsTasks: ['query_planner', 'answerer'],
  },

  // Synthesis & Answering
  ollama_llama3: {
    id: 'ollama_llama3',
    name: 'Meta Llama 3.1 8B',
    provider: 'Local Ollama Sovereign',
    category: 'synthesis',
    categoryLabel: 'Synthesis & Answering',
    context: '128K ctx',
    badges: ['128K Ctx', 'Local Sovereign', 'Low VRAM', '100% Offline'],
    supportsTasks: ['answerer', 'metadata_extraction', 'classification', 'query_condenser', 'query_planner'],
  },
  ollama_gemma4_12b: {
    id: 'ollama_gemma4_12b',
    name: 'Google Gemma 4 12B',
    provider: 'Local Ollama Sovereign',
    category: 'synthesis',
    categoryLabel: 'Synthesis & Answering',
    context: '131K ctx',
    badges: ['131K Ctx', 'High Fidelity Prose', 'Local Sovereign'],
    supportsTasks: ['answerer', 'query_condenser', 'classification'],
  },
  openai_gpt4o: {
    id: 'openai_gpt4o',
    name: 'OpenAI GPT-4o',
    provider: 'OpenAI Direct API',
    category: 'synthesis',
    categoryLabel: 'Synthesis & Answering',
    context: '128K ctx',
    badges: ['128K Ctx', 'Multimodal', 'Hosted Cloud'],
    supportsTasks: ['query_planner', 'answerer', 'visual_extraction', 'metadata_extraction'],
  },
  openai_gpt4o_mini: {
    id: 'openai_gpt4o_mini',
    name: 'OpenAI GPT-4o Mini',
    provider: 'OpenAI Direct API',
    category: 'synthesis',
    categoryLabel: 'Synthesis & Answering',
    context: '128K ctx',
    badges: ['128K Ctx', 'Fast Synthesis', 'Cost Efficient'],
    supportsTasks: ['answerer', 'classification', 'metadata_extraction'],
  },
  openrouter_gemma4_26b: {
    id: 'openrouter_gemma4_26b',
    name: 'Gemma 4 26B (Free)',
    provider: 'OpenRouter Gateway',
    category: 'synthesis',
    categoryLabel: 'Synthesis & Answering',
    context: '262K ctx',
    badges: ['262K Ctx', 'Free Tier', 'Hosted Gateway'],
    supportsTasks: ['answerer', 'query_planner'],
  },

  // Multimodal Vision / VLM
  gemini_vision: {
    id: 'gemini_vision',
    name: 'Google Gemini 2.5 Flash VLM',
    provider: 'Google Vertex AI / AI Studio',
    category: 'vision',
    categoryLabel: 'Multimodal Vision & VLM',
    context: '1M ctx',
    badges: ['1M Ctx', 'High Res VLM', 'Table Markdown', 'BBox Parsing'],
    supportsTasks: ['visual_extraction', 'layout_analysis'],
  },
  ollama_gemma4_e4b: {
    id: 'ollama_gemma4_e4b',
    name: 'Google Gemma 4 E4B VLM',
    provider: 'Local Ollama Sovereign',
    category: 'vision',
    categoryLabel: 'Multimodal Vision & VLM',
    context: '131K ctx',
    badges: ['131K Ctx', 'Local Vision', 'Edge Optimized'],
    supportsTasks: ['visual_extraction'],
  },
  ollama_vlm: {
    id: 'ollama_vlm',
    name: 'Qwen 2.5-VL 7B',
    provider: 'Local Ollama Sovereign',
    category: 'vision',
    categoryLabel: 'Multimodal Vision & VLM',
    context: '32K ctx',
    badges: ['32K Ctx', 'Local VLM', 'Spatial Detection'],
    supportsTasks: ['visual_extraction', 'layout_analysis'],
  },
  ollama_qwen3vl: {
    id: 'ollama_qwen3vl',
    name: 'Qwen 3-VL Next-Gen',
    provider: 'Local Ollama Sovereign',
    category: 'vision',
    categoryLabel: 'Multimodal Vision & VLM',
    context: '32K ctx',
    badges: ['32K Ctx', 'Next-Gen VLM', 'Local Sovereign'],
    supportsTasks: ['visual_extraction'],
  },
  nvidia_llama_vision: {
    id: 'nvidia_llama_vision',
    name: 'Meta Llama 3.2 11B Vision',
    provider: 'NVIDIA NIM Gateway',
    category: 'vision',
    categoryLabel: 'Multimodal Vision & VLM',
    context: '128K ctx',
    badges: ['128K Ctx', 'Hosted NIM', 'Vision Specialist'],
    supportsTasks: ['visual_extraction'],
  },

  // Spatial Layout & Document OCR
  docling_cloud_parser: {
    id: 'docling_cloud_parser',
    name: 'IBM Docling Cloud API',
    provider: 'IBM Cloud / Hosted Docling',
    category: 'layout',
    categoryLabel: 'Spatial Layout & Document OCR',
    badges: ['DocLayNet 2D', 'Zero Local RAM', 'High Precision Tables'],
    supportsTasks: ['layout_analysis', 'document_parser', 'ocr'],
  },
  docling_parser: {
    id: 'docling_parser',
    name: 'Docling Local (DocLayNet + RapidOCR)',
    provider: 'Local ONNX Engine',
    category: 'layout',
    categoryLabel: 'Spatial Layout & Document OCR',
    badges: ['100% Offline', 'DocLayNet ONNX', 'RapidOCR Local'],
    supportsTasks: ['layout_analysis', 'document_parser', 'ocr'],
  },
  google_cloud_vision: {
    id: 'google_cloud_vision',
    name: 'Google Cloud Vision OCR',
    provider: 'Google Cloud Platform',
    category: 'layout',
    categoryLabel: 'Spatial Layout & Document OCR',
    badges: ['GCP Cloud', 'Historical Print OCR'],
    supportsTasks: ['ocr', 'document_parser'],
  },

  // Dense Vector Embeddings
  gemini_embedding: {
    id: 'gemini_embedding',
    name: 'gemini-embedding-001 (768d MRL)',
    provider: 'Google AI Studio / Vertex AI',
    category: 'embedding',
    categoryLabel: 'Dense Vector Embeddings & MRL',
    badges: ['768d MRL', 'article_chunks_v2', 'Zero Container RAM'],
    supportsTasks: ['embedding'],
  },
  local_embed_bge: {
    id: 'local_embed_bge',
    name: 'BAAI BGE-M3 (1024d Multi-Lingual)',
    provider: 'Local HuggingFace / Transformers',
    category: 'embedding',
    categoryLabel: 'Dense Vector Embeddings & MRL',
    badges: ['1024d Dense', 'article_chunks', 'Multi-Lingual', '100% Offline'],
    supportsTasks: ['embedding'],
  },
  embedding_gemma2: {
    id: 'embedding_gemma2',
    name: 'Google EmbeddingGemma 2 (768d)',
    provider: 'Local Ollama Sovereign',
    category: 'embedding',
    categoryLabel: 'Dense Vector Embeddings & MRL',
    badges: ['768d Dense', 'article_chunks_v2', 'Low VRAM', 'Local Sovereign'],
    supportsTasks: ['embedding'],
  },
  ollama_embed: {
    id: 'ollama_embed',
    name: 'Nomic Embed Text (768d)',
    provider: 'Local Ollama Sovereign',
    category: 'embedding',
    categoryLabel: 'Dense Vector Embeddings & MRL',
    badges: ['768d Dense', 'Fast Ollama', '100% Offline'],
    supportsTasks: ['embedding'],
  },
  openai_embed: {
    id: 'openai_embed',
    name: 'OpenAI text-embedding-3-large (3072d)',
    provider: 'OpenAI Direct API',
    category: 'embedding',
    categoryLabel: 'Dense Vector Embeddings & MRL',
    badges: ['3072d High Dim', 'Hosted Cloud'],
    supportsTasks: ['embedding'],
  },
};

// ---------------------------------------------------------------------------
// Architecture Presets Configuration
// ---------------------------------------------------------------------------
const PRESET_PROFILES = [
  {
    id: 'cloud_full',
    name: 'Full Cloud Google Gemini',
    badge: 'Zero Container RAM (GCP Prod)',
    icon: Cloud,
    tagline: 'Recommended for ultra-fast, robust reasoning and vision with zero local GPU / CPU resource consumption.',
    pipelineHighlights: {
      reasoning: 'Google Gemini 2.5 Flash (1M ctx, Vertex AI / AI Studio)',
      vision: 'Gemini 2.5 Flash Single-Pass VLM + IBM Docling Cloud',
      indexing: 'gemini-embedding-001 (768d MRL → article_chunks_v2)',
    },
    description:
      'Routes all LLM reasoning, article segmentation, and visual extraction through Google Gemini 2.5 Flash, with document parsing powered by IBM Cloud Docling API, and vector indexing routed to Qdrant article_chunks_v2 via Gemini Embedding 001.',
    bindings: {
      query_planner: 'gemini_flash',
      answerer: 'gemini_flash',
      metadata_extraction: 'gemini_flash',
      classification: 'gemini_flash',
      article_segmentation: 'gemini_flash',
      visual_extraction: 'gemini_flash',
      layout_analysis: 'gemini_flash',
      document_parser: 'docling_cloud_parser',
      ocr: 'docling_cloud_parser',
      embedding: 'gemini_embedding',
    },
  },
  {
    id: 'cloud_hybrid',
    name: 'High-Speed Hybrid',
    badge: 'Lightning Fast',
    icon: Zap,
    tagline: 'Ultra-low latency reasoning and specialized 2D document parsing.',
    pipelineHighlights: {
      reasoning: 'Gemini 2.5 Flash + NVIDIA Nemotron 3.5 Lightning',
      vision: 'Gemini 2.5 Flash / Gemma 4 E4B + Local Docling 2D',
      indexing: 'BAAI/bge-m3 (1024d → article_chunks) or EmbeddingGemma 2 (768d)',
    },
    description:
      'Combines Google Gemini 2.5 Flash and NVIDIA Nemotron 3.5 Lightning for reasoning, paired with specialized 2D document parsing and local/cloud vector embeddings.',
    bindings: {
      query_planner: 'gemini_flash',
      answerer: 'gemini_flash',
      metadata_extraction: 'gemini_flash',
      classification: 'gemini_flash',
      article_segmentation: 'gemini_flash',
      visual_extraction: 'gemini_vision',
      layout_analysis: 'docling_parser',
      document_parser: 'docling_parser',
      ocr: 'google_cloud_vision',
      embedding: 'local_embed_bge',
    },
  },
  {
    id: 'local_offline',
    name: 'Local Sovereign',
    badge: '100% On-Premise (No API Keys)',
    icon: Laptop,
    tagline: 'Private on-premise execution with zero external API calls.',
    pipelineHighlights: {
      reasoning: 'Gemma 4 12B / Llama 3.1 8B / DeepSeek-R1 14B (Ollama)',
      vision: 'Gemma 4 E4B / Qwen3-VL + Local Docling DocLayNet',
      indexing: 'BAAI/bge-m3 (1024d Local → article_chunks)',
    },
    description:
      'Runs locally via Ollama hardware acceleration. Llama 3.1 8B, Gemma 4 12B, and DeepSeek R1 handle reasoning, Gemma 4 E4B / Qwen3-VL handle multimodal vision, and Docling handles 2D layout.',
    bindings: {
      query_planner: 'ollama_llama3',
      answerer: 'ollama_llama3',
      metadata_extraction: 'ollama_llama3',
      classification: 'ollama_llama3',
      article_segmentation: 'ollama_deepseek',
      visual_extraction: 'ollama_gemma4_e4b',
      layout_analysis: 'docling_parser',
      document_parser: 'docling_parser',
      ocr: 'docling_parser',
      embedding: 'local_embed_bge',
    },
  },
];

// ---------------------------------------------------------------------------
// Pipeline Tasks Grouped by Stage
// ---------------------------------------------------------------------------
const PIPELINE_STAGES = [
  {
    stageId: 'reasoning',
    title: 'Stage 1: Agentic Reasoning & Synthesis',
    description: 'Autonomous multi-step tool planning, query decomposition, and executive answer synthesis with citations.',
    icon: Sparkles,
    badgeColor: 'text-accent-editorial bg-accent-editorial/10 border-accent-editorial/30',
    tasks: [
      {
        id: 'query_planner',
        name: 'Query Planner',
        role: 'Autonomous tool sequence planner & sub-query generator',
        requiredCapability: 'supports_tool_use',
        requiredLabel: 'Requires Tool Calling',
        type: 'Reasoning LLM',
      },
      {
        id: 'answerer',
        name: 'Answer Synthesizer',
        role: 'Multi-newspaper factual synthesizer & citation linker',
        requiredCapability: 'supports_tool_use',
        requiredLabel: 'Requires Tool Calling',
        type: 'Synthesis LLM',
      },
    ],
  },
  {
    stageId: 'vision_ingestion',
    title: 'Stage 2: Vision & Broadsheet Ingestion',
    description: 'High-resolution newspaper layout parsing, OCR text extraction, and figure/chart analysis.',
    icon: Eye,
    badgeColor: 'text-accent-primary bg-accent-primary/10 border-accent-primary/30',
    tasks: [
      {
        id: 'layout_analysis',
        name: 'Layout Analysis',
        role: '2D broadsheet coordinate parser & column bounding',
        requiredCapability: 'supports_vision',
        requiredLabel: 'Requires Vision',
        type: 'VLM / Layout',
      },
      {
        id: 'document_parser',
        name: 'Document Parser',
        role: 'Hierarchical article DOM tree & reading-order assembler',
        requiredCapability: 'supports_vision',
        requiredLabel: 'Requires Vision',
        type: 'Layout Engine',
      },
      {
        id: 'ocr',
        name: 'High-Res OCR Engine',
        role: 'Glyph-to-text transcription for micro-print columns',
        requiredCapability: 'supports_vision',
        requiredLabel: 'Vision / OCR',
        type: 'OCR',
      },
      {
        id: 'visual_extraction',
        name: 'Visual & Photo Analysis',
        role: 'Photo, chart, infobox extraction & visual captioning',
        requiredCapability: 'supports_vision',
        requiredLabel: 'Requires Vision',
        type: 'VLM',
      },
    ],
  },
  {
    stageId: 'semantic_enrichment',
    title: 'Stage 3: Enrichment & Semantic Intelligence',
    description: 'Column boundary segmentation, entity recognition, topical classification, and dense vector embeddings.',
    icon: Database,
    badgeColor: 'text-accent-primary bg-accent-primary/10 border-accent-primary/30',
    tasks: [
      {
        id: 'article_segmentation',
        name: 'Article Segmentation',
        role: 'Splits raw layout blocks into unified logical articles',
        requiredCapability: null,
        requiredLabel: 'Text Reasoning',
        type: 'LLM',
      },
      {
        id: 'metadata_extraction',
        name: 'Metadata & NER Extraction',
        role: 'Extracts bylines, organizations, locations & event dates',
        requiredCapability: null,
        requiredLabel: 'Text Reasoning',
        type: 'LLM',
      },
      {
        id: 'classification',
        name: 'Topical Classification',
        role: 'Assigns broadsheet categories (Politics, Economy, Health, etc.)',
        requiredCapability: null,
        requiredLabel: 'Text Reasoning',
        type: 'LLM',
      },
      {
        id: 'embedding',
        name: 'Dense Vector Embedder',
        role: 'Multilingual dense vector generation for Qdrant hybrid search',
        requiredCapability: 'embedding_dim',
        requiredLabel: 'Requires Embedding Dim',
        type: 'Embedder',
      },
    ],
  },
];

// Pre-defined API catalog for API Studio
const API_ENDPOINTS = [
  {
    id: 'bindings',
    method: 'GET',
    name: 'Model Task Bindings',
    path: '/api/settings/model-bindings',
    description: 'Inspect active task-to-provider bindings, configured models, and live reachability status.',
    category: 'Settings',
  },
  {
    id: 'models_available',
    method: 'GET',
    name: 'Available Models Introspection',
    path: '/api/models/available',
    description: 'Comprehensive registry of configured providers with capabilities, context windows, and reachability.',
    category: 'Models',
  },
  {
    id: 'health',
    method: 'GET',
    name: 'System Infrastructure Health',
    path: '/health',
    description: 'Health check for MySQL database, Qdrant vector store, MinIO storage, and background workers.',
    category: 'System',
  },
  {
    id: 'newspapers',
    method: 'GET',
    name: 'Corpus Newspapers',
    path: '/api/newspapers',
    description: 'List all onboarded broadsheet publications, publishing cycles, and editions in the archive.',
    category: 'Corpus',
  },
  {
    id: 'issues',
    method: 'GET',
    name: 'Cataloged Newspaper Issues',
    path: '/api/issues?limit=10',
    description: 'Recent 10 newspaper issues with page manifests and article counts.',
    category: 'Corpus',
  },
  {
    id: 'query',
    method: 'POST',
    name: 'Agentic Conversational RAG',
    path: '/api/query',
    description: 'Execute multi-step CRAG research over newspaper articles and multimodal image chunks.',
    category: 'Agent',
  },
];

// ---------------------------------------------------------------------------
// Custom Classified Task Model Dropdown Component
// ---------------------------------------------------------------------------
function TaskModelDropdown({ taskId, value, onChange, reachabilityMap, configuredProviders }) {
  const [isOpen, setIsOpen] = useState(false);
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [customValue, setCustomValue] = useState('');
  const popoverRef = useRef(null);

  // Close when clicking outside
  useEffect(() => {
    function handleClickOutside(e) {
      if (popoverRef.current && !popoverRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // Determine which models can do this task based on task ID
  const eligibleModels = useMemo(() => {
    const list = Object.values(MODEL_CATALOG).filter((m) => {
      if (!m.supportsTasks) return false;
      return m.supportsTasks.includes(taskId);
    });

    // Also include configured providers from backend if they aren't already represented
    configuredProviders.forEach((cp) => {
      if (!list.some((m) => m.id === cp.id)) {
        // Classify based on capabilities
        let match = false;
        if (taskId === 'embedding' && cp.embedding_dim) match = true;
        if (['visual_extraction', 'layout_analysis', 'document_parser', 'ocr'].includes(taskId) && cp.supports_vision) match = true;
        if (['query_planner', 'answerer'].includes(taskId) && cp.supports_tool_use) match = true;
        if (['article_segmentation', 'metadata_extraction', 'classification'].includes(taskId)) match = true;

        if (match) {
          list.push({
            id: cp.id,
            name: reachabilityMap[cp.id]?.name || cp.model || cp.id,
            provider: cp.provider,
            category: cp.provider === 'ollama' ? 'local' : 'cloud',
            categoryLabel: cp.provider === 'ollama' ? 'Local Hardware' : 'Cloud Direct',
            context: cp.context_window ? `${Math.round(cp.context_window / 1000)}K ctx` : '',
            badges: [cp.provider],
            supportsTasks: [taskId],
          });
        }
      }
    });

    return list;
  }, [taskId, configuredProviders, reachabilityMap]);

  // Group eligible models by category
  const groupedModels = useMemo(() => {
    const groups = {};
    eligibleModels.forEach((m) => {
      const cat = m.categoryLabel || 'General Models';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(m);
    });
    return groups;
  }, [eligibleModels]);

  const currentModelMeta = MODEL_CATALOG[value] || configuredProviders.find((p) => p.id === value);
  const displayName = currentModelMeta?.name || currentModelMeta?.model || value;
  const isOnline = reachabilityMap[value]?.is_reachable === true;

  const handleCustomSubmit = (e) => {
    e.preventDefault();
    if (customValue.trim()) {
      onChange(customValue.trim());
      setShowCustomInput(false);
      setIsOpen(false);
    }
  };

  return (
    <div className="relative" ref={popoverRef}>
      {/* Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between gap-2.5 px-3 py-2 rounded-xl bg-surface hover:bg-elevated border border-border-subtle hover:border-accent-primary/50 text-xs font-mono transition-all text-left shadow-xs cursor-pointer group"
      >
        <div className="flex items-center gap-2 truncate">
          <span
            className={`w-2 h-2 rounded-full shrink-0 ${
              isOnline ? 'bg-accent-primary' : reachabilityMap[value]?.is_reachable === false ? 'bg-rose-400' : 'bg-slate-400'
            }`}
            title={isOnline ? 'Online & Reachable' : 'Offline / Unchecked'}
          />
          <span className="font-semibold text-text-primary truncate">{displayName}</span>
          {MODEL_CATALOG[value]?.context && (
            <span className="text-[10px] text-text-muted hidden sm:inline">
              ({MODEL_CATALOG[value].context})
            </span>
          )}
        </div>
        <ChevronDown className="w-3.5 h-3.5 text-text-muted group-hover:text-text-primary shrink-0 transition-transform duration-150" />
      </button>

      {/* Popover Dropdown */}
      {isOpen && (
        <div className="absolute right-0 top-full mt-1.5 w-80 sm:w-96 bg-surface border border-border-subtle rounded-2xl shadow-2xl p-2 z-50 text-xs font-sans animate-fade-in max-h-96 overflow-y-auto">
          <div className="px-3 py-2 border-b border-border-subtle flex items-center justify-between text-[11px] font-mono text-text-muted">
            <span className="font-bold uppercase tracking-wider text-accent-primary">Classified Compatible Models</span>
            <span>{eligibleModels.length} Qualified</span>
          </div>

          <div className="py-1 space-y-3">
            {Object.entries(groupedModels).map(([groupTitle, models]) => (
              <div key={groupTitle} className="space-y-1">
                <span className="px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider text-text-muted font-bold block">
                  {groupTitle}
                </span>

                {models.map((m) => {
                  const isSelected = m.id === value;
                  const isReachable = reachabilityMap[m.id]?.is_reachable === true;

                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => {
                        onChange(m.id);
                        setIsOpen(false);
                      }}
                      className={`w-full flex items-start justify-between p-2 rounded-xl text-left transition-colors cursor-pointer ${
                        isSelected
                          ? 'bg-accent-primary/10 border border-accent-primary/30 text-text-primary'
                          : 'hover:bg-elevated text-text-secondary hover:text-text-primary'
                      }`}
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-1.5">
                          <span
                            className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                              isReachable ? 'bg-accent-primary' : 'bg-slate-400'
                            }`}
                          />
                          <span className="font-semibold text-xs text-text-primary">{m.name}</span>
                        </div>
                        <p className="text-[10px] font-mono text-text-muted">{m.provider}</p>
                        {Array.isArray(m.badges) && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {m.badges.map((b, bIdx) => (
                              <span
                                key={bIdx}
                                className="text-[9px] font-mono px-1 py-0.2 rounded bg-elevated border border-border-subtle text-text-secondary"
                              >
                                {b}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      {isSelected && <Check className="w-4 h-4 text-accent-primary shrink-0 mt-0.5" />}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Custom Model Option */}
          <div className="pt-2 border-t border-border-subtle font-mono">
            {showCustomInput ? (
              <form onSubmit={handleCustomSubmit} className="p-2 bg-canvas rounded-xl space-y-2">
                <label className="block text-[10px] text-text-muted">Enter Custom Model or Provider String:</label>
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    value={customValue}
                    onChange={(e) => setCustomValue(e.target.value)}
                    placeholder="e.g. ollama/mistral:latest"
                    autoFocus
                    className="flex-1 bg-surface border border-border-subtle rounded-lg px-2.5 py-1 text-xs text-text-primary outline-none focus:border-accent-primary"
                  />
                  <button
                    type="submit"
                    className="px-2.5 py-1 bg-accent-primary text-white rounded-lg text-xs font-semibold hover:opacity-90"
                  >
                    Set
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowCustomInput(false)}
                    className="p-1 text-text-muted hover:text-text-primary"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              </form>
            ) : (
              <button
                type="button"
                onClick={() => setShowCustomInput(true)}
                className="w-full py-1.5 text-center text-[11px] text-accent-primary hover:bg-elevated rounded-xl transition-colors cursor-pointer flex items-center justify-center gap-1.5"
              >
                <Plus className="w-3 h-3" />
                <span>+ Custom Model / Node Override...</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main ModelSettingsStudio Component
// ---------------------------------------------------------------------------
export default function ModelSettingsStudio() {
  const { updateTaskBindings, taskBindings, refreshTaskBindings } = useActiveHighlight();

  // Navigation Sub-views: 'presets' | 'pipeline' | 'providers' | 'api'
  const [activeSubView, setActiveSubView] = useState('pipeline');

  // Sovereign Laptop Node State
  const [laptopState, setLaptopState] = useState({
    enabled: false,
    online: false,
    available_models: [],
    default_model: 'llama3.1:8b',
    tunnel_url: '',
  });
  const [adminPin, setAdminPin] = useState('');
  const [tunnelUrlInput, setTunnelUrlInput] = useState('');
  const [laptopLoading, setLaptopLoading] = useState(false);

  // Backend Data State
  const [configuredProviders, setConfiguredProviders] = useState([]);
  const [reachabilityList, setReachabilityList] = useState([]);
  const [currentBindings, setCurrentBindings] = useState(() => taskBindings || {});
  const [loadingInitial, setLoadingInitial] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [statusNotification, setStatusNotification] = useState(null);

  // Granular Pipeline Task Staging State
  const [stagedBindings, setStagedBindings] = useState(() => ({ ...(taskBindings || {}) }));
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [autoSave, setAutoSave] = useState(true);

  // Provider Registry Search & Filter
  const [providerSearch, setProviderSearch] = useState('');
  const [providerFilter, setProviderFilter] = useState('all');

  // API Studio State
  const [apiEndpoint, setApiEndpoint] = useState('/api/settings/model-bindings');
  const [apiMethod, setApiMethod] = useState('GET');
  const [apiData, setApiData] = useState(null);
  const [apiLoading, setApiLoading] = useState(false);
  const [apiLatency, setApiLatency] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);
  const [apiPayloadSize, setApiPayloadSize] = useState(null);
  const [apiSearchQuery, setApiSearchQuery] = useState('');
  const [apiCopied, setApiCopied] = useState(false);

  // Synchronize with ActiveHighlightContext
  useEffect(() => {
    if (taskBindings && Object.keys(taskBindings).length > 0) {
      setCurrentBindings(taskBindings);
      setStagedBindings((prev) => ({ ...taskBindings, ...prev }));
    }
  }, [taskBindings]);

  // Load System Settings
  const loadSystemSettings = useCallback(
    async (isInitial = false) => {
      if (isInitial) setLoadingInitial(true);
      else setRefreshing(true);

      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4000);

        const [bindingsRes, availableRes] = await Promise.all([
          fetch('/api/settings/model-bindings', { signal: controller.signal }),
          fetch('/api/models/available', { signal: controller.signal }),
        ]);

        clearTimeout(timeoutId);

        if (bindingsRes.ok) {
          const bData = await bindingsRes.json();
          if (bData.task_bindings) {
            setCurrentBindings(bData.task_bindings);
            setStagedBindings(bData.task_bindings);
          }
        }

        if (availableRes.ok) {
          const aData = await availableRes.json();
          if (Array.isArray(aData.configured_providers)) {
            setConfiguredProviders(aData.configured_providers);
          }
          if (Array.isArray(aData.reachability)) {
            setReachabilityList(aData.reachability);
          }
        }
      } catch (err) {
        console.warn('Backend unavailable, using default model topology:', err);
      } finally {
        if (isInitial) setLoadingInitial(false);
        else setRefreshing(false);
      }
    },
    []
  );

  useEffect(() => {
    loadSystemSettings(true);
  }, [loadSystemSettings]);

  // Reachability map lookup
  const reachabilityMap = useMemo(() => {
    const map = {};
    reachabilityList.forEach((r) => {
      map[r.provider_id] = r;
    });
    return map;
  }, [reachabilityList]);

  // Detect Unsaved Changes
  useEffect(() => {
    let diff = false;
    const allKeys = Array.from(new Set([...Object.keys(currentBindings), ...Object.keys(stagedBindings)]));
    for (const key of allKeys) {
      if (currentBindings[key] !== stagedBindings[key]) {
        diff = true;
        break;
      }
    }
    setHasUnsavedChanges(diff);
  }, [currentBindings, stagedBindings]);

  // Stage a Task Binding Change
  const handleStageTaskBinding = (taskId, providerId) => {
    const updated = { ...stagedBindings, [taskId]: providerId };
    setStagedBindings(updated);

    if (autoSave) {
      handleSaveTaskBindings(updated);
    }
  };

  // Save Task Bindings to Backend & Global Context
  const handleSaveTaskBindings = async (bindingsToSave = stagedBindings) => {
    try {
      const res = await fetch('/api/settings/model-bindings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_bindings: bindingsToSave }),
      });

      if (res.ok) {
        const data = await res.json();
        const newBindings = data.task_bindings || bindingsToSave;
        setCurrentBindings(newBindings);
        setStagedBindings(newBindings);
        setHasUnsavedChanges(false);

        if (updateTaskBindings) {
          updateTaskBindings(newBindings);
        }

        setStatusNotification({
          type: 'success',
          message: 'Model task matrix updated and broadcast across all workspaces.',
        });
        setTimeout(() => setStatusNotification(null), 4000);
      } else {
        // Fallback optimistic apply
        setCurrentBindings(bindingsToSave);
        setHasUnsavedChanges(false);
        if (updateTaskBindings) {
          updateTaskBindings(bindingsToSave);
        }
        setStatusNotification({
          type: 'success',
          message: 'Model bindings applied locally to active workspace session.',
        });
        setTimeout(() => setStatusNotification(null), 3000);
      }
    } catch (err) {
      setCurrentBindings(bindingsToSave);
      setHasUnsavedChanges(false);
      if (updateTaskBindings) {
        updateTaskBindings(bindingsToSave);
      }
      setStatusNotification({
        type: 'success',
        message: 'Model bindings applied locally to active workspace session.',
      });
      setTimeout(() => setStatusNotification(null), 3000);
    }
  };

  // Apply Architecture Preset
  const handleApplyPreset = async (preset) => {
    setStagedBindings(preset.bindings);
    await handleSaveTaskBindings(preset.bindings);
  };

  // Determine Active Architecture Preset
  const activePreset = useMemo(() => {
    for (const profile of PRESET_PROFILES) {
      let match = true;
      for (const [task, expectedProvider] of Object.entries(profile.bindings)) {
        if (currentBindings[task] !== expectedProvider) {
          match = false;
          break;
        }
      }
      if (match) return profile;
    }
    return null;
  }, [currentBindings]);

  // Execute API Request in Studio
  const handleExecuteApi = async () => {
    setApiLoading(true);
    setApiData(null);
    const start = performance.now();

    try {
      const res = await fetch(apiEndpoint, { method: apiMethod });
      const duration = Math.round(performance.now() - start);
      setApiLatency(duration);
      setApiStatus(res.status);

      const json = await res.json().catch(() => ({ status: 'Non-JSON response' }));
      setApiData(json);
      setApiPayloadSize(JSON.stringify(json).length);
    } catch (err) {
      setApiStatus(500);
      setApiData({ error: err.message, note: 'Failed to communicate with endpoint' });
      setApiLatency(Math.round(performance.now() - start));
    } finally {
      setApiLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0 bg-canvas text-text-primary w-full max-w-7xl 2xl:max-w-[1600px] mx-auto px-3 sm:px-6 md:px-8 py-4 sm:py-6 overflow-y-auto space-y-6 font-sans transition-colors duration-200">
      {/* Top Header & Editorial Folio */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border-subtle">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-mono tracking-widest text-accent-editorial uppercase font-bold px-2 py-0.5 rounded bg-accent-editorial/10 border border-accent-editorial/30">
              Neural Studio · Runtime Telemetry
            </span>
            <span className="text-border-subtle">•</span>
            <span className="text-[11px] font-mono text-text-muted">
              Model Orchestration & API Matrix
            </span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold font-serif text-text-primary tracking-tight flex items-center gap-2.5">
            <Sliders className="w-7 h-7 text-accent-primary" />
            Model Architecture & API Studio
          </h1>
          <p className="text-xs text-text-secondary mt-1">
            Dynamic runtime model orchestration, task capability filtering, local laptop node bridge, and live API telemetry.
          </p>
        </div>

        {/* Global Controls & Status Pill */}
        <div className="flex items-center gap-2.5 flex-wrap">
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-border-subtle bg-surface text-xs font-mono shadow-xs">
            <span
              className={`w-2 h-2 rounded-full animate-pulse ${
                activePreset?.id === 'cloud_full'
                  ? 'bg-sky-400'
                  : activePreset?.id === 'cloud_hybrid'
                  ? 'bg-accent-primary'
                  : 'bg-amber-400'
              }`}
            />
            <span className="font-semibold text-text-primary">
              {activePreset ? activePreset.name : 'Custom Pipeline Matrix'}
            </span>
          </div>

          <button
            type="button"
            onClick={() => loadSystemSettings(false)}
            disabled={refreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-surface hover:bg-elevated text-xs text-text-secondary hover:text-text-primary transition-all border border-border-subtle cursor-pointer font-mono shadow-xs"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-accent-primary ${refreshing ? 'animate-spin' : ''}`} />
            <span>Introspect</span>
          </button>
        </div>
      </header>

      {/* Notification Banner */}
      {statusNotification && (
        <div
          className={`p-3 px-4 rounded-xl text-xs font-mono flex items-center justify-between border ${
            statusNotification.type === 'success'
              ? 'bg-accent-primary/10 border-accent-primary/30 text-accent-primary'
              : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
          }`}
        >
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span>{statusNotification.message}</span>
          </div>
          <button type="button" onClick={() => setStatusNotification(null)} className="text-text-muted hover:text-text-primary">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Sovereign Laptop Node Card (Local Ollama Bridge) */}
      <div className="bg-surface border border-border-subtle rounded-2xl p-5 shadow-xs space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border-subtle">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-elevated border border-border-subtle flex items-center justify-center text-accent-editorial">
              <Laptop className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-serif font-bold text-sm text-text-primary flex items-center gap-2">
                Sovereign Laptop Local LLM Bridge
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-elevated border border-border-subtle text-text-muted">
                  {laptopState.online ? '🟢 Connected (Ollama Online)' : '⚪ Standby / Offline'}
                </span>
              </h3>
              <p className="text-[11px] text-text-secondary mt-0.5">
                Route queries to Ollama on your personal laptop GPU/NPU with zero cloud token consumption and 100% data privacy.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="text"
              value={tunnelUrlInput}
              onChange={(e) => setTunnelUrlInput(e.target.value)}
              placeholder="http://localhost:11434"
              className="bg-canvas border border-border-subtle rounded-xl px-3 py-1.5 text-xs text-text-primary outline-none focus:border-accent-primary font-mono w-48 sm:w-56"
            />
            <button
              type="button"
              onClick={() => {
                setStatusNotification({
                  type: 'success',
                  message: 'Pinging Ollama node at ' + (tunnelUrlInput || 'http://localhost:11434') + '...',
                });
              }}
              className="px-3 py-1.5 rounded-xl bg-accent-editorial/10 hover:bg-accent-editorial/20 text-accent-editorial text-xs font-semibold font-mono border border-accent-editorial/30 transition-all cursor-pointer"
            >
              Ping Node
            </button>
          </div>
        </div>
      </div>

      {/* Segmented Sub-View Switcher */}
      <div className="flex items-center justify-between border-b border-border-subtle pb-3">
        <div className="flex items-center gap-1.5 bg-surface p-1 rounded-xl border border-border-subtle font-mono text-xs">
          <button
            type="button"
            onClick={() => setActiveSubView('pipeline')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
              activeSubView === 'pipeline'
                ? 'bg-accent-primary text-white shadow-xs'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Task Bindings Matrix</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveSubView('presets')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
              activeSubView === 'presets'
                ? 'bg-accent-primary text-white shadow-xs'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Cloud className="w-3.5 h-3.5" />
            <span>Architecture Presets</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveSubView('providers')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
              activeSubView === 'providers'
                ? 'bg-accent-primary text-white shadow-xs'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Server className="w-3.5 h-3.5" />
            <span>Provider Registry</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveSubView('api')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
              activeSubView === 'api'
                ? 'bg-accent-primary text-white shadow-xs'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            <span>API Studio</span>
          </button>
        </div>

        {/* Auto-save Toggle */}
        <label className="hidden sm:flex items-center gap-2 text-xs font-mono text-text-muted cursor-pointer select-none">
          <input
            type="checkbox"
            checked={autoSave}
            onChange={(e) => setAutoSave(e.target.checked)}
            className="rounded border-border-subtle text-accent-primary focus:ring-accent-primary"
          />
          <span>Auto-save changes to backend</span>
        </label>
      </div>

      {/* SUBVIEW 1: GRANULAR PIPELINE TASK MATRIX */}
      {activeSubView === 'pipeline' && (
        <div className="space-y-6">
          <div className="p-4 rounded-xl bg-surface border border-border-subtle text-xs space-y-1">
            <span className="font-serif font-bold text-text-primary block text-sm">
              Classified Model Binding Matrix:
            </span>
            <p className="text-text-secondary leading-relaxed">
              Each pipeline task only displays models verified to support its specific functional requirements (e.g. Reasoning, Tool Calling, Multimodal Vision, or Dense Embeddings). Selecting a model stages or persists changes across all tabs.
            </p>
          </div>

          <div className="space-y-6">
            {PIPELINE_STAGES.map((stage) => {
              const StageIcon = stage.icon;
              return (
                <div key={stage.stageId} className="bg-surface border border-border-subtle rounded-2xl p-5 shadow-xs space-y-4">
                  <div className="flex items-center justify-between pb-3 border-b border-border-subtle">
                    <div className="flex items-center gap-2.5">
                      <div className="w-7 h-7 rounded-lg bg-elevated border border-border-subtle flex items-center justify-center text-accent-primary">
                        <StageIcon className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="font-serif font-bold text-sm text-text-primary">{stage.title}</h3>
                        <p className="text-[11px] text-text-muted">{stage.description}</p>
                      </div>
                    </div>
                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${stage.badgeColor}`}>
                      {stage.tasks.length} Active Tasks
                    </span>
                  </div>

                  {/* Task Rows */}
                  <div className="grid grid-cols-1 gap-3">
                    {stage.tasks.map((task) => {
                      const selectedProviderId = stagedBindings[task.id] || currentBindings[task.id] || '';
                      const isModified = stagedBindings[task.id] !== currentBindings[task.id];

                      return (
                        <div
                          key={task.id}
                          className={`p-3.5 rounded-xl border transition-all flex flex-col md:flex-row md:items-center justify-between gap-3 ${
                            isModified
                              ? 'bg-amber-500/10 border-amber-500/40 ring-1 ring-amber-500/30'
                              : 'bg-canvas border-border-subtle hover:border-border-subtle/80'
                          }`}
                        >
                          <div className="space-y-1 flex-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-semibold text-xs text-text-primary font-sans">{task.name}</span>
                              <code className="text-[10px] font-mono text-accent-primary bg-elevated px-1.5 py-0.5 rounded border border-border-subtle">
                                {task.id}
                              </code>
                              <span className="text-[10px] text-text-muted font-mono px-1.5 py-0.5 rounded bg-elevated">
                                {task.type}
                              </span>
                              {task.requiredLabel && (
                                <span className="text-[10px] text-accent-editorial font-mono px-1.5 py-0.5 rounded bg-accent-editorial/10 border border-accent-editorial/30">
                                  {task.requiredLabel}
                                </span>
                              )}
                              {isModified && (
                                <span className="text-[10px] font-bold text-amber-400 bg-amber-500/10 border border-amber-500/30 px-1.5 py-0.5 rounded animate-pulse">
                                  Modified
                                </span>
                              )}
                            </div>
                            <p className="text-[11px] text-text-secondary">{task.role}</p>
                          </div>

                          {/* Custom Classified Dropdown */}
                          <div className="min-w-[280px] md:min-w-[340px]">
                            <TaskModelDropdown
                              taskId={task.id}
                              value={selectedProviderId}
                              onChange={(newVal) => handleStageTaskBinding(task.id, newVal)}
                              reachabilityMap={reachabilityMap}
                              configuredProviders={configuredProviders}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* SUBVIEW 2: ARCHITECTURE PRESETS */}
      {activeSubView === 'presets' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {PRESET_PROFILES.map((profile) => {
            const Icon = profile.icon;
            const isCurrent = activePreset?.id === profile.id;

            return (
              <div
                key={profile.id}
                className={`bg-surface border rounded-2xl p-6 shadow-xs flex flex-col justify-between transition-all ${
                  isCurrent
                    ? 'border-accent-primary ring-2 ring-accent-primary/30 shadow-lg'
                    : 'border-border-subtle hover:border-accent-primary/40'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between pb-3 mb-3 border-b border-border-subtle">
                    <div className="flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-xl bg-elevated border border-border-subtle flex items-center justify-center text-accent-primary">
                        <Icon className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="font-serif font-bold text-base text-text-primary">{profile.name}</h3>
                        <span className="text-[10px] font-mono text-accent-editorial block font-semibold">
                          {profile.badge}
                        </span>
                      </div>
                    </div>
                  </div>

                  <p className="text-xs text-text-secondary leading-relaxed mb-4">{profile.description}</p>

                  {/* Highlights */}
                  <div className="space-y-2 py-3 px-3.5 rounded-xl bg-canvas border border-border-subtle text-xs font-mono mb-4">
                    <div>
                      <span className="text-[10px] text-text-muted block uppercase">Reasoning & Synthesis:</span>
                      <span className="text-text-primary text-[11px] font-semibold">{profile.pipelineHighlights.reasoning}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-text-muted block uppercase">Vision & Layout:</span>
                      <span className="text-accent-primary text-[11px] font-semibold">{profile.pipelineHighlights.vision}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-text-muted block uppercase">Vector Index:</span>
                      <span className="text-accent-editorial text-[11px] font-semibold">{profile.pipelineHighlights.indexing}</span>
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => handleApplyPreset(profile)}
                  className={`w-full py-2.5 px-4 rounded-xl font-semibold text-xs transition-all cursor-pointer flex items-center justify-center gap-2 ${
                    isCurrent
                      ? 'bg-accent-primary/20 text-accent-primary border border-accent-primary/40 cursor-default'
                      : 'bg-accent-primary hover:opacity-90 text-white shadow-sm'
                  }`}
                >
                  {isCurrent ? (
                    <>
                      <CheckCircle2 className="w-4 h-4" />
                      <span>Active Operating Profile</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      <span>Apply Profile</span>
                    </>
                  )}
                </button>
              </div>
            );
          })}
        </div>
      )}

      {/* SUBVIEW 3: PROVIDER REGISTRY */}
      {activeSubView === 'providers' && (
        <div className="bg-surface border border-border-subtle rounded-2xl p-6 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border-subtle">
            <h3 className="font-serif font-bold text-base text-text-primary flex items-center gap-2">
              <Server className="w-4 h-4 text-accent-primary" />
              Configured Model Providers ({configuredProviders.length})
            </h3>
            <div className="relative w-full sm:w-64">
              <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-text-muted" />
              <input
                type="text"
                value={providerSearch}
                onChange={(e) => setProviderSearch(e.target.value)}
                placeholder="Filter providers..."
                className="w-full bg-canvas border border-border-subtle rounded-xl pl-9 pr-3 py-1.5 text-xs text-text-primary outline-none focus:border-accent-primary font-mono"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {configuredProviders
              .filter(
                (p) =>
                  !providerSearch ||
                  p.id.toLowerCase().includes(providerSearch.toLowerCase()) ||
                  (p.model && p.model.toLowerCase().includes(providerSearch.toLowerCase()))
              )
              .map((p) => {
                const reach = reachabilityMap[p.id];
                const isReachable = reach?.is_reachable === true;

                return (
                  <div key={p.id} className="p-3.5 rounded-xl bg-canvas border border-border-subtle space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-text-primary font-mono">{p.id}</span>
                      <span
                        className={`text-[9px] font-mono px-2 py-0.5 rounded-full border ${
                          isReachable
                            ? 'bg-accent-primary/10 text-accent-primary border-accent-primary/30'
                            : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                        }`}
                      >
                        {isReachable ? 'Online' : 'Offline'}
                      </span>
                    </div>
                    <div className="text-[11px] text-text-secondary font-mono">
                      <span>Model: {p.model || 'N/A'}</span>
                      <span className="block text-text-muted">Provider: {p.provider}</span>
                    </div>
                    <div className="flex flex-wrap gap-1 pt-1 border-t border-border-subtle/60 text-[9px] font-mono text-text-muted">
                      {p.supports_vision && <span className="bg-elevated px-1.5 py-0.2 rounded">Vision</span>}
                      {p.supports_tool_use && <span className="bg-elevated px-1.5 py-0.2 rounded">Tool Calling</span>}
                      {p.embedding_dim && <span className="bg-elevated px-1.5 py-0.2 rounded">{p.embedding_dim}d</span>}
                      {p.context_window && (
                        <span className="bg-elevated px-1.5 py-0.2 rounded">
                          {Math.round(p.context_window / 1000)}K ctx
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* SUBVIEW 4: INTERACTIVE API STUDIO */}
      {activeSubView === 'api' && (
        <div className="bg-surface border border-border-subtle rounded-2xl p-6 shadow-xs space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-border-subtle">
            <h3 className="font-serif font-bold text-base text-text-primary flex items-center gap-2">
              <Code2 className="w-4 h-4 text-accent-primary" />
              Live Interactive API Studio
            </h3>
            <span className="text-xs font-mono text-text-muted">FastAPI Endpoints</span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Endpoints Sidebar */}
            <div className="space-y-1.5">
              <span className="text-[10px] font-mono uppercase tracking-wider text-text-muted font-bold block mb-2">
                Available System Endpoints:
              </span>
              {API_ENDPOINTS.map((ep) => (
                <button
                  key={ep.id}
                  type="button"
                  onClick={() => {
                    setApiEndpoint(ep.path);
                    setApiMethod(ep.method);
                  }}
                  className={`w-full text-left p-2.5 rounded-xl border text-xs transition-colors cursor-pointer ${
                    apiEndpoint === ep.path
                      ? 'bg-accent-primary/10 border-accent-primary/40 text-text-primary'
                      : 'bg-canvas border-border-subtle hover:bg-elevated text-text-secondary'
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-mono mb-1">
                    <span className="text-[9px] font-bold px-1.5 py-0.2 rounded bg-elevated text-accent-editorial">
                      {ep.method}
                    </span>
                    <span className="font-semibold text-text-primary">{ep.name}</span>
                  </div>
                  <p className="text-[10px] text-text-muted font-mono truncate">{ep.path}</p>
                </button>
              ))}
            </div>

            {/* Request & Response Inspector */}
            <div className="lg:col-span-2 space-y-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-bold px-2 py-1.5 rounded-xl bg-elevated border border-border-subtle text-accent-editorial">
                  {apiMethod}
                </span>
                <input
                  type="text"
                  value={apiEndpoint}
                  onChange={(e) => setApiEndpoint(e.target.value)}
                  className="flex-1 bg-canvas border border-border-subtle rounded-xl px-3 py-1.5 text-xs text-text-primary font-mono outline-none focus:border-accent-primary"
                />
                <button
                  type="button"
                  onClick={handleExecuteApi}
                  disabled={apiLoading}
                  className="px-4 py-1.5 rounded-xl bg-accent-primary hover:opacity-90 disabled:opacity-40 text-white font-semibold text-xs flex items-center gap-1.5 transition-all cursor-pointer font-mono shadow-xs"
                >
                  {apiLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  <span>Send</span>
                </button>
              </div>

              {/* Response Viewer */}
              <div className="bg-canvas border border-border-subtle rounded-xl p-4 font-mono text-xs max-h-96 overflow-auto">
                <div className="flex items-center justify-between pb-2 mb-2 border-b border-border-subtle text-[11px] text-text-muted">
                  <span>
                    Status: <strong className={apiStatus === 200 ? 'text-accent-primary' : 'text-rose-400'}>{apiStatus || 'Ready'}</strong>
                  </span>
                  {apiLatency && <span>Latency: {apiLatency} ms</span>}
                </div>
                <pre className="text-[11px] text-text-secondary leading-relaxed overflow-x-auto">
                  {apiData ? JSON.stringify(apiData, null, 2) : '// Click Send to execute request and introspect response payload'}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Sticky Bottom Bar for Unsaved Changes (if autoSave is unchecked) */}
      {!autoSave && hasUnsavedChanges && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-surface border-2 border-accent-primary rounded-2xl p-4 shadow-2xl flex items-center gap-4 z-40 animate-bounce-short">
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" />
            <span className="font-semibold text-text-primary">Unsaved Task Bindings Pending</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setStagedBindings(currentBindings)}
              className="px-3 py-1.5 rounded-xl bg-canvas hover:bg-elevated text-xs font-mono text-text-muted hover:text-text-primary border border-border-subtle transition-colors cursor-pointer"
            >
              Discard
            </button>
            <button
              type="button"
              onClick={() => handleSaveTaskBindings(stagedBindings)}
              className="px-4 py-1.5 rounded-xl bg-accent-primary hover:opacity-90 text-white font-semibold text-xs font-mono transition-all cursor-pointer shadow-sm"
            >
              Save Changes
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
