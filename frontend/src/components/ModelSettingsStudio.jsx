import React, { useState, useEffect, useMemo, useCallback } from 'react';
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
} from 'lucide-react';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

// ---------------------------------------------------------------------------
// Architecture Presets Configuration
// ---------------------------------------------------------------------------
const PRESET_PROFILES = [
  {
    id: 'cloud_full',
    name: 'Full Cloud Dual-Key',
    badge: 'Zero Local Stress',
    icon: Cloud,
    tagline: 'Recommended for instant setup with 0% local GPU / CPU resource consumption.',
    theme: {
      border: 'border-emerald-500/50 hover:border-emerald-400',
      activeBorder: 'border-emerald-400 ring-2 ring-emerald-500/40 shadow-emerald-950/50 shadow-xl',
      bg: 'bg-emerald-950/20 hover:bg-emerald-950/30',
      badgeBg: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
      text: 'text-emerald-400',
      btn: 'bg-emerald-600 hover:bg-emerald-500 text-white',
    },
    pipelineHighlights: {
      reasoning: 'Google Gemma 4 26B (262k ctx)',
      vision: 'Google Cloud Vision OCR',
      indexing: 'BAAI BGE-M3 (1024d)',
    },
    description:
      'Routes all LLM reasoning, article segmentation, and visual inspection through OpenRouter Dual-Key Gemma 4 26B, paired with Google Cloud Vision OCR for broadsheet layout parsing.',
    bindings: {
      query_planner: 'openrouter_gemma4_26b',
      answerer: 'openrouter_gemma4_26b',
      metadata_extraction: 'openrouter_gemma4_26b',
      classification: 'openrouter_gemma4_26b',
      article_segmentation: 'openrouter_gemma4_26b',
      visual_extraction: 'openrouter_gemma4_26b',
      layout_analysis: 'google_cloud_vision',
      document_parser: 'google_cloud_vision',
      ocr: 'google_cloud_vision',
      embedding: 'local_embed_bge',
    },
  },
  {
    id: 'cloud_hybrid',
    name: 'High-Speed Hybrid',
    badge: 'Lightning Fast',
    icon: Zap,
    tagline: 'Ultra-low latency reasoning and specialized 2D document parsing.',
    theme: {
      border: 'border-sky-500/50 hover:border-sky-400',
      activeBorder: 'border-sky-400 ring-2 ring-sky-500/40 shadow-sky-950/50 shadow-xl',
      bg: 'bg-sky-950/20 hover:bg-sky-950/30',
      badgeBg: 'bg-sky-500/10 text-sky-300 border-sky-500/30',
      text: 'text-sky-400',
      btn: 'bg-sky-600 hover:bg-sky-500 text-white',
    },
    pipelineHighlights: {
      reasoning: 'NVIDIA Nemotron 3.5 Lightning (1M ctx)',
      vision: 'Docling 2D Layout Engine + Cloud Vision',
      indexing: 'BAAI BGE-M3 (1024d)',
    },
    description:
      'Employs NVIDIA Nemotron 3.5 Lightning with 1M context window for rapid synthesis, paired with IBM Docling 2D for table & column geometry parsing and Gemma 4 for visual extraction.',
    bindings: {
      query_planner: 'openrouter_nemotron',
      answerer: 'openrouter_nemotron',
      metadata_extraction: 'openrouter_gemma4_26b',
      classification: 'openrouter_nemotron',
      article_segmentation: 'openrouter_gemma4_26b',
      visual_extraction: 'openrouter_gemma4_26b',
      layout_analysis: 'docling_parser',
      document_parser: 'docling_parser',
      ocr: 'google_cloud_vision',
      embedding: 'local_embed_bge',
    },
  },
  {
    id: 'local_offline',
    name: 'Local Sovereign',
    badge: '100% Air-Gapped',
    icon: Laptop,
    tagline: 'Private on-premise execution with zero external API calls.',
    theme: {
      border: 'border-amber-500/50 hover:border-amber-400',
      activeBorder: 'border-amber-400 ring-2 ring-amber-500/40 shadow-amber-950/50 shadow-xl',
      bg: 'bg-amber-950/20 hover:bg-amber-950/30',
      badgeBg: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
      text: 'text-amber-400',
      btn: 'bg-amber-600 hover:bg-amber-500 text-white',
    },
    pipelineHighlights: {
      reasoning: 'Meta Llama 3.1 8B + DeepSeek R1 14B',
      vision: 'Qwen 3 VL + Docling Layout',
      indexing: 'BAAI BGE-M3 (1024d)',
    },
    description:
      'Runs locally via Ollama hardware acceleration. Llama 3.1 8B handles fast planning and synthesis, DeepSeek R1 handles reasoning, Qwen 3 VL handles images, and Docling handles 2D layout.',
    bindings: {
      query_planner: 'ollama_llama3',
      answerer: 'ollama_llama3',
      metadata_extraction: 'ollama_llama3',
      classification: 'ollama_llama3',
      article_segmentation: 'ollama_deepseek',
      visual_extraction: 'ollama_qwen3vl',
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
    badgeColor: 'text-purple-400 bg-purple-950/40 border-purple-800',
    tasks: [
      {
        id: 'query_planner',
        name: 'Query Planner',
        role: 'Autonomous tool sequence planner & sub-query generator',
        requiredCapability: 'supports_tool_use',
        requiredLabel: 'Requires Tool Use',
        type: 'LLM',
      },
      {
        id: 'answerer',
        name: 'Answer Synthesizer',
        role: 'Multi-newspaper factual synthesizer & citation linker',
        requiredCapability: 'supports_tool_use',
        requiredLabel: 'Requires Tool Use',
        type: 'LLM',
      },
    ],
  },
  {
    stageId: 'vision_ingestion',
    title: 'Stage 2: Vision & Broadsheet Ingestion',
    description: 'High-resolution newspaper layout parsing, OCR text extraction, and figure/chart analysis.',
    icon: Eye,
    badgeColor: 'text-cyan-400 bg-cyan-950/40 border-cyan-800',
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
    badgeColor: 'text-emerald-400 bg-emerald-950/40 border-emerald-800',
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
    id: 'reset_bindings',
    method: 'POST',
    name: 'Reset Task Bindings',
    path: '/api/settings/model-bindings/reset',
    description: 'Restore all pipeline task bindings back to system defaults.',
    category: 'Settings',
  },
];

export default function ModelSettingsStudio() {
  const { updateTaskBindings, taskBindings, refreshTaskBindings } = useActiveHighlight();

  // Navigation Sub-views: 'presets' | 'pipeline' | 'providers' | 'api'
  const [activeSubView, setActiveSubView] = useState('presets');

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

  // Provider Registry Search & Filter
  const [providerSearch, setProviderSearch] = useState('');
  const [providerFilter, setProviderFilter] = useState('all'); // 'all' | 'reachable' | 'cloud' | 'local' | 'vision' | 'tools'

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

  // Load Model Settings and Provider Status from backend
  const loadSystemSettings = useCallback(async (isSilent = false) => {
    if (!isSilent) setRefreshing(true);
    try {
      const res = await fetch('/api/settings/model-bindings');
      if (res.ok) {
        const json = await res.json();
        setConfiguredProviders(json.configured_providers || []);
        setReachabilityList(json.provider_reachability || []);
        if (json.task_bindings) {
          setCurrentBindings(json.task_bindings);
          setStagedBindings(json.task_bindings);
          updateTaskBindings(json.task_bindings, false);
          setHasUnsavedChanges(false);
        }
      }
    } catch (err) {
      console.error('Failed to load model settings:', err);
      setStatusNotification({
        type: 'error',
        message: `Failed to fetch settings: ${err.message}`,
      });
    } finally {
      setLoadingInitial(false);
      setRefreshing(false);
    }
  }, [updateTaskBindings]);

  useEffect(() => {
    loadSystemSettings();
    executeApiRequest('/api/settings/model-bindings', 'GET');
  }, [loadSystemSettings]);

  // Toast Notification Auto-dismiss
  useEffect(() => {
    if (statusNotification) {
      const timer = setTimeout(() => {
        setStatusNotification(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [statusNotification]);

  // Check if a preset is currently active
  const isPresetActive = useCallback(
    (preset) => {
      if (!currentBindings || Object.keys(currentBindings).length === 0) return false;
      return Object.entries(preset.bindings).every(
        ([task, providerId]) => currentBindings[task] === providerId
      );
    },
    [currentBindings]
  );

  // Active preset object if one matches
  const activePreset = useMemo(() => {
    return PRESET_PROFILES.find((p) => isPresetActive(p)) || null;
  }, [isPresetActive]);

  // 1-Click Apply Preset
  const handleApplyPreset = async (preset) => {
    setStatusNotification({
      type: 'loading',
      message: `Activating ${preset.name}...`,
    });
    try {
      const res = await fetch('/api/settings/model-bindings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_bindings: preset.bindings }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`);
      }

      const newBindings = data.task_bindings || preset.bindings;
      setCurrentBindings(newBindings);
      setStagedBindings(newBindings);
      setHasUnsavedChanges(false);
      updateTaskBindings(newBindings, true);

      setStatusNotification({
        type: 'success',
        message: `Activated ${preset.name}! Configuration synchronized across all workspace tabs.`,
      });

      // Refresh API Studio if showing bindings
      if (apiEndpoint === '/api/settings/model-bindings') {
        executeApiRequest('/api/settings/model-bindings', 'GET');
      }
    } catch (err) {
      setStatusNotification({
        type: 'error',
        message: `Failed to activate preset: ${err.message}`,
      });
    }
  };

  // Reset to Factory Defaults
  const handleResetDefaults = async () => {
    if (!window.confirm('Reset all task-provider bindings back to system defaults?')) {
      return;
    }
    setStatusNotification({
      type: 'loading',
      message: 'Resetting task bindings to factory defaults...',
    });
    try {
      const res = await fetch('/api/settings/model-bindings/reset', {
        method: 'POST',
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`);
      }

      const newBindings = data.task_bindings || {};
      setCurrentBindings(newBindings);
      setStagedBindings(newBindings);
      setHasUnsavedChanges(false);
      updateTaskBindings(newBindings, true);

      setStatusNotification({
        type: 'success',
        message: 'Factory default task bindings successfully restored!',
      });

      if (apiEndpoint === '/api/settings/model-bindings') {
        executeApiRequest('/api/settings/model-bindings', 'GET');
      }
    } catch (err) {
      setStatusNotification({
        type: 'error',
        message: `Reset failed: ${err.message}`,
      });
    }
  };

  // Stage a change to a single task binding
  const handleStageTaskBinding = (taskId, providerId) => {
    setStagedBindings((prev) => {
      const updated = { ...prev, [taskId]: providerId };
      const hasChanges = Object.keys(updated).some(
        (key) => updated[key] !== currentBindings[key]
      );
      setHasUnsavedChanges(hasChanges);
      return updated;
    });
  };

  // Save Staged Pipeline Bindings
  const handleSavePipelineBindings = async () => {
    setStatusNotification({
      type: 'loading',
      message: 'Saving updated task bindings to model_config.yaml...',
    });
    try {
      const res = await fetch('/api/settings/model-bindings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_bindings: stagedBindings }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`);
      }

      const newBindings = data.task_bindings || stagedBindings;
      setCurrentBindings(newBindings);
      setStagedBindings(newBindings);
      setHasUnsavedChanges(false);
      updateTaskBindings(newBindings, true);

      setStatusNotification({
        type: 'success',
        message: 'Pipeline bindings successfully saved and persisted to disk!',
      });

      if (apiEndpoint === '/api/settings/model-bindings') {
        executeApiRequest('/api/settings/model-bindings', 'GET');
      }
    } catch (err) {
      setStatusNotification({
        type: 'error',
        message: `Failed to save bindings: ${err.message}`,
      });
    }
  };

  // Revert Staged Changes
  const handleRevertStaged = () => {
    setStagedBindings({ ...currentBindings });
    setHasUnsavedChanges(false);
  };

  // Execute API Request in Studio
  const executeApiRequest = async (targetUrl = apiEndpoint, targetMethod = apiMethod) => {
    setApiLoading(true);
    setApiData(null);
    setApiStatus(null);
    setApiLatency(null);
    setApiPayloadSize(null);

    const startTime = performance.now();
    try {
      const options = { method: targetMethod };
      if (targetMethod === 'POST') {
        options.headers = { 'Content-Type': 'application/json' };
      }

      const res = await fetch(targetUrl, options);
      const endTime = performance.now();
      const latencyMs = Math.round(endTime - startTime);
      setApiLatency(latencyMs);
      setApiStatus({ code: res.status, text: res.statusText || (res.ok ? 'OK' : 'Error') });

      const text = await res.text();
      setApiPayloadSize((text.length / 1024).toFixed(2));

      let json = null;
      try {
        json = JSON.parse(text);
      } catch {
        json = { raw_response: text };
      }
      setApiData(json);
    } catch (err) {
      const endTime = performance.now();
      setApiLatency(Math.round(endTime - startTime));
      setApiStatus({ code: 0, text: 'Network Failure' });
      setApiData({ error: err.message });
    } finally {
      setApiLoading(false);
    }
  };

  // Copy API JSON Response
  const handleCopyApiJson = () => {
    if (!apiData) return;
    navigator.clipboard.writeText(JSON.stringify(apiData, null, 2));
    setApiCopied(true);
    setTimeout(() => setApiCopied(false), 2000);
  };

  // Download API JSON Response
  const handleDownloadApiJson = () => {
    if (!apiData) return;
    const blob = new Blob([JSON.stringify(apiData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `newslens_${apiEndpoint.replace(/[^a-zA-Z0-9]/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Provider Reachability Map for O(1) lookup
  const reachabilityMap = useMemo(() => {
    const map = {};
    reachabilityList.forEach((p) => {
      map[p.id] = p;
    });
    return map;
  }, [reachabilityList]);

  // Filtered Provider Registry List
  const filteredProviders = useMemo(() => {
    return configuredProviders.filter((p) => {
      const reach = reachabilityMap[p.id] || {};
      const matchesSearch =
        p.id.toLowerCase().includes(providerSearch.toLowerCase()) ||
        (p.provider && p.provider.toLowerCase().includes(providerSearch.toLowerCase())) ||
        (p.model && p.model.toLowerCase().includes(providerSearch.toLowerCase())) ||
        (reach.name && reach.name.toLowerCase().includes(providerSearch.toLowerCase()));

      if (!matchesSearch) return false;

      if (providerFilter === 'reachable') {
        return reach.is_reachable === true;
      }
      if (providerFilter === 'cloud') {
        return (
          p.provider === 'openrouter' ||
          p.provider === 'gemini' ||
          p.provider === 'groq' ||
          p.provider === 'openai' ||
          p.provider === 'google_cloud_vision' ||
          p.provider === 'nvidia'
        );
      }
      if (providerFilter === 'local') {
        return p.provider === 'ollama' || p.provider === 'local_sentence_transformers' || p.provider === 'docling';
      }
      if (providerFilter === 'vision') {
        return p.supports_vision === true;
      }
      if (providerFilter === 'tools') {
        return p.supports_tool_use === true;
      }
      return true;
    });
  }, [configuredProviders, reachabilityMap, providerSearch, providerFilter]);

  // Provider metrics
  const providerMetrics = useMemo(() => {
    const total = configuredProviders.length;
    const reachableCount = reachabilityList.filter((p) => p.is_reachable === true).length;
    const boundTasksCount = Object.keys(currentBindings).length;
    const cloudCount = configuredProviders.filter(
      (p) =>
        p.provider === 'openrouter' ||
        p.provider === 'gemini' ||
        p.provider === 'groq' ||
        p.provider === 'openai' ||
        p.provider === 'google_cloud_vision' ||
        p.provider === 'nvidia'
    ).length;
    const localCount = total - cloudCount;

    return { total, reachableCount, boundTasksCount, cloudCount, localCount };
  }, [configuredProviders, reachabilityList, currentBindings]);

  // Get tasks bound to a provider ID
  const getTasksBoundToProvider = useCallback(
    (providerId) => {
      const tasks = [];
      Object.entries(currentBindings).forEach(([task, pId]) => {
        if (pId === providerId) tasks.push(task);
      });
      return tasks;
    },
    [currentBindings]
  );

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 w-full max-w-7xl 2xl:max-w-[1600px] mx-auto px-4 md:px-8 py-6 overflow-y-auto space-y-6">
      {/* ------------------------------------------------------------------- */}
      {/* Top Header & Studio Navigation Bar                                  */}
      {/* ------------------------------------------------------------------- */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 flex items-center justify-center text-white shadow-lg shadow-emerald-950/60">
              <Sliders className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl md:text-2xl font-bold font-serif text-slate-100 flex items-center gap-2.5">
                Model Architecture & API Studio
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">
                Dynamic runtime model orchestration, multi-stage task bindings, provider health observability, and interactive API explorer.
              </p>
            </div>
          </div>
        </div>

        {/* Global Controls & Status Pill */}
        <div className="flex items-center gap-2.5 flex-wrap">
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-mono shadow-sm transition-all ${
              activePreset
                ? `${activePreset.theme.badgeBg} ${activePreset.theme.border}`
                : 'bg-slate-900 border-slate-700 text-slate-300'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full animate-pulse ${
                activePreset?.id === 'cloud_full'
                  ? 'bg-emerald-400'
                  : activePreset?.id === 'cloud_hybrid'
                  ? 'bg-sky-400'
                  : 'bg-amber-400'
              }`}
            />
            <span className="font-semibold">{activePreset ? activePreset.name : 'Custom Pipeline'}</span>
          </div>

          <button
            type="button"
            onClick={() => loadSystemSettings(false)}
            disabled={refreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-emerald-400 text-xs font-medium border border-slate-800 hover:border-slate-700 transition-colors disabled:opacity-50"
            title="Ping all 20 configured model providers and refresh status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin text-emerald-400' : ''}`} />
            <span>{refreshing ? 'Pinging...' : 'Refresh Status'}</span>
          </button>

          <button
            type="button"
            onClick={handleResetDefaults}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-amber-400 text-xs font-medium border border-slate-800 hover:border-slate-700 transition-colors"
            title="Reset all task bindings to factory defaults"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Defaults</span>
          </button>
        </div>
      </header>

      {/* ------------------------------------------------------------------- */}
      {/* Toast Alert Notification                                            */}
      {/* ------------------------------------------------------------------- */}
      {statusNotification && (
        <div
          className={`px-4 py-3 rounded-xl text-xs flex items-center justify-between gap-3 border shadow-lg transition-all animate-fadeIn ${
            statusNotification.type === 'success'
              ? 'bg-emerald-950/80 text-emerald-300 border-emerald-500/40 shadow-emerald-950/40'
              : statusNotification.type === 'loading'
              ? 'bg-sky-950/80 text-sky-300 border-sky-500/40 shadow-sky-950/40'
              : 'bg-rose-950/80 text-rose-300 border-rose-500/40 shadow-rose-950/40'
          }`}
        >
          <div className="flex items-center gap-2">
            {statusNotification.type === 'loading' && <RefreshCw className="w-4 h-4 animate-spin text-sky-400" />}
            {statusNotification.type === 'success' && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
            {statusNotification.type === 'error' && <AlertCircle className="w-4 h-4 text-rose-400" />}
            <span className="font-medium">{statusNotification.message}</span>
          </div>
          <button
            onClick={() => setStatusNotification(null)}
            className="text-slate-400 hover:text-slate-200 text-xs px-2 py-0.5 rounded hover:bg-slate-800"
          >
            ✕
          </button>
        </div>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* Clean Segmented Sub-View Switcher (Tabs)                             */}
      {/* ------------------------------------------------------------------- */}
      <div className="flex items-center gap-2 bg-slate-900/90 p-1.5 rounded-2xl border border-slate-800 shadow-inner">
        {[
          {
            id: 'presets',
            label: 'Architecture Presets',
            icon: Sparkles,
            badge: '1-Click',
          },
          {
            id: 'pipeline',
            label: 'Pipeline Task Matrix',
            icon: Cpu,
            badge: hasUnsavedChanges ? 'Unsaved' : '10 Tasks',
            badgeHighlight: hasUnsavedChanges,
          },
          {
            id: 'providers',
            label: 'Provider Registry & Health',
            icon: Server,
            badge: `${providerMetrics.reachableCount}/${providerMetrics.total} Online`,
          },
          {
            id: 'api',
            label: 'API Studio & Inspector',
            icon: Terminal,
            badge: 'Interactive',
          },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeSubView === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubView(tab.id)}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl text-xs font-semibold transition-all duration-150 ${
                isActive
                  ? 'bg-emerald-600 text-white shadow-md shadow-emerald-950/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span className="hidden sm:inline">{tab.label}</span>
              <span className="sm:hidden">{tab.label.split(' ')[0]}</span>
              {tab.badge && (
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded-full font-mono font-medium ${
                    tab.badgeHighlight
                      ? 'bg-amber-500 text-slate-950 animate-pulse font-bold'
                      : isActive
                      ? 'bg-emerald-800/80 text-emerald-200'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* ------------------------------------------------------------------- */}
      {/* SUB-VIEW 1: ARCHITECTURE PRESETS                                    */}
      {/* ------------------------------------------------------------------- */}
      {activeSubView === 'presets' && (
        <section className="space-y-4 animate-fadeIn">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-emerald-400" />
                Operational Architecture Profiles
              </h2>
              <p className="text-xs text-slate-400">
                Instantly switch the operational profile of the entire NewsLens-AI intelligence pipeline with one click.
              </p>
            </div>
            <span className="text-[11px] text-slate-500 font-mono">
              Active: {activePreset ? activePreset.name : 'Custom Matrix'}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {PRESET_PROFILES.map((preset) => {
              const IconComp = preset.icon;
              const active = isPresetActive(preset);
              return (
                <div
                  key={preset.id}
                  className={`rounded-2xl border p-5 transition-all duration-200 flex flex-col justify-between ${
                    active ? preset.theme.activeBorder : preset.theme.border
                  } ${preset.theme.bg} shadow-lg`}
                >
                  <div className="space-y-3">
                    {/* Preset Header */}
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <div
                          className={`w-9 h-9 rounded-xl flex items-center justify-center ${preset.theme.badgeBg} border`}
                        >
                          <IconComp className={`w-5 h-5 ${preset.theme.text}`} />
                        </div>
                        <div>
                          <h3 className="font-bold text-sm text-slate-100">{preset.name}</h3>
                          <span
                            className={`inline-block text-[10px] font-mono px-2 py-0.5 rounded-full border mt-0.5 ${preset.theme.badgeBg}`}
                          >
                            {preset.badge}
                          </span>
                        </div>
                      </div>

                      {active && (
                        <span className="text-[11px] font-mono px-2 py-1 rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold flex items-center gap-1">
                          <Check className="w-3 h-3" /> ACTIVE
                        </span>
                      )}
                    </div>

                    <p className="text-xs text-slate-300 font-medium leading-snug">{preset.tagline}</p>
                    <p className="text-[11px] text-slate-400 leading-relaxed">{preset.description}</p>

                    {/* Pipeline Breakdown Highlights */}
                    <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3 space-y-2 text-[11px]">
                      <div className="flex items-center justify-between text-slate-400">
                        <span className="font-semibold text-slate-300 flex items-center gap-1">
                          <Sparkles className="w-3 h-3 text-purple-400" /> Reasoning:
                        </span>
                        <span className="font-mono text-slate-200 truncate ml-2">
                          {preset.pipelineHighlights.reasoning}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-slate-400">
                        <span className="font-semibold text-slate-300 flex items-center gap-1">
                          <Eye className="w-3 h-3 text-cyan-400" /> Vision & Layout:
                        </span>
                        <span className="font-mono text-slate-200 truncate ml-2">
                          {preset.pipelineHighlights.vision}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-slate-400">
                        <span className="font-semibold text-slate-300 flex items-center gap-1">
                          <Database className="w-3 h-3 text-emerald-400" /> Vector Index:
                        </span>
                        <span className="font-mono text-slate-200 truncate ml-2">
                          {preset.pipelineHighlights.indexing}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="pt-4 mt-2">
                    <button
                      type="button"
                      onClick={() => handleApplyPreset(preset)}
                      disabled={active}
                      className={`w-full py-2.5 px-4 rounded-xl text-xs font-semibold transition-all shadow-md flex items-center justify-center gap-2 ${
                        active
                          ? 'bg-emerald-600/60 text-emerald-200 border border-emerald-500/40 cursor-default'
                          : `${preset.theme.btn}`
                      }`}
                    >
                      {active ? (
                        <>
                          <CheckCircle2 className="w-4 h-4" />
                          <span>Currently Deployed Profile</span>
                        </>
                      ) : (
                        <>
                          <span>Activate Architecture</span>
                          <ArrowRight className="w-4 h-4" />
                        </>
                      )}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Architecture Details Banner */}
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs text-slate-400">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-400 flex-shrink-0" />
              <span>
                Presets update both in-memory provider instances and persist to{' '}
                <code className="text-emerald-300 bg-slate-950 px-1.5 py-0.5 rounded font-mono">model_config.yaml</code>{' '}
                immediately, synchronizing Broadsheet Reader, Agent Assistant, and Ingest workers with zero downtime.
              </span>
            </div>
            <button
              onClick={() => setActiveSubView('pipeline')}
              className="text-emerald-400 hover:text-emerald-300 font-semibold flex items-center gap-1 flex-shrink-0"
            >
              <span>Customize Individual Tasks</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* SUB-VIEW 2: PIPELINE TASK MATRIX                                    */}
      {/* ------------------------------------------------------------------- */}
      {activeSubView === 'pipeline' && (
        <section className="space-y-5 animate-fadeIn">
          {/* Action Bar for Pipeline Changes */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-md">
            <div>
              <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                <Cpu className="w-4 h-4 text-emerald-400" />
                Pipeline Task Orchestrator
              </h2>
              <p className="text-xs text-slate-400">
                Granularly re-bind individual AI models, VLMs, OCR engines, and embedders across the 3 pipeline stages.
              </p>
            </div>

            <div className="flex items-center gap-2 flex-wrap">
              {hasUnsavedChanges && (
                <>
                  <button
                    type="button"
                    onClick={handleRevertStaged}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition-colors"
                  >
                    Discard Changes
                  </button>
                  <button
                    type="button"
                    onClick={handleSavePipelineBindings}
                    className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold shadow-lg shadow-emerald-950/50 transition-colors animate-pulse"
                  >
                    <CheckSquare className="w-4 h-4" />
                    <span>Apply & Persist Changes</span>
                  </button>
                </>
              )}
              {!hasUnsavedChanges && (
                <span className="text-xs text-slate-500 font-mono flex items-center gap-1">
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  All 10 tasks synchronized
                </span>
              )}
            </div>
          </div>

          {/* Grouped Stages */}
          <div className="space-y-4">
            {PIPELINE_STAGES.map((stage) => {
              const StageIcon = stage.icon;
              return (
                <div
                  key={stage.stageId}
                  className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4"
                >
                  {/* Stage Header */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
                    <div className="flex items-center gap-2.5">
                      <div className="w-7 h-7 rounded-lg bg-slate-800 flex items-center justify-center">
                        <StageIcon className="w-4 h-4 text-emerald-400" />
                      </div>
                      <div>
                        <h3 className="font-bold text-sm text-slate-200">{stage.title}</h3>
                        <p className="text-[11px] text-slate-400">{stage.description}</p>
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
                      const providerMeta = configuredProviders.find((p) => p.id === selectedProviderId);
                      const reachMeta = reachabilityMap[selectedProviderId];
                      const isModified = stagedBindings[task.id] !== currentBindings[task.id];

                      // Capability Mismatch Warning
                      let capabilityWarning = null;
                      if (task.requiredCapability && providerMeta) {
                        if (task.requiredCapability === 'supports_vision' && !providerMeta.supports_vision) {
                          capabilityWarning = '⚠️ Provider lacks Vision capability required for this task';
                        } else if (
                          task.requiredCapability === 'supports_tool_use' &&
                          !providerMeta.supports_tool_use
                        ) {
                          capabilityWarning = '⚠️ Provider lacks Tool Calling required for agentic planning';
                        } else if (
                          task.requiredCapability === 'embedding_dim' &&
                          !providerMeta.embedding_dim
                        ) {
                          capabilityWarning = '⚠️ Provider lacks vector embedding output dimensions';
                        }
                      }

                      return (
                        <div
                          key={task.id}
                          className={`p-3.5 rounded-xl border transition-all flex flex-col md:flex-row md:items-center justify-between gap-3 ${
                            isModified
                              ? 'bg-amber-950/20 border-amber-500/40 ring-1 ring-amber-500/30'
                              : 'bg-slate-950/70 border-slate-800/70 hover:border-slate-700'
                          }`}
                        >
                          {/* Task Description */}
                          <div className="flex-1 space-y-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-semibold text-xs text-slate-100">{task.name}</span>
                              <code className="text-[10px] font-mono text-emerald-400 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                                {task.id}
                              </code>
                              <span className="text-[10px] text-slate-400 font-mono px-1.5 py-0.5 rounded bg-slate-800">
                                {task.type}
                              </span>
                              {task.requiredLabel && (
                                <span className="text-[10px] text-sky-400 font-mono px-1.5 py-0.5 rounded bg-sky-950/60 border border-sky-800/60">
                                  {task.requiredLabel}
                                </span>
                              )}
                              {isModified && (
                                <span className="text-[10px] font-bold text-amber-400 bg-amber-950/60 border border-amber-800 px-1.5 py-0.5 rounded animate-pulse">
                                  Pending Save
                                </span>
                              )}
                            </div>
                            <p className="text-[11px] text-slate-400">{task.role}</p>
                            {capabilityWarning && (
                              <p className="text-[10px] font-semibold text-amber-400">{capabilityWarning}</p>
                            )}
                          </div>

                          {/* Target Provider Selector Dropdown */}
                          <div className="flex items-center gap-2 min-w-[280px] md:min-w-[340px]">
                            {/* Reachability Dot */}
                            <div
                              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                              title={
                                reachMeta?.is_reachable === true
                                  ? 'Provider Online & Reachable'
                                  : reachMeta?.is_reachable === false
                                  ? 'Provider Offline / Unreachable'
                                  : 'Status Unknown'
                              }
                              style={{
                                backgroundColor:
                                  reachMeta?.is_reachable === true
                                    ? '#10b981'
                                    : reachMeta?.is_reachable === false
                                    ? '#f43f5e'
                                    : '#94a3b8',
                              }}
                            />

                            <select
                              value={selectedProviderId}
                              onChange={(e) => handleStageTaskBinding(task.id, e.target.value)}
                              className="flex-1 bg-slate-900 border border-slate-700 hover:border-slate-600 rounded-lg px-3 py-2 text-xs text-slate-200 outline-none focus:border-emerald-500 cursor-pointer font-mono"
                            >
                              <optgroup label="☁️ Cloud Dual-Key (OpenRouter)">
                                {configuredProviders
                                  .filter((p) => p.provider === 'openrouter')
                                  .map((p) => (
                                    <option key={p.id} value={p.id}>
                                      {reachabilityMap[p.id]?.name || p.id} ({p.model})
                                    </option>
                                  ))}
                              </optgroup>
                              <optgroup label="☁️ Cloud Direct (Google, Groq, OpenAI, NVIDIA)">
                                {configuredProviders
                                  .filter(
                                    (p) =>
                                      p.provider !== 'openrouter' &&
                                      p.provider !== 'ollama' &&
                                      p.provider !== 'local_sentence_transformers' &&
                                      p.provider !== 'docling'
                                  )
                                  .map((p) => (
                                    <option key={p.id} value={p.id}>
                                      {reachabilityMap[p.id]?.name || p.id} ({p.provider})
                                    </option>
                                  ))}
                              </optgroup>
                              <optgroup label="🖥️ Local Hardware (Ollama, BGE-M3, Docling)">
                                {configuredProviders
                                  .filter(
                                    (p) =>
                                      p.provider === 'ollama' ||
                                      p.provider === 'local_sentence_transformers' ||
                                      p.provider === 'docling'
                                  )
                                  .map((p) => (
                                    <option key={p.id} value={p.id}>
                                      {reachabilityMap[p.id]?.name || p.id} ({p.model || p.provider})
                                    </option>
                                  ))}
                              </optgroup>
                            </select>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* SUB-VIEW 3: PROVIDER REGISTRY & HEALTH OBSERVABILITY                */}
      {/* ------------------------------------------------------------------- */}
      {activeSubView === 'providers' && (
        <section className="space-y-4 animate-fadeIn">
          {/* Top Metrics Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block">
                Total Configured
              </span>
              <div className="text-2xl font-bold text-slate-100 mt-1 font-mono">{providerMetrics.total}</div>
              <span className="text-[10px] text-slate-500">In model_config.yaml</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block">
                Live Connectivity
              </span>
              <div className="text-2xl font-bold text-emerald-400 mt-1 font-mono">
                {providerMetrics.reachableCount} / {providerMetrics.total}
              </div>
              <span className="text-[10px] text-emerald-500/80">Online & Reachable</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block">
                Active Bound Tasks
              </span>
              <div className="text-2xl font-bold text-sky-400 mt-1 font-mono">
                {providerMetrics.boundTasksCount} / 10
              </div>
              <span className="text-[10px] text-sky-500/80">Orchestrating Pipeline</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block">
                Compute Distribution
              </span>
              <div className="text-2xl font-bold text-amber-400 mt-1 font-mono">
                {providerMetrics.cloudCount}C / {providerMetrics.localCount}L
              </div>
              <span className="text-[10px] text-amber-500/80">Cloud vs Local split</span>
            </div>
          </div>

          {/* Search & Filter Toolbar */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3 shadow-md">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="relative flex-1">
                <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={providerSearch}
                  onChange={(e) => setProviderSearch(e.target.value)}
                  placeholder="Search providers by model name, ID, or framework (e.g. gemma, nemotron, vision)..."
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none focus:border-emerald-500"
                />
              </div>

              <div className="flex items-center gap-1.5 flex-wrap">
                {[
                  { id: 'all', label: 'All' },
                  { id: 'reachable', label: 'Reachable Only' },
                  { id: 'cloud', label: 'Cloud' },
                  { id: 'local', label: 'Local' },
                  { id: 'vision', label: 'Vision Capable' },
                  { id: 'tools', label: 'Tool Use' },
                ].map((f) => (
                  <button
                    key={f.id}
                    onClick={() => setProviderFilter(f.id)}
                    className={`text-xs px-2.5 py-1.5 rounded-lg border font-medium transition-colors ${
                      providerFilter === f.id
                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 font-semibold'
                        : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Provider Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {filteredProviders.map((p) => {
              const reach = reachabilityMap[p.id];
              const isReachable = reach?.is_reachable;
              const boundTasks = getTasksBoundToProvider(p.id);

              return (
                <div
                  key={p.id}
                  className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-md flex flex-col justify-between hover:border-slate-700 transition-colors"
                >
                  <div className="space-y-2.5">
                    {/* Title & Ping Dot */}
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="flex items-center gap-2">
                          <span
                            className="w-2.5 h-2.5 rounded-full"
                            style={{
                              backgroundColor:
                                isReachable === true ? '#10b981' : isReachable === false ? '#f43f5e' : '#94a3b8',
                            }}
                          />
                          <h4 className="font-bold text-sm text-slate-100">{reach?.name || p.id}</h4>
                        </div>
                        <span className="text-[10px] font-mono text-slate-500 block mt-0.5">
                          ID: {p.id} • Provider: {p.provider}
                        </span>
                      </div>

                      <span
                        className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                          isReachable === true
                            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                            : isReachable === false
                            ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                            : 'bg-slate-800 text-slate-400 border-slate-700'
                        }`}
                      >
                        {isReachable === true ? 'Reachable' : isReachable === false ? 'Unreachable' : 'Checking'}
                      </span>
                    </div>

                    {/* Model & Base URL */}
                    <div className="bg-slate-950/80 border border-slate-800/80 rounded-xl p-2.5 text-[11px] font-mono space-y-1">
                      <div className="flex items-center justify-between text-slate-400">
                        <span className="text-slate-500">Model:</span>
                        <span className="text-slate-200 truncate ml-2">{p.model || 'N/A'}</span>
                      </div>
                      {p.base_url && (
                        <div className="flex items-center justify-between text-slate-400">
                          <span className="text-slate-500">Endpoint:</span>
                          <span className="text-slate-300 truncate ml-2">{p.base_url}</span>
                        </div>
                      )}
                    </div>

                    {/* Capabilities Badges */}
                    <div className="flex items-center gap-1.5 flex-wrap">
                      {p.context_window && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                          {p.context_window >= 1000000
                            ? `${(p.context_window / 1000000).toFixed(0)}M Context`
                            : `${(p.context_window / 1000).toFixed(0)}k Context`}
                        </span>
                      )}
                      {p.supports_vision && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-300 border border-cyan-800 flex items-center gap-1">
                          <Eye className="w-3 h-3" /> Vision
                        </span>
                      )}
                      {p.supports_tool_use && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950/60 text-purple-300 border border-purple-800 flex items-center gap-1">
                          <Wrench className="w-3 h-3" /> Tools
                        </span>
                      )}
                      {p.embedding_dim && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800">
                          {p.embedding_dim}d Dense
                        </span>
                      )}
                    </div>

                    {/* Bound Tasks Indicator */}
                    {boundTasks.length > 0 && (
                      <div className="pt-2 border-t border-slate-800/80">
                        <span className="text-[10px] text-slate-400 font-semibold block mb-1">
                          Bound to {boundTasks.length} pipeline task{boundTasks.length > 1 ? 's' : ''}:
                        </span>
                        <div className="flex items-center gap-1 flex-wrap">
                          {boundTasks.map((t) => (
                            <span
                              key={t}
                              className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800"
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* SUB-VIEW 4: INTERACTIVE API STUDIO & EXPLORER                       */}
      {/* ------------------------------------------------------------------- */}
      {activeSubView === 'api' && (
        <section className="space-y-4 animate-fadeIn">
          {/* Quick Endpoint Preset Selector */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                Curated API Endpoints Catalog
              </span>
              <span className="text-[11px] text-slate-500">Live FastAPI / Proxy Console</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
              {API_ENDPOINTS.map((endpoint) => {
                const isActive = apiEndpoint === endpoint.path;
                return (
                  <button
                    key={endpoint.id}
                    onClick={() => {
                      setApiEndpoint(endpoint.path);
                      setApiMethod(endpoint.method);
                      executeApiRequest(endpoint.path, endpoint.method);
                    }}
                    className={`p-2.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                      isActive
                        ? 'bg-emerald-950/40 border-emerald-500/50 ring-1 ring-emerald-500/30'
                        : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span
                          className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded ${
                            endpoint.method === 'GET'
                              ? 'bg-emerald-500/20 text-emerald-300'
                              : 'bg-blue-500/20 text-blue-300'
                          }`}
                        >
                          {endpoint.method}
                        </span>
                        <span className="text-[9px] font-mono text-slate-500">{endpoint.category}</span>
                      </div>
                      <div className="font-semibold text-xs text-slate-200 truncate">{endpoint.name}</div>
                    </div>
                    <div className="text-[10px] font-mono text-slate-400 truncate mt-1">{endpoint.path}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Interactive Request Bar */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-3">
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
              <select
                value={apiMethod}
                onChange={(e) => setApiMethod(e.target.value)}
                className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs font-mono font-bold text-emerald-400 outline-none focus:border-emerald-500 cursor-pointer"
              >
                <option value="GET">GET</option>
                <option value="POST">POST</option>
              </select>

              <div className="flex-1 relative">
                <input
                  type="text"
                  value={apiEndpoint}
                  onChange={(e) => setApiEndpoint(e.target.value)}
                  placeholder="/api/..."
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs font-mono text-slate-100 outline-none focus:border-emerald-500"
                />
              </div>

              <button
                type="button"
                onClick={() => executeApiRequest(apiEndpoint, apiMethod)}
                disabled={apiLoading}
                className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold px-5 py-2 rounded-xl flex items-center justify-center gap-2 transition-colors shadow-md disabled:opacity-50"
              >
                {apiLoading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Executing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" />
                    <span>Send Request</span>
                  </>
                )}
              </button>
            </div>

            {/* Response Metrics Strip */}
            <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-800/80 text-xs">
              <div className="flex items-center gap-3">
                {apiStatus && (
                  <span
                    className={`font-mono text-xs px-2 py-0.5 rounded-md font-bold ${
                      apiStatus.code >= 200 && apiStatus.code < 300
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                        : 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                    }`}
                  >
                    {apiStatus.code} {apiStatus.text}
                  </span>
                )}

                {apiLatency !== null && (
                  <span className="font-mono text-slate-400 text-xs flex items-center gap-1">
                    <Zap className="w-3 h-3 text-sky-400" />
                    {apiLatency} ms
                  </span>
                )}

                {apiPayloadSize !== null && (
                  <span className="font-mono text-slate-400 text-xs flex items-center gap-1">
                    <Database className="w-3 h-3 text-purple-400" />
                    {apiPayloadSize} KB
                  </span>
                )}
              </div>

              {/* Data Search & Actions */}
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={apiSearchQuery}
                  onChange={(e) => setApiSearchQuery(e.target.value)}
                  placeholder="Filter keys/values in JSON..."
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-slate-200 placeholder-slate-500 outline-none focus:border-emerald-500 font-mono w-48"
                />

                <button
                  type="button"
                  onClick={handleCopyApiJson}
                  disabled={!apiData}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-slate-300 hover:text-white text-xs transition-colors disabled:opacity-30"
                  title="Copy full JSON payload to clipboard"
                >
                  {apiCopied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  <span>{apiCopied ? 'Copied' : 'Copy'}</span>
                </button>

                <button
                  type="button"
                  onClick={handleDownloadApiJson}
                  disabled={!apiData}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-slate-300 hover:text-white text-xs transition-colors disabled:opacity-30"
                  title="Download JSON payload"
                >
                  <Download className="w-3 h-3" />
                  <span>Download</span>
                </button>
              </div>
            </div>
          </div>

          {/* Formatted JSON Tree / Console */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
            <div className="bg-slate-950 border border-slate-800/90 rounded-xl p-4 overflow-x-auto max-h-[480px]">
              {apiLoading ? (
                <div className="flex flex-col items-center justify-center py-16 text-slate-500 gap-3">
                  <RefreshCw className="w-6 h-6 animate-spin text-emerald-400" />
                  <span className="text-xs font-mono">Executing HTTP request to {apiEndpoint}...</span>
                </div>
              ) : apiData ? (
                <pre className="text-xs font-mono text-emerald-400 leading-relaxed">
                  {JSON.stringify(
                    apiSearchQuery
                      ? filterJsonByQuery(apiData, apiSearchQuery)
                      : apiData,
                    null,
                    2
                  )}
                </pre>
              ) : (
                <div className="text-center py-16 text-slate-500 text-xs">
                  Select an endpoint or enter a custom path and click Send Request.
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helper: Filter nested JSON object by query key/value string
// ---------------------------------------------------------------------------
function filterJsonByQuery(obj, query) {
  if (!query) return obj;
  const q = query.toLowerCase();

  if (Array.isArray(obj)) {
    return obj
      .map((item) => filterJsonByQuery(item, query))
      .filter((item) => {
        if (item === null || item === undefined) return false;
        if (typeof item === 'object') return Object.keys(item).length > 0;
        return String(item).toLowerCase().includes(q);
      });
  }

  if (typeof obj === 'object' && obj !== null) {
    const result = {};
    for (const [k, v] of Object.entries(obj)) {
      if (k.toLowerCase().includes(q)) {
        result[k] = v;
      } else if (typeof v === 'object' && v !== null) {
        const filtered = filterJsonByQuery(v, query);
        if (filtered && (Array.isArray(filtered) ? filtered.length > 0 : Object.keys(filtered).length > 0)) {
          result[k] = filtered;
        }
      } else if (String(v).toLowerCase().includes(q)) {
        result[k] = v;
      }
    }
    return result;
  }

  return String(obj).toLowerCase().includes(q) ? obj : null;
}
