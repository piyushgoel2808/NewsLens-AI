import React, { useState, useEffect } from 'react';
import {
  Database,
  Cpu,
  Sliders,
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
} from 'lucide-react';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

const PRESET_PROFILES = [
  {
    id: 'cloud_full',
    name: 'Full Cloud Mode (Recommended)',
    badge: '0% Machine Stress',
    icon: Cloud,
    theme: {
      border: 'border-emerald-500/40 hover:border-emerald-400',
      bg: 'bg-emerald-950/20 hover:bg-emerald-950/40',
      text: 'text-emerald-400',
      btn: 'bg-emerald-600 hover:bg-emerald-500 text-white',
    },
    description: 'Zero local GPU/CPU load. Routes LLM reasoning, visual analysis & NER to OpenRouter Dual-Key Gemma 4 26B and Google Cloud Vision.',
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
    badge: 'Fast Reasoning',
    icon: Zap,
    theme: {
      border: 'border-sky-500/40 hover:border-sky-400',
      bg: 'bg-sky-950/20 hover:bg-sky-950/40',
      text: 'text-sky-400',
      btn: 'bg-sky-600 hover:bg-sky-500 text-white',
    },
    description: 'Blazing speed using NVIDIA Nemotron 3.5 Lightning for planning & synthesis, paired with Docling 2D for layout parsing and Gemma 4 for vision.',
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
    name: 'Local Offline Mode',
    badge: 'Local 2D + Compute',
    icon: Laptop,
    theme: {
      border: 'border-amber-500/40 hover:border-amber-400',
      bg: 'bg-amber-950/20 hover:bg-amber-950/40',
      text: 'text-amber-400',
      btn: 'bg-amber-600 hover:bg-amber-500 text-white',
    },
    description: 'Executes locally using Llama 3.1 8B for fast planning & synthesis, DeepSeek R1 for reasoning, Qwen 3 VL for vision, and IBM Docling for 2D layout.',
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

const TASK_OPTIONS = [
  { value: 'query_planner', label: 'Query Planner (LLM)' },
  { value: 'answerer', label: 'Answer Synthesizer (LLM)' },
  { value: 'layout_analysis', label: 'Layout Analysis (VLM)' },
  { value: 'article_segmentation', label: 'Article Segmentation (LLM)' },
  { value: 'metadata_extraction', label: 'Metadata & NER Extraction (LLM)' },
  { value: 'classification', label: 'Article Classification (LLM)' },
  { value: 'visual_extraction', label: 'Visual Extraction & Photo Analysis (VLM)' },
  { value: 'embedding', label: 'Vector Embedding (Embedder)' },
  { value: 'ocr', label: 'OCR Engine' },
];

export default function RawDataViewer() {
  const { updateTaskBindings, taskBindings, refreshTaskBindings } = useActiveHighlight();

  const [activeEndpoint, setActiveEndpoint] = useState('/api/settings/model-bindings');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  // Dynamic Settings Swapper State
  const [selectedTask, setSelectedTask] = useState('query_planner');
  const [selectedProvider, setSelectedProvider] = useState('');
  const [configuredProviders, setConfiguredProviders] = useState([]);
  const [currentBindings, setCurrentBindings] = useState(() => taskBindings || {});
  const [swapMessage, setSwapMessage] = useState(null);

  // Keep local state aligned if taskBindings in context changes
  useEffect(() => {
    if (taskBindings && Object.keys(taskBindings).length > 0) {
      setCurrentBindings(taskBindings);
    }
  }, [taskBindings]);

  async function loadSettings() {
    try {
      const res = await fetch('/api/settings/model-bindings');
      if (res.ok) {
        const json = await res.json();
        setConfiguredProviders(json.configured_providers || []);
        if (json.task_bindings) {
          setCurrentBindings(json.task_bindings);
          updateTaskBindings(json.task_bindings, false);
        }
        if (!selectedProvider && json.configured_providers?.length > 0) {
          setSelectedProvider(json.configured_providers[0].id);
        }
      }
    } catch (err) {
      console.error('Failed to load settings:', err);
    }
  }

  useEffect(() => {
    loadSettings();
    fetchEndpoint('/api/settings/model-bindings');
  }, []);

  async function fetchEndpoint(endpoint) {
    setActiveEndpoint(endpoint);
    setLoading(true);
    setData(null);

    try {
      const response = await fetch(endpoint);
      const json = await response.json();
      setData(json);
      if (endpoint === '/api/settings/model-bindings' && response.ok) {
        setConfiguredProviders(json.configured_providers || []);
        if (json.task_bindings) {
          setCurrentBindings(json.task_bindings);
          updateTaskBindings(json.task_bindings, false);
        }
      }
    } catch (err) {
      setData({ error: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleSwapBinding(e) {
    e.preventDefault();
    if (!selectedTask || !selectedProvider) return;

    setSwapMessage({ status: 'updating', text: `Assigning ${selectedTask} -> ${selectedProvider}...` });
    try {
      const payload = {
        task_bindings: {
          [selectedTask]: selectedProvider,
        },
      };

      const response = await fetch('/api/settings/model-bindings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || `Update failed (${response.status})`);
      }

      setSwapMessage({ status: 'success', text: `Successfully bound ${selectedTask} to ${selectedProvider}!` });
      const newBindings = result.task_bindings || {};
      setCurrentBindings(newBindings);
      // Synchronize immediately across other tabs (Agent Assistant, Timeline, Ingest)
      updateTaskBindings(newBindings, true);

      if (activeEndpoint === '/api/settings/model-bindings') {
        fetchEndpoint('/api/settings/model-bindings');
      }
    } catch (err) {
      setSwapMessage({ status: 'error', text: err.message });
    }
  }

  async function handleApplyPreset(preset) {
    setSwapMessage({ status: 'updating', text: `Activating ${preset.name}...` });
    try {
      const response = await fetch('/api/settings/model-bindings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_bindings: preset.bindings }),
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || `Preset update failed (${response.status})`);
      }

      setSwapMessage({ status: 'success', text: `Successfully activated ${preset.name} across all workspace tabs!` });
      const newBindings = result.task_bindings || preset.bindings;
      setCurrentBindings(newBindings);
      // Synchronize immediately across all other tabs
      updateTaskBindings(newBindings, true);

      if (activeEndpoint === '/api/settings/model-bindings') {
        fetchEndpoint('/api/settings/model-bindings');
      }
    } catch (err) {
      setSwapMessage({ status: 'error', text: err.message });
    }
  }

  async function handleResetToDefault() {
    if (!window.confirm('Reset all model task bindings back to system defaults?')) {
      return;
    }
    setSwapMessage({ status: 'updating', text: 'Resetting to default configuration...' });
    try {
      const response = await fetch('/api/settings/model-bindings/reset', {
        method: 'POST',
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || 'Reset failed');
      }
      setSwapMessage({ status: 'success', text: 'Task bindings successfully restored to defaults across all tabs!' });
      const newBindings = result.task_bindings || {};
      setCurrentBindings(newBindings);
      // Synchronize immediately across all other tabs
      updateTaskBindings(newBindings, true);

      if (activeEndpoint === '/api/settings/model-bindings') {
        fetchEndpoint('/api/settings/model-bindings');
      }
    } catch (err) {
      setSwapMessage({ status: 'error', text: err.message });
    }
  }

  const handleCopyJson = () => {
    if (!data) return;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isPresetActive = (preset) => {
    if (!currentBindings || Object.keys(currentBindings).length === 0) return false;
    return Object.entries(preset.bindings).every(
      ([task, providerId]) => currentBindings[task] === providerId
    );
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 max-w-5xl mx-auto p-4 md:p-6 overflow-y-auto space-y-6">
      {/* Header */}
      <div className="pb-4 border-b border-slate-800 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold font-serif text-slate-100 flex items-center gap-2.5">
            <Sliders className="w-6 h-6 text-emerald-400" />
            Model Registry & Ingestion Manifest Inspector
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Dynamically re-bind LLM, VLM, Embedding, and OCR providers at runtime without service restarts, and inspect live endpoint responses.
          </p>
        </div>
        <button
          type="button"
          onClick={handleResetToDefault}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-emerald-400 hover:text-emerald-300 transition-colors border border-slate-700 font-medium"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Reset to Default Bindings</span>
        </button>
      </div>

      {/* Model Provider Re-Binding Panel */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-emerald-400" />
            Dynamic Provider Binding Manager
          </h2>
          <button
            type="button"
            onClick={handleResetToDefault}
            className="text-xs text-slate-400 hover:text-emerald-400 transition-colors flex items-center gap-1"
          >
            <RefreshCw className="w-3 h-3" /> Restore Defaults
          </button>
        </div>

        {/* 1-Click System Presets */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
              1-Click System Presets (Instant Migration)
            </span>
            <span className="text-[11px] text-slate-500">
              One-click whole-pipeline assignment
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {PRESET_PROFILES.map((preset) => {
              const IconComp = preset.icon;
              const active = isPresetActive(preset);
              return (
                <div
                  key={preset.id}
                  className={`p-3.5 rounded-xl border transition-all duration-150 flex flex-col justify-between ${
                    active
                      ? `${preset.theme.border} ${preset.theme.bg} ring-2 ring-emerald-500/50 shadow-lg`
                      : `${preset.theme.border} ${preset.theme.bg}`
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-1.5">
                        <IconComp className={`w-4 h-4 ${preset.theme.text}`} />
                        <span className="font-semibold text-xs text-slate-100">{preset.name}</span>
                      </div>
                      <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border border-current ${preset.theme.text} bg-slate-900/60`}>
                        {active ? 'Active Profile ✓' : preset.badge}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed mb-3">
                      {preset.description}
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={() => handleApplyPreset(preset)}
                    className={`w-full py-1.5 px-3 rounded-lg text-xs font-semibold transition-colors shadow-sm flex items-center justify-center gap-1.5 ${
                      active ? 'bg-emerald-600 text-white shadow-emerald-900/40 cursor-default' : preset.theme.btn
                    }`}
                  >
                    <span>{active ? 'Active Profile ✓' : 'Activate Preset'}</span>
                    {!active && <span>→</span>}
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Divider for Granular Form */}
        <div className="relative flex py-1 items-center">
          <div className="flex-grow border-t border-slate-800"></div>
          <span className="flex-shrink mx-3 text-[11px] text-slate-500 font-medium uppercase tracking-wider">
            Or Granular Task Customization
          </span>
          <div className="flex-grow border-t border-slate-800"></div>
        </div>

        <form onSubmit={handleSwapBinding} className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div>
            <label className="block text-slate-400 font-medium mb-1">Pipeline Task</label>
            <select
              value={selectedTask}
              onChange={(e) => setSelectedTask(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 outline-none focus:border-emerald-500 cursor-pointer"
            >
              {TASK_OPTIONS.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-slate-400 font-medium mb-1">Target Provider Binding</label>
            <select
              value={selectedProvider}
              onChange={(e) => setSelectedProvider(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 outline-none focus:border-emerald-500 cursor-pointer"
            >
              {configuredProviders.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name || p.id} ({p.provider} - {p.model || 'default'})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-end">
            <button
              type="submit"
              className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors shadow-md"
            >
              <CheckCircle2 className="w-4 h-4" />
              <span>Apply & Persist</span>
            </button>
          </div>
        </form>

        {swapMessage && (
          <div
            className={`p-3 rounded-lg text-xs flex items-center gap-2 ${
              swapMessage.status === 'success'
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                : swapMessage.status === 'updating'
                ? 'bg-blue-500/10 text-blue-400 border border-blue-500/30'
                : 'bg-red-500/10 text-red-400 border border-red-500/30'
            }`}
          >
            {swapMessage.status === 'updating' && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
            {swapMessage.status === 'success' && <CheckCircle2 className="w-3.5 h-3.5" />}
            {swapMessage.status === 'error' && <AlertCircle className="w-3.5 h-3.5" />}
            <span>{swapMessage.text}</span>
          </div>
        )}

        {/* Live Active Bindings Grid */}
        <div className="pt-3 border-t border-slate-800">
          <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block mb-2">
            Active System Bindings:
          </span>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {Object.entries(currentBindings).map(([task, prov]) => {
              const isCloudOR = prov.startsWith('openrouter');
              const isCloudDirect =
                prov.startsWith('gemini') ||
                prov.startsWith('groq') ||
                prov.startsWith('openai') ||
                prov.startsWith('google');
              const isLocal = prov.startsWith('ollama') || prov.startsWith('local');

              return (
                <div
                  key={task}
                  className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-2.5 text-xs flex flex-col justify-between"
                >
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase font-mono">{task}</div>
                    <div className="text-emerald-300 font-semibold truncate mt-0.5" title={prov}>
                      {prov}
                    </div>
                  </div>
                  <div className="mt-2 flex items-center gap-1 text-[10px]">
                    {isCloudOR && (
                      <span className="text-emerald-400 font-medium flex items-center gap-1">
                        <Cloud className="w-3 h-3" /> Dual-Key Cloud
                      </span>
                    )}
                    {isCloudDirect && (
                      <span className="text-sky-400 font-medium flex items-center gap-1">
                        <Sparkles className="w-3 h-3" /> Cloud Hosted
                      </span>
                    )}
                    {isLocal && (
                      <span className="text-amber-400 font-medium flex items-center gap-1">
                        <Laptop className="w-3 h-3" /> Local Machine
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Endpoint Inspector & JSON Console */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Inspect API:</span>
            {[
              { label: 'Model Bindings', url: '/api/settings/model-bindings' },
              { label: 'Corpus Newspapers', url: '/api/newspapers' },
              { label: 'Issues Catalog', url: '/api/issues?limit=10' },
              { label: 'System Health', url: '/health' },
            ].map((btn) => (
              <button
                key={btn.url}
                onClick={() => fetchEndpoint(btn.url)}
                className={`text-xs px-3 py-1.5 rounded-lg border font-medium transition-colors ${
                  activeEndpoint === btn.url
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                    : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                }`}
              >
                {btn.label}
              </button>
            ))}
          </div>

          <button
            onClick={handleCopyJson}
            disabled={!data}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 bg-slate-950 border border-slate-800 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-30"
          >
            <Copy className="w-3.5 h-3.5" />
            <span>{copied ? 'Copied!' : 'Copy JSON'}</span>
          </button>
        </div>

        {/* JSON Code Viewer */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 overflow-x-auto max-h-[420px]">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-slate-500 gap-2">
              <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
              <span className="text-xs">Fetching live JSON payload...</span>
            </div>
          ) : data ? (
            <pre className="text-xs font-mono text-emerald-400 leading-relaxed">
              {JSON.stringify(data, null, 2)}
            </pre>
          ) : (
            <div className="text-center py-12 text-slate-500 text-xs">
              Select an endpoint above to view structured response payload.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
