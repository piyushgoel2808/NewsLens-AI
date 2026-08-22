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
} from 'lucide-react';

const TASK_OPTIONS = [
  { value: 'query_planner', label: 'Query Planner (LLM)' },
  { value: 'answerer', label: 'Answer Synthesizer (LLM)' },
  { value: 'layout_analysis', label: 'Layout Analysis (VLM)' },
  { value: 'article_segmentation', label: 'Article Segmentation (LLM)' },
  { value: 'metadata_extraction', label: 'Metadata & NER Extraction (LLM)' },
  { value: 'classification', label: 'Article Classification (LLM)' },
  { value: 'embedding', label: 'Vector Embedding (Embedder)' },
  { value: 'ocr', label: 'OCR Engine' },
];

export default function RawDataViewer() {
  const [activeEndpoint, setActiveEndpoint] = useState('/api/settings/model-bindings');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  // Dynamic Settings Swapper State
  const [selectedTask, setSelectedTask] = useState('query_planner');
  const [selectedProvider, setSelectedProvider] = useState('');
  const [configuredProviders, setConfiguredProviders] = useState([]);
  const [currentBindings, setCurrentBindings] = useState({});
  const [swapMessage, setSwapMessage] = useState(null);

  async function loadSettings() {
    try {
      const res = await fetch('/api/settings/model-bindings');
      if (res.ok) {
        const json = await res.json();
        setConfiguredProviders(json.configured_providers || []);
        setCurrentBindings(json.task_bindings || {});
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
        setCurrentBindings(json.task_bindings || {});
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
      setCurrentBindings(result.task_bindings || {});
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

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 max-w-5xl mx-auto p-4 md:p-6 overflow-y-auto space-y-6">
      {/* Header */}
      <div className="pb-4 border-b border-slate-800">
        <h1 className="text-2xl font-bold font-serif text-slate-100 flex items-center gap-2.5">
          <Sliders className="w-6 h-6 text-emerald-400" />
          Model Registry & Ingestion Manifest Inspector
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Dynamically re-bind LLM, VLM, Embedding, and OCR providers at runtime without service restarts, and inspect live endpoint responses.
        </p>
      </div>

      {/* Model Provider Re-Binding Panel */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-emerald-400" />
          Dynamic Provider Binding Manager
        </h2>

        <form onSubmit={handleSwapBinding} className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div>
            <label className="block text-slate-400 font-medium mb-1">Pipeline Task</label>
            <select
              value={selectedTask}
              onChange={(e) => setSelectedTask(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 outline-none focus:border-emerald-500"
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
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 outline-none focus:border-emerald-500"
            >
              {configuredProviders.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.id} ({p.provider} - {p.model || 'default'})
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
              <span>Apply Binding</span>
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
