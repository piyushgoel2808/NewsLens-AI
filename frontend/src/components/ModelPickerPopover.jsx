import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  Cpu,
  Cloud,
  Laptop,
  Sparkles,
  ChevronDown,
  Check,
  Search,
  ExternalLink,
  Zap,
  Eye,
  Sliders,
  Server,
} from 'lucide-react';
import { useModelConfig } from '../context/ModelConfigContext';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

export default function ModelPickerPopover({
  value,
  onChange,
  className = '',
  compact = false,
  showLabel = true,
}) {
  const {
    configuredProviders,
    providerReachability,
    laptopBridgeStatus,
    selectedModel,
    setSelectedModel,
  } = useModelConfig();
  const { setActiveTab } = useActiveHighlight();

  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const popoverRef = useRef(null);

  const currentModelId = value || selectedModel || 'gemini_flash';

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (popoverRef.current && !popoverRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  // Categorize configured providers into 3 prominent architectural groups
  const categorized = useMemo(() => {
    const gemini = [];
    const local = [];
    const hosted = [];

    configuredProviders.forEach((p) => {
      const pId = p.id || p.name || '';
      const provider = (p.provider || '').toLowerCase();
      const lowerId = pId.toLowerCase();

      if (lowerId.includes('gemini') || provider.includes('gemini') || provider.includes('google')) {
        gemini.push(p);
      } else if (lowerId.includes('ollama') || lowerId.includes('local') || lowerId.startsWith('laptop:')) {
        local.push(p);
      } else {
        hosted.push(p);
      }
    });

    return { gemini, local, hosted };
  }, [configuredProviders]);

  // Find active model details
  const activeDetails = useMemo(() => {
    const found = configuredProviders.find((p) => p.id === currentModelId || p.model === currentModelId);
    if (found) return found;

    if (currentModelId.startsWith('laptop:')) {
      return {
        id: currentModelId,
        provider: 'ollama',
        model: currentModelId.replace('laptop:', ''),
        context_window: 128000,
        supports_vision: false,
        supports_tool_use: true,
      };
    }

    return {
      id: currentModelId,
      provider: 'unknown',
      model: currentModelId,
      context_window: 131072,
    };
  }, [configuredProviders, currentModelId]);

  const handleSelect = (id) => {
    if (onChange) {
      onChange(id);
    } else {
      setSelectedModel(id);
    }
    setIsOpen(false);
  };

  const getFriendlyName = (p) => {
    if (!p) return currentModelId;
    if (p.id === 'gemini_flash') return 'Google Gemini 2.5 Flash';
    if (p.id === 'gemini_pro') return 'Google Gemini 2.5 Pro';
    if (p.id === 'gemini_flash_lite') return 'Gemini 2.5 Flash-Lite';
    if (p.id === 'ollama_llama3') return 'Llama 3.1 8B (Local)';
    if (p.id === 'ollama_deepseek') return 'DeepSeek R1 14B (Local)';
    if (p.id === 'ollama_gemma4_e4b') return 'Gemma 4 E4B (Vision)';
    if (p.id === 'ollama_gemma4_12b') return 'Gemma 4 12B (Local)';
    if (p.id === 'ollama_qwen3vl') return 'Qwen3-VL (Vision)';
    if (p.id === 'ollama_nemotron') return 'Nemotron 3.5 Lightning';
    if (p.id === 'openai_gpt4o') return 'OpenAI GPT-4o';
    if (p.id === 'openai_gpt4o_mini') return 'OpenAI GPT-4o Mini';
    if (p.id === 'openrouter_gemma4_26b') return 'Gemma 4 26B (OpenRouter)';
    return p.model || p.id;
  };

  const formatContext = (ctx) => {
    if (!ctx) return '128K';
    if (ctx >= 1000000) return '1M ctx';
    if (ctx >= 1000) return `${Math.round(ctx / 1000)}K ctx`;
    return `${ctx} ctx`;
  };

  const isOnline = (pId) => {
    if (pId.startsWith('laptop:')) return laptopBridgeStatus.online;
    if (pId.startsWith('gemini')) return true; // cloud reachability defaults to ready
    return providerReachability[pId] !== false;
  };

  const filterMatches = (p) => {
    if (!search.trim()) return true;
    const q = search.toLowerCase();
    const name = getFriendlyName(p).toLowerCase();
    const id = (p.id || '').toLowerCase();
    const model = (p.model || '').toLowerCase();
    return name.includes(q) || id.includes(q) || model.includes(q);
  };

  return (
    <div className={`relative inline-block text-left ${className}`} ref={popoverRef}>
      {showLabel && !compact && (
        <span className="text-[10px] uppercase font-mono tracking-wider text-slate-500 mr-2 inline-flex items-center gap-1">
          <Cpu className="w-3 h-3 text-emerald-400" />
          Model:
        </span>
      )}

      {/* Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 rounded-xl text-xs font-mono font-medium transition-all ${
          compact
            ? 'px-2.5 py-1 bg-surface border border-border-subtle hover:border-slate-600 text-text-primary'
            : 'px-3 py-1.5 bg-surface border border-border-subtle hover:border-emerald-500/50 text-text-primary shadow-sm'
        }`}
        title={`Active inference model: ${currentModelId}`}
      >
        <span
          className={`w-2 h-2 rounded-full shrink-0 ${
            isOnline(currentModelId) ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
          }`}
        />
        <span className="font-semibold truncate max-w-[170px] sm:max-w-[210px]">
          {getFriendlyName(activeDetails)}
        </span>
        <span className="px-1.5 py-0.2 rounded text-[10px] bg-elevated text-text-secondary border border-border-subtle shrink-0">
          {formatContext(activeDetails?.context_window)}
        </span>
        <ChevronDown className={`w-3.5 h-3.5 text-text-muted transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Popover Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 sm:w-96 rounded-2xl bg-surface border border-border-subtle shadow-2xl z-50 overflow-hidden backdrop-blur-xl animate-in fade-in zoom-in-95 duration-100">
          {/* Header & Search */}
          <div className="p-3 border-b border-border-subtle bg-elevated/40 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-text-primary flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-accent-primary" />
                Active Model Pipeline
              </span>
              <span className="text-[10px] font-mono text-text-muted uppercase">Hot-Swappable</span>
            </div>
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Filter models, providers, context..."
                className="w-full pl-8 pr-3 py-1 text-xs bg-canvas border border-border-subtle rounded-lg text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-primary font-sans"
                autoFocus
              />
            </div>
          </div>

          {/* Grouped Options List */}
          <div className="max-h-80 overflow-y-auto p-2 space-y-3 font-sans">
            {/* Group 1: Google Cloud Vertex / Gemini */}
            {categorized.gemini.filter(filterMatches).length > 0 && (
              <div>
                <div className="px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider font-bold text-sky-400 flex items-center gap-1.5">
                  <Cloud className="w-3 h-3 text-sky-400" />
                  Google Cloud Vertex / Gemini
                </div>
                <div className="space-y-1 mt-1">
                  {categorized.gemini.filter(filterMatches).map((p) => {
                    const isSelected = p.id === currentModelId;
                    const online = isOnline(p.id);
                    return (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => handleSelect(p.id)}
                        className={`w-full text-left px-2.5 py-2 rounded-xl text-xs flex items-center justify-between transition-all ${
                          isSelected
                            ? 'bg-sky-500/15 border border-sky-500/40 text-sky-200'
                            : 'hover:bg-elevated border border-transparent text-text-primary'
                        }`}
                      >
                        <div className="space-y-0.5 min-w-0 pr-2">
                          <div className="flex items-center gap-1.5 font-semibold text-xs truncate">
                            <span className={`w-1.5 h-1.5 rounded-full ${online ? 'bg-emerald-400' : 'bg-red-400'}`} />
                            <span className="truncate">{getFriendlyName(p)}</span>
                          </div>
                          <div className="flex items-center gap-1.5 text-[10px] text-text-muted font-mono">
                            <span>{formatContext(p.context_window)}</span>
                            {p.supports_vision && <span className="text-purple-400">· Vision</span>}
                            {p.supports_tool_use && <span className="text-emerald-400">· Tools</span>}
                          </div>
                        </div>
                        {isSelected && <Check className="w-4 h-4 text-sky-400 shrink-0" />}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Group 2: Local Sovereign Ollama */}
            {categorized.local.filter(filterMatches).length > 0 && (
              <div>
                <div className="px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider font-bold text-amber-400 flex items-center gap-1.5">
                  <Laptop className="w-3 h-3 text-amber-400" />
                  Local Sovereign Ollama
                </div>
                <div className="space-y-1 mt-1">
                  {categorized.local.filter(filterMatches).map((p) => {
                    const isSelected = p.id === currentModelId;
                    const online = isOnline(p.id);
                    return (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => handleSelect(p.id)}
                        className={`w-full text-left px-2.5 py-2 rounded-xl text-xs flex items-center justify-between transition-all ${
                          isSelected
                            ? 'bg-amber-500/15 border border-amber-500/40 text-amber-200'
                            : 'hover:bg-elevated border border-transparent text-text-primary'
                        }`}
                      >
                        <div className="space-y-0.5 min-w-0 pr-2">
                          <div className="flex items-center gap-1.5 font-semibold text-xs truncate">
                            <span className={`w-1.5 h-1.5 rounded-full ${online ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                            <span className="truncate">{getFriendlyName(p)}</span>
                          </div>
                          <div className="flex items-center gap-1.5 text-[10px] text-text-muted font-mono">
                            <span>{formatContext(p.context_window)}</span>
                            {p.supports_vision && <span className="text-purple-400">· Vision</span>}
                            {p.supports_tool_use && <span className="text-emerald-400">· Tools</span>}
                            <span>· Zero Cloud Egress</span>
                          </div>
                        </div>
                        {isSelected && <Check className="w-4 h-4 text-amber-400 shrink-0" />}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Group 3: Hosted Gateways — Groq / NVIDIA / OpenAI */}
            {categorized.hosted.filter(filterMatches).length > 0 && (
              <div>
                <div className="px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider font-bold text-purple-400 flex items-center gap-1.5">
                  <Zap className="w-3 h-3 text-purple-400" />
                  Hosted Gateways — Groq / NVIDIA / OpenAI
                </div>
                <div className="space-y-1 mt-1">
                  {categorized.hosted.filter(filterMatches).map((p) => {
                    const isSelected = p.id === currentModelId;
                    const online = isOnline(p.id);
                    return (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => handleSelect(p.id)}
                        className={`w-full text-left px-2.5 py-2 rounded-xl text-xs flex items-center justify-between transition-all ${
                          isSelected
                            ? 'bg-purple-500/15 border border-purple-500/40 text-purple-200'
                            : 'hover:bg-elevated border border-transparent text-text-primary'
                        }`}
                      >
                        <div className="space-y-0.5 min-w-0 pr-2">
                          <div className="flex items-center gap-1.5 font-semibold text-xs truncate">
                            <span className={`w-1.5 h-1.5 rounded-full ${online ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                            <span className="truncate">{getFriendlyName(p)}</span>
                          </div>
                          <div className="flex items-center gap-1.5 text-[10px] text-text-muted font-mono">
                            <span>{formatContext(p.context_window)}</span>
                            {p.supports_vision && <span className="text-purple-400">· Vision</span>}
                            <span>· High Throughput</span>
                          </div>
                        </div>
                        {isSelected && <Check className="w-4 h-4 text-purple-400 shrink-0" />}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Footer Quick Link to Model Settings */}
          <div className="p-2.5 border-t border-border-subtle bg-elevated/60 flex items-center justify-between text-xs">
            <span className="text-[11px] text-text-muted">Need fine-grained bindings?</span>
            <button
              type="button"
              onClick={() => {
                setIsOpen(false);
                setActiveTab('settings');
              }}
              className="text-xs font-bold text-accent-primary hover:underline flex items-center gap-1"
            >
              <span>Configure Pipeline Matrix</span>
              <ExternalLink className="w-3 h-3" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
