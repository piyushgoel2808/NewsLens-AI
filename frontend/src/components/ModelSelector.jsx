import React, { useState, useEffect } from 'react';
import { Cpu, Cloud, Laptop, Sparkles, ShieldCheck } from 'lucide-react';

// Hardcoded reference models for clean display labels
const CORE_MODELS = {
  openrouter: [
    { id: 'openrouter_gemma4_26b', label: 'Gemma 4 26B (Dual-Key Free)', hint: 'Vision + Chat' },
    { id: 'openrouter_nemotron', label: 'Nemotron 3.5 Lightning (Dual-Key Free)', hint: 'Reasoning' },
  ],
  cloud_direct: [
    { id: 'gemini_flash', label: 'Gemini 3.7 Flash', hint: 'Google Grounding' },
    { id: 'gemini_pro', label: 'Gemini Pro Latest', hint: 'Google Deep Analysis' },
    { id: 'groq_compound', label: 'Groq Compound AI', hint: 'Ultra-Fast' },
    { id: 'groq_qwen', label: 'Groq Qwen 3.6 27B', hint: 'Fast Reasoning' },
    { id: 'groq_gpt_oss', label: 'Groq GPT-OSS 120B', hint: 'Open Weight' },
    { id: 'openai_gpt4o', label: 'OpenAI GPT-4o', hint: 'Omni Multimodal' },
    { id: 'openai_gpt4o_mini', label: 'OpenAI GPT-4o Mini', hint: 'Lightweight' },
  ],
  local: [
    { id: 'ollama_llama3', label: 'Llama 3.1 8B (Local)', hint: 'Fast General (~5GB)' },
    { id: 'ollama_deepseek', label: 'DeepSeek R1 14B (Local)', hint: 'Local Reasoning' },
    { id: 'ollama_qwen3vl', label: 'Qwen 3 VL (Local)', hint: 'Vision & Multimodal' },
    { id: 'ollama_vlm', label: 'Qwen 2.5 VL 7B (Local)', hint: 'Vision Extraction' },
  ],
};

export default function ModelSelector({
  value,
  onChange,
  availableModels: propModels,
  className = '',
  compact = false,
  showLabel = true,
}) {
  const [fetchedModels, setFetchedModels] = useState([]);

  useEffect(() => {
    if (!propModels || propModels.length === 0) {
      fetch('/api/models/available')
        .then((r) => r.json())
        .then((data) => {
          if (data.models || data.providers) {
            setFetchedModels(data.models || data.providers || []);
          }
        })
        .catch((err) => console.error('ModelSelector failed to fetch models:', err));
    }
  }, [propModels]);

  const allAvailable = propModels && propModels.length > 0 ? propModels : fetchedModels;

  // Determine environment tag for currently selected model
  const getModelEnv = (modelId) => {
    if (!modelId) return { type: 'cloud_or', label: 'Dual-Key Cloud', color: 'emerald' };
    const lower = modelId.toLowerCase();
    if (
      lower.startsWith('openrouter') ||
      lower.includes('google/gemma-4') ||
      lower.includes('nemotron-3.5-lightning:free')
    ) {
      return { type: 'cloud_or', label: 'Dual-Key Cloud', color: 'emerald' };
    }
    if (
      lower.startsWith('gemini') ||
      lower.startsWith('groq') ||
      lower.startsWith('openai') ||
      lower.includes('gpt')
    ) {
      return { type: 'cloud', label: 'Cloud Direct', color: 'sky' };
    }
    if (lower.startsWith('ollama') || lower.startsWith('local')) {
      return { type: 'local', label: 'Local Machine', color: 'amber' };
    }
    return { type: 'custom', label: 'Custom', color: 'slate' };
  };

  const currentEnv = getModelEnv(value);

  // Filter out any models from dynamically fetched that are already represented in core lists
  const knownIds = new Set([
    ...CORE_MODELS.openrouter.map((m) => m.id),
    ...CORE_MODELS.cloud_direct.map((m) => m.id),
    ...CORE_MODELS.local.map((m) => m.id),
  ]);

  const extraCloudOR = allAvailable.filter(
    (m) =>
      !knownIds.has(m.id || m.name) &&
      (m.provider === 'openrouter' || (m.id && m.id.includes('openrouter')))
  );

  const extraCloudDirect = allAvailable.filter(
    (m) =>
      !knownIds.has(m.id || m.name) &&
      !m.is_local &&
      m.provider !== 'openrouter' &&
      !(m.id && m.id.includes('openrouter'))
  );

  const extraLocal = allAvailable.filter(
    (m) => !knownIds.has(m.id || m.name) && m.is_local
  );

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {showLabel && !compact && (
        <div className="flex items-center gap-1.5 text-slate-400 font-medium text-xs select-none">
          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
          <span>Model:</span>
        </div>
      )}

      {/* Grouped Model Dropdown */}
      <div className="relative inline-flex items-center">
        <select
          value={value || 'openrouter_gemma4_26b'}
          onChange={(e) => onChange && onChange(e.target.value)}
          className={`appearance-none bg-slate-900 border rounded-lg pl-2.5 pr-8 py-1 font-medium text-xs outline-none cursor-pointer transition-all duration-150 ${
            currentEnv.color === 'emerald'
              ? 'border-emerald-500/50 text-emerald-300 hover:border-emerald-400 focus:border-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.15)]'
              : currentEnv.color === 'sky'
              ? 'border-sky-500/50 text-sky-300 hover:border-sky-400 focus:border-sky-400'
              : 'border-amber-500/50 text-amber-300 hover:border-amber-400 focus:border-amber-400'
          }`}
          title={`Active inference model: ${value}`}
        >
          {/* Group 1: OpenRouter Dual-Key Cloud */}
          <optgroup label="☁️ OpenRouter Cloud (Dual-Key Free Tier)" className="bg-slate-900 text-slate-400 font-semibold">
            {CORE_MODELS.openrouter.map((m) => (
              <option key={m.id} value={m.id} className="bg-slate-900 text-slate-200 font-normal">
                ☁️ {m.label} — {m.hint}
              </option>
            ))}
            {extraCloudOR.map((m) => (
              <option key={m.id || m.name} value={m.id || m.name} className="bg-slate-900 text-slate-200 font-normal">
                ☁️ {m.name || m.id} ({m.model || m.provider})
              </option>
            ))}
          </optgroup>

          {/* Group 2: Direct Cloud Hosted */}
          <optgroup label="⚡ Cloud Hosted Direct" className="bg-slate-900 text-slate-400 font-semibold">
            {CORE_MODELS.cloud_direct.map((m) => (
              <option key={m.id} value={m.id} className="bg-slate-900 text-slate-200 font-normal">
                ✨ {m.label} — {m.hint}
              </option>
            ))}
            {extraCloudDirect.map((m) => (
              <option key={m.id || m.name} value={m.id || m.name} className="bg-slate-900 text-slate-200 font-normal">
                ⚡ {m.name || m.id} ({m.provider})
              </option>
            ))}
          </optgroup>

          {/* Group 3: Local Offline Inference */}
          <optgroup label="🖥️ Local Offline (Ollama - High Compute)" className="bg-slate-900 text-slate-400 font-semibold">
            {CORE_MODELS.local.map((m) => (
              <option key={m.id} value={m.id} className="bg-slate-900 text-slate-200 font-normal">
                🟢 {m.label} — {m.hint}
              </option>
            ))}
            {extraLocal.map((m) => (
              <option key={m.id || m.name} value={m.id || m.name} className="bg-slate-900 text-slate-200 font-normal">
                🟢 {m.name || m.id} (Local {m.provider})
              </option>
            ))}
          </optgroup>
        </select>

        {/* Custom Chevron Indicator */}
        <span className="pointer-events-none absolute right-2.5 flex items-center text-slate-500 text-[10px]">
          ▼
        </span>
      </div>

      {/* Visual Environment Badge */}
      {!compact && (
        <span
          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold tracking-wide border transition-all ${
            currentEnv.color === 'emerald'
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 shadow-[0_0_8px_rgba(16,185,129,0.12)]'
              : currentEnv.color === 'sky'
              ? 'bg-sky-500/10 text-sky-400 border-sky-500/30'
              : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
          }`}
          title={
            currentEnv.type === 'cloud_or'
              ? 'Dual-Account Round-Robin distribution over OpenRouter'
              : currentEnv.type === 'cloud'
              ? 'Direct API calls to cloud provider'
              : 'Runs locally on device hardware via Ollama'
          }
        >
          {currentEnv.type === 'cloud_or' && <ShieldCheck className="w-3 h-3 text-emerald-400" />}
          {currentEnv.type === 'cloud' && <Cloud className="w-3 h-3 text-sky-400" />}
          {currentEnv.type === 'local' && <Laptop className="w-3 h-3 text-amber-400" />}
          <span>{currentEnv.label}</span>
        </span>
      )}
    </div>
  );
}
