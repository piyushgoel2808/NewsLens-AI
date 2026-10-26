import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from 'react';
import { deduplicatedFetch } from '../utils/apiDeduplicator';

export const DEFAULT_CLOUD_FULL_BINDINGS = {
  query_planner: 'gemini_flash',
  answerer: 'gemini_flash',
  answer_verifier: 'gemini_flash',
  metadata_extraction: 'gemini_flash',
  classification: 'gemini_flash',
  article_segmentation: 'gemini_flash',
  visual_extraction: 'gemini_flash',
  layout_analysis: 'gemini_flash',
  document_parser: 'docling_cloud_parser',
  ocr: 'docling_cloud_parser',
  embedding: 'gemini_embedding',
};

export const PRODUCTION_ARCHITECTURE_PRESETS = {
  cloud_full: {
    id: 'cloud_full',
    name: 'Full Cloud Google Gemini',
    badge: 'Zero Container RAM (GCP Prod)',
    description:
      'Routes all LLM reasoning, article segmentation, and visual extraction through Google Gemini 2.5 Flash, with document parsing and OCR powered by high-speed IBM Cloud Docling API, and vector indexing routed to Qdrant article_chunks_v2 via Gemini Embedding 001.',
    pipelineHighlights: {
      reasoning: 'Google Gemini 2.5 Flash (1M ctx, Vertex AI / AI Studio)',
      vision: 'Gemini 2.5 Flash Single-Pass VLM + IBM Docling Cloud',
      indexing: 'gemini-embedding-001 (768d MRL → article_chunks_v2)',
    },
    bindings: {
      query_planner: 'gemini_flash',
      answerer: 'gemini_flash',
      answer_verifier: 'gemini_flash',
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
  cloud_hybrid: {
    id: 'cloud_hybrid',
    name: 'High-Speed Hybrid',
    badge: 'Lightning Fast',
    description:
      'Combines Google Gemini 2.5 Flash and NVIDIA Nemotron 3.5 Lightning for reasoning, paired with specialized 2D document parsing and local/cloud vector embeddings.',
    pipelineHighlights: {
      reasoning: 'Gemini 2.5 Flash + NVIDIA Nemotron 3.5 Lightning',
      vision: 'Gemini 2.5 Flash / Gemma 4 E4B + Local Docling 2D',
      indexing: 'BAAI/bge-m3 (1024d → article_chunks) or EmbeddingGemma 2 (768d)',
    },
    bindings: {
      query_planner: 'gemini_flash',
      answerer: 'gemini_flash',
      answer_verifier: 'gemini_flash',
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
  local_offline: {
    id: 'local_offline',
    name: 'Local Sovereign',
    badge: '100% On-Premise (No API Keys)',
    description:
      'Private on-premise execution with zero external API calls via Ollama hardware acceleration and local DocLayNet layout parsing.',
    pipelineHighlights: {
      reasoning: 'Gemma 4 12B / Llama 3.1 8B / DeepSeek-R1 14B (Ollama)',
      vision: 'Gemma 4 E4B / Qwen3-VL + Local Docling DocLayNet',
      indexing: 'BAAI/bge-m3 (1024d Local → article_chunks)',
    },
    bindings: {
      query_planner: 'ollama_llama3',
      answerer: 'ollama_llama3',
      answer_verifier: 'ollama_deepseek',
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
};

// Fallback providers matching backend model_config.yaml
export const FALLBACK_CONFIGURED_PROVIDERS = [
  {
    id: 'gemini_flash',
    provider: 'gemini',
    model: 'gemini-2.5-flash',
    context_window: 1048576,
    supports_vision: true,
    supports_tool_use: true,
    family: 'Google Gemini',
  },
  {
    id: 'gemini_pro',
    provider: 'gemini',
    model: 'gemini-2.5-pro',
    context_window: 1048576,
    supports_vision: true,
    supports_tool_use: true,
    family: 'Google Gemini',
  },
  {
    id: 'gemini_flash_lite',
    provider: 'gemini',
    model: 'gemini-2.5-flash',
    context_window: 1048576,
    supports_vision: true,
    supports_tool_use: true,
    family: 'Google Gemini',
  },
  {
    id: 'ollama_llama3',
    provider: 'ollama',
    model: 'llama3.1:8b',
    context_window: 128000,
    supports_vision: false,
    supports_tool_use: true,
    family: 'Local Sovereign Ollama',
  },
  {
    id: 'ollama_deepseek',
    provider: 'ollama',
    model: 'deepseek-r1:14b',
    context_window: 65536,
    supports_vision: false,
    supports_tool_use: true,
    family: 'Local Sovereign Ollama',
  },
  {
    id: 'ollama_gemma4_e4b',
    provider: 'ollama',
    model: 'gemma4:e4b',
    context_window: 131072,
    supports_vision: true,
    supports_tool_use: true,
    family: 'Local Sovereign Ollama',
  },
  {
    id: 'ollama_gemma4_12b',
    provider: 'ollama',
    model: 'gemma4:12b',
    context_window: 131072,
    supports_vision: false,
    supports_tool_use: true,
    family: 'Local Sovereign Ollama',
  },
  {
    id: 'ollama_qwen3vl',
    provider: 'ollama',
    model: 'qwen3-vl:latest',
    context_window: 32768,
    supports_vision: true,
    supports_tool_use: true,
    family: 'Local Sovereign Ollama',
  },
  {
    id: 'ollama_nemotron',
    provider: 'ollama',
    model: 'nemotron-3.5-lightning:latest',
    context_window: 128000,
    supports_vision: false,
    supports_tool_use: true,
    family: 'Local Sovereign Ollama',
  },
  {
    id: 'openai_gpt4o',
    provider: 'openai',
    model: 'gpt-4o',
    context_window: 128000,
    supports_vision: true,
    supports_tool_use: true,
    family: 'Hosted Gateways',
  },
  {
    id: 'openai_gpt4o_mini',
    provider: 'openai',
    model: 'gpt-4o-mini',
    context_window: 128000,
    supports_vision: true,
    supports_tool_use: true,
    family: 'Hosted Gateways',
  },
  {
    id: 'openrouter_gemma4_26b',
    provider: 'openrouter',
    model: 'google/gemma-4-26b-a4b-it:free',
    context_window: 262144,
    supports_vision: true,
    supports_tool_use: true,
    family: 'Hosted Gateways',
  },
];

const ModelConfigContext = createContext(null);

export function ModelConfigProvider({ children }) {
  // Theme state: 'midnight' or 'archival'
  const [theme, setThemeState] = useState(() => {
    try {
      return localStorage.getItem('newslens_theme') || 'midnight';
    } catch {
      return 'midnight';
    }
  });

  const setTheme = useCallback((newTheme) => {
    const valid = newTheme === 'archival' ? 'archival' : 'midnight';
    setThemeState(valid);
    try {
      localStorage.setItem('newslens_theme', valid);
    } catch {
      // ignore
    }
    document.documentElement.setAttribute('data-theme', valid);
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'midnight' ? 'archival' : 'midnight');
  }, [theme, setTheme]);

  // Sync initial theme with DOM
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Task bindings
  const [taskBindings, setTaskBindings] = useState(() => {
    try {
      const saved = localStorage.getItem('newslens_task_bindings');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed && typeof parsed === 'object' && Object.keys(parsed).length > 0) {
          return { ...DEFAULT_CLOUD_FULL_BINDINGS, ...parsed };
        }
      }
    } catch {
      // ignore
    }
    return { ...DEFAULT_CLOUD_FULL_BINDINGS };
  });

  // Configured providers list from backend
  const [configuredProviders, setConfiguredProviders] = useState(FALLBACK_CONFIGURED_PROVIDERS);
  const [providerReachability, setProviderReachability] = useState({});

  // Laptop Bridge Node Status
  const [laptopBridgeStatus, setLaptopBridgeStatus] = useState({
    enabled: false,
    online: false,
    available_models: [],
    default_model: 'llama3.1:8b',
  });

  // Selected LLM override for per-query workflows
  const [selectedModel, setSelectedModelState] = useState(() => {
    return localStorage.getItem('newslens_selected_model') || 'gemini_flash';
  });

  // Fetch bindings & providers on app mount
  const refreshModelBindings = useCallback(async () => {
    try {
      const data = await deduplicatedFetch('/api/settings/model-bindings');
      if (data) {
        if (data.task_bindings) {
          setTaskBindings((prev) => {
            const merged = { ...prev, ...data.task_bindings };
            try {
              localStorage.setItem('newslens_task_bindings', JSON.stringify(merged));
            } catch {
              // ignore
            }
            return merged;
          });
        }
        if (Array.isArray(data.configured_providers) && data.configured_providers.length > 0) {
          setConfiguredProviders(data.configured_providers);
        }
        if (data.provider_reachability) {
          setProviderReachability(data.provider_reachability);
        }
      }
    } catch (err) {
      console.warn('Failed to load backend model bindings:', err);
    }
  }, []);

  // Poll laptop node status periodically
  useEffect(() => {
    refreshModelBindings();

    const checkLaptop = () => {
      fetch('/api/settings/laptop-node/status')
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data) setLaptopBridgeStatus(data);
        })
        .catch(() => {});
    };
    checkLaptop();
    const interval = setInterval(checkLaptop, 30000);
    return () => clearInterval(interval);
  }, [refreshModelBindings]);

  // Determine active preset based on taskBindings
  const activePreset = useMemo(() => {
    const qp = taskBindings?.query_planner || '';
    const ans = taskBindings?.answerer || '';
    const emb = taskBindings?.embedding || '';

    if (qp === 'gemini_flash' && ans === 'gemini_flash' && emb === 'gemini_embedding') {
      return 'cloud_full';
    }
    if (qp === 'gemini_flash' && (emb === 'local_embed_bge' || taskBindings?.layout_analysis === 'docling_parser')) {
      return 'cloud_hybrid';
    }
    if (qp.startsWith('ollama') || qp.startsWith('laptop')) {
      return 'local_offline';
    }
    return 'custom';
  }, [taskBindings]);

  // Update a single task binding and sync with backend
  const updateTaskBinding = useCallback(
    async (task, providerId) => {
      if (!task || !providerId) return;

      const updated = { ...taskBindings, [task]: providerId };
      setTaskBindings(updated);
      try {
        localStorage.setItem('newslens_task_bindings', JSON.stringify(updated));
      } catch {
        // ignore
      }

      if (task === 'answerer' || task === 'query_planner') {
        setSelectedModelState(providerId);
        try {
          localStorage.setItem('newslens_selected_model', providerId);
        } catch {
          // ignore
        }
      }

      try {
        await fetch('/api/settings/model-bindings', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            task_bindings: { [task]: providerId },
          }),
        });
      } catch (err) {
        console.warn(`Failed to persist binding ${task}=${providerId}:`, err);
      }
    },
    [taskBindings]
  );

  // Apply an entire architecture preset
  const applyArchitecturePreset = useCallback(async (presetKey) => {
    const preset = PRODUCTION_ARCHITECTURE_PRESETS[presetKey];
    if (!preset) return;

    setTaskBindings((prev) => {
      const merged = { ...prev, ...preset.bindings };
      try {
        localStorage.setItem('newslens_task_bindings', JSON.stringify(merged));
      } catch {
        // ignore
      }
      return merged;
    });

    const activeModel = preset.bindings.answerer || preset.bindings.query_planner || 'gemini_flash';
    setSelectedModelState(activeModel);
    try {
      localStorage.setItem('newslens_selected_model', activeModel);
    } catch {
      // ignore
    }

    try {
      await fetch('/api/settings/model-bindings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_bindings: preset.bindings,
        }),
      });
    } catch (err) {
      console.warn('Failed to persist architecture preset to backend:', err);
    }
  }, []);

  const setSelectedModel = useCallback((model, syncToBackend = true) => {
    if (!model) return;
    setSelectedModelState(model);
    try {
      localStorage.setItem('newslens_selected_model', model);
    } catch {
      // ignore
    }

    setTaskBindings((prev) => {
      const updated = {
        ...prev,
        query_planner: model,
        answerer: model,
      };
      try {
        localStorage.setItem('newslens_task_bindings', JSON.stringify(updated));
      } catch {
        // ignore
      }
      return updated;
    });

    if (syncToBackend) {
      fetch('/api/settings/model-bindings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_bindings: {
            query_planner: model,
            answerer: model,
          },
        }),
      }).catch((err) => console.warn('Failed to sync model to backend:', err));
    }
  }, []);

  return (
    <ModelConfigContext.Provider
      value={{
        theme,
        setTheme,
        toggleTheme,
        taskBindings,
        setTaskBindings,
        configuredProviders,
        providerReachability,
        laptopBridgeStatus,
        activePreset,
        selectedModel,
        setSelectedModel,
        updateTaskBinding,
        applyArchitecturePreset,
        refreshModelBindings,
      }}
    >
      {children}
    </ModelConfigContext.Provider>
  );
}

export function useModelConfig() {
  const ctx = useContext(ModelConfigContext);
  if (!ctx) {
    throw new Error('useModelConfig must be used within a ModelConfigProvider');
  }
  return ctx;
}
