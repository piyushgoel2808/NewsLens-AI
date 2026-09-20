import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { deduplicatedFetch, clearApiCache } from '../utils/apiDeduplicator';


export const DEFAULT_CLOUD_FULL_BINDINGS = {
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
};

const ActiveHighlightContext = createContext(null);

export function ActiveHighlightProvider({ children }) {
  const [activeTab, setActiveTab] = useState('home');
  const [selectedIssueId, setSelectedIssueId] = useState(null);
  const [selectedPageNumber, setSelectedPageNumber] = useState(1);
  const [selectedArticleId, setSelectedArticleId] = useState(null);
  const [highlightedBboxes, setHighlightedBboxes] = useState([]);
  const [isPulsing, setIsPulsing] = useState(false);
  const [hoveredArticleId, setHoveredArticleId] = useState(null);

  // Synchronized Task Bindings State across all tabs (Default: Full Cloud Google Gemini)
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

  // Persistent Selected LLM Model (Default: gemini_flash)
  const [selectedModel, setSelectedModelState] = useState(() => {
    return localStorage.getItem('newslens_selected_model') || 'gemini_flash';
  });

  // Refresh and synchronize task bindings from backend on mount
  const refreshTaskBindings = useCallback(async () => {
    try {
      const json = await deduplicatedFetch('/api/settings/model-bindings');
      if (json && json.task_bindings) {
        setTaskBindings(json.task_bindings);
        try {
          localStorage.setItem('newslens_task_bindings', JSON.stringify(json.task_bindings));
        } catch {
          // ignore
        }

        // If no explicitly saved model in localStorage, bind to backend answerer
        const activeLlm = json.task_bindings.answerer || json.task_bindings.query_planner;
        const storedModel = localStorage.getItem('newslens_selected_model');
        if (!storedModel && activeLlm) {
          setSelectedModelState(activeLlm);
          localStorage.setItem('newslens_selected_model', activeLlm);
        }
      }
    } catch (err) {
      console.warn('Failed to load initial model bindings in context:', err);
    }
  }, []);


  useEffect(() => {
    refreshTaskBindings();
  }, [refreshTaskBindings]);

  // Synchronized updater called when presets or settings change
  const updateTaskBindings = useCallback((newBindings, syncSelectedModel = true) => {
    if (!newBindings || typeof newBindings !== 'object') return;
    setTaskBindings((prev) => {
      const merged = { ...prev, ...newBindings };
      try {
        localStorage.setItem('newslens_task_bindings', JSON.stringify(merged));
      } catch {
        // ignore
      }
      return merged;
    });

    if (syncSelectedModel) {
      const nextModel = newBindings.answerer || newBindings.query_planner;
      if (nextModel) {
        setSelectedModelState(nextModel);
        localStorage.setItem('newslens_selected_model', nextModel);
      }
    }
  }, []);

  // Set selected model and sync with backend bindings so all tabs stay in harmony
  const setSelectedModel = useCallback((model, syncToBackend = true) => {
    if (!model) return;
    setSelectedModelState(model);
    localStorage.setItem('newslens_selected_model', model);

    // Update in-memory bindings for query planner & answerer
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

    // Optionally propagate to backend settings
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
      }).catch((err) => console.warn('Failed to sync model binding to backend:', err));
    }
  }, []);

  // Persistent Chat Messages across tab switches and reloads
  const [chatMessages, setChatMessagesState] = useState(() => {
    try {
      const saved = localStorage.getItem('newslens_chat_messages');
      if (saved) return JSON.parse(saved);
    } catch {
      // ignore
    }
    return [
      {
        role: 'assistant',
        content:
          'Hello! I am your **NewsLens-AI Research Assistant**. I can perform multi-step newspaper intelligence investigations, cross-newspaper comparative analysis, quantitative trend tracking, and temporal timeline reconstruction with verifiable spatial citations.',
        isStreaming: false,
      },
    ];
  });

  const setChatMessages = useCallback((updater) => {
    setChatMessagesState((prev) => {
      const next = typeof updater === 'function' ? updater(prev) : updater;
      try {
        localStorage.setItem('newslens_chat_messages', JSON.stringify(next));
      } catch {
        // ignore
      }
      return next;
    });
  }, []);

  const highlightArticle = useCallback((issueId, pageNumber, articleId, bboxes = []) => {
    const pageNum = Number(pageNumber) || 1;
    if (issueId) setSelectedIssueId(Number(issueId));
    setSelectedPageNumber(pageNum);
    if (articleId) {
      const artIdNum = Number(articleId);
      setSelectedArticleId(artIdNum);
      setActiveAttachedAsset((prev) => (prev && prev.articleId && Number(prev.articleId) !== artIdNum ? null : prev));
    }
    const safeBboxes = Array.isArray(bboxes)
      ? bboxes
      : bboxes && typeof bboxes === 'object'
      ? Object.values(bboxes)
      : [];
    setHighlightedBboxes(safeBboxes);
    setIsPulsing(true);
    setActiveTab('reader');

    setTimeout(() => {
      setIsPulsing(false);
    }, 4000);
  }, []);

  const [timelineQuery, setTimelineQuery] = useState('');

  const openTimeline = useCallback((query = '') => {
    if (query) setTimelineQuery(query);
    setActiveTab('timeline');
  }, []);

  const openIssueInReader = useCallback((issueId, pageNumber = 1) => {
    const pageNum = Number(pageNumber) || 1;
    if (issueId) setSelectedIssueId(Number(issueId));
    setSelectedPageNumber(pageNum);
    setSelectedArticleId(null);
    setHighlightedBboxes([]);
    setActiveAttachedAsset(null);
    setActiveTab('reader');
  }, []);

  // Attached Visual Asset (Infographics, Data Charts, Tables) for Agent Assistant interrogation
  const [activeAttachedAsset, setActiveAttachedAsset] = useState(null);

  const attachAssetForAgent = useCallback((asset) => {
    setActiveAttachedAsset(asset);
    setActiveTab('agent');
  }, []);

  const clearAttachedAsset = useCallback(() => {
    setActiveAttachedAsset(null);
  }, []);

  return (
    <ActiveHighlightContext.Provider
      value={{
        activeTab,
        setActiveTab,
        selectedIssueId,
        setSelectedIssueId,
        selectedPageNumber,
        setSelectedPageNumber,
        selectedArticleId,
        setSelectedArticleId,
        highlightedBboxes,
        setHighlightedBboxes,
        isPulsing,
        setIsPulsing,
        hoveredArticleId,
        setHoveredArticleId,
        selectedModel,
        setSelectedModel,
        taskBindings,
        updateTaskBindings,
        refreshTaskBindings,
        chatMessages,
        setChatMessages,
        timelineQuery,
        setTimelineQuery,
        openTimeline,
        highlightArticle,
        openIssueInReader,
        activeAttachedAsset,
        setActiveAttachedAsset,
        attachAssetForAgent,
        clearAttachedAsset,
      }}
    >
      {children}
    </ActiveHighlightContext.Provider>
  );
}

export function useActiveHighlight() {
  const context = useContext(ActiveHighlightContext);
  if (!context) {
    throw new Error('useActiveHighlight must be used within an ActiveHighlightProvider');
  }
  return context;
}
