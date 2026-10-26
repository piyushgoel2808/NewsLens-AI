import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useModelConfig } from './ModelConfigContext';

const ActiveHighlightContext = createContext(null);

export function ActiveHighlightProvider({ children }) {
  const modelConfig = useModelConfig();
  const [activeTab, setActiveTab] = useState('home');
  const [selectedIssueId, setSelectedIssueId] = useState(null);
  const [selectedPageNumber, setSelectedPageNumber] = useState(1);
  const [selectedArticleId, setSelectedArticleId] = useState(null);
  const [highlightedBboxes, setHighlightedBboxes] = useState([]);
  const [isPulsing, setIsPulsing] = useState(false);
  const [hoveredArticleId, setHoveredArticleId] = useState(null);

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

  const [pendingQuery, setPendingQuery] = useState('');

  const runAgentQuery = useCallback((queryStr) => {
    if (queryStr) {
      setPendingQuery(queryStr);
    }
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
        // Model config bridge properties
        selectedModel: modelConfig.selectedModel,
        setSelectedModel: modelConfig.setSelectedModel,
        taskBindings: modelConfig.taskBindings,
        updateTaskBindings: modelConfig.updateTaskBinding,
        refreshTaskBindings: modelConfig.refreshModelBindings,
        configuredProviders: modelConfig.configuredProviders,
        providerReachability: modelConfig.providerReachability,
        activePreset: modelConfig.activePreset,
        applyArchitecturePreset: modelConfig.applyArchitecturePreset,
        theme: modelConfig.theme,
        setTheme: modelConfig.setTheme,
        toggleTheme: modelConfig.toggleTheme,
        // Chat & navigation properties
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
        pendingQuery,
        setPendingQuery,
        runAgentQuery,
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
