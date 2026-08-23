import React, { createContext, useContext, useState, useCallback } from 'react';

const ActiveHighlightContext = createContext(null);

export function ActiveHighlightProvider({ children }) {
  const [activeTab, setActiveTab] = useState('reader');
  const [selectedIssueId, setSelectedIssueId] = useState(null);
  const [selectedPageNumber, setSelectedPageNumber] = useState(1);
  const [selectedArticleId, setSelectedArticleId] = useState(null);
  const [highlightedBboxes, setHighlightedBboxes] = useState([]);
  const [isPulsing, setIsPulsing] = useState(false);
  const [hoveredArticleId, setHoveredArticleId] = useState(null);

  // Persistent Selected LLM Model (Default: groq_qwen)
  const [selectedModel, setSelectedModelState] = useState(() => {
    return localStorage.getItem('newslens_selected_model') || 'groq_qwen';
  });

  const setSelectedModel = useCallback((model) => {
    setSelectedModelState(model);
    localStorage.setItem('newslens_selected_model', model);
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
    if (articleId) setSelectedArticleId(Number(articleId));
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

  const openIssueInReader = useCallback((issueId, pageNumber = 1) => {
    const pageNum = Number(pageNumber) || 1;
    if (issueId) setSelectedIssueId(Number(issueId));
    setSelectedPageNumber(pageNum);
    setSelectedArticleId(null);
    setHighlightedBboxes([]);
    setActiveTab('reader');
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
        chatMessages,
        setChatMessages,
        highlightArticle,
        openIssueInReader,
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
