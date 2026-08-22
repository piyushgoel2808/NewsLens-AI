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

  const highlightArticle = useCallback((issueId, pageNumber, articleId, bboxes = []) => {
    if (issueId) setSelectedIssueId(issueId);
    if (pageNumber) setSelectedPageNumber(pageNumber);
    if (articleId) setSelectedArticleId(articleId);
    setHighlightedBboxes(bboxes);
    setIsPulsing(true);
    setActiveTab('reader');

    setTimeout(() => {
      setIsPulsing(false);
    }, 4000);
  }, []);

  const openIssueInReader = useCallback((issueId, pageNumber = 1) => {
    setSelectedIssueId(issueId);
    setSelectedPageNumber(pageNumber);
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
