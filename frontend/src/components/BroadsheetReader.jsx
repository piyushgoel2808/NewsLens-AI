import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Calendar,
  Newspaper,
  Tag,
  Users,
  Sparkles,
  ExternalLink,
  BookOpen,
  Info,
  Search,
  Filter,
} from 'lucide-react';
import CanvasOverlay from './CanvasOverlay';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

export default function BroadsheetReader() {
  const {
    selectedIssueId,
    setSelectedIssueId,
    selectedPageNumber,
    setSelectedPageNumber,
    selectedArticleId,
    setSelectedArticleId,
    highlightedBboxes,
    isPulsing,
    hoveredArticleId,
    setHoveredArticleId,
    highlightArticle,
  } = useActiveHighlight();

  const [issues, setIssues] = useState([]);
  const [issueData, setIssueData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [articleDetails, setArticleDetails] = useState(null);
  const [articleLoading, setArticleLoading] = useState(false);

  // Canvas Viewport Controls
  const [zoom, setZoom] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSection, setSelectedSection] = useState('ALL');

  const canvasContainerRef = useRef(null);
  const articleListRef = useRef(null);

  // 1. Fetch Issues Catalog
  useEffect(() => {
    fetch('/api/issues?limit=50')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const list = Array.isArray(data) ? data : [];
        setIssues(list);
        if (list.length > 0 && !selectedIssueId) {
          setSelectedIssueId(list[0].id);
        }
      })
      .catch((err) => {
        console.error('Failed to load issues:', err);
        setIssues([]);
      });
  }, [selectedIssueId, setSelectedIssueId]);

  // 2. Fetch Issue Inspection Data (pages, articles, extraction modes)
  useEffect(() => {
    if (!selectedIssueId) return;
    setLoading(true);
    fetch(`/api/issues/${selectedIssueId}/inspection?chunk_limit=100`)
      .then((res) => {
        if (!res.ok) throw new Error('Issue not found');
        return res.json();
      })
      .then((data) => {
        if (data && !data.error && Array.isArray(data.pages)) {
          setIssueData(data);
        } else {
          setIssueData(null);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load issue inspection:', err);
        setIssueData(null);
        setLoading(false);
      });
  }, [selectedIssueId]);

  // 3. Fetch Full Article Details when an article is selected
  useEffect(() => {
    if (!selectedArticleId) {
      setArticleDetails(null);
      return;
    }
    setArticleLoading(true);
    fetch(`/api/articles/${selectedArticleId}`)
      .then((res) => {
        if (!res.ok) throw new Error('Article details failed');
        return res.json();
      })
      .then((data) => {
        if (data && !data.error) {
          setArticleDetails(data);
        } else {
          setArticleDetails(null);
        }
        setArticleLoading(false);
      })
      .catch((err) => {
        console.error('Failed to fetch article details:', err);
        setArticleDetails(null);
        setArticleLoading(false);
      });
  }, [selectedArticleId]);

  // Current Active Page
  const currentPage = useMemo(() => {
    if (!issueData || !Array.isArray(issueData.pages) || issueData.pages.length === 0) return null;
    return (
      issueData.pages.find((p) => p.page_number === selectedPageNumber) ||
      issueData.pages[0] ||
      null
    );
  }, [issueData, selectedPageNumber]);

  // Articles on the current page with bounding boxes
  const pageArticles = useMemo(() => {
    if (!issueData || !Array.isArray(issueData.articles) || !currentPage) return [];
    return issueData.articles
      .filter((art) => {
        const pages = Array.isArray(art.pages) ? art.pages : [];
        return pages.includes(currentPage.page_number);
      })
      .map((art) => {
        const pageBboxes =
          (art.page_bboxes && art.page_bboxes[currentPage.page_number]) ||
          (art.bboxes_by_page && art.bboxes_by_page[currentPage.page_number]) ||
          (Array.isArray(art.bboxes) ? art.bboxes : []);
        return {
          ...art,
          bboxes: pageBboxes,
        };
      });
  }, [issueData, currentPage]);

  // Sections on the current page for filtering
  const availableSections = useMemo(() => {
    const set = new Set();
    pageArticles.forEach((a) => {
      if (a.section) set.add(a.section);
    });
    return ['ALL', ...Array.from(set)];
  }, [pageArticles]);

  // Filtered articles
  const filteredArticles = useMemo(() => {
    return pageArticles.filter((art) => {
      const matchSection =
        selectedSection === 'ALL' || art.section === selectedSection;
      const matchSearch =
        !searchTerm ||
        (art.headline && art.headline.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (art.full_text_preview &&
          art.full_text_preview.toLowerCase().includes(searchTerm.toLowerCase()));
      return matchSection && matchSearch;
    });
  }, [pageArticles, selectedSection, searchTerm]);

  // Zoom handlers
  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.5));
  const handleResetZoom = () => setZoom(1);

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 overflow-hidden">
      {/* Top Header & Reader Toolbar */}
      <header className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-slate-900 border-b border-slate-800 gap-3 z-20">
        {/* Issue Selector & Publication Info */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700">
            <Newspaper className="w-4 h-4 text-emerald-400" />
            <select
              value={selectedIssueId || ''}
              onChange={(e) => {
                setSelectedIssueId(Number(e.target.value));
                setSelectedPageNumber(1);
                setSelectedArticleId(null);
              }}
              className="bg-transparent text-sm font-semibold text-slate-100 outline-none cursor-pointer"
            >
              {Array.isArray(issues) && issues.length > 0 ? (
                issues.map((iss) => (
                  <option key={iss.id} value={iss.id} className="bg-slate-900 text-slate-200">
                    {iss.newspaper_name} — {iss.issue_date} (Issue #{iss.id})
                  </option>
                ))
              ) : (
                <option value="" className="bg-slate-900 text-slate-400">
                  No issues archived yet
                </option>
              )}
            </select>
          </div>

          {issueData?.issue && (
            <div className="hidden md:flex items-center gap-2 text-xs text-slate-400">
              <span className="flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5 text-slate-400" />
                {issueData.issue.issue_date}
              </span>
              <span>•</span>
              <span className="bg-slate-800 px-2 py-0.5 rounded text-emerald-400 font-medium">
                {issueData.issue.edition || 'Main Edition'}
              </span>
              <span>•</span>
              <span>{issueData.issue.total_pages} Pages</span>
            </div>
          )}
        </div>

        {/* Page Switcher & Canvas Zoom Controls */}
        <div className="flex items-center gap-3">
          {/* Page Carousel / Stepper */}
          <div className="flex items-center gap-1 bg-slate-800/80 px-2 py-1 rounded-lg border border-slate-700">
            <button
              onClick={() => setSelectedPageNumber((p) => Math.max(p - 1, 1))}
              disabled={selectedPageNumber <= 1}
              className="p-1 hover:bg-slate-700 rounded disabled:opacity-30 disabled:hover:bg-transparent"
              title="Previous Page"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-xs font-semibold px-2">
              Page {selectedPageNumber}
              {currentPage?.printed_page_number && (
                <span className="text-slate-400 font-normal"> (Folio {currentPage.printed_page_number})</span>
              )}
              <span className="text-slate-500 font-normal"> / {issueData?.issue?.total_pages || 1}</span>
            </span>
            <button
              onClick={() =>
                setSelectedPageNumber((p) =>
                  Math.min(p + 1, issueData?.issue?.total_pages || p)
                )
              }
              disabled={selectedPageNumber >= (issueData?.issue?.total_pages || 1)}
              className="p-1 hover:bg-slate-700 rounded disabled:opacity-30 disabled:hover:bg-transparent"
              title="Next Page"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* Zoom Controls */}
          <div className="flex items-center gap-1 bg-slate-800/80 px-2 py-1 rounded-lg border border-slate-700">
            <button
              onClick={handleZoomOut}
              className="p-1 hover:bg-slate-700 rounded text-slate-300"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="text-xs font-mono w-10 text-center text-slate-300">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={handleZoomIn}
              className="p-1 hover:bg-slate-700 rounded text-slate-300"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-1 hover:bg-slate-700 rounded text-slate-300 ml-1 border-l border-slate-700 pl-1.5"
              title="Reset View"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Toggle Spatial Overlays */}
          <button
            onClick={() => setShowOverlays((prev) => !prev)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
              showOverlays
                ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Overlays</span>
          </button>
        </div>
      </header>

      {/* Main Content Area (Split Screen) */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden relative">
        {/* LEFT PANE: Broadsheet Canvas & Raster Viewer */}
        <section
          ref={canvasContainerRef}
          className="flex-1 bg-slate-950/80 overflow-auto p-4 flex items-start justify-center relative select-none"
        >
          {loading ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-3">
              <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm font-medium">Loading high-resolution broadsheet...</p>
            </div>
          ) : currentPage ? (
            <div
              className="relative shadow-2xl rounded-sm transition-transform duration-100 ease-out origin-top"
              style={{
                transform: `scale(${zoom})`,
                width: '100%',
                maxWidth: '900px',
              }}
            >
              {/* Raster Image from MinIO Proxy */}
              <img
                src={currentPage.image_url}
                alt={`Broadsheet Page ${currentPage.page_number}`}
                className="w-full h-auto block rounded-sm pointer-events-none"
                loading="eager"
              />

              {/* Spatial SVG Overlay */}
              {showOverlays && (
                <CanvasOverlay
                  pageWidth={currentPage.width_px || 2800}
                  pageHeight={currentPage.height_px || 4399}
                  articles={pageArticles}
                  selectedArticleId={selectedArticleId}
                  hoveredArticleId={hoveredArticleId}
                  pulsingArticleId={isPulsing ? selectedArticleId : null}
                  customHighlightBboxes={highlightedBboxes}
                  onSelectArticle={(artId) => setSelectedArticleId(artId)}
                  onHoverArticle={(artId) => setHoveredArticleId(artId)}
                />
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-slate-500">
              <p className="text-sm">No page available for this issue.</p>
            </div>
          )}
        </section>

        {/* RIGHT PANE: Article Explorer & Inspector */}
        <aside className="w-full lg:w-[480px] xl:w-[540px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-slate-800 flex flex-col h-full z-10">
          {/* Article Inspector View when an article is selected */}
          {selectedArticleId && articleDetails ? (
            <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-900">
              {/* Article Header & Navigation Back */}
              <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/90 backdrop-blur-sm sticky top-0 z-10">
                <button
                  onClick={() => setSelectedArticleId(null)}
                  className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-100 font-medium px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 transition-colors"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  Back to Page Articles
                </button>
                <div className="flex items-center gap-2">
                  {articleDetails.prominence_score !== undefined && (
                    <span
                      className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                      title="Page Prominence Score"
                    >
                      Prominence: {Math.round(articleDetails.prominence_score * 100)}%
                    </span>
                  )}
                  <span className="text-xs uppercase font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                    {articleDetails.section || articleDetails.article_type || 'News'}
                  </span>
                </div>
              </div>

              {/* Article Content Body (Scrollable) */}
              <div className="flex-1 overflow-y-auto p-5 space-y-4">
                {articleDetails.subheadline && (
                  <p className="text-xs uppercase tracking-widest text-emerald-400 font-semibold font-mono">
                    {articleDetails.subheadline}
                  </p>
                )}

                <h1 className="text-xl md:text-2xl font-bold font-serif leading-tight text-slate-50">
                  {articleDetails.headline || 'Untitled Article'}
                </h1>

                {/* Author & Issue Info */}
                <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400 border-b border-slate-800 pb-3">
                  {articleDetails.byline_author && (
                    <span className="text-slate-200 font-medium">By {articleDetails.byline_author}</span>
                  )}
                  <span>•</span>
                  <span>{articleDetails.word_count} words</span>
                  <span>•</span>
                  <span>
                    Spans Page {articleDetails.pages?.map((p) => p.page_number).join(', ')}
                  </span>
                </div>

                {/* AI Generated Summary Pill */}
                {articleDetails.summary && (
                  <div className="bg-slate-800/60 border border-slate-700/80 rounded-lg p-3 text-xs leading-relaxed text-slate-300">
                    <div className="flex items-center gap-1.5 text-emerald-400 font-semibold mb-1">
                      <Sparkles className="w-3.5 h-3.5" />
                      Executive Summary
                    </div>
                    {articleDetails.summary}
                  </div>
                )}

                {/* Full Article Text */}
                <div className="prose prose-invert prose-sm max-w-none text-slate-300 leading-relaxed font-sans whitespace-pre-wrap">
                  {articleDetails.full_text}
                </div>

                {/* Page Jump / Continuation Links */}
                {articleDetails.pages && articleDetails.pages.length > 1 && (
                  <div className="bg-slate-800/80 border border-slate-700 p-3 rounded-lg flex items-center justify-between text-xs mt-4">
                    <span className="text-slate-300">Article spans multiple pages:</span>
                    <div className="flex gap-1.5">
                      {articleDetails.pages.map((p) => (
                        <button
                          key={p.page_number}
                          onClick={() => setSelectedPageNumber(p.page_number)}
                          className={`px-2.5 py-1 rounded text-xs font-semibold transition-colors ${
                            selectedPageNumber === p.page_number
                              ? 'bg-emerald-500 text-white'
                              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                          }`}
                        >
                          Page {p.page_number}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Article Chunks & Vector Decomposition Breakdown */}
                {articleDetails.chunks && articleDetails.chunks.length > 0 && (
                  <div className="mt-6 border-t border-slate-800 pt-4">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-1.5">
                        <Sparkles className="w-4 h-4 text-emerald-400" />
                        Indexed RAG Chunks ({articleDetails.chunks.length})
                      </h3>
                      <span className="text-[11px] text-slate-400 font-mono">
                        Vector DB: Qdrant
                      </span>
                    </div>

                    <div className="space-y-2.5">
                      {articleDetails.chunks.map((chk) => (
                        <div
                          key={chk.id || chk.chunk_index}
                          className="bg-slate-950/80 border border-slate-800 hover:border-emerald-500/60 rounded-lg p-3 transition-colors text-xs"
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="font-mono text-[10px] uppercase font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/50">
                              Chunk #{chk.chunk_index + 1}
                            </span>
                            <span className="text-[10px] text-slate-500 font-mono">
                              {chk.token_count || '~'} tokens
                            </span>
                          </div>
                          <p className="text-slate-300 font-mono text-[11px] leading-relaxed whitespace-pre-wrap bg-slate-900/60 p-2.5 rounded border border-slate-800/80">
                            {chk.text}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Extracted Photos Section */}
                {articleDetails.photos && articleDetails.photos.length > 0 && (
                  <div className="mt-6 border-t border-slate-800 pt-4">
                    <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-1.5">
                      <Tag className="w-4 h-4 text-purple-400" />
                      Associated Photos & Graphics ({articleDetails.photos.length})
                    </h3>
                    <div className="grid grid-cols-1 gap-3">
                      {articleDetails.photos.map((ph, idx) => (
                        <div key={ph.id || idx} className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs overflow-hidden flex flex-col gap-2.5">
                          {ph.image_url && (
                            <div className="bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center max-h-64 border border-slate-800 relative group">
                              <img
                                src={ph.image_url}
                                alt={ph.caption || `Photo #${ph.id}`}
                                className="w-full h-auto object-contain max-h-64 rounded-lg"
                                loading="lazy"
                                onError={(e) => {
                                  e.currentTarget.style.display = 'none';
                                }}
                              />
                            </div>
                          )}
                          <div className="flex items-center justify-between gap-2 flex-wrap">
                            <div className="flex items-center gap-1.5">
                              <span className="text-[10px] text-purple-400 font-mono font-bold bg-purple-950/60 px-2 py-0.5 rounded border border-purple-800/40">
                                Asset #{ph.id}
                              </span>
                              {ph.visual_type && (
                                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-blue-950/80 text-blue-300 border border-blue-800/50">
                                  {ph.visual_type === 'data_chart' ? '📊 Data Chart' :
                                   ph.visual_type === 'table' ? '🔢 Table' :
                                   ph.visual_type === 'infographic' ? '📈 Infographic' :
                                   ph.visual_type === 'logo' ? '🏷️ Logo' : '📷 Editorial Photo'}
                                </span>
                              )}
                            </div>
                            {ph.bbox && (
                              <button
                                onClick={() => {
                                  if (ph.bbox) {
                                    highlightArticle(
                                      selectedIssueId,
                                      selectedPageNumber,
                                      articleDetails?.id,
                                      [ph.bbox]
                                    );
                                  }
                                }}
                                className="text-[10px] text-emerald-400 hover:text-emerald-300 font-medium flex items-center gap-1 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/40 transition-colors"
                              >
                                <Maximize2 className="w-3 h-3" />
                                Highlight on Page
                              </button>
                            )}
                          </div>
                          {ph.caption && (
                            <div className="bg-slate-900/60 p-2 rounded border border-slate-800/60">
                              <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5">Caption:</span>
                              <p className="text-slate-300 text-xs italic leading-relaxed">{ph.caption}</p>
                            </div>
                          )}
                          {ph.vlm_description && (
                            <div className="p-2.5 bg-slate-900/80 border border-blue-900/40 rounded-lg text-xs text-slate-300 font-sans whitespace-pre-wrap max-h-48 overflow-y-auto leading-relaxed">
                              <span className="text-[10px] uppercase tracking-wider font-bold text-blue-400 flex items-center gap-1 mb-1">
                                <Sparkles className="w-3 h-3" />
                                AI Visual Intelligence:
                              </span>
                              {ph.vlm_description}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Extracted Tables Section */}
                {articleDetails.tables && articleDetails.tables.length > 0 && (
                  <div className="mt-6 border-t border-slate-800 pt-4">
                    <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-1.5">
                      <Layers className="w-4 h-4 text-blue-400" />
                      Extracted Data Tables ({articleDetails.tables.length})
                    </h3>
                    {articleDetails.tables.map((tb, idx) => (
                      <div key={tb.id || idx} className="bg-slate-950 border border-slate-800 rounded p-3 text-xs overflow-x-auto">
                        <pre className="font-mono text-slate-300">
                          {JSON.stringify(tb.extracted_json, null, 2)}
                        </pre>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Page Article List View */
            <div className="flex-1 flex flex-col h-full overflow-hidden">
              {/* Filter & Search Header */}
              <div className="p-3 border-b border-slate-800 space-y-2 bg-slate-900">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <BookOpen className="w-3.5 h-3.5 text-emerald-400" />
                    Page {selectedPageNumber} Articles ({filteredArticles.length})
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <div className="relative flex-1">
                    <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-500" />
                    <input
                      type="text"
                      placeholder="Filter articles..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-8 pr-3 py-1.5 bg-slate-950 border border-slate-800 rounded-md text-xs text-slate-200 placeholder-slate-500 outline-none focus:border-emerald-500"
                    />
                  </div>

                  {availableSections.length > 1 && (
                    <select
                      value={selectedSection}
                      onChange={(e) => setSelectedSection(e.target.value)}
                      className="bg-slate-950 border border-slate-800 rounded-md px-2 py-1.5 text-xs text-slate-300 outline-none cursor-pointer"
                    >
                      {availableSections.map((sec) => (
                        <option key={sec} value={sec}>
                          {sec}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
              </div>

              {/* Scrollable Article Cards */}
              <div ref={articleListRef} className="flex-1 overflow-y-auto p-3 space-y-2.5">
                {filteredArticles.length === 0 ? (
                  <div className="text-center py-10 text-slate-500 text-xs">
                    No articles found matching filters on this page.
                  </div>
                ) : (
                  filteredArticles.map((art) => {
                    const isSelected = selectedArticleId === art.id;
                    const isHovered = hoveredArticleId === art.id;

                    return (
                      <div
                        key={art.id}
                        onClick={() => setSelectedArticleId(art.id)}
                        onMouseEnter={() => setHoveredArticleId(art.id)}
                        onMouseLeave={() => setHoveredArticleId(null)}
                        className={`p-3.5 rounded-lg border cursor-pointer transition-all duration-150 ${
                          isSelected
                            ? 'bg-slate-800/90 border-emerald-500 shadow-md ring-1 ring-emerald-500'
                            : isHovered
                            ? 'bg-slate-800/60 border-slate-700'
                            : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1.5 flex-wrap gap-1">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span
                              className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded font-semibold ${
                                art.article_type === 'advertisement'
                                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                                  : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                              }`}
                            >
                              {art.section || art.article_type || 'News'}
                            </span>

                            {/* Visual Badges for Photos & Infographics */}
                            {art.has_infographic && (
                              <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-cyan-950/90 text-cyan-300 border border-cyan-800/60 flex items-center gap-1 shadow-sm">
                                📊 Infographic
                              </span>
                            )}
                            {art.photo_count > 0 && !art.has_infographic && (
                              <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-purple-950/90 text-purple-300 border border-purple-800/60 flex items-center gap-1 shadow-sm">
                                📷 {art.photo_count > 1 ? `${art.photo_count} Photos` : 'Photo'}
                              </span>
                            )}
                            {art.table_count > 0 && !art.has_infographic && (
                              <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-blue-950/90 text-blue-300 border border-blue-800/60 flex items-center gap-1 shadow-sm">
                                🔢 Table
                              </span>
                            )}
                          </div>
                          <span className="text-[11px] text-slate-400">
                            {art.word_count} words
                          </span>
                        </div>

                        <h3 className="text-sm font-bold font-serif leading-snug text-slate-100 mb-1.5 line-clamp-2">
                          {art.headline || 'Untitled Article'}
                        </h3>

                        {art.full_text_preview && (
                          <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed font-sans">
                            {art.full_text_preview}
                          </p>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
