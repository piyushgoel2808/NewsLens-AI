import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  GitMerge,
  Search,
  Sparkles,
  Calendar,
  Layers,
  AlertTriangle,
  ExternalLink,
  ChevronRight,
  RefreshCw,
  Clock,
  CheckCircle2,
  FileText,
  Table,
  Zap,
  Cpu,
  User,
  Building,
  MapPin,
  TrendingUp,
  ArrowRight,
} from 'lucide-react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useActiveHighlight } from '../context/ActiveHighlightContext';
import { deduplicatedFetch } from '../utils/apiDeduplicator';
import ModelPickerPopover from './ModelPickerPopover';

// ---------------------------------------------------------------------------
// Noise Filters (Adverts & Low-Quality Fragments)
// ---------------------------------------------------------------------------
const AD_FILTER_REGEX =
  /safe payments|save more with|\bemi\b|off\b|subscribe|teaser set to|pre-rel|download now|click here|classified|limited offer/i;

const isCleanSuggestion = (headline) => {
  if (!headline || typeof headline !== 'string') return false;
  if (AD_FILTER_REGEX.test(headline)) return false;
  const words = headline.trim().split(/\s+/);
  if (words.length < 4) return false;
  return true;
};

// ---------------------------------------------------------------------------
// Actor Badges Noise Filter & Entity Typing
// ---------------------------------------------------------------------------
const INVALID_ACTOR_REGEX =
  /^(pti|ani|reuters|afp|ians|rs\.?|\$|page\s*\d+|photo\s*by|advertisement|express news service|staff reporter|bureau|our correspondent)$/i;

const isValidActor = (name) => {
  if (!name || typeof name !== 'string') return false;
  const clean = name.trim();
  if (clean.length < 2) return false;
  if (INVALID_ACTOR_REGEX.test(clean)) return false;
  if (/^\d+$/.test(clean)) return false;
  return true;
};

const getActorColor = (type = '') => {
  const t = (type || '').toLowerCase();
  if (t.includes('person') || t.includes('per')) {
    return 'bg-sky-500/10 text-sky-400 border-sky-500/30';
  }
  if (t.includes('org') || t.includes('comp') || t.includes('corp')) {
    return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
  }
  if (t.includes('loc') || t.includes('gpe') || t.includes('city') || t.includes('state')) {
    return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
  }
  return 'bg-purple-500/10 text-purple-400 border-purple-500/30';
};

const PHASE_STYLES = {
  Breaking: {
    badge: 'bg-accent-primary/20 text-accent-primary border-accent-primary/40',
    dot: 'bg-accent-primary ring-accent-primary/30',
    label: 'Breaking / Inception',
  },
  Development: {
    badge: 'bg-accent-editorial/20 text-accent-editorial border-accent-editorial/40',
    dot: 'bg-accent-editorial ring-accent-editorial/30',
    label: 'Development & Escalation',
  },
  Financial: {
    badge: 'bg-accent-visual/20 text-accent-visual border-accent-visual/40',
    dot: 'bg-accent-visual ring-accent-visual/30',
    label: 'Financial & Market Impact',
  },
  Regulatory: {
    badge: 'bg-accent-info/20 text-accent-info border-accent-info/40',
    dot: 'bg-accent-info ring-accent-info/30',
    label: 'Regulatory & Outcome',
  },
};

const getPublicationStyle = (newspaperName = '') => {
  const name = newspaperName.toLowerCase();
  if (name.includes('mint')) {
    return {
      card: 'bg-surface border-border-subtle hover:border-accent-primary/60',
      badge: 'bg-accent-primary/15 text-accent-primary border-accent-primary/30',
      accent: 'text-accent-primary',
    };
  }
  if (name.includes('business standard')) {
    return {
      card: 'bg-surface border-border-subtle hover:border-accent-info/60',
      badge: 'bg-accent-info/15 text-accent-info border-accent-info/30',
      accent: 'text-accent-info',
    };
  }
  if (name.includes('hindu')) {
    return {
      card: 'bg-surface border-border-subtle hover:border-accent-visual/60',
      badge: 'bg-accent-visual/15 text-accent-visual border-accent-visual/30',
      accent: 'text-accent-visual',
    };
  }
  return {
    card: 'bg-surface border-border-subtle hover:border-accent-editorial/60',
    badge: 'bg-accent-editorial/15 text-accent-editorial border-accent-editorial/30',
    accent: 'text-accent-editorial',
  };
};

export default function TimelineWorkspace() {
  const {
    timelineQuery,
    setTimelineQuery,
    highlightArticle,
    selectedModel,
    setSelectedModel,
  } = useActiveHighlight();

  const [query, setQuery] = useState(timelineQuery || '');
  const [suggestions, setSuggestions] = useState([]);
  const [viewMode, setViewMode] = useState('spine'); // 'spine' | 'swimlane'
  const [loading, setLoading] = useState(false);
  const [progressStage, setProgressStage] = useState('');
  const [trajectoryData, setTrajectoryData] = useState(null);
  const [error, setError] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);

  useEffect(() => {
    deduplicatedFetch('/api/models/available')
      .then((data) => {
        if (data && data.models) {
          setAvailableModels(data.models);
        }
      })
      .catch((err) => console.error('Failed to load models in Timeline:', err));

    deduplicatedFetch('/api/query/timeline/suggestions')
      .then((data) => {
        if (data && data.suggestions && Array.isArray(data.suggestions)) {
          // Apply strict noise filter to suggestions
          const clean = data.suggestions.filter(isCleanSuggestion);
          setSuggestions(clean);
        }
      })
      .catch((err) => console.error('Failed to load timeline suggestions:', err));
  }, []);

  // Sync external timelineQuery from context
  useEffect(() => {
    if (timelineQuery && timelineQuery !== query) {
      setQuery(timelineQuery);
      fetchTrajectory(timelineQuery);
    }
  }, [timelineQuery]);

  // Initial load if query is provided
  useEffect(() => {
    if (timelineQuery && !trajectoryData && !loading) {
      fetchTrajectory(timelineQuery);
    }
  }, []);

  const fetchTrajectory = async (targetQuery) => {
    if (!targetQuery || !targetQuery.trim()) return;
    setLoading(true);
    setError(null);
    setProgressStage('Initializing narrative synthesis...');

    try {
      const response = await fetch('/api/query/timeline/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: targetQuery.trim(),
          model: selectedModel,
          model_override: selectedModel,
          use_cache: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const block of lines) {
          if (!block.trim()) continue;
          const matchEvent = block.match(/^event:\s*(\w+)/m);
          const matchData = block.match(/^data:\s*(.+)$/m);
          const eventType = matchEvent ? matchEvent[1] : 'message';

          if (matchData) {
            try {
              const data = JSON.parse(matchData[1]);
              if (eventType === 'stage') {
                if (data.message) setProgressStage(data.message);
                else if (data.stage === 'fetching_articles')
                  setProgressStage('Retrieving broadsheets across editions...');
                else if (data.stage === 'clustering_dates')
                  setProgressStage('Clustering coverage by calendar date...');
                else if (data.stage === 'synthesizing_perspectives')
                  setProgressStage('Synthesizing perspectives & detecting discrepancies...');
              } else if (eventType === 'result') {
                setTrajectoryData(data);
              } else if (eventType === 'error') {
                throw new Error(data.error || 'Failed to synthesize timeline');
              }
            } catch (err) {
              console.error('Failed to parse SSE payload', err);
            }
          }
        }
      }
    } catch (err) {
      console.error('Trajectory stream error', err);
      // Fallback to standard REST endpoint
      try {
        setProgressStage('Querying trajectory engine...');
        const restRes = await fetch('/api/query/timeline', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: targetQuery.trim(),
            model: selectedModel,
            model_override: selectedModel,
            use_cache: true,
          }),
        });
        if (restRes.ok) {
          const data = await restRes.json();
          setTrajectoryData(data);
        } else {
          setError(err.message || 'Failed to load timeline');
        }
      } catch (fallbackErr) {
        setError(fallbackErr.message || 'Failed to load timeline');
      }
    } finally {
      setLoading(false);
      setProgressStage('');
    }
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (setTimelineQuery) setTimelineQuery(query);
    fetchTrajectory(query);
  };

  // Virtualization for large lists of milestones
  const parentRef = useRef(null);
  const milestones = trajectoryData?.milestones || [];
  const virtualizer = useVirtualizer({
    count: milestones.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 240,
    overscan: 5,
  });

  // Extract unique publications for swimlane
  const allPublications = useMemo(() => {
    const pubSet = new Set();
    milestones.forEach((m) => {
      (m.perspectives || []).forEach((p) => {
        if (p.newspaper_name) pubSet.add(p.newspaper_name);
      });
    });
    return Array.from(pubSet);
  }, [milestones]);

  const totalDiscrepancies = useMemo(() => {
    return milestones.reduce((acc, m) => acc + (m.discrepancies?.length || 0), 0);
  }, [milestones]);

  // Determine if we should show the enlarged center hero search or top header search
  const hasResultsOrLoading = Boolean(trajectoryData || loading);

  return (
    <div className="flex flex-col h-full min-h-0 bg-canvas text-text-primary overflow-hidden font-sans transition-colors duration-200">
      {/* Top Header Toolbar */}
      <header className="px-6 py-3.5 bg-surface border-b border-border-subtle shrink-0 space-y-3 z-10 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          {/* Query Form (Top Bar visible if results or loading exist) */}
          {hasResultsOrLoading ? (
            <form onSubmit={handleSearchSubmit} className="flex-1 flex items-center gap-2 max-w-2xl">
              <div className="relative flex-1">
                <Search className="w-4 h-4 text-text-muted absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Trace cross-newspaper storyline (e.g. Tata Power clean energy deal)..."
                  className="w-full bg-elevated border border-border-subtle rounded-xl pl-10 pr-4 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-accent-primary"
                />
              </div>
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="flex items-center gap-1.5 bg-accent-primary hover:opacity-90 disabled:opacity-50 text-white font-semibold px-4 py-2 rounded-xl text-sm transition-all shadow-md cursor-pointer"
              >
                {loading ? (
                  <RefreshCw className="w-4 h-4 animate-spin text-white" />
                ) : (
                  <Zap className="w-4 h-4" />
                )}
                <span>Reconstruct</span>
              </button>
            </form>
          ) : (
            <div className="flex items-center gap-2 text-xs font-mono text-text-muted">
              <GitMerge className="w-4 h-4 text-accent-primary" />
              <span className="font-bold text-text-primary uppercase">Storyline Trajectory Canvas</span>
              <span>· Chronological Multi-Issue Clustering</span>
            </div>
          )}

          {/* View Switcher & Model Selector */}
          <div className="flex items-center gap-2.5">
            <ModelPickerPopover value={selectedModel} onChange={setSelectedModel} compact={false} />

            <div className="flex items-center bg-elevated p-1 rounded-xl border border-border-subtle font-mono text-xs">
              <button
                type="button"
                onClick={() => setViewMode('spine')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
                  viewMode === 'spine'
                    ? 'bg-accent-primary text-white shadow-xs'
                    : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                <GitMerge className="w-3.5 h-3.5" />
                <span>Chronological Spine</span>
              </button>
              <button
                type="button"
                onClick={() => setViewMode('swimlane')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
                  viewMode === 'swimlane'
                    ? 'bg-accent-primary text-white shadow-xs'
                    : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                <Table className="w-3.5 h-3.5" />
                <span>Multi-Track Matrix</span>
              </button>
            </div>
          </div>
        </div>

        {/* Live SSE Progress Banner */}
        {loading && (
          <div className="flex items-center gap-2.5 bg-accent-primary/10 border border-accent-primary/30 px-3.5 py-2 rounded-xl text-xs text-accent-primary animate-pulse font-mono">
            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
            <span className="font-semibold">{progressStage || 'Reconstructing storyline...'}</span>
          </div>
        )}
      </header>

      {/* Main Workspace Body */}
      <div className="flex-1 overflow-hidden flex flex-col p-6 space-y-4">
        {/* Error Alert */}
        {error && (
          <div className="bg-rose-500/10 border border-rose-500/30 text-rose-400 p-4 rounded-xl flex items-center gap-3 text-sm font-mono">
            <AlertTriangle className="w-5 h-5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* CENTER ENLARGED SEARCH (when no results yet, smooth transition) */}
        {!hasResultsOrLoading && (
          <div className="flex-1 flex flex-col items-center justify-center max-w-3xl mx-auto w-full px-4 text-center my-auto">
            <div className="w-14 h-14 rounded-2xl bg-surface border border-border-subtle flex items-center justify-center text-accent-primary mb-4 shadow-xl">
              <GitMerge className="w-7 h-7" />
            </div>

            <h2 className="text-2xl sm:text-3xl font-serif font-bold text-text-primary mb-2">
              Investigative Storyline Trajectory
            </h2>
            <p className="text-sm text-text-secondary max-w-lg mb-8 leading-relaxed">
              Trace developing stories across multi-week broadsheet editions, isolate editorial divergence,
              and detect reporting contradictions on a unified chronological spine.
            </p>

            {/* Enlarged Center Search Box */}
            <form onSubmit={handleSearchSubmit} className="w-full flex items-center gap-2.5 mb-8">
              <div className="relative flex-1">
                <Search className="w-5 h-5 text-text-muted absolute left-4 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter a storyline prompt (e.g. Tata Power clean energy expansion)..."
                  className="w-full bg-surface border-2 border-border-subtle focus:border-accent-primary rounded-2xl pl-12 pr-4 py-3.5 text-base text-text-primary placeholder-text-muted outline-none shadow-lg transition-all"
                  autoFocus
                />
              </div>
              <button
                type="submit"
                disabled={!query.trim()}
                className="bg-accent-primary hover:opacity-90 disabled:opacity-40 text-white font-semibold px-6 py-3.5 rounded-2xl text-sm flex items-center gap-2 transition-all shadow-lg shadow-emerald-950/30 cursor-pointer"
              >
                <span>Reconstruct</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </form>

            {/* Filtered Suggestion Chips */}
            {suggestions.length > 0 && (
              <div className="w-full text-left">
                <span className="text-[11px] font-mono uppercase tracking-wider text-text-muted block mb-2.5 text-center">
                  Recommended Investigative Trajectories:
                </span>
                <div className="flex flex-wrap justify-center gap-2">
                  {suggestions.slice(0, 6).map((sq, sIdx) => (
                    <button
                      key={sIdx}
                      type="button"
                      onClick={() => {
                        setQuery(sq);
                        fetchTrajectory(sq);
                      }}
                      className="bg-surface hover:bg-elevated text-text-secondary hover:text-text-primary px-3 py-1.5 rounded-xl border border-border-subtle text-xs transition-all shadow-xs cursor-pointer truncate max-w-md"
                    >
                      "{sq}"
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Summary Telemetry Bar (when results exist) */}
        {trajectoryData && (
          <div className="bg-surface border border-border-subtle p-4 rounded-2xl flex flex-wrap items-center justify-between gap-4 shadow-sm">
            <div className="space-y-1 max-w-3xl">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-accent-primary" />
                <h2 className="text-sm font-bold text-text-primary font-serif">
                  {trajectoryData.query}
                </h2>
                {trajectoryData.cached && (
                  <span className="bg-accent-info/20 text-accent-info text-[10px] font-mono px-2 py-0.5 rounded border border-accent-info/30">
                    ⚡ Redis Cache (0ms)
                  </span>
                )}
              </div>
              <p className="text-xs text-text-secondary leading-relaxed font-sans">
                {trajectoryData.topic_summary}
              </p>
            </div>

            <div className="flex items-center gap-3 text-xs font-mono text-text-muted">
              {trajectoryData.date_range?.length >= 2 && (
                <div className="flex items-center gap-1.5 bg-elevated px-2.5 py-1 rounded-lg border border-border-subtle">
                  <Calendar className="w-3.5 h-3.5 text-text-muted" />
                  <span>
                    {trajectoryData.date_range[0]} → {trajectoryData.date_range[1]}
                  </span>
                </div>
              )}
              <div className="flex items-center gap-1.5 bg-elevated px-2.5 py-1 rounded-lg border border-border-subtle">
                <Layers className="w-3.5 h-3.5 text-accent-primary" />
                <span>{milestones.length} Milestones</span>
              </div>
              {totalDiscrepancies > 0 && (
                <div className="flex items-center gap-1.5 bg-accent-editorial/15 text-accent-editorial px-2.5 py-1 rounded-lg border border-accent-editorial/30">
                  <AlertTriangle className="w-3.5 h-3.5 text-accent-editorial" />
                  <span>{totalDiscrepancies} Discrepancies</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* View Mode 1: Vertical Chronological Spine */}
        {hasResultsOrLoading && viewMode === 'spine' && (
          <div
            ref={parentRef}
            className="flex-1 overflow-y-auto pr-2 relative"
            style={{ height: '100%' }}
          >
            {milestones.length === 0 && !loading ? (
              <div className="flex flex-col items-center justify-center h-64 text-text-muted space-y-2 font-mono text-xs">
                <GitMerge className="w-8 h-8 opacity-40" />
                <p>No narrative milestones found for this query.</p>
              </div>
            ) : (
              <div
                className="relative"
                style={{
                  height: `${virtualizer.getTotalSize()}px`,
                  width: '100%',
                }}
              >
                {/* Continuous Vertical Left Spine Line */}
                <div className="absolute left-6 top-4 bottom-4 w-0.5 bg-gradient-to-b from-accent-primary via-accent-info to-border-subtle pointer-events-none" />

                {virtualizer.getVirtualItems().map((virtualRow) => {
                  const m = milestones[virtualRow.index];
                  const phaseInfo = PHASE_STYLES[m.event_phase] || PHASE_STYLES.Development;

                  // Clean Key Actors using the NER filter
                  const validActors = (m.active_entities || []).filter((ent) =>
                    isValidActor(ent?.name || ent)
                  );

                  return (
                    <div
                      key={m.milestone_id || virtualRow.index}
                      data-index={virtualRow.index}
                      ref={virtualizer.measureElement}
                      className="absolute top-0 left-0 w-full pl-16 pb-8"
                      style={{
                        transform: `translateY(${virtualRow.start}px)`,
                      }}
                    >
                      {/* Node Bullet on Spine */}
                      <div
                        className={`absolute left-[19px] top-4 w-3.5 h-3.5 rounded-full border-2 border-canvas ring-4 ${phaseInfo.dot} transition-transform hover:scale-125 z-10`}
                      />

                      {/* Milestone Container Card */}
                      <div className="bg-surface border border-border-subtle hover:border-text-secondary rounded-2xl p-5 shadow-lg transition-all space-y-4">
                        {/* Header: Date + Phase + Canonical Event */}
                        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border-subtle pb-3 font-mono">
                          <div className="flex items-center gap-3">
                            <span className="text-sm font-bold text-text-primary flex items-center gap-1.5">
                              <Calendar className="w-4 h-4 text-accent-primary" />
                              {m.date}
                            </span>
                            <span
                              className={`text-[11px] font-semibold uppercase tracking-wider px-2.5 py-0.5 rounded-md border ${phaseInfo.badge}`}
                            >
                              {phaseInfo.label}
                            </span>
                          </div>
                        </div>

                        {/* Canonical Event Description */}
                        <div>
                          <p className="text-sm font-semibold font-serif text-text-primary leading-snug">
                            {m.canonical_event}
                          </p>
                        </div>

                        {/* Filtered Key Actors Badges with Entity Color Coding */}
                        {validActors.length > 0 && (
                          <div className="flex flex-wrap items-center gap-1.5 pt-1">
                            <span className="text-[10px] uppercase font-bold text-text-muted mr-1 font-mono flex items-center gap-1">
                              <span className="w-1.5 h-1.5 rounded-full bg-accent-primary animate-pulse" />
                              Key Actors:
                            </span>
                            {validActors.map((ent, entIdx) => {
                              const entName = ent?.name || ent;
                              const entType = ent?.type || 'Entity';
                              const colorStyle = getActorColor(entType);

                              return (
                                <span
                                  key={entIdx}
                                  className={`px-2 py-0.5 rounded-md border text-[10px] font-mono font-medium flex items-center gap-1 transition-colors ${colorStyle}`}
                                >
                                  <span>{entName}</span>
                                  <span className="text-[9px] opacity-70 capitalize">({entType})</span>
                                </span>
                              );
                            })}
                          </div>
                        )}

                        {/* Discrepancy Alert Banner */}
                        {m.discrepancies && m.discrepancies.length > 0 && (
                          <div className="bg-accent-editorial/15 border border-accent-editorial/30 rounded-xl p-3 text-xs text-accent-editorial space-y-1 font-mono">
                            <div className="flex items-center gap-1.5 font-bold uppercase text-[10px] tracking-wider">
                              <AlertTriangle className="w-3.5 h-3.5" />
                              <span>Reporting Contradiction / Discrepancy:</span>
                            </div>
                            <ul className="list-disc list-inside space-y-0.5 pl-1 font-sans text-text-secondary">
                              {m.discrepancies.map((disc, dIdx) => (
                                <li key={dIdx}>{disc}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Multi-Perspective Cards Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                          {(m.perspectives || []).map((p, pIdx) => {
                            const pubStyle = getPublicationStyle(p.newspaper_name);

                            return (
                              <div
                                key={pIdx}
                                className={`rounded-xl border p-3.5 flex flex-col justify-between space-y-3 transition-all ${pubStyle.card}`}
                              >
                                <div className="space-y-2">
                                  {/* Publication Badge & Editorial Angle */}
                                  <div className="flex items-center justify-between gap-1 font-mono">
                                    <span
                                      className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${pubStyle.badge}`}
                                    >
                                      {p.newspaper_name}
                                    </span>
                                    <span className="text-[10px] text-text-muted bg-elevated px-1.5 py-0.5 rounded border border-border-subtle truncate max-w-[120px]">
                                      {p.angle}
                                    </span>
                                  </div>

                                  {/* Headline */}
                                  <h4 className="text-xs font-bold font-serif text-text-primary line-clamp-2 leading-snug">
                                    "{p.headline}"
                                  </h4>

                                  {/* Key Takeaway */}
                                  <p className="text-xs text-text-secondary line-clamp-3 leading-relaxed font-sans">
                                    {p.key_takeaway}
                                  </p>
                                </div>

                                {/* Deep Link to Broadsheet Page & Canvas Pulse */}
                                <button
                                  type="button"
                                  onClick={() =>
                                    highlightArticle(
                                      p.issue_id,
                                      p.pdf_page || 1,
                                      p.article_id,
                                      p.bboxes || []
                                    )
                                  }
                                  className="flex items-center justify-between w-full pt-2 border-t border-border-subtle text-[11px] font-mono text-text-muted hover:text-accent-primary transition-colors cursor-pointer group"
                                >
                                  <span className="flex items-center gap-1">
                                    <FileText className="w-3 h-3 text-text-muted group-hover:text-accent-primary" />
                                    <span>Read Article in Broadsheet (p.{p.pdf_page || 1})</span>
                                  </span>
                                  <ExternalLink className="w-3 h-3 opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 transition-transform" />
                                </button>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* View Mode 2: Multi-Track Swimlane Matrix View */}
        {hasResultsOrLoading && viewMode === 'swimlane' && (
          <div className="flex-1 overflow-auto rounded-xl border border-border-subtle bg-surface shadow-md">
            <table className="min-w-full text-xs text-left border-collapse">
              <thead className="bg-elevated text-text-primary uppercase tracking-wider font-mono border-b border-border-subtle sticky top-0 z-10">
                <tr>
                  <th className="p-3.5 border-r border-border-subtle w-48">Date & Phase</th>
                  <th className="p-3.5 border-r border-border-subtle w-72">Canonical Milestone</th>
                  {allPublications.map((pub, idx) => (
                    <th key={idx} className="p-3.5 border-r border-border-subtle last:border-r-0 min-w-[260px]">
                      <span className="font-bold">{pub}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle">
                {milestones.map((m, mIdx) => {
                  const phaseInfo = PHASE_STYLES[m.event_phase] || PHASE_STYLES.Development;

                  return (
                    <tr key={mIdx} className="hover:bg-elevated/40 transition-colors">
                      {/* Date & Phase Cell */}
                      <td className="p-3.5 border-r border-border-subtle align-top space-y-1 font-mono">
                        <div className="font-bold text-text-primary">{m.date}</div>
                        <span className={`inline-block text-[10px] px-2 py-0.5 rounded border ${phaseInfo.badge}`}>
                          {m.event_phase}
                        </span>
                      </td>

                      {/* Canonical Milestone Summary Cell */}
                      <td className="p-3.5 border-r border-border-subtle align-top space-y-2">
                        <p className="font-semibold font-serif text-text-primary leading-snug">
                          {m.canonical_event}
                        </p>
                        {m.discrepancies && m.discrepancies.length > 0 && (
                          <div className="text-[10px] text-accent-editorial font-mono">
                            ⚠ {m.discrepancies.length} reporting contradiction(s) noted
                          </div>
                        )}
                      </td>

                      {/* Publication Track Cells */}
                      {allPublications.map((pubName, pIdx) => {
                        const matchedPerspective = (m.perspectives || []).find(
                          (p) => p.newspaper_name?.toLowerCase() === pubName.toLowerCase()
                        );

                        if (!matchedPerspective) {
                          return (
                            <td
                              key={pIdx}
                              className="p-3.5 border-r border-border-subtle last:border-r-0 align-middle text-center text-text-muted italic bg-canvas/30 text-[11px]"
                            >
                              No verified coverage
                            </td>
                          );
                        }

                        return (
                          <td
                            key={pIdx}
                            className="p-3.5 border-r border-border-subtle last:border-r-0 align-top space-y-2"
                          >
                            <div className="flex items-center justify-between text-[10px] font-mono">
                              <span className="font-semibold text-accent-primary">
                                Page {matchedPerspective.pdf_page || 1}
                              </span>
                              <span className="text-text-muted">{matchedPerspective.angle}</span>
                            </div>
                            <h4 className="font-serif font-bold text-text-primary text-xs line-clamp-2">
                              "{matchedPerspective.headline}"
                            </h4>
                            <p className="text-text-secondary text-[11px] leading-relaxed line-clamp-3 font-sans">
                              {matchedPerspective.key_takeaway}
                            </p>
                            <button
                              type="button"
                              onClick={() =>
                                highlightArticle(
                                  matchedPerspective.issue_id,
                                  matchedPerspective.pdf_page || 1,
                                  matchedPerspective.article_id,
                                  matchedPerspective.bboxes || []
                                )
                              }
                              className="text-[10px] font-mono text-accent-primary hover:underline flex items-center gap-1 cursor-pointer"
                            >
                              <span>Read Article in Broadsheet &rarr;</span>
                            </button>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
