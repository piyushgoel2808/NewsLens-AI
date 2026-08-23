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
} from 'lucide-react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

const SAMPLE_QUERIES = [
  'Indo-Pacific bilateral trade negotiations',
  'Tata Power clean energy expansion in Odisha',
  'Telecom AGR dues Supreme Court dispute',
  'Semiconductor fabrication incentives & foundry rollout',
];

const PHASE_STYLES = {
  Breaking: {
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    dot: 'bg-emerald-400 ring-emerald-500/30',
    label: 'Breaking / Inception',
  },
  Development: {
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
    dot: 'bg-amber-400 ring-amber-500/30',
    label: 'Development & Escalation',
  },
  Financial: {
    badge: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
    dot: 'bg-purple-400 ring-purple-500/30',
    label: 'Financial & Market Impact',
  },
  Regulatory: {
    badge: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
    dot: 'bg-cyan-400 ring-cyan-500/30',
    label: 'Regulatory & Outcome',
  },
};

const getPublicationStyle = (newspaperName = '') => {
  const name = newspaperName.toLowerCase();
  if (name.includes('mint')) {
    return {
      card: 'bg-emerald-950/20 border-emerald-500/30 hover:border-emerald-500/60',
      badge: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
      accent: 'text-emerald-400',
    };
  }
  if (name.includes('business standard')) {
    return {
      card: 'bg-blue-950/20 border-blue-500/30 hover:border-blue-500/60',
      badge: 'bg-blue-500/15 text-blue-300 border-blue-500/30',
      accent: 'text-blue-400',
    };
  }
  if (name.includes('hindu')) {
    return {
      card: 'bg-purple-950/20 border-purple-500/30 hover:border-purple-500/60',
      badge: 'bg-purple-500/15 text-purple-300 border-purple-500/30',
      accent: 'text-purple-400',
    };
  }
  return {
    card: 'bg-slate-900/60 border-slate-700/60 hover:border-cyan-500/50',
    badge: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30',
    accent: 'text-cyan-400',
  };
};

export default function TimelineWorkspace() {
  const { timelineQuery, setTimelineQuery, highlightArticle, selectedModel } =
    useActiveHighlight();

  const [query, setQuery] = useState(timelineQuery || 'Tata Power clean energy expansion');
  const [viewMode, setViewMode] = useState('spine'); // 'spine' | 'swimlane'
  const [loading, setLoading] = useState(false);
  const [progressStage, setProgressStage] = useState('');
  const [trajectoryData, setTrajectoryData] = useState(null);
  const [error, setError] = useState(null);

  // Sync external timelineQuery from context (e.g. from chat button)
  useEffect(() => {
    if (timelineQuery && timelineQuery !== query) {
      setQuery(timelineQuery);
      fetchTrajectory(timelineQuery);
    }
  }, [timelineQuery]);

  // Initial load
  useEffect(() => {
    if (!trajectoryData && !loading) {
      fetchTrajectory(query);
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

        for (const line of lines) {
          if (!line.trim()) continue;
          const matchEvent = line.match(/^event:\s*(\w+)/m);
          const matchData = line.match(/^data:\s*(.+)$/m);
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

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 overflow-hidden">
      {/* Top Controls Header */}
      <header className="px-6 py-4 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 shrink-0 space-y-3 z-10">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          {/* Query Form */}
          <form onSubmit={handleSearchSubmit} className="flex-1 flex items-center gap-2 max-w-2xl">
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Trace cross-newspaper storyline (e.g. Tata Power nuclear deal)..."
                className="w-full bg-slate-950/90 border border-slate-700/80 rounded-xl pl-10 pr-4 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500/80 focus:ring-1 focus:ring-emerald-500/80"
              />
            </div>
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="flex items-center gap-1.5 bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 text-slate-950 font-semibold px-4 py-2 rounded-xl text-sm transition-all shadow-md shadow-emerald-950/40"
            >
              {loading ? (
                <RefreshCw className="w-4 h-4 animate-spin text-slate-950" />
              ) : (
                <Zap className="w-4 h-4" />
              )}
              <span>Reconstruct</span>
            </button>
          </form>

          {/* View Switcher & Actions */}
          <div className="flex items-center gap-2">
            <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800">
              <button
                onClick={() => setViewMode('spine')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  viewMode === 'spine'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <GitMerge className="w-3.5 h-3.5" />
                <span>Chronological Spine</span>
              </button>
              <button
                onClick={() => setViewMode('swimlane')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  viewMode === 'swimlane'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Table className="w-3.5 h-3.5" />
                <span>Multi-Track Matrix</span>
              </button>
            </div>
          </div>
        </div>

        {/* Quick Sample Query Suggestions */}
        <div className="flex items-center gap-2 overflow-x-auto pb-1 text-xs text-slate-400">
          <span className="shrink-0 text-slate-500 font-medium">Suggested Topics:</span>
          {SAMPLE_QUERIES.map((sq, sIdx) => (
            <button
              key={sIdx}
              onClick={() => {
                setQuery(sq);
                fetchTrajectory(sq);
              }}
              className="shrink-0 bg-slate-800/80 hover:bg-slate-800 text-slate-300 hover:text-emerald-300 px-2.5 py-1 rounded-md border border-slate-700/60 transition-colors"
            >
              {sq}
            </button>
          ))}
        </div>

        {/* Live SSE Progress Banner */}
        {loading && (
          <div className="flex items-center gap-2.5 bg-emerald-950/40 border border-emerald-500/30 px-3.5 py-2 rounded-lg text-xs text-emerald-300 animate-pulse">
            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
            <span className="font-medium">{progressStage || 'Reconstructing storyline...'}</span>
          </div>
        )}
      </header>

      {/* Main Workspace Body */}
      <div className="flex-1 overflow-hidden flex flex-col p-6 space-y-4">
        {/* Error Alert */}
        {error && (
          <div className="bg-rose-950/40 border border-rose-500/30 text-rose-300 p-4 rounded-xl flex items-center gap-3 text-sm">
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Summary Telemetry Bar */}
        {trajectoryData && (
          <div className="bg-slate-900/70 border border-slate-800 p-4 rounded-xl flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-1 max-w-3xl">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-bold text-slate-100">
                  {trajectoryData.query}
                </h2>
                {trajectoryData.cached && (
                  <span className="bg-cyan-500/20 text-cyan-300 text-[10px] font-mono px-2 py-0.5 rounded border border-cyan-500/40">
                    ⚡ Cached (0ms)
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                {trajectoryData.topic_summary}
              </p>
            </div>

            <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
              {trajectoryData.date_range?.length >= 2 && (
                <div className="flex items-center gap-1.5 bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800">
                  <Calendar className="w-3.5 h-3.5 text-slate-500" />
                  <span>
                    {trajectoryData.date_range[0]} → {trajectoryData.date_range[1]}
                  </span>
                </div>
              )}
              <div className="flex items-center gap-1.5 bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800">
                <Layers className="w-3.5 h-3.5 text-emerald-400" />
                <span>{milestones.length} Milestones</span>
              </div>
              {totalDiscrepancies > 0 && (
                <div className="flex items-center gap-1.5 bg-amber-950/40 text-amber-300 px-2.5 py-1 rounded-md border border-amber-500/40">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                  <span>{totalDiscrepancies} Discrepancies</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* View Mode 1: Vertical Chronological Spine */}
        {viewMode === 'spine' && (
          <div
            ref={parentRef}
            className="flex-1 overflow-y-auto pr-2 relative"
            style={{ height: '100%' }}
          >
            {milestones.length === 0 && !loading ? (
              <div className="flex flex-col items-center justify-center h-64 text-slate-500 space-y-2">
                <GitMerge className="w-8 h-8 text-slate-600" />
                <p className="text-sm">No narrative milestones found for this query.</p>
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
                <div className="absolute left-6 top-4 bottom-4 w-0.5 bg-gradient-to-b from-emerald-500/60 via-blue-500/40 to-slate-800 pointer-events-none" />

                {virtualizer.getVirtualItems().map((virtualRow) => {
                  const m = milestones[virtualRow.index];
                  const phaseInfo = PHASE_STYLES[m.event_phase] || PHASE_STYLES.Development;

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
                        className={`absolute left-[19px] top-4 w-3.5 h-3.5 rounded-full border-2 border-slate-950 ring-4 ${phaseInfo.dot} transition-transform hover:scale-125 z-10`}
                      />

                      {/* Milestone Container Card */}
                      <div className="bg-slate-900/80 border border-slate-800 hover:border-slate-700/90 rounded-2xl p-5 shadow-xl transition-all space-y-4">
                        {/* Header: Date + Phase + Canonical Event */}
                        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
                          <div className="flex items-center gap-3">
                            <span className="text-sm font-bold font-mono text-slate-100 flex items-center gap-1.5">
                              <Calendar className="w-4 h-4 text-emerald-400" />
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
                          <p className="text-sm font-semibold text-slate-100 leading-snug">
                            {m.canonical_event}
                          </p>
                        </div>

                        {/* Discrepancy Alert Banner */}
                        {m.discrepancies && m.discrepancies.length > 0 && (
                          <div className="bg-amber-950/30 border border-amber-500/40 rounded-xl p-3 text-xs text-amber-200 space-y-1">
                            <div className="flex items-center gap-1.5 font-bold uppercase text-[10px] tracking-wider text-amber-400">
                              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                              <span>Reporting Contradiction / Discrepancy:</span>
                            </div>
                            <ul className="list-disc list-inside space-y-0.5 text-amber-200/90 pl-1">
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
                                  <div className="flex items-center justify-between gap-1">
                                    <span
                                      className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${pubStyle.badge}`}
                                    >
                                      {p.newspaper_name}
                                    </span>
                                    <span className="text-[10px] font-mono text-slate-400 bg-slate-950/60 px-1.5 py-0.5 rounded border border-slate-800">
                                      {p.angle}
                                    </span>
                                  </div>

                                  {/* Headline */}
                                  <h4 className="text-xs font-bold text-slate-100 line-clamp-2 leading-snug">
                                    "{p.headline}"
                                  </h4>

                                  {/* Key Takeaway */}
                                  <p className="text-xs text-slate-300/90 line-clamp-3 leading-relaxed">
                                    {p.key_takeaway}
                                  </p>
                                </div>

                                {/* Deep Link to Broadsheet Page & Canvas Pulse */}
                                <button
                                  onClick={() =>
                                    highlightArticle(
                                      p.issue_id,
                                      p.pdf_page || 1,
                                      p.article_id,
                                      p.bboxes || []
                                    )
                                  }
                                  className="flex items-center justify-between w-full pt-2 border-t border-slate-800/80 text-[11px] font-medium text-slate-400 hover:text-emerald-300 transition-colors group"
                                >
                                  <span className="flex items-center gap-1">
                                    <FileText className="w-3 h-3 text-slate-400 group-hover:text-emerald-400" />
                                    <span>Verify on Broadsheet (Page {p.pdf_page || 1})</span>
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
        {viewMode === 'swimlane' && (
          <div className="flex-1 overflow-auto rounded-xl border border-slate-800 bg-slate-900/60">
            <table className="min-w-full text-xs text-left border-collapse">
              <thead className="bg-slate-900 text-slate-200 uppercase tracking-wider font-mono border-b border-slate-800 sticky top-0 z-10">
                <tr>
                  <th className="p-3.5 border-r border-slate-800 w-48">Date & Phase</th>
                  <th className="p-3.5 border-r border-slate-800 w-72">Canonical Milestone</th>
                  {allPublications.map((pub, idx) => (
                    <th key={idx} className="p-3.5 border-r border-slate-800 last:border-r-0 min-w-[260px]">
                      <span className="font-bold text-slate-100">{pub}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80">
                {milestones.map((m, mIdx) => {
                  const phaseInfo = PHASE_STYLES[m.event_phase] || PHASE_STYLES.Development;

                  return (
                    <tr key={mIdx} className="hover:bg-slate-800/30 transition-colors">
                      {/* Date & Phase Cell */}
                      <td className="p-3.5 border-r border-slate-800 align-top space-y-1 font-mono">
                        <div className="font-bold text-slate-100">{m.date}</div>
                        <span className={`inline-block text-[10px] px-2 py-0.5 rounded border ${phaseInfo.badge}`}>
                          {m.event_phase}
                        </span>
                      </td>

                      {/* Canonical Milestone Summary Cell */}
                      <td className="p-3.5 border-r border-slate-800 align-top space-y-2">
                        <p className="font-semibold text-slate-200 leading-snug">
                          {m.canonical_event}
                        </p>
                        {m.discrepancies && m.discrepancies.length > 0 && (
                          <div className="bg-amber-950/30 border border-amber-500/40 p-2 rounded text-[11px] text-amber-200">
                            <span className="font-bold block mb-0.5 text-amber-400">⚠️ Discrepancy:</span>
                            {m.discrepancies[0]}
                          </div>
                        )}
                      </td>

                      {/* Publication Columns */}
                      {allPublications.map((pub, pIdx) => {
                        const perspective = (m.perspectives || []).find(
                          (p) => p.newspaper_name.toLowerCase() === pub.toLowerCase()
                        );

                        if (!perspective) {
                          return (
                            <td
                              key={pIdx}
                              className="p-3.5 border-r border-slate-800 last:border-r-0 align-top text-slate-600 italic text-[11px]"
                            >
                              No direct coverage on this date
                            </td>
                          );
                        }

                        const pubStyle = getPublicationStyle(pub);

                        return (
                          <td
                            key={pIdx}
                            className="p-3.5 border-r border-slate-800 last:border-r-0 align-top space-y-2"
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] font-mono text-slate-400 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                                {perspective.angle}
                              </span>
                              <span className="text-[10px] text-slate-400 font-mono">
                                Page {perspective.pdf_page || 1}
                              </span>
                            </div>
                            <h5 className="font-bold text-slate-100 text-xs line-clamp-2">
                              "{perspective.headline}"
                            </h5>
                            <p className="text-slate-300 text-[11px] line-clamp-3 leading-relaxed">
                              {perspective.key_takeaway}
                            </p>
                            <button
                              onClick={() =>
                                highlightArticle(
                                  perspective.issue_id,
                                  perspective.pdf_page || 1,
                                  perspective.article_id,
                                  perspective.bboxes || []
                                )
                              }
                              className="inline-flex items-center gap-1 text-[10px] text-emerald-400 hover:text-emerald-300 font-medium"
                            >
                              <span>View Broadsheet Scan</span>
                              <ExternalLink className="w-2.5 h-2.5" />
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
