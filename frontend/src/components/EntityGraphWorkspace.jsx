import React, { useState, useEffect } from 'react';
import {
  Network,
  Share2,
  Layers,
  Search,
  RefreshCw,
  ZoomIn,
  ZoomOut,
  Filter,
  Info,
  ExternalLink,
  Bot,
  Newspaper,
  User,
  Building,
  MapPin,
  Sparkles,
  ChevronRight,
  TrendingUp,
} from 'lucide-react';
import { useActiveHighlight } from '../context/ActiveHighlightContext';
import { deduplicatedFetch } from '../utils/apiDeduplicator';

export default function EntityGraphWorkspace() {
  const { highlightArticle, runAgentQuery, setActiveTab } = useActiveHighlight();

  const [searchQuery, setSearchQuery] = useState('Tata');
  const [depth, setDepth] = useState(2);
  const [minCooccurrence, setMinCooccurrence] = useState(1);
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [filterType, setFilterType] = useState('all');

  const SUGGESTED_ENTITIES = ['Tata', 'RBI', 'Adani', 'SEBI', 'Reliance', 'Supreme Court', 'NTPC'];

  const fetchGraph = async (query = searchQuery) => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(
        `/api/entities/${encodeURIComponent(query.trim())}/graph?depth=${depth}&min_cooccurrence=${minCooccurrence}&top_neighbors=15`
      );
      if (res.ok) {
        const data = await res.json();
        setGraphData(data);
        if (data.nodes && data.nodes.length > 0) {
          setSelectedNode(data.nodes[0]);
        }
      }
    } catch (err) {
      console.error('Failed to load entity graph:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGraph();
  }, [depth, minCooccurrence]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    fetchGraph();
  };

  const getNodeColorStyles = (type) => {
    const t = (type || '').toLowerCase();
    if (t.includes('person') || t.includes('per')) {
      return {
        dot: 'bg-sky-400',
        badge: 'bg-sky-500/15 text-sky-400 border-sky-500/30',
        card: 'border-sky-500/40 hover:border-sky-400',
        label: 'Person',
      };
    }
    if (t.includes('org') || t.includes('comp') || t.includes('corp')) {
      return {
        dot: 'bg-emerald-400',
        badge: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
        card: 'border-emerald-500/40 hover:border-emerald-400',
        label: 'Organization',
      };
    }
    if (t.includes('loc') || t.includes('gpe') || t.includes('city') || t.includes('state')) {
      return {
        dot: 'bg-amber-400',
        badge: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
        card: 'border-amber-500/40 hover:border-amber-400',
        label: 'Location',
      };
    }
    return {
      dot: 'bg-purple-400',
      badge: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
      card: 'border-purple-500/40 hover:border-purple-400',
      label: 'Event / Concept',
    };
  };

  const filteredNodes =
    graphData?.nodes?.filter((n) => {
      if (filterType === 'all') return true;
      const t = (n.type || '').toLowerCase();
      if (filterType === 'org') return t.includes('org') || t.includes('comp');
      if (filterType === 'person') return t.includes('person') || t.includes('per');
      if (filterType === 'location') return t.includes('loc') || t.includes('gpe');
      return true;
    }) || [];

  return (
    <div className="flex flex-col h-full min-h-0 bg-canvas text-text-primary overflow-hidden font-sans transition-colors duration-200">
      {/* Header & Controls Toolbar */}
      <header className="p-4 border-b border-border-subtle bg-surface shrink-0 flex flex-wrap items-center justify-between gap-4 shadow-xs">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-accent-visual/20 border border-accent-visual/30 text-accent-visual">
            <Network className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold font-serif text-text-primary flex items-center gap-2">
              Entity Knowledge Graph{' '}
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-accent-visual/15 text-accent-visual border border-accent-visual/30">
                GraphRAG Multi-Hop
              </span>
            </h2>
            <p className="text-xs text-text-muted">
              Interactive narrative co-occurrences & relational cross-newspaper topology
            </p>
          </div>
        </div>

        {/* Search Input */}
        <form onSubmit={handleSearchSubmit} className="flex items-center gap-2">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search focal entity (e.g. Tata, RBI)..."
              className="bg-elevated border border-border-subtle rounded-xl pl-9 pr-4 py-1.5 text-xs text-text-primary placeholder-text-muted focus:outline-none focus:border-accent-primary w-60 shadow-inner font-sans"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="px-3.5 py-1.5 bg-accent-primary hover:opacity-90 text-white rounded-xl text-xs font-semibold font-mono transition-all flex items-center gap-1.5 shadow-sm cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Explore
          </button>
        </form>

        {/* Filters & Hops */}
        <div className="flex items-center gap-3 font-mono text-xs">
          {/* Depth Hops */}
          <div className="flex items-center gap-1.5 bg-elevated border border-border-subtle rounded-xl px-2.5 py-1">
            <span className="text-text-muted">Hops:</span>
            {[1, 2, 3].map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setDepth(d)}
                className={`px-2 py-0.5 rounded font-bold transition-all cursor-pointer ${
                  depth === d
                    ? 'bg-accent-primary text-white shadow-xs'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                {d}
              </button>
            ))}
          </div>

          {/* Min Co-occurrence Filter */}
          <div className="flex items-center gap-1.5 bg-elevated border border-border-subtle rounded-xl px-2.5 py-1">
            <span className="text-text-muted">Min Co-occur:</span>
            {[1, 2, 3].map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMinCooccurrence(m)}
                className={`px-2 py-0.5 rounded font-bold transition-all cursor-pointer ${
                  minCooccurrence === m
                    ? 'bg-accent-visual text-white shadow-xs'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                {m}x
              </button>
            ))}
          </div>

          {/* Type Filter */}
          <div className="flex items-center gap-1.5 bg-elevated border border-border-subtle rounded-xl px-2.5 py-1">
            <Filter className="w-3.5 h-3.5 text-text-muted" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="bg-transparent text-text-primary focus:outline-none text-xs cursor-pointer font-sans"
            >
              <option value="all" className="bg-surface">All Types</option>
              <option value="org" className="bg-surface">Organizations</option>
              <option value="person" className="bg-surface">Persons</option>
              <option value="location" className="bg-surface">Locations</option>
            </select>
          </div>
        </div>
      </header>

      {/* Suggested Entities Ribbon */}
      <div className="px-4 py-2 border-b border-border-subtle bg-surface/50 flex items-center gap-2 overflow-x-auto text-xs font-mono">
        <span className="text-text-muted shrink-0 text-[11px]">Popular Entities:</span>
        {SUGGESTED_ENTITIES.map((ent, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => {
              setSearchQuery(ent);
              fetchGraph(ent);
            }}
            className="px-2.5 py-0.5 rounded-lg bg-elevated hover:bg-surface text-text-secondary hover:text-text-primary border border-border-subtle text-[11px] transition-all cursor-pointer shrink-0"
          >
            {ent}
          </button>
        ))}
      </div>

      {/* Main Canvas & Details Drawer */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Visual Graph Viewport */}
        <div className="flex-1 p-6 overflow-auto flex flex-col justify-between relative bg-canvas">
          {loading ? (
            <div className="flex flex-col items-center justify-center my-auto gap-3 text-text-muted font-mono text-xs">
              <RefreshCw className="w-8 h-8 text-accent-visual animate-spin" />
              <p>Synthesizing relational entity co-occurrences across broadsheet corpus...</p>
            </div>
          ) : !graphData || filteredNodes.length === 0 ? (
            <div className="text-center max-w-sm mx-auto my-auto text-text-muted">
              <Share2 className="w-12 h-12 mx-auto mb-3 opacity-40 text-accent-visual" />
              <p className="text-sm font-semibold text-text-primary font-serif">
                No Multi-Hop Entity Connections Found
              </p>
              <p className="text-xs mt-1 leading-relaxed">
                Try expanding search query, increasing depth hops to 3, or adjusting entity type filters.
              </p>
            </div>
          ) : (
            <div className="w-full h-full flex flex-col justify-between">
              {/* Graph Stats Bar & Legend */}
              <div className="flex flex-wrap items-center justify-between gap-3 bg-surface/90 border border-border-subtle px-4 py-2.5 rounded-2xl w-full backdrop-blur-md shadow-sm font-mono text-xs">
                <div className="flex items-center gap-3">
                  <span className="text-accent-primary font-bold">
                    Focal: <span className="text-text-primary">{graphData.root_entity}</span>
                  </span>
                  <span className="text-border-subtle">•</span>
                  <span className="text-text-secondary">
                    Nodes: <strong className="text-text-primary">{filteredNodes.length}</strong>
                  </span>
                  <span className="text-border-subtle">•</span>
                  <span className="text-text-secondary">
                    Edges: <strong className="text-text-primary">{graphData.edges?.length || 0}</strong>
                  </span>
                </div>

                {/* Color Legend */}
                <div className="flex items-center gap-3 text-[11px]">
                  <span className="flex items-center gap-1 text-sky-400">
                    <span className="w-2 h-2 rounded-full bg-sky-400" /> Person
                  </span>
                  <span className="flex items-center gap-1 text-emerald-400">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" /> Organization
                  </span>
                  <span className="flex items-center gap-1 text-amber-400">
                    <span className="w-2 h-2 rounded-full bg-amber-400" /> Location
                  </span>
                  <span className="flex items-center gap-1 text-purple-400">
                    <span className="w-2 h-2 rounded-full bg-purple-400" /> Event
                  </span>
                </div>
              </div>

              {/* Node Cluster Display */}
              <div className="my-auto py-8 flex flex-wrap items-center justify-center gap-3.5 max-w-4xl mx-auto">
                {filteredNodes.map((node) => {
                  const isSelected = selectedNode?.id === node.id;
                  const isRoot = node.name.toLowerCase() === graphData.root_entity?.toLowerCase();
                  const styles = getNodeColorStyles(node.type);

                  return (
                    <button
                      key={node.id}
                      type="button"
                      onClick={() => setSelectedNode(node)}
                      className={`group relative p-3 rounded-2xl border transition-all flex items-center gap-3 shadow-md cursor-pointer ${
                        isSelected
                          ? 'ring-2 ring-accent-primary scale-105 bg-elevated border-accent-primary'
                          : 'bg-surface border-border-subtle hover:border-text-secondary hover:scale-102'
                      }`}
                    >
                      <div className={`w-3.5 h-3.5 rounded-full ${styles.dot} shadow-xs shrink-0`} />
                      <div className="text-left min-w-0">
                        <div className="text-xs font-bold font-serif text-text-primary flex items-center gap-1.5 truncate">
                          {node.name}
                          {isRoot && (
                            <span className="text-[9px] font-mono px-1.5 py-0.2 rounded-full bg-accent-primary/20 text-accent-primary border border-accent-primary/40">
                              Root
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-text-muted font-mono capitalize">
                          {styles.label} • {node.article_count} mentions
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Edge Connections Feed */}
              <div className="w-full bg-surface/90 border border-border-subtle rounded-2xl p-3 backdrop-blur-md shadow-sm font-mono text-xs">
                <div className="text-[11px] font-bold text-text-muted mb-2 flex items-center gap-1.5 uppercase">
                  <Share2 className="w-3.5 h-3.5 text-accent-visual" />
                  Strongest Interconnected Story Edges (Co-occurrences)
                </div>
                <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
                  {graphData.edges?.slice(0, 8).map((edge, idx) => (
                    <div
                      key={idx}
                      className="px-3 py-1.5 bg-elevated border border-border-subtle rounded-xl text-xs flex items-center gap-2 shrink-0"
                    >
                      <span className="font-semibold text-text-primary">{edge.source_name}</span>
                      <span className="text-accent-visual font-bold text-[10px]">
                        ↔ ({edge.weight}x)
                      </span>
                      <span className="font-semibold text-text-primary">{edge.target_name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Selected Entity Dossier Drawer (Right Pane) */}
        {selectedNode && (
          <aside className="w-80 lg:w-96 border-l border-border-subtle bg-surface p-5 flex flex-col justify-between overflow-y-auto shrink-0 shadow-xl">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <span
                  className={`text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded border ${
                    getNodeColorStyles(selectedNode.type).badge
                  }`}
                >
                  {selectedNode.type || 'Entity'}
                </span>
                <span className="text-[10px] font-mono text-text-muted">
                  Hop Depth: {selectedNode.depth}
                </span>
              </div>

              <h3 className="text-xl font-bold font-serif text-text-primary mb-1">
                {selectedNode.name}
              </h3>
              <p className="text-xs text-text-muted font-mono mb-5">
                Archival Graph Entity #{selectedNode.id}
              </p>

              <div className="space-y-3 font-mono">
                {/* Salience & Frequency Metric Card */}
                <div className="p-3.5 bg-elevated border border-border-subtle rounded-xl">
                  <div className="text-[10px] text-text-muted uppercase tracking-wider mb-1 font-bold">
                    Corpus Mentions
                  </div>
                  <div className="text-xl font-extrabold text-accent-primary">
                    {selectedNode.article_count} Stories
                  </div>
                  <p className="text-[10px] text-text-secondary mt-1 font-sans">
                    Indexed across verified broadsheet newsprint editions.
                  </p>
                </div>

                {/* Top Connected Co-Occurrences */}
                <div className="p-3.5 bg-elevated border border-border-subtle rounded-xl">
                  <div className="text-[10px] text-text-muted uppercase tracking-wider mb-2 font-bold">
                    Top Connected Co-Occurrences
                  </div>
                  <div className="space-y-1.5">
                    {graphData.edges
                      ?.filter(
                        (e) =>
                          String(e.source) === String(selectedNode.id) ||
                          String(e.target) === String(selectedNode.id)
                      )
                      .slice(0, 5)
                      .map((e, idx) => {
                        const otherName =
                          String(e.source) === String(selectedNode.id)
                            ? e.target_name
                            : e.source_name;
                        return (
                          <div
                            key={idx}
                            onClick={() => {
                              const found = graphData.nodes?.find(
                                (n) => n.name.toLowerCase() === otherName.toLowerCase()
                              );
                              if (found) setSelectedNode(found);
                            }}
                            className="flex items-center justify-between text-xs text-text-secondary hover:text-accent-primary cursor-pointer p-1 rounded hover:bg-surface transition-colors"
                          >
                            <span className="font-sans font-medium truncate">{otherName}</span>
                            <span className="text-[10px] text-text-muted shrink-0 ml-2">
                              {e.weight} co-mentions
                            </span>
                          </div>
                        );
                      })}
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="space-y-2 pt-6">
              <button
                type="button"
                onClick={() =>
                  runAgentQuery(`Investigate all recent broadsheet coverage regarding ${selectedNode.name}.`)
                }
                className="w-full py-2.5 bg-accent-primary hover:opacity-90 text-white rounded-xl text-xs font-semibold font-sans transition-all flex items-center justify-center gap-1.5 shadow-sm cursor-pointer"
              >
                <Bot className="w-3.5 h-3.5" />
                <span>Investigate Entity in AI Newsroom</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setSearchQuery(selectedNode.name);
                  fetchGraph(selectedNode.name);
                }}
                className="w-full py-2 bg-elevated hover:bg-surface border border-border-subtle text-text-secondary hover:text-text-primary rounded-xl text-xs font-semibold font-mono transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Re-Center Graph on {selectedNode.name}</span>
              </button>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
