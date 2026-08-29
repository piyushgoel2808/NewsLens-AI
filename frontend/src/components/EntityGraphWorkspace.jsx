import React, { useState, useEffect } from 'react';
import { Network, Share2, Layers, Search, RefreshCw, ZoomIn, ZoomOut, Filter, Info, ExternalLink } from 'lucide-react';

export default function EntityGraphWorkspace({ onSelectArticle, onSelectNewspaper }) {
  const [searchQuery, setSearchQuery] = useState('Tata');
  const [depth, setDepth] = useState(2);
  const [minCooccurrence, setMinCooccurrence] = useState(1);
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [filterType, setFilterType] = useState('all');

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

  const getNodeColor = (type) => {
    switch (type?.toLowerCase()) {
      case 'org':
        return 'bg-blue-600 border-blue-400 text-blue-100 shadow-blue-500/20';
      case 'person':
        return 'bg-amber-600 border-amber-400 text-amber-100 shadow-amber-500/20';
      case 'location':
        return 'bg-emerald-600 border-emerald-400 text-emerald-100 shadow-emerald-500/20';
      default:
        return 'bg-purple-600 border-purple-400 text-purple-100 shadow-purple-500/20';
    }
  };

  const filteredNodes = graphData?.nodes?.filter((n) => {
    if (filterType === 'all') return true;
    return n.type?.toLowerCase() === filterType;
  }) || [];

  return (
    <div className="flex flex-col h-full bg-zinc-950 text-zinc-100 overflow-hidden">
      {/* Header & Controls */}
      <div className="p-4 border-b border-zinc-800 bg-zinc-900/60 backdrop-blur-md flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-blue-600/20 border border-blue-500/30 text-blue-400">
            <Network className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              Entity Knowledge Graph <span className="text-xs px-2 py-0.5 rounded-full bg-blue-900/60 border border-blue-700 text-blue-300">GraphRAG Multi-Hop</span>
            </h2>
            <p className="text-xs text-zinc-400">
              Interactive narrative co-occurrences & relational cross-newspaper topology
            </p>
          </div>
        </div>

        <form onSubmit={handleSearchSubmit} className="flex items-center gap-2">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search focal entity (e.g. Tata, RBI)..."
              className="bg-zinc-900 border border-zinc-700/80 rounded-xl pl-9 pr-4 py-1.5 text-sm text-zinc-100 focus:outline-none focus:border-blue-500 w-64 shadow-inner"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="px-3.5 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-sm font-medium transition-colors flex items-center gap-1.5 shadow-md shadow-blue-600/20"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Explore
          </button>
        </form>

        <div className="flex items-center gap-3">
          {/* Depth & Co-occurrence filters */}
          <div className="flex items-center gap-2 text-xs bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-1.5">
            <span className="text-zinc-400">Hops:</span>
            {[1, 2, 3].map((d) => (
              <button
                key={d}
                onClick={() => setDepth(d)}
                className={`px-2 py-0.5 rounded-md font-semibold transition-all ${
                  depth === d ? 'bg-blue-600 text-white shadow' : 'text-zinc-400 hover:text-white'
                }`}
              >
                {d}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-1.5">
            <Filter className="w-3.5 h-3.5 text-zinc-400" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="bg-transparent text-zinc-300 focus:outline-none text-xs"
            >
              <option value="all">All Types</option>
              <option value="org">Organizations</option>
              <option value="person">Persons</option>
              <option value="location">Locations</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Canvas & Details Drawer */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Visual Graph Viewport */}
        <div className="flex-1 bg-[radial-gradient(#27272a_1px,transparent_1px)] [background-size:20px_20px] p-6 overflow-auto flex flex-col items-center justify-center relative">
          {loading ? (
            <div className="flex flex-col items-center gap-3">
              <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
              <p className="text-sm text-zinc-400 font-medium">Synthesizing relational entity co-occurrences...</p>
            </div>
          ) : !graphData || filteredNodes.length === 0 ? (
            <div className="text-center max-w-sm text-zinc-500">
              <Share2 className="w-12 h-12 mx-auto mb-3 opacity-40 text-blue-400" />
              <p className="text-sm font-semibold text-zinc-400">No Multi-Hop Entity Connections Found</p>
              <p className="text-xs mt-1">Try expanding search query, adjusting depth hops, or clearing entity type filters.</p>
            </div>
          ) : (
            <div className="w-full h-full flex flex-col justify-between">
              {/* Graph Stats Bar */}
              <div className="flex items-center gap-4 bg-zinc-900/90 border border-zinc-800/80 px-4 py-2 rounded-2xl w-fit backdrop-blur-md shadow-xl">
                <span className="text-xs font-semibold text-blue-400">Focal Entity: <span className="text-white">{graphData.root_entity}</span></span>
                <span className="text-xs text-zinc-500">|</span>
                <span className="text-xs text-zinc-300">Nodes: <span className="font-bold text-white">{filteredNodes.length}</span></span>
                <span className="text-xs text-zinc-500">|</span>
                <span className="text-xs text-zinc-300">Co-Occurrence Edges: <span className="font-bold text-white">{graphData.edges?.length || 0}</span></span>
              </div>

              {/* Node Matrix / Cluster Display */}
              <div className="my-auto py-8 flex flex-wrap items-center justify-center gap-4 max-w-4xl mx-auto">
                {filteredNodes.map((node) => {
                  const isSelected = selectedNode?.id === node.id;
                  const isRoot = node.name.toLowerCase() === graphData.root_entity?.toLowerCase();
                  return (
                    <button
                      key={node.id}
                      onClick={() => setSelectedNode(node)}
                      className={`group relative p-3.5 rounded-2xl border transition-all duration-200 flex items-center gap-3 shadow-lg ${
                        isSelected
                          ? 'ring-2 ring-blue-500 scale-105 bg-zinc-800 border-zinc-600'
                          : 'hover:scale-102 bg-zinc-900/90 border-zinc-800/90 hover:border-zinc-700'
                      }`}
                    >
                      <div className={`w-3.5 h-3.5 rounded-full border shadow-sm ${getNodeColor(node.type)}`} />
                      <div className="text-left">
                        <div className="text-xs font-bold text-zinc-100 flex items-center gap-1.5">
                          {node.name}
                          {isRoot && (
                            <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-blue-900/80 border border-blue-600 text-blue-200">
                              Root
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-zinc-400 capitalize">
                          {node.type} • {node.article_count} story mentions
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Edge Connections Feed */}
              <div className="w-full bg-zinc-900/80 border border-zinc-800/80 rounded-2xl p-3 backdrop-blur-md">
                <div className="text-xs font-semibold text-zinc-400 mb-2 flex items-center gap-1.5">
                  <Share2 className="w-3.5 h-3.5 text-blue-400" />
                  Strongest Interconnected Story Edges (Co-occurrences)
                </div>
                <div className="flex gap-2 overflow-x-auto pb-1">
                  {graphData.edges?.slice(0, 8).map((edge, idx) => (
                    <div
                      key={idx}
                      className="px-3 py-1.5 bg-zinc-950/80 border border-zinc-800 rounded-xl text-xs flex items-center gap-2 shrink-0"
                    >
                      <span className="font-semibold text-zinc-200">{edge.source_name}</span>
                      <span className="text-blue-400 font-mono text-[10px]">↔ ({edge.weight}x)</span>
                      <span className="font-semibold text-zinc-200">{edge.target_name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Selected Entity Knowledge Inspector Drawer */}
        {selectedNode && (
          <div className="w-80 border-l border-zinc-800 bg-zinc-900/90 backdrop-blur-md p-5 flex flex-col justify-between overflow-y-auto">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className={`w-3 h-3 rounded-full ${getNodeColor(selectedNode.type)}`} />
                <span className="text-xs font-bold uppercase tracking-wider text-zinc-400">{selectedNode.type}</span>
              </div>
              <h3 className="text-base font-bold text-white mb-1">{selectedNode.name}</h3>
              <p className="text-xs text-zinc-400 mb-4">
                Discovered via Multi-Hop Hop Depth {selectedNode.depth}
              </p>

              <div className="space-y-3">
                <div className="p-3 bg-zinc-950/70 border border-zinc-800/80 rounded-xl">
                  <div className="text-[11px] text-zinc-400 font-semibold mb-1">Archive Frequency</div>
                  <div className="text-lg font-extrabold text-blue-400">{selectedNode.article_count} Articles</div>
                </div>

                <div className="p-3 bg-zinc-950/70 border border-zinc-800/80 rounded-xl">
                  <div className="text-[11px] text-zinc-400 font-semibold mb-1">Connected Co-Occurrences</div>
                  <div className="space-y-1.5 mt-2">
                    {graphData.edges
                      ?.filter((e) => e.source === selectedNode.id || e.target === selectedNode.id)
                      .slice(0, 5)
                      .map((e, idx) => {
                        const otherName = e.source === selectedNode.id ? e.target_name : e.source_name;
                        return (
                          <div key={idx} className="flex items-center justify-between text-xs text-zinc-300">
                            <span>{otherName}</span>
                            <span className="text-[10px] font-mono text-zinc-500">{e.weight} co-mentions</span>
                          </div>
                        );
                      })}
                  </div>
                </div>
              </div>
            </div>

            <button
              onClick={() => {
                setSearchQuery(selectedNode.name);
                fetchGraph(selectedNode.name);
              }}
              className="mt-6 w-full py-2 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/40 text-blue-300 rounded-xl text-xs font-semibold transition-colors flex items-center justify-center gap-1.5 shadow-sm"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Re-Center Graph on {selectedNode.name}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}