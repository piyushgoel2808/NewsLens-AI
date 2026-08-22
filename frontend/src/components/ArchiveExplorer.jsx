import React, { useState, useEffect, useMemo } from 'react';
import {
  Newspaper,
  Calendar,
  Layers,
  BookOpen,
  Search,
  Filter,
  CheckCircle2,
  Clock,
  AlertTriangle,
  Trash2,
  ExternalLink,
  ChevronRight,
  Database,
  Tag,
  BarChart3,
} from 'lucide-react';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

export default function ArchiveExplorer() {
  const { openIssueInReader, setActiveTab } = useActiveHighlight();

  const [newspapers, setNewspapers] = useState([]);
  const [issues, setIssues] = useState([]);
  const [loading, setLoading] = useState(true);

  const [selectedNewspaperId, setSelectedNewspaperId] = useState('ALL');
  const [selectedStatus, setSelectedStatus] = useState('ALL');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  // 1. Fetch Newspapers & Issues
  const fetchData = async () => {
    setLoading(true);
    try {
      const [newsRes, issuesRes] = await Promise.all([
        fetch('/api/newspapers'),
        fetch('/api/issues?limit=100'),
      ]);

      const newsData = await newsRes.json();
      const issuesData = await issuesRes.json();

      setNewspapers(newsData || []);
      setIssues(issuesData || []);
    } catch (err) {
      console.error('Failed to load archive data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Handle Delete Issue
  const handleDeleteIssue = async (issueId, e) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to permanently delete Issue #${issueId}? This will remove all vectors, database records, and MinIO files.`)) {
      return;
    }

    try {
      const res = await fetch(`/api/issues/${issueId}`, { method: 'DELETE' });
      if (res.ok) {
        setIssues((prev) => prev.filter((i) => i.id !== issueId));
      } else {
        alert('Failed to delete issue.');
      }
    } catch (err) {
      console.error('Error deleting issue:', err);
    }
  };

  // Filtered Issues
  const filteredIssues = useMemo(() => {
    return issues.filter((iss) => {
      const matchNews =
        selectedNewspaperId === 'ALL' || iss.newspaper_id === Number(selectedNewspaperId);
      const matchStatus =
        selectedStatus === 'ALL' || iss.ingestion_status === selectedStatus;
      const matchDateFrom = !dateFrom || iss.issue_date >= dateFrom;
      const matchDateTo = !dateTo || iss.issue_date <= dateTo;
      const matchSearch =
        !searchQuery ||
        (iss.newspaper_name && iss.newspaper_name.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (iss.edition && iss.edition.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (iss.issue_date && iss.issue_date.includes(searchQuery));

      return matchNews && matchStatus && matchDateFrom && matchDateTo && matchSearch;
    });
  }, [issues, selectedNewspaperId, selectedStatus, dateFrom, dateTo, searchQuery]);

  // Aggregate Metrics
  const metrics = useMemo(() => {
    const totalArticles = issues.reduce((acc, i) => acc + (i.article_count || 0), 0);
    const totalPages = issues.reduce((acc, i) => acc + (i.total_pages || 0), 0);
    return {
      newspapersCount: newspapers.length,
      issuesCount: issues.length,
      articlesCount: totalArticles,
      pagesCount: totalPages,
    };
  }, [newspapers, issues]);

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 max-w-7xl mx-auto p-4 md:p-6 overflow-y-auto">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold font-serif text-slate-100 flex items-center gap-2.5">
            <Newspaper className="w-6 h-6 text-emerald-400" />
            Newspaper Archive & Corpus Navigator
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Browse, inspect, and analyze historical broadsheet issues, structured extractions, and OCR layers.
          </p>
        </div>

        <button
          onClick={() => setActiveTab('ingest')}
          className="self-start md:self-auto bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-4 py-2 rounded-lg flex items-center gap-2 transition-colors shadow-md"
        >
          <Layers className="w-4 h-4" />
          <span>Upload New Issue</span>
        </button>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-6">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <Newspaper className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xl font-bold text-slate-100">{metrics.newspapersCount}</span>
            <p className="text-[11px] text-slate-400 uppercase tracking-wider">Publications</p>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
            <Calendar className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xl font-bold text-slate-100">{metrics.issuesCount}</span>
            <p className="text-[11px] text-slate-400 uppercase tracking-wider">Total Issues</p>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xl font-bold text-slate-100">{metrics.articlesCount}</span>
            <p className="text-[11px] text-slate-400 uppercase tracking-wider">Indexed Articles</p>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xl font-bold text-slate-100">{metrics.totalPages}</span>
            <p className="text-[11px] text-slate-400 uppercase tracking-wider">Digitized Pages</p>
          </div>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 mb-6 space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          {/* Publication Tabs */}
          <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setSelectedNewspaperId('ALL')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                selectedNewspaperId === 'ALL'
                  ? 'bg-emerald-500 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All Publications
            </button>
            {newspapers.map((n) => (
              <button
                key={n.id}
                onClick={() => setSelectedNewspaperId(n.id)}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  selectedNewspaperId === n.id
                    ? 'bg-emerald-500 text-white'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {n.name}
              </button>
            ))}
          </div>

          {/* Search Box */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
            <input
              type="text"
              placeholder="Search issues by date, edition..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 outline-none focus:border-emerald-500"
            />
          </div>

          {/* Status Filter */}
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300 outline-none cursor-pointer"
          >
            <option value="ALL">All Ingestion Statuses</option>
            <option value="COMPLETED">Completed</option>
            <option value="PROCESSING">Processing</option>
            <option value="FAILED">Failed</option>
          </select>
        </div>
      </div>

      {/* Issues Grid */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 text-slate-400 gap-3">
          <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm font-medium">Loading historical archives...</p>
        </div>
      ) : filteredIssues.length === 0 ? (
        <div className="text-center py-20 text-slate-500 bg-slate-900/50 border border-slate-800 rounded-xl">
          <Newspaper className="w-10 h-10 mx-auto mb-2 opacity-40" />
          <p className="text-sm font-medium">No newspaper issues found matching your filters.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredIssues.map((iss) => (
            <div
              key={iss.id}
              onClick={() => openIssueInReader(iss.id, 1)}
              className="bg-slate-900 border border-slate-800 hover:border-emerald-500/60 rounded-xl p-4.5 cursor-pointer transition-all duration-200 hover:shadow-xl hover:bg-slate-800/80 group flex flex-col justify-between"
            >
              <div>
                {/* Header Row */}
                <div className="flex items-center justify-between mb-2.5">
                  <span className="font-bold text-xs uppercase tracking-wider text-emerald-400 font-mono">
                    {iss.newspaper_name || 'Daily Broadsheet'}
                  </span>
                  <span
                    className={`text-[10px] font-semibold px-2 py-0.5 rounded-full flex items-center gap-1 ${
                      iss.ingestion_status === 'COMPLETED'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        : iss.ingestion_status === 'PROCESSING'
                        ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                        : 'bg-red-500/10 text-red-400 border border-red-500/20'
                    }`}
                  >
                    {iss.ingestion_status === 'COMPLETED' && <CheckCircle2 className="w-3 h-3" />}
                    {iss.ingestion_status === 'PROCESSING' && <Clock className="w-3 h-3 animate-spin" />}
                    {iss.ingestion_status === 'FAILED' && <AlertTriangle className="w-3 h-3" />}
                    {iss.ingestion_status}
                  </span>
                </div>

                {/* Date & Edition */}
                <h3 className="text-lg font-bold font-serif text-slate-100 group-hover:text-emerald-300 transition-colors mb-1">
                  {iss.issue_date}
                </h3>
                <p className="text-xs text-slate-400 mb-3">
                  {iss.edition || 'Main Broadsheet Edition'} • {iss.language || 'eng'}
                </p>

                {/* Badges Info */}
                <div className="flex items-center gap-3 text-xs text-slate-400 pt-2 border-t border-slate-800/80">
                  <span>{iss.total_pages || 0} Pages</span>
                  <span>•</span>
                  <span>{iss.article_count || 0} Articles</span>
                  <span>•</span>
                  <span>{iss.chunk_count || 0} Chunks</span>
                </div>
              </div>

              {/* Action Buttons Footer */}
              <div className="flex items-center justify-between mt-4 pt-3 border-t border-slate-800/80 text-xs">
                <span className="text-emerald-400 group-hover:translate-x-1 transition-transform flex items-center gap-1 font-semibold">
                  Read Issue <ChevronRight className="w-3.5 h-3.5" />
                </span>

                <button
                  onClick={(e) => handleDeleteIssue(iss.id, e)}
                  className="text-slate-500 hover:text-red-400 p-1 rounded hover:bg-slate-800 transition-colors"
                  title="Delete issue and purge vectors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
