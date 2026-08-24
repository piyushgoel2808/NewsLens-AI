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
  Edit2,
  Building2,
  Plus,
  X,
  RefreshCw,
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

  // Edit Issue State
  const [editingIssue, setEditingIssue] = useState(null);
  const [editDate, setEditDate] = useState('');
  const [editEdition, setEditEdition] = useState('');
  const [editNewspaperId, setEditNewspaperId] = useState('');
  const [isSavingEdit, setIsSavingEdit] = useState(false);

  // Manage Newspapers Modal
  const [showManageModal, setShowManageModal] = useState(false);
  const [newPubName, setNewPubName] = useState('');
  const [newPubPublisher, setNewPubPublisher] = useState('');
  const [newPubLang, setNewPubLang] = useState('en');
  const [newPubCountry, setNewPubCountry] = useState('IN');
  const [isCreatingPub, setIsCreatingPub] = useState(false);

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

      setNewspapers(Array.isArray(newsData) ? newsData : []);
      setIssues(Array.isArray(issuesData) ? issuesData : []);
    } catch (err) {
      console.error('Failed to load archive data:', err);
      setNewspapers([]);
      setIssues([]);
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
    if (
      !window.confirm(
        `Are you sure you want to permanently delete Issue #${issueId}? This will remove all vectors, database records, and MinIO files.`
      )
    ) {
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

  // Open Edit Modal
  const handleOpenEdit = (iss, e) => {
    e.stopPropagation();
    setEditingIssue(iss);
    setEditDate(iss.issue_date || '');
    setEditEdition(iss.edition || 'morning');
    setEditNewspaperId(iss.newspaper_id || '');
  };

  // Save Issue Edit (Cascades date to Qdrant vectors and DB)
  const handleSaveIssueEdit = async (e) => {
    e.preventDefault();
    if (!editingIssue || !editDate) return;

    setIsSavingEdit(true);
    try {
      const res = await fetch(`/api/issues/${editingIssue.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          issue_date: editDate,
          edition: editEdition,
          newspaper_id: editNewspaperId ? Number(editNewspaperId) : undefined,
        }),
      });

      if (res.ok) {
        const updated = await res.json();
        setIssues((prev) =>
          prev.map((item) =>
            item.id === editingIssue.id
              ? {
                  ...item,
                  issue_date: updated.issue_date,
                  edition: updated.edition,
                  newspaper_id: updated.newspaper_id,
                  newspaper_name: updated.newspaper_name,
                }
              : item
          )
        );
        setEditingIssue(null);
      } else {
        const err = await res.json();
        alert(err.detail || 'Failed to update issue metadata.');
      }
    } catch (err) {
      console.error('Error updating issue:', err);
    } finally {
      setIsSavingEdit(false);
    }
  };

  // Create Newspaper
  const handleCreateNewspaper = async (e) => {
    e.preventDefault();
    if (!newPubName.trim()) return;

    setIsCreatingPub(true);
    try {
      const res = await fetch('/api/newspapers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newPubName.trim(),
          publisher: newPubPublisher.trim() || null,
          default_language: newPubLang,
          country: newPubCountry,
        }),
      });
      if (res.ok) {
        const created = await res.json();
        setNewspapers((prev) => [...prev, created]);
        setNewPubName('');
        setNewPubPublisher('');
        setShowManageModal(false);
      } else {
        const err = await res.json();
        alert(err.detail || 'Failed to create newspaper.');
      }
    } catch (err) {
      console.error('Error creating newspaper:', err);
    } finally {
      setIsCreatingPub(false);
    }
  };

  // Delete Newspaper
  const handleDeleteNewspaper = async (id, name) => {
    if (
      !window.confirm(
        `Are you sure you want to delete "${name}"? This will delete the publication and ALL associated issues and vector embeddings!`
      )
    ) {
      return;
    }

    try {
      const res = await fetch(`/api/newspapers/${id}`, { method: 'DELETE' });
      if (res.ok) {
        setNewspapers((prev) => prev.filter((n) => n.id !== id));
        setIssues((prev) => prev.filter((i) => i.newspaper_id !== id));
      } else {
        alert('Failed to delete newspaper.');
      }
    } catch (err) {
      console.error('Error deleting newspaper:', err);
    }
  };

  // Filtered Issues
  const filteredIssues = useMemo(() => {
    return issues.filter((iss) => {
      const matchNews =
        selectedNewspaperId === 'ALL' || iss.newspaper_id === Number(selectedNewspaperId);

      const st = (iss.ingestion_status || '').toLowerCase();
      let matchStatus = selectedStatus === 'ALL';
      if (selectedStatus === 'COMPLETED') {
        matchStatus = st === 'completed' || st === 'indexed' || st === 'parsed';
      } else if (selectedStatus === 'PROCESSING') {
        matchStatus = st !== 'completed' && st !== 'indexed' && st !== 'parsed' && st !== 'failed';
      } else if (selectedStatus === 'FAILED') {
        matchStatus = st === 'failed';
      }

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

        {/* Action and Aggregate Stats Bar */}
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setShowManageModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 transition-colors border border-slate-700"
          >
            <Building2 className="w-3.5 h-3.5 text-emerald-400" />
            <span>Manage Publications</span>
          </button>

          <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 rounded-xl p-2 px-4 text-xs">
            <div className="flex items-center gap-1.5 text-slate-300">
              <Newspaper className="w-4 h-4 text-emerald-400" />
              <span className="font-semibold">{metrics.newspapersCount}</span>
              <span className="text-slate-500">Newspapers</span>
            </div>
            <span className="text-slate-700">|</span>
            <div className="flex items-center gap-1.5 text-slate-300">
              <Layers className="w-4 h-4 text-teal-400" />
              <span className="font-semibold">{metrics.issuesCount}</span>
              <span className="text-slate-500">Issues</span>
            </div>
            <span className="text-slate-700">|</span>
            <div className="flex items-center gap-1.5 text-slate-300">
              <BookOpen className="w-4 h-4 text-emerald-400" />
              <span className="font-semibold">{metrics.articlesCount}</span>
              <span className="text-slate-500">Articles</span>
            </div>
          </div>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="py-5 flex flex-col md:flex-row items-center justify-between gap-3 border-b border-slate-800/80">
        <div className="flex flex-wrap items-center gap-2 w-full md:w-auto text-xs">
          {/* Newspaper Pills */}
          <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
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

          {/* Date Range Inputs */}
          <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs">
            <Calendar className="w-3.5 h-3.5 text-slate-500" />
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="bg-transparent text-slate-300 outline-none text-xs"
              placeholder="From Date"
            />
            <span className="text-slate-600">→</span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="bg-transparent text-slate-300 outline-none text-xs"
              placeholder="To Date"
            />
          </div>
        </div>

        {/* Search & Status Filters */}
        <div className="flex items-center gap-3 w-full md:w-auto">
          <div className="relative w-full md:w-56">
            <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search issues..."
              className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 outline-none focus:border-emerald-500 placeholder-slate-500"
            />
          </div>

          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300 outline-none cursor-pointer"
          >
            <option value="ALL">All Ingestion Statuses</option>
            <option value="COMPLETED">Completed / Indexed</option>
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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
          {filteredIssues.map((iss) => {
            const st = (iss.ingestion_status || '').toLowerCase();
            const isDone = st === 'completed' || st === 'indexed' || st === 'parsed';
            const isFail = st === 'failed';
            const isProc = !isDone && !isFail;

            return (
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
                        isDone
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : isProc
                          ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                          : 'bg-red-500/10 text-red-400 border border-red-500/20'
                      }`}
                    >
                      {isDone && <CheckCircle2 className="w-3 h-3" />}
                      {isProc && <Clock className="w-3 h-3 animate-spin" />}
                      {isFail && <AlertTriangle className="w-3 h-3" />}
                      {isDone ? 'Indexed' : isProc ? 'Processing' : 'Failed'}
                    </span>
                  </div>

                  {/* Date & Edition */}
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="text-lg font-bold font-serif text-slate-100 group-hover:text-emerald-300 transition-colors">
                      {iss.issue_date}
                    </h3>
                    <button
                      type="button"
                      onClick={(e) => handleOpenEdit(iss, e)}
                      className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-emerald-300 transition-colors text-[11px] flex items-center gap-1"
                      title="Edit Date or Newspaper"
                    >
                      <Edit2 className="w-3 h-3" /> Edit Date
                    </button>
                  </div>

                  <p className="text-xs text-slate-400 mb-3">
                    {iss.edition || 'Main Broadsheet Edition'} • {iss.language || 'eng'}
                  </p>

                  {/* Badges Info */}
                  <div className="flex items-center gap-3 text-xs text-slate-400 pt-2 border-t border-slate-800/80">
                    <span>{iss.total_pages || 0} Pages</span>
                    <span>•</span>
                    <span className="text-emerald-400 font-medium">
                      {iss.article_count || 0} Articles
                    </span>
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
            );
          })}
        </div>
      )}

      {/* Edit Issue Metadata Modal */}
      {editingIssue && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
                <Edit2 className="w-4 h-4 text-emerald-400" />
                Edit Issue Metadata & Sync Vectors
              </h2>
              <button
                type="button"
                onClick={() => setEditingIssue(null)}
                className="text-slate-400 hover:text-slate-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleSaveIssueEdit} className="space-y-4 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Issue Date *</label>
                <input
                  type="date"
                  required
                  value={editDate}
                  onChange={(e) => setEditDate(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 outline-none focus:border-emerald-500"
                />
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Updating the date automatically synchronizes all Qdrant chunk vector payloads and database records.
                </span>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Publication Title</label>
                <select
                  value={editNewspaperId}
                  onChange={(e) => setEditNewspaperId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 outline-none focus:border-emerald-500"
                >
                  {newspapers.map((np) => (
                    <option key={np.id} value={np.id}>
                      {np.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Edition Tag</label>
                <input
                  type="text"
                  value={editEdition}
                  onChange={(e) => setEditEdition(e.target.value)}
                  placeholder="e.g. morning"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 outline-none focus:border-emerald-500"
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setEditingIssue(null)}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSavingEdit || !editDate}
                  className="px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs flex items-center gap-1.5 disabled:opacity-40"
                >
                  {isSavingEdit ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Syncing Payloads...</span>
                    </>
                  ) : (
                    <span>Save & Update Vectors</span>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Manage Newspapers Modal */}
      {showManageModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg p-6 shadow-2xl space-y-5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
                <Building2 className="w-4 h-4 text-emerald-400" />
                Manage Newspaper Publications
              </h2>
              <button
                type="button"
                onClick={() => setShowManageModal(false)}
                className="text-slate-400 hover:text-slate-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Add Newspaper Form */}
            <form onSubmit={handleCreateNewspaper} className="bg-slate-950 border border-slate-800/80 rounded-xl p-4 space-y-3">
              <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">
                Register New Newspaper
              </h3>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">Title / Name *</label>
                  <input
                    type="text"
                    required
                    value={newPubName}
                    onChange={(e) => setNewPubName(e.target.value)}
                    placeholder="e.g. The Financial Times"
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Publisher</label>
                  <input
                    type="text"
                    value={newPubPublisher}
                    onChange={(e) => setNewPubPublisher(e.target.value)}
                    placeholder="e.g. Nikkei / Pearson"
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 outline-none focus:border-emerald-500"
                  />
                </div>
              </div>
              <button
                type="submit"
                disabled={!newPubName.trim() || isCreatingPub}
                className="w-full bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold py-2 rounded-lg transition-colors disabled:opacity-40"
              >
                {isCreatingPub ? 'Registering...' : '+ Add Newspaper to Archive'}
              </button>
            </form>

            {/* Existing Newspapers List */}
            <div className="space-y-2 max-h-60 overflow-y-auto">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Existing Publications ({newspapers.length})
              </h3>
              {newspapers.map((np) => (
                <div
                  key={np.id}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-800/80 text-xs"
                >
                  <div>
                    <span className="font-semibold text-slate-200 block">{np.name}</span>
                    <span className="text-[11px] text-slate-400">
                      {np.publisher || 'Independent'} • {np.issue_count || 0} issues indexed
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleDeleteNewspaper(np.id, np.name)}
                    className="p-1.5 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded transition-colors"
                    title="Delete publication and all issues"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
