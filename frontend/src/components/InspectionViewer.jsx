import React, { useState, useEffect } from 'react';
import {
  Layers,
  Sparkles,
  FileText,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Trash2,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  X,
  Database,
  Image as ImageIcon,
} from 'lucide-react';

export default function InspectionViewer({ initialIssueId = '', onClose = null }) {
  const [issues, setIssues] = useState([]);
  const [selectedIssueId, setSelectedIssueId] = useState(String(initialIssueId || ''));
  const [inspectionData, setInspectionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chunks'); // 'chunks' | 'pages' | 'articles'
  const [chunkOffset, setChunkOffset] = useState(0);
  const chunkLimit = 50;

  async function loadIssues() {
    try {
      const res = await fetch('/api/issues');
      if (res.ok) {
        const data = await res.json();
        setIssues(data);
        if (data.length > 0 && !selectedIssueId && !initialIssueId) {
          setSelectedIssueId(String(data[0].id));
        }
      }
    } catch (err) {
      console.error('Failed to load issues:', err);
    }
  }

  useEffect(() => {
    loadIssues();
  }, []);

  useEffect(() => {
    if (initialIssueId) {
      setSelectedIssueId(String(initialIssueId));
    }
  }, [initialIssueId]);

  async function fetchInspection(issueId, offset = 0) {
    if (!issueId) return;
    setLoading(true);
    try {
      const res = await fetch(
        `/api/issues/${issueId}/inspection?chunk_limit=${chunkLimit}&chunk_offset=${offset}`
      );
      if (res.ok) {
        const data = await res.json();
        setInspectionData(data);
        setChunkOffset(offset);
      } else {
        setInspectionData({ error: `HTTP ${res.status}: ${res.statusText}` });
      }
    } catch (err) {
      setInspectionData({ error: err.message });
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (selectedIssueId) {
      fetchInspection(selectedIssueId, 0);
    }
  }, [selectedIssueId]);

  const [deleteMessage, setDeleteMessage] = useState(null);

  async function handleDeleteIssue() {
    if (!selectedIssueId) return;
    const confirmDelete = window.confirm(
      `Are you sure you want to permanently delete Issue #${selectedIssueId} and all its vectors, files, and chunks?`
    );
    if (!confirmDelete) return;

    setLoading(true);
    setDeleteMessage({
      status: 'deleting',
      text: `Purging Issue #${selectedIssueId} across Qdrant, MinIO, and MySQL...`,
    });
    try {
      const res = await fetch(`/api/issues/${selectedIssueId}`, {
        method: 'DELETE',
      });
      const result = await res.json();
      if (!res.ok) {
        throw new Error(result.detail || 'Deletion failed');
      }
      setDeleteMessage({
        status: 'success',
        text: `Issue #${selectedIssueId} successfully purged from Qdrant, MinIO, and MySQL!`,
      });
      setSelectedIssueId('');
      setInspectionData(null);
      await loadIssues();
    } catch (err) {
      setDeleteMessage({ status: 'error', text: err.message });
    } finally {
      setLoading(false);
    }
  }

  const issue = inspectionData?.issue;
  const pages = inspectionData?.pages || [];
  const articles = inspectionData?.articles || [];
  const chunks = inspectionData?.chunks || [];
  const pagination = inspectionData?.pagination;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 md:p-6 shadow-2xl text-slate-100 space-y-4">
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3.5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
              Ingestion & Chunking Transparency Inspector
            </h3>
            <p className="text-[11px] text-slate-400 font-mono">
              Live inspection of raw MinIO storage, OCR fallbacks, and Qdrant hierarchical chunks
            </p>
          </div>
        </div>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Issue Selector Controls */}
      <div className="flex flex-wrap items-center gap-3 bg-slate-950 p-3 rounded-xl border border-slate-800/80 text-xs">
        <div className="flex items-center gap-2 flex-1 min-w-[260px]">
          <label className="text-slate-400 font-semibold whitespace-nowrap">Target Issue:</label>
          <select
            value={selectedIssueId}
            onChange={(e) => setSelectedIssueId(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 outline-none focus:border-emerald-500 font-mono text-xs"
          >
            <option value="">-- Choose Issue to Inspect --</option>
            {issues.map((iss) => (
              <option key={iss.id} value={iss.id}>
                #{iss.id} | {iss.newspaper_name} ({iss.issue_date}) — {iss.chunk_count || 0} chunks [{iss.ingestion_status}]
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={() => fetchInspection(selectedIssueId, chunkOffset)}
          disabled={!selectedIssueId || loading}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors font-semibold disabled:opacity-40"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>

        <button
          onClick={handleDeleteIssue}
          disabled={!selectedIssueId || loading}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 rounded-lg transition-colors font-semibold disabled:opacity-40"
        >
          <Trash2 className="w-3.5 h-3.5" />
          Purge Issue
        </button>
      </div>

      {deleteMessage && (
        <div
          className={`p-3 text-xs rounded-lg border font-mono ${
            deleteMessage.status === 'success'
              ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
              : deleteMessage.status === 'error'
              ? 'bg-rose-950/40 border-rose-500/40 text-rose-300'
              : 'bg-amber-950/40 border-amber-500/40 text-amber-300'
          }`}
        >
          {deleteMessage.text}
        </div>
      )}

      {/* Overview Stat Badges */}
      {issue && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 bg-slate-950/60 p-3 rounded-xl border border-slate-800/60 text-xs">
          <div>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Newspaper</span>
            <span className="font-bold text-slate-200">{issue.newspaper_name}</span>
          </div>
          <div>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Date & Edition</span>
            <span className="font-semibold text-slate-300">{issue.issue_date} ({issue.edition})</span>
          </div>
          <div>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider block">MinIO Scans</span>
            <span className="font-bold text-emerald-400">{issue.total_pages} Pages Rendered</span>
          </div>
          <div>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Qdrant Vectors</span>
            <span className="font-bold text-emerald-400">{issue.total_chunks} Total Chunks</span>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex items-center gap-1.5 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab('chunks')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            activeTab === 'chunks'
              ? 'bg-emerald-500 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5" />
          Hierarchical Chunks ({issue?.total_chunks || chunks.length})
        </button>

        <button
          onClick={() => setActiveTab('pages')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            activeTab === 'pages'
              ? 'bg-emerald-500 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          <ImageIcon className="w-3.5 h-3.5" />
          MinIO Pages & OCR ({pages.length})
        </button>

        <button
          onClick={() => setActiveTab('articles')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            activeTab === 'articles'
              ? 'bg-emerald-500 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          <FileText className="w-3.5 h-3.5" />
          Segmented Articles ({articles.length})
        </button>
      </div>

      {loading && (
        <div className="py-12 flex flex-col items-center justify-center text-slate-400 space-y-2">
          <RefreshCw className="w-5 h-5 animate-spin text-emerald-400" />
          <p className="text-xs font-mono">Loading inspection details from MySQL & Qdrant...</p>
        </div>
      )}

      {/* Tab 1: Hierarchical Chunks */}
      {activeTab === 'chunks' && !loading && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 font-mono">
              Showing {chunkOffset + 1} – {Math.min(chunkOffset + chunkLimit, pagination?.total || chunks.length)} of{' '}
              <strong className="text-emerald-400">{pagination?.total || chunks.length}</strong> indexed chunks
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => fetchInspection(selectedIssueId, Math.max(0, chunkOffset - chunkLimit))}
                disabled={chunkOffset === 0}
                className="flex items-center gap-1 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded disabled:opacity-40 transition-colors font-medium text-xs"
              >
                <ChevronLeft className="w-3.5 h-3.5" /> Prev {chunkLimit}
              </button>
              <button
                onClick={() => fetchInspection(selectedIssueId, chunkOffset + chunkLimit)}
                disabled={!pagination?.has_more}
                className="flex items-center gap-1 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded disabled:opacity-40 transition-colors font-medium text-xs"
              >
                Next {chunkLimit} <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {chunks.length === 0 ? (
            <p className="text-xs text-slate-500 py-6 text-center font-mono">
              No chunks generated for this issue.
            </p>
          ) : (
            <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
              {chunks.map((chk) => (
                <div
                  key={chk.id}
                  className="bg-slate-950 border border-slate-800/90 hover:border-emerald-500/50 rounded-xl p-3.5 transition-colors space-y-2"
                >
                  <div className="flex items-center justify-between text-xs flex-wrap gap-2">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[11px] font-bold text-emerald-400 bg-emerald-950/70 border border-emerald-800/60 px-2 py-0.5 rounded">
                        Chunk #{chk.chunk_index + 1}
                      </span>
                      <span className="font-semibold text-slate-200 truncate max-w-sm">
                        {chk.headline}
                      </span>
                      {chk.chunk_type === 'visual' && (
                        <span className="bg-sky-950/60 text-sky-400 border border-sky-800/50 px-2 py-0.5 rounded text-[10px] font-mono">
                          📊 Infographic / Visual
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
                      <span>Tokens: <strong className="text-slate-200">{chk.token_count}</strong></span>
                      {chk.embedding_vector_id && (
                        <span className="text-slate-500 truncate max-w-[120px]" title={chk.embedding_vector_id}>
                          ID: {chk.embedding_vector_id.slice(0, 8)}...
                        </span>
                      )}
                    </div>
                  </div>

                  <pre className="m-0 p-3 bg-slate-900/90 border border-slate-800/80 rounded-lg text-emerald-300 font-mono text-xs leading-relaxed whitespace-pre-wrap">
                    {chk.text}
                  </pre>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 2: MinIO Pages */}
      {activeTab === 'pages' && !loading && (
        <div className="space-y-3">
          <p className="text-xs text-slate-400">
            Rendered PNG broadsheet scans stored directly in MinIO bucket (<code>newslens-pages</code>).
          </p>
          {pages.length === 0 ? (
            <p className="text-xs text-slate-500 py-6 text-center font-mono">
              No pages ingested for this issue.
            </p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 max-h-[500px] overflow-y-auto pr-1">
              {pages.map((p) => (
                <div
                  key={p.id}
                  className={`bg-slate-950 border rounded-xl p-3.5 space-y-2 text-xs flex flex-col justify-between ${
                    p.is_advertisement_page ? 'border-amber-500/40 bg-amber-950/10' : 'border-slate-800'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <strong className="text-slate-200 font-serif text-sm">Page {p.page_number}</strong>
                      {p.is_advertisement_page && (
                        <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 text-[10px] px-2 py-0.5 rounded font-mono">
                          📢 Ad Wrap
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-400 font-mono">
                      Dimensions: {p.width_px} × {p.height_px} px
                    </div>
                    <div className="text-[11px] text-slate-400 mt-1">
                      Mode: <span className="font-semibold text-slate-200">{p.extraction_mode}</span>
                    </div>

                    {p.ocr_fallback_triggered && (
                      <div className="mt-2 p-2 rounded bg-amber-950/40 border border-amber-500/30 text-[11px] text-amber-300 space-y-0.5">
                        <div className="font-semibold">⚠️ OCR Fallback Triggered</div>
                        <div>{p.ocr_fallback_reason}</div>
                        <div>Confidence: {p.ocr_confidence ? (p.ocr_confidence * 100).toFixed(1) + '%' : 'N/A'}</div>
                      </div>
                    )}
                  </div>

                  <div className="pt-2 border-t border-slate-800/80">
                    <a
                      href={p.image_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-emerald-400 hover:text-emerald-300 font-semibold text-xs flex items-center gap-1"
                    >
                      <ExternalLink className="w-3.5 h-3.5" /> View MinIO Scan PNG
                    </a>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Articles */}
      {activeTab === 'articles' && !loading && (
        <div className="space-y-3">
          <p className="text-xs text-slate-400">
            Segmented articles with polygon boundaries, multi-page continuation stitching, and visual assets.
          </p>
          {articles.length === 0 ? (
            <p className="text-xs text-slate-500 py-6 text-center font-mono">
              No articles segmented for this issue.
            </p>
          ) : (
            <div className="space-y-2.5 max-h-[500px] overflow-y-auto pr-1">
              {articles.map((art) => (
                <div
                  key={art.id}
                  className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2 text-xs"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h4 className="font-bold text-slate-200 text-sm">{art.headline}</h4>
                      <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                        Section: {art.section || 'General'} • Type: <code>{art.article_type}</code> • Page(s): {art.pages?.join(', ') || '1'}
                      </div>
                    </div>
                    <div className="text-right text-[11px] text-slate-400 font-mono whitespace-nowrap">
                      <div>Words: <strong className="text-slate-200">{art.word_count}</strong></div>
                      <div>Prominence: <strong className="text-emerald-400">{art.prominence_score?.toFixed(2)}</strong></div>
                    </div>
                  </div>

                  {art.summary && (
                    <p className="text-slate-300 bg-slate-900/70 p-2 rounded text-xs italic border border-slate-800/60">
                      {art.summary}
                    </p>
                  )}

                  {art.full_text_preview && (
                    <div className="text-[11px] text-slate-400">
                      <strong className="text-slate-300">Text: </strong>
                      {art.full_text_preview}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
