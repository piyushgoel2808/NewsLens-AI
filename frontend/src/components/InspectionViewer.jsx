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
    <div className="bg-surface border border-border-subtle rounded-2xl p-5 md:p-6 shadow-2xl text-text-primary space-y-4 font-sans transition-colors duration-200">
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-border-subtle pb-3.5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-accent-primary/10 border border-accent-primary/30 flex items-center justify-center text-accent-primary">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold font-serif text-text-primary flex items-center gap-2">
              Ingestion & Chunking Transparency Inspector
            </h3>
            <p className="text-[11px] text-text-muted font-mono">
              Live inspection of raw MinIO storage, OCR fallbacks, and Qdrant hierarchical chunks
            </p>
          </div>
        </div>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 text-text-muted hover:text-text-primary hover:bg-elevated rounded-lg transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Issue Selector Controls */}
      <div className="flex flex-wrap items-center gap-3 bg-canvas p-3 rounded-xl border border-border-subtle text-xs">
        <div className="flex items-center gap-2 flex-1 min-w-[260px]">
          <label className="text-text-secondary font-semibold font-mono whitespace-nowrap">Target Issue:</label>
          <select
            value={selectedIssueId}
            onChange={(e) => setSelectedIssueId(e.target.value)}
            className="w-full bg-surface border border-border-subtle rounded-lg px-3 py-1.5 text-text-primary outline-none focus:border-accent-primary font-mono text-xs cursor-pointer"
          >
            <option value="">-- Choose Issue to Inspect --</option>
            {issues.map((iss) => (
              <option key={iss.id} value={iss.id} className="bg-surface text-text-primary">
                #{iss.id} | {iss.newspaper_name} ({iss.issue_date}) — {iss.chunk_count || 0} chunks [{iss.ingestion_status}]
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={() => fetchInspection(selectedIssueId, chunkOffset)}
          disabled={!selectedIssueId || loading}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-surface hover:bg-elevated text-text-primary border border-border-subtle rounded-lg transition-colors font-semibold font-mono disabled:opacity-40 cursor-pointer shadow-xs"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-accent-primary ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>

        <button
          onClick={handleDeleteIssue}
          disabled={!selectedIssueId || loading}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 rounded-lg transition-colors font-semibold font-mono disabled:opacity-40 cursor-pointer"
        >
          <Trash2 className="w-3.5 h-3.5" />
          Purge Issue
        </button>
      </div>

      {deleteMessage && (
        <div
          className={`p-3 text-xs rounded-lg border font-mono ${
            deleteMessage.status === 'success'
              ? 'bg-accent-primary/10 border-accent-primary/30 text-accent-primary'
              : deleteMessage.status === 'error'
              ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
              : 'bg-amber-500/10 border-amber-500/30 text-amber-400'
          }`}
        >
          {deleteMessage.text}
        </div>
      )}

      {/* Overview Stat Badges */}
      {issue && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 bg-canvas p-3 rounded-xl border border-border-subtle text-xs font-mono">
          <div>
            <span className="text-[10px] text-text-muted uppercase tracking-wider block">Newspaper</span>
            <span className="font-bold text-text-primary font-serif">{issue.newspaper_name}</span>
          </div>
          <div>
            <span className="text-[10px] text-text-muted uppercase tracking-wider block">Date & Edition</span>
            <span className="font-semibold text-text-secondary">{issue.issue_date} ({issue.edition})</span>
          </div>
          <div>
            <span className="text-[10px] text-text-muted uppercase tracking-wider block">MinIO Scans</span>
            <span className="font-bold text-accent-primary">{issue.total_pages} Pages Rendered</span>
          </div>
          <div>
            <span className="text-[10px] text-text-muted uppercase tracking-wider block">Qdrant Vectors</span>
            <span className="font-bold text-accent-editorial">{issue.total_chunks} Total Chunks</span>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex items-center gap-1.5 border-b border-border-subtle pb-2 font-mono">
        <button
          onClick={() => setActiveTab('chunks')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
            activeTab === 'chunks'
              ? 'bg-accent-primary text-white shadow-xs'
              : 'text-text-secondary hover:text-text-primary hover:bg-elevated'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5" />
          Hierarchical Chunks ({issue?.total_chunks || chunks.length})
        </button>

        <button
          onClick={() => setActiveTab('pages')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
            activeTab === 'pages'
              ? 'bg-accent-primary text-white shadow-xs'
              : 'text-text-secondary hover:text-text-primary hover:bg-elevated'
          }`}
        >
          <ImageIcon className="w-3.5 h-3.5" />
          MinIO Pages & OCR ({pages.length})
        </button>

        <button
          onClick={() => setActiveTab('articles')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
            activeTab === 'articles'
              ? 'bg-accent-primary text-white shadow-xs'
              : 'text-text-secondary hover:text-text-primary hover:bg-elevated'
          }`}
        >
          <FileText className="w-3.5 h-3.5" />
          Segmented Articles ({articles.length})
        </button>
      </div>

      {loading && (
        <div className="py-12 flex flex-col items-center justify-center text-text-muted space-y-2 font-mono">
          <RefreshCw className="w-5 h-5 animate-spin text-accent-primary" />
          <p className="text-xs">Loading inspection details from MySQL & Qdrant...</p>
        </div>
      )}

      {/* Tab 1: Hierarchical Chunks */}
      {activeTab === 'chunks' && !loading && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-text-muted">
              Showing {chunkOffset + 1} – {Math.min(chunkOffset + chunkLimit, pagination?.total || chunks.length)} of{' '}
              <strong className="text-accent-primary">{pagination?.total || chunks.length}</strong> indexed chunks
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => fetchInspection(selectedIssueId, Math.max(0, chunkOffset - chunkLimit))}
                disabled={chunkOffset === 0}
                className="flex items-center gap-1 px-2.5 py-1 bg-surface hover:bg-elevated text-text-primary rounded border border-border-subtle disabled:opacity-40 transition-colors font-medium text-xs cursor-pointer"
              >
                <ChevronLeft className="w-3.5 h-3.5" /> Prev {chunkLimit}
              </button>
              <button
                onClick={() => fetchInspection(selectedIssueId, chunkOffset + chunkLimit)}
                disabled={!pagination?.has_more}
                className="flex items-center gap-1 px-2.5 py-1 bg-surface hover:bg-elevated text-text-primary rounded border border-border-subtle disabled:opacity-40 transition-colors font-medium text-xs cursor-pointer"
              >
                Next {chunkLimit} <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {chunks.length === 0 ? (
            <p className="text-xs text-text-muted py-6 text-center font-mono">
              No chunks generated for this issue.
            </p>
          ) : (
            <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
              {chunks.map((chk) => (
                <div
                  key={chk.id}
                  className="bg-canvas border border-border-subtle hover:border-accent-primary/50 rounded-xl p-3.5 transition-colors space-y-2"
                >
                  <div className="flex items-center justify-between text-xs flex-wrap gap-2">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[11px] font-bold text-accent-primary bg-accent-primary/10 border border-accent-primary/30 px-2 py-0.5 rounded">
                        Chunk #{chk.chunk_index + 1}
                      </span>
                      <span className="font-semibold text-text-primary truncate max-w-sm font-serif">
                        {chk.headline}
                      </span>
                      {chk.chunk_type === 'visual' && (
                        <span className="bg-accent-editorial/10 text-accent-editorial border border-accent-editorial/30 px-2 py-0.5 rounded text-[10px] font-mono">
                          📊 Infographic / Visual
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-[11px] text-text-muted font-mono">
                      <span>Tokens: <strong className="text-text-primary">{chk.token_count}</strong></span>
                      {chk.embedding_vector_id && (
                        <span className="text-text-muted truncate max-w-[120px]" title={chk.embedding_vector_id}>
                          ID: {chk.embedding_vector_id.slice(0, 8)}...
                        </span>
                      )}
                    </div>
                  </div>

                  <pre className="m-0 p-3 bg-surface border border-border-subtle rounded-lg text-accent-primary font-mono text-xs leading-relaxed whitespace-pre-wrap">
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
          <p className="text-xs text-text-secondary font-mono">
            Rendered PNG broadsheet scans stored directly in MinIO bucket (<code>newslens-pages</code>).
          </p>
          {pages.length === 0 ? (
            <p className="text-xs text-text-muted py-6 text-center font-mono">
              No pages ingested for this issue.
            </p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 max-h-[500px] overflow-y-auto pr-1">
              {pages.map((p) => (
                <div
                  key={p.id}
                  className={`bg-canvas border rounded-xl p-3.5 space-y-2 text-xs flex flex-col justify-between ${
                    p.is_advertisement_page ? 'border-amber-500/40 bg-amber-500/10' : 'border-border-subtle'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <strong className="text-text-primary font-serif text-sm">Page {p.page_number}</strong>
                      {p.is_advertisement_page && (
                        <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 text-[10px] px-2 py-0.5 rounded font-mono">
                          📢 Ad Wrap
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-text-muted font-mono">
                      Dimensions: {p.width_px} × {p.height_px} px
                    </div>
                    <div className="text-[11px] text-text-secondary mt-1">
                      Mode: <span className="font-semibold text-text-primary font-mono">{p.extraction_mode}</span>
                    </div>

                    {p.ocr_fallback_triggered && (
                      <div className="mt-2 p-2 rounded bg-amber-500/10 border border-amber-500/30 text-[11px] text-amber-400 space-y-0.5 font-mono">
                        <div className="font-semibold">⚠️ OCR Fallback Triggered</div>
                        <div>{p.ocr_fallback_reason}</div>
                        <div>Confidence: {p.ocr_confidence ? (p.ocr_confidence * 100).toFixed(1) + '%' : 'N/A'}</div>
                      </div>
                    )}
                  </div>

                  <div className="pt-2 border-t border-border-subtle">
                    <a
                      href={p.image_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-accent-primary hover:opacity-80 font-semibold text-xs flex items-center gap-1 font-mono"
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
          <p className="text-xs text-text-secondary font-mono">
            Segmented articles with polygon boundaries, multi-page continuation stitching, and visual assets.
          </p>
          {articles.length === 0 ? (
            <p className="text-xs text-text-muted py-6 text-center font-mono">
              No articles segmented for this issue.
            </p>
          ) : (
            <div className="space-y-2.5 max-h-[500px] overflow-y-auto pr-1">
              {articles.map((art) => (
                <div
                  key={art.id}
                  className="bg-canvas border border-border-subtle rounded-xl p-3.5 space-y-2 text-xs"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h4 className="font-bold text-text-primary text-sm font-serif">{art.headline}</h4>
                      <div className="text-[11px] text-text-muted font-mono mt-0.5">
                        Section: {art.section || 'General'} • Type: <code>{art.article_type}</code> • Page(s): {art.pages?.join(', ') || '1'}
                      </div>
                    </div>
                    <div className="text-right text-[11px] text-text-muted font-mono whitespace-nowrap">
                      <div>Words: <strong className="text-text-primary">{art.word_count}</strong></div>
                      <div>Prominence: <strong className="text-accent-primary">{art.prominence_score?.toFixed(2)}</strong></div>
                    </div>
                  </div>

                  {art.summary && (
                    <p className="text-text-secondary bg-surface p-2.5 rounded-lg text-xs italic border border-border-subtle font-serif leading-relaxed">
                      {art.summary}
                    </p>
                  )}

                  {art.full_text_preview && (
                    <div className="text-[11px] text-text-muted">
                      <strong className="text-text-secondary">Text: </strong>
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
