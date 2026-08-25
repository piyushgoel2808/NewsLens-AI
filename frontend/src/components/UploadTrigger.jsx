import React, { useState, useEffect } from 'react';
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  Clock,
  ExternalLink,
  RefreshCw,
  Layers,
  Sparkles,
  Plus,
  Trash2,
  Settings2,
  X,
  HelpCircle,
  Building2,
  Calendar,
} from 'lucide-react';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

export default function UploadTrigger() {
  const { openIssueInReader } = useActiveHighlight();

  const [selectedFiles, setSelectedFiles] = useState([]);
  const [newspapers, setNewspapers] = useState([]);
  const [loadingNewspapers, setLoadingNewspapers] = useState(true);

  const [newspaperMode, setNewspaperMode] = useState('auto'); // 'auto' | custom id
  const [customNewspaperName, setCustomNewspaperName] = useState('');
  const [issueDate, setIssueDate] = useState(''); // empty = auto-detect
  const [edition, setEdition] = useState('morning');
  const [parserEngine, setParserEngine] = useState('auto');
  const [forceReingest, setForceReingest] = useState(false);

  const [uploadQueue, setUploadQueue] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [inspectingFiles, setInspectingFiles] = useState(false);

  // Manage Newspapers Modal
  const [showManageModal, setShowManageModal] = useState(false);
  const [newPubName, setNewPubName] = useState('');
  const [newPubPublisher, setNewPubPublisher] = useState('');
  const [newPubLang, setNewPubLang] = useState('en');
  const [newPubCountry, setNewPubCountry] = useState('IN');
  const [isCreatingPub, setIsCreatingPub] = useState(false);

  // Fetch registered newspapers
  const loadNewspapers = async () => {
    try {
      setLoadingNewspapers(true);
      const res = await fetch('/api/newspapers');
      const data = await res.json();
      if (Array.isArray(data)) {
        setNewspapers(data);
      }
    } catch (err) {
      console.error('Failed to load newspapers:', err);
    } finally {
      setLoadingNewspapers(false);
    }
  };

  useEffect(() => {
    loadNewspapers();
  }, []);

  // Inspect uploaded files to extract consensus metadata
  async function handleFileChange(e) {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    setSelectedFiles(files);
    setInspectingFiles(true);

    const initialQueue = files.map((f, idx) => ({
      id: idx,
      file: f,
      name: f.name,
      size: (f.size / (1024 * 1024)).toFixed(2) + ' MB',
      status: 'inspecting',
      detectedNewspaper: null,
      detectedDate: null,
      isNewNewspaper: false,
      dateVotes: null,
      issueId: null,
      detail: 'Analyzing multi-page consensus...',
    }));
    setUploadQueue(initialQueue);

    // Run inspection on each PDF
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch('/api/ingest/inspect-preview', {
          method: 'POST',
          body: formData,
        });

        if (res.ok) {
          const preview = await res.json();
          setUploadQueue((prev) =>
            prev.map((item, idx) =>
              idx === i
                ? {
                    ...item,
                    status: 'ready',
                    detectedNewspaper: preview.detected_newspaper,
                    detectedDate: preview.detected_date,
                    isNewNewspaper: preview.is_new_newspaper,
                    dateVotes: preview.telemetry?.date_votes,
                    detail: `Detected: ${preview.detected_newspaper || 'Daily Broadsheet'} • Date: ${preview.detected_date || 'Auto'}`,
                  }
                : item
            )
          );
        } else {
          setUploadQueue((prev) =>
            prev.map((item, idx) =>
              idx === i
                ? {
                    ...item,
                    status: 'ready',
                    detail: 'Auto-detection will resolve during ingestion pipeline',
                  }
                : item
            )
          );
        }
      } catch (err) {
        setUploadQueue((prev) =>
          prev.map((item, idx) =>
            idx === i
              ? {
                  ...item,
                  status: 'ready',
                  detail: 'Ready for upload',
                }
              : item
          )
        );
      }
    }
    setInspectingFiles(false);
  }

  // Handle Ingest Batch
  async function handleBatchUpload(e) {
    e.preventDefault();
    if (selectedFiles.length === 0 || isUploading) return;

    setIsUploading(true);

    for (let i = 0; i < uploadQueue.length; i++) {
      const item = uploadQueue[i];
      const file = item.file || selectedFiles[i];

      setUploadQueue((prev) =>
        prev.map((q, idx) => (idx === i ? { ...q, status: 'uploading' } : q))
      );

      try {
        const formData = new FormData();
        formData.append('file', file);

        // Effective newspaper name
        let effectiveName = 'auto';
        if (newspaperMode !== 'auto') {
          effectiveName = newspaperMode;
        } else if (item.detectedNewspaper) {
          effectiveName = item.detectedNewspaper;
        }
        formData.append('newspaper_name', effectiveName);

        // Effective date
        let effectiveDate = issueDate || item.detectedDate || '';
        if (effectiveDate) {
          formData.append('issue_date', effectiveDate);
        }

        formData.append('edition', edition);
        formData.append('parser_engine', parserEngine);
        formData.append('force', forceReingest ? 'true' : 'false');

        const response = await fetch('/api/ingest/upload', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || `HTTP ${response.status}`);
        }

        const isDuplicate =
          data.status === 'skipped_duplicate' ||
          data.is_duplicate ||
          (data.skipped_duplicates && data.skipped_duplicates.length > 0);

        let infoDetail = '';
        if (isDuplicate) {
          infoDetail = 'Already in archive. Check "Force Re-ingest" to overwrite.';
        } else if (data.pipeline_results && data.pipeline_results.length > 0) {
          const res = data.pipeline_results[0];
          infoDetail = `Issue #${data.issue_id} • ${res.articles_created || 0} articles • ${res.chunks_created || 0} chunks`;
        } else {
          infoDetail = `Issue #${data.issue_id || 'Queued'}`;
        }

        setUploadQueue((prev) =>
          prev.map((q, idx) =>
            idx === i
              ? {
                  ...q,
                  status: isDuplicate ? 'skipped (duplicate)' : 'completed',
                  issueId: data.issue_id,
                  detail: infoDetail,
                }
              : q
          )
        );

        // Refresh newspapers list in case a new one was auto-created
        loadNewspapers();
      } catch (err) {
        setUploadQueue((prev) =>
          prev.map((q, idx) =>
            idx === i
              ? {
                  ...q,
                  status: 'failed',
                  detail: err.message,
                }
              : q
          )
        );
      }
    }

    setIsUploading(false);
  }

  // Create new newspaper
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
        setNewspaperMode(created.name);
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

  // Delete newspaper
  const handleDeleteNewspaper = async (id, name) => {
    if (
      !window.confirm(
        `Are you sure you want to delete "${name}"? This will permanently delete the publication and ALL associated issues and vector embeddings!`
      )
    ) {
      return;
    }

    try {
      const res = await fetch(`/api/newspapers/${id}`, { method: 'DELETE' });
      if (res.ok) {
        setNewspapers((prev) => prev.filter((n) => n.id !== id));
        if (newspaperMode === name) {
          setNewspaperMode('auto');
        }
      } else {
        alert('Failed to delete newspaper.');
      }
    } catch (err) {
      console.error('Error deleting newspaper:', err);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 max-w-4xl mx-auto p-4 md:p-6 overflow-y-auto">
      {/* Header */}
      <div className="pb-5 mb-5 border-b border-slate-800 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold font-serif text-slate-100 flex items-center gap-2.5">
            <UploadCloud className="w-6 h-6 text-emerald-400" />
            Ingestion Console & Zero-Friction PDF Uploader
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Automated multi-page date consensus, masthead brand recognition, neural layout parsing, and vector indexing.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShowManageModal(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 transition-colors border border-slate-700"
        >
          <Building2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>Manage Newspapers ({newspapers.length})</span>
        </button>
      </div>

      {/* Upload Form */}
      <form
        onSubmit={handleBatchUpload}
        className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl mb-6 space-y-5"
      >
        {/* Dropzone */}
        <div className="border-2 border-dashed border-slate-700 hover:border-emerald-500/70 rounded-xl p-8 text-center cursor-pointer transition-colors bg-slate-950/50">
          <input
            type="file"
            id="pdf-upload"
            accept="application/pdf"
            multiple
            onChange={handleFileChange}
            disabled={isUploading || inspectingFiles}
            className="hidden"
          />
          <label htmlFor="pdf-upload" className="cursor-pointer block">
            <UploadCloud className="w-12 h-12 text-emerald-400 mx-auto mb-3 opacity-80" />
            <span className="text-sm font-semibold text-slate-200 block mb-1">
              Click to select broadsheet PDFs or drag and drop
            </span>
            <span className="text-xs text-slate-500">
              Zero manual metadata required — automatic multi-page consensus extracts date and publication title
            </span>
          </label>
        </div>

        {/* Configuration Row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-slate-400 font-medium">Newspaper Publication</label>
              <button
                type="button"
                onClick={() => setShowManageModal(true)}
                className="text-[10px] text-emerald-400 hover:underline flex items-center gap-0.5"
              >
                <Plus className="w-2.5 h-2.5" /> New
              </button>
            </div>
            <select
              value={newspaperMode}
              onChange={(e) => setNewspaperMode(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 outline-none focus:border-emerald-500"
            >
              <option value="auto">✨ Auto-Detect from Document</option>
              {newspapers.map((np) => (
                <option key={np.id} value={np.name}>
                  {np.name} ({np.issue_count || 0} issues)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-slate-400 font-medium mb-1.5">
              Issue Date (Override)
            </label>
            <input
              type="date"
              value={issueDate}
              onChange={(e) => setIssueDate(e.target.value)}
              placeholder="Auto-detect date"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 outline-none focus:border-emerald-500"
            />
          </div>

          <div>
            <label className="block text-slate-400 font-medium mb-1.5">Edition Tag</label>
            <input
              type="text"
              value={edition}
              onChange={(e) => setEdition(e.target.value)}
              placeholder="e.g. morning, national"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 outline-none focus:border-emerald-500"
            />
          </div>

          <div>
            <label className="block text-emerald-400 font-semibold mb-1.5 flex items-center gap-1">
              <Sparkles className="w-3 h-3 text-emerald-400" />
              Parsing & Layout Engine
            </label>
            <select
              value={parserEngine}
              onChange={(e) => setParserEngine(e.target.value)}
              className="w-full bg-slate-950 border border-emerald-500/50 rounded-lg px-3 py-2 text-emerald-300 font-medium outline-none focus:border-emerald-400"
            >
              <option value="auto">✨ Auto (Google Cloud Vision Pure OCR + DeepSeek-R1 / Gemma 4)</option>
              <option value="google_cloud_vision">🔍 Google Cloud Vision API (Pure OCR Engine)</option>
              <option value="gemma4:26b">🦙 Local Ollama Gemma 4 (26B)</option>
            </select>
          </div>
        </div>

        {/* Options & Action Button */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2 border-t border-slate-800/80">
          <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-400">
            <input
              type="checkbox"
              checked={forceReingest}
              onChange={(e) => setForceReingest(e.target.checked)}
              className="rounded bg-slate-950 border-slate-800 text-emerald-500 focus:ring-0"
            />
            <span>Force Re-ingest (overwrite existing issue)</span>
          </label>

          <button
            type="submit"
            disabled={selectedFiles.length === 0 || isUploading || inspectingFiles}
            className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-6 py-2.5 rounded-xl flex items-center justify-center gap-2 transition-colors disabled:opacity-40 shadow-lg"
          >
            {isUploading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Processing Pipeline...</span>
              </>
            ) : inspectingFiles ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Inspecting Consensus...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Ingest {selectedFiles.length} File(s)</span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* Upload Queue Progress */}
      {uploadQueue.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Ingestion Queue & Consensus Inspection ({uploadQueue.length})
          </h3>

          <div className="space-y-2">
            {uploadQueue.map((item) => (
              <div
                key={item.id}
                className="bg-slate-950 border border-slate-800/80 rounded-xl p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="font-semibold text-slate-200 block">{item.name}</span>
                    <div className="flex items-center gap-2 text-[11px] text-slate-400 mt-0.5">
                      <span>{item.size}</span>
                      <span>•</span>
                      <span className="text-emerald-400">{item.detail}</span>
                    </div>

                    {/* New Newspaper Detected Alert / Confirmation */}
                    {item.isNewNewspaper && item.detectedNewspaper && (
                      <div className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 text-[11px]">
                        <AlertCircle className="w-3.5 h-3.5" />
                        <span>
                          New Publication Detected: <strong>{item.detectedNewspaper}</strong> (will be registered automatically)
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-3 self-end sm:self-center">
                  {item.status === 'uploading' && (
                    <span className="flex items-center gap-1.5 text-amber-400 font-medium">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Ingesting...
                    </span>
                  )}
                  {item.status === 'inspecting' && (
                    <span className="flex items-center gap-1.5 text-cyan-400 font-medium">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Extracting Date...
                    </span>
                  )}
                  {item.status === 'completed' && (
                    <div className="flex items-center gap-2">
                      <span className="flex items-center gap-1 text-emerald-400 font-medium">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Indexed
                      </span>
                      {item.issueId && (
                        <button
                          type="button"
                          onClick={() => openIssueInReader(item.issueId)}
                          className="px-2.5 py-1 bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 rounded border border-emerald-500/30 font-medium"
                        >
                          View
                        </button>
                      )}
                    </div>
                  )}
                  {item.status === 'failed' && (
                    <span className="flex items-center gap-1 text-rose-400 font-medium">
                      <AlertCircle className="w-3.5 h-3.5" /> Failed
                    </span>
                  )}
                </div>
              </div>
            ))}
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
