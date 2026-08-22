import React, { useState } from 'react';
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
} from 'lucide-react';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

export default function UploadTrigger() {
  const { openIssueInReader } = useActiveHighlight();

  const [selectedFiles, setSelectedFiles] = useState([]);
  const [newspaperName, setNewspaperName] = useState('Mint');
  const [issueDate, setIssueDate] = useState('2026-08-22');
  const [edition, setEdition] = useState('morning');
  const [forceReingest, setForceReingest] = useState(false);
  const [uploadQueue, setUploadQueue] = useState([]);
  const [isUploading, setIsUploading] = useState(false);

  function handleFileChange(e) {
    const files = Array.from(e.target.files || []);
    setSelectedFiles(files);
    setUploadQueue(
      files.map((f, idx) => ({
        id: idx,
        name: f.name,
        size: (f.size / (1024 * 1024)).toFixed(2) + ' MB',
        status: 'queued',
        issueId: null,
        detail: null,
      }))
    );
  }

  async function handleBatchUpload(e) {
    e.preventDefault();
    if (selectedFiles.length === 0 || isUploading) return;

    setIsUploading(true);

    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];

      setUploadQueue((prev) =>
        prev.map((item, idx) =>
          idx === i ? { ...item, status: 'uploading' } : item
        )
      );

      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('newspaper_name', newspaperName);
        formData.append('issue_date', issueDate);
        formData.append('edition', edition);
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
          prev.map((item, idx) =>
            idx === i
              ? {
                  ...item,
                  status: isDuplicate ? 'skipped (duplicate)' : 'completed',
                  issueId: data.issue_id,
                  detail: infoDetail,
                }
              : item
          )
        );
      } catch (err) {
        setUploadQueue((prev) =>
          prev.map((item, idx) =>
            idx === i
              ? {
                  ...item,
                  status: 'failed',
                  detail: err.message,
                }
              : item
          )
        );
      }
    }

    setIsUploading(false);
  }

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 max-w-4xl mx-auto p-4 md:p-6 overflow-y-auto">
      {/* Header */}
      <div className="pb-5 mb-5 border-b border-slate-800">
        <h1 className="text-2xl font-bold font-serif text-slate-100 flex items-center gap-2.5">
          <UploadCloud className="w-6 h-6 text-emerald-400" />
          Ingestion Console & PDF Uploader
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Upload multi-page broadsheet newspaper PDFs for automated Docling layout analysis, OCR extraction, article segmentation, and vector indexing.
        </p>
      </div>

      {/* Upload Form */}
      <form onSubmit={handleBatchUpload} className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl mb-6 space-y-5">
        {/* Dropzone */}
        <div className="border-2 border-dashed border-slate-700 hover:border-emerald-500/70 rounded-xl p-8 text-center cursor-pointer transition-colors bg-slate-950/50">
          <input
            type="file"
            id="pdf-upload"
            accept="application/pdf"
            multiple
            onChange={handleFileChange}
            disabled={isUploading}
            className="hidden"
          />
          <label htmlFor="pdf-upload" className="cursor-pointer block">
            <UploadCloud className="w-12 h-12 text-emerald-400 mx-auto mb-3 opacity-80" />
            <span className="text-sm font-semibold text-slate-200 block mb-1">
              Click to select broadsheet PDFs or drag and drop
            </span>
            <span className="text-xs text-slate-500">
              Supports scanned broadsheets, digital vector PDFs, and hybrid editions
            </span>
          </label>
        </div>

        {/* Configuration Row */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div>
            <label className="block text-slate-400 font-medium mb-1.5">Newspaper Title</label>
            <select
              value={newspaperName}
              onChange={(e) => setNewspaperName(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 outline-none focus:border-emerald-500"
            >
              <option value="Mint">Mint</option>
              <option value="The Hindu">The Hindu</option>
              <option value="Business Standard">Business Standard</option>
              <option value="The Indian Express">The Indian Express</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 font-medium mb-1.5">Issue Date</label>
            <input
              type="date"
              value={issueDate}
              onChange={(e) => setIssueDate(e.target.value)}
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
            disabled={selectedFiles.length === 0 || isUploading}
            className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-6 py-2.5 rounded-xl flex items-center justify-center gap-2 transition-colors disabled:opacity-40 shadow-lg"
          >
            {isUploading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Processing Pipeline...</span>
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
            Ingestion Progress Queue ({uploadQueue.length})
          </h3>

          <div className="space-y-2">
            {uploadQueue.map((item) => (
              <div
                key={item.id}
                className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3"
              >
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-slate-400 shrink-0" />
                  <div>
                    <span className="text-sm font-semibold text-slate-200 block">{item.name}</span>
                    <span className="text-xs text-slate-500">{item.size}</span>
                  </div>
                </div>

                <div className="flex items-center gap-3 self-end sm:self-auto">
                  {item.detail && (
                    <span className="text-xs text-slate-400 max-w-xs truncate">{item.detail}</span>
                  )}

                  <span
                    className={`text-xs font-semibold px-2.5 py-1 rounded-full flex items-center gap-1.5 ${
                      item.status === 'completed'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                        : item.status === 'uploading'
                        ? 'bg-blue-500/10 text-blue-400 border border-blue-500/30'
                        : item.status === 'failed'
                        ? 'bg-red-500/10 text-red-400 border border-red-500/30'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {item.status === 'completed' && <CheckCircle2 className="w-3.5 h-3.5" />}
                    {item.status === 'uploading' && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
                    {item.status === 'failed' && <AlertCircle className="w-3.5 h-3.5" />}
                    {item.status}
                  </span>

                  {item.issueId && (
                    <button
                      onClick={() => openIssueInReader(item.issueId, 1)}
                      className="bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 text-xs font-medium px-3 py-1 rounded-lg flex items-center gap-1 transition-colors"
                    >
                      <span>Read</span>
                      <ExternalLink className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
