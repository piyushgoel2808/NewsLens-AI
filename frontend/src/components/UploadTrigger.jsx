import React, { useState } from 'react';

export default function UploadTrigger() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [newspaperName, setNewspaperName] = useState('Business Standard');
  const [issueDate, setIssueDate] = useState('2026-08-21');
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
        size: (f.size / 1024).toFixed(1) + ' KB',
        status: 'queued',
        jobId: null,
        detail: null,
      }))
    );
  }

  async function handleBatchUpload(e) {
    e.preventDefault();
    if (selectedFiles.length === 0 || isUploading) return;

    setIsUploading(true);

    // Sequential for...of loop to avoid overwhelming the backend/broker
    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];

      // Mark current file as uploading
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
          infoDetail = `Issue #${data.issue_id} | ${res.articles_created || 0} articles, ${res.chunks_created || 0} chunks`;
        } else {
          infoDetail = `Issue #${data.issue_id || 'Queued'}`;
        }

        setUploadQueue((prev) =>
          prev.map((item, idx) =>
            idx === i
              ? {
                  ...item,
                  status: isDuplicate ? 'skipped (duplicate)' : 'completed',
                  jobId: data.job_id || 'N/A',
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
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '12px 0' }}>
      <h3>2. UploadTrigger (Multi-PDF / ZIP Sequential Ingestion)</h3>
      <form onSubmit={handleBatchUpload}>
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '8px', alignItems: 'center' }}>
          <div>
            <label>Select Files (Multiple Allowed): </label>
            <input
              type="file"
              multiple
              accept=".pdf,.zip"
              onChange={handleFileChange}
              disabled={isUploading}
            />
          </div>

          <div>
            <label>Newspaper: </label>
            <input
              type="text"
              value={newspaperName}
              onChange={(e) => setNewspaperName(e.target.value)}
              disabled={isUploading}
              style={{ width: '160px' }}
            />
          </div>

          <div>
            <label>Date: </label>
            <input
              type="date"
              value={issueDate}
              onChange={(e) => setIssueDate(e.target.value)}
              disabled={isUploading}
            />
          </div>

          <div>
            <label>Edition: </label>
            <select
              value={edition}
              onChange={(e) => setEdition(e.target.value)}
              disabled={isUploading}
            >
              <option value="morning">Morning</option>
              <option value="evening">Evening</option>
              <option value="special">Special</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <input
              type="checkbox"
              id="forceReingestCheck"
              checked={forceReingest}
              onChange={(e) => setForceReingest(e.target.checked)}
              disabled={isUploading}
            />
            <label htmlFor="forceReingestCheck" style={{ cursor: 'pointer', fontWeight: forceReingest ? 'bold' : 'normal', color: forceReingest ? '#d9534f' : 'inherit' }}>
              ⚡ Force Re-ingest (Overwrite)
            </label>
          </div>

          <button
            type="submit"
            disabled={selectedFiles.length === 0 || isUploading}
            style={{ padding: '6px 16px', fontWeight: 'bold', background: '#0066cc', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
          >
            {isUploading ? `Uploading (${uploadQueue.filter(u => u.status === 'completed' || u.status === 'skipped (duplicate)').length}/${selectedFiles.length})...` : `Upload ${selectedFiles.length} File(s)`}
          </button>
        </div>
      </form>

      {uploadQueue.length > 0 && (
        <div style={{ marginTop: '12px' }}>
          <strong>Upload Queue & Progress:</strong>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '6px', fontSize: '13px' }}>
            <thead>
              <tr style={{ background: '#f0f0f0', textAlign: 'left' }}>
                <th style={{ padding: '6px', border: '1px solid #ddd' }}>File Name</th>
                <th style={{ padding: '6px', border: '1px solid #ddd' }}>Size</th>
                <th style={{ padding: '6px', border: '1px solid #ddd' }}>Status</th>
                <th style={{ padding: '6px', border: '1px solid #ddd' }}>Job / Info</th>
              </tr>
            </thead>
            <tbody>
              {uploadQueue.map((item) => (
                <tr key={item.id} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '6px', border: '1px solid #ddd' }}>{item.name}</td>
                  <td style={{ padding: '6px', border: '1px solid #ddd' }}>{item.size}</td>
                  <td style={{ padding: '6px', border: '1px solid #ddd' }}>
                    <span
                      style={{
                        padding: '2px 6px',
                        borderRadius: '3px',
                        background:
                          item.status === 'completed'
                            ? '#d4edda'
                            : item.status === 'uploading'
                            ? '#cce5ff'
                            : item.status === 'skipped (duplicate)'
                            ? '#fff3cd'
                            : item.status === 'failed'
                            ? '#f8d7da'
                            : '#e2e3e5',
                      }}
                    >
                      {item.status}
                    </span>
                  </td>
                  <td style={{ padding: '6px', border: '1px solid #ddd' }}>
                    {item.jobId ? <code>Job: {item.jobId}</code> : ''} {item.detail || ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
