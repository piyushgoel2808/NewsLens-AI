import React, { useState } from 'react';

export default function UploadTrigger() {
  const [file, setFile] = useState(null);
  const [newspaperName, setNewspaperName] = useState('The Daily Record');
  const [issueDate, setIssueDate] = useState('2026-08-21');
  const [edition, setEdition] = useState('morning');
  const [status, setStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  async function handleUpload(e) {
    e.preventDefault();
    if (!file || isUploading) return;

    setIsUploading(true);
    setStatus('Uploading file to /api/ingest/upload...');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('newspaper_name', newspaperName);
      formData.append('issue_date', issueDate);
      formData.append('edition', edition);

      const response = await fetch('/api/ingest/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `Upload failed with status ${response.status}`);
      }

      setStatus(data);
    } catch (err) {
      setStatus({ error: err.message });
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '12px 0' }}>
      <h3>2. UploadTrigger (POST /api/ingest/upload)</h3>
      <form onSubmit={handleUpload}>
        <div style={{ marginBottom: '8px' }}>
          <label>File (.pdf / .zip): </label>
          <input
            type="file"
            accept=".pdf,.zip"
            onChange={(e) => setFile(e.target.files[0] || null)}
          />
        </div>

        <div style={{ marginBottom: '8px' }}>
          <label>Newspaper: </label>
          <input
            type="text"
            value={newspaperName}
            onChange={(e) => setNewspaperName(e.target.value)}
            style={{ marginRight: '12px' }}
          />

          <label>Date: </label>
          <input
            type="date"
            value={issueDate}
            onChange={(e) => setIssueDate(e.target.value)}
            style={{ marginRight: '12px' }}
          />

          <label>Edition: </label>
          <input
            type="text"
            value={edition}
            onChange={(e) => setEdition(e.target.value)}
          />
        </div>

        <button type="submit" disabled={!file || isUploading} style={{ padding: '6px 14px' }}>
          {isUploading ? 'Uploading...' : 'Submit Ingestion Job'}
        </button>
      </form>

      {status && (
        <div style={{ marginTop: '12px' }}>
          <strong>Upload Response:</strong>
          <pre style={{ background: '#f5f5f5', padding: '8px', overflow: 'auto' }}>
            {JSON.stringify(status, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
