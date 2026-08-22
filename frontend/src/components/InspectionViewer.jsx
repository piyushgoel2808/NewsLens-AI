import React, { useState, useEffect } from 'react';

export default function InspectionViewer() {
  const [issues, setIssues] = useState([]);
  const [selectedIssueId, setSelectedIssueId] = useState('');
  const [inspectionData, setInspectionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('pages'); // 'pages' | 'articles' | 'chunks'
  const [chunkOffset, setChunkOffset] = useState(0);
  const chunkLimit = 50;

  async function loadIssues() {
    try {
      const res = await fetch('/api/issues');
      if (res.ok) {
        const data = await res.json();
        setIssues(data);
        if (data.length > 0 && !selectedIssueId) {
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

  async function fetchInspection(issueId, offset = 0) {
    if (!issueId) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/issues/${issueId}/inspection?chunk_limit=${chunkLimit}&chunk_offset=${offset}`);
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
    const confirmDelete = window.confirm(`Are you sure you want to permanently delete Issue #${selectedIssueId} and all its vectors, files, and chunks?`);
    if (!confirmDelete) return;

    setLoading(true);
    setDeleteMessage({ status: 'deleting', text: `Purging Issue #${selectedIssueId} across Qdrant, MinIO, and MySQL...` });
    try {
      const res = await fetch(`/api/issues/${selectedIssueId}`, {
        method: 'DELETE',
      });
      const result = await res.json();
      if (!res.ok) {
        throw new Error(result.detail || 'Deletion failed');
      }
      setDeleteMessage({ status: 'success', text: `Issue #${selectedIssueId} successfully purged from Qdrant, MinIO, and MySQL!` });
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
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '12px 0' }}>
      <h3>4. Ingestion & Chunking Transparency Inspector</h3>

      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap' }}>
        <label><strong>Select Ingested Issue: </strong></label>
        <select
          value={selectedIssueId}
          onChange={(e) => setSelectedIssueId(e.target.value)}
          style={{ padding: '6px' }}
        >
          <option value="">-- Choose Issue --</option>
          {issues.map((iss) => (
            <option key={iss.id} value={iss.id}>
              ID: {iss.id} | {iss.newspaper_name} ({iss.issue_date}) - {iss.total_pages} pages [{iss.ingestion_status}]
            </option>
          ))}
        </select>

        <button onClick={() => fetchInspection(selectedIssueId, chunkOffset)} disabled={!selectedIssueId || loading}>
          {loading ? 'Refreshing...' : 'Refresh Inspection Data'}
        </button>

        <button
          onClick={handleDeleteIssue}
          disabled={!selectedIssueId || loading}
          style={{ background: '#ffebee', color: '#c62828', borderColor: '#ef9a9a', cursor: 'pointer' }}
        >
          🗑️ Delete Issue
        </button>
      </div>

      {deleteMessage && (
        <div
          style={{
            marginBottom: '12px',
            padding: '8px 12px',
            fontSize: '13px',
            background: deleteMessage.status === 'success' ? '#e8f5e9' : deleteMessage.status === 'error' ? '#ffebee' : '#fff3e0',
            border: '1px solid #ccc',
            borderRadius: '4px',
          }}
        >
          {deleteMessage.text}
        </div>
      )}

      {issue && (
        <div style={{ background: '#f8f9fa', border: '1px solid #e9ecef', padding: '10px', marginBottom: '12px', fontSize: '13px' }}>
          <strong>Issue Overview: </strong>
          <span>{issue.newspaper_name} | Date: {issue.issue_date} | Edition: {issue.edition} | Status: <code>{issue.ingestion_status}</code></span>
          <div style={{ marginTop: '4px', color: '#555' }}>
            Total Pages: <strong>{issue.total_pages}</strong> | Segmented Articles: <strong>{issue.article_count}</strong> | Total Chunks Indexed: <strong>{issue.total_chunks}</strong>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '4px', borderBottom: '2px solid #ccc', marginBottom: '12px' }}>
        <button
          onClick={() => setActiveTab('pages')}
          style={{
            padding: '8px 16px',
            border: 'none',
            background: activeTab === 'pages' ? '#0066cc' : '#eee',
            color: activeTab === 'pages' ? '#fff' : '#000',
            cursor: 'pointer',
            fontWeight: activeTab === 'pages' ? 'bold' : 'normal',
          }}
        >
          1. Pages & OCR Fallback ({pages.length})
        </button>
        <button
          onClick={() => setActiveTab('articles')}
          style={{
            padding: '8px 16px',
            border: 'none',
            background: activeTab === 'articles' ? '#0066cc' : '#eee',
            color: activeTab === 'articles' ? '#fff' : '#000',
            cursor: 'pointer',
            fontWeight: activeTab === 'articles' ? 'bold' : 'normal',
          }}
        >
          2. Segmented Articles ({articles.length})
        </button>
        <button
          onClick={() => setActiveTab('chunks')}
          style={{
            padding: '8px 16px',
            border: 'none',
            background: activeTab === 'chunks' ? '#0066cc' : '#eee',
            color: activeTab === 'chunks' ? '#fff' : '#000',
            cursor: 'pointer',
            fontWeight: activeTab === 'chunks' ? 'bold' : 'normal',
          }}
        >
          3. Hierarchical Chunks ({issue?.total_chunks || chunks.length})
        </button>
      </div>

      {loading && <p>Loading inspection details...</p>}

      {/* Tab 1: Pages */}
      {activeTab === 'pages' && !loading && (
        <div>
          <h4>Page Extraction Analysis</h4>
          {pages.length === 0 ? <p>No pages ingested for this issue.</p> : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '12px' }}>
              {pages.map((p) => (
                <div key={p.id} style={{ border: '1px solid #ddd', padding: '10px', borderRadius: '4px', background: p.is_advertisement_page ? '#fffdf7' : '#fff' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <strong>Printed Folio: Page {p.printed_page_number || p.page_number}</strong>
                      <div style={{ fontSize: '11px', color: '#666' }}>PDF Index: p.{p.page_number} ({p.width_px} x {p.height_px} px)</div>
                    </div>
                    {p.is_advertisement_page && (
                      <span style={{ fontSize: '11px', background: '#ffe082', color: '#b78103', padding: '2px 6px', borderRadius: '3px', fontWeight: 'bold' }}>
                        📢 Ad Wrap
                      </span>
                    )}
                  </div>

                  <div style={{ marginTop: '6px', fontSize: '13px' }}>
                    <strong>Extraction Mode: </strong>
                    <span style={{ fontWeight: 'bold' }}>{p.extraction_mode}</span>
                  </div>

                  {p.ocr_fallback_triggered && (
                    <div style={{ marginTop: '6px', padding: '4px 8px', background: '#fff3cd', border: '1px solid #ffeeba', borderRadius: '3px', fontSize: '12px', color: '#856404' }}>
                      ⚠️ <strong>OCR Fallback Triggered</strong>
                      <div>{p.ocr_fallback_reason}</div>
                      <div>OCR Confidence: {p.ocr_confidence ? (p.ocr_confidence * 100).toFixed(1) + '%' : 'N/A'}</div>
                    </div>
                  )}

                  <div style={{ marginTop: '8px' }}>
                    <a href={p.image_url} target="_blank" rel="noreferrer" style={{ fontSize: '12px', color: '#0066cc' }}>
                      View 300 DPI Scan PNG ↗
                    </a>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Articles */}
      {activeTab === 'articles' && !loading && (
        <div>
          <h4>Segmented Articles Manifest ({articles.length})</h4>
          {articles.length === 0 ? <p>No articles segmented.</p> : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {articles.map((art) => (
                <div key={art.id} style={{ border: '1px solid #ddd', padding: '10px', background: art.is_advertisement ? '#fffdf7' : '#fff' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <strong style={{ fontSize: '15px' }}>{art.headline}</strong>
                      <div style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                        Section: {art.section || 'General'} | Type: <code>{art.article_type}</code> | Folio: <strong>{art.printed_pages?.length ? art.printed_pages.join(', ') : (art.pages?.join(', ') || '1')}</strong> <span style={{ color: '#888' }}>(PDF p.{art.pages?.join(', ')})</span>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right', fontSize: '12px' }}>
                      {art.is_advertisement ? (
                        <span style={{ fontSize: '11px', background: '#ffe082', color: '#b78103', padding: '2px 6px', borderRadius: '3px', fontWeight: 'bold' }}>
                          📢 Ad (No Vectors)
                        </span>
                      ) : (
                        <div>Prominence: <strong>{art.prominence_score?.toFixed(2)}</strong></div>
                      )}
                      <div>Word Count: {art.word_count}</div>
                    </div>
                  </div>

                  {art.summary && (
                    <div style={{ marginTop: '6px', fontSize: '13px', background: '#f9f9f9', padding: '6px', fontStyle: 'italic' }}>
                      Summary: {art.summary}
                    </div>
                  )}

                  {art.full_text_preview && (
                    <div style={{ marginTop: '6px', fontSize: '12px', color: '#333' }}>
                      <strong>Text Preview: </strong>{art.full_text_preview}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Hierarchical Chunks */}
      {activeTab === 'chunks' && !loading && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <h4>Hierarchical Chunks (Showing {chunkOffset + 1} - {Math.min(chunkOffset + chunkLimit, pagination?.total || chunks.length)} of {pagination?.total || chunks.length})</h4>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => fetchInspection(selectedIssueId, Math.max(0, chunkOffset - chunkLimit))}
                disabled={chunkOffset === 0}
              >
                « Prev {chunkLimit} Chunks
              </button>
              <button
                onClick={() => fetchInspection(selectedIssueId, chunkOffset + chunkLimit)}
                disabled={!pagination?.has_more}
              >
                Next {chunkLimit} Chunks »
              </button>
            </div>
          </div>

          {chunks.length === 0 ? <p>No chunks generated for this issue.</p> : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {chunks.map((chk) => (
                <div key={chk.id} style={{ border: '1px solid #e0e0e0', padding: '10px', background: '#fafafa' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                    <span><strong>Chunk #{chk.chunk_index}</strong> (Article ID: {chk.article_id} - <em>{chk.headline}</em>)</span>
                    <span>Tokens: <strong>{chk.token_count}</strong> | Vector ID: <code>{chk.embedding_vector_id || 'N/A'}</code></span>
                  </div>
                  <pre style={{ margin: 0, padding: '8px', background: '#fff', border: '1px solid #ddd', fontSize: '12px', whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                    {chk.text}
                  </pre>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
