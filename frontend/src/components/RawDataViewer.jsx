import React, { useState } from 'react';

export default function RawDataViewer() {
  const [activeEndpoint, setActiveEndpoint] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [swapProvider, setSwapProvider] = useState('groq_llama');
  const [swapMessage, setSwapMessage] = useState(null);

  async function fetchEndpoint(endpoint) {
    setActiveEndpoint(endpoint);
    setLoading(true);
    setData(null);

    try {
      const response = await fetch(endpoint);
      const json = await response.json();
      setData(json);
    } catch (err) {
      setData({ error: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleSwapBinding(e) {
    e.preventDefault();
    setSwapMessage('Swapping binding...');
    try {
      const response = await fetch('/api/settings/model-bindings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_bindings: {
            query_planner: swapProvider,
            answerer: swapProvider,
          },
        }),
      });
      const result = await response.json();
      setSwapMessage(result);
      if (activeEndpoint === '/api/settings/model-bindings') {
        fetchEndpoint('/api/settings/model-bindings');
      }
    } catch (err) {
      setSwapMessage({ error: err.message });
    }
  }

  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '12px 0' }}>
      <h3>3. RawDataViewer (Corpus, Settings & History Verification)</h3>

      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
        <button onClick={() => fetchEndpoint('/api/newspapers')}>GET /api/newspapers</button>
        <button onClick={() => fetchEndpoint('/api/issues')}>GET /api/issues</button>
        <button onClick={() => fetchEndpoint('/api/settings/model-bindings')}>GET /api/settings/model-bindings</button>
        <button onClick={() => fetchEndpoint('/api/entities')}>GET /api/entities</button>
        <button onClick={() => fetchEndpoint('/api/topics')}>GET /api/topics</button>
        <button onClick={() => fetchEndpoint('/api/query/history')}>GET /api/query/history</button>
      </div>

      <div style={{ borderTop: '1px solid #eee', paddingTop: '10px', marginBottom: '12px' }}>
        <strong>Runtime Model-Binding Swapper (PUT /api/settings/model-bindings):</strong>
        <form onSubmit={handleSwapBinding} style={{ marginTop: '6px' }}>
          <label>Assign query_planner & answerer to: </label>
          <input
            type="text"
            value={swapProvider}
            onChange={(e) => setSwapProvider(e.target.value)}
            style={{ marginRight: '8px' }}
          />
          <button type="submit">Update Binding</button>
        </form>
        {swapMessage && (
          <div style={{ marginTop: '6px', fontSize: '12px' }}>
            Result: <code>{JSON.stringify(swapMessage)}</code>
          </div>
        )}
      </div>

      <div>
        <strong>Data from: </strong>
        <code>{activeEndpoint || 'No endpoint selected'}</code>
        {loading && <span> (Loading...)</span>}

        <pre
          style={{
            maxHeight: '300px',
            overflow: 'auto',
            background: '#f9f9f9',
            border: '1px solid #ddd',
            padding: '8px',
            marginTop: '8px',
          }}
        >
          {data ? JSON.stringify(data, null, 2) : '// Click an endpoint button above to inspect JSON output'}
        </pre>
      </div>
    </div>
  );
}
