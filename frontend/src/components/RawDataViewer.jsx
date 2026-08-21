import React, { useState, useEffect } from 'react';

const TASK_OPTIONS = [
  { value: 'query_planner', label: 'Query Planner (LLM)' },
  { value: 'answerer', label: 'Answer Synthesizer (LLM)' },
  { value: 'layout_analysis', label: 'Layout Analysis (VLM)' },
  { value: 'article_segmentation', label: 'Article Segmentation (LLM)' },
  { value: 'metadata_extraction', label: 'Metadata & NER Extraction (LLM)' },
  { value: 'classification', label: 'Article Classification (LLM)' },
  { value: 'embedding', label: 'Vector Embedding (Embedder)' },
  { value: 'ocr', label: 'OCR Engine' },
];

export default function RawDataViewer() {
  const [activeEndpoint, setActiveEndpoint] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Dynamic Settings Swapper State
  const [selectedTask, setSelectedTask] = useState('query_planner');
  const [selectedProvider, setSelectedProvider] = useState('');
  const [configuredProviders, setConfiguredProviders] = useState([]);
  const [currentBindings, setCurrentBindings] = useState({});
  const [swapMessage, setSwapMessage] = useState(null);

  async function loadSettings() {
    try {
      const res = await fetch('/api/settings/model-bindings');
      if (res.ok) {
        const json = await res.json();
        setConfiguredProviders(json.configured_providers || []);
        setCurrentBindings(json.task_bindings || {});
        if (!selectedProvider && json.configured_providers?.length > 0) {
          setSelectedProvider(json.configured_providers[0].id);
        }
      }
    } catch (err) {
      console.error('Failed to load settings:', err);
    }
  }

  useEffect(() => {
    loadSettings();
  }, []);

  async function fetchEndpoint(endpoint) {
    setActiveEndpoint(endpoint);
    setLoading(true);
    setData(null);

    try {
      const response = await fetch(endpoint);
      const json = await response.json();
      setData(json);
      if (endpoint === '/api/settings/model-bindings' && response.ok) {
        setConfiguredProviders(json.configured_providers || []);
        setCurrentBindings(json.task_bindings || {});
      }
    } catch (err) {
      setData({ error: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleSwapBinding(e) {
    e.preventDefault();
    if (!selectedTask || !selectedProvider) return;

    setSwapMessage({ status: 'updating', text: `Assigning ${selectedTask} -> ${selectedProvider}...` });
    try {
      const payload = {
        task_bindings: {
          [selectedTask]: selectedProvider,
        },
      };

      const response = await fetch('/api/settings/model-bindings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || `Update failed (${response.status})`);
      }

      setSwapMessage({ status: 'success', text: `Successfully updated ${selectedTask} to ${selectedProvider}!` });
      setCurrentBindings(result.task_bindings || {});
      if (activeEndpoint === '/api/settings/model-bindings') {
        fetchEndpoint('/api/settings/model-bindings');
      }
    } catch (err) {
      setSwapMessage({ status: 'error', text: err.message });
    }
  }

  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '12px 0' }}>
      <h3>3. RawDataViewer & Model Settings Swapper</h3>

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
        <form onSubmit={handleSwapBinding} style={{ marginTop: '8px', display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <label style={{ fontSize: '13px' }}>Task: </label>
          <select
            value={selectedTask}
            onChange={(e) => {
              setSelectedTask(e.target.value);
              const active = currentBindings[e.target.value];
              if (active) setSelectedProvider(active);
            }}
            style={{ padding: '6px' }}
          >
            {TASK_OPTIONS.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label} {currentBindings[t.value] ? `(Active: ${currentBindings[t.value]})` : ''}
              </option>
            ))}
          </select>

          <label style={{ fontSize: '13px' }}>Provider: </label>
          <select
            value={selectedProvider}
            onChange={(e) => setSelectedProvider(e.target.value)}
            style={{ padding: '6px' }}
          >
            {configuredProviders.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id} ({p.provider} - {p.model || 'default'})
              </option>
            ))}
          </select>

          <button type="submit" style={{ padding: '6px 12px' }}>Update Task Binding</button>
        </form>

        {swapMessage && (
          <div
            style={{
              marginTop: '8px',
              padding: '6px 10px',
              fontSize: '13px',
              background: swapMessage.status === 'success' ? '#e6ffe6' : swapMessage.status === 'error' ? '#ffe6e6' : '#f0f0f0',
              border: '1px solid #ccc',
            }}
          >
            {swapMessage.text}
          </div>
        )}
      </div>

      <div>
        <strong>Endpoint JSON Output: </strong>
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
          {data ? JSON.stringify(data, null, 2) : '// Click an endpoint button above to inspect raw JSON output'}
        </pre>
      </div>
    </div>
  );
}
