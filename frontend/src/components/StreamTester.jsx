import React, { useState, useEffect } from 'react';

export default function StreamTester() {
  const [query, setQuery] = useState('What happened to the financial markets?');
  const [stage, setStage] = useState('idle');
  const [plan, setPlan] = useState(null);
  const [tokens, setTokens] = useState('');
  const [citations, setCitations] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [telemetry, setTelemetry] = useState(null);

  // Model selection state
  const [availableProviders, setAvailableProviders] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [loadingProviders, setLoadingProviders] = useState(true);

  useEffect(() => {
    async function loadModelBindings() {
      try {
        const res = await fetch('/api/settings/model-bindings');
        if (res.ok) {
          const data = await res.json();
          const configured = data.configured_providers || [];
          setAvailableProviders(configured);
          // Set default model from active task binding for query_planner / answerer
          const defaultBinding = data.task_bindings?.answerer || data.task_bindings?.query_planner;
          if (defaultBinding) {
            setSelectedModel(defaultBinding);
          } else if (configured.length > 0) {
            setSelectedModel(configured[0].id);
          }
        }
      } catch (err) {
        console.error('Failed to fetch model bindings:', err);
      } finally {
        setLoadingProviders(false);
      }
    }
    loadModelBindings();
  }, []);

  async function handleStream(e) {
    e.preventDefault();
    if (!query.trim() || isStreaming) return;

    setIsStreaming(true);
    setStage('initiating');
    setPlan(null);
    setTokens('');
    setCitations([]);
    setTelemetry(null);

    try {
      const payload = {
        query: query.trim(),
        model_override: selectedModel || undefined,
      };

      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = 'message';
        for (const line of lines) {
          if (line.startsWith('event:')) {
            currentEvent = line.replace('event:', '').trim();
          } else if (line.startsWith('data:')) {
            const jsonStr = line.replace('data:', '').trim();
            if (!jsonStr) continue;

            try {
              const data = JSON.parse(jsonStr);

              if (currentEvent === 'stage') {
                setStage(data.stage);
              } else if (currentEvent === 'plan') {
                setPlan(data);
              } else if (currentEvent === 'token') {
                setTokens((prev) => prev + data.delta);
              } else if (currentEvent === 'citations') {
                setCitations(data.citations || []);
              } else if (currentEvent === 'done') {
                setStage('completed');
                setTelemetry(data);
              }
            } catch (err) {
              console.error('Error parsing SSE json:', err, jsonStr);
            }
          }
        }
      }
    } catch (err) {
      console.error('Stream error:', err);
      setStage(`error: ${err.message}`);
    } finally {
      setIsStreaming(false);
    }
  }

  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '12px 0' }}>
      <h3>1. StreamTester (SSE POST /api/query/stream)</h3>
      <form onSubmit={handleStream}>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center', marginBottom: '8px' }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter research query..."
            style={{ flex: '1', minWidth: '280px', padding: '8px' }}
          />

          <label style={{ fontSize: '14px', fontWeight: 'bold' }}>Model: </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={loadingProviders || isStreaming}
            style={{ padding: '8px' }}
          >
            {loadingProviders && <option value="">Loading providers...</option>}
            {availableProviders.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id} ({p.provider} - {p.model || 'default'})
              </option>
            ))}
          </select>

          <button type="submit" disabled={isStreaming} style={{ padding: '8px 16px' }}>
            {isStreaming ? 'Streaming...' : 'Stream Query'}
          </button>
        </div>
      </form>

      <div style={{ marginTop: '12px' }}>
        <strong>Current Stage: </strong>
        <code>{stage}</code>
        {selectedModel && (
          <span style={{ marginLeft: '12px', fontSize: '13px', color: '#555' }}>
            [Executing on: <code>{selectedModel}</code>]
          </span>
        )}
      </div>

      {plan && (
        <div style={{ marginTop: '12px', background: '#f5f5f5', padding: '8px' }}>
          <strong>Archetype: </strong> {plan.archetype}
          <div><strong>Planned Tools:</strong> {plan.plan.map((t) => t.tool_name).join(' -> ')}</div>
        </div>
      )}

      <div style={{ marginTop: '12px' }}>
        <strong>Live Token Stream:</strong>
        <div
          style={{
            minHeight: '100px',
            border: '1px solid #ddd',
            padding: '8px',
            background: '#fafafa',
            whiteSpace: 'pre-wrap',
            fontFamily: 'monospace',
          }}
        >
          {tokens || (isStreaming ? 'Waiting for tokens...' : 'Tokens will appear here...')}
        </div>
      </div>

      {citations.length > 0 && (
        <div style={{ marginTop: '12px' }}>
          <strong>Citations ({citations.length}):</strong>
          <ul>
            {citations.map((c, i) => (
              <li key={i}>
                [{c.newspaper_name}, {c.issue_date}, Page {c.page_number}] - <em>{c.headline}</em>
              </li>
            ))}
          </ul>
        </div>
      )}

      {telemetry && (
        <div style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
          Telemetry: {telemetry.latency_ms}ms | Cost: ${telemetry.cost_usd}
        </div>
      )}
    </div>
  );
}
