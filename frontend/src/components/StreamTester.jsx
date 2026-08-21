import React, { useState } from 'react';

export default function StreamTester() {
  const [query, setQuery] = useState('What happened to the financial markets?');
  const [stage, setStage] = useState('idle');
  const [plan, setPlan] = useState(null);
  const [tokens, setTokens] = useState('');
  const [citations, setCitations] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [telemetry, setTelemetry] = useState(null);

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
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim() }),
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
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter research query..."
          style={{ width: '60%', padding: '8px' }}
        />
        <button type="submit" disabled={isStreaming} style={{ marginLeft: '8px', padding: '8px 16px' }}>
          {isStreaming ? 'Streaming...' : 'Stream Query'}
        </button>
      </form>

      <div style={{ marginTop: '12px' }}>
        <strong>Current Stage: </strong>
        <code>{stage}</code>
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
