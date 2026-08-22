import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Sparkles,
  Bot,
  User,
  ChevronDown,
  ChevronRight,
  Database,
  Search,
  Clock,
  Layers,
  MapPin,
  ExternalLink,
  Cpu,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

export default function AgentAssistant() {
  const {
    highlightArticle,
    selectedModel,
    setSelectedModel,
    chatMessages: messages,
    setChatMessages: setMessages,
  } = useActiveHighlight();

  const [query, setQuery] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);

  const messagesEndRef = useRef(null);

  // Fetch configured model providers
  useEffect(() => {
    fetch('/api/models/available')
      .then((res) => res.json())
      .then((data) => {
        if (data.models) {
          setAvailableModels(data.models);
        }
      })
      .catch((err) => console.error('Failed to load models:', err));
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (customQuery) => {
    const queryText = customQuery || query;
    if (!queryText.trim() || isStreaming) return;

    const userMessage = { role: 'user', content: queryText };
    const assistantMessage = {
      role: 'assistant',
      content: '',
      stage: 'planning',
      condensedQuery: null,
      archetype: null,
      plan: [],
      toolExecutions: [],
      citations: [],
      telemetry: null,
      isStreaming: true,
      isTelemetryOpen: false,
    };

    const historyPayload = messages
      .filter((m) => !m.isStreaming && m.content)
      .slice(-8)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setQuery('');
    setIsStreaming(true);

    try {
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          chat_history: historyPayload,
          model_override: selectedModel || undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const block of lines) {
          if (!block.trim()) continue;

          let eventType = 'message';
          let dataStr = '';

          const blockLines = block.split('\n');
          for (const line of blockLines) {
            if (line.startsWith('event: ')) {
              eventType = line.replace('event: ', '').trim();
            } else if (line.startsWith('data: ')) {
              dataStr = line.replace('data: ', '').trim();
            }
          }

          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);

            setMessages((prev) => {
              const updated = [...prev];
              const current = { ...updated[updated.length - 1] };

              if (eventType === 'stage') {
                current.stage = data.stage;
              } else if (eventType === 'query_condensed') {
                current.condensedQuery = data.condensed_query;
              } else if (eventType === 'plan') {
                current.archetype = data.archetype;
                current.plan = data.plan || [];
              } else if (eventType === 'tool_result') {
                current.toolExecutions = [
                  ...(current.toolExecutions || []),
                  data,
                ];
              } else if (eventType === 'token') {
                current.content = (current.content || '') + (data.delta || '');
              } else if (eventType === 'citations') {
                current.citations = data.citations || [];
              } else if (eventType === 'done') {
                current.telemetry = {
                  latency_ms: data.latency_ms,
                  cost_usd: data.cost_usd,
                  evidence_count: data.evidence_count,
                };
                current.isStreaming = false;
                current.stage = 'completed';
              }

              updated[updated.length - 1] = current;
              return updated;
            });
          } catch (jsonErr) {
            console.error('Error parsing SSE event data:', jsonErr, dataStr);
          }
        }
      }
    } catch (err) {
      console.error('Streaming request failed:', err);
      setMessages((prev) => {
        const updated = [...prev];
        const current = { ...updated[updated.length - 1] };
        current.content += `\n\n*(Error: Execution failed — ${err.message})*`;
        current.isStreaming = false;
        updated[updated.length - 1] = current;
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const toggleTelemetry = (index) => {
    setMessages((prev) => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        isTelemetryOpen: !updated[index].isTelemetryOpen,
      };
      return updated;
    });
  };

  const handleClearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content:
          'Hello! I am your **NewsLens-AI Research Assistant**. I can perform multi-step newspaper intelligence investigations, cross-newspaper comparative analysis, quantitative trend tracking, and temporal timeline reconstruction with verifiable spatial citations.',
        isStreaming: false,
      },
    ]);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 max-w-5xl mx-auto p-4">
      {/* Top Controls & Model Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-3 border-b border-slate-800 text-xs">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-emerald-400" />
          <span className="font-semibold text-slate-200">Agentic Research Planner & Answerer</span>
        </div>

        <div className="flex items-center gap-2">
          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-slate-400 font-medium">LLM Model:</span>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-slate-900 border border-emerald-500/50 rounded-lg px-2.5 py-1 text-emerald-300 font-medium text-xs outline-none cursor-pointer hover:border-emerald-400 focus:border-emerald-400 transition-colors"
          >
            <option value="groq_qwen">⚡ Groq / Qwen 3.6 (Fastest)</option>
            <option value="groq_llama">⚡ Groq / Llama 3.3 70B</option>
            <option value="groq_gpt_oss">⚡ Groq / GPT-OSS 120B</option>
            <option value="ollama_chat">💻 Ollama / Llama 3.2 (Local)</option>
            <option value="anthropic_sonnet">☁️ Anthropic / Claude 3.5 Sonnet</option>
            <option value="openai_gpt4o">☁️ OpenAI / GPT-4o</option>
            {availableModels
              .filter(
                (m) =>
                  ![
                    'groq_qwen',
                    'groq_llama',
                    'groq_gpt_oss',
                    'ollama_chat',
                    'anthropic_sonnet',
                    'openai_gpt4o',
                  ].includes(m.name)
              )
              .map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} ({m.provider})
                </option>
              ))}
          </select>
          <button
            onClick={handleClearChat}
            className="text-slate-400 hover:text-slate-200 text-xs px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 hover:bg-slate-800 transition-colors ml-1"
            title="Reset Chat History"
          >
            Clear History
          </button>
        </div>
      </div>

      {/* Message Chat Feed */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex gap-3 ${
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 shrink-0 mt-1">
                <Bot className="w-4 h-4" />
              </div>
            )}

            <div
              className={`max-w-[85%] rounded-xl p-4 ${
                msg.role === 'user'
                  ? 'bg-emerald-600 text-white rounded-br-none'
                  : 'bg-slate-900 border border-slate-800 rounded-bl-none shadow-lg'
              }`}
            >
              {/* Dynamic Stage Indicator (while assistant is streaming) */}
              {msg.role === 'assistant' && msg.isStreaming && (
                <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-800 text-xs text-emerald-400 font-medium">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span className="capitalize">
                    {msg.stage === 'condensing_query'
                      ? 'Resolving conversational coreference...'
                      : msg.stage === 'planning'
                      ? 'Formulating multi-step investigation plan...'
                      : msg.stage === 'tool_execution'
                      ? 'Executing hybrid vector & SQL retrieval tools...'
                      : 'Synthesizing evidence-grounded response...'}
                  </span>
                </div>
              )}

              {/* Condensed Query Pill */}
              {msg.role === 'assistant' && msg.condensedQuery && (
                <div className="flex items-center gap-1.5 text-[11px] text-emerald-300 font-mono bg-emerald-950/60 border border-emerald-500/30 px-2.5 py-1 rounded-md mb-2.5">
                  <Search className="w-3 h-3 text-emerald-400 shrink-0" />
                  <span className="truncate">
                    Resolved Context: <strong className="text-emerald-200">"{msg.condensedQuery}"</strong>
                  </span>
                </div>
              )}

              {/* Execution Plan & Tool Telemetry Disclosure */}
              {msg.role === 'assistant' &&
                (msg.plan?.length > 0 || msg.toolExecutions?.length > 0) && (
                  <div className="mb-3 border border-slate-800 rounded-lg overflow-hidden bg-slate-950/60">
                    <button
                      onClick={() => toggleTelemetry(idx)}
                      className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-800/50 transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <Database className="w-3.5 h-3.5 text-blue-400" />
                        <span>
                          Plan & Tool Telemetry ({msg.toolExecutions?.length || 0} tools run)
                        </span>
                        {msg.archetype && (
                          <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300">
                            {msg.archetype}
                          </span>
                        )}
                      </div>
                      {msg.isTelemetryOpen ? (
                        <ChevronDown className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronRight className="w-3.5 h-3.5" />
                      )}
                    </button>

                    {msg.isTelemetryOpen && (
                      <div className="p-3 border-t border-slate-800 space-y-2 text-xs text-slate-300 font-mono">
                        {/* Planned Steps */}
                        {msg.plan && msg.plan.length > 0 && (
                          <div>
                            <span className="text-[10px] text-slate-500 uppercase block mb-1">
                              Planned Tool Invocations:
                            </span>
                            <ul className="space-y-1 list-disc list-inside text-slate-400">
                              {msg.plan.map((p, pIdx) => (
                                <li key={pIdx}>
                                  <strong className="text-slate-200">{p.tool_name}</strong>: {p.purpose}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Executed Tools */}
                        {msg.toolExecutions && msg.toolExecutions.length > 0 && (
                          <div className="pt-2 border-t border-slate-800/80">
                            <span className="text-[10px] text-slate-500 uppercase block mb-1">
                              Tool Execution Results:
                            </span>
                            {msg.toolExecutions.map((t, tIdx) => (
                              <div key={tIdx} className="bg-slate-900 p-2 rounded border border-slate-800 mb-1.5">
                                <span className="text-emerald-400 font-semibold">{t.tool_name || 'Tool'}:</span>
                                <pre className="text-[11px] text-slate-400 overflow-x-auto mt-1 whitespace-pre-wrap">
                                  {JSON.stringify(t.execution || t, null, 2)}
                                </pre>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

              {/* Message Content */}
              <div className="text-sm leading-relaxed whitespace-pre-wrap font-sans text-slate-200">
                {msg.content || (msg.isStreaming ? 'Thinking...' : '')}
              </div>

              {/* Clickable Citations Grounding */}
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-800">
                  <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block mb-2">
                    Verified Newspaper Sources (Click to inspect spatial bboxes):
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {msg.citations.map((cit, cIdx) => (
                      <button
                        key={cIdx}
                        onClick={() =>
                          highlightArticle(
                            cit.issue_id,
                            cit.page_number || 1,
                            cit.article_id,
                            cit.bboxes || []
                          )
                        }
                        className="flex items-center gap-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2.5 py-1 rounded-md text-xs transition-all hover:scale-105"
                        title="Jump to broadsheet scan and pulse article bounding box"
                      >
                        <MapPin className="w-3 h-3 text-emerald-400" />
                        <span className="font-medium">
                          {cit.newspaper_name || 'Daily News'}, Page {cit.page_number || 1}
                        </span>
                        <ExternalLink className="w-2.5 h-2.5 opacity-60" />
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Telemetry Summary Footer */}
              {msg.telemetry && (
                <div className="flex items-center gap-3 mt-3 pt-2 text-[10px] text-slate-500 font-mono border-t border-slate-800">
                  <span>Latency: {msg.telemetry.latency_ms}ms</span>
                  <span>•</span>
                  <span>Evidence: {msg.telemetry.evidence_count} chunks</span>
                  <span>•</span>
                  <span>Cost: ${msg.telemetry.cost_usd.toFixed(4)}</span>
                </div>
              )}
            </div>

            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shrink-0 mt-1">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Prompt Chips */}
      {messages.length <= 1 && (
        <div className="py-2 flex flex-wrap gap-2 text-xs">
          <button
            onClick={() =>
              handleSend(
                'What are the key policy decisions and macroeconomic indicators announced in the latest issues?'
              )
            }
            className="bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 px-3 py-1.5 rounded-full transition-colors"
          >
            📊 Latest Policy & Macro Indicators
          </button>
          <button
            onClick={() =>
              handleSend(
                'Compare corporate deals and capital market disclosures between technology and energy sectors.'
              )
            }
            className="bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 px-3 py-1.5 rounded-full transition-colors"
          >
            🏢 Corporate Deals & Market Disclosures
          </button>
          <button
            onClick={() =>
              handleSend(
                'Identify all statutory public notices, tenders, and initial public offerings (IPOs).'
              )
            }
            className="bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 px-3 py-1.5 rounded-full transition-colors"
          >
            📜 Public Notices & IPO Listings
          </button>
        </div>
      )}

      {/* Query Input Bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
        className="flex items-center gap-2 pt-2"
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask research questions across historical newspaper issues..."
          disabled={isStreaming}
          className="flex-1 bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-emerald-500 transition-colors disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!query.trim() || isStreaming}
          className="bg-emerald-600 hover:bg-emerald-500 text-white px-5 py-3 rounded-xl text-sm font-semibold flex items-center gap-2 transition-colors disabled:opacity-40 disabled:hover:bg-emerald-600"
        >
          <span>Ask</span>
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
