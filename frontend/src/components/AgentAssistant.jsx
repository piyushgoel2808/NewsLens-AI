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
  Globe,
  GitMerge,
  Compass,
  X,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useActiveHighlight } from '../context/ActiveHighlightContext';
import ModelSelector from './ModelSelector';

// Custom dark-mode theme components for ReactMarkdown
const markdownComponents = {
  h1: ({ children, ...props }) => (
    <h1 className="text-base font-bold text-emerald-300 mt-3 mb-1.5 pb-1 border-b border-slate-800" {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }) => (
    <h2 className="text-sm font-bold text-emerald-400 mt-3 mb-1" {...props}>
      {children}
    </h2>
  ),
  h3: ({ children, ...props }) => (
    <h3 className="text-xs uppercase font-bold tracking-wider text-emerald-300 mt-2.5 mb-1 flex items-center gap-1.5" {...props}>
      {children}
    </h3>
  ),
  h4: ({ children, ...props }) => (
    <h4 className="text-xs font-semibold text-cyan-300 mt-2 mb-1" {...props}>
      {children}
    </h4>
  ),
  p: ({ children, ...props }) => (
    <p className="mb-2 leading-relaxed text-slate-200" {...props}>
      {children}
    </p>
  ),
  blockquote: ({ children, ...props }) => (
    <blockquote className="border-l-2 border-emerald-500/60 bg-emerald-500/10 px-3 py-1.5 rounded-r my-2 text-xs font-medium text-emerald-200 shadow-sm" {...props}>
      {children}
    </blockquote>
  ),
  ul: ({ children, ...props }) => (
    <ul className="list-disc list-outside ml-4 space-y-1 mb-2.5 text-slate-300 text-xs" {...props}>
      {children}
    </ul>
  ),
  ol: ({ children, ...props }) => (
    <ol className="list-decimal list-outside ml-4 space-y-1 mb-2.5 text-slate-300 text-xs" {...props}>
      {children}
    </ol>
  ),
  li: ({ children, ...props }) => (
    <li className="leading-relaxed pl-0.5" {...props}>
      {children}
    </li>
  ),
  strong: ({ children, ...props }) => (
    <strong className="font-semibold text-emerald-300" {...props}>
      {children}
    </strong>
  ),
  em: ({ children, ...props }) => (
    <em className="text-cyan-200 not-italic font-medium" {...props}>
      {children}
    </em>
  ),
  code: ({ inline, className, children, ...props }) =>
    inline ? (
      <code className="bg-slate-900 text-emerald-300 px-1 py-0.5 rounded font-mono text-[11px] border border-slate-800" {...props}>
        {children}
      </code>
    ) : (
      <pre className="bg-slate-900/90 border border-slate-800 p-2.5 rounded-lg overflow-x-auto my-2 text-xs font-mono text-emerald-300">
        <code {...props}>{children}</code>
      </pre>
    ),
  table: ({ children, ...props }) => (
    <div className="overflow-x-auto my-2.5 rounded border border-slate-800">
      <table className="min-w-full text-xs text-left border-collapse" {...props}>
        {children}
      </table>
    </div>
  ),
  thead: ({ children, ...props }) => (
    <thead className="bg-slate-900/90 text-emerald-300 font-semibold border-b border-slate-800" {...props}>
      {children}
    </thead>
  ),
  th: ({ children, ...props }) => (
    <th className="p-2 border-r border-slate-800 last:border-r-0" {...props}>
      {children}
    </th>
  ),
  td: ({ children, ...props }) => (
    <td className="p-2 border-t border-slate-800/60 border-r border-slate-800/60 last:border-r-0 text-slate-300" {...props}>
      {children}
    </td>
  ),
  a: ({ href, children, ...props }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2 transition-colors"
      {...props}
    >
      {children}
    </a>
  ),
};

// Robust separation of reasoning trace vs answer
const splitThoughtAndAnswer = (text) => {
  if (!text) return ['', ''];
  if (text.includes('<think>')) {
    if (text.includes('</think>')) {
      const match = text.match(/<think>([\s\S]*?)<\/think>([\s\S]*)/i);
      if (match) {
        const thought = match[1].trim();
        const ans = match[2].trim();
        if (ans) return [thought, ans];
        const splitMatch = thought.search(
          /\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|In summary|Summary:|Answer:|Draft:|Executive Summary)/i
        );
        if (splitMatch !== -1) {
          return [thought.slice(0, splitMatch).trim(), thought.slice(splitMatch).trim()];
        }
        return [thought, ''];
      }
    } else {
      const parts = text.split('<think>');
      const after = parts.slice(1).join('<think>');
      const splitMatch = after.search(
        /\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|In summary|Summary:|Answer:|Draft:|Executive Summary)/i
      );
      if (splitMatch !== -1) {
        return [
          after.slice(0, splitMatch).trim(),
          after.slice(splitMatch).trim(),
        ];
      }
      return [after.trim(), ''];
    }
  }
  const prefixMatch = text.match(/^(?:Here'?s a thinking process:?|Thinking Process:?|Thought:?)\s*/i);
  if (prefixMatch) {
    const splitMatch = text.search(
      /\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|In summary|Summary:|Answer:|Draft:|Executive Summary)/i
    );
    if (splitMatch !== -1) {
      const thoughtPart = text.slice(0, splitMatch).trim();
      const ansPart = text.slice(splitMatch).replace(/^(?:Draft:?\s*\n*)/i, '').trim();
      return [thoughtPart, ansPart];
    }
    // If no answer section header found yet, the text is purely reasoning
    return [text.trim(), ''];
  }
  return ['', text.trim()];
};

// Helper to strip <think>...</think> and unclosed reasoning tags from main answer text
const sanitizeAnswerText = (text, fallbackThought = '') => {
  if (!text && fallbackThought) {
    const [, ans] = splitThoughtAndAnswer(fallbackThought);
    if (ans) return ans;
    return '';
  }
  if (!text) return '';

  const [th, ans] = splitThoughtAndAnswer(text);
  if (th && ans) {
    return ans;
  }
  if (th && !ans) {
    return '';
  }

  let cleaned = text
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<think>[\s\S]*/gi, '')
    .trim();

  cleaned = cleaned.replace(/^(?:Here'?s a thinking process:?|Thinking Process:?|Thought:?)\s*[\s\S]*?(?=\n#{1,4}\s+|$)/i, '').trim();
  return cleaned;
};

const extractFallbackThought = (text) => {
  if (!text) return '';
  const match = text.match(/<think>([\s\S]*?)(?:<\/think>|$)/i);
  if (match) return match[1].trim();

  const prefixMatch = text.match(/^(?:Here'?s a thinking process:?|Thinking Process:?|Thought:?)\s*/i);
  if (prefixMatch) {
    const [thought] = splitThoughtAndAnswer(text);
    return thought;
  }
  return '';
};

// Helper to extract follow-up exploration questions / angles from the assistant's answer
const extractExplorationPills = (text) => {
  if (!text) return [];
  const pills = [];
  const lines = text.split('\n');
  for (const line of lines) {
    const match = line.match(
      /^\s*(?:>|-|\*|\d+\.)\s*(?:💡\s*)?(?:Explore|Follow-up|Explore Further|Explore angle):\s*(.+)$/i
    );
    if (match && match[1]) {
      const cleanPrompt = match[1].trim().replace(/^["'`]|["'`]$/g, '');
      if (cleanPrompt.length > 5 && !pills.includes(cleanPrompt)) {
        pills.push(cleanPrompt);
      }
    }
  }
  return pills;
};

export default function AgentAssistant() {
  const {
    highlightArticle,
    selectedModel,
    setSelectedModel,
    chatMessages: messages,
    setChatMessages: setMessages,
    openTimeline,
    activeAttachedAsset,
    clearAttachedAsset,
    selectedArticleId,
    selectedIssueId,
    selectedPageNumber,
  } = useActiveHighlight();

  const [query, setQuery] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [enableWebSearch, setEnableWebSearch] = useState(false);

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

    const userMessage = {
      role: 'user',
      content: queryText,
      attachedAsset: activeAttachedAsset ? { ...activeAttachedAsset } : null,
    };
    const assistantMessage = {
      role: 'assistant',
      content: '',
      thought: '',
      thoughtDurationSec: null,
      isThoughtOpen: false,
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
      .filter(
        (m) =>
          !m.isStreaming &&
          m.content &&
          !m.content.startsWith('Hello! I am your **NewsLens-AI')
      )
      .slice(-8)
      .map((m) => ({ role: m.role, content: sanitizeAnswerText(m.content) }));

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
          model: selectedModel || undefined,
          model_override: selectedModel || undefined,
          enable_web_search: enableWebSearch,
          attached_article_id: activeAttachedAsset?.articleId || selectedArticleId || undefined,
          attached_photo_id: activeAttachedAsset?.photoId || undefined,
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
              } else if (eventType === 'thought') {
                current.thought = (current.thought || '') + (data.delta || '');
              } else if (eventType === 'thought_done') {
                current.thought = data.thought || current.thought;
                current.thoughtDurationSec = data.duration_sec;
              } else if (eventType === 'plan') {
                current.archetype = data.archetype;
                current.plan = data.plan || [];
              } else if (eventType === 'tool_result' || eventType === 'tool_results') {
                const toolsToAdd = Array.isArray(data?.tools) ? data.tools : (data ? [data] : []);
                current.toolExecutions = [
                  ...(current.toolExecutions || []),
                  ...toolsToAdd,
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

                // Safety fallback: if content is empty but thought exists, recover answer if an answer was drafted
                if (!current.content && current.thought) {
                  const [th, ans] = splitThoughtAndAnswer(current.thought);
                  if (ans) {
                    current.thought = th;
                    current.content = ans;
                  }
                }
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

  const toggleThought = (index) => {
    setMessages((prev) => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        isThoughtOpen: !updated[index].isThoughtOpen,
      };
      return updated;
    });
  };

  const handleClearChat = () => {
    try {
      localStorage.removeItem('newslens_chat_messages');
    } catch {
      // ignore
    }
    if (typeof highlightArticle === 'function') {
      highlightArticle(null, 1, null, []);
    }
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
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 w-full max-w-7xl 2xl:max-w-[1600px] mx-auto px-4 md:px-8 py-4">
      {/* Top Controls & Model Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-3 border-b border-slate-800 text-xs">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-emerald-400" />
          <span className="font-semibold text-slate-200">Agentic Research Planner & Answerer</span>
        </div>

        <div className="flex items-center gap-2">
          <ModelSelector
            value={selectedModel}
            onChange={setSelectedModel}
            availableModels={availableModels}
          />

          {/* Live Web Search Grounding Toggle */}
          <button
            type="button"
            onClick={() => setEnableWebSearch(!enableWebSearch)}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium border transition-all ${
              enableWebSearch
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 shadow-[0_0_12px_rgba(6,182,212,0.25)] hover:bg-cyan-500/30'
                : 'bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-700 hover:text-slate-300'
            }`}
            title="Toggle live internet search to complement primary newspaper archives"
          >
            <Globe
              className={`w-3.5 h-3.5 ${
                enableWebSearch ? 'text-cyan-400 animate-pulse' : 'text-slate-500'
              }`}
            />
            <span>Web Search:</span>
            <span
              className={`font-semibold ${
                enableWebSearch ? 'text-cyan-300' : 'text-slate-500'
              }`}
            >
              {enableWebSearch ? 'ON' : 'OFF'}
            </span>
          </button>

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
              className={`rounded-xl p-4 md:p-5 ${
                msg.role === 'user'
                  ? 'max-w-[85%] md:max-w-[75%] bg-emerald-600 text-white rounded-br-none shadow-md'
                  : 'w-full bg-slate-900/95 border border-slate-800/80 rounded-bl-none shadow-xl'
              }`}
            >
              {/* User Attached Visual Asset Chip */}
              {msg.role === 'user' && msg.attachedAsset && (
                <div className="mb-2 flex items-center gap-2 text-xs bg-emerald-700/80 border border-emerald-400/40 rounded-lg p-2 text-emerald-100 shadow-sm">
                  {msg.attachedAsset.imageUrl && (
                    <img
                      src={msg.attachedAsset.imageUrl}
                      alt="Attached"
                      className="w-8 h-8 rounded object-cover border border-emerald-300/40 shrink-0 bg-slate-900"
                    />
                  )}
                  <div className="min-w-0">
                    <span className="font-bold text-[11px] block text-white">
                      📊 Attached {msg.attachedAsset.visualType ? msg.attachedAsset.visualType.replace('_', ' ') : 'Visual Asset'} #{msg.attachedAsset.photoId}
                    </span>
                    {msg.attachedAsset.headline && (
                      <span className="text-[10px] text-emerald-200 truncate block">
                        {msg.attachedAsset.headline}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Dynamic Stage Indicator (while assistant is streaming) */}
              {msg.role === 'assistant' && msg.isStreaming && (
                <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-800 text-xs text-emerald-400 font-medium">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span className="capitalize">
                    {msg.stage === 'condensing_query'
                      ? 'Resolving conversational coreference...'
                      : msg.stage === 'planning'
                      ? 'Formulating multi-step investigation plan...'
                      : msg.stage === 'inspecting_visual_asset'
                      ? 'Inspecting visual infographic & extracting numerical data...'
                      : msg.stage === 'web_search'
                      ? 'Searching live web and open sources...'
                      : msg.stage === 'tool_execution'
                      ? 'Executing hybrid vector & SQL retrieval tools...'
                      : msg.stage === 'thinking'
                      ? 'Reasoning and analyzing evidence...'
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

              {/* Collapsible Thought Process (Reasoning block kept outside the output chat) */}
              {msg.role === 'assistant' &&
              (msg.thought || extractFallbackThought(msg.content)) ? (
                <div className="mb-3">
                  <button
                    type="button"
                    onClick={() => toggleThought(idx)}
                    className="flex items-center gap-2 text-xs font-mono text-slate-400 hover:text-slate-200 transition-colors py-1 px-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80 hover:border-slate-700"
                  >
                    <ChevronRight
                      className={`w-3.5 h-3.5 text-purple-400 transition-transform duration-200 ${
                        msg.isThoughtOpen ? 'rotate-90' : ''
                      }`}
                    />
                    <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                    <span className="text-slate-300 font-medium">
                      {msg.thoughtDurationSec
                        ? `Thought for ${msg.thoughtDurationSec}s`
                        : 'Thought Process'}
                    </span>
                    <span className="text-[10px] text-slate-500">
                      {msg.isThoughtOpen ? '(click to collapse)' : '(click to view)'}
                    </span>
                  </button>

                  {msg.isThoughtOpen && (
                    <div className="mt-2 p-3 bg-slate-950/90 border border-purple-500/20 rounded-lg text-xs font-mono text-slate-300 whitespace-pre-wrap leading-relaxed max-h-64 overflow-y-auto shadow-inner">
                      {msg.thought || extractFallbackThought(msg.content)}
                    </div>
                  )}
                </div>
              ) : null}

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
              <div className="text-sm leading-relaxed font-sans text-slate-200">
                {msg.role === 'user' ? (
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                ) : (
                  (() => {
                    const sanitized = sanitizeAnswerText(msg.content, msg.thought);
                    if (sanitized) {
                      return (
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={markdownComponents}
                        >
                          {sanitized}
                        </ReactMarkdown>
                      );
                    }
                    if (msg.isStreaming) {
                      return (
                        <span className="text-slate-400 italic flex items-center gap-2">
                          <RefreshCw className="w-3.5 h-3.5 animate-spin text-emerald-400" />
                          <span>
                            {msg.stage === 'thinking'
                              ? 'Reasoning through evidence...'
                              : msg.stage === 'inspecting_visual_asset'
                              ? 'Inspecting visual infographic & extracting numerical data...'
                              : msg.stage === 'web_search'
                              ? 'Gathering web context...'
                              : 'Synthesizing response...'}
                          </span>
                        </span>
                      );
                    }
                    return <span className="text-slate-500 italic">No response generated.</span>;
                  })()
                )}
              </div>

              {/* Clickable Citations Grounding */}
              {Array.isArray(msg.citations) && msg.citations.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-800">
                  <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block mb-2">
                    Verified Sources & Citations:
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {msg.citations.map((cit, cIdx) => {
                      if (!cit) return null;
                      const isWeb = cit.is_web || cit.source_type === 'web' || !!cit.url;
                      if (isWeb && cit.url) {
                        return (
                          <a
                            key={cIdx}
                            href={cit.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1.5 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 px-2.5 py-1 rounded-md text-xs transition-all hover:scale-105"
                            title={`Open live web source: ${cit.url}`}
                          >
                            <Globe className="w-3 h-3 text-cyan-400 shrink-0" />
                            <span className="font-medium truncate max-w-[220px]">
                              {cit.headline || cit.newspaper_name || 'Web Source'}
                            </span>
                            <ExternalLink className="w-2.5 h-2.5 opacity-70 shrink-0" />
                          </a>
                        );
                      }

                      // Visual Infographic / Chart Citation
                      const isVisual = cit.source_type === 'visual_asset' || Boolean(cit.photo_id);
                      if (isVisual) {
                        const vType = cit.visual_type ? cit.visual_type.replace('_', ' ') : 'infographic';
                        const imgUrl = cit.image_url || (cit.photo_id ? `/api/photos/${cit.photo_id}/image` : null);
                        return (
                          <button
                            key={cIdx}
                            onClick={() =>
                              highlightArticle(
                                cit.issue_id,
                                cit.page_number || 1,
                                cit.article_id,
                                Array.isArray(cit.bboxes) ? cit.bboxes : []
                              )
                            }
                            className="flex items-center gap-2 bg-purple-950/40 hover:bg-purple-900/50 text-purple-300 border border-purple-800/40 px-2.5 py-1 rounded-md text-xs transition-all hover:scale-105 max-w-full"
                            title={`Visual Asset #${cit.photo_id} (${vType}) on Page ${cit.page_number || 1}. Click to view in broadsheet.`}
                          >
                            {imgUrl && (
                              <img
                                src={imgUrl}
                                alt={vType}
                                className="w-5 h-5 object-cover rounded border border-purple-700/50 shrink-0 bg-slate-950"
                                onError={(e) => {
                                  e.currentTarget.style.display = 'none';
                                }}
                              />
                            )}
                            <span className="font-semibold text-purple-200 uppercase text-[11px] shrink-0">
                              📊 {vType} #{cit.photo_id}
                            </span>
                            {cit.headline && (
                              <span className="truncate max-w-[160px] sm:max-w-[220px] text-purple-300/80 font-normal">
                                {cit.headline}
                              </span>
                            )}
                            <ExternalLink className="w-2.5 h-2.5 opacity-60 shrink-0" />
                          </button>
                        );
                      }
                      const pub = cit.newspaper_name || 'Daily News';
                      const pg = cit.page_number || 1;
                      const hl = cit.headline || '';
                      const tooltip = `${pub} (Page ${pg})${hl ? `: "${hl}"` : ''}\nClick to jump and highlight article.`;
                      return (
                        <button
                          key={cIdx}
                          onClick={() =>
                            highlightArticle(
                              cit.issue_id,
                              pg,
                              cit.article_id,
                              Array.isArray(cit.bboxes) ? cit.bboxes : []
                            )
                          }
                          className="flex items-center gap-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2.5 py-1 rounded-md text-xs transition-all hover:scale-105 max-w-full"
                          title={tooltip}
                        >
                          <MapPin className="w-3 h-3 text-emerald-400 shrink-0" />
                          <span className="font-medium shrink-0">
                            {pub} (p.{pg}){hl ? ':' : ''}
                          </span>
                          {hl && (
                            <span className="truncate max-w-[180px] sm:max-w-[260px] text-emerald-300/90 font-normal">
                              {hl}
                            </span>
                          )}
                          <ExternalLink className="w-2.5 h-2.5 opacity-60 shrink-0" />
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Interactive Multi-Newspaper Deep-Dive & Exploration Pills */}
              {msg.role === 'assistant' && !msg.isStreaming && (() => {
                const pills = extractExplorationPills(msg.content);
                if (pills.length === 0) return null;
                return (
                  <div className="mt-3 pt-3 border-t border-slate-800/80">
                    <div className="flex items-center gap-1.5 text-[11px] font-semibold text-emerald-400 uppercase tracking-wider mb-2">
                      <Compass className="w-3.5 h-3.5" />
                      <span>Explore Broadsheet Perspectives:</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {pills.map((pill, pIdx) => (
                        <button
                          key={pIdx}
                          type="button"
                          onClick={() => handleSend(pill)}
                          className="flex items-center gap-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 hover:text-emerald-200 border border-emerald-500/30 hover:border-emerald-400/50 px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:scale-105 shadow-sm text-left group"
                        >
                          <Sparkles className="w-3 h-3 text-emerald-400 group-hover:rotate-12 transition-transform shrink-0" />
                          <span>{pill}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {/* Storyline Trajectory Quick Action */}
              {msg.role === 'assistant' && !msg.isStreaming && (
                <div className="mt-3 pt-2 flex items-center justify-between border-t border-slate-800/80">
                  <button
                    onClick={() => {
                      const userQuery =
                        idx > 0 && messages[idx - 1]?.role === 'user'
                          ? messages[idx - 1].content
                          : query || 'Tata Power clean energy';
                      openTimeline(userQuery);
                    }}
                    className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 transition-all hover:scale-105"
                  >
                    <GitMerge className="w-3.5 h-3.5" />
                    <span>Explore in Storyline Trajectory Canvas</span>
                  </button>

                  {/* Telemetry Summary Footer */}
                  {msg.telemetry && (
                    <div className="flex items-center gap-2 text-[10px] text-slate-500 font-mono">
                      <span>{msg.telemetry.latency_ms}ms</span>
                      <span>•</span>
                      <span>${msg.telemetry.cost_usd.toFixed(4)}</span>
                    </div>
                  )}
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

      {/* Attached Visual Asset Pill Banner (if an asset is attached from Broadsheet Reader) */}
      {activeAttachedAsset && (
        <div className="flex items-center justify-between gap-3 bg-gradient-to-r from-cyan-950/80 via-slate-900/90 to-purple-950/60 border border-cyan-700/50 rounded-xl px-3 py-2 shadow-lg mb-2">
          <div className="flex items-center gap-2.5 overflow-hidden">
            {activeAttachedAsset.imageUrl ? (
              <img
                src={activeAttachedAsset.imageUrl}
                alt="Attached asset"
                className="w-10 h-10 object-cover rounded-lg border border-cyan-800/60 shrink-0 bg-slate-950"
              />
            ) : (
              <div className="w-10 h-10 rounded-lg bg-cyan-950 border border-cyan-800/60 flex items-center justify-center shrink-0 text-cyan-400">
                <Layers className="w-5 h-5" />
              </div>
            )}
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-[10px] uppercase font-mono font-bold px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  Attached {activeAttachedAsset.visualType ? activeAttachedAsset.visualType.replace('_', ' ') : 'Visual Asset'} #{activeAttachedAsset.photoId}
                </span>
                {activeAttachedAsset.articleId && (
                  <span className="text-[10px] text-slate-400 font-mono">
                    Article #{activeAttachedAsset.articleId}
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-200 font-medium truncate max-w-[320px] sm:max-w-[480px] mt-0.5">
                {activeAttachedAsset.headline || activeAttachedAsset.caption || 'Attached visual clipping from broadsheet'}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={clearAttachedAsset}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors shrink-0"
            title="Detach asset"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Query Input Bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
        className="flex items-center gap-2 pt-1"
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={
            activeAttachedAsset
              ? `Ask about this ${activeAttachedAsset.visualType ? activeAttachedAsset.visualType.replace('_', ' ') : 'infographic'} (e.g. "Extract all numbers and explain the trend")...`
              : "Ask research questions across historical newspaper issues..."
          }
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
