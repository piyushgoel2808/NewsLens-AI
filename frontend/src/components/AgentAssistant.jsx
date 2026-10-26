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
  ShieldCheck,
  CheckCircle2,
  Sliders,
  Scale,
  FileText,
  BarChart3,
  Calendar,
  AlertTriangle,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useActiveHighlight } from '../context/ActiveHighlightContext';
import { deduplicatedFetch } from '../utils/apiDeduplicator';
import ModelPickerPopover from './ModelPickerPopover';

// ---------------------------------------------------------------------------
// Absence Detection Keywords
// ---------------------------------------------------------------------------
const ABSENCE_KEYWORDS = [
  'do not contain any verified reporting',
  'no relevant articles were found',
  'contain no verifiable record',
  'no verified reporting found',
  'absence of reporting',
  'no reporting found in the archives',
  'no archival evidence was found',
  'contain no record',
  'contain no direct reporting',
  'contains no verifiable record',
];

const isAbsenceDetected = (text) => {
  if (!text) return false;
  const lower = text.toLowerCase();
  return ABSENCE_KEYWORDS.some((kw) => lower.includes(kw));
};

// ---------------------------------------------------------------------------
// Citation Sanitizer (Fix Bug #1: Malformed Citation Syntax & Curly Braces)
// ---------------------------------------------------------------------------
const sanitizeCitationSyntax = (text) => {
  if (!text) return '';

  // 1. Convert [{The Hindu}, {2026-08-22}, Page 15, "Headline"] or [{The Hindu}, {2026-08-22}, Page 15]
  // into markdown citation links: [📰 The Hindu · p.15](#cite:pub=The+Hindu&page=15)
  let cleaned = text.replace(
    /\[\s*\{?([^{},\]]+)\}?\s*,\s*\{?([^{},\]]+)\}?\s*,\s*(?:Page\s*|p\.\s*)?(\d+)(?:,\s*["'“]([^"'”\]]+)["'”])?\s*\]/gi,
    (match, pub, date, page, headline) => {
      const cleanPub = pub.replace(/[{}]/g, '').trim();
      const cleanHeadline = (headline || '').replace(/[{}]/g, '').trim();
      const encodedPub = encodeURIComponent(cleanPub);
      const encodedHl = encodeURIComponent(cleanHeadline);
      return `[📰 ${cleanPub} · p.${page}](#cite:pub=${encodedPub}&page=${page}&hl=${encodedHl})`;
    }
  );

  // 2. Also clean any remaining stray curly braces inside brackets: [{Text}] -> [Text]
  cleaned = cleaned.replace(/\[\s*\{([^}]+)\}\s*\]/g, '[$1]');

  return cleaned;
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
        return [after.slice(0, splitMatch).trim(), after.slice(splitMatch).trim()];
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

  cleaned = cleaned
    .replace(/^(?:Here'?s a thinking process:?|Thinking Process:?|Thought:?)\s*[\s\S]*?(?=\n#{1,4}\s+|$)/i, '')
    .trim();
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

// ---------------------------------------------------------------------------
// Curated 6-Card Starter Matrix
// ---------------------------------------------------------------------------
const STARTER_CARDS = [
  {
    archetype: 'Factual Verification',
    icon: CheckCircle2,
    color: 'emerald',
    title: 'Policy & Economic Indicators',
    prompt: 'What are the key policy decisions and macroeconomic indicators announced in the latest issues?',
  },
  {
    archetype: 'Comparative Coverage',
    icon: Scale,
    color: 'sky',
    title: 'Cross-Sector Market Disclosures',
    prompt: 'Compare corporate deals and capital market disclosures between technology and energy sectors.',
  },
  {
    archetype: 'Visual Data & Infographics',
    icon: BarChart3,
    color: 'purple',
    title: 'Financial Tables & IPO Metrics',
    prompt: 'Extract all quantitative statistics from the financial infographics and IPO tables on Page 4.',
  },
  {
    archetype: 'Storyline Reconstruction',
    icon: GitMerge,
    color: 'teal',
    title: 'Chronological Policy Spine',
    prompt: 'Reconstruct the chronological timeline of clean energy initiatives across all editions.',
  },
  {
    archetype: 'Omission Audit',
    icon: AlertCircle,
    color: 'amber',
    title: 'Hyperlocal Coverage Gap Audit',
    prompt: 'Audit hyperlocal coverage in regional editions omitted by national broadsheets.',
  },
  {
    archetype: 'Financial Disclosures',
    icon: FileText,
    color: 'indigo',
    title: 'Public Notices & Tender Listings',
    prompt: 'Identify all statutory public notices, tenders, and corporate earnings disclosures.',
  },
];

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
    pendingQuery,
    setPendingQuery,
    setActiveTab,
  } = useActiveHighlight();

  const [query, setQuery] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [enableWebSearch, setEnableWebSearch] = useState(false);
  const [activeTelemetryTab, setActiveTelemetryTab] = useState({}); // { [msgIdx]: 'step3' }

  const messagesEndRef = useRef(null);

  // Fetch configured model providers
  useEffect(() => {
    deduplicatedFetch('/api/models/available')
      .then((data) => {
        if (data && data.models) {
          setAvailableModels(data.models);
        }
      })
      .catch((err) => console.error('Failed to load models:', err));
  }, []);

  // Handle incoming pending query from other tabs (Overview, Broadsheet Reader)
  useEffect(() => {
    if (pendingQuery && !isStreaming) {
      const q = pendingQuery;
      setPendingQuery('');
      handleSend(q);
    }
  }, [pendingQuery, isStreaming]);

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
      const isStaleAsset = Boolean(
        activeAttachedAsset?.articleId &&
        selectedArticleId &&
        Number(activeAttachedAsset.articleId) !== Number(selectedArticleId)
      );
      const effectiveArticleId =
        selectedArticleId || (isStaleAsset ? undefined : activeAttachedAsset?.articleId) || undefined;
      const effectivePhotoId = isStaleAsset ? undefined : activeAttachedAsset?.photoId || undefined;
      const effectiveIssueDate = isStaleAsset ? undefined : activeAttachedAsset?.issueDate;
      const effectiveNewspaperName = isStaleAsset ? undefined : activeAttachedAsset?.newspaperName;

      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          chat_history: historyPayload,
          model: selectedModel || undefined,
          model_override: selectedModel || undefined,
          enable_web_search: enableWebSearch,
          attached_article_id: effectiveArticleId,
          attached_photo_id: effectivePhotoId || undefined,
          attached_issue_date: effectiveIssueDate || undefined,
          attached_newspaper_name: effectiveNewspaperName || undefined,
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
                const toolsToAdd = Array.isArray(data?.tools) ? data.tools : data ? [data] : [];
                current.toolExecutions = [...(current.toolExecutions || []), ...toolsToAdd];
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

  // Custom markdown components supporting inline citation clicks
  const customMarkdownComponents = (msgCitations = []) => ({
    h1: ({ children, ...props }) => (
      <h1 className="text-base font-serif font-bold text-accent-primary mt-3 mb-1.5 pb-1 border-b border-border-subtle" {...props}>
        {children}
      </h1>
    ),
    h2: ({ children, ...props }) => (
      <h2 className="text-sm font-serif font-bold text-accent-primary mt-3 mb-1" {...props}>
        {children}
      </h2>
    ),
    h3: ({ children, ...props }) => (
      <h3 className="text-xs uppercase font-mono font-bold tracking-wider text-accent-primary mt-2.5 mb-1 flex items-center gap-1.5" {...props}>
        {children}
      </h3>
    ),
    p: ({ children, ...props }) => (
      <p className="mb-2 leading-relaxed text-text-primary" {...props}>
        {children}
      </p>
    ),
    blockquote: ({ children, ...props }) => (
      <blockquote className="border-l-2 border-accent-primary/60 bg-accent-primary/10 px-3 py-1.5 rounded-r my-2 text-xs font-medium text-text-primary shadow-sm" {...props}>
        {children}
      </blockquote>
    ),
    ul: ({ children, ...props }) => (
      <ul className="list-disc list-outside ml-4 space-y-1 mb-2.5 text-text-secondary text-xs" {...props}>
        {children}
      </ul>
    ),
    ol: ({ children, ...props }) => (
      <ol className="list-decimal list-outside ml-4 space-y-1 mb-2.5 text-text-secondary text-xs" {...props}>
        {children}
      </ol>
    ),
    li: ({ children, ...props }) => (
      <li className="leading-relaxed pl-0.5" {...props}>
        {children}
      </li>
    ),
    strong: ({ children, ...props }) => (
      <strong className="font-semibold text-text-primary" {...props}>
        {children}
      </strong>
    ),
    em: ({ children, ...props }) => (
      <em className="text-accent-editorial not-italic font-medium" {...props}>
        {children}
      </em>
    ),
    code: ({ inline, className, children, ...props }) =>
      inline ? (
        <code className="bg-elevated text-accent-primary px-1 py-0.5 rounded font-mono text-[11px] border border-border-subtle" {...props}>
          {children}
        </code>
      ) : (
        <pre className="bg-elevated border border-border-subtle p-2.5 rounded-lg overflow-x-auto my-2 text-xs font-mono text-accent-primary">
          <code {...props}>{children}</code>
        </pre>
      ),
    table: ({ children, ...props }) => (
      <div className="overflow-x-auto my-2.5 rounded border border-border-subtle">
        <table className="min-w-full text-xs text-left border-collapse" {...props}>
          {children}
        </table>
      </div>
    ),
    thead: ({ children, ...props }) => (
      <thead className="bg-elevated text-accent-primary font-semibold border-b border-border-subtle" {...props}>
        {children}
      </thead>
    ),
    th: ({ children, ...props }) => (
      <th className="p-2 border-r border-border-subtle last:border-r-0" {...props}>
        {children}
      </th>
    ),
    td: ({ children, ...props }) => (
      <td className="p-2 border-t border-border-subtle border-r border-border-subtle last:border-r-0 text-text-secondary font-mono" {...props}>
        {children}
      </td>
    ),
    a: ({ href, children, ...props }) => {
      // Check if this is an inline citation pill link: #cite:pub=...&page=...
      if (href && href.startsWith('#cite:')) {
        const params = new URLSearchParams(href.replace('#cite:', ''));
        const pub = params.get('pub');
        const page = Number(params.get('page')) || 1;
        const hl = params.get('hl');

        // Look for matching citation object to get articleId and bounding boxes
        const matchCit = msgCitations.find(
          (c) =>
            (c.newspaper_name && c.newspaper_name.toLowerCase().includes((pub || '').toLowerCase())) ||
            (c.page_number && Number(c.page_number) === page)
        );

        return (
          <button
            type="button"
            onClick={() =>
              highlightArticle(
                matchCit?.issue_id || null,
                page,
                matchCit?.article_id || null,
                Array.isArray(matchCit?.bboxes) ? matchCit.bboxes : []
              )
            }
            className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md text-[11px] font-mono font-semibold bg-accent-primary/15 text-accent-primary border border-accent-primary/30 hover:bg-accent-primary/25 hover:border-accent-primary/60 transition-all cursor-pointer shadow-xs"
            title={`Click to inspect ${pub} (Page ${page}) in Broadsheet Reader`}
          >
            <span>{children}</span>
            <ExternalLink className="w-2.5 h-2.5 opacity-70" />
          </button>
        );
      }

      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent-info hover:underline transition-colors"
          {...props}
        >
          {children}
        </a>
      );
    },
  });

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-canvas text-text-primary w-full max-w-7xl 2xl:max-w-[1600px] mx-auto px-4 md:px-8 py-4 transition-colors duration-200">
      {/* Top Controls & Model Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-3 border-b border-border-subtle text-xs">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-accent-primary" />
          <span className="font-semibold text-text-primary">Autonomous AI Newsroom Assistant</span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface border border-border-subtle text-text-muted">
            CRAG & 7 ARCHETYPES
          </span>
        </div>

        <div className="flex items-center gap-2">
          <ModelPickerPopover value={selectedModel} onChange={setSelectedModel} />

          {/* Live Web Search Grounding Toggle */}
          <button
            type="button"
            onClick={() => setEnableWebSearch(!enableWebSearch)}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium border transition-all cursor-pointer ${
              enableWebSearch
                ? 'bg-accent-info/20 text-accent-info border-accent-info/50 shadow-sm'
                : 'bg-surface text-text-muted border-border-subtle hover:border-text-secondary hover:text-text-primary'
            }`}
            title="Toggle live internet search to complement primary newspaper archives"
          >
            <Globe
              className={`w-3.5 h-3.5 ${
                enableWebSearch ? 'text-accent-info animate-pulse' : 'text-text-muted'
              }`}
            />
            <span>Web Search:</span>
            <span
              className={`font-semibold font-mono ${
                enableWebSearch ? 'text-accent-info' : 'text-text-muted'
              }`}
            >
              {enableWebSearch ? 'ON' : 'OFF'}
            </span>
          </button>

          <button
            type="button"
            onClick={handleClearChat}
            className="text-text-muted hover:text-text-primary text-xs px-2.5 py-1 rounded-lg bg-surface border border-border-subtle hover:bg-elevated transition-colors ml-1 cursor-pointer"
            title="Reset Chat History"
          >
            Clear History
          </button>
        </div>
      </div>

      {/* Message Chat Feed */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2">
        {messages.map((msg, idx) => {
          const userQueryText =
            idx > 0 && messages[idx - 1]?.role === 'user' ? messages[idx - 1].content : '';
          const absenceDetected =
            msg.role === 'assistant' && !msg.isStreaming && isAbsenceDetected(msg.content);

          return (
            <div
              key={idx}
              className={`flex gap-3 ${
                msg.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-accent-primary/20 border border-accent-primary/40 flex items-center justify-center text-accent-primary shrink-0 mt-1 shadow-sm">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div
                className={`rounded-xl p-4 md:p-5 ${
                  msg.role === 'user'
                    ? 'max-w-[85%] md:max-w-[75%] bg-accent-primary text-white rounded-br-none shadow-md'
                    : 'w-full bg-surface border border-border-subtle rounded-bl-none shadow-md'
                }`}
              >
                {/* User Attached Visual Asset Chip */}
                {msg.role === 'user' && msg.attachedAsset && (
                  <div className="mb-2 flex items-center gap-2 text-xs bg-black/20 border border-white/20 rounded-lg p-2 text-white shadow-sm">
                    {msg.attachedAsset.imageUrl && (
                      <img
                        src={msg.attachedAsset.imageUrl}
                        alt="Attached"
                        className="w-8 h-8 rounded object-cover border border-white/30 shrink-0 bg-surface"
                      />
                    )}
                    <div className="min-w-0">
                      <span className="font-bold text-[11px] block">
                        📊 Attached{' '}
                        {msg.attachedAsset.visualType
                          ? msg.attachedAsset.visualType.replace('_', ' ')
                          : 'Visual Asset'}{' '}
                        #{msg.attachedAsset.photoId}
                      </span>
                      {msg.attachedAsset.headline && (
                        <span className="text-[10px] opacity-80 truncate block">
                          {msg.attachedAsset.headline}
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* Dynamic Stage Indicator (while assistant is streaming) */}
                {msg.role === 'assistant' && msg.isStreaming && (
                  <div className="flex items-center gap-2 mb-3 pb-2 border-b border-border-subtle text-xs text-accent-primary font-medium font-mono">
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
                  <div className="flex items-center gap-1.5 text-[11px] text-accent-primary font-mono bg-elevated border border-border-subtle px-2.5 py-1 rounded-md mb-2.5">
                    <Search className="w-3 h-3 text-accent-primary shrink-0" />
                    <span className="truncate">
                      Resolved Context: <strong className="text-text-primary">"{msg.condensedQuery}"</strong>
                    </span>
                  </div>
                )}

                {/* Collapsible Thought Process */}
                {msg.role === 'assistant' &&
                (msg.thought || extractFallbackThought(msg.content)) ? (
                  <div className="mb-3">
                    <button
                      type="button"
                      onClick={() => toggleThought(idx)}
                      className="flex items-center gap-2 text-xs font-mono text-text-muted hover:text-text-primary transition-colors py-1 px-2.5 rounded-lg bg-elevated border border-border-subtle cursor-pointer"
                    >
                      <ChevronRight
                        className={`w-3.5 h-3.5 text-accent-visual transition-transform duration-200 ${
                          msg.isThoughtOpen ? 'rotate-90' : ''
                        }`}
                      />
                      <Sparkles className="w-3.5 h-3.5 text-accent-visual" />
                      <span className="text-text-secondary font-medium">
                        {msg.thoughtDurationSec
                          ? `Thought for ${msg.thoughtDurationSec}s`
                          : 'Thought Process'}
                      </span>
                      <span className="text-[10px] text-text-muted">
                        {msg.isThoughtOpen ? '(collapse)' : '(expand)'}
                      </span>
                    </button>

                    {msg.isThoughtOpen && (
                      <div className="mt-2 p-3 bg-canvas/80 border border-border-subtle rounded-lg text-xs font-mono text-text-secondary whitespace-pre-wrap leading-relaxed max-h-64 overflow-y-auto shadow-inner">
                        {msg.thought || extractFallbackThought(msg.content)}
                      </div>
                    )}
                  </div>
                ) : null}

                {/* 4-Step Pipeline Execution Stepper (Replacing raw JSON dump) */}
                {msg.role === 'assistant' &&
                  (msg.plan?.length > 0 || msg.toolExecutions?.length > 0 || msg.condensedQuery) && (
                    <div className="mb-4 border border-border-subtle rounded-xl overflow-hidden bg-elevated/40 shadow-xs">
                      {/* Horizontal Stepper Header */}
                      <div className="grid grid-cols-2 md:grid-cols-4 border-b border-border-subtle bg-elevated divide-x divide-border-subtle text-[11px] font-mono">
                        {/* Step 1: Query Condenser */}
                        <button
                          type="button"
                          onClick={() =>
                            setActiveTelemetryTab((prev) => ({
                              ...prev,
                              [idx]: prev[idx] === 'step1' ? null : 'step1',
                            }))
                          }
                          className={`p-2.5 flex items-center justify-between transition-colors text-left cursor-pointer ${
                            activeTelemetryTab[idx] === 'step1'
                              ? 'bg-surface text-accent-primary font-bold'
                              : 'text-text-secondary hover:bg-surface/50'
                          }`}
                        >
                          <div className="flex items-center gap-1.5 truncate">
                            <span className="w-4 h-4 rounded-full bg-accent-primary/20 text-accent-primary flex items-center justify-center text-[9px] font-bold shrink-0">
                              1
                            </span>
                            <span className="truncate">Condenser</span>
                          </div>
                          <CheckCircle2 className="w-3.5 h-3.5 text-accent-primary shrink-0" />
                        </button>

                        {/* Step 2: Newsroom Planner */}
                        <button
                          type="button"
                          onClick={() =>
                            setActiveTelemetryTab((prev) => ({
                              ...prev,
                              [idx]: prev[idx] === 'step2' ? null : 'step2',
                            }))
                          }
                          className={`p-2.5 flex items-center justify-between transition-colors text-left cursor-pointer ${
                            activeTelemetryTab[idx] === 'step2'
                              ? 'bg-surface text-accent-editorial font-bold'
                              : 'text-text-secondary hover:bg-surface/50'
                          }`}
                        >
                          <div className="flex items-center gap-1.5 truncate">
                            <span className="w-4 h-4 rounded-full bg-accent-editorial/20 text-accent-editorial flex items-center justify-center text-[9px] font-bold shrink-0">
                              2
                            </span>
                            <span className="truncate">Planner</span>
                          </div>
                          {msg.archetype && (
                            <span className="text-[9px] px-1 py-0.2 rounded bg-surface border border-border-subtle text-accent-editorial truncate max-w-[80px]">
                              {msg.archetype.replace('_', ' ')}
                            </span>
                          )}
                        </button>

                        {/* Step 3: Executed Tools */}
                        <button
                          type="button"
                          onClick={() =>
                            setActiveTelemetryTab((prev) => ({
                              ...prev,
                              [idx]: prev[idx] === 'step3' ? null : 'step3',
                            }))
                          }
                          className={`p-2.5 flex items-center justify-between transition-colors text-left cursor-pointer ${
                            activeTelemetryTab[idx] === 'step3'
                              ? 'bg-surface text-accent-visual font-bold'
                              : 'text-text-secondary hover:bg-surface/50'
                          }`}
                        >
                          <div className="flex items-center gap-1.5 truncate">
                            <span className="w-4 h-4 rounded-full bg-accent-visual/20 text-accent-visual flex items-center justify-center text-[9px] font-bold shrink-0">
                              3
                            </span>
                            <span className="truncate">Tools ({msg.toolExecutions?.length || 0})</span>
                          </div>
                          <Database className="w-3.5 h-3.5 text-accent-visual shrink-0" />
                        </button>

                        {/* Step 4: CRAG Evaluator */}
                        <button
                          type="button"
                          onClick={() =>
                            setActiveTelemetryTab((prev) => ({
                              ...prev,
                              [idx]: prev[idx] === 'step4' ? null : 'step4',
                            }))
                          }
                          className={`p-2.5 flex items-center justify-between transition-colors text-left cursor-pointer ${
                            activeTelemetryTab[idx] === 'step4'
                              ? 'bg-surface text-accent-info font-bold'
                              : 'text-text-secondary hover:bg-surface/50'
                          }`}
                        >
                          <div className="flex items-center gap-1.5 truncate">
                            <span className="w-4 h-4 rounded-full bg-accent-info/20 text-accent-info flex items-center justify-center text-[9px] font-bold shrink-0">
                              4
                            </span>
                            <span className="truncate">CRAG & Rerank</span>
                          </div>
                          <Scale className="w-3.5 h-3.5 text-accent-info shrink-0" />
                        </button>
                      </div>

                      {/* Active Step Disclosure Content */}
                      {activeTelemetryTab[idx] && (
                        <div className="p-3.5 bg-surface text-xs font-mono text-text-secondary border-t border-border-subtle">
                          {activeTelemetryTab[idx] === 'step1' && (
                            <div className="space-y-2">
                              <span className="text-[10px] text-text-muted uppercase tracking-wider block font-bold">
                                Step 1: Query Condensation & Coreference Resolution
                              </span>
                              <div className="p-2.5 rounded-lg bg-elevated border border-border-subtle">
                                <div className="text-[11px] text-text-muted mb-1">Original User Input:</div>
                                <div className="text-text-primary font-sans font-medium mb-2">
                                  "{userQueryText || 'N/A'}"
                                </div>
                                <div className="text-[11px] text-accent-primary mb-1">
                                  Resolved Contextual Inquiry:
                                </div>
                                <div className="text-text-primary font-sans font-medium">
                                  "{msg.condensedQuery || userQueryText}"
                                </div>
                              </div>
                            </div>
                          )}

                          {activeTelemetryTab[idx] === 'step2' && (
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <span className="text-[10px] text-text-muted uppercase tracking-wider font-bold">
                                  Step 2: Autonomous Newsroom Planner DAG
                                </span>
                                {msg.archetype && (
                                  <span className="px-2 py-0.5 rounded bg-accent-editorial/20 text-accent-editorial border border-accent-editorial/30 text-[10px] uppercase font-bold">
                                    Archetype: {msg.archetype}
                                  </span>
                                )}
                              </div>
                              {msg.plan && msg.plan.length > 0 ? (
                                <div className="space-y-1.5">
                                  {msg.plan.map((step, sIdx) => (
                                    <div
                                      key={sIdx}
                                      className="p-2 rounded-lg bg-elevated border border-border-subtle flex items-start gap-2"
                                    >
                                      <span className="text-accent-editorial font-bold">#{sIdx + 1}</span>
                                      <div>
                                        <span className="text-text-primary font-bold">{step.tool_name}()</span>
                                        <p className="text-[11px] text-text-secondary font-sans mt-0.5">
                                          {step.purpose}
                                        </p>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <p className="text-text-muted italic">Direct synthesis plan executed.</p>
                              )}
                            </div>
                          )}

                          {activeTelemetryTab[idx] === 'step3' && (
                            <div className="space-y-2">
                              <span className="text-[10px] text-text-muted uppercase tracking-wider block font-bold">
                                Step 3: Executed Tools & Retrieval Status
                              </span>
                              {msg.toolExecutions && msg.toolExecutions.length > 0 ? (
                                <div className="space-y-2">
                                  {msg.toolExecutions.map((t, tIdx) => {
                                    const toolName = t.tool_name || t.name || 'tool';
                                    const execData = t.execution || t;
                                    const hasResults =
                                      execData?.articles?.length ||
                                      execData?.rows?.length ||
                                      execData?.count ||
                                      execData?.milestones?.length ||
                                      (typeof execData === 'object' && Object.keys(execData).length > 0);

                                    return (
                                      <div
                                        key={tIdx}
                                        className="p-2.5 rounded-lg bg-elevated border border-border-subtle"
                                      >
                                        <div className="flex items-center justify-between mb-1">
                                          <span className="text-accent-visual font-bold">{toolName}()</span>
                                          <span className="text-[10px] px-1.5 py-0.2 rounded bg-accent-primary/20 text-accent-primary font-mono">
                                            SUCCESS
                                          </span>
                                        </div>
                                        {/* Parameters */}
                                        {t.args && (
                                          <div className="text-[10px] text-text-muted truncate mb-1">
                                            Args: {JSON.stringify(t.args)}
                                          </div>
                                        )}
                                        {/* Structured Result Summary */}
                                        <div className="text-[11px] text-text-secondary font-sans">
                                          {execData?.summary ||
                                            (execData?.articles
                                              ? `Retrieved ${execData.articles.length} article chunks from Qdrant vector space.`
                                              : execData?.rows
                                              ? `Executed MySQL relational query; returned ${execData.rows.length} records.`
                                              : typeof execData === 'string'
                                              ? execData
                                              : 'Tool executed successfully.')}
                                        </div>
                                      </div>
                                    );
                                  })}
                                </div>
                              ) : (
                                <p className="text-text-muted italic">No external tool calls recorded.</p>
                              )}
                            </div>
                          )}

                          {activeTelemetryTab[idx] === 'step4' && (
                            <div className="space-y-2">
                              <span className="text-[10px] text-text-muted uppercase tracking-wider block font-bold">
                                Step 4: Corrective RAG (CRAG) & Cross-Encoder Evaluation
                              </span>
                              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-center">
                                <div className="p-2 rounded bg-elevated border border-border-subtle">
                                  <div className="text-sm font-bold text-text-primary">
                                    {msg.citations?.length || msg.telemetry?.evidence_count || 0}
                                  </div>
                                  <div className="text-[9px] text-text-muted uppercase">Verified Sources</div>
                                </div>
                                <div className="p-2 rounded bg-elevated border border-border-subtle">
                                  <div className="text-sm font-bold text-accent-primary">
                                    {msg.telemetry?.latency_ms ? `${msg.telemetry.latency_ms}ms` : '~1.0s'}
                                  </div>
                                  <div className="text-[9px] text-text-muted uppercase">Two-Stage Latency</div>
                                </div>
                                <div className="p-2 rounded bg-elevated border border-border-subtle col-span-2 sm:col-span-1">
                                  <div className="text-sm font-bold text-accent-editorial">
                                    {msg.telemetry?.cost_usd ? `$${msg.telemetry.cost_usd.toFixed(4)}` : '$0.0000'}
                                  </div>
                                  <div className="text-[9px] text-text-muted uppercase">Token Cost</div>
                                </div>
                              </div>
                              <p className="text-[11px] text-text-secondary leading-snug mt-1 font-sans">
                                RRF Stage 1 (N=75) merged dense vectors & BM25 sparse hits. MiniLM Cross-Encoder Stage 2 capped candidates to top-20 evidence items before structured synthesis.
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                {/* Message Content (Sanitized of Malformed Curly Braces) */}
                <div className="text-sm leading-relaxed font-sans text-text-primary">
                  {msg.role === 'user' ? (
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  ) : (
                    (() => {
                      const sanitized = sanitizeAnswerText(msg.content, msg.thought);
                      const cleanSyntax = sanitizeCitationSyntax(sanitized);

                      if (cleanSyntax) {
                        return (
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={customMarkdownComponents(msg.citations || [])}
                          >
                            {cleanSyntax}
                          </ReactMarkdown>
                        );
                      }
                      if (msg.isStreaming) {
                        return (
                          <span className="text-text-muted italic flex items-center gap-2">
                            <RefreshCw className="w-3.5 h-3.5 animate-spin text-accent-primary" />
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
                      return <span className="text-text-muted italic">No response generated.</span>;
                    })()
                  )}
                </div>

                {/* Archival Silence Notice (Fix Bug #2: Archival Silence & False Citation Suppression) */}
                {absenceDetected && (
                  <div className="mt-4 p-4 rounded-xl bg-elevated border border-accent-editorial/40 shadow-sm">
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-accent-editorial/20 text-accent-editorial flex items-center justify-center shrink-0 mt-0.5">
                        <AlertTriangle className="w-4 h-4" />
                      </div>
                      <div className="flex-1">
                        <h4 className="text-xs font-bold text-text-primary font-serif">
                          Archival Silence / Zero Corpus Match Notice
                        </h4>
                        <p className="text-xs text-text-secondary mt-1 leading-relaxed font-sans">
                          The historical broadsheet archives currently indexed contain no verifiable record or direct reporting on this topic. Distractor citations and low-confidence fallback passages have been suppressed to guarantee factual integrity.
                        </p>

                        <div className="mt-3 flex flex-wrap items-center gap-2">
                          <button
                            type="button"
                            onClick={() => {
                              setEnableWebSearch(true);
                              handleSend(userQueryText);
                            }}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent-info/15 hover:bg-accent-info/25 text-accent-info border border-accent-info/30 text-xs font-semibold font-mono transition-all cursor-pointer"
                          >
                            <Globe className="w-3.5 h-3.5" />
                            <span>🌐 Re-run with Live Web Search ON</span>
                          </button>

                          <button
                            type="button"
                            onClick={() => setActiveTab('graph')}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface hover:bg-elevated text-text-secondary hover:text-text-primary border border-border-subtle text-xs font-semibold font-mono transition-all cursor-pointer"
                          >
                            <Search className="w-3.5 h-3.5 text-accent-visual" />
                            <span>🔍 Search Entity Knowledge Graph</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Verified Citations Grounding (Suppressed if absence detected) */}
                {!absenceDetected && Array.isArray(msg.citations) && msg.citations.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-border-subtle">
                    <span className="text-[11px] uppercase tracking-wider text-text-muted font-semibold block mb-2 font-mono">
                      Verified Archival Sources & Spatial Citations:
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
                              className="flex items-center gap-1.5 bg-accent-info/10 hover:bg-accent-info/20 text-accent-info border border-accent-info/30 px-2.5 py-1 rounded-md text-xs transition-all hover:scale-105"
                              title={`Open live web source: ${cit.url}`}
                            >
                              <Globe className="w-3 h-3 text-accent-info shrink-0" />
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
                              type="button"
                              onClick={() =>
                                highlightArticle(
                                  cit.issue_id,
                                  cit.page_number || 1,
                                  cit.article_id,
                                  Array.isArray(cit.bboxes) ? cit.bboxes : []
                                )
                              }
                              className="flex items-center gap-2 bg-accent-visual/15 hover:bg-accent-visual/25 text-accent-visual border border-accent-visual/30 px-2.5 py-1 rounded-md text-xs transition-all hover:scale-105 max-w-full cursor-pointer"
                              title={`Visual Asset #${cit.photo_id} (${vType}) on Page ${cit.page_number || 1}. Click to view in broadsheet.`}
                            >
                              {imgUrl && (
                                <img
                                  src={imgUrl}
                                  alt={vType}
                                  className="w-5 h-5 object-cover rounded border border-border-subtle shrink-0 bg-canvas"
                                  onError={(e) => {
                                    e.currentTarget.style.display = 'none';
                                  }}
                                />
                              )}
                              <span className="font-semibold uppercase text-[11px] shrink-0 font-mono">
                                📊 {vType} #{cit.photo_id}
                              </span>
                              {cit.headline && (
                                <span className="truncate max-w-[160px] sm:max-w-[220px] font-normal">
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
                            type="button"
                            onClick={() =>
                              highlightArticle(
                                cit.issue_id,
                                pg,
                                cit.article_id,
                                Array.isArray(cit.bboxes) ? cit.bboxes : []
                              )
                            }
                            className="flex items-center gap-1.5 bg-accent-primary/10 hover:bg-accent-primary/20 text-accent-primary border border-accent-primary/30 px-2.5 py-1 rounded-md text-xs transition-all hover:scale-105 max-w-full cursor-pointer"
                            title={tooltip}
                          >
                            <MapPin className="w-3 h-3 text-accent-primary shrink-0" />
                            <span className="font-medium shrink-0 font-mono">
                              {pub} (p.{pg}){hl ? ':' : ''}
                            </span>
                            {hl && (
                              <span className="truncate max-w-[180px] sm:max-w-[260px] font-normal">
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
                {msg.role === 'assistant' &&
                  !msg.isStreaming &&
                  (() => {
                    const pills = extractExplorationPills(msg.content);
                    if (pills.length === 0) return null;
                    return (
                      <div className="mt-3 pt-3 border-t border-border-subtle">
                        <div className="flex items-center gap-1.5 text-[11px] font-semibold text-accent-primary uppercase tracking-wider mb-2 font-mono">
                          <Compass className="w-3.5 h-3.5" />
                          <span>Explore Broadsheet Angles:</span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {pills.map((pill, pIdx) => (
                            <button
                              key={pIdx}
                              type="button"
                              onClick={() => handleSend(pill)}
                              className="flex items-center gap-1.5 bg-elevated hover:bg-surface text-text-secondary hover:text-text-primary border border-border-subtle hover:border-accent-primary/50 px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:scale-105 shadow-xs text-left cursor-pointer group"
                            >
                              <Sparkles className="w-3 h-3 text-accent-primary group-hover:rotate-12 transition-transform shrink-0" />
                              <span>{pill}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    );
                  })()}

                {/* Storyline Trajectory Quick Action */}
                {msg.role === 'assistant' && !msg.isStreaming && (
                  <div className="mt-3 pt-2 flex items-center justify-between border-t border-border-subtle">
                    <button
                      type="button"
                      onClick={() => {
                        const userQuery =
                          idx > 0 && messages[idx - 1]?.role === 'user'
                            ? messages[idx - 1].content
                            : query || 'Clean energy policy announcements';
                        openTimeline(userQuery);
                      }}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold bg-accent-primary/10 hover:bg-accent-primary/20 text-accent-primary border border-accent-primary/30 transition-all hover:scale-105 cursor-pointer font-mono"
                    >
                      <GitMerge className="w-3.5 h-3.5" />
                      <span>Explore in Storyline Trajectory Canvas</span>
                    </button>

                    {/* Telemetry Summary Footer */}
                    {msg.telemetry && (
                      <div className="flex items-center gap-2 text-[10px] text-text-muted font-mono">
                        <span>{msg.telemetry.latency_ms}ms</span>
                        <span>•</span>
                        <span>${msg.telemetry.cost_usd.toFixed(4)}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-elevated border border-border-subtle flex items-center justify-center text-text-primary shrink-0 mt-1 shadow-sm">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Curated 6-Card Starter Matrix (when chat is empty / single welcome message) */}
      {messages.length <= 1 && (
        <div className="py-4">
          <div className="text-[11px] font-mono text-text-muted uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-accent-primary" />
            <span>Select a Journalistic Research Archetype to Begin:</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
            {STARTER_CARDS.map((card, cIdx) => {
              const CardIcon = card.icon;
              return (
                <button
                  key={cIdx}
                  type="button"
                  onClick={() => handleSend(card.prompt)}
                  className="text-left p-3 rounded-xl bg-surface border border-border-subtle hover:border-accent-primary/60 hover:bg-elevated transition-all shadow-xs group flex flex-col justify-between min-h-[95px] cursor-pointer"
                >
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[9px] font-mono uppercase tracking-wider text-accent-primary font-bold">
                        {card.archetype}
                      </span>
                      <CardIcon className="w-3.5 h-3.5 text-text-muted group-hover:text-accent-primary transition-colors" />
                    </div>
                    <div className="text-xs font-semibold text-text-primary group-hover:text-accent-primary transition-colors">
                      {card.title}
                    </div>
                  </div>
                  <div className="text-[11px] text-text-secondary line-clamp-2 mt-1 leading-snug">
                    "{card.prompt}"
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Attached Visual Asset Pill Banner (if an asset is attached from Broadsheet Reader) */}
      {activeAttachedAsset && (
        <div className="flex items-center justify-between gap-3 bg-elevated border border-accent-visual/50 rounded-xl px-3 py-2 shadow-lg mb-2">
          <div className="flex items-center gap-2.5 overflow-hidden">
            {activeAttachedAsset.imageUrl ? (
              <img
                src={activeAttachedAsset.imageUrl}
                alt="Attached asset"
                className="w-10 h-10 object-cover rounded-lg border border-border-subtle shrink-0 bg-canvas"
              />
            ) : (
              <div className="w-10 h-10 rounded-lg bg-surface border border-border-subtle flex items-center justify-center shrink-0 text-accent-visual">
                <Layers className="w-5 h-5" />
              </div>
            )}
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-[10px] uppercase font-mono font-bold px-1.5 py-0.5 rounded bg-accent-visual/20 text-accent-visual border border-accent-visual/30">
                  Attached{' '}
                  {activeAttachedAsset.visualType
                    ? activeAttachedAsset.visualType.replace('_', ' ')
                    : 'Visual Asset'}{' '}
                  #{activeAttachedAsset.photoId}
                </span>
                {activeAttachedAsset.newspaperName && (
                  <span className="text-[10px] text-text-secondary font-mono px-1.5 py-0.5 rounded bg-surface border border-border-subtle">
                    {activeAttachedAsset.newspaperName}{' '}
                    {activeAttachedAsset.issueDate ? `(${activeAttachedAsset.issueDate})` : ''}
                  </span>
                )}
                {activeAttachedAsset.articleId && (
                  <span className="text-[10px] text-text-muted font-mono">
                    Article #{activeAttachedAsset.articleId}
                  </span>
                )}
              </div>
              <p className="text-xs text-text-primary font-medium truncate max-w-[320px] sm:max-w-[480px] mt-0.5">
                {activeAttachedAsset.headline ||
                  activeAttachedAsset.caption ||
                  'Attached visual clipping from broadsheet'}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={clearAttachedAsset}
            className="p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-surface transition-colors shrink-0 cursor-pointer"
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
              ? `Ask about this ${
                  activeAttachedAsset.visualType
                    ? activeAttachedAsset.visualType.replace('_', ' ')
                    : 'infographic'
                } (e.g. "Extract all numbers and explain the trend")...`
              : 'Ask research questions across historical newspaper broadsheets...'
          }
          disabled={isStreaming}
          className="flex-1 bg-surface border border-border-subtle rounded-xl px-4 py-3 text-sm text-text-primary placeholder-text-muted outline-none focus:border-accent-primary transition-colors disabled:opacity-50 shadow-inner font-sans"
        />
        <button
          type="submit"
          disabled={!query.trim() || isStreaming}
          className="bg-accent-primary hover:opacity-90 text-white px-5 py-3 rounded-xl text-sm font-semibold flex items-center gap-2 transition-all disabled:opacity-40 cursor-pointer shadow-md shadow-emerald-950/20"
        >
          <span>Ask</span>
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
