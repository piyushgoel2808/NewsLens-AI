import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Calendar,
  Newspaper,
  Tag,
  Users,
  Sparkles,
  ExternalLink,
  BookOpen,
  Info,
  Search,
  Filter,
  Loader2,
  RefreshCw,
  CheckCircle2,
  Bot,
  Sliders,
  Eye,
  Database,
  BarChart3,
  FileText,
  Table,
  Check,
  Minimize2,
  X,
  Compass,
} from 'lucide-react';
import CanvasOverlay from './CanvasOverlay';
import { useActiveHighlight } from '../context/ActiveHighlightContext';

// ---------------------------------------------------------------------------
// Fallback Archival Broadsheet Data (Zero-Spin Resilience)
// ---------------------------------------------------------------------------
const FALLBACK_PAGE_SVG = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2800 4399" width="2800" height="4399">
  <rect width="2800" height="4399" fill="%23fbf8f2"/>
  <rect x="80" y="80" width="2640" height="4239" fill="none" stroke="%23222222" stroke-width="4"/>
  <line x1="80" y1="200" x2="2720" y2="200" stroke="%23222222" stroke-width="2"/>
  <text x="100" y="170" font-family="serif" font-size="36" fill="%23444444" font-weight="bold">VOL. CXLVI NO. 64 · NEW DELHI · FRIDAY, MARCH 15, 2024</text>
  <text x="2400" y="170" font-family="monospace" font-size="34" fill="%23444444">12 PAGES · ₹5.00</text>
  <line x1="80" y1="220" x2="2720" y2="220" stroke="%23222222" stroke-width="4"/>
  <text x="1400" y="480" font-family="serif" font-size="190" font-weight="900" text-anchor="middle" letter-spacing="12" fill="%23111111">THE HINDU</text>
  <text x="1400" y="550" font-family="sans-serif" font-size="32" text-anchor="middle" letter-spacing="8" fill="%23555555">INDIA&apos;S NATIONAL NEWSPAPER SINCE 1878</text>
  <line x1="80" y1="600" x2="2720" y2="600" stroke="%23111111" stroke-width="6"/>
  <line x1="80" y1="620" x2="2720" y2="620" stroke="%23111111" stroke-width="2"/>
  <text x="120" y="740" font-family="serif" font-size="82" font-weight="900" fill="%23111111">SC Mandates Full Disclosure of Electoral Bond Data</text>
  <text x="120" y="810" font-family="serif" font-size="44" font-style="italic" fill="%23333333">Constitution Bench rejects SBI plea for extension; orders serial numbers handover to EC by tomorrow</text>
  <line x1="120" y1="840" x2="1950" y2="840" stroke="%23888888" stroke-width="2"/>
  <rect x="120" y="880" width="1100" height="900" fill="%23e8e2d5" stroke="%23999999" stroke-width="2"/>
  <text x="670" y="1330" font-family="sans-serif" font-size="36" text-anchor="middle" fill="%23666666">PRESIDING CONSTITUTION BENCH</text>
  <text x="120" y="1820" font-family="serif" font-size="28" font-style="italic" fill="%23444444">Chief Justice D.Y. Chandrachud along with Constitution Bench members during the proceedings.</text>
  <g fill="%23333333">
    <rect x="1260" y="880" width="680" height="18" rx="4"/>
    <rect x="1260" y="920" width="660" height="18" rx="4"/>
    <rect x="1260" y="960" width="670" height="18" rx="4"/>
    <rect x="1260" y="1000" width="650" height="18" rx="4"/>
    <rect x="1260" y="1040" width="680" height="18" rx="4"/>
    <rect x="1260" y="1080" width="630" height="18" rx="4"/>
    <rect x="1260" y="1120" width="670" height="18" rx="4"/>
    <rect x="1260" y="1160" width="660" height="18" rx="4"/>
    <rect x="1260" y="1200" width="680" height="18" rx="4"/>
    <rect x="1260" y="1240" width="640" height="18" rx="4"/>
    <rect x="1260" y="1280" width="670" height="18" rx="4"/>
    <rect x="1260" y="1320" width="660" height="18" rx="4"/>
    <rect x="1260" y="1360" width="680" height="18" rx="4"/>
    <rect x="1260" y="1400" width="650" height="18" rx="4"/>
    <rect x="1260" y="1440" width="670" height="18" rx="4"/>
    <rect x="1260" y="1480" width="620" height="18" rx="4"/>
    <rect x="1260" y="1520" width="670" height="18" rx="4"/>
    <rect x="1260" y="1560" width="650" height="18" rx="4"/>
    <rect x="1260" y="1600" width="660" height="18" rx="4"/>
    <rect x="1260" y="1640" width="680" height="18" rx="4"/>
    <rect x="1260" y="1680" width="640" height="18" rx="4"/>
    <rect x="1260" y="1720" width="670" height="18" rx="4"/>
  </g>
  <line x1="1980" y1="650" x2="1980" y2="4250" stroke="%23cccccc" stroke-width="2"/>
  <text x="2020" y="740" font-family="serif" font-size="52" font-weight="bold" fill="%23111111">RBI Tightens Call</text>
  <text x="2020" y="800" font-family="serif" font-size="52" font-weight="bold" fill="%23111111">Rates Framework</text>
  <text x="2020" y="860" font-family="serif" font-size="30" font-style="italic" fill="%23555555">Overnight liquidity smoothing</text>
  <line x1="2020" y1="890" x2="2680" y2="890" stroke="%23aaaaaa" stroke-width="2"/>
  <g fill="%23333333">
    <rect x="2020" y="930" width="640" height="18" rx="4"/>
    <rect x="2020" y="970" width="620" height="18" rx="4"/>
    <rect x="2020" y="1010" width="650" height="18" rx="4"/>
    <rect x="2020" y="1050" width="630" height="18" rx="4"/>
    <rect x="2020" y="1090" width="640" height="18" rx="4"/>
    <rect x="2020" y="1130" width="610" height="18" rx="4"/>
    <rect x="2020" y="1170" width="640" height="18" rx="4"/>
    <rect x="2020" y="1210" width="630" height="18" rx="4"/>
    <rect x="2020" y="1250" width="650" height="18" rx="4"/>
  </g>
</svg>`;

const FALLBACK_READER_ISSUES = [
  { id: 101, newspaper_name: 'The Hindu', issue_date: '2024-03-15', edition: 'Morning Edition', total_pages: 4 },
  { id: 102, newspaper_name: 'Mint', issue_date: '2024-03-14', edition: 'National Business', total_pages: 4 },
  { id: 103, newspaper_name: 'The Economic Times', issue_date: '2024-03-12', edition: 'Markets & Policy', total_pages: 4 },
];

const FALLBACK_INSPECTION_DATA = {
  issue: {
    id: 101,
    newspaper_name: 'The Hindu',
    issue_date: '2024-03-15',
    edition: 'Morning Edition',
    total_pages: 4,
  },
  pages: [
    {
      id: 1,
      page_number: 1,
      width_px: 2800,
      height_px: 4399,
      image_url: FALLBACK_PAGE_SVG,
      status: 'indexed',
    },
    {
      id: 2,
      page_number: 2,
      width_px: 2800,
      height_px: 4399,
      image_url: FALLBACK_PAGE_SVG,
      status: 'indexed',
    },
  ],
  articles: [
    {
      id: 1001,
      headline: 'Supreme Court Mandates Full Disclosure of Electoral Bond Data',
      section: 'National Front',
      prominence_score: 0.98,
      pages: [1],
      word_count: 540,
      article_type: 'lead_story',
      bboxes: [{ x0: 100, y0: 680, x1: 1960, y1: 1850 }],
      summary:
        'A five-judge Constitution Bench headed by CJI D.Y. Chandrachud dismissed the State Bank of India plea seeking an extension until June 30 to match electoral bonds with donors, ordering full disclosure of alphanumeric serial numbers to the Election Commission by 5 p.m. tomorrow.',
      full_text:
        'A five-judge Constitution Bench headed by Chief Justice of India D.Y. Chandrachud on Friday dismissed the State Bank of India application seeking an extension until June 30 to collate and disclose electoral bond purchase and redemption records.\n\nThe Bench directed the bank to disclose all details, including the unique alphanumeric serial numbers of every bond purchased and redeemed, to the Election Commission by 5 p.m. on Saturday. The Commission was ordered to upload the dataset on its official portal without delay.\n\n"The State Bank of India has maintained complete records in its Mumbai main branch. The submission that matching donor identities with political parties requires months is rejected," the Chief Justice declared.',
      photos: [
        {
          id: 501,
          caption: 'Chief Justice D.Y. Chandrachud heading the Constitution Bench during the hearing',
          bbox: [120, 880, 1220, 1780],
          page_number: 1,
        },
      ],
      chunks: [
        {
          id: 2001,
          chunk_index: 0,
          chunk_type: 'editorial',
          token_count: 142,
          text: 'Supreme Court Constitution Bench dismisses SBI plea for time; orders full release of alphanumeric serial numbers by Saturday 5 PM.',
        },
        {
          id: 2002,
          chunk_index: 1,
          chunk_type: 'editorial',
          token_count: 165,
          text: 'Chief Justice Chandrachud notes records were maintained at SBI Mumbai main branch, rejecting claims that matching donor names with encashments would take months.',
        },
      ],
    },
    {
      id: 1002,
      headline: 'RBI Tightens Liquidity Framework as Overnight Call Rates Flirt With MSF Ceiling',
      section: 'Economy & Markets',
      prominence_score: 0.82,
      pages: [1],
      word_count: 380,
      article_type: 'report',
      bboxes: [{ x0: 2000, y0: 680, x1: 2700, y1: 1850 }],
      summary:
        'The Reserve Bank of India conducted variable rate repo auctions to address transient tightness in overnight money markets caused by advance tax outflows and GST settlements.',
      full_text:
        'The Reserve Bank of India on Friday injected liquidity through two-day variable rate repo (VRR) auctions as the weighted average call rate moved closer to the Marginal Standing Facility (MSF) rate of 6.75%.\n\nMarket participants noted that advance tax quarterly outflows combined with ongoing government cash balances had drained banking system liquidity.',
      photos: [],
      chunks: [
        {
          id: 2003,
          chunk_index: 0,
          chunk_type: 'editorial',
          token_count: 110,
          text: 'RBI conducts VRR operations to keep overnight call money rates aligned with policy repo rate amid advance corporate tax drains.',
        },
      ],
    },
  ],
};

export default function BroadsheetReader() {
  const {
    selectedIssueId,
    setSelectedIssueId,
    selectedPageNumber,
    setSelectedPageNumber,
    selectedArticleId,
    setSelectedArticleId,
    highlightedBboxes,
    isPulsing,
    hoveredArticleId,
    setHoveredArticleId,
    highlightArticle,
    attachAssetForAgent,
    taskBindings,
    selectedModel,
    activePreset,
  } = useActiveHighlight();

  const [issues, setIssues] = useState([]);
  const [issueData, setIssueData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [articleDetails, setArticleDetails] = useState(null);
  const [articleLoading, setArticleLoading] = useState(false);

  // Inspector Drawer Active Tab: 'articles' | 'visuals' | 'chunks'
  const [activeInspectorTab, setActiveInspectorTab] = useState('articles');
  const [visualsCurrentPageOnly, setVisualsCurrentPageOnly] = useState(true);
  const [chunkSearchTerm, setChunkSearchTerm] = useState('');

  // Filmstrip Bottom Drawer
  const [showFilmstrip, setShowFilmstrip] = useState(false);

  // Canvas Viewport Controls
  const [zoom, setZoom] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);
  const [layers, setLayers] = useState({
    editorial: true,
    photos: true,
    tables: true,
    ads: true,
  });

  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSection, setSelectedSection] = useState('ALL');
  const [analyzingPhotoIds, setAnalyzingPhotoIds] = useState({});
  const [mobileView, setMobileView] = useState('page'); // 'page' | 'inspector'

  const canvasContainerRef = useRef(null);
  const articleListRef = useRef(null);

  const currentIssue = useMemo(() => {
    if (issueData?.issue) return issueData.issue;
    if (Array.isArray(issues)) {
      return issues.find((i) => i.id === selectedIssueId) || null;
    }
    return null;
  }, [issueData, issues, selectedIssueId]);

  // Derive active vector collection
  const isCloudCollection =
    taskBindings?.embedding === 'gemini_embedding' ||
    activePreset === 'cloud_full' ||
    (!taskBindings?.embedding && (selectedModel || '').startsWith('gemini'));
  const vectorCollectionBadge = isCloudCollection
    ? 'article_chunks_v2 · 768d MRL'
    : 'article_chunks · 1024d BGE-M3';

  // Active VLM model derivation
  const activeVlmModel = taskBindings?.visual_extraction || selectedModel || 'gemini_flash';
  const activeVlmLabel = useMemo(() => {
    const m = (activeVlmModel || '').toLowerCase();
    if (m.includes('gemini')) return 'Google Gemini VLM';
    if (m.includes('qwen')) return 'Qwen-VL';
    if (m.includes('llama')) return 'Llama Vision';
    return activeVlmModel;
  }, [activeVlmModel]);
  const activeVlmShort = useMemo(() => {
    const m = (activeVlmModel || '').toLowerCase();
    if (m.includes('gemini')) return 'Gemini VLM';
    if (m.includes('qwen')) return 'Qwen-VL';
    return 'VLM';
  }, [activeVlmModel]);

  const handleAnalyzePhoto = async (photoId) => {
    if (!photoId) return;
    setAnalyzingPhotoIds((prev) => ({ ...prev, [photoId]: true }));
    try {
      const vlmModel = taskBindings?.visual_extraction || selectedModel || 'gemini_flash';
      const res = await fetch(`/api/photos/${photoId}/analyze?model=${encodeURIComponent(vlmModel)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: vlmModel }),
      });
      if (!res.ok) throw new Error('Failed to analyze photo');
      const updated = await res.json();
      setArticleDetails((prev) => {
        if (!prev || !prev.photos) return prev;
        return {
          ...prev,
          photos: prev.photos.map((p) => (p.id === photoId ? { ...p, ...updated } : p)),
        };
      });
    } catch (err) {
      console.error('Photo analysis error:', err);
    } finally {
      setAnalyzingPhotoIds((prev) => ({ ...prev, [photoId]: false }));
    }
  };

  const [isReingestingPage, setIsReingestingPage] = useState(false);
  const [reingestStatus, setReingestStatus] = useState(null);

  const handleReingestCurrentPage = async () => {
    if (!selectedIssueId || !selectedPageNumber || isReingestingPage) return;

    const activeParserName =
      taskBindings?.document_parser === 'docling_cloud_parser'
        ? 'Docling Cloud'
        : taskBindings?.document_parser === 'docling_parser'
        ? 'Docling Local'
        : taskBindings?.document_parser || 'Docling Cloud';

    const activeVlmName =
      (taskBindings?.visual_extraction || selectedModel || '').includes('gemini')
        ? 'Gemini Flash VLM'
        : (taskBindings?.visual_extraction || selectedModel || '').includes('qwen')
        ? 'Qwen-VL'
        : taskBindings?.visual_extraction || selectedModel || 'Gemini Flash VLM';

    const activeEmbedName =
      taskBindings?.embedding === 'gemini_embedding'
        ? 'Gemini Embedding (768d)'
        : taskBindings?.embedding === 'local_embed_bge'
        ? 'BAAI BGE-M3 (1024d)'
        : taskBindings?.embedding || 'Gemini Embedding';

    const confirmed = window.confirm(
      `Re-ingest Page ${selectedPageNumber} using currently selected models?\n\n` +
      `• Document Parser: ${activeParserName}\n` +
      `• Visual Extraction (VLM): ${activeVlmName}\n` +
      `• Dense Vector Embedding: ${activeEmbedName}\n\n` +
      `This will purge previous page extractions, re-run layout parsing & OCR with ${activeParserName}, ` +
      `extract photo scenes & charts with ${activeVlmName}, and re-index dense embeddings with ${activeEmbedName} into Qdrant.`
    );
    if (!confirmed) return;

    setIsReingestingPage(true);
    setReingestStatus({
      type: 'loading',
      message: `Re-ingesting Page ${selectedPageNumber} using ${activeParserName}, ${activeVlmName}, and ${activeEmbedName}...`,
    });

    try {
      const parserEngine = taskBindings?.document_parser || 'auto';
      const res = await fetch(
        `/api/issues/${selectedIssueId}/pages/${selectedPageNumber}/reingest?parser_engine=${encodeURIComponent(parserEngine)}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            parser_engine: parserEngine,
            task_bindings: taskBindings,
          }),
        }
      );
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();

      // Refresh issue inspection data
      const refreshRes = await fetch(`/api/issues/${selectedIssueId}/inspection?chunk_limit=100`);
      if (refreshRes.ok) {
        const freshData = await refreshRes.json();
        setIssueData(freshData);
      }
      setSelectedArticleId(null);

      setReingestStatus({
        type: 'success',
        message: `Page ${selectedPageNumber} successfully re-ingested with ${activeParserName} & ${activeEmbedName}! (${data.articles_count} articles, ${data.chunks_count} chunks re-indexed).`,
      });
      setTimeout(() => setReingestStatus(null), 7000);
    } catch (err) {
      console.error('Page re-ingestion error:', err);
      setReingestStatus({
        type: 'error',
        message: `Re-ingestion failed: ${err.message}`,
      });
      setTimeout(() => setReingestStatus(null), 8000);
    } finally {
      setIsReingestingPage(false);
    }
  };

  // 1. Fetch Issues Catalog with Fallback Guarantee
  useEffect(() => {
    fetch('/api/issues?limit=50')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const list = Array.isArray(data) && data.length > 0 ? data : FALLBACK_READER_ISSUES;
        setIssues(list);
        if (list.length > 0 && !selectedIssueId) {
          setSelectedIssueId(list[0].id);
        }
      })
      .catch((err) => {
        console.warn('Using fallback archival catalog for reader:', err);
        setIssues(FALLBACK_READER_ISSUES);
        if (!selectedIssueId) {
          setSelectedIssueId(FALLBACK_READER_ISSUES[0].id);
        }
      });
  }, [selectedIssueId, setSelectedIssueId]);

  // 2. Fetch Issue Inspection Data with Fallback Guarantee
  useEffect(() => {
    if (!selectedIssueId) return;
    setLoading(true);

    fetch(`/api/issues/${selectedIssueId}/inspection?chunk_limit=100`)
      .then((res) => {
        if (!res.ok) throw new Error('Issue not found');
        return res.json();
      })
      .then((data) => {
        if (data && !data.error && Array.isArray(data.pages) && data.pages.length > 0) {
          setIssueData(data);
        } else {
          setIssueData(FALLBACK_INSPECTION_DATA);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.warn('Backend inspection unavailable, activating demo broadsheet:', err);
        setIssueData(FALLBACK_INSPECTION_DATA);
        setLoading(false);
      });
  }, [selectedIssueId]);

  // 3. Fetch Full Article Details when an article is selected
  useEffect(() => {
    if (!selectedArticleId) {
      setArticleDetails(null);
      return;
    }
    setArticleLoading(true);
    fetch(`/api/articles/${selectedArticleId}`)
      .then((res) => {
        if (!res.ok) throw new Error('Article details failed');
        return res.json();
      })
      .then((data) => {
        if (data && !data.error) {
          setArticleDetails(data);
        } else {
          const fallbackArt = FALLBACK_INSPECTION_DATA.articles.find((a) => a.id === selectedArticleId);
          setArticleDetails(fallbackArt || null);
        }
        setArticleLoading(false);
      })
      .catch((err) => {
        const fallbackArt = FALLBACK_INSPECTION_DATA.articles.find((a) => a.id === selectedArticleId);
        setArticleDetails(fallbackArt || null);
        setArticleLoading(false);
      });
  }, [selectedArticleId]);

  // When an article is selected on mobile/small screens, auto-switch to inspector
  useEffect(() => {
    if (selectedArticleId && typeof window !== 'undefined' && window.innerWidth < 1024) {
      setMobileView('inspector');
      setActiveInspectorTab('articles');
    }
  }, [selectedArticleId]);

  // Current Active Page
  const currentPage = useMemo(() => {
    if (!issueData || !Array.isArray(issueData.pages) || issueData.pages.length === 0) return null;
    return (
      issueData.pages.find((p) => p.page_number === selectedPageNumber) ||
      issueData.pages[0] ||
      null
    );
  }, [issueData, selectedPageNumber]);

  // Articles on the current page with bounding boxes
  const pageArticles = useMemo(() => {
    if (!issueData || !Array.isArray(issueData.articles) || !currentPage) return [];
    return issueData.articles
      .filter((art) => {
        const pages = Array.isArray(art.pages) ? art.pages : [];
        return pages.includes(currentPage.page_number);
      })
      .map((art) => {
        const pageBboxes =
          (art.page_bboxes && art.page_bboxes[currentPage.page_number]) ||
          (art.bboxes_by_page && art.bboxes_by_page[currentPage.page_number]) ||
          (Array.isArray(art.bboxes) ? art.bboxes : []);
        return {
          ...art,
          bboxes: pageBboxes,
        };
      });
  }, [issueData, currentPage]);

  // All Visual Assets in the current issue
  const allVisualAssets = useMemo(() => {
    if (!issueData || !Array.isArray(issueData.articles)) return [];
    const assets = [];
    issueData.articles.forEach((art) => {
      if (Array.isArray(art.photos)) {
        art.photos.forEach((ph) => {
          assets.push({
            ...ph,
            articleId: art.id,
            articleHeadline: art.headline,
            page_number: ph.page_number || art.pages?.[0] || 1,
          });
        });
      }
    });
    return assets;
  }, [issueData]);

  // Filtered Visual Assets (Page only vs All issue)
  const filteredVisualAssets = useMemo(() => {
    if (visualsCurrentPageOnly) {
      return allVisualAssets.filter((a) => a.page_number === selectedPageNumber);
    }
    return allVisualAssets;
  }, [allVisualAssets, visualsCurrentPageOnly, selectedPageNumber]);

  // All Chunks on the current page (if no article selected)
  const pageChunks = useMemo(() => {
    if (selectedArticleId && articleDetails?.chunks) {
      return articleDetails.chunks;
    }
    if (!issueData || !Array.isArray(issueData.articles) || !currentPage) return [];
    const chunks = [];
    pageArticles.forEach((art) => {
      if (Array.isArray(art.chunks)) {
        art.chunks.forEach((chk) => {
          chunks.push({
            ...chk,
            articleHeadline: art.headline,
            articleId: art.id,
          });
        });
      }
    });
    return chunks;
  }, [selectedArticleId, articleDetails, issueData, currentPage, pageArticles]);

  // Filtered Chunks by search term
  const filteredChunks = useMemo(() => {
    if (!chunkSearchTerm.trim()) return pageChunks;
    const term = chunkSearchTerm.toLowerCase();
    return pageChunks.filter(
      (c) =>
        (c.text && c.text.toLowerCase().includes(term)) ||
        (c.articleHeadline && c.articleHeadline.toLowerCase().includes(term))
    );
  }, [pageChunks, chunkSearchTerm]);

  // Available sections for filtering
  const availableSections = useMemo(() => {
    const set = new Set();
    pageArticles.forEach((a) => {
      if (a.section) set.add(a.section);
    });
    return ['ALL', ...Array.from(set)];
  }, [pageArticles]);

  // Filtered articles
  const filteredArticles = useMemo(() => {
    return pageArticles.filter((art) => {
      const matchSection = selectedSection === 'ALL' || art.section === selectedSection;
      const matchSearch =
        !searchTerm ||
        (art.headline && art.headline.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (art.full_text_preview &&
          art.full_text_preview.toLowerCase().includes(searchTerm.toLowerCase()));
      return matchSection && matchSearch;
    });
  }, [pageArticles, selectedSection, searchTerm]);

  // Zoom handlers
  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.5));
  const handleResetZoom = () => setZoom(1);

  const handleFitWidth = () => {
    if (canvasContainerRef.current) {
      const containerWidth = canvasContainerRef.current.clientWidth - 48;
      const targetScale = Math.min(containerWidth / 900, 1.6);
      setZoom(Math.max(targetScale, 0.7));
    }
  };

  // Prominence heatmap helper
  const getProminenceBadge = (score) => {
    if (score === undefined || score === null) return null;
    const num = Number(score);
    if (num >= 0.7) {
      return (
        <span
          className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30"
          title={`Lead Story Prominence: ${(num * 100).toFixed(0)}%`}
        >
          Gold · {(num * 100).toFixed(0)}%
        </span>
      );
    }
    if (num >= 0.3) {
      return (
        <span
          className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-400 border border-sky-500/30"
          title={`Secondary Story Prominence: ${(num * 100).toFixed(0)}%`}
        >
          Blue · {(num * 100).toFixed(0)}%
        </span>
      );
    }
    return (
      <span
        className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-400 border border-teal-500/30"
        title={`Standard Story Prominence: ${(num * 100).toFixed(0)}%`}
      >
        Teal · {(num * 100).toFixed(0)}%
      </span>
    );
  };

  return (
    <div className="flex flex-col h-full min-h-0 bg-canvas text-text-primary overflow-hidden font-sans transition-colors duration-200">
      {/* Top Header & Issue Selector Ribbon */}
      <header className="flex flex-wrap items-center justify-between px-3 sm:px-4 py-2 bg-surface border-b border-border-subtle gap-2 sm:gap-3 z-20 shrink-0">
        <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
          <div className="flex items-center gap-2 bg-elevated px-2.5 sm:px-3 py-1.5 rounded-xl border border-border-subtle">
            <Newspaper className="w-4 h-4 text-accent-primary shrink-0" />
            <select
              value={selectedIssueId || ''}
              onChange={(e) => {
                setSelectedIssueId(Number(e.target.value));
                setSelectedPageNumber(1);
                setSelectedArticleId(null);
              }}
              className="bg-transparent text-xs sm:text-sm font-semibold text-text-primary outline-none cursor-pointer max-w-[200px] sm:max-w-none truncate"
            >
              {Array.isArray(issues) && issues.length > 0 ? (
                issues.map((iss) => (
                  <option key={iss.id} value={iss.id} className="bg-surface text-text-primary">
                    {iss.newspaper_name} — {iss.issue_date} (Issue #{iss.id})
                  </option>
                ))
              ) : (
                <option value="" className="bg-surface text-text-muted">
                  No issues archived yet
                </option>
              )}
            </select>
          </div>

          {/* Mobile View Toggle Switcher (Canvas vs Inspector) */}
          <div className="lg:hidden flex items-center bg-elevated rounded-xl p-0.5 border border-border-subtle text-xs">
            <button
              type="button"
              onClick={() => setMobileView('page')}
              className={`px-2.5 py-1 rounded-lg font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                mobileView === 'page'
                  ? 'bg-accent-primary text-white shadow-xs'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              <Newspaper className="w-3.5 h-3.5" />
              <span>Canvas</span>
            </button>
            <button
              type="button"
              onClick={() => setMobileView('inspector')}
              className={`px-2.5 py-1 rounded-lg font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                mobileView === 'inspector'
                  ? 'bg-accent-editorial text-white shadow-xs'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Inspector</span>
              {filteredArticles.length > 0 && (
                <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-black/20 font-mono">
                  {filteredArticles.length}
                </span>
              )}
            </button>
          </div>

          {issueData?.issue && (
            <div className="hidden md:flex items-center gap-2 text-xs text-text-muted font-mono">
              <span className="flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5" />
                {issueData.issue.issue_date}
              </span>
              <span>•</span>
              <span className="bg-elevated px-2 py-0.5 rounded text-accent-primary font-medium">
                {issueData.issue.edition || 'Main Edition'}
              </span>
              <span>•</span>
              <span>{issueData.issue.total_pages} Pages</span>
            </div>
          )}
        </div>

        {/* Vector Space & Model Badges */}
        <div className="hidden sm:flex items-center gap-2 font-mono text-xs">
          <span className="text-[11px] px-2.5 py-1 rounded-full bg-elevated border border-border-subtle text-text-secondary">
            Vector Space: <strong className="text-accent-editorial">{vectorCollectionBadge}</strong>
          </span>
          <span className="text-[11px] px-2.5 py-1 rounded-full bg-elevated border border-border-subtle text-text-secondary">
            VLM: <strong className="text-accent-visual">{activeVlmShort}</strong>
          </span>
        </div>
      </header>

      {/* Re-ingestion Live Status Banner */}
      {reingestStatus && (
        <div
          className={`px-4 py-2 text-xs flex items-center justify-between border-b transition-colors font-mono ${
            reingestStatus.type === 'loading'
              ? 'bg-amber-500/15 border-amber-500/30 text-amber-300'
              : reingestStatus.type === 'success'
              ? 'bg-accent-primary/15 border-accent-primary/30 text-accent-primary'
              : 'bg-red-500/15 border-red-500/30 text-red-400'
          }`}
        >
          <div className="flex items-center gap-2">
            {reingestStatus.type === 'loading' ? (
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-amber-400" />
            ) : reingestStatus.type === 'success' ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-accent-primary" />
            ) : (
              <Info className="w-3.5 h-3.5 text-red-400" />
            )}
            <span>{reingestStatus.message}</span>
          </div>
          <button
            type="button"
            onClick={() => setReingestStatus(null)}
            className="text-[11px] opacity-70 hover:opacity-100 underline ml-3 cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Content Area (Split Screen on Desktop, Tabbed on Mobile) */}
      <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-hidden relative">
        {/* LEFT PANE: Broadsheet Canvas & Raster Viewer */}
        <section
          ref={canvasContainerRef}
          className={`${
            mobileView === 'page' ? 'flex' : 'hidden'
          } lg:flex flex-1 min-h-0 bg-canvas overflow-auto p-2 sm:p-4 items-start justify-center relative select-none`}
        >
          {loading ? (
            <div className="flex flex-col items-center justify-center h-full text-text-muted gap-3">
              <div className="w-8 h-8 border-2 border-accent-primary border-t-transparent rounded-full animate-spin" />
              <p className="text-sm font-medium">Loading high-resolution 150 DPI broadsheet...</p>
            </div>
          ) : currentPage ? (
            <div
              className="relative shadow-2xl rounded-sm transition-transform duration-100 ease-out origin-top mb-20"
              style={{
                transform: `scale(${zoom})`,
                width: '100%',
                maxWidth: '900px',
              }}
            >
              {/* Raster Image */}
              <img
                src={currentPage.image_url}
                alt={`Broadsheet Page ${currentPage.page_number}`}
                className="w-full h-auto block rounded-sm pointer-events-none border border-border-subtle"
                loading="eager"
              />

              {/* Spatial SVG Overlay */}
              {showOverlays && (
                <CanvasOverlay
                  pageWidth={currentPage.width_px || 2800}
                  pageHeight={currentPage.height_px || 4399}
                  articles={pageArticles}
                  selectedArticleId={selectedArticleId}
                  hoveredArticleId={hoveredArticleId}
                  pulsingArticleId={isPulsing ? selectedArticleId : null}
                  customHighlightBboxes={highlightedBboxes}
                  layers={layers}
                  onSelectArticle={(artId) => setSelectedArticleId(artId)}
                  onHoverArticle={(artId) => setHoveredArticleId(artId)}
                />
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-text-muted">
              <p className="text-sm">No page available for this issue.</p>
            </div>
          )}

          {/* ---------------------------------------------------------------- */}
          {/* FLOATING GLASSMORPHIC CANVAS HUD (Centered at bottom) */}
          {/* ---------------------------------------------------------------- */}
          <div className="fixed bottom-6 left-1/2 -translate-x-1/2 backdrop-blur-md bg-surface/90 border border-border-subtle rounded-full px-4 py-2 flex items-center gap-3 shadow-2xl z-30 font-mono text-xs text-text-primary">
            {/* Page Stepper */}
            <div className="flex items-center gap-1 bg-elevated px-2 py-1 rounded-full border border-border-subtle">
              <button
                type="button"
                onClick={() => setSelectedPageNumber((p) => Math.max(p - 1, 1))}
                disabled={selectedPageNumber <= 1}
                className="p-1 hover:text-accent-primary disabled:opacity-30 disabled:hover:text-inherit cursor-pointer"
                title="Previous Page"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <span className="text-[11px] font-bold px-1.5">
                Page {selectedPageNumber}
                <span className="text-text-muted font-normal"> / {issueData?.issue?.total_pages || 1}</span>
              </span>
              <button
                type="button"
                onClick={() =>
                  setSelectedPageNumber((p) =>
                    Math.min(p + 1, issueData?.issue?.total_pages || p)
                  )
                }
                disabled={selectedPageNumber >= (issueData?.issue?.total_pages || 1)}
                className="p-1 hover:text-accent-primary disabled:opacity-30 disabled:hover:text-inherit cursor-pointer"
                title="Next Page"
              >
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Quick Filmstrip Drawer Toggle */}
            <button
              type="button"
              onClick={() => setShowFilmstrip(!showFilmstrip)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-full border transition-all cursor-pointer ${
                showFilmstrip
                  ? 'bg-accent-primary text-white border-accent-primary shadow-xs'
                  : 'bg-elevated text-text-secondary hover:text-text-primary border-border-subtle'
              }`}
              title="Toggle All Pages Filmstrip Drawer"
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span className="hidden sm:inline text-[11px]">Pages</span>
            </button>

            {/* Zoom Controls */}
            <div className="flex items-center gap-1 bg-elevated px-2 py-1 rounded-full border border-border-subtle">
              <button
                type="button"
                onClick={handleZoomOut}
                className="p-1 hover:text-accent-primary text-text-secondary cursor-pointer"
                title="Zoom Out"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
              <span className="text-[11px] w-9 text-center font-bold">
                {Math.round(zoom * 100)}%
              </span>
              <button
                type="button"
                onClick={handleZoomIn}
                className="p-1 hover:text-accent-primary text-text-secondary cursor-pointer"
                title="Zoom In"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={handleResetZoom}
                className="p-1 hover:text-accent-primary text-text-secondary border-l border-border-subtle pl-1.5 cursor-pointer"
                title="Reset (100%)"
              >
                <RotateCcw className="w-3 h-3" />
              </button>
              <button
                type="button"
                onClick={handleFitWidth}
                className="p-1 hover:text-accent-primary text-text-secondary cursor-pointer"
                title="Fit Width"
              >
                <Maximize2 className="w-3 h-3" />
              </button>
            </div>

            {/* Layer Visibility Toggles */}
            <div className="hidden md:flex items-center gap-1 bg-elevated px-2 py-1 rounded-full border border-border-subtle">
              <span className="text-[10px] text-text-muted px-1">LAYERS:</span>
              <button
                type="button"
                onClick={() => setLayers((l) => ({ ...l, editorial: !l.editorial }))}
                className={`px-2 py-0.5 rounded-full text-[10px] font-bold transition-all cursor-pointer ${
                  layers.editorial
                    ? 'bg-accent-primary/20 text-accent-primary border border-accent-primary/40'
                    : 'text-text-muted line-through'
                }`}
                title="Toggle Editorial Articles Layer"
              >
                Editorial
              </button>
              <button
                type="button"
                onClick={() => setLayers((l) => ({ ...l, photos: !l.photos }))}
                className={`px-2 py-0.5 rounded-full text-[10px] font-bold transition-all cursor-pointer ${
                  layers.photos
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                    : 'text-text-muted line-through'
                }`}
                title="Toggle Photos & Diagrams Layer"
              >
                Photos
              </button>
              <button
                type="button"
                onClick={() => setLayers((l) => ({ ...l, tables: !l.tables }))}
                className={`px-2 py-0.5 rounded-full text-[10px] font-bold transition-all cursor-pointer ${
                  layers.tables
                    ? 'bg-sky-500/20 text-sky-400 border border-sky-500/40'
                    : 'text-text-muted line-through'
                }`}
                title="Toggle Tables & Infographics Layer"
              >
                Tables
              </button>
              <button
                type="button"
                onClick={() => setLayers((l) => ({ ...l, ads: !l.ads }))}
                className={`px-2 py-0.5 rounded-full text-[10px] font-bold transition-all cursor-pointer ${
                  layers.ads
                    ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                    : 'text-text-muted line-through'
                }`}
                title="Toggle Advertisements Layer"
              >
                Ads
              </button>
            </div>

            {/* Re-Ingest Single Page Trigger */}
            <button
              type="button"
              onClick={handleReingestCurrentPage}
              disabled={isReingestingPage || loading}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-[11px] font-semibold transition-all cursor-pointer ${
                isReingestingPage
                  ? 'bg-amber-500/20 border-amber-500 text-amber-300 animate-pulse'
                  : 'bg-elevated hover:bg-surface text-accent-editorial border-border-subtle hover:border-accent-editorial'
              }`}
              title="Re-ingest this page with active layout model"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isReingestingPage ? 'animate-spin' : ''}`} />
              <span className="hidden sm:inline">
                {isReingestingPage ? 'Re-ingesting...' : 'Re-ingest'}
              </span>
            </button>
          </div>

          {/* ---------------------------------------------------------------- */}
          {/* BOTTOM FILMSTRIP DRAWER (Slide Up) */}
          {/* ---------------------------------------------------------------- */}
          {showFilmstrip && issueData?.pages && (
            <div className="fixed bottom-20 left-4 right-4 lg:left-64 lg:right-96 bg-surface/95 backdrop-blur-md border border-border-subtle rounded-2xl p-3 shadow-2xl z-30">
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-border-subtle text-xs font-mono">
                <span className="text-text-primary font-bold flex items-center gap-1.5">
                  <BookOpen className="w-4 h-4 text-accent-primary" />
                  Issue Pages Filmstrip ({issueData.pages.length} Pages)
                </span>
                <button
                  type="button"
                  onClick={() => setShowFilmstrip(false)}
                  className="p-1 hover:bg-elevated rounded-lg text-text-muted hover:text-text-primary cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="flex items-center gap-3 overflow-x-auto pb-2 scrollbar-thin">
                {issueData.pages.map((p) => {
                  const isCurrent = p.page_number === selectedPageNumber;
                  return (
                    <button
                      key={p.page_number}
                      type="button"
                      onClick={() => {
                        setSelectedPageNumber(p.page_number);
                        setSelectedArticleId(null);
                      }}
                      className={`flex flex-col items-center gap-1 p-1.5 rounded-xl border transition-all shrink-0 cursor-pointer ${
                        isCurrent
                          ? 'bg-accent-primary/20 border-accent-primary ring-2 ring-accent-primary/40'
                          : 'bg-elevated border-border-subtle hover:border-text-secondary'
                      }`}
                    >
                      <div className="w-20 h-28 bg-canvas rounded overflow-hidden relative border border-border-subtle">
                        {p.image_url ? (
                          <img
                            src={p.image_url}
                            alt={`Page ${p.page_number}`}
                            className="w-full h-full object-cover"
                            loading="lazy"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-xs text-text-muted font-mono">
                            p.{p.page_number}
                          </div>
                        )}
                        <span className="absolute bottom-1 right-1 text-[9px] font-mono px-1 rounded bg-black/70 text-white">
                          p.{p.page_number}
                        </span>
                      </div>
                      <span className="text-[10px] font-mono font-semibold text-text-primary">
                        {isCurrent ? '● Active' : `Page ${p.page_number}`}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </section>

        {/* ------------------------------------------------------------------ */}
        {/* RIGHT PANE: SYNCHRONIZED 3-TAB INSPECTOR DRAWER */}
        {/* ------------------------------------------------------------------ */}
        <aside
          className={`${
            mobileView === 'inspector' ? 'flex' : 'hidden'
          } lg:flex w-full lg:w-[480px] xl:w-[540px] bg-surface border-t lg:border-t-0 lg:border-l border-border-subtle flex-col h-full min-h-0 z-10`}
        >
          {/* Top 3 Tabs Switcher */}
          <div className="grid grid-cols-3 border-b border-border-subtle bg-elevated text-xs font-mono">
            <button
              type="button"
              onClick={() => setActiveInspectorTab('articles')}
              className={`p-3 flex items-center justify-center gap-1.5 font-bold transition-all border-b-2 cursor-pointer ${
                activeInspectorTab === 'articles'
                  ? 'border-accent-primary text-accent-primary bg-surface'
                  : 'border-transparent text-text-muted hover:text-text-primary'
              }`}
            >
              <FileText className="w-4 h-4" />
              <span>Articles ({filteredArticles.length})</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveInspectorTab('visuals')}
              className={`p-3 flex items-center justify-center gap-1.5 font-bold transition-all border-b-2 cursor-pointer ${
                activeInspectorTab === 'visuals'
                  ? 'border-accent-visual text-accent-visual bg-surface'
                  : 'border-transparent text-text-muted hover:text-text-primary'
              }`}
            >
              <Eye className="w-4 h-4" />
              <span>Visuals ({filteredVisualAssets.length})</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveInspectorTab('chunks')}
              className={`p-3 flex items-center justify-center gap-1.5 font-bold transition-all border-b-2 cursor-pointer ${
                activeInspectorTab === 'chunks'
                  ? 'border-accent-info text-accent-info bg-surface'
                  : 'border-transparent text-text-muted hover:text-text-primary'
              }`}
            >
              <Database className="w-4 h-4" />
              <span>Chunks ({filteredChunks.length})</span>
            </button>
          </div>

          {/* TAB 1: SEGMENTED ARTICLES */}
          {activeInspectorTab === 'articles' && (
            <div className="flex-1 flex flex-col h-full overflow-hidden">
              {/* If an article is selected, display Article Deep-Dive */}
              {selectedArticleId && articleDetails ? (
                <div className="flex-1 flex flex-col h-full overflow-hidden bg-surface">
                  {/* Article Detail Header */}
                  <div className="p-3.5 border-b border-border-subtle flex items-center justify-between bg-elevated/70 backdrop-blur-sm sticky top-0 z-10">
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => setSelectedArticleId(null)}
                        className="flex items-center gap-1.5 text-xs text-text-secondary hover:text-text-primary font-medium px-2 py-1 rounded-lg bg-surface border border-border-subtle transition-colors cursor-pointer"
                      >
                        <ChevronLeft className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">Back to Page Articles</span>
                        <span className="sm:hidden">Articles</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => setMobileView('page')}
                        className="lg:hidden flex items-center gap-1 text-xs text-accent-primary hover:text-accent-primary/80 font-medium px-2 py-1 rounded-lg bg-accent-primary/10 border border-accent-primary/30 transition-colors cursor-pointer"
                        title="Return to Broadsheet Page View"
                      >
                        <Newspaper className="w-3.5 h-3.5" />
                        <span>Canvas</span>
                      </button>
                    </div>
                    <div className="flex items-center gap-2">
                      {getProminenceBadge(articleDetails.prominence_score)}
                      <span className="text-xs uppercase font-mono px-2 py-0.5 rounded bg-surface text-text-secondary border border-border-subtle">
                        {articleDetails.section || articleDetails.article_type || 'News'}
                      </span>
                    </div>
                  </div>

                  {/* Article Content Body (Scrollable) */}
                  <div className="flex-1 overflow-y-auto p-5 space-y-4">
                    {articleDetails.subheadline && (
                      <p className="text-xs uppercase tracking-widest text-accent-primary font-semibold font-mono">
                        {articleDetails.subheadline}
                      </p>
                    )}

                    <h1 className="text-xl md:text-2xl font-bold font-serif leading-tight text-text-primary">
                      {articleDetails.headline || 'Untitled Article'}
                    </h1>

                    {/* Author & Issue Info */}
                    <div className="flex flex-wrap items-center gap-3 text-xs text-text-muted border-b border-border-subtle pb-3 font-mono">
                      {articleDetails.byline_author && (
                        <span className="text-text-primary font-medium">
                          By {articleDetails.byline_author}
                        </span>
                      )}
                      <span>•</span>
                      <span>{articleDetails.word_count} words</span>
                      <span>•</span>
                      <span>
                        Spans Page {articleDetails.pages?.map((p) => p.page_number).join(', ')}
                      </span>
                    </div>

                    {/* AI Generated Executive Summary */}
                    {articleDetails.summary && (
                      <div className="bg-elevated border border-border-subtle rounded-xl p-3 text-xs leading-relaxed text-text-secondary">
                        <div className="flex items-center gap-1.5 text-accent-primary font-semibold mb-1 font-mono">
                          <Sparkles className="w-3.5 h-3.5" />
                          Executive Summary
                        </div>
                        {articleDetails.summary}
                      </div>
                    )}

                    {/* Full Article Text */}
                    <div className="text-sm text-text-secondary leading-relaxed font-sans whitespace-pre-wrap">
                      {articleDetails.full_text}
                    </div>

                    {/* Continuation Pages */}
                    {articleDetails.pages && articleDetails.pages.length > 1 && (
                      <div className="bg-elevated border border-border-subtle p-3 rounded-xl flex items-center justify-between text-xs mt-4 font-mono">
                        <span className="text-text-secondary">Article spans multiple pages:</span>
                        <div className="flex gap-1.5">
                          {articleDetails.pages.map((p) => (
                            <button
                              key={p.page_number}
                              type="button"
                              onClick={() => setSelectedPageNumber(p.page_number)}
                              className={`px-2.5 py-1 rounded text-xs font-semibold transition-colors cursor-pointer ${
                                selectedPageNumber === p.page_number
                                  ? 'bg-accent-primary text-white'
                                  : 'bg-surface text-text-secondary hover:bg-elevated'
                              }`}
                            >
                              Page {p.page_number}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                /* Segmented Articles List */
                <div className="flex-1 flex flex-col h-full overflow-hidden">
                  {/* Search and Section Filter */}
                  <div className="p-3 border-b border-border-subtle space-y-2 bg-surface">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-text-secondary font-bold flex items-center gap-1.5">
                        <BookOpen className="w-3.5 h-3.5 text-accent-primary" />
                        Page {selectedPageNumber} Articles ({filteredArticles.length})
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <div className="relative flex-1">
                        <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-text-muted" />
                        <input
                          type="text"
                          placeholder="Filter articles..."
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          className="w-full pl-8 pr-3 py-1.5 bg-elevated border border-border-subtle rounded-lg text-xs text-text-primary placeholder-text-muted outline-none focus:border-accent-primary font-sans"
                        />
                      </div>

                      {availableSections.length > 1 && (
                        <select
                          value={selectedSection}
                          onChange={(e) => setSelectedSection(e.target.value)}
                          className="bg-elevated border border-border-subtle rounded-lg px-2 py-1.5 text-xs text-text-secondary outline-none cursor-pointer font-mono"
                        >
                          {availableSections.map((sec) => (
                            <option key={sec} value={sec} className="bg-surface">
                              {sec}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                  </div>

                  {/* Article Cards */}
                  <div ref={articleListRef} className="flex-1 overflow-y-auto p-3 space-y-2.5">
                    {filteredArticles.length === 0 ? (
                      <div className="text-center py-12 px-4 text-text-muted text-xs flex flex-col items-center gap-2">
                        <p className="font-semibold text-text-primary">No articles match your filter.</p>
                      </div>
                    ) : (
                      filteredArticles.map((art) => {
                        const isSelected = selectedArticleId === art.id;
                        const isHovered = hoveredArticleId === art.id;

                        return (
                          <div
                            key={art.id}
                            onClick={() => setSelectedArticleId(art.id)}
                            onMouseEnter={() => setHoveredArticleId(art.id)}
                            onMouseLeave={() => setHoveredArticleId(null)}
                            className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                              isSelected
                                ? 'bg-elevated border-accent-primary shadow-md ring-1 ring-accent-primary'
                                : isHovered
                                ? 'bg-elevated/70 border-border-subtle'
                                : 'bg-surface border-border-subtle hover:border-text-secondary'
                            }`}
                          >
                            <div className="flex items-center justify-between mb-1.5 flex-wrap gap-1">
                              <div className="flex items-center gap-1.5 flex-wrap">
                                {getProminenceBadge(art.prominence_score)}
                                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-elevated text-text-muted border border-border-subtle">
                                  {art.section || art.article_type || 'News'}
                                </span>
                              </div>
                              <span className="text-[11px] text-text-muted font-mono">
                                {art.word_count} words
                              </span>
                            </div>

                            <h3 className="text-sm font-bold font-serif leading-snug text-text-primary mb-1 line-clamp-2">
                              {art.headline || 'Untitled Article'}
                            </h3>

                            {art.full_text_preview && (
                              <p className="text-xs text-text-secondary line-clamp-2 leading-relaxed font-sans">
                                {art.full_text_preview}
                              </p>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: VISUAL ASSETS (Photos, Charts, Infographics & Markdown Tables) */}
          {activeInspectorTab === 'visuals' && (
            <div className="flex-1 flex flex-col h-full overflow-hidden">
              {/* Filter Ribbon: Current Page vs All Issue */}
              <div className="p-3 border-b border-border-subtle flex items-center justify-between bg-surface font-mono text-xs">
                <span className="text-text-secondary font-bold flex items-center gap-1.5">
                  <Eye className="w-3.5 h-3.5 text-accent-visual" />
                  Visual Intelligence ({filteredVisualAssets.length})
                </span>

                <div className="flex items-center gap-1 bg-elevated p-0.5 rounded-lg border border-border-subtle">
                  <button
                    type="button"
                    onClick={() => setVisualsCurrentPageOnly(true)}
                    className={`px-2 py-1 rounded text-[10px] font-semibold transition-all cursor-pointer ${
                      visualsCurrentPageOnly
                        ? 'bg-surface text-accent-visual font-bold shadow-xs'
                        : 'text-text-muted hover:text-text-primary'
                    }`}
                  >
                    Page {selectedPageNumber} Only
                  </button>
                  <button
                    type="button"
                    onClick={() => setVisualsCurrentPageOnly(false)}
                    className={`px-2 py-1 rounded text-[10px] font-semibold transition-all cursor-pointer ${
                      !visualsCurrentPageOnly
                        ? 'bg-surface text-accent-visual font-bold shadow-xs'
                        : 'text-text-muted hover:text-text-primary'
                    }`}
                  >
                    All Issue Assets
                  </button>
                </div>
              </div>

              {/* Assets List */}
              <div className="flex-1 overflow-y-auto p-3 space-y-4">
                {filteredVisualAssets.length === 0 ? (
                  <div className="text-center py-12 px-4 text-text-muted text-xs flex flex-col items-center gap-2">
                    <p className="font-semibold text-text-primary">
                      No visual assets on Page {selectedPageNumber}.
                    </p>
                    <button
                      type="button"
                      onClick={() => setVisualsCurrentPageOnly(false)}
                      className="text-accent-visual underline cursor-pointer mt-1"
                    >
                      View all visual assets across this issue &rarr;
                    </button>
                  </div>
                ) : (
                  filteredVisualAssets.map((asset, aIdx) => {
                    const isAnalyzing = Boolean(analyzingPhotoIds[asset.id]);
                    return (
                      <div
                        key={asset.id || aIdx}
                        className="bg-elevated border border-border-subtle rounded-2xl p-4 flex flex-col gap-3 shadow-md"
                      >
                        {/* Image Preview */}
                        {asset.image_url && (
                          <div className="bg-canvas rounded-xl overflow-hidden flex items-center justify-center max-h-60 border border-border-subtle">
                            <img
                              src={asset.image_url}
                              alt={asset.caption || `Visual Asset #${asset.id}`}
                              className="w-full h-auto object-contain max-h-60 rounded-xl"
                              loading="lazy"
                            />
                          </div>
                        )}

                        {/* Top Badges & Actions */}
                        <div className="flex items-center justify-between flex-wrap gap-2">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-accent-visual/20 text-accent-visual border border-accent-visual/30">
                              Asset #{asset.id}
                            </span>
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface text-text-secondary border border-border-subtle">
                              Page {asset.page_number}
                            </span>
                            <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-surface text-text-secondary border border-border-subtle">
                              {asset.visual_type ? asset.visual_type.replace('_', ' ') : 'Infographic'}
                            </span>
                          </div>

                          <div className="flex items-center gap-1.5">
                            <button
                              type="button"
                              onClick={() =>
                                attachAssetForAgent({
                                  photoId: asset.id,
                                  articleId: asset.articleId,
                                  headline: asset.articleHeadline,
                                  caption: asset.caption,
                                  visualType: asset.visual_type,
                                  imageUrl: asset.image_url,
                                  issueDate: currentIssue?.issue_date,
                                  newspaperName: currentIssue?.newspaper_name,
                                  issueId: currentIssue?.id || selectedIssueId,
                                  pageNumber: asset.page_number,
                                })
                              }
                              className="text-[10px] px-2.5 py-1 rounded-lg bg-accent-visual text-white font-semibold flex items-center gap-1 transition-all shadow-xs hover:opacity-90 cursor-pointer"
                              title="Ask AI Newsroom Agent about this chart"
                            >
                              <Bot className="w-3 h-3" />
                              <span>Ask Agent</span>
                            </button>

                            {asset.bbox && (
                              <button
                                type="button"
                                onClick={() => {
                                  setSelectedPageNumber(asset.page_number);
                                  highlightArticle(
                                    selectedIssueId,
                                    asset.page_number,
                                    asset.articleId,
                                    [asset.bbox]
                                  );
                                }}
                                className="text-[10px] px-2 py-1 rounded-lg bg-surface hover:bg-elevated text-accent-primary border border-border-subtle font-mono cursor-pointer"
                              >
                                Highlight
                              </button>
                            )}
                          </div>
                        </div>

                        {/* Newspaper Caption */}
                        {asset.caption && (
                          <div className="bg-surface p-2.5 rounded-xl border border-border-subtle text-xs">
                            <span className="text-[10px] uppercase font-mono font-bold text-text-muted block mb-1">
                              📰 Original Caption:
                            </span>
                            <p className="text-text-secondary italic leading-relaxed">
                              {asset.caption}
                            </p>
                          </div>
                        )}

                        {/* VLM Scene Description or Markdown Table */}
                        {asset.vlm_description ? (
                          <div className="bg-surface p-3 rounded-xl border border-border-subtle text-xs space-y-2">
                            <div className="flex items-center justify-between border-b border-border-subtle pb-1.5">
                              <span className="text-[11px] font-mono font-bold text-accent-visual flex items-center gap-1.5">
                                <Sparkles className="w-3 h-3" />
                                {activeVlmLabel} Single-Pass Extraction
                              </span>
                              <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-elevated text-accent-visual">
                                Markdown Grounded
                              </span>
                            </div>
                            <div className="text-text-secondary text-xs whitespace-pre-wrap leading-relaxed font-sans">
                              {asset.vlm_description}
                            </div>
                          </div>
                        ) : (
                          <div className="bg-surface p-3 rounded-xl border border-dashed border-border-subtle text-center">
                            <p className="text-[11px] text-text-muted mb-2">
                              VLM scene analysis not yet generated.
                            </p>
                            <button
                              type="button"
                              disabled={isAnalyzing}
                              onClick={() => handleAnalyzePhoto(asset.id)}
                              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent-visual/20 text-accent-visual border border-accent-visual/40 text-xs font-semibold cursor-pointer"
                            >
                              <Sparkles className="w-3.5 h-3.5" />
                              <span>{isAnalyzing ? 'Analyzing Scene...' : `Analyze with ${activeVlmShort}`}</span>
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}

          {/* TAB 3: VECTOR CHUNKS (Qdrant Dense Embeddings) */}
          {activeInspectorTab === 'chunks' && (
            <div className="flex-1 flex flex-col h-full overflow-hidden">
              {/* Chunks Header */}
              <div className="p-3 border-b border-border-subtle space-y-2 bg-surface font-mono text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary font-bold flex items-center gap-1.5">
                    <Database className="w-3.5 h-3.5 text-accent-info" />
                    Indexed Vector Chunks ({filteredChunks.length})
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-accent-info/20 text-accent-info border border-accent-info/30">
                    {vectorCollectionBadge}
                  </span>
                </div>

                <div className="relative">
                  <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-text-muted" />
                  <input
                    type="text"
                    placeholder="Search chunk embeddings..."
                    value={chunkSearchTerm}
                    onChange={(e) => setChunkSearchTerm(e.target.value)}
                    className="w-full pl-8 pr-3 py-1.5 bg-elevated border border-border-subtle rounded-lg text-xs text-text-primary placeholder-text-muted outline-none focus:border-accent-info font-sans"
                  />
                </div>
              </div>

              {/* Chunks List */}
              <div className="flex-1 overflow-y-auto p-3 space-y-3 font-mono">
                {filteredChunks.length === 0 ? (
                  <div className="text-center py-12 px-4 text-text-muted text-xs">
                    <p className="font-semibold text-text-primary">No vector chunks found.</p>
                  </div>
                ) : (
                  filteredChunks.map((chk, cIdx) => (
                    <div
                      key={chk.id || cIdx}
                      className="bg-elevated border border-border-subtle rounded-xl p-3 text-xs space-y-2"
                    >
                      <div className="flex items-center justify-between flex-wrap gap-1">
                        <span className="font-bold text-accent-info text-[10px] uppercase bg-surface px-2 py-0.5 rounded border border-border-subtle">
                          Chunk #{chk.chunk_index !== undefined ? chk.chunk_index + 1 : cIdx + 1}
                        </span>
                        <span className="text-[10px] text-text-muted">
                          {chk.token_count || '~180'} tokens
                        </span>
                      </div>

                      {chk.articleHeadline && (
                        <div className="text-[11px] font-serif font-bold text-text-primary truncate">
                          {chk.articleHeadline}
                        </div>
                      )}

                      <p className="text-text-secondary text-[11px] leading-relaxed whitespace-pre-wrap bg-surface p-2.5 rounded-lg border border-border-subtle font-mono">
                        {chk.text}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
