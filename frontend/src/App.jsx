import React, { useEffect } from 'react';
import {
  Newspaper,
  Bot,
  Layers,
  Sliders,
  UploadCloud,
  Sparkles,
  GitMerge,
  Network,
  Compass,
  Moon,
  Sun,
  Laptop,
  Cloud,
  Zap,
} from 'lucide-react';
import {
  ActiveHighlightProvider,
  useActiveHighlight,
} from './context/ActiveHighlightContext';
import {
  ModelConfigProvider,
  useModelConfig,
} from './context/ModelConfigContext';
import HomePage from './components/HomePage';
import BroadsheetReader from './components/BroadsheetReader';
import AgentAssistant from './components/AgentAssistant';
import ArchiveExplorer from './components/ArchiveExplorer';
import TimelineWorkspace from './components/TimelineWorkspace';
import EntityGraphWorkspace from './components/EntityGraphWorkspace';
import UploadTrigger from './components/UploadTrigger';
import ModelSettingsStudio from './components/ModelSettingsStudio';

function MainApp() {
  const { activeTab, setActiveTab } = useActiveHighlight();
  const {
    theme,
    toggleTheme,
    selectedModel,
    taskBindings,
    activePreset,
    configuredProviders,
  } = useModelConfig();

  // Navigation items with keyboard shortcuts
  const navigationTabs = [
    { id: 'home', label: 'Overview', icon: Compass, keyHint: '⌘1' },
    { id: 'reader', label: 'Broadsheet Reader', icon: Newspaper, keyHint: '⌘2' },
    { id: 'agent', label: 'AI Newsroom', icon: Bot, keyHint: '⌘3' },
    { id: 'timeline', label: 'Storyline Trajectory', icon: GitMerge, keyHint: '⌘4' },
    { id: 'graph', label: 'Entity Graph', icon: Network, keyHint: '⌘5' },
    { id: 'archive', label: 'Archive Explorer', icon: Layers, keyHint: '⌘6' },
    { id: 'ingest', label: 'Ingestion Console', icon: UploadCloud, keyHint: '⌘7' },
    { id: 'settings', label: 'Model Studio', icon: Sliders },
  ];

  // Global Keyboard Shortcuts ⌘1–⌘7 / Ctrl+1–Ctrl+7
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey) {
        const keyMap = {
          '1': 'home',
          '2': 'reader',
          '3': 'agent',
          '4': 'timeline',
          '5': 'graph',
          '6': 'archive',
          '7': 'ingest',
        };
        if (keyMap[e.key]) {
          e.preventDefault();
          setActiveTab(keyMap[e.key]);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setActiveTab]);

  // Derive model display name & architecture badge
  const activeLlm = selectedModel || taskBindings?.answerer || taskBindings?.query_planner || 'gemini_flash';
  const providerDetails = configuredProviders?.find((p) => p.id === activeLlm);
  const activeModelName =
    activeLlm === 'gemini_flash'
      ? 'Gemini 2.5 Flash'
      : activeLlm === 'gemini_pro'
      ? 'Gemini 2.5 Pro'
      : activeLlm === 'ollama_llama3'
      ? 'Llama 3.1 8B'
      : activeLlm === 'ollama_deepseek'
      ? 'DeepSeek R1 14B'
      : activeLlm === 'ollama_gemma4_e4b'
      ? 'Gemma 4 E4B'
      : providerDetails?.model || activeLlm;

  const isLocal = activeLlm.startsWith('ollama') || activeLlm.startsWith('local') || activeLlm.startsWith('laptop:');
  const isHybrid = activePreset === 'cloud_hybrid';

  return (
    <div className="h-screen h-[100dvh] max-h-screen bg-canvas text-primary flex flex-col font-sans transition-colors duration-200 overflow-hidden">
      {/* Top Editorial Masthead & Navigation Bar */}
      <nav className="h-14 sm:h-16 bg-surface/90 backdrop-blur-md border-b border-border-subtle px-3 sm:px-4 md:px-6 flex items-center justify-between sticky top-0 z-30 shadow-sm shrink-0">
        {/* Left Masthead & Custom Geometric Emblem */}
        <div
          onClick={() => setActiveTab('home')}
          className="flex items-center gap-2 sm:gap-3 cursor-pointer select-none group shrink-0"
          title="Return to Overview (⌘1)"
        >
          {/* Custom Geometric Broadsheet-Lens SVG Emblem */}
          <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-elevated border border-border-subtle flex items-center justify-center text-accent-primary shadow-sm group-hover:border-accent-primary/60 transition-all shrink-0">
            <svg
              className="w-4 h-4 sm:w-5 sm:h-5 text-accent-primary"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" className="stroke-accent-primary/50" />
              <path d="M7 7h10M7 11h6M7 15h8" className="stroke-text-secondary" />
              <circle cx="16" cy="14" r="3.5" className="stroke-accent-editorial fill-accent-editorial/20" />
              <path d="M18.5 16.5L21 19" className="stroke-accent-editorial" />
            </svg>
          </div>

          <div>
            <span className="font-serif font-bold text-base sm:text-lg text-primary tracking-tight flex items-center gap-1.5 leading-none">
              NewsLens<span className="text-accent-primary font-sans font-extrabold text-[10px] sm:text-xs px-1 py-0.5 rounded bg-accent-primary/10 border border-accent-primary/30">AI</span>
            </span>
            <span className="text-[8px] sm:text-[9px] text-muted tracking-widest uppercase block font-mono mt-0.5">
              BROADSHEET INTELLIGENCE
            </span>
          </div>
        </div>

        {/* Center Segmented Workspace Switcher (Desktop) */}
        <div className="hidden lg:flex items-center gap-1 bg-elevated/70 p-1 rounded-xl border border-border-subtle">
          {navigationTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150 ${
                  isActive
                    ? 'bg-accent-primary text-white shadow-sm'
                    : 'text-secondary hover:text-primary hover:bg-surface'
                }`}
                title={tab.keyHint ? `${tab.label} (${tab.keyHint})` : tab.label}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                <span>{tab.label}</span>
                {tab.keyHint && (
                  <span
                    className={`text-[9px] font-mono px-1 rounded ${
                      isActive ? 'bg-white/20 text-white' : 'text-muted'
                    }`}
                  >
                    {tab.keyHint}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Right Status Cluster: Architecture Pill & Dual Theme Toggle */}
        <div className="flex items-center gap-1.5 sm:gap-2.5 shrink-0">
          {/* Active Architecture Pill */}
          <button
            type="button"
            onClick={() => setActiveTab('settings')}
            className={`flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-full border text-xs font-mono shadow-sm transition-all hover:scale-[1.02] cursor-pointer ${
              isLocal
                ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                : isHybrid
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                : 'bg-sky-500/10 border-sky-500/30 text-sky-300'
            }`}
            title={`Active Architecture: Click to configure in Model Studio`}
          >
            <span
              className={`w-2 h-2 rounded-full shrink-0 animate-pulse ${
                isLocal ? 'bg-amber-400' : isHybrid ? 'bg-emerald-400' : 'bg-sky-400'
              }`}
            />
            {/* Desktop / Tablet Label */}
            <span className="font-semibold hidden sm:inline truncate max-w-[130px] md:max-w-[200px]">
              {isLocal ? `● Local Sovereign · ${activeModelName}` : isHybrid ? `● Hybrid · ${activeModelName}` : `● Cloud · ${activeModelName}`}
            </span>
            {/* Compact Mobile Label */}
            <span className="font-semibold sm:hidden text-[11px]">
              {isLocal ? 'Local' : isHybrid ? 'Hybrid' : 'Cloud'}
            </span>
          </button>

          {/* Dual-Theme Toggle Button */}
          <button
            type="button"
            onClick={toggleTheme}
            className="p-1.5 sm:p-2 rounded-xl bg-elevated border border-border-subtle text-text-secondary hover:text-text-primary hover:border-slate-500 transition-colors shadow-sm shrink-0"
            title={`Switch to ${theme === 'midnight' ? 'Archival Newsprint (Light)' : 'Midnight Newsroom (Dark)'} theme`}
            aria-label="Toggle visual theme"
          >
            {theme === 'midnight' ? (
              <Sun className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-amber-400" />
            ) : (
              <Moon className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-sky-500" />
            )}
          </button>
        </div>
      </nav>

      {/* Mobile/Tablet Horizontal Sub-Navigation Ribbon for Viewports < 1024px */}
      <div className="lg:hidden flex items-center overflow-x-auto bg-surface/95 border-b border-border-subtle px-2 py-1.5 gap-1 text-xs shrink-0 no-scrollbar touch-pan-x">
        {navigationTabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap shrink-0 transition-colors ${
                isActive
                  ? 'bg-accent-primary text-white font-semibold shadow-xs'
                  : 'text-text-secondary hover:text-text-primary hover:bg-elevated'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Main View Container (Preserves tab state across switching) */}
      <main className="flex-1 min-h-0 overflow-hidden relative">
        <div className={activeTab === 'home' ? 'h-full w-full min-h-0 overflow-y-auto' : 'hidden'}>
          <HomePage />
        </div>
        <div className={activeTab === 'reader' ? 'h-full w-full min-h-0' : 'hidden'}>
          <BroadsheetReader />
        </div>
        <div className={activeTab === 'agent' ? 'h-full w-full min-h-0' : 'hidden'}>
          <AgentAssistant />
        </div>
        <div className={activeTab === 'graph' ? 'h-full w-full min-h-0' : 'hidden'}>
          <EntityGraphWorkspace />
        </div>
        <div className={activeTab === 'timeline' ? 'h-full w-full min-h-0' : 'hidden'}>
          <TimelineWorkspace />
        </div>
        <div className={activeTab === 'archive' ? 'h-full w-full min-h-0' : 'hidden'}>
          <ArchiveExplorer />
        </div>
        <div className={activeTab === 'ingest' ? 'h-full w-full min-h-0' : 'hidden'}>
          <UploadTrigger />
        </div>
        <div className={activeTab === 'settings' ? 'h-full w-full min-h-0' : 'hidden'}>
          <ModelSettingsStudio />
        </div>
      </main>
    </div>
  );
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('React ErrorBoundary caught error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6 text-center">
          <div className="bg-slate-900 border border-red-500/30 rounded-2xl p-8 max-w-lg shadow-2xl">
            <div className="w-12 h-12 rounded-full bg-red-500/20 text-red-400 flex items-center justify-center mx-auto mb-4">
              <Sparkles className="w-6 h-6" />
            </div>
            <h2 className="text-xl font-bold text-slate-100 mb-2 font-serif">
              Interface Encountered an Issue
            </h2>
            <p className="text-xs text-slate-400 mb-4 font-mono">
              {this.state.error?.message || 'An unexpected rendering error occurred.'}
            </p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-5 py-2.5 rounded-xl transition-colors shadow-lg"
            >
              Reload Interface
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <ErrorBoundary>
      <ModelConfigProvider>
        <ActiveHighlightProvider>
          <MainApp />
        </ActiveHighlightProvider>
      </ModelConfigProvider>
    </ErrorBoundary>
  );
}
