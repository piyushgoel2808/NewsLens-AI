import React from 'react';
import {
  Newspaper,
  Bot,
  Layers,
  Sliders,
  UploadCloud,
  Sparkles,
} from 'lucide-react';
import {
  ActiveHighlightProvider,
  useActiveHighlight,
} from './context/ActiveHighlightContext';
import BroadsheetReader from './components/BroadsheetReader';
import AgentAssistant from './components/AgentAssistant';
import ArchiveExplorer from './components/ArchiveExplorer';
import UploadTrigger from './components/UploadTrigger';
import RawDataViewer from './components/RawDataViewer';

function MainApp() {
  const { activeTab, setActiveTab } = useActiveHighlight();

  const navigationTabs = [
    { id: 'reader', label: 'Broadsheet Reader', icon: Newspaper },
    { id: 'agent', label: 'Agent Assistant', icon: Bot },
    { id: 'archive', label: 'Archive Explorer', icon: Layers },
    { id: 'ingest', label: 'Ingestion Console', icon: UploadCloud },
    { id: 'settings', label: 'Model Settings & API', icon: Sliders },
  ];

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col font-sans">
      {/* Top Main Navigation Bar */}
      <nav className="h-16 bg-slate-900/95 backdrop-blur-md border-b border-slate-800 px-4 md:px-6 flex items-center justify-between sticky top-0 z-30 shadow-md">
        {/* Brand & Logo */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center text-white shadow-lg shadow-emerald-900/30">
            <Newspaper className="w-5 h-5" />
          </div>
          <div>
            <span className="font-serif font-bold text-lg text-slate-50 tracking-tight flex items-center gap-1.5">
              NewsLens<span className="text-emerald-400 font-sans font-extrabold text-sm">AI</span>
            </span>
            <span className="text-[10px] text-slate-400 tracking-wider uppercase block font-mono -mt-0.5">
              Newspaper Intelligence RAG
            </span>
          </div>
        </div>

        {/* Center Tab Navigation */}
        <div className="flex items-center gap-1 bg-slate-950/80 p-1 rounded-xl border border-slate-800">
          {navigationTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150 ${
                  isActive
                    ? 'bg-emerald-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Status Pill */}
        <div className="hidden lg:flex items-center gap-2 bg-slate-800/80 px-3 py-1.5 rounded-full border border-slate-700 text-xs text-slate-300 font-mono">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>Local Engine Active</span>
        </div>
      </nav>

      {/* Main View Container (Preserves tab state across switching) */}
      <main className="flex-1 overflow-hidden relative">
        <div className={activeTab === 'reader' ? 'h-full w-full' : 'hidden'}>
          <BroadsheetReader />
        </div>
        <div className={activeTab === 'agent' ? 'h-full w-full' : 'hidden'}>
          <AgentAssistant />
        </div>
        <div className={activeTab === 'archive' ? 'h-full w-full' : 'hidden'}>
          <ArchiveExplorer />
        </div>
        <div className={activeTab === 'ingest' ? 'h-full w-full' : 'hidden'}>
          <UploadTrigger />
        </div>
        <div className={activeTab === 'settings' ? 'h-full w-full' : 'hidden'}>
          <RawDataViewer />
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ActiveHighlightProvider>
      <MainApp />
    </ActiveHighlightProvider>
  );
}
