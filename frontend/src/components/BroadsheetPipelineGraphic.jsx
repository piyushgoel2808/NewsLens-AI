import React, { useState } from 'react';
import {
  Layout,
  Eye,
  Bot,
  Layers,
  Sparkles,
  ArrowRight,
  Database,
  CheckCircle2,
  Cpu,
  Zap,
} from 'lucide-react';

export default function BroadsheetPipelineGraphic({ onSelectStage }) {
  const [activeStage, setActiveStage] = useState('layout'); // 'layout' | 'vlm' | 'crag'

  const stages = {
    layout: {
      title: 'Docling 2D Layout Engine',
      subtitle: '150 DPI Spatial Geometry & Convex Boundary Walls',
      description:
        'Deconstructs physical broadsheets into discrete articles, cleans author-box decks, isolates statutory ads, and maps exact [ymin, xmin, ymax, xmax] 2D coordinates.',
      metric: '2.5s / broadsheet page',
      color: 'emerald',
      borderColor: 'border-emerald-500/50',
      badgeBg: 'bg-emerald-500/20 text-emerald-300',
    },
    vlm: {
      title: 'Multimodal VLM Matrix',
      subtitle: 'Single-Pass Vision & Markdown Table Reconstruction',
      description:
        'Inspects financial infographics, IPO subscription matrices, and news photography in 1 unified pass. Transcribes tables to clean GitHub Markdown with 3–6 verifiable statistical metrics.',
      metric: 'Single-Pass Batch Extraction',
      color: 'sky',
      borderColor: 'border-sky-500/50',
      badgeBg: 'bg-sky-500/20 text-sky-300',
    },
    crag: {
      title: 'CRAG & Neural Reranker',
      subtitle: 'LangGraph 7-Archetype Cascade & Cross-Encoder MiniLM',
      description:
        'Classifies research queries into 7 journalistic archetypes, executes parallel tool gathering, and evaluates retrieved evidence with two-stage Cross-Encoder neural reranking in ~1.0s.',
      metric: '~1.0s Two-Stage Cascade',
      color: 'amber',
      borderColor: 'border-amber-500/50',
      badgeBg: 'bg-amber-500/20 text-amber-300',
    },
  };

  const current = stages[activeStage];

  return (
    <div className="w-full bg-slate-900/90 rounded-2xl border border-slate-800 p-5 shadow-2xl relative overflow-hidden backdrop-blur-md">
      {/* Subtle background ambient glow */}
      <div className="absolute -top-12 -right-12 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-12 -left-12 w-64 h-64 bg-sky-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header Bar */}
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800/80 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="font-mono text-slate-300 font-semibold tracking-wider uppercase text-[11px]">
            Spatial Telemetry Visualizer
          </span>
        </div>
        <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">
          INTERACTIVE PIPELINE HUD
        </span>
      </div>

      {/* Interactive Diagram Split: Broadsheet Front Page (Left) + Engine Nodes (Right) */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-center">
        {/* Left: Stylized Miniature Broadsheet Frontpage */}
        <div className="md:col-span-6 flex flex-col items-center">
          <div className="w-full max-w-[280px] bg-slate-950 rounded-xl border border-slate-700/80 p-3 shadow-xl relative overflow-hidden font-serif">
            {/* Newspaper Folio Strip */}
            <div className="border-b border-slate-800 pb-1 mb-2 text-center">
              <div className="text-[11px] font-bold tracking-widest text-slate-200 uppercase">
                The Archival Gazette
              </div>
              <div className="text-[8px] font-mono text-slate-500 flex justify-between px-1">
                <span>VOL. IV · NO. 88</span>
                <span>150 DPI RASTER SCAN</span>
              </div>
            </div>

            {/* Simulated Multi-Column Newspaper Layout with Bounding Boxes */}
            <div className="space-y-2">
              {/* Lead Headline Banner */}
              <div
                onClick={() => setActiveStage('layout')}
                onMouseEnter={() => setActiveStage('layout')}
                className={`cursor-pointer transition-all duration-200 p-1.5 rounded border ${
                  activeStage === 'layout'
                    ? 'border-emerald-400 bg-emerald-500/15 shadow-[0_0_12px_rgba(16,185,129,0.3)]'
                    : 'border-emerald-500/40 bg-emerald-950/20 hover:border-emerald-400'
                }`}
              >
                <div className="flex items-center justify-between text-[8px] font-mono text-emerald-400 mb-0.5">
                  <span className="font-bold flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    LEAD STORY
                  </span>
                  <span>0.94 PROMINENCE</span>
                </div>
                <div className="h-2 w-3/4 bg-slate-300/80 rounded mb-1" />
                <div className="space-y-0.5">
                  <div className="h-1 w-full bg-slate-600/60 rounded" />
                  <div className="h-1 w-5/6 bg-slate-600/60 rounded" />
                </div>
              </div>

              {/* 2-Column Middle Row: Visual Photo & Markdown Table */}
              <div className="grid grid-cols-2 gap-2">
                {/* Photo Asset Box */}
                <div
                  onClick={() => setActiveStage('vlm')}
                  onMouseEnter={() => setActiveStage('vlm')}
                  className={`cursor-pointer transition-all duration-200 p-1.5 rounded border ${
                    activeStage === 'vlm'
                      ? 'border-amber-400 bg-amber-500/15 shadow-[0_0_12px_rgba(245,158,11,0.3)]'
                      : 'border-amber-500/40 bg-amber-950/20 hover:border-amber-400'
                  }`}
                >
                  <div className="flex items-center justify-between text-[7px] font-mono text-amber-400 mb-1">
                    <span className="font-bold">PHOTO VLM</span>
                    <span>150 DPI</span>
                  </div>
                  <div className="h-10 bg-slate-900 rounded border border-dashed border-amber-500/30 flex items-center justify-center text-[8px] text-amber-300/80 font-mono">
                    Scene Intel
                  </div>
                  <div className="h-1 w-full bg-slate-600/40 rounded mt-1" />
                </div>

                {/* Data Table / IPO Box */}
                <div
                  onClick={() => setActiveStage('vlm')}
                  onMouseEnter={() => setActiveStage('vlm')}
                  className={`cursor-pointer transition-all duration-200 p-1.5 rounded border ${
                    activeStage === 'vlm'
                      ? 'border-sky-400 bg-sky-500/15 shadow-[0_0_12px_rgba(56,189,248,0.3)]'
                      : 'border-sky-500/40 bg-sky-950/20 hover:border-sky-400'
                  }`}
                >
                  <div className="flex items-center justify-between text-[7px] font-mono text-sky-400 mb-1">
                    <span className="font-bold">FINANCIAL TABLE</span>
                    <span>MARKDOWN</span>
                  </div>
                  <div className="space-y-1 py-1">
                    <div className="flex gap-1">
                      <div className="h-1.5 flex-1 bg-sky-400/40 rounded" />
                      <div className="h-1.5 flex-1 bg-sky-400/30 rounded" />
                    </div>
                    <div className="flex gap-1">
                      <div className="h-1 flex-1 bg-slate-600/40 rounded" />
                      <div className="h-1 flex-1 bg-slate-600/40 rounded" />
                    </div>
                    <div className="flex gap-1">
                      <div className="h-1 flex-1 bg-slate-600/40 rounded" />
                      <div className="h-1 flex-1 bg-slate-600/40 rounded" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Bottom Advert / Statutory Notice Isolated */}
              <div
                onClick={() => setActiveStage('crag')}
                onMouseEnter={() => setActiveStage('crag')}
                className={`cursor-pointer transition-all duration-200 p-1.5 rounded border ${
                  activeStage === 'crag'
                    ? 'border-indigo-400 bg-indigo-500/15 shadow-[0_0_12px_rgba(99,102,241,0.3)]'
                    : 'border-slate-700 bg-slate-900/60 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center justify-between text-[7px] font-mono text-slate-400 mb-0.5">
                  <span className="font-medium">STATUTORY PUBLIC NOTICE</span>
                  <span className="text-slate-500">ISOLATED WALL</span>
                </div>
                <div className="h-1 w-full bg-slate-700/60 rounded" />
              </div>
            </div>

            {/* Bottom Coordinates HUD */}
            <div className="mt-2 pt-1 border-t border-slate-800/80 flex items-center justify-between text-[8px] font-mono text-slate-500">
              <span>SCAN FIDELITY: 1500x2150px</span>
              <span className="text-emerald-400">NORM [0.0–1.0]</span>
            </div>
          </div>
          <span className="text-[10px] font-mono text-slate-500 mt-2">
            Click broadsheet zones to inspect data paths
          </span>
        </div>

        {/* Right: 3 Engine Pipeline Nodes */}
        <div className="md:col-span-6 space-y-2.5">
          {/* Node 1: Docling 2D Layout */}
          <button
            type="button"
            onClick={() => {
              setActiveStage('layout');
              onSelectStage?.('ingest_layout');
            }}
            className={`w-full text-left p-3 rounded-xl border transition-all ${
              activeStage === 'layout'
                ? 'bg-slate-950 border-emerald-500 shadow-md shadow-emerald-950/40 ring-1 ring-emerald-500/30'
                : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/60'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
                  <Layout className="w-3.5 h-3.5" />
                </div>
                <span className="text-xs font-bold text-slate-100 font-sans">
                  Docling 2D Layout Engine
                </span>
              </div>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                150 DPI Scan
              </span>
            </div>
            <p className="text-[11px] text-slate-400 leading-snug">
              Convex walls isolate ads, cleans headline author decks, and extracts 2D reading order.
            </p>
          </button>

          {/* Node 2: Multimodal VLM Matrix */}
          <button
            type="button"
            onClick={() => {
              setActiveStage('vlm');
              onSelectStage?.('vision_extract');
            }}
            className={`w-full text-left p-3 rounded-xl border transition-all ${
              activeStage === 'vlm'
                ? 'bg-slate-950 border-sky-500 shadow-md shadow-sky-950/40 ring-1 ring-sky-500/30'
                : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/60'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-lg bg-sky-500/20 text-sky-400 flex items-center justify-center">
                  <Eye className="w-3.5 h-3.5" />
                </div>
                <span className="text-xs font-bold text-slate-100 font-sans">
                  Multimodal VLM Matrix
                </span>
              </div>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">
                Single-Pass
              </span>
            </div>
            <p className="text-[11px] text-slate-400 leading-snug">
              Converts complex charts & IPO tables to Markdown tables with 3–6 verifiable metrics.
            </p>
          </button>

          {/* Node 3: CRAG & Neural Reranker */}
          <button
            type="button"
            onClick={() => {
              setActiveStage('crag');
              onSelectStage?.('crag_agent');
            }}
            className={`w-full text-left p-3 rounded-xl border transition-all ${
              activeStage === 'crag'
                ? 'bg-slate-950 border-amber-500 shadow-md shadow-amber-950/40 ring-1 ring-amber-500/30'
                : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/60'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center">
                  <Bot className="w-3.5 h-3.5" />
                </div>
                <span className="text-xs font-bold text-slate-100 font-sans">
                  CRAG & Neural Reranker
                </span>
              </div>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                ~1.0s Cascade
              </span>
            </div>
            <p className="text-[11px] text-slate-400 leading-snug">
              7-archetype planner with two-stage Cross-Encoder neural rerank and zero-hallucination citations.
            </p>
          </button>
        </div>
      </div>

      {/* Active Stage Live Telemetry Footer Pill */}
      <div className="mt-4 pt-3 border-t border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-[11px]">ACTIVE STREAM:</span>
          <span className={`px-2 py-0.5 rounded font-bold text-[10px] ${current.badgeBg}`}>
            {current.title}
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-slate-300">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          <span>{current.metric}</span>
        </div>
      </div>
    </div>
  );
}
