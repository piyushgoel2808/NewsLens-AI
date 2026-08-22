import React, { useState, useMemo } from 'react';

export default function CanvasOverlay({
  pageWidth = 2800,
  pageHeight = 4399,
  articles = [],
  selectedArticleId = null,
  hoveredArticleId = null,
  pulsingArticleId = null,
  customHighlightBboxes = [],
  onSelectArticle,
  onHoverArticle,
}) {
  const [tooltip, setTooltip] = useState(null);

  // Auto-detect coordinate domain: scale viewBox if bboxes are in PDF points (72 DPI) vs 300 DPI raster
  const { vbWidth, vbHeight } = useMemo(() => {
    let maxBx = 0;
    let maxBy = 0;

    articles.forEach((art) => {
      (art.bboxes || []).forEach((b) => {
        if (Array.isArray(b) && b.length >= 4) {
          if (b[2] > maxBx) maxBx = b[2];
          if (b[3] > maxBy) maxBy = b[3];
        }
      });
    });

    customHighlightBboxes.forEach((b) => {
      if (Array.isArray(b) && b.length >= 4) {
        if (b[2] > maxBx) maxBx = b[2];
        if (b[3] > maxBy) maxBy = b[3];
      }
    });

    // If coordinates are in 72 DPI PDF point space (< 1500) but raster is 300 DPI (> 1800)
    if (maxBx > 0 && maxBx < pageWidth * 0.65 && pageWidth > 1500) {
      const ptWidth = (pageWidth * 72) / 300;
      const ptHeight = (pageHeight * 72) / 300;
      return {
        vbWidth: ptWidth > maxBx ? ptWidth : maxBx * 1.05,
        vbHeight: ptHeight > maxBy ? ptHeight : maxBy * 1.05,
      };
    }

    return {
      vbWidth: pageWidth,
      vbHeight: pageHeight,
    };
  }, [articles, customHighlightBboxes, pageWidth, pageHeight]);

  const getColorStyles = (articleType, isSelected, isHovered, isPulse) => {
    const type = (articleType || '').toLowerCase();
    let baseStroke = 'rgb(34, 197, 94)'; // emerald-500
    let baseFill = 'rgba(34, 197, 94, 0.08)';
    let activeFill = 'rgba(34, 197, 94, 0.35)';

    if (type.includes('ad') || type.includes('notice')) {
      baseStroke = 'rgb(245, 158, 11)'; // amber-500
      baseFill = 'rgba(245, 158, 11, 0.08)';
      activeFill = 'rgba(245, 158, 11, 0.35)';
    } else if (type.includes('table') || type.includes('data')) {
      baseStroke = 'rgb(59, 130, 246)'; // blue-500
      baseFill = 'rgba(59, 130, 246, 0.1)';
      activeFill = 'rgba(59, 130, 246, 0.35)';
    } else if (type.includes('photo') || type.includes('image')) {
      baseStroke = 'rgb(168, 85, 247)'; // purple-500
      baseFill = 'rgba(168, 85, 247, 0.1)';
      activeFill = 'rgba(168, 85, 247, 0.35)';
    } else if (type.includes('toc') || type.includes('index')) {
      baseStroke = 'rgb(148, 163, 184)'; // slate-400
      baseFill = 'rgba(148, 163, 184, 0.08)';
      activeFill = 'rgba(148, 163, 184, 0.3)';
    }

    if (isPulse) {
      return {
        fill: 'rgba(239, 68, 68, 0.4)', // bright pulsing amber/red
        stroke: 'rgb(239, 68, 68)',
        strokeWidth: 6,
        strokeDasharray: '8 4',
      };
    }

    if (isSelected) {
      return {
        fill: activeFill,
        stroke: baseStroke,
        strokeWidth: 4,
      };
    }

    if (isHovered) {
      return {
        fill: activeFill,
        stroke: baseStroke,
        strokeWidth: 3,
      };
    }

    return {
      fill: baseFill,
      stroke: baseStroke,
      strokeWidth: 1.5,
      strokeOpacity: 0.7,
    };
  };

  return (
    <div className="absolute inset-0 w-full h-full pointer-events-none">
      <svg
        viewBox={`0 0 ${vbWidth} ${vbHeight}`}
        className="w-full h-full pointer-events-auto select-none"
        preserveAspectRatio="none"
      >
        <defs>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Render Article Bounding Boxes */}
        {articles.map((art) => {
          const isSelected = selectedArticleId === art.id;
          const isHovered = hoveredArticleId === art.id;
          const isPulse = pulsingArticleId === art.id;
          const styles = getColorStyles(art.article_type, isSelected, isHovered, isPulse);

          const bboxes = art.bboxes || [];

          return (
            <g
              key={art.id}
              className="cursor-pointer transition-all duration-150"
              onClick={(e) => {
                e.stopPropagation();
                if (onSelectArticle) onSelectArticle(art.id);
              }}
              onMouseEnter={(e) => {
                if (onHoverArticle) onHoverArticle(art.id);
                setTooltip({
                  headline: art.headline,
                  type: art.article_type || 'News',
                  x: e.clientX,
                  y: e.clientY - 40,
                });
              }}
              onMouseMove={(e) => {
                setTooltip({
                  headline: art.headline,
                  type: art.article_type || 'News',
                  x: e.clientX,
                  y: e.clientY - 40,
                });
              }}
              onMouseLeave={() => {
                if (onHoverArticle) onHoverArticle(null);
                setTooltip(null);
              }}
            >
              {bboxes.map((bbox, idx) => {
                const [x0, y0, x1, y1] = bbox;
                const width = Math.max(x1 - x0, 10);
                const height = Math.max(y1 - y0, 10);

                return (
                  <rect
                    key={`${art.id}-${idx}`}
                    x={x0}
                    y={y0}
                    width={width}
                    height={height}
                    fill={styles.fill}
                    stroke={styles.stroke}
                    strokeWidth={styles.strokeWidth}
                    strokeDasharray={styles.strokeDasharray}
                    strokeOpacity={styles.strokeOpacity}
                    filter={isPulse || isSelected ? 'url(#glow)' : undefined}
                    className={`transition-all duration-200 ${
                      isPulse ? 'animate-pulse' : ''
                    }`}
                  />
                );
              })}
            </g>
          );
        })}

        {/* Render Custom Highlight Bboxes (e.g. from citation badge clicks) */}
        {customHighlightBboxes.map((bbox, idx) => {
          const [x0, y0, x1, y1] = bbox;
          const width = Math.max(x1 - x0, 10);
          const height = Math.max(y1 - y0, 10);

          return (
            <rect
              key={`custom-${idx}`}
              x={x0}
              y={y0}
              width={width}
              height={height}
              fill="rgba(239, 68, 68, 0.45)"
              stroke="rgb(239, 68, 68)"
              strokeWidth={5}
              filter="url(#glow)"
              className="animate-pulse pointer-events-none"
            />
          );
        })}
      </svg>

      {/* Floating Canvas Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 pointer-events-none bg-slate-900/95 border border-slate-700 shadow-2xl px-3 py-1.5 rounded-md text-xs text-slate-100 max-w-xs backdrop-blur-sm transform -translate-x-1/2"
          style={{ left: `${tooltip.x}px`, top: `${tooltip.y}px` }}
        >
          <span className="font-semibold text-emerald-400 uppercase tracking-wider text-[10px] block mb-0.5">
            {tooltip.type}
          </span>
          <p className="line-clamp-2 leading-snug">{tooltip.headline || 'Untitled Article'}</p>
        </div>
      )}
    </div>
  );
}
