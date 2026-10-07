import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
  Eye,
  EyeOff,
  CheckCircle2,
  XCircle,
  ShieldCheck,
  Target,
  Sparkles,
  MousePointer2,
  Sliders,
  Crosshair,
  Layers,
  ChevronRight,
  Filter,
} from 'lucide-react';

export interface CorrespondencePoint {
  index: number;
  source: [number, number];
  reference: [number, number];
  initial_reference?: [number, number];
  refined_reference?: [number, number];
  refinement_delta?: [number, number];
  integer_coordinate?: [number, number];
  residual?: number;
  raw_residual?: number;
  refined_residual?: number;
  residual_improvement?: number;
  shift_magnitude?: number;
  iterations?: number;
  converged?: boolean;
  convergence_status?: string;
  local_correlation?: number;
  error?: number;
  confidence?: number;
  refinement_method?: string;
  status: 'INLIER' | 'OUTLIER';
  is_inlier: boolean;
  spatial_cell?: number;
  spatial_coverage?: string;
  source_cell?: number;
  reference_cell?: number;
}

export type CanonicalCorrespondence = CorrespondencePoint;

export interface ScientificCanonicalMatchViewerProps {
  sourceUrl?: string;
  referenceUrl?: string;
  sourceName?: string;
  referenceName?: string;
  matchVisualizationUrl?: string;
  correspondences?: CorrespondencePoint[];
  selectedMatchIndex?: number;
  onSelectMatch?: (index: number) => void;
  sourceDimensions?: { width: number; height: number };
  referenceDimensions?: { width: number; height: number };
  transformModel?: string;
  rmse?: number;
  inlierRatio?: number;
}

export function ScientificCanonicalMatchViewer({
  sourceUrl,
  referenceUrl,
  sourceName,
  referenceName,
  matchVisualizationUrl,
  correspondences = [],
  selectedMatchIndex = 0,
  onSelectMatch,
  sourceDimensions,
  referenceDimensions,
  transformModel = 'HOMOGRAPHY',
  rmse,
  inlierRatio,
}: ScientificCanonicalMatchViewerProps) {
  // Navigation State
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Display Toggles (Primary rule: no red rejected lines by default)
  const [showMatchLines, setShowMatchLines] = useState(true);
  const [showOutliersDiagnostic, setShowOutliersDiagnostic] = useState(false);
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [hoveredMatch, setHoveredMatch] = useState<CorrespondencePoint | null>(null);
  const [cursorPos, setCursorPos] = useState<{
    screenX: number;
    screenY: number;
    imageType: 'SOURCE' | 'REFERENCE' | 'NONE';
    imageX: number;
    imageY: number;
  } | null>(null);

  // Match Inspector Filtering State
  const [tableFilter, setTableFilter] = useState<'inliers' | 'all' | 'outliers'>('inliers');

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Cached image elements for offscreen rendering
  const [sourceImg, setSourceImg] = useState<HTMLImageElement | null>(null);
  const [referenceImg, setReferenceImg] = useState<HTMLImageElement | null>(null);
  const [matchVisImg, setMatchVisImg] = useState<HTMLImageElement | null>(null);

  // Load images
  useEffect(() => {
    if (sourceUrl) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = sourceUrl;
      img.onload = () => setSourceImg(img);
    }
  }, [sourceUrl]);

  useEffect(() => {
    if (referenceUrl) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = referenceUrl;
      img.onload = () => setReferenceImg(img);
    }
  }, [referenceUrl]);

  useEffect(() => {
    if (matchVisualizationUrl) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = matchVisualizationUrl;
      img.onload = () => setMatchVisImg(img);
    }
  }, [matchVisualizationUrl]);

  // Dimensions of canvas
  const srcW = sourceImg?.naturalWidth || sourceDimensions?.width || 1024;
  const srcH = sourceImg?.naturalHeight || sourceDimensions?.height || 1024;
  const refW = referenceImg?.naturalWidth || referenceDimensions?.width || 1024;
  const refH = referenceImg?.naturalHeight || referenceDimensions?.height || 1024;

  const totalVirtualWidth = srcW + refW;
  const totalVirtualHeight = Math.max(srcH, refH);

  // Filtered correspondences
  const inlierMatches = useMemo(() => correspondences.filter((c) => c.is_inlier), [correspondences]);
  const outlierMatches = useMemo(() => correspondences.filter((c) => !c.is_inlier), [correspondences]);

  const activeTableMatches = useMemo(() => {
    if (tableFilter === 'inliers') return inlierMatches;
    if (tableFilter === 'outliers') return outlierMatches;
    return correspondences;
  }, [tableFilter, inlierMatches, outlierMatches, correspondences]);

  const currentSelectedMatch = useMemo(() => {
    if (!correspondences.length) return null;
    return correspondences[selectedMatchIndex] || correspondences[0] || null;
  }, [correspondences, selectedMatchIndex]);

  // Coordinate transforms
  // Virtual space:
  // [0 .. srcW] = Source Image (Y: 0 .. srcH)
  // [srcW .. srcW + refW] = Reference Image (Y: 0 .. refH)
  const virtualToScreen = useCallback(
    (vx: number, vy: number, cw: number, ch: number) => {
      // Calculate fit scale
      const baseScale = Math.min(cw / totalVirtualWidth, ch / totalVirtualHeight) * 0.95;
      const effectiveScale = baseScale * zoom;

      const offsetX = (cw - totalVirtualWidth * effectiveScale) / 2 + pan.x;
      const offsetY = (ch - totalVirtualHeight * effectiveScale) / 2 + pan.y;

      return {
        x: vx * effectiveScale + offsetX,
        y: vy * effectiveScale + offsetY,
        scale: effectiveScale,
      };
    },
    [totalVirtualWidth, totalVirtualHeight, zoom, pan]
  );

  const screenToVirtual = useCallback(
    (sx: number, sy: number, cw: number, ch: number) => {
      const baseScale = Math.min(cw / totalVirtualWidth, ch / totalVirtualHeight) * 0.95;
      const effectiveScale = baseScale * zoom;

      const offsetX = (cw - totalVirtualWidth * effectiveScale) / 2 + pan.x;
      const offsetY = (ch - totalVirtualHeight * effectiveScale) / 2 + pan.y;

      return {
        vx: (sx - offsetX) / effectiveScale,
        vy: (sy - offsetY) / effectiveScale,
        scale: effectiveScale,
      };
    },
    [totalVirtualWidth, totalVirtualHeight, zoom, pan]
  );

  // Render on HTML5 Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const cw = rect.width;
    const ch = rect.height;

    if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
      canvas.width = cw * dpr;
      canvas.height = ch * dpr;
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cw, ch);

    // Aerospace dark clean neutral backdrop
    ctx.fillStyle = '#0B1119';
    ctx.fillRect(0, 0, cw, ch);

    // Compute coordinate mapping
    const baseScale = Math.min(cw / totalVirtualWidth, ch / totalVirtualHeight) * 0.95;
    const effectiveScale = baseScale * zoom;
    const offsetX = (cw - totalVirtualWidth * effectiveScale) / 2 + pan.x;
    const offsetY = (ch - totalVirtualHeight * effectiveScale) / 2 + pan.y;

    // 1. Draw Images
    const srcScreenW = srcW * effectiveScale;
    const srcScreenH = srcH * effectiveScale;
    const refScreenW = refW * effectiveScale;
    const refScreenH = refH * effectiveScale;

    const srcScreenX = offsetX;
    const srcScreenY = offsetY;
    const refScreenX = offsetX + srcScreenW;
    const refScreenY = offsetY;

    // Draw Source Image
    if (sourceImg) {
      ctx.drawImage(sourceImg, srcScreenX, srcScreenY, srcScreenW, srcScreenH);
    } else {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(srcScreenX, srcScreenY, srcScreenW, srcScreenH);
    }
    // Source Boundary
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(srcScreenX, srcScreenY, srcScreenW, srcScreenH);

    // Draw Reference Image
    if (referenceImg) {
      ctx.drawImage(referenceImg, refScreenX, refScreenY, refScreenW, refScreenH);
    } else {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(refScreenX, refScreenY, refScreenW, refScreenH);
    }
    // Reference Boundary
    ctx.strokeStyle = '#818cf8';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(refScreenX, refScreenY, refScreenW, refScreenH);

    // Image Header Badges on Canvas
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(srcScreenX + 4, srcScreenY + 4, 180, 22);
    ctx.fillStyle = '#38bdf8';
    ctx.fillText(`SOURCE (MOVING) ${srcW}x${srcH}`, srcScreenX + 10, srcScreenY + 19);

    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(refScreenX + 4, refScreenY + 4, 195, 22);
    ctx.fillStyle = '#818cf8';
    ctx.fillText(`REFERENCE (FIXED) ${refW}x${refH}`, refScreenX + 10, refScreenY + 19);

    // Divider Line between rasters
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(refScreenX, Math.min(srcScreenY, refScreenY));
    ctx.lineTo(refScreenX, Math.max(srcScreenY + srcScreenH, refScreenY + refScreenH));
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Draw Correspondence Lines
    if (showMatchLines && correspondences.length > 0) {
      // Outliers first if diagnostic toggle is active
      if (showOutliersDiagnostic) {
        outlierMatches.forEach((c) => {
          const isSelected = currentSelectedMatch?.index === c.index;
          const isHovered = hoveredMatch?.index === c.index;

          const p1 = virtualToScreen(c.source[0], c.source[1], cw, ch);
          const p2 = virtualToScreen(srcW + c.reference[0], c.reference[1], cw, ch);

          ctx.strokeStyle = isSelected ? '#fbbf24' : isHovered ? '#f87171' : 'rgba(239, 68, 68, 0.40)';
          ctx.lineWidth = isSelected || isHovered ? 2.5 : 1;

          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();

          if (showKeypoints) {
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.arc(p1.x, p1.y, 2.5, 0, Math.PI * 2);
            ctx.arc(p2.x, p2.y, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }

      // Verified Inliers prominently drawn in green
      inlierMatches.forEach((c) => {
        const isSelected = currentSelectedMatch?.index === c.index;
        const isHovered = hoveredMatch?.index === c.index;

        const p1 = virtualToScreen(c.source[0], c.source[1], cw, ch);
        const p2 = virtualToScreen(srcW + c.reference[0], c.reference[1], cw, ch);

        ctx.strokeStyle = isSelected
          ? '#f59e0b' // Gold for active selection
          : isHovered
          ? '#06b6d4' // Cyan for hover
          : 'rgba(16, 185, 129, 0.85)'; // Clean emerald green for verified inliers
        ctx.lineWidth = isSelected ? 3.0 : isHovered ? 2.5 : 1.6;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();

        if (showKeypoints) {
          // Source point marker
          ctx.fillStyle = isSelected ? '#f59e0b' : isHovered ? '#06b6d4' : '#10b981';
          ctx.beginPath();
          ctx.arc(p1.x, p1.y, isSelected ? 5.5 : isHovered ? 4.5 : 3.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1;
          ctx.stroke();

          // Reference point marker
          ctx.beginPath();
          ctx.arc(p2.x, p2.y, isSelected ? 5.5 : isHovered ? 4.5 : 3.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      });
    }

    ctx.restore();
  }, [
    sourceImg,
    referenceImg,
    correspondences,
    inlierMatches,
    outlierMatches,
    selectedMatchIndex,
    currentSelectedMatch,
    hoveredMatch,
    showMatchLines,
    showOutliersDiagnostic,
    showKeypoints,
    zoom,
    pan,
    srcW,
    srcH,
    refW,
    refH,
    totalVirtualWidth,
    totalVirtualHeight,
    virtualToScreen,
  ]);

  // Pointer interactions: Zoom via Wheel
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setZoom((prev) => Math.max(0.4, Math.min(8.0, prev * factor)));
  };

  // Pointer Drag: Pan
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cw = rect.width;
    const ch = rect.height;
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }

    // Accurate Coordinate Mapping for Telemetry Readout
    const { vx, vy } = screenToVirtual(sx, sy, cw, ch);

    let imgType: 'SOURCE' | 'REFERENCE' | 'NONE' = 'NONE';
    let imgX = 0;
    let imgY = 0;

    if (vx >= 0 && vx <= srcW && vy >= 0 && vy <= srcH) {
      imgType = 'SOURCE';
      imgX = vx;
      imgY = vy;
    } else if (vx >= srcW && vx <= srcW + refW && vy >= 0 && vy <= refH) {
      imgType = 'REFERENCE';
      imgX = vx - srcW;
      imgY = vy;
    }

    setCursorPos({
      screenX: sx,
      screenY: sy,
      imageType: imgType,
      imageX: imgX,
      imageY: imgY,
    });

    // Proximity search for hover tooltip
    if (correspondences.length > 0) {
      let closestMatch: CorrespondencePoint | null = null;
      let minDistance = 14; // pixels in screen space

      const targetList = showOutliersDiagnostic ? correspondences : inlierMatches;

      for (const c of targetList) {
        const p1 = virtualToScreen(c.source[0], c.source[1], cw, ch);
        const p2 = virtualToScreen(srcW + c.reference[0], c.reference[1], cw, ch);

        // Distance to endpoint 1
        const d1 = Math.hypot(sx - p1.x, sy - p1.y);
        // Distance to endpoint 2
        const d2 = Math.hypot(sx - p2.x, sy - p2.y);

        // Distance to line segment
        const lineLen = Math.hypot(p2.x - p1.x, p2.y - p1.y);
        let distToLine = 999;
        if (lineLen > 0.001) {
          const u = Math.max(0, Math.min(1, ((sx - p1.x) * (p2.x - p1.x) + (sy - p1.y) * (p2.y - p1.y)) / (lineLen * lineLen)));
          const projX = p1.x + u * (p2.x - p1.x);
          const projY = p1.y + u * (p2.y - p1.y);
          distToLine = Math.hypot(sx - projX, sy - projY);
        }

        const pointMin = Math.min(d1, d2, distToLine);
        if (pointMin < minDistance) {
          minDistance = pointMin;
          closestMatch = c;
        }
      }

      setHoveredMatch(closestMatch);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Click on correspondence to select it
  const handleClick = (e: React.MouseEvent) => {
    if (hoveredMatch && onSelectMatch) {
      const matchIdx = correspondences.findIndex((c) => c.index === hoveredMatch.index);
      if (matchIdx !== -1) {
        onSelectMatch(matchIdx);
      }
    }
  };

  const handleResetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  const handleFitToView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  const handleZoomIn = () => setZoom((z) => Math.min(8.0, z * 1.25));
  const handleZoomOut = () => setZoom((z) => Math.max(0.4, z * 0.8));

  return (
    <div className="space-y-4">
      {/* Primary Canonical View Hero Section */}
      <section className="panel rounded-xl p-4 border border-border bg-card shadow-xs">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3 border-b border-border/80 pb-3 bg-gradient-to-b from-card to-muted/50">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <p className="eyebrow text-sky-800">Primary Planetary Visualization</p>
            </div>
            <h3 className="mt-0.5 text-base font-display font-bold text-card-foreground">
              Canonical Correspondence Canvas
            </h3>
            <p className="text-[11px] font-mono text-muted-foreground">
              Dual-raster dual-coordinate projection · Inlier vectors &middot; Sub-pixel delta analysis
            </p>
          </div>

          {/* Quick Metrics Bar */}
          <div className="flex items-center gap-2 font-mono">
            <div className="flex items-center gap-1.5 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 text-xs shadow-2xs">
              <span className="size-1.5 rounded-full bg-emerald-600" />
              <span className="font-bold text-emerald-950">{inlierMatches.length}</span>
              <span className="text-emerald-500 font-medium">verified inliers</span>
            </div>
            {rmse !== undefined && (
              <div className="rounded-md border border-border bg-card px-2.5 py-1 text-xs shadow-2xs">
                <span className="text-muted-foreground">RMSE: </span>
                <span className="font-bold text-sky-700">{rmse.toFixed(3)} px</span>
              </div>
            )}
            <div className="rounded-md border border-border bg-card px-2.5 py-1 text-xs shadow-2xs">
              <span className="text-muted-foreground">Model: </span>
              <span className="font-bold text-card-foreground">{transformModel}</span>
            </div>
          </div>
        </div>

        {/* Interactive Viewer Toolbar */}
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-muted/90 p-1.5 text-xs font-mono">
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={handleZoomIn}
              className="rounded bg-card border border-border p-1 text-muted-foreground hover:bg-muted hover:border-border transition shadow-2xs"
              title="Zoom In (+)"
            >
              <ZoomIn className="size-3.5" />
            </button>
            <button
              type="button"
              onClick={handleZoomOut}
              className="rounded bg-card border border-border p-1 text-muted-foreground hover:bg-muted hover:border-border transition shadow-2xs"
              title="Zoom Out (-)"
            >
              <ZoomOut className="size-3.5" />
            </button>
            <span className="font-mono px-2 py-0.5 rounded bg-card border border-border text-[11px] font-bold text-card-foreground shadow-2xs">
              {Math.round(zoom * 100)}%
            </span>
            <button
              type="button"
              onClick={handleFitToView}
              className="rounded bg-card border border-border px-2 py-1 text-muted-foreground hover:bg-muted hover:border-border transition font-medium text-[11px] shadow-2xs"
            >
              Fit View
            </button>
            <button
              type="button"
              onClick={handleResetView}
              className="rounded bg-card border border-border p-1 text-muted-foreground hover:bg-muted hover:border-border transition shadow-2xs"
              title="Reset View"
            >
              <RotateCcw className="size-3.5" />
            </button>
          </div>

          {/* Visualization Layer Toggles */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowMatchLines((v) => !v)}
              className={`inline-flex items-center gap-1.5 rounded px-2.5 py-1 font-semibold transition text-[11px] shadow-2xs ${
                showMatchLines ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20' : 'bg-card text-muted-foreground border border-border'
              }`}
            >
              <span className={`size-1.5 rounded-full ${showMatchLines ? 'bg-emerald-500' : 'bg-slate-400'}`} />
              {showMatchLines ? 'Correspondence Lines' : 'Lines Hidden'}
            </button>

            {/* Diagnostic Outlier Toggle (Hidden by default for clean view) */}
            <button
              type="button"
              data-testid="toggle-outliers-diagnostic"
              onClick={() => setShowOutliersDiagnostic((v) => !v)}
              className={`inline-flex items-center gap-1 rounded px-2.5 py-1 font-medium transition text-[11px] shadow-2xs ${
                showOutliersDiagnostic
                  ? 'bg-destructive/10 text-destructive border border-destructive/20 font-bold'
                  : 'bg-card border border-border text-muted-foreground hover:bg-muted'
              }`}
              title="Toggle diagnostic rejected outliers (hidden by default to avoid visual noise)"
            >
              <Crosshair className="size-3 text-muted-foreground" />
              {showOutliersDiagnostic ? `Outliers (${outlierMatches.length})` : 'Inliers Only'}
            </button>
          </div>
        </div>

        {/* Main Canvas Viewport with Tooltip & Live Cursor Readout */}
        <div
          ref={containerRef}
          className="relative h-[560px] w-full overflow-hidden rounded-lg border border-border bg-muted shadow-inner cursor-crosshair"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={handleClick}
        >
          <canvas ref={canvasRef} className="h-full w-full block" />

          {/* Interactive Hover Tooltip */}
          {hoveredMatch && cursorPos && (
            <div
              className="pointer-events-none absolute z-20 rounded-lg border border-border bg-card/95 p-3 text-white shadow-2xl backdrop-blur-md"
              style={{
                left: Math.min(cursorPos.screenX + 16, (containerRef.current?.clientWidth || 800) - 240),
                top: Math.max(10, Math.min(cursorPos.screenY - 30, (containerRef.current?.clientHeight || 500) - 180)),
                width: '230px',
              }}
            >
              <div className="flex items-center justify-between border-b border-border pb-1.5 mb-1.5">
                <span className="font-mono text-xs font-bold text-amber-500">
                  CORRESPONDENCE #{hoveredMatch.index}
                </span>
                <span
                  className={`rounded px-1.5 py-0.5 text-[9px] font-bold uppercase ${
                    hoveredMatch.is_inlier ? 'bg-emerald-950 text-emerald-500 border border-emerald-800' : 'bg-rose-950 text-destructive'
                  }`}
                >
                  {hoveredMatch.status}
                </span>
              </div>
              <div className="space-y-1 font-mono text-[10px] text-muted-foreground">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Source:</span>
                  <span className="text-cyan-300">({hoveredMatch.source[0].toFixed(2)}, {hoveredMatch.source[1].toFixed(2)})</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Reference:</span>
                  <span className="text-indigo-300">({hoveredMatch.reference[0].toFixed(2)}, {hoveredMatch.reference[1].toFixed(2)})</span>
                </div>
                {hoveredMatch.residual !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Residual:</span>
                    <span className="font-semibold text-emerald-500">{hoveredMatch.residual.toFixed(3)} px</span>
                  </div>
                )}
                {hoveredMatch.confidence !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Confidence:</span>
                    <span className="text-foreground">{(hoveredMatch.confidence * 100).toFixed(1)}%</span>
                  </div>
                )}
                {hoveredMatch.refinement_method && hoveredMatch.refinement_method !== 'none' && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Sub-pixel:</span>
                    <span className="text-amber-300">{hoveredMatch.refinement_method}</span>
                  </div>
                )}
                {hoveredMatch.spatial_coverage && (
                  <div className="mt-1 border-t border-border pt-1 text-[9px] text-muted-foreground">
                    Coverage: {hoveredMatch.spatial_coverage}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Live Cursor Coordinate Telemetry Badge */}
          <div className="absolute bottom-2 left-2 z-10 flex items-center gap-3 rounded-md bg-card/95 border border-border px-3 py-1.5 text-xs text-card-foreground shadow-md backdrop-blur-xs font-mono">
            {cursorPos && cursorPos.imageType !== 'NONE' ? (
              <span className="text-primary font-semibold">
                {cursorPos.imageType}: X = {cursorPos.imageX.toFixed(1)} px, Y = {cursorPos.imageY.toFixed(1)} px
              </span>
            ) : (
              <span className="text-muted-foreground">Hover over image for coordinates</span>
            )}
            <span className="text-muted-foreground">|</span>
            <span className="text-muted-foreground">Scale: {zoom.toFixed(2)}x</span>
            {hoveredMatch && (
              <>
                <span className="text-muted-foreground">|</span>
                <span className="text-amber-300 font-semibold">Match #{hoveredMatch.index} (Click to inspect)</span>
              </>
            )}
          </div>
        </div>
      </section>

      {/* Section 2: Interactive Match Inspector Panel */}
      <section className="panel rounded-xl p-4 border border-border bg-card shadow-xs">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-2.5">
          <div>
            <div className="flex items-center gap-2">
              <Target className="size-4 text-primary" />
              <p className="eyebrow text-primary">Match Inspector</p>
            </div>
            <h3 className="mt-0.5 text-base font-semibold text-card-foreground">
              Verified Correspondence Evidence & Sub-pixel Telemetry
            </h3>
          </div>

          {/* Filter Pills */}
          <div className="flex items-center rounded-lg border border-border bg-muted p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setTableFilter('inliers')}
              className={`rounded-md px-2.5 py-1 font-medium transition ${
                tableFilter === 'inliers' ? 'bg-card text-emerald-500 shadow-2xs' : 'text-muted-foreground hover:text-card-foreground'
              }`}
            >
              Verified Inliers ({inlierMatches.length})
            </button>
            <button
              type="button"
              onClick={() => setTableFilter('all')}
              className={`rounded-md px-2.5 py-1 font-medium transition ${
                tableFilter === 'all' ? 'bg-card text-card-foreground shadow-2xs' : 'text-muted-foreground hover:text-card-foreground'
              }`}
            >
              All Matches ({correspondences.length})
            </button>
            <button
              type="button"
              onClick={() => setTableFilter('outliers')}
              className={`rounded-md px-2.5 py-1 font-medium transition ${
                tableFilter === 'outliers' ? 'bg-card text-destructive shadow-2xs' : 'text-muted-foreground hover:text-card-foreground'
              }`}
            >
              Outliers ({outlierMatches.length})
            </button>
          </div>
        </div>

        {/* Selected Match Details Card & Match Table Grid */}
        <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
          {/* Card: Active Selected Match Deep Dive */}
          {currentSelectedMatch ? (
            <div className="rounded-lg border border-primary/20 bg-primary/10/40 p-3.5 space-y-3">
              <div className="flex items-center justify-between border-b border-primary/20/80 pb-2">
                <span className="font-mono text-xs font-bold text-cyan-950">
                  SELECTED MATCH #{currentSelectedMatch.index}
                </span>
                <span
                  className={`rounded px-2 py-0.5 text-[10px] font-bold uppercase ${
                    currentSelectedMatch.is_inlier
                      ? 'bg-emerald-100 text-emerald-500 border border-emerald-500/20'
                      : 'bg-rose-100 text-destructive border border-destructive/20'
                  }`}
                >
                  {currentSelectedMatch.status}
                </span>
              </div>

              {/* Coordinates Grid */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded border border-border bg-card p-2">
                  <span className="font-mono text-[9px] text-muted-foreground uppercase block">Source (Moving)</span>
                  <span className="font-mono text-xs font-semibold text-card-foreground">
                    X: {currentSelectedMatch.source[0].toFixed(3)}
                  </span>
                  <span className="font-mono text-xs font-semibold text-card-foreground block">
                    Y: {currentSelectedMatch.source[1].toFixed(3)}
                  </span>
                </div>
                <div className="rounded border border-border bg-card p-2">
                  <span className="font-mono text-[9px] text-muted-foreground uppercase block">Reference (Fixed)</span>
                  <span className="font-mono text-xs font-semibold text-card-foreground">
                    X: {currentSelectedMatch.reference[0].toFixed(3)}
                  </span>
                  <span className="font-mono text-xs font-semibold text-card-foreground block">
                    Y: {currentSelectedMatch.reference[1].toFixed(3)}
                  </span>
                </div>
              </div>

              {/* Residual and Confidence */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded border border-border bg-card p-2">
                  <span className="font-mono text-[9px] text-muted-foreground uppercase block">Reprojection Residual</span>
                  <span className="font-mono text-sm font-bold text-emerald-500">
                    {currentSelectedMatch.residual !== undefined ? `${currentSelectedMatch.residual.toFixed(3)} px` : 'N/A'}
                  </span>
                </div>
                <div className="rounded border border-border bg-card p-2">
                  <span className="font-mono text-[9px] text-muted-foreground uppercase block">Match Confidence</span>
                  <span className="font-mono text-sm font-bold text-card-foreground">
                    {currentSelectedMatch.confidence !== undefined
                      ? `${(currentSelectedMatch.confidence * 100).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
              </div>

              {/* Sub-pixel Details (Section 8 & 9) */}
              <div className="rounded border border-border bg-card p-2.5 text-xs space-y-1.5">
                <span className="font-mono text-[9px] text-muted-foreground uppercase block font-semibold">Sub-pixel Inspection Telemetry</span>
                <div className="flex justify-between font-mono text-[11px]">
                  <span className="text-muted-foreground">Method:</span>
                  <span className="font-semibold text-primary">{currentSelectedMatch.refinement_method || 'none'}</span>
                </div>
                {currentSelectedMatch.integer_coordinate && (
                  <div className="flex justify-between font-mono text-[11px]">
                    <span className="text-muted-foreground">Integer Coordinate:</span>
                    <span className="text-muted-foreground">
                      [{currentSelectedMatch.integer_coordinate[0]}, {currentSelectedMatch.integer_coordinate[1]}]
                    </span>
                  </div>
                )}
                {currentSelectedMatch.refined_reference && (
                  <div className="flex justify-between font-mono text-[11px]">
                    <span className="text-muted-foreground">Refined Coordinate:</span>
                    <span className="text-indigo-800 font-semibold">
                      [{currentSelectedMatch.refined_reference[0].toFixed(3)}, {currentSelectedMatch.refined_reference[1].toFixed(3)}]
                    </span>
                  </div>
                )}
                {currentSelectedMatch.refinement_delta && (
                  <div className="flex justify-between font-mono text-[11px]">
                    <span className="text-muted-foreground">Sub-pixel Shift (dx, dy):</span>
                    <span className="text-card-foreground font-medium">
                      ({currentSelectedMatch.refinement_delta[0].toFixed(3)}, {currentSelectedMatch.refinement_delta[1].toFixed(3)}) px
                    </span>
                  </div>
                )}
                {currentSelectedMatch.shift_magnitude !== undefined && (
                  <div className="flex justify-between font-mono text-[11px]">
                    <span className="text-muted-foreground">Shift Magnitude (||Δ||):</span>
                    <span className="text-amber-700 font-semibold">
                      {currentSelectedMatch.shift_magnitude.toFixed(3)} px
                    </span>
                  </div>
                )}
                {currentSelectedMatch.raw_residual !== undefined && currentSelectedMatch.residual !== undefined && (
                  <div className="flex justify-between font-mono text-[11px]">
                    <span className="text-muted-foreground">Residual Before → After:</span>
                    <span className="text-emerald-500 font-semibold">
                      {currentSelectedMatch.raw_residual.toFixed(3)} px → {currentSelectedMatch.residual.toFixed(3)} px
                      {currentSelectedMatch.residual_improvement !== undefined && currentSelectedMatch.residual_improvement > 0 ? (
                        <span className="text-emerald-500 ml-1">(-{currentSelectedMatch.residual_improvement.toFixed(3)} px)</span>
                      ) : null}
                    </span>
                  </div>
                )}
                <div className="flex justify-between font-mono text-[11px]">
                  <span className="text-muted-foreground">Convergence / Iterations:</span>
                  <span className={currentSelectedMatch.converged ?? true ? 'text-emerald-500 font-medium' : 'text-amber-700 font-medium'}>
                    {currentSelectedMatch.convergence_status || (currentSelectedMatch.converged ?? true ? 'CONVERGED' : 'DIVERGED')}
                    {currentSelectedMatch.iterations ? ` (${currentSelectedMatch.iterations} iters)` : ''}
                  </span>
                </div>
                {currentSelectedMatch.local_correlation !== undefined && currentSelectedMatch.local_correlation !== null && (
                  <div className="flex justify-between font-mono text-[11px]">
                    <span className="text-muted-foreground">Local Patch Correlation:</span>
                    <span className="text-card-foreground">
                      {currentSelectedMatch.local_correlation.toFixed(3)}
                    </span>
                  </div>
                )}
              </div>

              {/* Spatial Coverage 4x4 Mini-Grid Visualization */}
              <div className="rounded border border-border bg-card p-2.5 text-xs">
                <span className="font-mono text-[9px] text-muted-foreground uppercase block mb-1.5">
                  Spatial Distribution Grid ({currentSelectedMatch.spatial_coverage || 'Cell 1..16'})
                </span>
                <div className="grid grid-cols-4 gap-1 w-32 mx-auto">
                  {Array.from({ length: 16 }, (_, i) => i + 1).map((cellNum) => {
                    const isCell = currentSelectedMatch.spatial_cell === cellNum;
                    return (
                      <div
                        key={cellNum}
                        className={`h-5 rounded flex items-center justify-center font-mono text-[9px] ${
                          isCell
                            ? 'bg-emerald-600 text-white font-bold ring-2 ring-emerald-400'
                            : 'bg-muted text-muted-foreground'
                        }`}
                      >
                        {cellNum}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-border p-8 text-center text-xs text-muted-foreground">
              No correspondence selected. Click a match point or table row to inspect.
            </div>
          )}

          {/* Table of Correspondences */}
          <div className="max-h-[360px] overflow-auto rounded-lg border border-border">
            <table className="w-full text-left text-[11px]">
              <thead className="sticky top-0 bg-muted text-[10px] uppercase tracking-wider text-muted-foreground font-semibold border-b border-border">
                <tr>
                  <th className="px-3 py-2">#</th>
                  <th className="px-3 py-2">Source (x, y)</th>
                  <th className="px-3 py-2">Reference (x, y)</th>
                  <th className="px-3 py-2">Residual (px)</th>
                  <th className="px-3 py-2">Confidence</th>
                  <th className="px-3 py-2">Method</th>
                  <th className="px-3 py-2">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-[10px]">
                {activeTableMatches.map((m, idx) => {
                  const isSelected = currentSelectedMatch?.index === m.index;
                  return (
                    <tr
                      key={m.index}
                      onClick={() => {
                        const originalIndex = correspondences.findIndex((c) => c.index === m.index);
                        if (originalIndex !== -1 && onSelectMatch) {
                          onSelectMatch(originalIndex);
                        }
                      }}
                      className={`cursor-pointer transition ${
                        isSelected
                          ? 'bg-amber-500/10 font-semibold text-amber-500'
                          : 'hover:bg-muted text-muted-foreground'
                      }`}
                    >
                      <td className="px-3 py-1.5 font-bold text-card-foreground">{m.index}</td>
                      <td className="px-3 py-1.5">
                        ({m.source[0].toFixed(2)}, {m.source[1].toFixed(2)})
                      </td>
                      <td className="px-3 py-1.5">
                        ({m.reference[0].toFixed(2)}, {m.reference[1].toFixed(2)})
                      </td>
                      <td className="px-3 py-1.5 font-semibold text-emerald-500">
                        {m.residual !== undefined ? m.residual.toFixed(3) : '—'}
                      </td>
                      <td className="px-3 py-1.5">
                        {m.confidence !== undefined ? `${(m.confidence * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="px-3 py-1.5 text-muted-foreground">{m.refinement_method || 'none'}</td>
                      <td className="px-3 py-1.5">
                        <span
                          className={`rounded px-1.5 py-0.5 text-[9px] font-bold uppercase ${
                            m.is_inlier ? 'bg-emerald-100 text-emerald-500' : 'bg-rose-100 text-destructive'
                          }`}
                        >
                          {m.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
