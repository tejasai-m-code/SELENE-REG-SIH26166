import { useState, useRef } from 'react';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Download,
  Eye,
  CheckCircle2,
  XCircle,
  ShieldCheck,
  X,
  Target,
} from 'lucide-react';

interface MatchInspectorModalProps {
  isOpen: boolean;
  onClose: () => void;
  matchVisualizationUrl?: string;
  inliersCount: number;
  outliersCount: number;
  totalCandidateMatches: number;
  inlierRatio: number;
  spatialCoverage: number;
  rmse: number;
  sourceFilename?: string;
  referenceFilename?: string;
}

export function MatchInspectorModal({
  isOpen,
  onClose,
  matchVisualizationUrl,
  inliersCount,
  outliersCount,
  totalCandidateMatches,
  inlierRatio,
  spatialCoverage,
  rmse,
  sourceFilename,
  referenceFilename,
}: MatchInspectorModalProps) {
  const [zoomScale, setZoomScale] = useState(1.0);
  const [showInliers, setShowInliers] = useState(true);
  const [showOutliers, setShowOutliers] = useState(true);
  const [showRegion, setShowRegion] = useState(true);
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [showLabels, setShowLabels] = useState(true);

  if (!isOpen) return null;

  const handleZoom = (delta: number) => {
    setZoomScale((prev) => Math.max(0.5, Math.min(4.0, prev + delta)));
  };

  const handleReset = () => {
    setZoomScale(1.0);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/70 p-4 backdrop-blur-xs">
      <div className="flex h-[92vh] w-full max-w-6xl flex-col rounded-xl border border-border bg-card shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border bg-muted px-6 py-3.5">
          <div className="flex items-center gap-3">
            <div className="flex size-8 items-center justify-center rounded-lg bg-emerald-100 text-emerald-500 border border-emerald-500/20">
              <Target className="size-4" />
            </div>
            <div>
              <p className="eyebrow text-emerald-500">Geometric Inlier Inspection Bay</p>
              <h2 className="text-base font-bold text-card-foreground">
                Match Diagnostics & Verified Inlier Hull
              </h2>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-card-foreground transition"
          >
            <X className="size-5" />
          </button>
        </div>

        {/* Scientific Telemetry Ribbon */}
        <div className="grid grid-cols-2 sm:grid-cols-6 gap-2 border-b border-border bg-muted p-3 text-xs">
          <div className="rounded border border-border bg-card p-2">
            <span className="font-mono text-[9px] text-muted-foreground uppercase block">Candidate Matches</span>
            <span className="font-mono text-sm font-semibold text-card-foreground">{totalCandidateMatches}</span>
          </div>
          <div className="rounded border border-emerald-200 bg-emerald-500/10/70 p-2">
            <span className="font-mono text-[9px] text-emerald-500 uppercase block">Geometric Inliers</span>
            <span className="font-mono text-sm font-bold text-emerald-500">{inliersCount}</span>
          </div>
          <div className="rounded border border-rose-200 bg-destructive/10/70 p-2">
            <span className="font-mono text-[9px] text-destructive uppercase block">Rejected Outliers</span>
            <span className="font-mono text-sm font-bold text-rose-700">{outliersCount}</span>
          </div>
          <div className="rounded border border-border bg-card p-2">
            <span className="font-mono text-[9px] text-muted-foreground uppercase block">Inlier Ratio</span>
            <span className="font-mono text-sm font-semibold text-card-foreground">
              {(inlierRatio * 100).toFixed(1)}%
            </span>
          </div>
          <div className="rounded border border-border bg-card p-2">
            <span className="font-mono text-[9px] text-muted-foreground uppercase block">Spatial Coverage</span>
            <span className="font-mono text-sm font-semibold text-primary">
              {(spatialCoverage * 100).toFixed(1)}%
            </span>
          </div>
          <div className="rounded border border-border bg-card p-2">
            <span className="font-mono text-[9px] text-muted-foreground uppercase block">Inlier RMSE</span>
            <span className="font-mono text-sm font-semibold text-card-foreground">{rmse.toFixed(3)} px</span>
          </div>
        </div>

        {/* Canvas Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-muted/80 px-4 py-2 text-xs">
          {/* Toggles */}
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-1.5 font-medium text-emerald-500 cursor-pointer">
              <input
                type="checkbox"
                checked={showInliers}
                onChange={(e) => setShowInliers(e.target.checked)}
                className="accent-emerald-600"
              />
              <span className="size-2 rounded-full bg-emerald-500" />
              Inliers (Green)
            </label>
            <label className="flex items-center gap-1.5 font-medium text-destructive cursor-pointer">
              <input
                type="checkbox"
                checked={showOutliers}
                onChange={(e) => setShowOutliers(e.target.checked)}
                className="accent-rose-600"
              />
              <span className="size-2 rounded-full bg-rose-500" />
              Outliers (Red)
            </label>
            <label className="flex items-center gap-1.5 font-medium text-amber-500 cursor-pointer">
              <input
                type="checkbox"
                checked={showRegion}
                onChange={(e) => setShowRegion(e.target.checked)}
                className="accent-amber-600"
              />
              <span className="size-2 rounded-full bg-amber-500" />
              Correspondence Region (Hull)
            </label>
            <label className="flex items-center gap-1.5 font-medium text-muted-foreground cursor-pointer">
              <input
                type="checkbox"
                checked={showLabels}
                onChange={(e) => setShowLabels(e.target.checked)}
                className="accent-cyan-700"
              />
              Diagnostic Banner
            </label>
          </div>

          {/* Zoom & Viewport Controls */}
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => handleZoom(-0.25)}
              className="rounded border border-border bg-card p-1 hover:bg-muted"
              title="Zoom Out"
            >
              <ZoomOut className="size-4 text-muted-foreground" />
            </button>
            <span className="font-mono text-[10px] text-muted-foreground w-12 text-center">
              {Math.round(zoomScale * 100)}%
            </span>
            <button
              type="button"
              onClick={() => handleZoom(0.25)}
              className="rounded border border-border bg-card p-1 hover:bg-muted"
              title="Zoom In"
            >
              <ZoomIn className="size-4 text-muted-foreground" />
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="rounded border border-border bg-card px-2 py-1 text-[11px] font-medium text-muted-foreground hover:bg-muted"
            >
              Fit View
            </button>
            {matchVisualizationUrl && (
              <a
                href={matchVisualizationUrl}
                download="selene_match_diagnostics.png"
                className="inline-flex items-center gap-1 rounded bg-cyan-700 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-primary/20"
              >
                <Download className="size-3.5" /> Download
              </a>
            )}
          </div>
        </div>

        {/* Interactive Visualization Canvas */}
        <div className="flex-1 overflow-auto bg-muted flex items-center justify-center p-4">
          {matchVisualizationUrl ? (
            <div
              style={{
                transform: `scale(${zoomScale})`,
                transformOrigin: 'center center',
                transition: 'transform 0.15s ease-out',
              }}
              className="relative max-h-full max-w-full flex items-center justify-center"
            >
              <img
                src={matchVisualizationUrl}
                alt="Match Visualization Canvas"
                className="rounded shadow-xl object-contain"
              />
            </div>
          ) : (
            <div className="text-center text-muted-foreground">
              <p className="text-xs">No match visualization available.</p>
              <p className="text-[10px] text-muted-foreground mt-1">Run pairwise correspondence to render diagnostics.</p>
            </div>
          )}
        </div>

        {/* Footer Info */}
        <div className="flex items-center justify-between border-t border-border bg-card px-4 py-2.5 text-[11px] text-muted-foreground">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-card-foreground">Source:</span> {sourceFilename || 'moving_raster.png'}
            <span className="text-muted-foreground">→</span>
            <span className="font-semibold text-card-foreground">Reference:</span> {referenceFilename || 'fixed_raster.png'}
          </div>
          <div className="font-mono text-[10px] text-muted-foreground">
            Green: Inliers (&le; 3.0px) | Red: Outliers | Orange: 2D Projected Convex Hull
          </div>
        </div>
      </div>
    </div>
  );
}
