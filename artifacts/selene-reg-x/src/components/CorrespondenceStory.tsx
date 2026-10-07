import { ArrowRight, CheckCircle2, ShieldCheck, Crosshair, Sparkles, Layers } from 'lucide-react';

interface CorrespondenceStoryProps {
  sourceUrl?: string;
  referenceUrl?: string;
  inliersCount: number;
  inlierRatio: number;
  rmse: number;
  homography?: number[][];
  sourcePoint?: [number, number];
  referencePoint?: [number, number];
  sourceName?: string;
  referenceName?: string;
  status: string;
}

export function CorrespondenceStory({
  sourceUrl,
  referenceUrl,
  inliersCount,
  inlierRatio,
  rmse,
  homography,
  sourcePoint,
  referencePoint,
  sourceName,
  referenceName,
  status,
}: CorrespondenceStoryProps) {
  const isPassed = status === 'PASS' || status === 'PASS_WITH_WARNING';

  // Format sample point coordinate
  const sx = sourcePoint ? sourcePoint[0].toFixed(2) : '—';
  const sy = sourcePoint ? sourcePoint[1].toFixed(2) : '—';
  const rx = referencePoint ? referencePoint[0].toFixed(2) : '—';
  const ry = referencePoint ? referencePoint[1].toFixed(2) : '—';

  return (
    <section className="bg-card border border-border rounded-lg p-4 shadow-xs transition-all duration-200">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/80 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="flex size-7 items-center justify-center rounded-md bg-emerald-500/10 text-emerald-500 border border-emerald-500/20/80 shadow-2xs">
            <ShieldCheck className="size-4 text-emerald-500" />
          </div>
          <div>
            <p className="eyebrow text-emerald-500">Verified Planetary Correspondence</p>
            <h3 className="text-sm font-display font-semibold tracking-tight text-card-foreground">
              Verified Lunar Terrain Feature Alignment Bridge
            </h3>
          </div>
        </div>
        <span
          className={`inline-flex items-center gap-1.5 rounded px-2.5 py-0.5 text-[10px] font-mono font-bold uppercase tracking-wider ${
            isPassed
              ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 shadow-2xs'
              : 'bg-destructive/10 text-destructive border border-destructive/20 shadow-2xs'
          }`}
        >
          <span className={`size-1.5 rounded-full ${isPassed ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
          {isPassed ? 'GEOMETRICALLY VERIFIED' : 'UNVERIFIED PAIR'}
        </span>
      </div>

      {/* 3-Stage Scientific Verification Bridge */}
      <div className="mt-3.5 grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
        {/* Stage 01: Moving Source Extract */}
        <div className="bg-card p-3 rounded border border-border hover:border-sky-300 transition-all duration-200 flex flex-col justify-between shadow-2xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono font-bold text-muted-foreground uppercase tracking-wider">
              STAGE 01 · SOURCE EXTRACT
            </span>
            <span className="text-[9px] font-mono font-semibold text-sky-800 bg-sky-50 px-1.5 py-0.5 rounded border border-sky-200">
              X:{sx} Y:{sy}
            </span>
          </div>
          <div className="relative aspect-video sm:aspect-[4/3] w-full overflow-hidden rounded border border-border bg-muted flex items-center justify-center group">
            {sourceUrl ? (
              <img
                src={sourceUrl}
                alt="Source feature crop"
                className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
              />
            ) : (
              <div className="text-xs text-muted-foreground">Source raster</div>
            )}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="size-8 rounded-full border border-amber-500 bg-amber-500/20 flex items-center justify-center animate-pulse">
                <Crosshair className="size-4 text-amber-600" />
              </div>
            </div>
            <div className="absolute bottom-1 left-1 px-1.5 py-0.2 rounded bg-black/60 text-white font-mono text-[9px] backdrop-blur-2xs">
              Feature Patch
            </div>
          </div>
          <p className="mt-2 truncate font-mono text-[10px] text-muted-foreground font-medium" title={sourceName}>
            {sourceName || 'moving_source.png'}
          </p>
        </div>

        {/* Stage 02: Forward Homography Transformation */}
        <div className="bg-card p-3 rounded border border-border hover:border-sky-300 transition-all duration-200 flex flex-col justify-between shadow-2xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono font-bold text-muted-foreground uppercase tracking-wider">
              STAGE 02 · FORWARD WARP H
            </span>
            <span className="text-[9px] font-mono font-bold text-emerald-500 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-200">
              x_ref = H · x_src
            </span>
          </div>
          <div className="aspect-video sm:aspect-[4/3] w-full rounded border border-border bg-card p-2.5 flex flex-col justify-between font-mono shadow-inner">
            <div className="text-[11px] font-semibold text-card-foreground border-b border-border pb-1 flex items-center justify-between">
              <span>Projective Alignment</span>
              <span className="text-[10px] text-sky-800 font-bold">Det ~ 1.0</span>
            </div>
            <div className="space-y-1 text-[11px] text-muted-foreground">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Verified inliers:</span>
                <span className="font-bold text-emerald-500">{inliersCount} pts</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Inlier ratio:</span>
                <span className="font-bold text-card-foreground">{(inlierRatio * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Residual RMSE:</span>
                <span className="font-bold text-sky-700">{rmse.toFixed(3)} px</span>
              </div>
            </div>
            <div className="text-[9px] text-muted-foreground font-mono pt-1 border-t border-border flex items-center justify-between">
              <span>Sub-pixel target</span>
              <span className="text-emerald-500 font-bold">&lt; 0.50 px</span>
            </div>
          </div>
          <p className="mt-2 font-mono text-[10px] text-muted-foreground flex items-center gap-1">
            <Layers className="size-3 text-sky-700" /> Planar Homography Projection
          </p>
        </div>

        {/* Stage 03: Fixed Reference Target Coincidence */}
        <div className="bg-card p-3 rounded border border-border hover:border-emerald-500/20 transition-all duration-200 flex flex-col justify-between shadow-2xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono font-bold text-muted-foreground uppercase tracking-wider">
              STAGE 03 · TARGET COINCIDENCE
            </span>
            <span className="text-[9px] font-mono font-semibold text-emerald-500 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-200">
              X:{rx} Y:{ry}
            </span>
          </div>
          <div className="relative aspect-video sm:aspect-[4/3] w-full overflow-hidden rounded border border-border bg-muted flex items-center justify-center group">
            {referenceUrl ? (
              <img
                src={referenceUrl}
                alt="Reference feature crop"
                className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
              />
            ) : (
              <div className="text-xs text-muted-foreground">Reference raster</div>
            )}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="size-8 rounded-full border border-emerald-500 bg-emerald-500/20 flex items-center justify-center animate-pulse">
                <Crosshair className="size-4 text-emerald-600" />
              </div>
            </div>
            <div className="absolute bottom-1 left-1 px-1.5 py-0.2 rounded bg-black/60 text-white font-mono text-[9px] backdrop-blur-2xs">
              Target Coincidence Lock
            </div>
          </div>
          <p className="mt-2 truncate font-mono text-[10px] text-muted-foreground font-medium" title={referenceName}>
            {referenceName || 'fixed_reference.png'}
          </p>
        </div>
      </div>
    </section>
  );
}
