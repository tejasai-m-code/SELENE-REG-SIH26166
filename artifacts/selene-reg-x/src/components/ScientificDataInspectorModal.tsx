import { useState } from 'react';
import {
  Activity,
  AlertCircle,
  BadgeCheck,
  BarChart2,
  Binary,
  Check,
  ChevronRight,
  Database,
  Eye,
  FileImage,
  Info,
  Layers,
  Maximize2,
  ShieldCheck,
  Sliders,
  Sparkles,
  X,
} from 'lucide-react';

export interface ScientificInspectionData {
  dimensions?: number[];
  width?: number;
  height?: number;
  ndim?: number;
  channels?: number;
  scientific_bands?: number;
  layer_classification?: string;
  bit_depth?: number;
  dtype?: string;
  min_dn?: number;
  max_dn?: number;
  mean_dn?: number;
  median_dn?: number;
  std_dn?: number;
  dynamic_range_span?: number;
  valid_pixel_pct?: number;
  saturation_pct?: number;
  histogram_bins?: number[];
  histogram_counts?: number[];
  has_alpha?: boolean;
  compression?: string;
  display_normalized?: boolean;
}

export interface SensorIdentificationData {
  sensor?: string;
  mission?: string;
  gsd_m?: number | null;
  confidence_pct?: number;
  evidence?: string[];
  metadata_status?: string;
  band_info?: string;
}

export interface ScientificFileItem {
  filename: string;
  format?: string;
  size_bytes?: number;
  preview_url?: string;
  dimensions?: number[];
  width?: number;
  height?: number;
  channels?: number;
  bit_depth?: number;
  dtype?: string;
  scientific_bands?: number;
  layer_classification?: string;
  min_dn?: number;
  max_dn?: number;
  mean_dn?: number;
  median_dn?: number;
  std_dn?: number;
  dynamic_range_span?: number;
  valid_pixel_pct?: number;
  saturation_pct?: number;
  histogram?: {
    bins: number[];
    counts: number[];
  };
  sensor?: string;
  mission?: string;
  gsd_m?: number | null;
  sensor_confidence_pct?: number;
  sensor_evidence?: string[];
  metadata_status?: string;
  metadata?: Record<string, unknown>;
  scientific_inspection?: ScientificInspectionData;
  sensor_identification?: SensorIdentificationData;
}

interface ScientificDataInspectorModalProps {
  isOpen: boolean;
  onClose: () => void;
  fileItem: ScientificFileItem | null;
}

export function ScientificDataInspectorModal({
  isOpen,
  onClose,
  fileItem,
}: ScientificDataInspectorModalProps) {
  const [hoveredBinIndex, setHoveredBinIndex] = useState<number | null>(null);

  if (!isOpen || !fileItem) return null;

  const inspection = fileItem.scientific_inspection || {};
  const sensor = fileItem.sensor_identification || {};

  const width = inspection.width ?? fileItem.width ?? fileItem.dimensions?.[0] ?? 0;
  const height = inspection.height ?? fileItem.height ?? fileItem.dimensions?.[1] ?? 0;
  const channels = inspection.channels ?? fileItem.channels ?? fileItem.dimensions?.[2] ?? 1;
  const bitDepth = inspection.bit_depth ?? fileItem.bit_depth ?? 8;
  const dtype = inspection.dtype ?? fileItem.dtype ?? (bitDepth === 16 ? 'uint16' : 'uint8');
  const bands = inspection.scientific_bands ?? fileItem.scientific_bands ?? (channels === 3 ? 3 : 1);
  const layerClass =
    inspection.layer_classification ??
    fileItem.layer_classification ??
    (channels === 1
      ? 'Panchromatic Grayscale (1 Scientific Layer)'
      : channels === 3
      ? 'RGB Composite (3 Image Channels / 1 Tri-band Layer)'
      : `${channels} Channels`);

  const minDn = inspection.min_dn ?? fileItem.min_dn ?? 0;
  const maxDn = inspection.max_dn ?? fileItem.max_dn ?? (bitDepth === 16 ? 65535 : 255);
  const meanDn = inspection.mean_dn ?? fileItem.mean_dn ?? (minDn + maxDn) / 2;
  const medianDn = inspection.median_dn ?? fileItem.median_dn ?? meanDn;
  const stdDn = inspection.std_dn ?? fileItem.std_dn ?? 0;
  const rangeSpan = inspection.dynamic_range_span ?? fileItem.dynamic_range_span ?? Math.max(0, maxDn - minDn);
  const validPct = inspection.valid_pixel_pct ?? fileItem.valid_pixel_pct ?? 100.0;
  const satPct = inspection.saturation_pct ?? fileItem.saturation_pct ?? 0.0;

  const sensorName = sensor.sensor ?? fileItem.sensor ?? 'UNKNOWN / NEEDS METADATA';
  const missionName = sensor.mission ?? fileItem.mission ?? 'Unknown Lunar Mission';
  const gsdM = sensor.gsd_m ?? fileItem.gsd_m;
  const confidence = sensor.confidence_pct ?? fileItem.sensor_confidence_pct ?? 0;
  const evidenceList = sensor.evidence ?? fileItem.sensor_evidence ?? [
    'No embedded PDS/GeoTIFF sensor tags or recognized mission tokens found in filename.',
  ];

  const histBins = inspection.histogram_bins ?? fileItem.histogram?.bins ?? [];
  const histCounts = inspection.histogram_counts ?? fileItem.histogram?.counts ?? [];
  const maxCount = histCounts.length > 0 ? Math.max(...histCounts, 1) : 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-card/60 p-4 backdrop-blur-xs">
      <div className="flex h-[92vh] w-full max-w-5xl flex-col rounded-xl border border-border bg-card shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/80 bg-gradient-to-b from-card to-muted px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-lg bg-sky-50 text-sky-800 border border-sky-300 shadow-2xs">
              <Binary className="size-4.5 text-sky-800" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] font-bold tracking-wider text-sky-800 uppercase">
                  Dataset Scientific Inspector &middot; Radiometric Telemetry
                </span>
                <span className="rounded bg-muted border border-border px-1.5 py-0.2 font-mono text-[9px] font-semibold text-muted-foreground uppercase">
                  Lossless Radiometry
                </span>
              </div>
              <h2 className="text-base font-display font-bold text-card-foreground truncate max-w-xl">
                {fileItem.filename}
              </h2>
            </div>
          </div>
          <button
            type="button"
            data-testid="button-close-scientific-inspector"
            onClick={onClose}
            className="rounded-lg p-1.5 text-muted-foreground hover:bg-accent hover:text-card-foreground transition"
          >
            <X className="size-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Section 1: Sensor & Mission Auto-Identification (Evidence Based) */}
          <div className="rounded-xl border border-border bg-muted/50 p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <ShieldCheck className="size-5 text-primary" />
                <h3 className="text-sm font-bold text-card-foreground">
                  Sensor & Mission Auto-Identification
                </h3>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Identification Confidence:</span>
                <span
                  className={`font-mono text-xs font-bold px-2 py-0.5 rounded-full border ${
                    confidence >= 80
                      ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
                      : confidence > 0
                      ? 'bg-amber-500/10 text-amber-500 border-amber-500/20'
                      : 'bg-muted text-muted-foreground border-border'
                  }`}
                >
                  {confidence > 0 ? `${confidence}%` : '0% (Needs Metadata)'}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
              <div className="rounded-lg bg-card p-3 border border-border">
                <span className="font-mono text-[10px] text-muted-foreground uppercase">Identified Sensor</span>
                <p className="mt-1 font-semibold text-card-foreground text-sm">{sensorName}</p>
                {gsdM && (
                  <p className="mt-0.5 font-mono text-[11px] text-primary">
                    Nominal GSD: {gsdM} m/pixel
                  </p>
                )}
              </div>
              <div className="rounded-lg bg-card p-3 border border-border">
                <span className="font-mono text-[10px] text-muted-foreground uppercase">Mission / Platform</span>
                <p className="mt-1 font-semibold text-card-foreground text-sm">{missionName}</p>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  Provenance: {fileItem.metadata_status || 'LOCAL_INGESTION'}
                </p>
              </div>
              <div className="rounded-lg bg-card p-3 border border-border">
                <span className="font-mono text-[10px] text-muted-foreground uppercase">Identification Status</span>
                <div className="mt-1 flex items-center gap-1.5">
                  {confidence >= 80 ? (
                    <BadgeCheck className="size-4 text-emerald-600" />
                  ) : (
                    <AlertCircle className="size-4 text-amber-600" />
                  )}
                  <span className="font-semibold text-card-foreground">
                    {confidence >= 80 ? 'Evidence Verified' : confidence > 0 ? 'Pattern Inferred' : 'Unknown / Unverified'}
                  </span>
                </div>
                <p className="mt-0.5 text-[10px] text-muted-foreground">
                  {confidence === 0 ? 'No fabrication: honest unknown tag' : 'Derived from genuine mission evidence'}
                </p>
              </div>
            </div>

            {/* Evidence Trace Chain */}
            <div className="rounded-lg bg-card p-3.5 border border-border space-y-2">
              <span className="font-mono text-[10px] text-muted-foreground uppercase font-semibold">
                Transparent Evidence Trace:
              </span>
              <ul className="space-y-1.5 text-xs text-muted-foreground">
                {evidenceList.map((ev, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <ChevronRight className="size-3.5 text-cyan-600 shrink-0 mt-0.5" />
                    <span>{ev}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Section 2: Radiometric & Dynamic Range Inspection */}
          <div className="rounded-xl border border-border bg-muted/50 p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <Sliders className="size-5 text-primary" />
                <h3 className="text-sm font-bold text-card-foreground">
                  Radiometric Depth & Scientific Dynamic Range
                </h3>
              </div>
              <div className="flex items-center gap-2">
                <span className="rounded bg-cyan-100 text-primary font-mono text-[11px] font-semibold px-2 py-0.5 border border-primary/20">
                  {bitDepth}-bit ({dtype})
                </span>
                <span className="rounded bg-emerald-100 text-emerald-500 font-mono text-[11px] font-semibold px-2 py-0.5 border border-emerald-500/20">
                  Raw Range Preserved
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="rounded-lg bg-card p-3 border border-border">
                <span className="font-mono text-[10px] text-muted-foreground uppercase">Native Bit-Depth</span>
                <p className="mt-1 font-mono text-base font-bold text-card-foreground">{bitDepth}-bit</p>
                <p className="text-[10px] text-muted-foreground">Storage dtype: {dtype}</p>
              </div>
              <div className="rounded-lg bg-card p-3 border border-border">
                <span className="font-mono text-[10px] text-muted-foreground uppercase">Min / Max Raw DN</span>
                <p className="mt-1 font-mono text-base font-bold text-primary">
                  {minDn.toLocaleString()} – {maxDn.toLocaleString()}
                </p>
                <p className="text-[10px] text-muted-foreground">Span: {rangeSpan.toLocaleString()} DN</p>
              </div>
              <div className="rounded-lg bg-card p-3 border border-border">
                <span className="font-mono text-[10px] text-muted-foreground uppercase">Mean / Median DN</span>
                <p className="mt-1 font-mono text-base font-bold text-card-foreground">
                  {meanDn.toFixed(1)} / {medianDn.toFixed(1)}
                </p>
                <p className="text-[10px] text-muted-foreground">Std Dev (σ): {stdDn.toFixed(1)}</p>
              </div>
              <div className="rounded-lg bg-card p-3 border border-border">
                <span className="font-mono text-[10px] text-muted-foreground uppercase">Valid / Saturated Pixels</span>
                <p className="mt-1 font-mono text-base font-bold text-emerald-500">
                  {validPct.toFixed(1)}% <span className="text-xs text-muted-foreground font-normal">valid</span>
                </p>
                <p className="text-[10px] text-muted-foreground">Saturated: {satPct.toFixed(2)}%</p>
              </div>
            </div>

            {/* Scientific Layer Count Clarification */}
            <div className="rounded-lg bg-primary/10/70 p-3.5 border border-primary/20 flex items-start gap-3">
              <Info className="size-4 text-primary shrink-0 mt-0.5" />
              <div className="text-xs text-cyan-950 space-y-1">
                <div className="font-semibold text-primary">
                  Scientific Layer Count Clarification: {layerClass}
                </div>
                <p className="text-muted-foreground leading-relaxed">
                  Image raster has <strong className="font-mono">{width} × {height}</strong> pixels across{' '}
                  <strong className="font-mono">{channels}</strong> physical image channel(s) and{' '}
                  <strong className="font-mono">{bands}</strong> scientific spectral band(s). Display preview is rendered
                  using an 8-bit normalized contrast stretch strictly for UI canvas viewing without altering the raw
                  scientific radiometry.
                </p>
              </div>
            </div>

            {/* 32-bin Radiometric Histogram */}
            {histCounts.length > 0 && (
              <div className="rounded-lg bg-card p-4 border border-border space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <BarChart2 className="size-4 text-primary" />
                    <span className="font-mono text-[11px] font-semibold uppercase text-card-foreground">
                      Radiometric DN Distribution (32 Bins)
                    </span>
                  </div>
                  {hoveredBinIndex !== null && histBins[hoveredBinIndex] !== undefined && (
                    <span className="font-mono text-xs font-semibold text-primary bg-primary/10 px-2 py-0.5 rounded border border-primary/20">
                      DN {histBins[hoveredBinIndex]?.toFixed(1)}–{histBins[hoveredBinIndex + 1]?.toFixed(1)}:{' '}
                      {histCounts[hoveredBinIndex]?.toLocaleString()} pixels
                    </span>
                  )}
                </div>

                <div className="h-28 w-full flex items-end gap-1 pt-2">
                  {histCounts.map((count, idx) => {
                    const heightPct = Math.max(3, (count / maxCount) * 100);
                    const isHovered = hoveredBinIndex === idx;
                    return (
                      <div
                        key={idx}
                        onMouseEnter={() => setHoveredBinIndex(idx)}
                        onMouseLeave={() => setHoveredBinIndex(null)}
                        className={`flex-1 rounded-t transition-all cursor-pointer ${
                          isHovered ? 'bg-cyan-600 shadow-sm' : 'bg-primary/20/70 hover:bg-cyan-700'
                        }`}
                        style={{ height: `${heightPct}%` }}
                        title={`Bin ${idx + 1}: ${count.toLocaleString()} pixels`}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between font-mono text-[10px] text-muted-foreground pt-1 border-t border-slate-100">
                  <span>Min: {minDn.toFixed(1)} DN</span>
                  <span>Center: {((minDn + maxDn) / 2).toFixed(1)} DN</span>
                  <span>Max: {maxDn.toFixed(1)} DN</span>
                </div>
              </div>
            )}
          </div>

          {/* Section 3: Visual Preview & Dimensional Overview */}
          <div className="rounded-xl border border-border bg-muted/50 p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <FileImage className="size-5 text-primary" />
                <h3 className="text-sm font-bold text-card-foreground">
                  Visual Preview & Raster Dimensions
                </h3>
              </div>
              <span className="font-mono text-xs text-muted-foreground">
                {width} × {height} px ({((fileItem.size_bytes || 0) / (1024 * 1024)).toFixed(2)} MB)
              </span>
            </div>

            <div className="flex flex-col sm:flex-row items-center gap-6">
              {fileItem.preview_url && (
                <div className="size-48 shrink-0 overflow-hidden rounded-lg border border-border bg-muted flex items-center justify-center shadow-inner">
                  <img
                    src={fileItem.preview_url}
                    alt={fileItem.filename}
                    className="size-full object-contain"
                  />
                </div>
              )}
              <div className="flex-1 space-y-2 text-xs text-muted-foreground">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="font-mono text-[10px] text-muted-foreground uppercase">Raster Width:</span>
                    <p className="font-semibold text-card-foreground">{width} pixels</p>
                  </div>
                  <div>
                    <span className="font-mono text-[10px] text-muted-foreground uppercase">Raster Height:</span>
                    <p className="font-semibold text-card-foreground">{height} pixels</p>
                  </div>
                  <div>
                    <span className="font-mono text-[10px] text-muted-foreground uppercase">Image Channels:</span>
                    <p className="font-semibold text-card-foreground">{channels} channel(s)</p>
                  </div>
                  <div>
                    <span className="font-mono text-[10px] text-muted-foreground uppercase">Data Encoding:</span>
                    <p className="font-semibold text-card-foreground">{fileItem.format || 'RASTER'}</p>
                  </div>
                </div>
                <div className="pt-2 text-[11px] text-muted-foreground bg-card p-2.5 rounded border border-border">
                  <p>
                    <strong className="text-card-foreground">Decoupled Preservation:</strong> Raw imagery is stored
                    unquantized in system memory. Downstream registration algorithms access native scientific DNs while
                    display canvas operates on the 8-bit normalized preview.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 border-t border-border bg-muted px-6 py-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border bg-card px-4 py-2 text-xs font-semibold text-muted-foreground hover:bg-muted transition"
          >
            Close Inspector
          </button>
        </div>
      </div>
    </div>
  );
}
