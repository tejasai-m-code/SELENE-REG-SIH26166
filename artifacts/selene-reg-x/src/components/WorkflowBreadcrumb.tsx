import React from 'react';
import { Database, Sliders, Sparkles, GitCompare, ShieldCheck, Target, Layers, BarChart, Grid3X3, Download } from 'lucide-react';

export type WorkflowStageId =
  | 'dataset'
  | 'preprocess'
  | 'detection'
  | 'matching'
  | 'geometry'
  | 'subpixel'
  | 'registration'
  | 'validation'
  | 'mosaic'
  | 'export';

interface WorkflowBreadcrumbProps {
  currentStage: WorkflowStageId;
  onSelectStage?: (stage: WorkflowStageId) => void;
}

const STAGES: Array<{ id: WorkflowStageId; num: string; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { id: 'dataset', num: '01', label: 'Dataset', icon: Database },
  { id: 'preprocess', num: '02', label: 'Preprocess', icon: Sliders },
  { id: 'detection', num: '03', label: 'Features', icon: Sparkles },
  { id: 'matching', num: '04', label: 'Matching', icon: GitCompare },
  { id: 'geometry', num: '05', label: 'Geometry', icon: ShieldCheck },
  { id: 'subpixel', num: '06', label: 'Subpixel', icon: Target },
  { id: 'registration', num: '07', label: 'Registration', icon: Layers },
  { id: 'validation', num: '08', label: 'Validation', icon: BarChart },
  { id: 'mosaic', num: '09', label: 'Mosaic', icon: Grid3X3 },
  { id: 'export', num: '10', label: 'Export', icon: Download },
];

export function WorkflowBreadcrumb({ currentStage, onSelectStage }: WorkflowBreadcrumbProps) {
  return (
    <div className="w-full border-b border-border/80 bg-background shadow-2xs overflow-hidden">
      <div className="mx-auto flex max-w-[1920px] items-center justify-between overflow-x-auto px-4 py-2 scrollbar-none sm:px-6">
        <div className="flex items-center gap-1.5 sm:gap-2">
          {STAGES.map((s, idx) => {
            const Icon = s.icon;
            const isCurrent = currentStage === s.id;
            return (
              <React.Fragment key={s.id}>
                {idx > 0 && <span className="text-[10px] text-muted-foreground select-none">›</span>}
                <button
                  type="button"
                  onClick={() => onSelectStage?.(s.id)}
                  className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-mono transition-all duration-150 cursor-pointer ${
                    isCurrent
                      ? 'border border-sky-400/90 bg-card font-bold text-sky-950 shadow-xs stepper-glow-active'
                      : 'text-muted-foreground hover:bg-card/80 hover:text-card-foreground border border-transparent'
                  }`}
                >
                  <span className={`font-mono text-[10px] font-bold ${isCurrent ? 'text-sky-700' : 'text-muted-foreground'}`}>{s.num}</span>
                  <Icon className={`size-3.5 ${isCurrent ? 'text-sky-700' : 'text-muted-foreground'}`} />
                  <span className="hidden sm:inline tracking-tight">{s.label}</span>
                  {isCurrent && (
                    <span className="ml-1 text-[9px] font-mono px-1 py-0.2 rounded bg-sky-600 text-white font-semibold uppercase tracking-wider shadow-2xs">
                      Active
                    </span>
                  )}
                </button>
              </React.Fragment>
            );
          })}
        </div>
        <div className="hidden xl:flex items-center gap-2 pl-4 text-[10px] font-mono text-muted-foreground border-l border-border">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="tracking-wider uppercase font-semibold text-muted-foreground">Deterministic Pipeline Active</span>
        </div>
      </div>
    </div>
  );
}
