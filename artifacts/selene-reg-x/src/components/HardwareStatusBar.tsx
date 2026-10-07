import { useEffect, useState } from 'react';
import { Cpu, Zap, Activity, Info, CheckCircle2, XCircle, RefreshCw, X, ShieldCheck } from 'lucide-react';

interface GpuSelfTest {
  cuda_device_detected: string;
  tensor_allocation: string;
  gpu_computation: string;
  synchronization: string;
  device_name: string;
  compute_capability?: string;
  vram_total_mb?: number;
  vram_allocated_mb?: number;
  vram_reserved_mb?: number;
  vram_free_mb?: number;
  compute_time_ms?: number;
  overall_status: string;
  execution_mode: string;
  note?: string;
}

interface HybridSchedulerTelemetry {
  total_tasks_scheduled?: number;
  gpu_tasks_executed?: number;
  cpu_tasks_executed?: number;
  fallbacks_count?: number;
  gpu_execution_percentage?: number;
  cpu_execution_percentage?: number;
  active_mode?: string;
  cuda_active?: boolean;
  gpu_device_name?: string;
  vram_free_mb?: number;
  vram_safety_margin_mb?: number;
}

interface HardwareStatus {
  device: string;
  acceleration_status: string;
  execution_mode?: string;
  status_display: string;
  hardware_gpu_name: string;
  hardware_gpu_detected: boolean;
  cuda_driver_version: string;
  vram_total_mb: number;
  vram_free_mb: number;
  vram_safety_reserve_mb?: number;
  opencv_cuda_devices: number;
  pytorch_cuda_available: boolean;
  pytorch_version?: string;
  system_ram_total_mb: number;
  system_ram_available_mb: number;
  cpu_logical_cores: number;
  bounded_worker_concurrency: number;
  gpu_self_test?: GpuSelfTest;
  hybrid_scheduler?: HybridSchedulerTelemetry;
  note: string;
}

export interface RuntimeTelemetry {
  requested_mode?: 'cpu' | 'gpu' | 'hybrid' | string;
  actual_mode?: 'cpu' | 'gpu' | 'hybrid' | string;
  running_status?: string;
  fallback_reason?: string;
  gpu_percentage?: number;
  cpu_percentage?: number;
  vram_used_mb?: number;
  vram_total_mb?: number;
  gpu_time_s?: number;
  cpu_time_s?: number;
  hybrid_time_s?: number;
  total_time_s?: number;
  tasks_breakdown?: Record<string, 'CPU' | 'GPU' | 'HYBRID'>;
}

export interface HardwareStatusBarProps {
  requestedMode?: 'cpu' | 'gpu' | 'hybrid';
  onSelectMode?: (mode: 'cpu' | 'gpu' | 'hybrid') => void;
  runtimeTelemetry?: RuntimeTelemetry;
  isProcessing?: boolean;
}

export function HardwareStatusBar({
  requestedMode: propRequestedMode,
  onSelectMode,
  runtimeTelemetry,
  isProcessing = false,
}: HardwareStatusBarProps = {}) {
  const [internalMode, setInternalMode] = useState<'cpu' | 'gpu' | 'hybrid'>(() => {
    try {
      return (localStorage.getItem('selene-execution-mode') as 'cpu' | 'gpu' | 'hybrid') || 'hybrid';
    } catch {
      return 'hybrid';
    }
  });

  const requestedMode = propRequestedMode || internalMode;

  const handleModeChange = (mode: 'cpu' | 'gpu' | 'hybrid') => {
    setInternalMode(mode);
    try {
      localStorage.setItem('selene-execution-mode', mode);
    } catch {}
    if (onSelectMode) onSelectMode(mode);
  };

  const [status, setStatus] = useState<HardwareStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [selfTesting, setSelfTesting] = useState(false);
  const [selfTestResult, setSelfTestResult] = useState<GpuSelfTest | null>(null);

  const fetchStatus = () => {
    fetch('/api/hardware/status')
      .then((res) => (res.ok ? res.json() : null))
      .then((data: HardwareStatus | null) => {
        if (data) {
          setStatus(data);
          if (data.gpu_self_test) {
            setSelfTestResult(data.gpu_self_test);
          }
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const runLiveSelfTest = async () => {
    setSelfTesting(true);
    try {
      const res = await fetch('/api/hardware/self-test', { method: 'POST' });
      if (res.ok) {
        const testData = (await res.json()) as GpuSelfTest;
        setSelfTestResult(testData);
      }
    } catch {
      // ignore
    } finally {
      setSelfTesting(false);
      fetchStatus();
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-border/50 bg-muted/80 px-2.5 py-1 text-[11px] text-muted-foreground">
        <Activity className="size-3 animate-spin text-primary" />
        <span className="font-mono text-[10px]">Detecting hardware…</span>
      </div>
    );
  }

  if (!status) return null;

  const isGpuHost = status.hardware_gpu_detected;
  const isCudaActive =
    status.pytorch_cuda_available ||
    status.acceleration_status === 'GPU ACCELERATED' ||
    status.acceleration_status === 'GPU AVAILABLE';

  const testPassed = (selfTestResult?.overall_status || status.gpu_self_test?.overall_status) === 'PASS';

  // Determine actual execution indicator
  const actualExecutionMode = runtimeTelemetry?.actual_mode?.toUpperCase() || (isProcessing ? (requestedMode === 'gpu' && !isCudaActive ? 'CPU FALLBACK' : requestedMode.toUpperCase()) : undefined);
  const fallbackReason = runtimeTelemetry?.fallback_reason || (!isCudaActive && requestedMode !== 'cpu' ? 'CUDA runtime not available on host' : undefined);

  return (
    <div className="flex items-center gap-2 sm:gap-3">
      {/* Explicit Execution Mode Selector Buttons: [ CPU ] [ GPU ] [ HYBRID ] */}
      <div
        className="flex items-center rounded-md border border-border/50 bg-muted/60 p-0.5 shadow-2xs"
        role="group"
        aria-label="Execution Mode"
      >
        <button
          type="button"
          data-testid="mode-select-cpu"
          onClick={() => handleModeChange('cpu')}
          title="Run workloads purely on CPU multi-core"
          className={`px-2.5 py-1 text-[11px] font-mono font-medium rounded transition ${
            requestedMode === 'cpu'
              ? 'bg-card text-foreground font-bold border border-border shadow-xs'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          CPU
        </button>
        <button
          type="button"
          data-testid="mode-select-gpu"
          onClick={() => handleModeChange('gpu')}
          title={isCudaActive ? 'Run supported workloads on GPU via CUDA' : 'GPU requested (will fallback if CUDA is unavailable)'}
          className={`px-2.5 py-1 text-[11px] font-mono font-medium rounded transition flex items-center gap-1.5 ${
            requestedMode === 'gpu'
              ? 'bg-primary/10 text-primary font-bold border border-primary/40 shadow-xs'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          <span className={`size-1.5 rounded-full ${requestedMode === 'gpu' ? 'bg-primary animate-pulse shadow-[0_0_8px_rgba(56,189,248,0.6)]' : 'bg-muted-foreground'}`} />
          GPU
        </button>
        <button
          type="button"
          data-testid="mode-select-hybrid"
          onClick={() => handleModeChange('hybrid')}
          title="Intelligent scheduling: CPU for IO/RANSAC, GPU for matching/warping"
          className={`px-2.5 py-1 text-[11px] font-mono font-medium rounded transition flex items-center gap-1.5 ${
            requestedMode === 'hybrid'
              ? 'bg-primary/10 text-primary font-bold border border-primary/40 shadow-xs'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          <span className={`size-1.5 rounded-full ${requestedMode === 'hybrid' ? 'bg-primary animate-pulse shadow-[0_0_8px_rgba(56,189,248,0.6)]' : 'bg-muted-foreground'}`} />
          HYBRID
        </button>
      </div>

      {/* Main Hardware Status Pill / Diagnostics Trigger */}
      <button
        type="button"
        data-testid="button-hardware-diagnostics"
        onClick={() => setModalOpen(true)}
        title="Click to view detailed GPU/CPU Hybrid Execution & Self-Test diagnostics"
        className="flex items-center gap-2 rounded-md border border-border/50 bg-muted/60 hover:border-border transition-colors px-2.5 sm:px-3 py-1 text-[11px] text-foreground shadow-2xs cursor-pointer text-left font-mono"
      >
        <div className="flex items-center gap-1.5">
          {isCudaActive ? (
            <span className="size-2 rounded-full bg-emerald-500 animate-pulse" />
          ) : (
            <span className="size-2 rounded-full bg-sky-600" />
          )}
          <span className="font-bold text-foreground hidden md:inline">
            {isGpuHost ? status.hardware_gpu_name.replace('NVIDIA ', '').replace('GeForce ', '') : 'CPU Host'}
          </span>
        </div>

        <span className="text-muted-foreground">|</span>

        {/* Dynamic Execution Badge */}
        {isProcessing ? (
          <span className="inline-flex items-center gap-1 font-mono text-[10px] uppercase px-1.5 py-0.5 rounded font-bold bg-amber-950/40 text-amber-300 border border-amber-900/50 animate-pulse">
            <Activity className="size-2.5 animate-spin text-amber-500" />
            RUNNING: {requestedMode.toUpperCase()}
          </span>
        ) : actualExecutionMode ? (
          <span
            className={`font-mono text-[10px] uppercase px-1.5 py-0.5 rounded font-bold ${
              actualExecutionMode === 'GPU'
                ? 'bg-emerald-950/40 text-emerald-300 border border-emerald-900/50'
                : actualExecutionMode === 'HYBRID'
                ? 'bg-sky-950/40 text-sky-300 border border-sky-900/50'
                : 'bg-muted/80 text-muted-foreground border border-border'
            }`}
          >
            ACTUAL: {actualExecutionMode}
          </span>
        ) : (
          <span
            className={`font-mono text-[10px] uppercase px-1.5 py-0.5 rounded font-bold ${
              isCudaActive
                ? 'bg-emerald-950/40 text-emerald-300 border border-emerald-900/50'
                : isGpuHost
                ? 'bg-sky-950/40 text-sky-300 border border-sky-900/50'
                : 'bg-muted/80 text-muted-foreground border border-border'
            }`}
          >
            {isCudaActive ? 'CUDA READY' : 'CPU MODE'}
          </span>
        )}

        {isCudaActive && status.vram_total_mb > 0 && (
          <span className="hidden lg:inline text-muted-foreground text-[10px] font-mono">
            VRAM <span className="font-bold text-sky-700">{(status.vram_free_mb / 1024).toFixed(1)}</span>/{(status.vram_total_mb / 1024).toFixed(1)} GB
          </span>
        )}

        <span
          className={`hidden xl:inline-flex items-center gap-1 font-mono text-[10px] px-1.5 py-0.5 rounded border ${
            testPassed
              ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20 font-bold'
              : 'bg-destructive/10 text-destructive border-destructive/20 font-bold'
          }`}
        >
          {testPassed ? <CheckCircle2 className="size-2.5 text-emerald-600" /> : <XCircle className="size-2.5 text-rose-600" />}
          SELF-TEST: {testPassed ? 'PASS' : 'FAIL'}
        </span>

        <Info className="size-3 text-muted-foreground opacity-60 ml-0.5" />
      </button>

      {/* Hardware Diagnostics Modal */}
      {modalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-xs p-4 animate-in fade-in duration-150">
          <div className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-xl border border-primary/20/60 bg-background p-6 text-foreground shadow-2xl">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-border pb-3 mb-4">
              <div className="flex items-center gap-2">
                <ShieldCheck className="size-5 text-primary" />
                <div>
                  <h3 className="font-semibold text-base text-foreground">
                    GPU + CPU Hybrid Execution & Hardware Self-Test
                  </h3>
                  <p className="text-[11px] text-muted-foreground font-mono">
                    Truthful hardware inspection & 4 GB VRAM memory safety envelope (SIH 26166 §22-§25)
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setModalOpen(false)}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                <X className="size-4" />
              </button>
            </div>

            {/* Hardware Environment Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
              <div className="rounded-lg border border-border bg-card/60 p-2.5">
                <span className="text-[10px] uppercase font-mono text-muted-foreground">GPU Device</span>
                <p className="text-xs font-semibold text-emerald-500 truncate" title={status.hardware_gpu_name}>
                  {status.hardware_gpu_name || 'None'}
                </p>
                <span className="text-[10px] text-muted-foreground font-mono">Driver {status.cuda_driver_version}</span>
              </div>

              <div className="rounded-lg border border-border bg-card/60 p-2.5">
                <span className="text-[10px] uppercase font-mono text-muted-foreground">CUDA Runtime</span>
                <p className="text-xs font-semibold text-cyan-300">
                  {isCudaActive ? 'Available (PyTorch)' : 'Unavailable'}
                </p>
                <span className="text-[10px] text-muted-foreground font-mono">{status.pytorch_version || 'N/A'}</span>
              </div>

              <div className="rounded-lg border border-border bg-card/60 p-2.5">
                <span className="text-[10px] uppercase font-mono text-muted-foreground">Dedicated VRAM</span>
                <p className="text-xs font-semibold text-foreground">
                  {status.vram_total_mb ? `${(status.vram_total_mb / 1024).toFixed(1)} GB Total` : '0 GB'}
                </p>
                <span className="text-[10px] text-emerald-500 font-mono">
                  {(status.vram_free_mb / 1024).toFixed(1)} GB Free
                </span>
              </div>

              <div className="rounded-lg border border-border bg-card/60 p-2.5">
                <span className="text-[10px] uppercase font-mono text-muted-foreground">Host RAM & Cores</span>
                <p className="text-xs font-semibold text-foreground">
                  {(status.system_ram_total_mb / 1024).toFixed(0)} GB RAM
                </p>
                <span className="text-[10px] text-muted-foreground font-mono">
                  {status.cpu_logical_cores} Cores ({status.bounded_worker_concurrency} Workers)
                </span>
              </div>
            </div>

            {/* Authoritative Real GPU Self-Test */}
            <div className="rounded-lg border border-primary/20/40 bg-card/80 p-4 mb-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Zap className="size-4 text-emerald-500" />
                  <h4 className="text-xs font-bold uppercase tracking-wider text-foreground">
                    Live GPU Self-Test Results (§24)
                  </h4>
                </div>
                <button
                  type="button"
                  onClick={runLiveSelfTest}
                  disabled={selfTesting}
                  className="flex items-center gap-1.5 rounded-md bg-cyan-950 hover:bg-primary/20 border border-cyan-700/60 px-2.5 py-1 text-[11px] font-medium text-cyan-200 disabled:opacity-50 cursor-pointer"
                >
                  <RefreshCw className={`size-3 ${selfTesting ? 'animate-spin text-primary' : ''}`} />
                  {selfTesting ? 'Running Self-Test…' : 'Run Live Self-Test'}
                </button>
              </div>

              {selfTestResult ? (
                <div className="space-y-2">
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
                    <div className="rounded border border-border bg-background p-2">
                      <span className="text-[10px] text-muted-foreground uppercase font-mono block">CUDA Device</span>
                      <span
                        className={`text-xs font-bold font-mono ${
                          selfTestResult.cuda_device_detected === 'PASS' ? 'text-emerald-500' : 'text-destructive'
                        }`}
                      >
                        {selfTestResult.cuda_device_detected}
                      </span>
                    </div>

                    <div className="rounded border border-border bg-background p-2">
                      <span className="text-[10px] text-muted-foreground uppercase font-mono block">Tensor Alloc</span>
                      <span
                        className={`text-xs font-bold font-mono ${
                          selfTestResult.tensor_allocation === 'PASS' ? 'text-emerald-500' : 'text-destructive'
                        }`}
                      >
                        {selfTestResult.tensor_allocation}
                      </span>
                    </div>

                    <div className="rounded border border-border bg-background p-2">
                      <span className="text-[10px] text-muted-foreground uppercase font-mono block">GPU Compute (GEMM)</span>
                      <span
                        className={`text-xs font-bold font-mono ${
                          selfTestResult.gpu_computation === 'PASS' ? 'text-emerald-500' : 'text-destructive'
                        }`}
                      >
                        {selfTestResult.gpu_computation}
                      </span>
                    </div>

                    <div className="rounded border border-border bg-background p-2">
                      <span className="text-[10px] text-muted-foreground uppercase font-mono block">Sync & Stream</span>
                      <span
                        className={`text-xs font-bold font-mono ${
                          selfTestResult.synchronization === 'PASS' ? 'text-emerald-500' : 'text-destructive'
                        }`}
                      >
                        {selfTestResult.synchronization}
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center justify-between text-[11px] text-muted-foreground font-mono bg-background/80 px-3 py-1.5 rounded border border-border/80">
                    <span>
                      Device: <strong className="text-foreground">{selfTestResult.device_name}</strong>
                    </span>
                    <span>
                      Compute Latency: <strong className="text-cyan-300">{selfTestResult.compute_time_ms} ms</strong>
                    </span>
                    <span>
                      Execution Mode: <strong className="text-emerald-300">{selfTestResult.execution_mode}</strong>
                    </span>
                  </div>
                  {selfTestResult.note && (
                    <p className="text-[10px] text-muted-foreground italic font-mono px-1">
                      {selfTestResult.note}
                    </p>
                  )}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground font-mono">No self-test result recorded yet.</p>
              )}
            </div>

            {/* Hybrid Scheduler & 4 GB VRAM Guard Policy */}
            <div className="rounded-lg border border-border bg-card/60 p-4 mb-4">
              <h4 className="text-xs font-bold uppercase tracking-wider text-foreground mb-2.5">
                Intelligent Hybrid Scheduling & 4 GB VRAM Safety (§23 & §25)
              </h4>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div className="rounded border border-border/80 bg-background/60 p-2.5 space-y-1">
                  <span className="font-semibold text-cyan-300 flex items-center gap-1.5">
                    <Cpu className="size-3 text-primary" /> CPU Workloads
                  </span>
                  <ul className="text-[11px] text-muted-foreground space-y-0.5 list-disc list-inside">
                    <li>PDS/PVL/XML metadata extraction</li>
                    <li>Image disk I/O & raster decoding</li>
                    <li>RANSAC & USAC-MAGSAC geometric estimation</li>
                    <li>Spatial distribution & adaptive grid filtering</li>
                    <li>Multi-image registration graph & MST</li>
                  </ul>
                </div>

                <div className="rounded border border-border/80 bg-background/60 p-2.5 space-y-1">
                  <span className="font-semibold text-emerald-300 flex items-center gap-1.5">
                    <Zap className="size-3 text-emerald-500" /> GPU CUDA Workloads
                  </span>
                  <ul className="text-[11px] text-muted-foreground space-y-0.5 list-disc list-inside">
                    <li>2D FFT Phase correlation & cross-power</li>
                    <li>Tensor distance matrix & Lowe ratio matching</li>
                    <li>Perspective warping via GPU grid sampling</li>
                    <li>2D Sobel gradients & multi-scale Retinex</li>
                    <li>Chunked batch execution for large sets</li>
                  </ul>
                </div>
              </div>

              <div className="mt-3 rounded border border-amber-950/60 bg-amber-950/20 px-3 py-2 text-[11px] text-amber-300/90 font-mono">
                <strong>4 GB VRAM Memory Safeguard:</strong> Minimum 512 MB safety reserve enforced. If free VRAM drops below threshold or tensor allocation exceeds envelope, operations automatically and transparently fall back to multi-core CPU without crashing.
              </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between pt-2 border-t border-border text-[11px] text-muted-foreground font-mono">
              <span>SELENE-REG-X Phase 7 System Telemetry</span>
              <button
                type="button"
                onClick={() => setModalOpen(false)}
                className="rounded-md bg-muted hover:bg-accent px-3 py-1 text-foreground cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
