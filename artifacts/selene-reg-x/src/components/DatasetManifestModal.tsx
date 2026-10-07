import { useState, useRef, type ChangeEvent, type DragEvent } from 'react';
import {
  FolderOpen,
  RefreshCw,
  Search,
  Check,
  AlertTriangle,
  FileImage,
  ArrowRight,
  Database,
  Layers3,
  X,
  SlidersHorizontal,
  Upload,
  Binary,
  ShieldCheck,
  ChevronDown,
} from 'lucide-react';
import {
  ScientificDataInspectorModal,
  type ScientificFileItem,
} from './ScientificDataInspectorModal';

export interface ManifestFileItem extends ScientificFileItem {
  relative_path: string;
  absolute_path: string;
  status: 'READY' | 'UNSUPPORTED' | 'METADATA_ONLY' | 'CORRUPT';
  reason?: string | null;
  metadata_summary?: string;
}

export interface DatasetManifest {
  dataset_name: string;
  manifest_id: string;
  root_path: string;
  files_discovered: number;
  images_count: number;
  metadata_count: number;
  unsupported_count: number;
  total_size_bytes: number;
  files: ManifestFileItem[];
}

interface DatasetManifestModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectPair?: (sourcePath: string, referencePath: string, sourceName: string, refName: string) => void;
  onSelectImage?: (path: string, name: string, target: 'source' | 'reference') => void;
  onSelectBatch?: (paths: string[], names: string[]) => void;
}

export function DatasetManifestModal({
  isOpen,
  onClose,
  onSelectPair,
  onSelectImage,
  onSelectBatch,
}: DatasetManifestModalProps) {
  const [folderPath, setFolderPath] = useState('demo_data');
  const [isScanning, setIsScanning] = useState(false);
  const [manifest, setManifest] = useState<DatasetManifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterFormat, setFilterFormat] = useState('ALL');
  const [filterStatus, setFilterStatus] = useState('ALL');
  const [selectedMoving, setSelectedMoving] = useState<ManifestFileItem | null>(null);
  const [selectedFixed, setSelectedFixed] = useState<ManifestFileItem | null>(null);
  const [selectedBatch, setSelectedBatch] = useState<Set<string>>(new Set());
  const [inspectingFile, setInspectingFile] = useState<ScientificFileItem | null>(null);
  const [showHostInput, setShowHostInput] = useState(false);

  const folderInputRef = useRef<HTMLInputElement>(null);
  const filesInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleScanHostPath = async () => {
    setIsScanning(true);
    setError(null);
    try {
      const res = await fetch('/api/dataset/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: folderPath.trim() }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Dataset folder scan failed.');
      }
      setManifest(data);
      const readyFiles = (data.files as ManifestFileItem[]).filter((f) => f.status === 'READY');
      if (readyFiles.length >= 2) {
        setSelectedMoving(readyFiles[0]);
        setSelectedFixed(readyFiles[1]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scan request failed.');
    } finally {
      setIsScanning(false);
    }
  };

  const handleNativeFolderUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;

    setIsScanning(true);
    setError(null);

    try {
      const formData = new FormData();
      let folderName = 'Uploaded Folder';
      for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        const relativePath = file.webkitRelativePath || file.name;
        if (i === 0 && file.webkitRelativePath) {
          const parts = file.webkitRelativePath.split('/');
          if (parts.length > 1) folderName = parts[0];
        }
        formData.append('files', file, relativePath);
      }
      formData.append('dataset_name', folderName);

      const res = await fetch('/api/dataset/upload-folder', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Folder upload and scanning failed.');
      }
      setManifest(data);
      const readyFiles = (data.files as ManifestFileItem[]).filter((f) => f.status === 'READY');
      if (readyFiles.length >= 2) {
        setSelectedMoving(readyFiles[0]);
        setSelectedFixed(readyFiles[1]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Folder upload failed.');
    } finally {
      setIsScanning(false);
      if (folderInputRef.current) folderInputRef.current.value = '';
    }
  };

  const handleNativeFilesUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;

    setIsScanning(true);
    setError(null);

    try {
      const formData = new FormData();
      for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        formData.append('files', file, file.name);
      }
      formData.append('dataset_name', `Batch Selection (${fileList.length} files)`);

      const res = await fetch('/api/dataset/upload-folder', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Files upload and scanning failed.');
      }
      setManifest(data);
      const readyFiles = (data.files as ManifestFileItem[]).filter((f) => f.status === 'READY');
      if (readyFiles.length >= 2) {
        setSelectedMoving(readyFiles[0]);
        setSelectedFixed(readyFiles[1]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Files upload failed.');
    } finally {
      setIsScanning(false);
      if (filesInputRef.current) filesInputRef.current.value = '';
    }
  };

  const filteredFiles = (manifest?.files || []).filter((f) => {
    if (searchQuery && !f.filename.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    if (filterFormat !== 'ALL' && (f.format || '').toUpperCase() !== filterFormat) return false;
    if (filterStatus !== 'ALL' && f.status !== filterStatus) return false;
    return true;
  });

  const formats = Array.from(new Set((manifest?.files || []).map((f) => (f.format || '').toUpperCase()))).filter(Boolean);

  const toggleBatchSelect = (path: string) => {
    setSelectedBatch((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const selectAllReady = () => {
    if (!manifest) return;
    const ready = manifest.files.filter((f) => f.status === 'READY').map((f) => f.absolute_path);
    setSelectedBatch(new Set(ready));
  };

  const handleApplyPair = () => {
    if (selectedMoving && selectedFixed) {
      onSelectPair?.(
        selectedMoving.absolute_path,
        selectedFixed.absolute_path,
        selectedMoving.filename,
        selectedFixed.filename
      );
      onClose();
    }
  };

  const handleApplyBatch = () => {
    if (!manifest || selectedBatch.size < 2) return;
    const selectedItems = manifest.files.filter((f) => selectedBatch.has(f.absolute_path));
    const paths = selectedItems.map((f) => f.absolute_path);
    const names = selectedItems.map((f) => f.filename);
    onSelectBatch?.(paths, names);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-card/60 p-4 backdrop-blur-xs">
      <div className="flex h-[92vh] w-full max-w-5xl flex-col rounded-xl border border-border bg-card shadow-2xl overflow-hidden">
        {/* Hidden native directory & files file-pickers */}
        <input
          type="file"
          ref={folderInputRef}
          // @ts-expect-error - webkitdirectory is standard in Chromium browsers
          webkitdirectory="true"
          directory="true"
          multiple
          onChange={handleNativeFolderUpload}
          className="hidden"
          data-testid="input-native-folder"
        />
        <input
          type="file"
          ref={filesInputRef}
          multiple
          accept="image/*,.tif,.tiff,.png,.jpg,.jpeg,.bmp,.img,.fit,.fits,.cub"
          onChange={handleNativeFilesUpload}
          className="hidden"
          data-testid="input-native-files"
        />

        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/80 bg-gradient-to-b from-card to-muted px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-lg bg-sky-50 text-sky-800 border border-sky-300 shadow-2xs">
              <Database className="size-4.5 text-sky-800" />
            </div>
            <div>
              <p className="font-mono text-[10px] font-bold tracking-wider text-sky-800 uppercase">
                Planetary Ingestion Engine &middot; Manifest Explorer
              </p>
              <h2 className="text-base font-display font-bold text-card-foreground">
                Lunar Dataset Directory &amp; Multi-Image Manifest
              </h2>
            </div>
          </div>
          <button
            type="button"
            data-testid="button-close-manifest-modal"
            onClick={onClose}
            className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-card-foreground transition"
          >
            <X className="size-5" />
          </button>
        </div>

        {/* Ingestion Action Bar */}
        <div className="border-b border-border bg-muted/70 p-4 space-y-3 font-mono">
          <div className="flex flex-wrap items-center gap-3">
            {/* Primary Native Directory Browser Button */}
            <button
              type="button"
              data-testid="button-browse-folder"
              disabled={isScanning}
              onClick={() => folderInputRef.current?.click()}
              className="focus-ring flex h-10 items-center justify-center gap-2 rounded-md bg-sky-700 px-5 text-xs font-mono font-bold text-white shadow-xs hover:bg-primary/20 disabled:opacity-50 transition border border-sky-800 cursor-pointer"
            >
              {isScanning ? (
                <RefreshCw className="size-4 animate-spin text-white" />
              ) : (
                <FolderOpen className="size-4 text-sky-200" />
              )}
              <span>{isScanning ? 'Ingesting Dataset…' : 'Select Image Folder (Native Picker)'}</span>
            </button>

            {/* Secondary Multi-File Selection Button */}
            <button
              type="button"
              data-testid="button-browse-files"
              disabled={isScanning}
              onClick={() => filesInputRef.current?.click()}
              className="focus-ring flex h-10 items-center justify-center gap-2 rounded-md border border-border bg-card px-4 text-xs font-semibold text-muted-foreground hover:bg-muted disabled:opacity-50 transition"
            >
              <Upload className="size-4 text-muted-foreground" />
              <span>Select Multiple Images</span>
            </button>

            {/* Toggle Host Folder Input */}
            <button
              type="button"
              onClick={() => setShowHostInput(!showHostInput)}
              className="ml-auto text-[11px] font-medium text-muted-foreground hover:text-primary flex items-center gap-1"
            >
              <span>Host directory path</span>
              <ChevronDown className={`size-3 transition ${showHostInput ? 'rotate-180' : ''}`} />
            </button>
          </div>

          {showHostInput && (
            <div className="flex items-center gap-2 pt-2 border-t border-border">
              <input
                type="text"
                data-testid="input-folder-path"
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                placeholder="Enter local folder path (e.g. demo_data, D:/LunarData)"
                className="focus-ring h-9 flex-1 rounded-md border border-border bg-card px-3 text-xs text-card-foreground outline-none"
              />
              <button
                type="button"
                data-testid="button-scan-dataset"
                onClick={handleScanHostPath}
                disabled={isScanning || !folderPath.trim()}
                className="focus-ring flex h-9 items-center justify-center gap-1.5 rounded-md bg-muted px-4 text-xs font-semibold text-white hover:bg-card disabled:opacity-50"
              >
                <Search className="size-3.5" />
                <span>Scan Path</span>
              </button>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 rounded-md border border-red-200 bg-red-50 p-2.5 text-xs text-red-800">
              <AlertTriangle className="size-4 shrink-0 text-red-600" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Manifest Overview Bar */}
        {manifest && (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 border-b border-border bg-primary/10/40 p-4 text-xs">
            <div>
              <p className="font-mono text-[10px] text-muted-foreground uppercase">Dataset Name</p>
              <p className="font-semibold text-card-foreground truncate">{manifest.dataset_name}</p>
            </div>
            <div>
              <p className="font-mono text-[10px] text-muted-foreground uppercase">Files Discovered</p>
              <p className="font-semibold text-primary">{manifest.files_discovered}</p>
            </div>
            <div>
              <p className="font-mono text-[10px] text-muted-foreground uppercase">Images (Ready)</p>
              <p className="font-semibold text-emerald-500">{manifest.images_count}</p>
            </div>
            <div>
              <p className="font-mono text-[10px] text-muted-foreground uppercase">Metadata Sidecars</p>
              <p className="font-semibold text-muted-foreground">{manifest.metadata_count}</p>
            </div>
            <div>
              <p className="font-mono text-[10px] text-muted-foreground uppercase">Unsupported</p>
              <p className="font-semibold text-rose-700">{manifest.unsupported_count}</p>
            </div>
          </div>
        )}

        {/* Search & Filters */}
        {manifest && (
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-2.5 bg-card text-xs">
            <div className="flex flex-wrap items-center gap-2">
              <div className="relative">
                <Search className="pointer-events-none absolute left-2.5 top-2 size-3.5 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Filter by name…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-7 w-48 rounded border border-border pl-8 pr-2 text-xs"
                />
              </div>
              <select
                value={filterFormat}
                onChange={(e) => setFilterFormat(e.target.value)}
                className="h-7 rounded border border-border px-2 text-xs"
              >
                <option value="ALL">All Formats</option>
                {formats.map((fmt) => (
                  <option key={fmt} value={fmt}>
                    {fmt}
                  </option>
                ))}
              </select>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="h-7 rounded border border-border px-2 text-xs"
              >
                <option value="ALL">All Statuses</option>
                <option value="READY">Ready</option>
                <option value="UNSUPPORTED">Unsupported</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={selectAllReady}
                className="text-[11px] font-semibold text-primary hover:underline"
              >
                Select All Ready ({manifest.images_count})
              </button>
            </div>
          </div>
        )}

        {/* Manifest Table */}
        <div className="flex-1 overflow-y-auto p-4">
          {!manifest ? (
            <div className="flex h-full flex-col items-center justify-center text-center text-muted-foreground p-8 space-y-3">
              <div className="size-16 rounded-full bg-muted border border-border flex items-center justify-center">
                <FolderOpen className="size-8 text-muted-foreground" />
              </div>
              <div>
                <p className="text-sm font-semibold text-muted-foreground">No Dataset Directory Loaded</p>
                <p className="text-xs text-muted-foreground max-w-md mt-1">
                  Click <strong className="text-primary">&ldquo;Select Image Folder&rdquo;</strong> to open the native
                  directory browser or select multiple lunar image files.
                </p>
              </div>
              <button
                type="button"
                onClick={() => folderInputRef.current?.click()}
                className="mt-2 inline-flex items-center gap-2 rounded-md bg-cyan-700 px-4 py-2 text-xs font-semibold text-white hover:bg-primary/20"
              >
                <FolderOpen className="size-3.5 text-cyan-200" />
                <span>Browse Folder Now</span>
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-left text-xs border-collapse">
                <thead className="bg-muted text-muted-foreground uppercase font-mono text-[10px]">
                  <tr>
                    <th className="p-2.5 w-10 text-center">Batch</th>
                    <th className="p-2.5 w-14">Preview</th>
                    <th className="p-2.5">File / Relative Path</th>
                    <th className="p-2.5 w-24">Sensor & GSD</th>
                    <th className="p-2.5 w-20">Bit Depth</th>
                    <th className="p-2.5 w-24">Dimensions</th>
                    <th className="p-2.5 w-20">Size</th>
                    <th className="p-2.5 w-24">Status</th>
                    <th className="p-2.5 w-48 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200">
                  {filteredFiles.map((file) => {
                    const isMoving = selectedMoving?.absolute_path === file.absolute_path;
                    const isFixed = selectedFixed?.absolute_path === file.absolute_path;
                    const isBatchSelected = selectedBatch.has(file.absolute_path);
                    const isReady = file.status === 'READY';

                    return (
                      <tr
                        key={file.absolute_path}
                        className={`hover:bg-muted transition ${
                          isMoving
                            ? 'bg-amber-500/10/60'
                            : isFixed
                            ? 'bg-primary/10/60'
                            : ''
                        }`}
                      >
                        <td className="p-2.5 text-center">
                          {isReady && (
                            <input
                              type="checkbox"
                              checked={isBatchSelected}
                              onChange={() => toggleBatchSelect(file.absolute_path)}
                              className="accent-cyan-700"
                            />
                          )}
                        </td>
                        <td className="p-2.5">
                          {file.preview_url ? (
                            <img
                              src={file.preview_url}
                              alt={file.filename}
                              className="size-10 rounded border border-border object-cover bg-card"
                            />
                          ) : (
                            <div className="size-10 rounded border border-border bg-muted flex items-center justify-center text-[9px] text-muted-foreground font-mono">
                              N/A
                            </div>
                          )}
                        </td>
                        <td className="p-2.5 font-medium text-card-foreground">
                          <p className="truncate max-w-[200px]" title={file.filename}>
                            {file.filename}
                          </p>
                          <p className="font-mono text-[10px] text-muted-foreground truncate max-w-[200px]">
                            {file.relative_path}
                          </p>
                        </td>
                        <td className="p-2.5 text-[11px] text-muted-foreground">
                          {file.sensor ? (
                            <span className="truncate block max-w-[150px] font-medium text-primary" title={file.sensor}>
                              {file.sensor}
                            </span>
                          ) : (
                            <span className="text-muted-foreground font-mono text-[10px]">UNKNOWN</span>
                          )}
                        </td>
                        <td className="p-2.5 font-mono text-[11px]">
                          <span className={`px-1.5 py-0.5 rounded font-semibold ${
                            file.bit_depth === 16
                              ? 'bg-purple-50 text-purple-800 border border-purple-200'
                              : 'bg-muted text-muted-foreground'
                          }`}>
                            {file.bit_depth ? `${file.bit_depth}-bit` : '8-bit'}
                          </span>
                        </td>
                        <td className="p-2.5 font-mono text-[11px] text-muted-foreground">
                          {file.dimensions ? `${file.dimensions[0]}×${file.dimensions[1]}` : '—'}
                        </td>
                        <td className="p-2.5 font-mono text-[11px] text-muted-foreground">
                          {file.size_bytes !== undefined ? (file.size_bytes / (1024 * 1024)).toFixed(2) + ' MB' : '—'}
                        </td>
                        <td className="p-2.5">
                          {file.status === 'READY' ? (
                            <span className="inline-flex items-center gap-1 font-mono text-[10px] font-bold text-emerald-500 bg-emerald-100 px-2 py-0.5 rounded-full">
                              <Check className="size-3" /> Ready
                            </span>
                          ) : (
                            <span
                              title={file.reason || 'Unsupported scientific raster format'}
                              className="inline-flex items-center gap-1 font-mono text-[10px] font-bold text-destructive bg-rose-100 px-2 py-0.5 rounded-full cursor-help"
                            >
                              <AlertTriangle className="size-3" /> Unsupported
                            </span>
                          )}
                        </td>
                        <td className="p-2.5 text-right space-x-1 whitespace-nowrap">
                          {isReady && (
                            <>
                              <button
                                type="button"
                                data-testid={`button-inspect-file-${file.filename}`}
                                onClick={() => setInspectingFile(file)}
                                className="px-2 py-1 text-[10px] font-semibold rounded border border-primary/20 bg-primary/10 text-primary hover:bg-cyan-100 transition inline-flex items-center gap-1"
                                title="Inspect scientific data, bit depth, and histogram"
                              >
                                <Binary className="size-3" />
                                <span>Inspect</span>
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  setSelectedMoving(file);
                                  onSelectImage?.(file.absolute_path, file.filename, 'source');
                                }}
                                className={`px-2 py-1 text-[10px] font-semibold rounded border transition ${
                                  isMoving
                                    ? 'bg-amber-600 text-white border-amber-700'
                                    : 'border-border text-muted-foreground hover:bg-muted'
                                }`}
                              >
                                Moving
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  setSelectedFixed(file);
                                  onSelectImage?.(file.absolute_path, file.filename, 'reference');
                                }}
                                className={`px-2 py-1 text-[10px] font-semibold rounded border transition ${
                                  isFixed
                                    ? 'bg-cyan-700 text-white border-primary/20'
                                    : 'border-border text-muted-foreground hover:bg-muted'
                                }`}
                              >
                                Fixed
                              </button>
                            </>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border bg-muted p-4 rounded-b-xl">
          <div className="text-xs text-muted-foreground">
            {selectedMoving && selectedFixed ? (
              <span>
                Selected Pair: <strong className="text-amber-500">{selectedMoving.filename}</strong> (Moving) →{' '}
                <strong className="text-primary">{selectedFixed.filename}</strong> (Fixed)
              </span>
            ) : (
              <span>Select Moving (Source) and Fixed (Reference) images to register.</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {selectedBatch.size >= 2 && (
              <button
                type="button"
                onClick={handleApplyBatch}
                className="focus-ring flex items-center gap-1.5 rounded-md bg-cyan-700 px-4 py-2 text-xs font-semibold text-white hover:bg-primary/20"
              >
                <Layers3 className="size-3.5" /> Load {selectedBatch.size} Images into Strip
              </button>
            )}
            <button
              type="button"
              disabled={!selectedMoving || !selectedFixed}
              onClick={handleApplyPair}
              className="focus-ring flex items-center gap-1.5 rounded-md bg-orange-600 px-4 py-2 text-xs font-semibold text-white hover:bg-orange-700 disabled:opacity-40"
            >
              <span>Load Pair to Workstation</span>
              <ArrowRight className="size-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Scientific Data Inspector Modal */}
      <ScientificDataInspectorModal
        isOpen={Boolean(inspectingFile)}
        onClose={() => setInspectingFile(null)}
        fileItem={inspectingFile}
      />
    </div>
  );
}
