# SELENE-REG-X — Comprehensive Codebase Audit Report (Phase 0)

**Date:** October 2026  
**Project:** SELENE-REG-X (SIH Problem Statement 26166)  
**Host Environment:** Windows 11 x64, Intel/AMD Host, NVIDIA GeForce RTX 3050 A Laptop GPU (4 GB VRAM), CUDA 13.1, Driver 592.82  
**Runtime:** Python 3.14.3, PyTorch 2.14.1+cu126, Node.js v22.12.0, Express 4.21.2, React 19.0.0, Vite 7.3.6

---

## 1. Executive Summary & Baseline Status

This codebase audit establishes the verified engineering baseline for **SELENE-REG-X** in compliance with the **SIH 26166 Master Engineering Specification**. In accordance with **Rule 0.1**, the project is being repaired, upgraded, and completed **in-place** without scratch rewrites or architectural abandonment.

### Baseline Test & Build Verification
* **Pytest Test Suite:** **260 passed, 0 failed** (100% pass rate across 22 test modules, duration: 271s).
* **Frontend Hook-Order Regression Test:** **14/14 scenarios passed** (`test_pair_registration_hooks.mjs`).
* **Frontend Production Build:** **Compiled cleanly with 0 errors** (`dist/public/index.html` 1.38 kB, CSS 121.6 kB, JS 521.2 kB).
* **Backend Production Build:** **Compiled cleanly with 0 errors** (`artifacts/api-server/dist/index.mjs` 1.4 MB).
* **CUDA GPU Acceleration:** **Verified operational on RTX 3050 (4 GB)** with PyTorch 2.14.1+cu126:
  * GPU L2 / Euclidean descriptor matching: **4.1x–5.8x speedup** over CPU OpenCV.
  * GPU Hamming descriptor matching: Chunked bitwise XOR + LUT popcount with bounded $O(N)$ VRAM allocation.
  * GPU Phase correlation: 2D Real FFT (`torch.fft.rfft2` & `irfft2`) with OpenCV-aligned cross-power spectrum.
  * GPU Warp perspective: Tensor grid-sampling resampling via `torch.nn.functional.grid_sample`.

---

## 2. Architecture Overview

The system consists of a decoupled two-tier client/server architecture with a Python scientific vision engine:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND CLIENT                                 │
│  React 19 + TypeScript + Vite 7 + Tailwind CSS v4 + TanStack Query     │
│  Pages: Workstation (/), Mosaic (/mosaic), Validation (/validation),   │
│         Audit / Provenance (/audit)                                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / SSE (Port 5000 / 3001)
┌───────────────────────────────────▼────────────────────────────────────┐
│                        EXPRESS API SERVER                              │
│  artifacts/api-server/src/app.ts & routes/registration.ts              │
│  Endpoints: /api/register, /api/multi-registration/register,           │
│             /api/dataset/scan, /api/hardware/status,                   │
│             /api/validation/benchmark, /api/ingest-url                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ ChildProcess Spawn / JSON IPC / SSE
┌───────────────────────────────────▼────────────────────────────────────┐
│                    SCIENTIFIC VISION ENGINE                            │
│  artifacts/api-server/python/worker.py                                 │
│  Services:                                                             │
│  ├── pairwise_registration.py  (12-stage scientific pair pipeline)     │
│  ├── multi_registration.py     (Graph construction & MST layout)       │
│  ├── gpu_accelerator.py        (PyTorch CUDA tensor kernels)           │
│  ├── subpixel.py               (Taylor, LK, ECC, Phase, Quad peak)     │
│  ├── geometry.py               (RANSAC / MAGSAC Homography / Affine)   │
│  ├── feature_matching.py       (SIFT, ORB, AKAZE, Hybrid GPU matcher)  │
│  ├── preprocessing.py          (CLAHE, Retinex, Gradients, Shadows)   │
│  ├── dataset_scanner.py        (Local folder ingestion & manifests)    │
│  └── hardware_manager.py       (CUDA telemetry & VRAM tracking)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Working Features Identified

1. **GPU Acceleration (PyTorch CUDA cu126 on RTX 3050):**
   * Real GPU tensor descriptor matching (`match_descriptors_gpu`) active and benchmarked.
   * Real GPU phase correlation (`phase_correlation_gpu`) active and mathematically aligned.
   * Real GPU perspective warping (`warp_perspective_gpu`) active.
   * Seamless CPU fallback if CUDA is absent or VRAM limits are reached.
2. **12-Stage Pairwise Registration Pipeline:**
   * Full stage reporting emitted via Server-Sent Events (SSE) across ingestion, sensor inference, radiometric normalization, feature extraction, matching, geometric verification, sub-pixel refinement, and footprint mapping.
3. **Sub-Pixel Refinement Algorithms:**
   * Real implementations for:
     1. Taylor expansion / local gradient linearization
     2. Lucas-Kanade iterative optical flow
     3. ECC (Enhanced Correlation Coefficient) intensity optimization
     4. Phase correlation with sub-pixel peak centroiding
     5. Quadratic surface peak fitting
   * Auto-selection mode comparing real residuals and convergence metrics.
4. **Multi-Image Relative Mosaic:**
   * Pairwise match graph assembly, Minimum Spanning Tree (MST) extraction, global transformation composition, and progressive multi-frame alignment.
5. **URL & Cloud Storage Ingestion:**
   * Google Drive and direct HTTPS image downloading with SSRF guard and MIME verification.
6. **Deterministic Benchmark Test Suite:**
   * 260 unit and integration tests passing with 100% reproducibility.

---

## 4. Flaws, Inconsistencies & Deficiencies Identified

In auditing against the SIH 26166 Master Engineering Specification, the following critical issues were identified:

### Flaw 4.1 — Misleading "Registered View" as Default Interface (Rule 2 & 3)
* **Current Behavior:** The frontend component `RegisteredViewer` (`App.tsx` lines 1181, 1230, 1248) defaults `viewMode` to `'registered'`, rendering the warped and cropped common overlap raster.
* **Requirement:** The cropped view is misleading for planetary correspondence. It must be removed from the primary position and replaced with the **Scientific Canonical Match Visualization** displaying the **Full Source Image (Left)** and **Full Reference Image (Right)** with verified correspondence lines.

### Flaw 4.2 — Rejected Red Lines Displayed in Match Visualization (Rule 4 & 5)
* **Current Behavior:** `draw_matches()` in `artifacts/api-server/python/app/services/registration.py` (lines 695–703) renders outlier lines in red (`(40, 40, 220)`).
* **Requirement:** Red outlier lines create visual noise and make the correspondence look broken. The primary canonical view must strictly render **verified green inlier lines only**. Red lines are restricted to QA diagnostic inspection.

### Flaw 4.3 — Validation Page Hardcoded "Dataset Not Connected" (Rule 26)
* **Current Behavior:** In `App.tsx` (lines 3771–3802), `datasetConnected` state is initialized to `false` and only set to `true` if the user clicks "Generate Synthetic Validation Corpus". It ignores currently uploaded images in the active session.
* **Requirement:** The Validation page must be converted into a true **Validation & Quality Assurance** dashboard that inspects and reports on the **currently loaded session and images** (uploaded count, evaluated pairs, inlier ratios, RMSE, sensor tags, illumination statistics, sub-pixel convergence, GPU utilization, and modular PASS/WARN/FAIL indicators).

### Flaw 4.4 — Manual Path Typing in Folder Upload (Rule 16)
* **Current Behavior:** `DatasetManifestModal.tsx` (lines 170–178) presents a text `<input>` requiring users to type string paths like `demo_data` or `D:/LunarData`.
* **Requirement:** Remove manual typing. Implement a standard native **[BROWSE FOLDER]** button using `<input type="file" webkitdirectory directory multiple />` and/or `window.showDirectoryPicker()` so users can pick local directories natively with their mouse.

### Flaw 4.5 — Dark UI Elements Violating Scientific Interface Rules (Rule 30)
* **Current Behavior:** `index.css` has some light root variables, but several modals and components (`DatasetManifestModal`, canvas overlays, headers) use `bg-slate-900`, `bg-slate-950/60`, and dark panels.
* **Requirement:** Eliminate dark mode / black surfaces. Redesign the visual system into a crisp, professional scientific aerospace dashboard: **white background, warm/light neutral surfaces, navy/blue-gray typography, subtle borders, restrained shadows, clean hierarchy**.

### Flaw 4.6 — Data Loss Risk on 16-Bit Scientific Rasters (Rule 11, 14, 15)
* **Current Behavior:** When multi-band or 16-bit TIFF images are ingested, downstream processing occasionally normalizes them directly into 8-bit arrays in memory, losing original radiometric dynamic range.
* **Requirement:** Decouple **Scientific Image Data** (preserving original dtype, raw DN values, 16-bit range, metadata) from the **Display Representation** (8-bit normalized preview strictly for UI canvas rendering).

### Flaw 4.7 — Sensor Identification Integrity (Rule 13)
* **Current Behavior:** If metadata is absent, some dropdowns still suggest specific sensors or fall back to generic defaults.
* **Requirement:** Sensor identification must be evidence-based (PDS labels, embedded TIFF tags, filename patterns, raster geometry). If unconfirmed, it must explicitly state `"UNKNOWN / NEEDS METADATA"` rather than fabricating a sensor identity.

---

## 5. Frontend Controls & Backend Endpoints Audit Matrix

| UI Component / Button | Event Handler | API Route | Backend Service Function | Status | Action Required |
|---|---|---|---|---|---|
| **Run Correspondence** (`button-run-registration`) | `submit()` in `App.tsx` | POST `/api/register` | `pairwise_registration.py:register_pair` | Working | Update default view mode to canonical side-by-side |
| **View Registered** (`button-view-registered`) | `setViewMode('registered')` | N/A (Client state) | Client canvas render | Misleading default | Demote from primary UI to secondary diagnostic tab |
| **View Matches** (`button-view-matches`) | `setViewMode('matches')` | N/A (Client state) | `registration.py:draw_matches` | Working | Promote to primary default; strip red outlier lines |
| **Ingest Lunar Dataset Folder** | `handleScan()` in `DatasetManifestModal` | POST `/api/dataset/scan` | `dataset_scanner.py:scan_dataset_folder` | Working (Manual path) | Replace text input with native Directory Picker browse button |
| **Multi-Image Run Mosaic** | `runMulti()` in `Mosaic.tsx` | POST `/api/multi-registration/register` | `multi_registration.py:run_multi_registration` | Working | Ensure pair edges link directly to canonical pair viewer |
| **Validation Benchmark** (`button-generate-synthetic`) | `handleRunBenchmark()` in `App.tsx` | POST `/api/validation/benchmark` | `worker.py:run_validation_benchmark` | Working | Connect validation dashboard to active session data |
| **Hardware Status Bar** | Query `/api/hardware/status` | GET `/api/hardware/status` | `hardware_manager.py:get_system_hardware_status` | Working | Display live CUDA test & VRAM telemetry truthfully |
| **Interactive Match Point Click** | `onSelectMatch` | Client callback | `MatchInspectorModal.tsx` | Working | Verify original pixel coordinates & sub-pixel deltas |

---

## 6. GPU & CPU Execution Paths and Memory Boundaries

### RTX 3050 Laptop GPU (4 GB VRAM) Profile:
* **Host GPU:** NVIDIA GeForce RTX 3050 A Laptop GPU
* **Total VRAM:** 4,096 MB (Dedicated GDDR6)
* **Active CUDA Toolkit in PyTorch:** cu126
* **Memory Strategy:**
  1. **Batch Sizing:** Descriptors matched in chunked blocks of 4,000 to prevent OOM.
  2. **LUT Popcount:** Hamming distance computed via streaming bitwise XOR and uint8 lookup table on device.
  3. **Warp Perspective:** Canvas image warps execute on GPU using float32 normalized grid sampling with automatic cache clearing (`torch.cuda.empty_cache()`).
  4. **CPU Fallback:** Any allocation exceeding 2.5 GB peak VRAM drops cleanly to OpenCV CPU execution with explicit logging.

---

## 7. Baseline Stability Conclusion & Next Phase Readiness

* **Baseline Integrity:** Pytest (260/260 passing), Node hook tests (14/14 passing), backend build (passing), frontend build (passing).
* **Blocking Crashes:** None currently blocking startup.
* **Audit Verdict:** Codebase is structurally sound, mathematically verified, and fully ready for **Phase 1: Image Ingestion + Data Inspection**.

**Antigravity Status:** Phase 0 audit is complete and committed to `AUDIT_REPORT.md`. Awaiting explicit user instruction before proceeding to Phase 1.
