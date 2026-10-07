import { randomUUID } from "node:crypto";
import { mkdir, writeFile, readdir, stat, rm } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import type { Request, Response, Router as RouterType } from "express";
import { Router } from "express";

import { existsSync } from "node:fs";

type MultipartPart = {
  name: string;
  filename?: string;
  contentType?: string;
  data: Buffer;
};

const router: RouterType = Router();
const baseDir = existsSync(path.resolve(process.cwd(), "python/worker.py"))
  ? process.cwd()
  : existsSync(path.resolve(process.cwd(), "artifacts/api-server/python/worker.py"))
  ? path.resolve(process.cwd(), "artifacts/api-server")
  : process.cwd();
const dataRoot = path.join(baseDir, "data");
const pythonRoot = path.join(baseDir, "python");
const pythonBin =
  process.env.PYTHON_BIN || (process.platform === "win32" ? "python" : "python3");
const maxUploadBytes = 80 * 1024 * 1024;

async function readRequestBody(req: Request): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > maxUploadBytes) {
      throw new Error("Upload exceeds the 80 MB pair-registration limit.");
    }
    chunks.push(buffer);
  }
  return Buffer.concat(chunks);
}

function parseHeaderBlock(block: string): Record<string, string> {
  return Object.fromEntries(
    block
      .split("\r\n")
      .map((line) => {
        const index = line.indexOf(":");
        return index < 0
          ? [line.toLowerCase(), ""]
          : [line.slice(0, index).trim().toLowerCase(), line.slice(index + 1).trim()];
      })
      .filter(([key]) => key),
  );
}

function parseDisposition(value: string) {
  const name = /name="([^"]+)"/i.exec(value)?.[1];
  const filename = /filename="([^"]*)"/i.exec(value)?.[1];
  return { name, filename };
}

function parseMultipart(body: Buffer, boundary: string): MultipartPart[] {
  const marker = Buffer.from(`--${boundary}`);
  const parts: MultipartPart[] = [];
  let cursor = 0;
  while (cursor < body.length) {
    const start = body.indexOf(marker, cursor);
    if (start < 0) break;
    const afterMarker = start + marker.length;
    if (body.subarray(afterMarker, afterMarker + 2).toString() === "--") break;
    const partStart = afterMarker + 2;
    const headerEnd = body.indexOf(Buffer.from("\r\n\r\n"), partStart);
    if (headerEnd < 0) break;
    const nextBoundary = body.indexOf(marker, headerEnd + 4);
    if (nextBoundary < 0) break;
    const headers = parseHeaderBlock(body.subarray(partStart, headerEnd).toString("utf8"));
    const disposition = parseDisposition(headers["content-disposition"] ?? "");
    if (disposition.name) {
      const end = Math.max(headerEnd + 4, nextBoundary - 2);
      parts.push({
        name: disposition.name,
        filename: disposition.filename || undefined,
        contentType: headers["content-type"],
        data: body.subarray(headerEnd + 4, end),
      });
    }
    cursor = nextBoundary;
  }
  return parts;
}

function parseScalar(parts: MultipartPart[], name: string, fallback: string) {
  const part = parts.find((item) => item.name === name);
  return part ? part.data.toString("utf8") : fallback;
}

function parseBool(parts: MultipartPart[], name: string, fallback: boolean) {
  const value = parseScalar(parts, name, String(fallback)).toLowerCase();
  return value === "true" || value === "1" || value === "on";
}

function parseNumber(parts: MultipartPart[], name: string): number | null {
  const value = parseScalar(parts, name, "");
  if (!value.trim()) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function safeFilename(filename: string | undefined, fallback: string) {
  const clean = path.basename(filename || fallback).replace(/[^a-zA-Z0-9._-]/g, "_");
  return clean || fallback;
}

export interface RegistrationProgressEvent {
  job_id: string;
  stage: string;
  stage_index: number;
  stage_count: number;
  progress: number;
  message: string;
  status: "RUNNING" | "COMPLETE" | "FAILED";
  error?: string;
}

export type ProgressCallback = (event: RegistrationProgressEvent) => void;

interface RegistrationJobState {
  job_id: string;
  stage: string;
  stage_index: number;
  stage_count: number;
  progress: number;
  message: string;
  status: "INITIALIZING" | "RUNNING" | "COMPLETE" | "FAILED";
  error?: string;
  created_at: number;
  updated_at: number;
  listeners: Set<(event: RegistrationProgressEvent) => void>;
}

const registrationJobs = new Map<string, RegistrationJobState>();

function cleanupOldJobs(maxAgeMs = 1800_000) {
  const now = Date.now();
  for (const [jobId, job] of registrationJobs.entries()) {
    if (now - job.updated_at > maxAgeMs) {
      registrationJobs.delete(jobId);
    }
  }
}

function getOrCreateJob(jobId: string): RegistrationJobState {
  cleanupOldJobs();
  let job = registrationJobs.get(jobId);
  if (!job) {
    job = {
      job_id: jobId,
      stage: "INITIALIZING",
      stage_index: 1,
      stage_count: 12,
      progress: 0.08,
      message: "Initializing registration pipeline",
      status: "INITIALIZING",
      created_at: Date.now(),
      updated_at: Date.now(),
      listeners: new Set(),
    };
    registrationJobs.set(jobId, job);
  }
  return job;
}

function updateJobProgress(jobId: string, event: RegistrationProgressEvent): void {
  const job = getOrCreateJob(jobId);
  job.stage = event.stage;
  job.stage_index = event.stage_index;
  job.stage_count = event.stage_count;
  job.progress = event.progress;
  job.message = event.message;
  job.status = event.status;
  if (event.error) job.error = event.error;
  job.updated_at = Date.now();

  for (const listener of job.listeners) {
    try {
      listener(event);
    } catch {
      job.listeners.delete(listener);
    }
  }
}

async function runPython(
  args: string[],
  timeoutMs = 180_000,
  onProgress?: ProgressCallback,
): Promise<unknown> {
  return await new Promise((resolve, reject) => {
    const child = spawn(pythonBin, [path.join(pythonRoot, "worker.py"), ...args], {
      env: { ...process.env, PYTHONPATH: pythonRoot, PYTHONUNBUFFERED: "1" },
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let stderrRemainder = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error("The vision pipeline exceeded its processing time limit."));
    }, timeoutMs);
    child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => {
      stderr.push(chunk);
      if (onProgress) {
        stderrRemainder += chunk.toString("utf8");
        const lines = stderrRemainder.split("\n");
        stderrRemainder = lines.pop() ?? "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("PROGRESS:")) {
            try {
              const event = JSON.parse(trimmed.slice("PROGRESS:".length)) as RegistrationProgressEvent;
              onProgress(event);
            } catch {
              // ignore malformed progress json
            }
          }
        }
      }
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      const outStr = Buffer.concat(stdout).toString("utf8").trim();
      const details = Buffer.concat(stderr).toString("utf8").trim();
      if (code !== 0) {
        try {
          const errObj = JSON.parse(outStr);
          if (errObj && errObj.detail) {
            const err = new Error(errObj.detail);
            Object.assign(err, errObj);
            reject(err);
            return;
          }
        } catch {
          // not JSON in stdout, fallback
        }
        reject(new Error(details || `Vision worker exited with code ${code ?? "unknown"}.`));
        return;
      }
      try {
        resolve(JSON.parse(outStr));
      } catch {
        reject(new Error("Vision worker returned invalid JSON."));
      }
    });
  });
}

function buildSettings(parts: MultipartPart[]) {
  return {
    detector: parseScalar(parts, "detector", "sift"),
    ratio: Number(parseScalar(parts, "ratio", "0.72")),
    ransac_threshold: Number(parseScalar(parts, "ransac_threshold", "3")),
    max_features: Number(parseScalar(parts, "max_features", "8000")),
    illumination_normalization: parseBool(parts, "illumination_normalization", true),
    spatial_distribution: parseBool(parts, "spatial_distribution", true),
    radiometric_mode: parseScalar(parts, "radiometric_mode", "safe_normalization"),
    representation: parseScalar(parts, "representation", "structural"),
    geometric_model: parseScalar(parts, "geometric_model", "auto"),
    execution_mode: parseScalar(parts, "execution_mode", "hybrid").toLowerCase(),
    refinement_methods: parseScalar(
      parts,
      "refinement_methods",
      "taylor,ecc,lucas_kanade,phase,quadratic",
    ),
    source_metadata: {
      sensor: parseScalar(parts, "source_sensor", "UNKNOWN"),
      gsd_m: parseNumber(parts, "source_gsd_m"),
      solar_incidence_deg: parseNumber(parts, "source_solar_incidence_deg"),
      solar_azimuth_deg: parseNumber(parts, "source_solar_azimuth_deg"),
      radiance_scale: parseNumber(parts, "source_radiance_scale"),
      radiance_offset: parseNumber(parts, "source_radiance_offset"),
    },
    reference_metadata: {
      sensor: parseScalar(parts, "reference_sensor", "UNKNOWN"),
      gsd_m: parseNumber(parts, "reference_gsd_m"),
      solar_incidence_deg: parseNumber(parts, "reference_solar_incidence_deg"),
      solar_azimuth_deg: parseNumber(parts, "reference_solar_azimuth_deg"),
      radiance_scale: parseNumber(parts, "reference_radiance_scale"),
      radiance_offset: parseNumber(parts, "reference_radiance_offset"),
    },
  };
}

async function writeUpload(root: string, part: MultipartPart, fallback: string) {
  const filePath = path.join(root, safeFilename(part.filename, fallback));
  await writeFile(filePath, part.data);
  return filePath;
}

async function registerPair(req: Request, res: Response) {
  const contentType = req.header("content-type") ?? "";
  const boundary = /boundary="?([^";]+)"?/i.exec(contentType)?.[1];
  if (!boundary) {
    res.status(400).json({ detail: "Pair registration requires a multipart/form-data request." });
    return;
  }
  try {
    const parts = parseMultipart(await readRequestBody(req), boundary);
    const sourcePart = parts.find((part) => part.name === "source" && part.filename);
    const referencePart = parts.find((part) => part.name === "reference" && part.filename);
    if (!sourcePart || !referencePart) {
      res.status(400).json({ detail: "Both source and reference image files are required." });
      return;
    }
    const clientJobId = parseScalar(parts, "job_id", "") || req.header("x-job-id") || (req.query.job_id as string);
    const jobId = clientJobId ? safeFilename(clientJobId, randomUUID().replaceAll("-", "").slice(0, 12)) : randomUUID().replaceAll("-", "").slice(0, 12);
    const jobRoot = path.join(dataRoot, "jobs", jobId);
    const outputRoot = path.join(dataRoot, "outputs", jobId);
    await mkdir(jobRoot, { recursive: true });
    await mkdir(outputRoot, { recursive: true });
    const sourcePath = await writeUpload(jobRoot, sourcePart, "source.png");
    const referencePath = await writeUpload(jobRoot, referencePart, "reference.png");
    const settings = { ...buildSettings(parts), job_id: jobId };
    const settingsPath = path.join(jobRoot, "settings.json");
    await writeFile(settingsPath, JSON.stringify(settings, null, 2), "utf8");

    getOrCreateJob(jobId);

    const onProgress = (event: RegistrationProgressEvent) => {
      updateJobProgress(jobId, event);
    };

    let payload: Record<string, unknown>;
    try {
      payload = (await runPython([
        "--mode",
        "pair",
        "--source",
        sourcePath,
        "--reference",
        referencePath,
        "--out-dir",
        outputRoot,
        "--public-prefix",
        `/api/outputs/${jobId}`,
        "--settings-file",
        settingsPath,
        "--settings-json",
        JSON.stringify(settings),
        "--job-id",
        jobId,
      ], 180_000, onProgress)) as Record<string, unknown>;
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : "Pair registration failed.";
      updateJobProgress(jobId, {
        job_id: jobId,
        stage: "FAILED",
        stage_index: 0,
        stage_count: 12,
        progress: 1.0,
        message: errMsg,
        status: "FAILED",
        error: errMsg,
      });
      throw err;
    }

    payload.job_id = jobId;
    res.setHeader("X-Job-ID", jobId);
    res.json(payload);
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Pair registration failed.";
    res.status(detail.includes("limit") ? 413 : 500).json({ detail });
  }
}

function streamProgress(req: Request, res: Response) {
  const rawJobId = req.params.jobId;
  const jobId = Array.isArray(rawJobId) ? rawJobId[0] : (rawJobId ?? "");
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");

  const job = getOrCreateJob(jobId);

  const initialPayload: RegistrationProgressEvent = {
    job_id: job.job_id,
    stage: job.stage,
    stage_index: job.stage_index,
    stage_count: job.stage_count,
    progress: job.progress,
    message: job.message,
    status: job.status === "INITIALIZING" ? "RUNNING" : job.status,
    error: job.error,
  };
  res.write(`data: ${JSON.stringify(initialPayload)}\n\n`);

  if (job.status === "COMPLETE" || job.status === "FAILED") {
    res.end();
    return;
  }

  const listener = (event: RegistrationProgressEvent) => {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
    if (event.status === "COMPLETE" || event.status === "FAILED") {
      res.end();
    }
  };

  job.listeners.add(listener);

  req.on("close", () => {
    job.listeners.delete(listener);
  });
}

function getJobStatus(req: Request, res: Response) {
  const rawJobId = req.params.jobId;
  const jobId = Array.isArray(rawJobId) ? rawJobId[0] : (rawJobId ?? "");
  const job = registrationJobs.get(jobId);
  if (!job) {
    res.status(404).json({ detail: `Registration job '${jobId}' not found.` });
    return;
  }
  res.json({
    job_id: job.job_id,
    stage: job.stage,
    stage_index: job.stage_index,
    stage_count: job.stage_count,
    progress: job.progress,
    message: job.message,
    status: job.status,
    error: job.error,
    updated_at: job.updated_at,
  });
}

async function syntheticRobustness(req: Request, res: Response) {
  const contentType = req.header("content-type") ?? "";
  const boundary = /boundary="?([^";]+)"?/i.exec(contentType)?.[1];
  if (!boundary) {
    res.status(400).json({ detail: "Synthetic validation requires a multipart/form-data request." });
    return;
  }
  try {
    const parts = parseMultipart(await readRequestBody(req), boundary);
    const imagePart = parts.find((part) => part.name === "image" && part.filename);
    if (!imagePart) {
      res.status(400).json({ detail: "A source image is required for synthetic validation." });
      return;
    }
    const jobRoot = path.join(dataRoot, "jobs", randomUUID().slice(0, 12));
    await mkdir(jobRoot, { recursive: true });
    const imagePath = await writeUpload(jobRoot, imagePart, "source.png");
    const settings = buildSettings(parts);
    const payload = await runPython([
      "--mode",
      "robustness",
      "--source",
      imagePath,
      "--out-dir",
      dataRoot,
      "--public-prefix",
      "/api/outputs",
      "--settings-json",
      JSON.stringify(settings),
    ]);
    res.json(payload);
  } catch (error) {
    res.status(500).json({ detail: error instanceof Error ? error.message : "Synthetic validation failed." });
  }
}

async function registerCollection(req: Request, res: Response) {
  const contentType = req.header("content-type") ?? "";
  const boundary = /boundary="?([^";]+)"?/i.exec(contentType)?.[1];
  if (!boundary) {
    res.status(400).json({ detail: "Multi-image registration requires multipart/form-data." });
    return;
  }
  try {
    const parts = parseMultipart(await readRequestBody(req), boundary);
    const imageParts = parts.filter((part) => part.name === "images" && part.filename);
    
    // Ingest any URLs provided in the request
    let suppliedUrls: string[] = [];
    try {
      const urlsRaw = parseScalar(parts, "urls_json", "") || parseScalar(parts, "urls", "");
      if (urlsRaw) {
        const parsed = JSON.parse(urlsRaw);
        if (Array.isArray(parsed)) suppliedUrls = parsed.filter((u) => typeof u === "string" && u.trim());
      }
    } catch {
      // not JSON list
    }
    for (const p of parts) {
      if (p.name === "url" && !p.filename && p.data.toString("utf8").trim()) {
        suppliedUrls.push(p.data.toString("utf8").trim());
      }
    }

    const totalInputCount = imageParts.length + suppliedUrls.length;
    if (totalInputCount < 2 || totalInputCount > 200) {
      res.status(400).json({ detail: "Provide between 2 and 200 total images (uploaded files + URLs) for a relative mosaic." });
      return;
    }

    const clientJobId = parseScalar(parts, "job_id", "") || req.header("x-job-id") || (req.query.job_id as string);
    const jobId = clientJobId ? safeFilename(clientJobId, randomUUID().replaceAll("-", "").slice(0, 12)) : randomUUID().replaceAll("-", "").slice(0, 12);
    const jobRoot = path.join(dataRoot, "jobs", jobId);
    const outputRoot = path.join(dataRoot, "outputs", jobId);
    await mkdir(jobRoot, { recursive: true });
    await mkdir(outputRoot, { recursive: true });

    getOrCreateJob(jobId);
    const onProgress = (event: RegistrationProgressEvent) => {
      updateJobProgress(jobId, event);
    };

    const imagePaths = [];
    for (const [index, part] of imageParts.entries()) {
      imagePaths.push(await writeUpload(jobRoot, part, `image-${index + 1}.png`));
    }

    let urlMetadata: Record<string, unknown>[] = [];
    for (const [urlIdx, urlStr] of suppliedUrls.entries()) {
      const ingestTarget = path.join(jobRoot, `url-${urlIdx + 1}`);
      await mkdir(ingestTarget, { recursive: true });
      const ingestRes = (await runPython([
        "--mode",
        "ingest-url",
        "--url",
        urlStr.trim(),
        "--out-dir",
        ingestTarget,
        "--public-prefix",
        `/api/outputs/${jobId}/url-${urlIdx + 1}`,
      ])) as Record<string, unknown>;

      if (ingestRes && ingestRes.local_path) {
        imagePaths.push(String(ingestRes.local_path));
        urlMetadata.push({
          label: String(ingestRes.filename || `url_frame_${urlIdx + 1}`),
          provenance: ingestRes.provenance,
          source_type: (ingestRes.provenance as any)?.source_type || "HTTPS_URL",
        });
      }
    }

    let metadata: unknown[] = [];
    try {
      const supplied = JSON.parse(parseScalar(parts, "metadata_json", "[]"));
      if (Array.isArray(supplied)) metadata = supplied;
    } catch {
      res.status(400).json({ detail: "metadata_json must be a JSON list when supplied." });
      return;
    }

    // Merge metadata
    const combinedMetadata = [...metadata, ...urlMetadata];
    const settings = { ...buildSettings(parts), multi_metadata: combinedMetadata, job_id: jobId };
    const settingsPath = path.join(jobRoot, "settings.json");
    await writeFile(settingsPath, JSON.stringify(settings, null, 2), "utf8");

    let payload: Record<string, unknown>;
    try {
      payload = (await runPython([
        "--mode",
        "multi",
        "--images",
        ...imagePaths,
        "--out-dir",
        outputRoot,
        "--public-prefix",
        `/api/outputs/${jobId}`,
        "--settings-file",
        settingsPath,
        "--settings-json",
        JSON.stringify(settings),
        "--job-id",
        jobId,
      ], 300_000, onProgress)) as Record<string, unknown>;
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : "Multi-image registration failed.";
      updateJobProgress(jobId, {
        job_id: jobId,
        stage: "FAILED",
        stage_index: 0,
        stage_count: 15,
        progress: 1.0,
        message: errMsg,
        status: "FAILED",
        error: errMsg,
      });
      throw err;
    }

    payload.job_id = jobId;
    res.setHeader("X-Job-ID", jobId);
    res.json(payload);
  } catch (error) {
    res.status(500).json({ detail: error instanceof Error ? error.message : "Multi-image registration failed." });
  }
}

async function runBenchmark(req: Request, res: Response) {
  const clientJobId = req.header("x-job-id") || (req.query.job_id as string);
  const jobId = clientJobId ? safeFilename(clientJobId, randomUUID().replaceAll("-", "").slice(0, 12)) : randomUUID().replaceAll("-", "").slice(0, 12);
  const jobRoot = path.join(dataRoot, "jobs", jobId);
  const outputRoot = path.join(dataRoot, "outputs", jobId);
  await mkdir(jobRoot, { recursive: true });
  await mkdir(outputRoot, { recursive: true });

  getOrCreateJob(jobId);
  const onProgress = (event: RegistrationProgressEvent) => {
    updateJobProgress(jobId, event);
  };

  try {
    const payload = (await runPython([
      "--mode",
      "validation-benchmark",
      "--out-dir",
      outputRoot,
      "--public-prefix",
      `/api/outputs/${jobId}`,
      "--job-id",
      jobId,
    ], 180_000, onProgress)) as Record<string, unknown>;

    payload.job_id = jobId;
    res.setHeader("X-Job-ID", jobId);
    res.json(payload);
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : "Validation benchmark execution failed.";
    updateJobProgress(jobId, {
      job_id: jobId,
      stage: "FAILED",
      stage_index: 0,
      stage_count: 24,
      progress: 1.0,
      message: errMsg,
      status: "FAILED",
      error: errMsg,
    });
    res.status(500).json({ detail: errMsg });
  }
}

async function cleanupOldIngestions(ingestedRoot: string, maxAgeMs = 7200_000) {
  try {
    if (!existsSync(ingestedRoot)) return;
    const entries = await readdir(ingestedRoot, { withFileTypes: true });
    const now = Date.now();
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const fullPath = path.join(ingestedRoot, entry.name);
      try {
        const stats = await stat(fullPath);
        if (now - stats.mtimeMs > maxAgeMs) {
          await rm(fullPath, { recursive: true, force: true });
        }
      } catch {
        // ignore individual errors
      }
    }
  } catch {
    // ignore directory-level error
  }
}

async function ingestUrl(req: Request, res: Response) {
  const { url } = req.body || {};
  if (!url || typeof url !== "string" || !url.trim()) {
    res.status(400).json({
      detail: "A valid 'url' string is required in the JSON body.",
      error_code: "URL_INVALID",
      recovery_hint: "Provide an HTTP/HTTPS image URL or Google Drive link.",
    });
    return;
  }

  const ingestId = randomUUID().replaceAll("-", "").slice(0, 12);
  const ingestedRoot = path.join(dataRoot, "outputs", "ingested");
  const ingestDir = path.join(ingestedRoot, ingestId);

  // Background cleanup of old temporary ingestions (older than 2 hours)
  cleanupOldIngestions(ingestedRoot).catch(() => {});

  try {
    await mkdir(ingestDir, { recursive: true });
    const payload = (await runPython([
      "--mode",
      "ingest-url",
      "--url",
      url.trim(),
      "--out-dir",
      ingestDir,
      "--public-prefix",
      `/api/outputs/ingested/${ingestId}`,
    ])) as Record<string, unknown>;

    payload.id = ingestId;
    res.json(payload);
  } catch (error) {
    const errObj = error as { message?: string; error_code?: string; recovery_hint?: string };
    const detail = errObj.message || "URL ingestion failed.";
    const status =
      errObj.error_code === "FILE_TOO_LARGE"
        ? 413
        : errObj.error_code === "DOWNLOAD_TIMEOUT"
        ? 504
        : errObj.error_code === "SSRF_BLOCKED"
        ? 403
        : 400;

    res.status(status).json({
      detail,
      error_code: errObj.error_code || "INGESTION_ERROR",
      recovery_hint: errObj.recovery_hint || "Verify the URL and try again.",
    });
  }
}

async function scanDataset(req: Request, res: Response) {
  const { folder_path, dataset_name } = req.body || {};
  let targetFolder = folder_path ? String(folder_path).trim() : "";
  if (!targetFolder) {
    targetFolder = path.resolve(process.cwd(), "demo_data");
    if (!existsSync(targetFolder)) {
      targetFolder = path.resolve(process.cwd(), "../../demo_data");
    }
  } else if (!path.isAbsolute(targetFolder)) {
    const direct = path.resolve(process.cwd(), targetFolder);
    const parent = path.resolve(process.cwd(), "..", "..", targetFolder);
    targetFolder = existsSync(direct) ? direct : existsSync(parent) ? parent : direct;
  }

  if (!existsSync(targetFolder)) {
    res.status(404).json({
      detail: `Specified dataset directory not found on host: ${targetFolder}`,
      error_code: "FOLDER_NOT_FOUND",
      recovery_hint: "Verify that the path exists on the local machine.",
    });
    return;
  }

  const manifestsDir = path.join(dataRoot, "outputs", "manifests");
  await mkdir(manifestsDir, { recursive: true });

  try {
    const payload = (await runPython([
      "--mode",
      "scan-dataset",
      "--folder-path",
      targetFolder,
      "--dataset-name",
      dataset_name || path.basename(targetFolder),
      "--out-dir",
      manifestsDir,
      "--public-prefix",
      "/api/outputs/manifests",
    ])) as Record<string, unknown>;

    res.json(payload);
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Dataset folder scanning failed.";
    res.status(500).json({ detail, error_code: "SCAN_FAILED" });
  }
}

async function getHardwareInfo(req: Request, res: Response) {
  try {
    const payload = (await runPython([
      "--mode",
      "hardware-status",
    ])) as Record<string, unknown>;
    res.json(payload);
  } catch (error) {
    res.status(500).json({ detail: "Failed querying hardware status." });
  }
}

async function runHardwareSelfTest(req: Request, res: Response) {
  try {
    const payload = (await runPython([
      "--mode",
      "gpu-self-test",
    ])) as Record<string, unknown>;
    res.json(payload);
  } catch (error) {
    res.status(500).json({ detail: "Failed executing GPU self-test." });
  }
}

async function getDatasetRawFile(req: Request, res: Response) {
  const filePath = req.query.path ? String(req.query.path) : "";
  if (!filePath) {
    res.status(400).json({ error: "Missing path parameter" });
    return;
  }
  const resolved = path.resolve(filePath);
  if (!existsSync(resolved)) {
    res.status(404).json({ error: "File not found" });
    return;
  }
  res.sendFile(resolved);
}

async function inspectImageFile(req: Request, res: Response) {
  const contentType = req.header("content-type") ?? "";
  const boundary = /boundary="?([^";]+)"?/i.exec(contentType)?.[1];
  
  try {
    let sourcePath = "";
    const inspectionId = randomUUID().replaceAll("-", "").slice(0, 12);
    const inspectDir = path.join(dataRoot, "outputs", "inspections", inspectionId);

    if (boundary) {
      const parts = parseMultipart(await readRequestBody(req), boundary);
      const filePart = parts.find((p) => (p.name === "image" || p.name === "file") && p.filename);
      if (!filePart) {
        res.status(400).json({ detail: "An image file is required for inspection." });
        return;
      }
      await mkdir(inspectDir, { recursive: true });
      sourcePath = await writeUpload(inspectDir, filePart, "inspect_target.png");
    } else {
      const filePath = req.body?.path ? String(req.body.path) : (req.query.path ? String(req.query.path) : "");
      if (!filePath) {
        res.status(400).json({ detail: "Missing image path or uploaded file." });
        return;
      }
      sourcePath = path.resolve(filePath);
      if (!existsSync(sourcePath)) {
        res.status(404).json({ detail: `Image file not found: ${sourcePath}` });
        return;
      }
      await mkdir(inspectDir, { recursive: true });
    }

    const payload = (await runPython([
      "--mode",
      "inspect-image",
      "--source",
      sourcePath,
      "--out-dir",
      inspectDir,
      "--public-prefix",
      `/api/outputs/inspections/${inspectionId}`,
    ])) as Record<string, unknown>;

    res.json(payload);
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Scientific image inspection failed.";
    res.status(500).json({ detail, error_code: "INSPECTION_FAILED" });
  }
}

async function uploadFolder(req: Request, res: Response) {
  const contentType = req.header("content-type") ?? "";
  const boundary = /boundary="?([^";]+)"?/i.exec(contentType)?.[1];
  if (!boundary) {
    res.status(400).json({ detail: "Folder upload requires a multipart/form-data request." });
    return;
  }

  try {
    const parts = parseMultipart(await readRequestBody(req), boundary);
    const fileParts = parts.filter((p) => p.filename && p.data && p.data.length > 0);
    if (fileParts.length === 0) {
      res.status(400).json({ detail: "No files found in folder upload." });
      return;
    }

    const uploadId = randomUUID().replaceAll("-", "").slice(0, 12);
    const folderUploadRoot = path.join(dataRoot, "outputs", "uploads", uploadId);
    await mkdir(folderUploadRoot, { recursive: true });

    for (const part of fileParts) {
      const relPath = (part.filename || "image.png").replace(/\\/g, "/");
      const safeRel = relPath.split("/").map((seg) => safeFilename(seg, "item")).join("/");
      const targetPath = path.join(folderUploadRoot, safeRel);
      await mkdir(path.dirname(targetPath), { recursive: true });
      await writeFile(targetPath, part.data);
    }

    const manifestsDir = path.join(dataRoot, "outputs", "manifests");
    await mkdir(manifestsDir, { recursive: true });

    const payload = (await runPython([
      "--mode",
      "scan-dataset",
      "--folder-path",
      folderUploadRoot,
      "--dataset-name",
      parseScalar(parts, "dataset_name", `Uploaded Folder (${fileParts.length} files)`),
      "--out-dir",
      manifestsDir,
      "--public-prefix",
      "/api/outputs/manifests",
    ])) as Record<string, unknown>;

    res.json(payload);
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Folder upload failed.";
    res.status(500).json({ detail, error_code: "UPLOAD_FAILED" });
  }
}

router.post("/register", registerPair);
router.get("/register/progress/:jobId", streamProgress);
router.get("/register/status/:jobId", getJobStatus);
router.post("/multi-registration/register", registerCollection);
router.get("/multi-registration/progress/:jobId", streamProgress);
router.get("/multi-registration/status/:jobId", getJobStatus);
router.post("/evaluation/synthetic-robustness", syntheticRobustness);
router.post("/ingest-url", ingestUrl);
router.post("/dataset/scan", scanDataset);
router.post("/dataset/upload-folder", uploadFolder);
router.post("/dataset/inspect-file", inspectImageFile);
router.get("/dataset/raw-file", getDatasetRawFile);
router.get("/detectors/capabilities", async (_req: Request, res: Response) => {
  try {
    const payload = (await runPython(["--mode", "list-detectors"])) as Record<string, unknown>;
    res.json(payload);
  } catch (error) {
    res.status(500).json({ detail: error instanceof Error ? error.message : "Failed to query detector capabilities" });
  }
});
router.get("/hardware/status", getHardwareInfo);
router.get("/hardware/self-test", runHardwareSelfTest);
router.post("/hardware/self-test", runHardwareSelfTest);
router.post("/validation/benchmark", runBenchmark);
router.get("/validation/progress/:jobId", streamProgress);
router.get("/validation/status/:jobId", getJobStatus);

export default router;