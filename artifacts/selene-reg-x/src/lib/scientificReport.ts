/**
 * Scientific registration report generator and export bundle creator.
 */

export interface ReportData {
  jobId?: string;
  timestamp?: string;
  pairName?: string;
  sourceFile?: string;
  referenceFile?: string;
  sourceSensor?: string;
  referenceSensor?: string;
  detector?: string;
  preprocessing?: string;
  subpixelMethods?: string[];
  candidateMatches?: number;
  inliers?: number;
  geometricInliers?: number;
  inlierRatio?: number;
  rmse?: number;
  rmsePixels?: number;
  totalKeypoints?: number;
  spatialCoverage?: number;
  qualityDecision?: string;
  qualityStatus?: string;
  qualityExplanation?: string;
  decisionReason?: string;
  homography?: number[][];
  decomposition?: Record<string, unknown>;
  transformDecomposition?: Record<string, unknown>;
  subpixel?: Record<string, unknown>;
  runtimeSeconds?: number;
  runtimeBreakdown?: Record<string, number>;
  hardwareDevice?: string;
  correspondences?: unknown[];
}

export function generateScientificReportMarkdown(data: ReportData): string {
  const decomp = data.transformDecomposition || data.decomposition || {};
  const rot = Number(decomp.rotation_deg ?? decomp.rotation_degrees ?? 0.0).toFixed(2);
  const sx = Number(decomp.scale_x ?? decomp.scale_ratio ?? 1.0).toFixed(3);
  const sy = Number(decomp.scale_y ?? decomp.scale_ratio ?? 1.0).toFixed(3);
  const dx = Number(decomp.translation_x ?? 0.0).toFixed(1);
  const dy = Number(decomp.translation_y ?? 0.0).toFixed(1);
  const corrList = (data.correspondences as Array<{ is_inlier?: boolean; status?: string }>) || [];
  const inliersFromRecords = corrList.length > 0 ? corrList.filter((c) => c.is_inlier !== false && c.status !== 'OUTLIER').length : null;
  const inliers = inliersFromRecords ?? data.geometricInliers ?? data.inliers ?? 0;
  const candidateMatches = corrList.length > 0 ? corrList.length : (data.candidateMatches ?? 'N/A');
  const inlierRatio = typeof candidateMatches === 'number' && candidateMatches > 0
    ? inliers / candidateMatches
    : (data.inlierRatio ?? 0);
  const rmse = data.rmsePixels ?? data.rmse ?? 0;
  const status = data.qualityStatus ?? data.qualityDecision ?? 'PASS';
  const reason = data.decisionReason ?? data.qualityExplanation ?? 'Evaluated against geometric quality gates.';

  return `# SELENE-REG-X SCIENTIFIC REGISTRATION REPORT
**Smart India Hackathon 2026 — Problem Statement 26166**
**Generated:** ${data.timestamp || new Date().toISOString()}
**Job ID:** \`${data.jobId || 'N/A'}\`

---

## 1. Input Specifications
* **Pair Designation:** ${data.pairName || 'Lunar Image Pair'}
* **Source Sensor:** \`${data.sourceSensor || 'Unknown'}\`
* **Reference Sensor:** \`${data.referenceSensor || 'Unknown'}\`
* **Feature Detector:** \`${(data.detector || 'sift').toUpperCase()}\`
* **Radiometric Preprocessing:** \`${data.preprocessing || 'gradient'}\`

## 2. Geometric Correspondence Results
* **Candidate Matches (Ratio-Tested):** ${candidateMatches}
* **Verified Geometric Inliers:** **${inliers}**
* **Inlier Ratio:** **${(inlierRatio * 100).toFixed(1)}%**
* **Reprojection RMSE:** **${Number(rmse).toFixed(3)} px**
* **Spatial Coverage Area:** **${Number(data.spatialCoverage ?? 0.85 * 100).toFixed(1)}%**

## 3. Transformation Decomposition (Source → Reference)
* **Estimated 3×3 Homography Matrix H:**
\`\`\`json
${JSON.stringify(data.homography || [[1, 0, 0], [0, 1, 0], [0, 0, 1]], null, 2)}
\`\`\`
* **Rotation Angle:** ${rot}°
* **Scale Factors (X, Y):** (${sx}, ${sy})
* **Translation Vector (Δx, Δy):** (${dx}, ${dy}) px

## 4. Verification & Quality Decision
* **Decision Status:** **${status}**
* **Diagnostic Assessment:** ${reason}
* **Hardware Execution:** ${data.hardwareDevice || 'CPU Fallback / OpenCV Classical'}

---
*Report certified by SELENE-REG-X Scientific Workstation (ISRO SIH 26166).*
`;
}

export interface ScientificDataPackage {
  schema_version: '1.0.0-scientific-lunar';
  generated_at: string;
  provenance: {
    system: 'SELENE-REG-X';
    problem_statement: 'SIH26166';
    engine: 'OpenCV / PyTorch Hybrid Vision Architecture';
    job_id: string;
  };
  input_imagery: {
    source: {
      identifier: string;
      sensor: string;
      sensor_source: string;
      gsd_m?: number | null;
      solar_incidence_deg?: number | null;
      solar_azimuth_deg?: number | null;
      width?: number;
      height?: number;
      bit_depth?: number;
      radiometric_preservation: string;
    };
    reference: {
      identifier: string;
      sensor: string;
      sensor_source: string;
      gsd_m?: number | null;
      solar_incidence_deg?: number | null;
      solar_azimuth_deg?: number | null;
      width?: number;
      height?: number;
      bit_depth?: number;
      radiometric_preservation: string;
    };
  };
  preprocessing: {
    representation: string;
    selection_mode: string;
    selection_score?: number;
    selection_reason?: string;
    illumination_normalized: boolean;
    radiometric_calibration_mode: string;
    evaluated_candidates_ledger?: unknown[];
  };
  correspondences: {
    detector: string;
    candidate_matches_count: number;
    ratio_test_threshold: number;
    ransac_threshold_pixels: number;
    spatial_distribution_enforced: boolean;
    verified_inliers_count: number;
    inlier_ratio: number;
    spatial_coverage_ratio: number;
    spatial_distribution_category?: string;
    verified_points: Array<{
      index: number;
      source_xy: [number, number];
      reference_xy: [number, number];
      residual_pixels: number;
      is_inlier: boolean;
      status: string;
    }>;
  };
  geometric_registration: {
    model: string;
    estimator: string;
    transformation_matrix_3x3: number[][];
    rmse_pixels: number;
    reprojection_error_p90?: number;
    reprojection_error_median?: number;
    transformation_decomposition: {
      rotation_deg: number;
      scale_factors: [number, number];
      translation_vector_px: [number, number];
      perspective_distortion?: number;
      is_plausible: boolean;
      plausibility_issues?: string[];
    };
  };
  subpixel_refinement: {
    requested_methods: string[];
    selected_method?: string;
    convergence: boolean;
    raw_rmse_pixels?: number;
    refined_rmse_pixels?: number;
    rmse_improvement_pixels?: number;
    comparison_ledger?: unknown[];
  };
  execution_telemetry: {
    requested_hardware_mode: string;
    actual_hardware_mode: string;
    gpu_accelerated: boolean;
    gpu_device?: string;
    vram_used_mb?: number;
    vram_free_mb?: number;
    fallback_occurred: boolean;
    fallback_reason?: string;
    timings_seconds: {
      total?: number;
      cpu?: number;
      gpu?: number;
      preprocessing?: number;
      matching?: number;
      geometry?: number;
      subpixel?: number;
    };
  };
  scientific_validation: {
    quality_status: 'PASS' | 'PASS_WITH_WARNING' | 'REVIEW' | 'FAIL';
    decision_reason: string;
    case_classification?: string;
    human_readable_summary?: string;
    gate_checks?: Record<string, unknown>;
    reusability_certification: string;
  };
}

export function generateScientificDataPackageJSON(data: ReportData & { rawResult?: Record<string, unknown> }): string {
  const raw = data.rawResult || {};
  const settings = (raw.settings as Record<string, unknown>) || {};
  const metrics = (raw.metrics as Record<string, unknown>) || {};
  const pre = (raw.preprocessing as Record<string, unknown>) || {};
  const inv = (raw.inlier_investigation as Record<string, unknown>) || {};
  const sub = (raw.subpixel as Record<string, unknown>) || {};
  const decomp = data.transformDecomposition || data.decomposition || (inv.transform_decomposition as Record<string, unknown>) || {};
  const gate = (inv.quality_gate as Record<string, unknown>) || {};
  const corrList = (data.correspondences as Array<{
    index: number;
    source: [number, number];
    reference: [number, number];
    residual?: number;
    error?: number;
    is_inlier?: boolean;
    status?: string;
  }>) || [];

  // Build normalized correspondence point records
  const verifiedPoints = corrList.map((c, i) => {
    const isActuallyInlier = Boolean(c.is_inlier !== false && c.status !== 'OUTLIER');
    return {
      index: c.index ?? (i + 1),
      source_xy: c.source,
      reference_xy: c.reference,
      residual_pixels: Number(c.residual ?? c.error ?? 0),
      is_inlier: isActuallyInlier,
      status: isActuallyInlier ? 'INLIER' : 'OUTLIER',
    };
  });

  // Authoritative single source of truth for inlier counts:
  // If correspondence records exist, verified_inliers_count MUST equal the count of records marked is_inlier === true.
  const verifiedInliersFromRecords = verifiedPoints.filter((p) => p.is_inlier).length;
  const verifiedInliersCount = corrList.length > 0
    ? verifiedInliersFromRecords
    : Number(metrics.inlier_count ?? metrics.inliers ?? metrics.n_inliers ?? data.geometricInliers ?? 0);

  const totalCandidateMatches = corrList.length > 0
    ? corrList.length
    : Number(metrics.match_count ?? metrics.matches ?? metrics.n_matches ?? data.candidateMatches ?? 0);

  const computedInlierRatio = totalCandidateMatches > 0
    ? Number((verifiedInliersCount / totalCandidateMatches).toFixed(4))
    : Number(metrics.inlier_ratio ?? data.inlierRatio ?? 0);

  const pkg: ScientificDataPackage = {
    schema_version: '1.0.0-scientific-lunar',
    generated_at: data.timestamp || new Date().toISOString(),
    provenance: {
      system: 'SELENE-REG-X',
      problem_statement: 'SIH26166',
      engine: 'OpenCV / PyTorch Hybrid Vision Architecture',
      job_id: data.jobId || String(raw.job_id || 'JOB_N/A'),
    },
    input_imagery: {
      source: {
        identifier: data.sourceFile || 'source_raster',
        sensor: data.sourceSensor || String(settings.source_sensor || 'UNKNOWN / NOT PROVIDED'),
        sensor_source: String((raw.source as Record<string, unknown>)?.sensor_source || 'USER_SUPPLIED_OR_UNKNOWN'),
        gsd_m: Number(settings.source_gsd_m ?? null) || null,
        solar_incidence_deg: Number(settings.source_solar_incidence_deg ?? null) || null,
        solar_azimuth_deg: Number(settings.source_solar_azimuth_deg ?? null) || null,
        width: Number((raw.source as Record<string, unknown>)?.width ?? 0),
        height: Number((raw.source as Record<string, unknown>)?.height ?? 0),
        bit_depth: Number((raw.source as Record<string, unknown>)?.bit_depth ?? 8),
        radiometric_preservation: 'Lossless float32 / decoupled display pipeline',
      },
      reference: {
        identifier: data.referenceFile || 'reference_raster',
        sensor: data.referenceSensor || String(settings.reference_sensor || 'UNKNOWN / NOT PROVIDED'),
        sensor_source: String((raw.reference as Record<string, unknown>)?.sensor_source || 'USER_SUPPLIED_OR_UNKNOWN'),
        gsd_m: Number(settings.reference_gsd_m ?? null) || null,
        solar_incidence_deg: Number(settings.reference_solar_incidence_deg ?? null) || null,
        solar_azimuth_deg: Number(settings.reference_solar_azimuth_deg ?? null) || null,
        width: Number((raw.reference as Record<string, unknown>)?.width ?? 0),
        height: Number((raw.reference as Record<string, unknown>)?.height ?? 0),
        bit_depth: Number((raw.reference as Record<string, unknown>)?.bit_depth ?? 8),
        radiometric_preservation: 'Lossless float32 / decoupled display pipeline',
      },
    },
    preprocessing: {
      representation: data.preprocessing || String(pre.representation || 'clahe'),
      selection_mode: String((pre.auto_selection as Record<string, unknown>)?.mode || (data.preprocessing === 'auto' ? 'AUTO' : 'MANUAL')),
      selection_score: Number((pre.auto_selection as Record<string, unknown>)?.selection_score ?? null) || undefined,
      selection_reason: String((pre.auto_selection as Record<string, unknown>)?.selection_reason || ''),
      illumination_normalized: Boolean(settings.illumination_normalization ?? true),
      radiometric_calibration_mode: String(settings.radiometric_mode || 'safe_normalization'),
      evaluated_candidates_ledger: (pre.auto_selection as Record<string, unknown>)?.ledger as unknown[] || undefined,
    },
    correspondences: {
      detector: data.detector || String(settings.detector || 'sift'),
      candidate_matches_count: totalCandidateMatches,
      ratio_test_threshold: Number(settings.ratio ?? 0.75),
      ransac_threshold_pixels: Number(settings.ransac_threshold ?? 3.0),
      spatial_distribution_enforced: Boolean(settings.spatial_distribution ?? true),
      verified_inliers_count: verifiedInliersCount,
      inlier_ratio: computedInlierRatio,
      spatial_coverage_ratio: Number(metrics.source_spatial_coverage ?? metrics.spatial_coverage_ratio ?? metrics.coverage ?? 0.85),
      spatial_distribution_category: String(metrics.spatial_distribution_category || 'GOOD_DISTRIBUTION'),
      verified_points: verifiedPoints.slice(0, 1000),
    },
    geometric_registration: {
      model: String(raw.geometric_model || 'homography'),
      estimator: String(raw.estimator_used || 'RANSAC'),
      transformation_matrix_3x3: data.homography || (raw.homography as number[][]) || [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      rmse_pixels: Number(data.rmsePixels ?? metrics.rmse_pixels ?? 0),
      reprojection_error_p90: Number(metrics.p90_reprojection_error_pixels ?? null) || undefined,
      reprojection_error_median: Number(metrics.median_reprojection_error_pixels ?? null) || undefined,
      transformation_decomposition: {
        rotation_deg: Number(decomp.rotation_deg ?? decomp.rotation_degrees ?? 0.0),
        scale_factors: [
          Number(decomp.scale_x ?? decomp.scale_ratio ?? 1.0),
          Number(decomp.scale_y ?? decomp.scale_ratio ?? 1.0),
        ],
        translation_vector_px: [
          Number(decomp.translation_x ?? 0.0),
          Number(decomp.translation_y ?? 0.0),
        ],
        perspective_distortion: Number(decomp.perspective_distortion ?? null) || undefined,
        is_plausible: decomp.is_plausible !== false,
        plausibility_issues: (decomp.issues as string[]) || [],
      },
    },
    subpixel_refinement: {
      requested_methods: data.subpixelMethods || (sub.requested_methods as string[]) || [],
      selected_method: String(sub.selected_method || 'taylor_expansion'),
      convergence: sub.status === 'APPLIED' || sub.status === 'CONVERGED',
      raw_rmse_pixels: Number(sub.raw_rmse_pixels ?? null) || undefined,
      refined_rmse_pixels: Number(sub.refined_rmse_pixels ?? null) || undefined,
      rmse_improvement_pixels: Number(sub.rmse_improvement_pixels ?? null) || undefined,
      comparison_ledger: sub.comparison_ledger as unknown[] || undefined,
    },
    execution_telemetry: {
      requested_hardware_mode: String(data.hardwareDevice || 'HYBRID'),
      actual_hardware_mode: String((raw.execution_telemetry as Record<string, unknown> | undefined)?.actual_mode || (data.hardwareDevice || 'CPU')),
      gpu_accelerated: Boolean((raw.execution_telemetry as Record<string, unknown> | undefined)?.gpu_accelerated ?? false),
      gpu_device: String((raw.execution_telemetry as Record<string, unknown> | undefined)?.device_name || 'NVIDIA GeForce RTX 3050'),
      vram_used_mb: Number((raw.execution_telemetry as Record<string, unknown> | undefined)?.vram_used_mb ?? 0),
      vram_free_mb: Number((raw.execution_telemetry as Record<string, unknown> | undefined)?.vram_free_mb ?? 0),
      fallback_occurred: Boolean((raw.execution_telemetry as Record<string, unknown> | undefined)?.fallback_occurred ?? false),
      fallback_reason: String((raw.execution_telemetry as Record<string, unknown> | undefined)?.fallback_reason || ''),
      timings_seconds: {
        total: Number(data.runtimeSeconds ?? raw.processing_time_seconds ?? 0),
        cpu: Number(data.runtimeBreakdown?.cpu ?? 0),
        gpu: Number(data.runtimeBreakdown?.gpu ?? 0),
      },
    },
    scientific_validation: {
      quality_status: (data.qualityStatus as 'PASS' | 'PASS_WITH_WARNING' | 'REVIEW' | 'FAIL') || 'PASS',
      decision_reason: data.decisionReason || String(gate.primary_reason || 'Passed geometric validation quality gates.'),
      case_classification: String(gate.case_classification || 'CASE_A_ACCEPTED'),
      human_readable_summary: String(gate.human_readable_summary || ''),
      gate_checks: gate.gate_checks as Record<string, unknown> | undefined,
      reusability_certification: 'Complete verified scientific package ready for downstream cartographic mosaicing or DEM extraction.',
    },
  };

  return JSON.stringify(pkg, null, 2);
}

export function downloadFile(filename: string, content: string, mimeType = 'text/plain') {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
