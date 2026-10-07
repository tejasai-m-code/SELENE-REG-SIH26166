/*
 * SELENE-REG-X — Match Point Inspector (Phase 8)
 *
 * Coordinate pipeline (one canonical chain):
 *   Backend pixel coord (source/reference image space)
 *     → transmitted as-is in inlier_points JSON
 *     → drawn on canvas via camera transform: screen = coord * camera.scale + camera.offset
 *     → inverted on hover: coord = (screen - camera.offset) / camera.scale
 *
 * All points, residual vectors, and images share the SAME camera transform.
 * No separate ad-hoc transforms exist.
 *
 * Residual vector semantics:
 *   projected_point (source transformed by H) → reference_point
 *   This is the geometric reprojection error vector, NOT the source↔reference correspondence.
 *
 * Phase 7 note:
 *   Phase 7 subpixel refinement is PAIR-LEVEL, not per-match.
 *   Individual match points do not have independent subpixel estimates.
 *   The pair-level summary is shown in the Transform Inspector panel.
 */

const Inspector = {
  data: null,
  images: { ref: new Image(), reg: new Image() },
  camera: { x: 0, y: 0, scale: 1 },
  isDragging: false,
  dragStart: { x: 0, y: 0 },
  dragMoved: false,
  hoveredMatch: null,
  selectedMatch: null,

  settings: {
    showRef: true,
    showReg: true,
    opacityReg: 0.5,
    filter: 'all',   // 'all' | 'inliers' | 'outliers' | 'high_confidence' | 'low_confidence'
    showResiduals: true,
  },

  // ---------- init ----------
  init() {
    this.canvas = document.getElementById('inspectorCanvas');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
    this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
    this.canvas.addEventListener('mouseup',   this.onMouseUp.bind(this));
    this.canvas.addEventListener('mouseleave', this.onMouseLeave.bind(this));
    this.canvas.addEventListener('wheel', this.onWheel.bind(this), { passive: false });

    document.getElementById('inspShowRef').addEventListener('change', e => {
      this.settings.showRef = e.target.checked; this.draw();
    });
    document.getElementById('inspShowReg').addEventListener('change', e => {
      this.settings.showReg = e.target.checked; this.draw();
    });
    document.getElementById('inspOpacityReg').addEventListener('input', e => {
      this.settings.opacityReg = parseFloat(e.target.value); this.draw();
    });
    document.getElementById('inspFilter').addEventListener('change', e => {
      this.settings.filter = e.target.value; this.draw();
    });
    document.getElementById('inspShowResiduals').addEventListener('change', e => {
      this.settings.showResiduals = e.target.checked; this.draw();
    });
    document.getElementById('inspReset').addEventListener('click', () => this.fitView());

    new ResizeObserver(() => { if (this.data) this.resizeCanvas(); })
      .observe(this.canvas.parentElement);
  },

  // ---------- load ----------
  loadData(data, refUrl, regUrl) {
    // Explicitly clear all stale state before loading new registration
    this.data = null;
    this.selectedMatch = null;
    this.hoveredMatch = null;

    // Reset panels to loading state
    const ptEl = document.getElementById('inspPointInfo');
    const txEl = document.getElementById('inspTransformInfo');
    if (ptEl) ptEl.innerHTML = '<p style="color:#8892b0;font-size:12px">Loading registration…</p>';
    if (txEl) txEl.innerHTML = '<p style="color:#8892b0;font-size:12px">Loading…</p>';

    this.images.ref = new Image();
    this.images.reg = new Image();

    let loaded = 0;
    const onLoad = () => {
      loaded++;
      if (loaded === 2) {
        this.data = data;
        this.resizeCanvas();
        this.fitView();
        this.updatePanels();
      }
    };
    const onErr = (label) => () => {
      console.warn('Inspector image failed to load:', label);
      loaded++;
      if (loaded === 2 && this.data === null) {
        this.data = data;
        this.resizeCanvas();
        this.fitView();
        this.updatePanels();
      }
    };

    this.images.ref.onload = onLoad;
    this.images.reg.onload = onLoad;
    this.images.ref.onerror = onErr('reference');
    this.images.reg.onerror = onErr('registered');
    this.images.ref.src = refUrl;
    this.images.reg.src = regUrl;
  },

  // ---------- canvas sizing ----------
  resizeCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width  = Math.max(100, rect.width);
    this.canvas.height = Math.max(100, rect.height);
    this.draw();
  },

  fitView() {
    if (!this.data) return;
    const imgW = this.images.ref.naturalWidth  || (this.data.reference && this.data.reference.width)  || 800;
    const imgH = this.images.ref.naturalHeight || (this.data.reference && this.data.reference.height) || 600;
    const scaleX = this.canvas.width  / imgW;
    const scaleY = this.canvas.height / imgH;
    this.camera.scale = Math.min(scaleX, scaleY) * 0.92;
    this.camera.x = (this.canvas.width  - imgW * this.camera.scale) / 2;
    this.camera.y = (this.canvas.height - imgH * this.camera.scale) / 2;
    this.draw();
  },

  // ---------- point filtering ----------
  pointVisible(pt) {
    const f = this.settings.filter;
    if (f === 'inliers')  return pt.inlier === true;
    if (f === 'outliers') return pt.inlier === false;
    if (f === 'high_confidence') {
      // Use pair-level confidence from subpixel_refinement — the only genuine confidence value
      const pairConf = this.data && this.data.subpixel_refinement
        ? this.data.subpixel_refinement.confidence : null;
      // Show if this point is an inlier AND the pair registered with HIGH confidence
      return pt.inlier && pairConf === 'HIGH';
    }
    if (f === 'low_confidence') {
      const pairConf = this.data && this.data.subpixel_refinement
        ? this.data.subpixel_refinement.confidence : null;
      return pt.inlier && (pairConf === 'LOW' || pairConf === 'MEDIUM' || pairConf === 'UNSTABLE');
    }
    return true; // 'all'
  },

  // ---------- mouse events ----------
  onMouseDown(e) {
    this.isDragging = true;
    this.dragMoved  = false;
    this.dragStart  = { x: e.clientX - this.camera.x, y: e.clientY - this.camera.y };
  },

  onMouseMove(e) {
    if (this.isDragging) {
      const newX = e.clientX - this.dragStart.x;
      const newY = e.clientY - this.dragStart.y;
      if (Math.abs(newX - this.camera.x) > 1 || Math.abs(newY - this.camera.y) > 1) {
        this.dragMoved = true;
      }
      this.camera.x = newX;
      this.camera.y = newY;
      this.draw();
      return;
    }
    if (!this.data) return;
    const rect = this.canvas.getBoundingClientRect();
    const wx = (e.clientX - rect.left  - this.camera.x) / this.camera.scale;
    const wy = (e.clientY - rect.top   - this.camera.y) / this.camera.scale;
    const hitRadius = 8 / this.camera.scale;

    let closest = null, minD = hitRadius;
    for (const pt of (this.data.inlier_points || [])) {
      if (!this.pointVisible(pt)) continue;
      const dx = pt.reference_x - wx, dy = pt.reference_y - wy;
      const d  = Math.sqrt(dx*dx + dy*dy);
      if (d < minD) { minD = d; closest = pt; }
    }
    if (this.hoveredMatch !== closest) {
      this.hoveredMatch = closest;
      this.canvas.style.cursor = closest ? 'pointer' : 'default';
      this.draw();
    }
  },

  onMouseUp(e) {
    const wasDragging = this.isDragging;
    this.isDragging = false;
    if (wasDragging && !this.dragMoved) {
      // Treat as click — select hovered point
      this.selectedMatch = this.hoveredMatch;
      this.updatePointPanel();
      this.draw();
    }
  },

  onMouseLeave() {
    this.isDragging   = false;
    this.hoveredMatch = null;
    this.canvas.style.cursor = 'default';
    this.draw();
  },

  onWheel(e) {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : (1 / 1.12);
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    // Zoom around mouse position
    const wx = (mx - this.camera.x) / this.camera.scale;
    const wy = (my - this.camera.y) / this.camera.scale;
    this.camera.scale = Math.max(0.05, Math.min(32, this.camera.scale * factor));
    this.camera.x = mx - wx * this.camera.scale;
    this.camera.y = my - wy * this.camera.scale;
    this.draw();
  },

  // ---------- draw ----------
  draw() {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.data) {
      // Show no-data message
      this.ctx.fillStyle = '#8892b0';
      this.ctx.font = '13px monospace';
      this.ctx.textAlign = 'center';
      this.ctx.fillText('No registration correspondences available.', this.canvas.width / 2, this.canvas.height / 2);
      this.ctx.textAlign = 'left';
      return;
    }

    const pts = this.data.inlier_points || [];

    this.ctx.save();
    this.ctx.translate(this.camera.x, this.camera.y);
    this.ctx.scale(this.camera.scale, this.camera.scale);

    // 1. Reference image
    if (this.settings.showRef && this.images.ref.naturalWidth) {
      this.ctx.globalAlpha = 1.0;
      this.ctx.drawImage(this.images.ref, 0, 0);
    }
    // 2. Registered (transformed source) image at configurable opacity
    if (this.settings.showReg && this.images.reg.naturalWidth) {
      this.ctx.globalAlpha = this.settings.opacityReg;
      this.ctx.drawImage(this.images.reg, 0, 0);
    }
    this.ctx.globalAlpha = 1.0;

    // 3. Residual vectors: projected_point → reference_point
    //    These show the geometric reprojection error direction and magnitude.
    if (this.settings.showResiduals) {
      this.ctx.lineWidth = Math.max(0.3, 1 / this.camera.scale);
      for (const pt of pts) {
        if (!this.pointVisible(pt)) continue;
        if (pt.projected_x == null || pt.projected_y == null) continue;
        this.ctx.strokeStyle = pt.inlier ? '#7cf2bd' : '#f27c7c';
        this.ctx.globalAlpha = 0.55;
        this.ctx.beginPath();
        this.ctx.moveTo(pt.projected_x, pt.projected_y);  // projected (source→ref space)
        this.ctx.lineTo(pt.reference_x, pt.reference_y);  // actual reference keypoint
        this.ctx.stroke();
      }
      this.ctx.globalAlpha = 1.0;
    }

    // 4. Points (reference-space keypoints)
    const r0 = 3 / this.camera.scale;
    for (const pt of pts) {
      if (!this.pointVisible(pt)) continue;

      if (pt === this.selectedMatch) {
        // Selected: yellow ring + larger dot
        this.ctx.strokeStyle = '#ffeb3b';
        this.ctx.lineWidth = 1.5 / this.camera.scale;
        this.ctx.beginPath();
        this.ctx.arc(pt.reference_x, pt.reference_y, r0 * 2.5, 0, Math.PI * 2);
        this.ctx.stroke();
        this.ctx.fillStyle = '#ffeb3b';
      } else if (pt === this.hoveredMatch) {
        this.ctx.fillStyle = '#ffffff';
      } else {
        this.ctx.fillStyle = pt.inlier ? '#7cf2bd' : '#f27c7c';
      }
      this.ctx.beginPath();
      this.ctx.arc(pt.reference_x, pt.reference_y, r0, 0, Math.PI * 2);
      this.ctx.fill();
    }

    this.ctx.restore();

    // 5. No-points overlay
    if (pts.length === 0) {
      this.ctx.fillStyle = '#8892b0';
      this.ctx.font = '13px monospace';
      this.ctx.textAlign = 'center';
      this.ctx.fillText('No registration correspondences available.', this.canvas.width / 2, this.canvas.height / 2);
      this.ctx.textAlign = 'left';
    }
  },

  // ---------- panels ----------
  updatePanels() {
    this.updateTransformPanel();
    this.updatePointPanel();
  },

  _fmtN(v, digits=3) {
    if (v === null || v === undefined || isNaN(Number(v))) return '—';
    return Number(v).toFixed(digits);
  },

  updateTransformPanel() {
    const el = document.getElementById('inspTransformInfo');
    if (!el) return;
    if (!this.data) { el.innerHTML = '<p style="color:#8892b0">No data</p>'; return; }

    const cm  = this.data.cross_modal || {};
    const met = this.data.metrics     || {};
    const sub = this.data.subpixel_refinement;

    // Build Phase 7 block from actual subpixel_refinement key (PAIR-LEVEL)
    let subHtml = '<p style="color:#8892b0;font-size:11px">Phase 7 not run or disabled.</p>';
    if (sub) {
      const accepted = sub.accepted;
      const reason   = sub.acceptance_reason || sub.rejection_reason || '—';
      const diags    = (sub.diagnostics || []).map(d => {
        const dx = this._fmtN(d.dx); const dy = this._fmtN(d.dy);
        return `<div class="inspector-kv" style="font-size:11px">
          <span style="color:#6ee7f9">${d.method}</span>
          <b>${d.status} &nbsp;dx=${dx} dy=${dy}</b></div>`;
      }).join('');

      subHtml = `
        <div class="inspector-kv"><span>Consensus dx / dy</span>
          <b>${this._fmtN(sub.dx)} / ${this._fmtN(sub.dy)} px</b></div>
        <div class="inspector-kv"><span>Confidence</span><b>${sub.confidence || '—'}</b></div>
        <div class="inspector-kv"><span>Decision</span>
          <b style="color:${accepted ? '#7cf2bd' : '#f27c7c'}">${accepted ? 'ACCEPTED' : 'REJECTED'}</b></div>
        <div class="inspector-kv"><span>Reason</span><b>${reason}</b></div>
        <div style="margin-top:6px;border-top:1px solid #2a3a50;padding-top:6px">${diags}</div>`;
    }

    // Final transform matrix
    const H = this.data.homography || [];
    const hStr = H.length ? H.map(row => row.map(v => Number(v).toFixed(7)).join('  ')).join('\n') : '—';

    el.innerHTML = `
      <h4>PHASE 6 — GEOMETRIC MODEL</h4>
      <div class="inspector-kv"><span>Selected Model</span>
        <b>${cm.selected_geometric_model || '—'}</b></div>
      <div class="inspector-kv"><span>Estimator</span>
        <b>${cm.geometric_estimator || '—'}</b></div>
      <div class="inspector-kv"><span>Match Count</span>
        <b>${met.match_count != null ? met.match_count : '—'}</b></div>
      <div class="inspector-kv"><span>Inlier Count</span>
        <b>${met.inlier_count != null ? met.inlier_count : '—'}</b></div>
      <div class="inspector-kv"><span>Inlier Ratio</span>
        <b>${met.inlier_ratio != null ? (met.inlier_ratio*100).toFixed(1)+'%' : '—'}</b></div>
      <div class="inspector-kv"><span>RMSE</span>
        <b>${this._fmtN(met.rmse_pixels)} px</b></div>
      <div class="inspector-kv"><span>Median Reprojection</span>
        <b>${this._fmtN(met.median_reprojection_error_pixels)} px</b></div>
      <div class="inspector-kv"><span>Spatial Coverage</span>
        <b>${met.source_spatial_coverage != null ? (met.source_spatial_coverage*100).toFixed(0)+'%' : '—'}</b></div>

      <h4 style="margin-top:12px">PHASE 7 — PAIR-LEVEL SUBPIXEL REFINEMENT</h4>
      <p style="color:#8892b0;font-size:10px;margin:0 0 6px">
        Phase 7 operates on the aligned pair as a whole — not on individual feature points.</p>
      ${subHtml}

      <h4 style="margin-top:12px">FINAL TRANSFORM MATRIX</h4>
      <p style="color:#8892b0;font-size:10px;margin:0 0 4px">
        ${sub && sub.accepted ? 'Phase 6 + Phase 7 subpixel correction applied' : 'Phase 6 transform (Phase 7 ' + (sub ? 'rejected' : 'not run') + ')'}</p>
      <pre style="font-size:9px;line-height:1.6;overflow-x:auto;background:#0a1320;padding:8px;border-radius:4px">${hStr}</pre>
    `;
  },

  updatePointPanel() {
    const el = document.getElementById('inspPointInfo');
    if (!el) return;
    const pt = this.selectedMatch;
    if (!pt) {
      el.innerHTML = '<p style="color:#8892b0;font-size:12px">Click a correspondence point in the overlay to inspect it.</p>';
      return;
    }

    const f3 = v => this._fmtN(v, 3);
    const f2 = v => this._fmtN(v, 2);
    const projStr = (pt.projected_x != null && pt.projected_y != null)
      ? `(${f2(pt.projected_x)}, ${f2(pt.projected_y)})` : '—';

    el.innerHTML = `
      <h4>MATCH POINT ${pt.match_id}</h4>

      <div class="inspector-kv">
        <span>Status</span>
        <b style="color:${pt.inlier ? '#7cf2bd' : '#f27c7c'}">${pt.inlier ? 'INLIER' : 'OUTLIER'}</b>
      </div>
      <div class="inspector-kv"><span>Source Coord</span>
        <b>(${f2(pt.source_x)}, ${f2(pt.source_y)})</b></div>
      <div class="inspector-kv"><span>Reference Coord</span>
        <b>(${f2(pt.reference_x)}, ${f2(pt.reference_y)})</b></div>
      <div class="inspector-kv"><span>Projected Coord</span>
        <b title="Source point projected into reference space via H">${projStr}</b></div>
      <div class="inspector-kv"><span>Residual ‖ref−proj‖</span>
        <b>${pt.residual != null ? f3(pt.residual) + ' px' : '—'}</b></div>
      <div class="inspector-kv"><span>Lowe Ratio</span>
        <b>${pt.ratio_test_value != null ? f3(pt.ratio_test_value) : 'N/A (no ratio test)'}</b></div>
      <div class="inspector-kv"><span>Detector</span>
        <b>${pt.detector || '—'}</b></div>
      <div class="inspector-kv"><span>Geometric Model</span>
        <b>${pt.geometric_model || '—'}</b></div>
      <div class="inspector-kv"><span>Estimator</span>
        <b>${pt.estimator || '—'}</b></div>
      <div class="inspector-kv"><span>Phase 6 Status</span>
        <b>${pt.phase6_status || '—'}</b></div>
      <div class="inspector-kv"><span>Spatial Cell</span>
        <b>Row ${pt.spatial_cell.row}, Col ${pt.spatial_cell.col}
           (grid ${pt.spatial_cell.configuration})</b></div>

      <div style="margin-top:8px;padding-top:8px;border-top:1px solid #2a3a50;
                  font-size:10px;color:#8892b0">
        Phase 7 subpixel information is PAIR-LEVEL — see Transform Inspector above.
        No per-match subpixel estimate is computed.
      </div>
    `;
  },
};

window.Inspector = Inspector;
