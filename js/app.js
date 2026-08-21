import { loadReferenceBands, runLiveLoop, analyzeVideoFull, compareConditions } from './pose.js';

const DEMO_METRICS = ['stride_sec', 'cadence_per_min', 'peak_knee_flexion_deg', 'trunk_lean_mean_deg', 'elbow_rom_deg'];
const METRIC_LABEL = {
  stride_sec: 'Stride duration (s)',
  cadence_per_min: 'Cadence (steps/min)',
  peak_knee_flexion_deg: 'Peak knee flexion (deg)',
  trunk_lean_mean_deg: 'Trunk lean (deg)',
  elbow_rom_deg: 'Elbow ROM (deg)',
};

const state = { bands: null, results: {} };
const isWebcamRunning = { value: false };

// MediaPipe's landmarker is a single shared instance (see pose.js) that
// can't handle two concurrent detectForVideo streams -- their timestamps
// interleave and both analyses silently produce garbage. Every call to
// analyzeVideoFull must be serialized through this queue, even if the user
// clicks two "Run analysis" buttons close together.
let analysisQueue = Promise.resolve();
function queueAnalysis(fn) {
  const run = analysisQueue.then(fn);
  analysisQueue = run.catch(() => {}); // don't let one failure block the queue
  return run;
}

async function boot() {
  state.bands = await loadReferenceBands();
  wireWebcam();
  wireVideoUpload();
  wireSampleAnalysis();
}

// ---------- Webcam live demo ----------

function wireWebcam() {
  const startBtn = document.getElementById('webcam-start');
  const stopBtn = document.getElementById('webcam-stop');
  const video = document.getElementById('webcam-video');
  const canvas = document.getElementById('webcam-canvas');
  const angleOut = document.getElementById('webcam-angles');

  startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      await video.play();
      isWebcamRunning.value = true;
      stopBtn.disabled = false;
      runLiveLoop(video, canvas, (angles) => {
        if (!angles) { angleOut.textContent = 'No person detected'; return; }
        angleOut.innerHTML = `
          <div><span>Knee flexion</span><strong>${fmt(angles.knee)}&deg;</strong></div>
          <div><span>Trunk lean</span><strong>${fmt(angles.trunk)}&deg;</strong></div>
          <div><span>Elbow angle</span><strong>${fmt(angles.elbow)}&deg;</strong></div>
        `;
      }, isWebcamRunning);
    } catch (e) {
      angleOut.innerHTML = `<p class="warn">Couldn't access the webcam: ${e.message}. Check browser camera permissions.</p>`;
      startBtn.disabled = false;
    }
  });

  stopBtn.addEventListener('click', () => {
    isWebcamRunning.value = false;
    const stream = video.srcObject;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
  });
}

function fmt(v) {
  return v == null || Number.isNaN(v) ? '--' : v.toFixed(1);
}

// ---------- Upload-your-own video ----------

function wireVideoUpload() {
  const input = document.getElementById('video-upload');
  const video = document.getElementById('upload-video');
  const canvas = document.getElementById('upload-canvas');
  const runBtn = document.getElementById('upload-run');
  const out = document.getElementById('upload-result');
  const progress = document.getElementById('upload-progress');

  input.addEventListener('change', () => {
    const file = input.files[0];
    if (!file) return;
    video.src = URL.createObjectURL(file);
    runBtn.disabled = false;
    out.innerHTML = '';
  });

  runBtn.addEventListener('click', async () => {
    runBtn.disabled = true;
    progress.style.display = 'block';
    try {
      const result = await queueAnalysis(() => analyzeVideoFull(video, canvas, (p) => {
        progress.querySelector('.bar').style.width = `${Math.round(p * 100)}%`;
      }));
      out.innerHTML = renderStrideSummary(result);
    } catch (e) {
      out.innerHTML = `<p class="warn">Analysis failed: ${e.message}</p>`;
    } finally {
      progress.style.display = 'none';
      runBtn.disabled = false;
    }
  });
}

function renderStrideSummary(result) {
  if (!result.strides.length) {
    return `<p class="warn">No confident strides detected -- try a clearer, fully-visible sagittal walking clip.</p>`;
  }
  const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const strideSec = mean(result.strides.map((s) => s.strideSec));
  const knee = mean(result.strides.map((s) => s.peakKneeFlexionDeg));
  const elbow = mean(result.strides.map((s) => s.elbowRomDeg));
  return `
    <div class="metric-grid">
      <div><span>Strides detected</span><strong>${result.strides.length}</strong></div>
      <div><span>Foot visibility</span><strong>${(result.footVisibility * 100).toFixed(0)}%</strong></div>
      <div><span>Avg stride duration</span><strong>${strideSec.toFixed(2)}s</strong></div>
      <div><span>Avg peak knee flexion</span><strong>${knee.toFixed(1)}&deg;</strong></div>
      <div><span>Avg elbow ROM</span><strong>${elbow.toFixed(1)}&deg;</strong></div>
    </div>
  `;
}

// ---------- Sample video comparison (Crutch A vs Crutch B demo) ----------

const uploadedSlot = { a: false, b: false };

function updateCompareSourceNote() {
  const note = document.getElementById('compare-source-note');
  if (uploadedSlot.a && uploadedSlot.b) {
    note.textContent = 'Both slots are your uploaded videos -- this is a real comparison.';
  } else if (uploadedSlot.a || uploadedSlot.b) {
    note.textContent = 'One slot is still the placeholder sample -- upload a video into the other slot for a real comparison.';
  } else {
    note.textContent = 'Currently loaded: the same validated sample clip in both slots (a mechanism demo, not a real comparison) -- upload your own two videos above for an actual verdict.';
  }
}

function wireSampleAnalysis() {
  for (const key of ['a', 'b']) {
    const video = document.getElementById(`sample-${key}-video`);
    const canvas = document.getElementById(`sample-${key}-canvas`);
    const btn = document.getElementById(`sample-${key}-run`);
    const out = document.getElementById(`sample-${key}-result`);
    const progress = document.getElementById(`sample-${key}-progress`);
    const upload = document.getElementById(`sample-${key}-upload`);

    upload.addEventListener('change', () => {
      const file = upload.files[0];
      if (!file) return;
      video.src = URL.createObjectURL(file);
      video.load();
      uploadedSlot[key] = true;
      updateCompareSourceNote();
      out.innerHTML = '';
      state.results[key] = null;
      document.getElementById('comparison-section').style.display = 'none';
    });

    btn.addEventListener('click', async () => {
      btn.disabled = true;
      progress.style.display = 'block';
      try {
        const result = await queueAnalysis(() => analyzeVideoFull(video, canvas, (p) => {
          progress.querySelector('.bar').style.width = `${Math.round(p * 100)}%`;
        }));
        state.results[key] = result;
        out.innerHTML = renderStrideSummary(result);
        btn.textContent = 'Re-run analysis';
        maybeShowComparison();
      } catch (e) {
        out.innerHTML = `<p class="warn">Analysis failed: ${e.message}</p>`;
      } finally {
        progress.style.display = 'none';
        btn.disabled = false;
      }
    });
  }
}

function maybeShowComparison() {
  if (!state.results.a || !state.results.b) return;
  const section = document.getElementById('comparison-section');
  section.style.display = 'block';

  // compareConditions expects the strides in their native camelCase field
  // names (it translates DEMO_METRICS' snake_case names internally via
  // METRIC_KEY_MAP) -- no conversion needed here.
  const { rows, overallA, overallB, verdict, usedMetrics } = compareConditions(
    state.results.a.strides, state.results.b.strides, state.bands, DEMO_METRICS, 'Crutch A', 'Crutch B'
  );

  const tableRows = rows.map((r) => `
    <tr class="${r.wellSeparated ? '' : 'excluded'}">
      <td>${METRIC_LABEL[r.metric]}</td>
      <td>${fmt(r.rawMeanA)}</td>
      <td>${fmt(r.rawMeanB)}</td>
      <td>${fmt(r.scoreA)}</td>
      <td>${fmt(r.scoreB)}</td>
      <td>${r.wellSeparated ? 'yes' : 'excluded (low separation)'}</td>
    </tr>
  `).join('');

  document.getElementById('comparison-table-body').innerHTML = tableRows;
  document.getElementById('comparison-verdict').innerHTML = `
    <div class="verdict-box">
      <div class="verdict-scores">
        <div><span>Crutch A overall score</span><strong>${fmt(overallA)}</strong></div>
        <div><span>Crutch B overall score</span><strong>${fmt(overallB)}</strong></div>
      </div>
      <p>${verdict}</p>
      <p class="note">Composite computed from: ${usedMetrics ? usedMetrics.map((m) => METRIC_LABEL[m]).join(', ') : 'none'}. Metrics with near-identical healthy/impaired medians are excluded from the composite (unstable score, not a real signal) but still shown individually above.</p>
    </div>
  `;
}

boot();
