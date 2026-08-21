// Port of pipeline/gait_events.py, pipeline/metrics.py, pipeline/scoring.py to JS.
// Runs entirely client-side against MediaPipe's live landmark output --
// no server, no Python -- this IS the inference step for the website.

const LANDMARK = {
  left_shoulder: 11, right_shoulder: 12,
  left_elbow: 13, right_elbow: 14,
  left_wrist: 15, right_wrist: 16,
  left_hip: 23, right_hip: 24,
  left_knee: 25, right_knee: 26,
  left_ankle: 27, right_ankle: 28,
  left_heel: 29, right_heel: 30,
  left_foot_index: 31, right_foot_index: 32,
};

const FOOT_VISIBILITY_INDICES = (side) => [
  side === 'left' ? LANDMARK.left_heel : LANDMARK.right_heel,
  side === 'left' ? LANDMARK.left_foot_index : LANDMARK.right_foot_index,
  side === 'left' ? LANDMARK.left_ankle : LANDMARK.right_ankle,
];

const STANCE_RATIO_MIN_FOOT_VISIBILITY = 0.85;

// ---------- signal helpers ----------

function interpolateNaN(arr) {
  const out = arr.slice();
  const n = out.length;
  let lastValid = null;
  // forward fill gaps via linear interpolation
  for (let i = 0; i < n; i++) {
    if (out[i] != null && !Number.isNaN(out[i])) {
      if (lastValid !== null && lastValid < i - 1) {
        const v0 = out[lastValid], v1 = out[i];
        for (let j = lastValid + 1; j < i; j++) {
          const t = (j - lastValid) / (i - lastValid);
          out[j] = v0 + t * (v1 - v0);
        }
      }
      lastValid = i;
    }
  }
  // boundary fill (limit_direction='both' equivalent)
  let firstValid = out.findIndex((v) => v != null && !Number.isNaN(v));
  if (firstValid === -1) return out.fill(0);
  for (let j = 0; j < firstValid; j++) out[j] = out[firstValid];
  let lastValidIdx = n - 1;
  while (lastValidIdx >= 0 && (out[lastValidIdx] == null || Number.isNaN(out[lastValidIdx]))) lastValidIdx--;
  for (let j = lastValidIdx + 1; j < n; j++) out[j] = out[lastValidIdx];
  return out;
}

// Lightweight smoothing stand-in for scipy's Savitzky-Golay: symmetric moving
// average. Not identical math, but same purpose (denoise per-frame jitter
// before peak-finding / angle computation) and fine for this precision level.
function smooth(arr, window = 9) {
  const filled = interpolateNaN(arr);
  const n = filled.length;
  const half = Math.floor(window / 2);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0, count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(n - 1, i + half); j++) {
      sum += filled[j];
      count++;
    }
    out[i] = sum / count;
  }
  return out;
}

function gradient(arr, fps) {
  const n = arr.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    if (i === 0) out[i] = (arr[1] - arr[0]) * fps;
    else if (i === n - 1) out[i] = (arr[n - 1] - arr[n - 2]) * fps;
    else out[i] = ((arr[i + 1] - arr[i - 1]) / 2) * fps;
  }
  return out;
}

// Local maxima with a minimum spacing constraint -- same intent as
// scipy.signal.find_peaks(distance=..., prominence=...).
// A point is a candidate peak if it's the max within its own +/-minDistance
// neighborhood (this enforces spacing directly, avoiding a separate
// prominence-walk that's easy to get subtly wrong) and is at least
// minProminence above that neighborhood's minimum.
function findPeaks(arr, minDistance, minProminence = 0.01) {
  const n = arr.length;
  const candidates = [];
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - minDistance);
    const hi = Math.min(n - 1, i + minDistance);
    let isMax = true;
    let localMin = arr[i];
    for (let j = lo; j <= hi; j++) {
      if (arr[j] > arr[i]) isMax = false;
      if (arr[j] < localMin) localMin = arr[j];
    }
    if (isMax && arr[i] - localMin >= minProminence) candidates.push(i);
  }
  // collapse adjacent too-close candidates (flat-top plateaus), keeping the taller
  const kept = [];
  for (const c of candidates) {
    if (kept.length === 0 || c - kept[kept.length - 1] >= minDistance) {
      kept.push(c);
    } else if (arr[c] > arr[kept[kept.length - 1]]) {
      kept[kept.length - 1] = c;
    }
  }
  return kept;
}

// ---------- geometry ----------

function angleDeg(ax, ay, bx, by, cx, cy) {
  const abx = ax - bx, aby = ay - by;
  const cbx = cx - bx, cby = cy - by;
  const dot = abx * cbx + aby * cby;
  const norm = Math.hypot(abx, aby) * Math.hypot(cbx, cby);
  if (norm === 0) return NaN;
  const cos = Math.max(-1, Math.min(1, dot / norm));
  return (Math.acos(cos) * 180) / Math.PI;
}

// frames: array of {landmarks: [{x,y,visibility}, ...33]}
function seriesFor(frames, idx, axis) {
  return frames.map((f) => (f.landmarks && f.landmarks[idx] ? f.landmarks[idx][axis] : NaN));
}

function kneeFlexionSeries(frames, side) {
  const hipX = seriesFor(frames, LANDMARK[`${side}_hip`], 'x');
  const hipY = seriesFor(frames, LANDMARK[`${side}_hip`], 'y');
  const kneeX = seriesFor(frames, LANDMARK[`${side}_knee`], 'x');
  const kneeY = seriesFor(frames, LANDMARK[`${side}_knee`], 'y');
  const ankleX = seriesFor(frames, LANDMARK[`${side}_ankle`], 'x');
  const ankleY = seriesFor(frames, LANDMARK[`${side}_ankle`], 'y');
  return frames.map((_, i) => angleDeg(hipX[i], hipY[i], kneeX[i], kneeY[i], ankleX[i], ankleY[i]));
}

function trunkLeanSeries(frames, side) {
  const shX = seriesFor(frames, LANDMARK[`${side}_shoulder`], 'x');
  const shY = seriesFor(frames, LANDMARK[`${side}_shoulder`], 'y');
  const hipX = seriesFor(frames, LANDMARK[`${side}_hip`], 'x');
  const hipY = seriesFor(frames, LANDMARK[`${side}_hip`], 'y');
  return frames.map((_, i) => {
    const dx = Math.abs(hipX[i] - shX[i]);
    const dy = Math.abs(hipY[i] - shY[i]);
    return (Math.atan2(dx, dy) * 180) / Math.PI;
  });
}

function elbowAngleSeries(frames, side) {
  const shX = seriesFor(frames, LANDMARK[`${side}_shoulder`], 'x');
  const shY = seriesFor(frames, LANDMARK[`${side}_shoulder`], 'y');
  const elX = seriesFor(frames, LANDMARK[`${side}_elbow`], 'x');
  const elY = seriesFor(frames, LANDMARK[`${side}_elbow`], 'y');
  const wrX = seriesFor(frames, LANDMARK[`${side}_wrist`], 'x');
  const wrY = seriesFor(frames, LANDMARK[`${side}_wrist`], 'y');
  return frames.map((_, i) => angleDeg(shX[i], shY[i], elX[i], elY[i], wrX[i], wrY[i]));
}

function footVisibility(frames, side) {
  const idxs = FOOT_VISIBILITY_INDICES(side);
  let sum = 0, count = 0;
  for (const f of frames) {
    if (!f.landmarks) continue;
    for (const idx of idxs) {
      const v = f.landmarks[idx] && f.landmarks[idx].visibility;
      if (v != null) { sum += v; count++; }
    }
  }
  return count ? sum / count : 0;
}

// ---------- gait events ----------

function detectGaitEvents(frames, side, fps, opts = {}) {
  const minStrideSec = opts.minStrideSec ?? 0.5;
  const minStanceSec = opts.minStanceSec ?? 0.15;
  const persistFramesSec = opts.persistFramesSec ?? 0.05;
  const plausibleRange = opts.plausibleStrideSec ?? [0.6, 2.0];

  const heelY = smooth(seriesFor(frames, LANDMARK[`${side}_heel`], 'y'));
  const toeY = smooth(seriesFor(frames, LANDMARK[`${side}_foot_index`], 'y'));
  const toeVelocity = gradient(toeY, fps);

  const minDistance = Math.round(minStrideSec * fps);
  const heelStrikes = findPeaks(heelY, minDistance, 0.01);

  const mean = toeVelocity.reduce((a, b) => a + b, 0) / toeVelocity.length;
  const variance = toeVelocity.reduce((a, b) => a + (b - mean) ** 2, 0) / toeVelocity.length;
  const velStd = Math.sqrt(variance);
  const threshold = -0.5 * velStd;
  const minStanceFrames = Math.round(minStanceSec * fps);
  const persistFrames = Math.max(1, Math.round(persistFramesSec * fps));

  const toeOffs = [];
  const plausible = [];
  for (let i = 0; i < heelStrikes.length - 1; i++) {
    const start = heelStrikes[i], end = heelStrikes[i + 1];
    const strideSec = (end - start) / fps;
    plausible.push(strideSec >= plausibleRange[0] && strideSec <= plausibleRange[1]);

    const searchStart = start + minStanceFrames;
    if (searchStart >= end) { toeOffs.push(null); continue; }
    let found = null;
    for (let j = searchStart; j <= end - persistFrames; j++) {
      let allBelow = true;
      for (let k = j; k < j + persistFrames; k++) {
        if (toeVelocity[k] >= threshold) { allBelow = false; break; }
      }
      if (allBelow) { found = j; break; }
    }
    toeOffs.push(found);
  }

  return { heelY, toeY, toeVelocity, heelStrikes, toeOffs, plausible };
}

function perStrideMetrics(frames, fps, side = 'left') {
  const events = detectGaitEvents(frames, side, fps);
  const knee = kneeFlexionSeries(frames, side);
  const trunk = trunkLeanSeries(frames, side);
  const elbow = elbowAngleSeries(frames, side);
  const footVis = footVisibility(frames, side);

  const rows = [];
  for (let i = 0; i < events.heelStrikes.length - 1; i++) {
    if (!events.plausible[i]) continue;
    const start = events.heelStrikes[i], end = events.heelStrikes[i + 1];
    const strideSec = (end - start) / fps;

    const toeOff = events.toeOffs[i];
    let stanceRatio = null;
    if (toeOff != null && footVis >= STANCE_RATIO_MIN_FOOT_VISIBILITY) {
      stanceRatio = (toeOff - start) / (end - start);
    }

    const kneeSlice = knee.slice(start, end).filter((v) => !Number.isNaN(v));
    const trunkSlice = trunk.slice(start, end).filter((v) => !Number.isNaN(v));
    const elbowSlice = elbow.slice(start, end).filter((v) => !Number.isNaN(v));
    if (!kneeSlice.length || !trunkSlice.length || !elbowSlice.length) continue;

    const peakKneeFlexion = 180 - Math.min(...kneeSlice);
    const trunkLeanMean = trunkSlice.reduce((a, b) => a + b, 0) / trunkSlice.length;
    const elbowRom = Math.max(...elbowSlice) - Math.min(...elbowSlice);

    if (peakKneeFlexion < 0 || peakKneeFlexion > 120) continue;
    if (trunkLeanMean < 0 || trunkLeanMean > 60) continue;
    if (elbowRom < 0 || elbowRom > 150) continue;

    rows.push({
      strideIdx: i,
      strideSec,
      cadencePerMin: 60 / strideSec,
      stanceStrideRatio: stanceRatio,
      peakKneeFlexionDeg: peakKneeFlexion,
      trunkLeanMeanDeg: trunkLeanMean,
      elbowRomDeg: elbowRom,
    });
  }
  return rows;
}

// ---------- scoring against reference bands ----------

const METRIC_KEY_MAP = {
  stride_sec: 'strideSec',
  cadence_per_min: 'cadencePerMin',
  stance_stride_ratio: 'stanceStrideRatio',
  peak_knee_flexion_deg: 'peakKneeFlexionDeg',
  trunk_lean_mean_deg: 'trunkLeanMeanDeg',
  elbow_rom_deg: 'elbowRomDeg',
};

function scoreValue(value, bands, metric) {
  const healthy = bands.healthy[metric];
  const impaired = bands.impaired[metric];
  if (!healthy || !impaired || value == null || Number.isNaN(value)) return null;
  const span = healthy.median - impaired.median;
  if (span === 0) return null;
  return (value - impaired.median) / span;
}

function bandSeparationOk(bands, metric, minRatio = 0.2) {
  const healthy = bands.healthy[metric];
  const impaired = bands.impaired[metric];
  if (!healthy || !impaired) return false;
  const span = Math.abs(healthy.median - impaired.median);
  const pooledSpread = (healthy.iqr + impaired.iqr) / 2;
  if (pooledSpread === 0) return false;
  return span >= minRatio * pooledSpread;
}

function mean(arr) {
  const valid = arr.filter((v) => v != null && !Number.isNaN(v));
  if (!valid.length) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

function compareConditions(stridesA, stridesB, bands, metrics, labelA = 'A', labelB = 'B') {
  const rows = metrics.map((m) => {
    const key = METRIC_KEY_MAP[m];
    const scoresA = stridesA.map((s) => scoreValue(s[key], bands, m));
    const scoresB = stridesB.map((s) => scoreValue(s[key], bands, m));
    return {
      metric: m,
      wellSeparated: bandSeparationOk(bands, m),
      rawMeanA: mean(stridesA.map((s) => s[key])),
      rawMeanB: mean(stridesB.map((s) => s[key])),
      scoreA: mean(scoresA),
      scoreB: mean(scoresB),
    };
  });

  const trustworthy = rows.filter((r) => r.wellSeparated && r.scoreA != null && r.scoreB != null);
  if (!trustworthy.length) {
    return { rows, overallA: null, overallB: null, verdict: 'No metric had well-separated reference bands -- cannot compute a reliable verdict.' };
  }
  const overallA = mean(trustworthy.map((r) => r.scoreA));
  const overallB = mean(trustworthy.map((r) => r.scoreB));
  const verdict = overallB > overallA
    ? `${labelB} scores closer to the healthy reference band overall`
    : `${labelA} scores closer to the healthy reference band overall`;
  return { rows, overallA, overallB, verdict, usedMetrics: trustworthy.map((r) => r.metric) };
}

export {
  LANDMARK, detectGaitEvents, perStrideMetrics, kneeFlexionSeries, trunkLeanSeries,
  elbowAngleSeries, scoreValue, bandSeparationOk, compareConditions, footVisibility,
};
