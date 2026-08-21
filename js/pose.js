import {
  PoseLandmarker, FilesetResolver, DrawingUtils,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
import {
  LANDMARK, kneeFlexionSeries, trunkLeanSeries, elbowAngleSeries,
  perStrideMetrics, compareConditions, footVisibility,
} from './metrics.js';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';

let landmarker = null;
let landmarkerMode = null;
let referenceBands = null;

async function initLandmarker(runningMode = 'VIDEO') {
  // setOptions() tears down and rebuilds the whole MediaPipe graph (visible as
  // "Graph finished closing" / GL context destroy+recreate in the console) --
  // calling it on every analysis, even when the mode hasn't changed, corrupts
  // whichever analysis runs right after. Only call it on an actual mode change.
  if (landmarker && landmarkerMode === runningMode) return landmarker;
  if (landmarker) { await landmarker.setOptions({ runningMode }); landmarkerMode = runningMode; return landmarker; }
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode,
    numPoses: 1,
  }).catch(async () => {
    // GPU delegate isn't guaranteed on every device/browser -- fall back to CPU.
    return PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
      runningMode,
      numPoses: 1,
    });
  });
  landmarkerMode = runningMode;
  return landmarker;
}

async function loadReferenceBands() {
  if (referenceBands) return referenceBands;
  const res = await fetch('./data/reference_bands.json');
  referenceBands = await res.json();
  return referenceBands;
}

function frameToRecord(result) {
  if (!result.landmarks || !result.landmarks.length) return { landmarks: null };
  const pose = result.landmarks[0];
  return { landmarks: pose.map((p) => ({ x: p.x, y: p.y, visibility: p.visibility ?? 1 })) };
}

function drawSkeleton(ctx, canvas, result, videoWidth, videoHeight) {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!result.landmarks || !result.landmarks.length) { ctx.restore(); return; }
  const drawer = new DrawingUtils(ctx);
  for (const pose of result.landmarks) {
    drawer.drawLandmarks(pose, { radius: 3, color: '#5eead4' });
    drawer.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, { color: '#38bdf8', lineWidth: 2 });
  }
  ctx.restore();
}

function currentAngles(record, side = 'left') {
  if (!record.landmarks) return null;
  const frames = [record];
  return {
    knee: kneeFlexionSeries(frames, side)[0],
    trunk: trunkLeanSeries(frames, side)[0],
    elbow: elbowAngleSeries(frames, side)[0],
  };
}

// ---------- Live webcam / video skeleton + live angle readout ----------

async function runLiveLoop(videoEl, canvasEl, onAngles, isRunningRef) {
  await initLandmarker('VIDEO');
  const ctx = canvasEl.getContext('2d');

  function frameLoop() {
    if (!isRunningRef.value) return;
    if (videoEl.readyState >= 2) {
      canvasEl.width = videoEl.videoWidth;
      canvasEl.height = videoEl.videoHeight;
      const result = landmarker.detectForVideo(videoEl, performance.now());
      drawSkeleton(ctx, canvasEl, result, videoEl.videoWidth, videoEl.videoHeight);
      const record = frameToRecord(result);
      const angles = currentAngles(record);
      if (onAngles) onAngles(angles, record);
    }
    requestAnimationFrame(frameLoop);
  }
  requestAnimationFrame(frameLoop);
}

// ---------- Full-video buffered analysis (the "sample video" mode) ----------

async function analyzeVideoFull(videoEl, canvasEl, onProgress) {
  await initLandmarker('VIDEO');
  const ctx = canvasEl.getContext('2d');
  const fps = 15; // sampling rate we drive detectForVideo at -- enough for ~1s strides, keeps demo runtime reasonable
  const frames = [];

  // videoEl.duration is NaN until metadata has loaded -- if a caller reads it
  // too early (e.g. a button clicked right after setting a new src), the
  // frame loop below silently runs zero iterations and this fails with no
  // error at all. Guard against that here so every caller is protected.
  if (Number.isNaN(videoEl.duration) || videoEl.readyState < 1) {
    await new Promise((resolve) => {
      if (videoEl.readyState >= 1) return resolve();
      videoEl.addEventListener('loadedmetadata', resolve, { once: true });
    });
  }
  const duration = videoEl.duration;
  if (!duration || Number.isNaN(duration)) {
    throw new Error('video has no readable duration -- check the source loaded correctly');
  }

  videoEl.pause();
  videoEl.currentTime = 0;
  await new Promise((resolve) => { videoEl.onseeked = resolve; });

  const totalFrames = Math.floor(duration * fps);
  for (let i = 0; i < totalFrames; i++) {
    const t = i / fps;
    videoEl.currentTime = t;
    await new Promise((resolve) => { videoEl.onseeked = resolve; });
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    const result = landmarker.detectForVideo(videoEl, performance.now());
    drawSkeleton(ctx, canvasEl, result, videoEl.videoWidth, videoEl.videoHeight);
    frames.push(frameToRecord(result));
    if (onProgress) onProgress(i / totalFrames);
  }

  const strides = perStrideMetrics(frames, fps, 'left');
  const visibility = footVisibility(frames, 'left');
  return { frames, strides, fps, footVisibility: visibility };
}

export {
  initLandmarker, loadReferenceBands, runLiveLoop, analyzeVideoFull, compareConditions,
};
