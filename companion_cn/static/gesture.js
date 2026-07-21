import {
  FilesetResolver,
  GestureRecognizer,
} from './vendor/mediapipe/vision_bundle.mjs';
import { palmCenterX, WaveDetector } from './wave_detector.mjs?v=20260720-wave4';

const CHECK_INTERVAL_MS = 130;
const RETRY_INTERVAL_MS = 500;
const OPEN_PALM_SCORE = 0.5;
const READY_TEXT = '视觉对话和挥手唤醒已开启；对着镜头自然挥手，AI 会主动和你打招呼。';
const PALM_TEXT = '已经看到你张开的手掌，请自然地左右挥动一下。';

const video = document.getElementById('companionVideo');
const noteText = document.getElementById('videoNoteText');
const detector = new WaveDetector({ scoreThreshold: OPEN_PALM_SCORE });

let recognizer = null;
let recognizerPromise = null;
let lastProcessedAt = 0;
let lastVideoTime = -1;
let stopped = false;
let greetingStatusUntil = 0;

function cameraActive() {
  return Boolean(window.companionCamera?.isActive?.());
}

function setCameraNote(text) {
  if (cameraActive() && noteText) noteText.textContent = text;
}

async function createRecognizer(delegate) {
  const vision = await FilesetResolver.forVisionTasks('/vendor/mediapipe/wasm');
  return GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: '/vendor/mediapipe/gesture_recognizer.task',
      delegate,
    },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    cannedGesturesClassifierOptions: {
      scoreThreshold: OPEN_PALM_SCORE,
      categoryAllowlist: ['Open_Palm'],
    },
  });
}

async function ensureRecognizer() {
  if (recognizer) return recognizer;
  if (recognizerPromise) return recognizerPromise;

  setCameraNote('正在准备本地挥手识别，摄像头画面不会上传……');
  recognizerPromise = (async () => {
    try {
      recognizer = await createRecognizer('GPU');
    } catch (gpuError) {
      console.warn('GPU gesture recognition unavailable; using CPU.', gpuError);
      recognizer = await createRecognizer('CPU');
    }
    setCameraNote(READY_TEXT);
    return recognizer;
  })().catch(error => {
    recognizerPromise = null;
    setCameraNote('摄像头可用，但挥手识别加载失败；普通视觉对话不受影响。');
    console.error('Failed to initialize gesture recognition:', error);
    throw error;
  });
  return recognizerPromise;
}

function handleRecognition(result, timestamp) {
  const gesture = result?.gestures?.[0]?.[0];
  const landmarks = result?.landmarks?.[0];
  const openPalm = gesture?.categoryName === 'Open_Palm';
  const score = Number(gesture?.score || 0);
  const x = palmCenterX(landmarks);
  const waved = detector.update({ timestamp, openPalm, score, x });
  if (!waved) {
    if (timestamp >= greetingStatusUntil) {
      setCameraNote(openPalm && score >= OPEN_PALM_SCORE && Number.isFinite(x)
        ? PALM_TEXT
        : READY_TEXT);
    }
    return;
  }

  greetingStatusUntil = timestamp + 2200;
  setCameraNote('检测到你挥手啦，正在开启语音陪伴并请 AI 主动打招呼……');
  if (typeof window.companionWaveGreeting === 'function') {
    Promise.resolve(window.companionWaveGreeting()).then(voiceReady => {
      if (!voiceReady) {
        greetingStatusUntil = performance.now() + 4000;
        setCameraNote('AI 会继续问候；请允许麦克风权限后再挥手，即可直接语音对话。');
      }
    }).catch(error => {
      console.warn('Wave greeting startup failed:', error);
    });
  }
  window.setTimeout(() => setCameraNote(READY_TEXT), 2200);
}

async function processFrame(now) {
  if (
    !cameraActive()
    || document.hidden
    || !video
    || video.readyState < 2
  ) {
    detector.reset();
    lastVideoTime = -1;
    return;
  }

  const activeRecognizer = await ensureRecognizer();
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const result = activeRecognizer.recognizeForVideo(video, now);
  handleRecognition(result, now);
}

async function loop() {
  if (stopped) return;
  const now = performance.now();
  const interval = cameraActive() ? CHECK_INTERVAL_MS : RETRY_INTERVAL_MS;
  if (now - lastProcessedAt >= interval) {
    lastProcessedAt = now;
    try {
      await processFrame(now);
    } catch (error) {
      console.warn('Gesture frame processing failed:', error);
      detector.reset();
    }
  }
  window.setTimeout(loop, interval);
}

document.addEventListener('visibilitychange', () => detector.reset());
window.addEventListener('beforeunload', () => {
  stopped = true;
  recognizer?.close?.();
});

window.companionWaveGesture = {
  reset: () => detector.reset(),
  isReady: () => Boolean(recognizer),
};

void loop();
