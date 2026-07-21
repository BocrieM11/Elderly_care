const DEFAULT_OPTIONS = Object.freeze({
  scoreThreshold: 0.5,
  windowMs: 2200,
  minDurationMs: 350,
  maxOpenPalmGapMs: 700,
  minSamples: 4,
  minStep: 0.012,
  minSpan: 0.07,
  minTravel: 0.14,
  minDirectionChanges: 1,
  cooldownMs: 5000,
});

export function palmCenterX(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length < 18) return null;
  const palmIndexes = [0, 5, 9, 13, 17];
  const values = palmIndexes
    .map(index => Number(landmarks[index]?.x))
    .filter(Number.isFinite);
  if (values.length !== palmIndexes.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export class WaveDetector {
  constructor(options = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.samples = [];
    this.lastOpenPalmAt = null;
    this.lastTriggeredAt = -Infinity;
  }

  reset() {
    this.samples = [];
    this.lastOpenPalmAt = null;
  }

  update({ timestamp, openPalm, score = 0, x = null }) {
    const now = Number(timestamp);
    const position = Number(x);
    if (!Number.isFinite(now)) return false;

    if (now - this.lastTriggeredAt < this.options.cooldownMs) {
      this.reset();
      return false;
    }

    const validPalm = Boolean(openPalm)
      && Number(score) >= this.options.scoreThreshold
      && Number.isFinite(position);
    if (!validPalm) {
      if (
        this.lastOpenPalmAt !== null
        && now - this.lastOpenPalmAt > this.options.maxOpenPalmGapMs
      ) {
        this.reset();
      }
      return false;
    }

    if (
      this.lastOpenPalmAt !== null
      && now - this.lastOpenPalmAt > this.options.maxOpenPalmGapMs
    ) {
      this.reset();
    }
    this.lastOpenPalmAt = now;
    this.samples.push({ timestamp: now, x: position });
    this.samples = this.samples.filter(
      sample => now - sample.timestamp <= this.options.windowMs,
    );

    if (this.samples.length < this.options.minSamples) return false;
    const duration = now - this.samples[0].timestamp;
    if (duration < this.options.minDurationMs) return false;

    const positions = this.samples.map(sample => sample.x);
    const span = Math.max(...positions) - Math.min(...positions);
    let travel = 0;
    let previousDirection = 0;
    let directionChanges = 0;
    for (let index = 1; index < positions.length; index += 1) {
      const delta = positions[index] - positions[index - 1];
      travel += Math.abs(delta);
      if (Math.abs(delta) < this.options.minStep) continue;
      const direction = Math.sign(delta);
      if (previousDirection && direction !== previousDirection) directionChanges += 1;
      previousDirection = direction;
    }

    const detected = span >= this.options.minSpan
      && travel >= this.options.minTravel
      && directionChanges >= this.options.minDirectionChanges;
    if (!detected) return false;

    this.lastTriggeredAt = now;
    this.reset();
    return true;
  }
}
