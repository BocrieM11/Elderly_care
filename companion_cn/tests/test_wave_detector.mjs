import assert from 'node:assert/strict';
import {
  palmCenterX,
  WaveDetector,
} from '../static/wave_detector.mjs';

function feed(detector, positions, start = 0, step = 150) {
  return positions.map((x, index) => detector.update({
    timestamp: start + index * step,
    openPalm: true,
    score: 0.9,
    x,
  }));
}

assert.equal(
  palmCenterX(Array.from({ length: 21 }, (_, index) => ({ x: index / 20 }))),
  (0 + 5 / 20 + 9 / 20 + 13 / 20 + 17 / 20) / 5,
);

const stationary = new WaveDetector();
assert.equal(feed(stationary, [0.50, 0.51, 0.49, 0.51, 0.50, 0.49]).some(Boolean), false);

const oneDirection = new WaveDetector();
assert.equal(feed(oneDirection, [0.30, 0.34, 0.39, 0.45, 0.51, 0.57]).some(Boolean), false);

const waving = new WaveDetector();
assert.equal(feed(waving, [0.35, 0.42, 0.51, 0.45, 0.35, 0.42]).some(Boolean), true);

const casualWave = new WaveDetector();
assert.equal(feed(casualWave, [0.42, 0.46, 0.50, 0.46, 0.42], 0, 130).some(Boolean), true);

const moderateConfidence = new WaveDetector();
assert.equal(
  [0.42, 0.46, 0.50, 0.46, 0.42].map((x, index) => moderateConfidence.update({
    timestamp: index * 130,
    openPalm: true,
    score: 0.55,
    x,
  })).some(Boolean),
  true,
);

const interrupted = new WaveDetector();
feed(interrupted, [0.35, 0.43, 0.51], 0);
interrupted.update({ timestamp: 1100, openPalm: false, score: 0, x: null });
assert.equal(feed(interrupted, [0.45, 0.39, 0.45], 1250).some(Boolean), false);

const cooldown = new WaveDetector();
assert.equal(feed(cooldown, [0.35, 0.42, 0.51, 0.45, 0.35, 0.42]).some(Boolean), true);
assert.equal(
  feed(cooldown, [0.35, 0.42, 0.51, 0.45, 0.35, 0.42], 900).some(Boolean),
  false,
);
assert.equal(
  feed(cooldown, [0.35, 0.42, 0.51, 0.45, 0.35, 0.42], 6000).some(Boolean),
  true,
);

process.stdout.write('wave detector regression passed\n');
