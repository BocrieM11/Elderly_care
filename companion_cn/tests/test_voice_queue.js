const assert = require('node:assert/strict');
const path = require('node:path');

const synthesizedTexts = [];

class FakeAudioContext {
  constructor() {
    this.currentTime = 0;
    this.state = 'running';
    this.destination = {};
  }

  resume() {
    return Promise.resolve();
  }

  createBuffer(_channels, length, sampleRate) {
    return {
      duration: length / sampleRate,
      copyToChannel() {},
    };
  }

  createBufferSource() {
    const source = {
      onended: null,
      connect() {},
      start() {
        setTimeout(() => source.onended?.(), 0);
      },
    };
    return source;
  }
}

global.document = { getElementById: () => null };
global.window = {
  AudioContext: FakeAudioContext,
  webkitAudioContext: FakeAudioContext,
  addEventListener() {},
};
global.fetch = async (_url, options) => {
  synthesizedTexts.push(JSON.parse(options.body).text);
  return {
    ok: true,
    headers: { get: () => '44100' },
    arrayBuffer: async () => new Uint8Array([0, 0]).buffer,
  };
};

require(path.resolve(__dirname, '../static/voice.js'));

const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function main() {
  window.companionVoice.begin();
  for (const character of '他说：“今天外面的天气很好！”\n下一句也应该单独播报。') {
    window.companionVoice.push(character);
  }
  window.companionVoice.finish();
  await wait(30);

  assert.deepEqual(synthesizedTexts, [
    '他说：“今天外面的天气很好！”',
    '下一句也应该单独播报。',
  ]);

  synthesizedTexts.length = 0;
  window.companionVoice.begin();
  for (const character of '那可不！今天太阳挺舒服。') {
    window.companionVoice.push(character);
  }
  window.companionVoice.finish();
  await wait(30);

  assert.deepEqual(synthesizedTexts, ['那可不！今天太阳挺舒服。']);

  window.companionVoice.begin();
  window.companionVoice.push('）');
  window.companionVoice.finish();
  await wait(30);

  assert.deepEqual(synthesizedTexts, ['那可不！今天太阳挺舒服。']);
  process.stdout.write('voice queue streaming regression passed\n');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
