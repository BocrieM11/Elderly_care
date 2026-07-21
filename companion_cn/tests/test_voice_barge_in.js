const assert = require('node:assert/strict');
const path = require('node:path');

let captureProcessor = null;
let stoppedSources = 0;
let interruptedTurns = 0;
const submittedTexts = [];
const wakeEvents = [];
const sockets = [];

class FakeAudioContext {
  constructor() {
    this.currentTime = 0;
    this.state = 'running';
    this.sampleRate = 48000;
    this.destination = {};
  }

  resume() { return Promise.resolve(); }
  close() { return Promise.resolve(); }
  createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
  createScriptProcessor() {
    captureProcessor = { connect() {}, disconnect() {}, onaudioprocess: null };
    return captureProcessor;
  }
  createGain() {
    return {
      gain: {
        value: 1,
        setValueAtTime() {},
        exponentialRampToValueAtTime() {},
      },
      connect() {}, disconnect() {},
    };
  }
  createOscillator() {
    return {
      type: 'sine',
      frequency: { setValueAtTime() {}, linearRampToValueAtTime() {} },
      connect() {}, start() {}, stop() {},
    };
  }
  createBuffer(_channels, length, sampleRate) {
    return { duration: length / sampleRate, copyToChannel() {} };
  }
  createBufferSource() {
    return {
      onended: null,
      connect() {},
      start() {},
      stop() { stoppedSources += 1; },
    };
  }
}

class FakeWebSocket {
  static OPEN = 1;
  static CONNECTING = 0;

  constructor() {
    this.readyState = FakeWebSocket.OPEN;
    this.sent = [];
    sockets.push(this);
  }

  send(value) { this.sent.push(value); }
  close() { this.readyState = 3; }
  emit(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }
}

const micButton = { disabled: false, textContent: '', classList: { toggle() {} } };
const voiceStatus = { textContent: '' };
global.document = {
  getElementById(id) {
    if (id === 'micBtn') return micButton;
    if (id === 'voiceStatus') return voiceStatus;
    return null;
  },
};
Object.defineProperty(global, 'navigator', {
  configurable: true,
  value: {
    mediaDevices: {
      async getUserMedia() {
        return { getTracks: () => [{ stop() {} }] };
      },
    },
  },
});
global.WebSocket = FakeWebSocket;
global.window = {
  AudioContext: FakeAudioContext,
  webkitAudioContext: FakeAudioContext,
  location: { protocol: 'http:', host: '127.0.0.1:8016' },
  addEventListener() {},
  async interruptAssistantTurn() {
    interruptedTurns += 1;
  },
  async send({ text }) {
    submittedTexts.push(text);
    return true;
  },
  async companionVoiceWakeup(wake, options = {}) {
    wakeEvents.push({ ...wake, options });
  },
};
global.fetch = async () => ({
  ok: true,
  headers: { get: () => '44100' },
  arrayBuffer: async () => new Uint8Array(44100 * 2).buffer,
  json: async () => ({ asr: true, tts: true }),
});

require(path.resolve(__dirname, '../static/voice.js'));

const wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function main() {
  const started = await Promise.all([
    window.companionVoice.start(),
    window.companionVoice.start(),
  ]);
  assert.deepEqual(started, [true, true]);
  assert.equal(sockets.length, 1, 'concurrent wave wakeups must share one microphone session');
  const socket = sockets[0];
  socket.emit({ type: 'ready' });

  assert.deepEqual(
    window.companionVoice.detectWakePhrase('你好，暖暖，今天天气怎么样？'),
    { phrase: '你好暖暖', command: '今天天气怎么样', transcript: '你好，暖暖，今天天气怎么样？' },
  );
  assert.deepEqual(
    window.companionVoice.detectWakePhrase('小暖同学，在吗？'),
    { phrase: '小暖同学', command: '', transcript: '小暖同学，在吗？' },
  );
  assert.equal(window.companionVoice.detectWakePhrase('电视里有人说你好暖暖'), null);
  assert.equal(window.companionVoice.classifySleepIntent('再见'), 'sleep');
  assert.equal(window.companionVoice.classifySleepIntent('我不想聊了'), 'sleep');
  assert.equal(window.companionVoice.classifySleepIntent('我不想和你再聊了'), 'sleep');
  assert.equal(window.companionVoice.classifySleepIntent('今天就聊到这里吧'), 'sleep');
  assert.equal(window.companionVoice.classifySleepIntent('今天先到这里吧'), 'sleep');
  assert.equal(window.companionVoice.classifySleepIntent('下次再聊吧'), 'sleep');
  assert.equal(window.companionVoice.classifySleepIntent('我工作很累，所以先不聊了'), 'sleep');
  assert.equal(window.companionVoice.classifySleepIntent('我有点累了，今天先不聊了'), 'sleep');
  assert.equal(window.companionVoice.classifySleepIntent('先这样吧'), 'confirm');
  assert.equal(window.companionVoice.classifySleepIntent('到这里吧'), 'confirm');
  assert.equal(window.companionVoice.classifySleepIntent('我有点累了'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我工作很累'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我今天工作特别累'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我困了'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我要睡觉了'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我想休息一下'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我想安静一会儿'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我想一个人静静'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我没什么想说的了'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('算了吧'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('你是不是不想聊了'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我不想聊这个话题了'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我不想继续聊工作了'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我不想说这件事'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('朋友说他不想聊了'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('如果我说不想聊了会怎样'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('我不是不想聊'), 'none');
  assert.equal(window.companionVoice.classifySleepIntent('别不跟我聊'), 'none');

  const recordWakeup = window.companionVoiceWakeup;
  window.companionVoiceWakeup = async (wake, options = {}) => {
    wakeEvents.push({ ...wake, options });
    window.companionVoice.speak('我在呢', {
      preserveWakeCapture: Boolean(options.preserveWakeCapture),
    });
  };
  const resetsBeforeWake = socket.sent.filter(value =>
    typeof value === 'string' && value.includes('"action":"reset"')
  ).length;
  socket.emit({ type: 'wake_word', phrase: '你好暖暖', detected_at_ms: Date.now() });
  assert.equal(wakeEvents.length, 1, 'KWS should acknowledge as soon as the keyword is detected');
  assert.equal(wakeEvents[0].options.preserveWakeCapture, true);
  assert.equal(
    socket.sent.filter(value =>
      typeof value === 'string' && value.includes('"action":"reset"')
    ).length,
    resetsBeforeWake,
    'immediate acknowledgement must not reset the in-flight wake recording',
  );
  socket.emit({ type: 'speech_end', duration_ms: 1200 });
  socket.emit({
    type: 'transcript',
    text: '你好，暖暖，我在呢，今天天气怎么样？',
    wake_transition: true,
  });
  await wait(20);
  assert.deepEqual(
    submittedTexts,
    ['今天天气怎么样'],
    'a wake phrase and a fast follow-up question must submit only the question',
  );
  window.companionVoice.cancel();
  window.companionVoice.sleep();
  window.companionVoiceWakeup = recordWakeup;
  submittedTexts.length = 0;
  wakeEvents.length = 0;

  socket.emit({ type: 'transcript', text: '今天天气不错' });
  await wait(20);
  assert.deepEqual(submittedTexts, [], 'standby speech without a wake phrase must stay local');
  assert.equal(window.companionVoice.isConversationAwake(), false);

  socket.emit({ type: 'wake_word', phrase: '暖暖你好', detected_at_ms: Date.now() });
  assert.equal(wakeEvents.length, 1, 'KWS acknowledgement must not wait for FunASR');
  assert.equal(wakeEvents[0].options.preserveWakeCapture, true);
  socket.emit({ type: 'speech_end', duration_ms: 900 });
  socket.emit({ type: 'transcript', text: '暖暖，你好。', wake_transition: true });
  await wait(20);
  assert.equal(wakeEvents.length, 1, 'the ASR result must not duplicate the acknowledgement');
  assert.equal(wakeEvents[0].phrase, '暖暖你好');
  assert.equal(window.companionVoice.isConversationAwake(), true);

  window.companionVoice.begin();
  window.companionVoice.push('这是一段正在播放、可以被用户打断的回复。');
  window.companionVoice.finish();
  await wait(20);

  const loudSamples = new Float32Array(4096).fill(0.05);
  captureProcessor.onaudioprocess({
    inputBuffer: { getChannelData: () => loudSamples },
  });
  socket.emit({ type: 'speech_start' });
  await wait(420);

  assert.equal(interruptedTurns, 1, 'confirmed speech should abort the active chat turn');
  assert.ok(stoppedSources >= 1, 'confirmed speech should stop scheduled TTS audio');

  socket.emit({ type: 'speech_end', duration_ms: 900 });
  socket.emit({ type: 'transcript', text: '先停一下' });
  await wait(20);
  assert.deepEqual(submittedTexts, ['先停一下']);

  // A short VAD trigger during another assistant turn is treated as residual
  // speaker echo and must not be submitted as a new user message.
  window.companionVoice.begin();
  socket.emit({ type: 'speech_start' });
  socket.emit({ type: 'speech_end', duration_ms: 120 });
  socket.emit({ type: 'transcript', text: '助手自己的声音' });
  await wait(20);
  assert.deepEqual(submittedTexts, ['先停一下']);
  assert.equal(interruptedTurns, 1);

  window.companionVoice.cancel();
  socket.emit({ type: 'transcript', text: '我不想聊了' });
  await wait(20);
  assert.deepEqual(submittedTexts, ['先停一下', '我不想聊了']);
  window.companionVoice.begin();
  window.companionVoice.cancel();
  assert.equal(
    window.companionVoice.isConversationAwake(),
    false,
    'a completed goodbye turn should return to wake-word standby',
  );

  socket.emit({ type: 'wake_word', phrase: '你好暖暖', detected_at_ms: Date.now() });
  socket.emit({ type: 'speech_end', duration_ms: 800 });
  socket.emit({ type: 'transcript', text: '你好暖暖', wake_transition: true });
  await wait(20);
  assert.equal(window.companionVoice.isConversationAwake(), true);
  assert.deepEqual(
    submittedTexts,
    ['先停一下', '我不想聊了'],
    'a standalone wake phrase must not be sent to the LLM',
  );
  await wait(5200);
  assert.equal(
    window.companionVoice.isConversationAwake(),
    false,
    'five seconds without user speech should return to wake-word standby',
  );

  await window.companionVoice.toggle();
  process.stdout.write('voice barge-in regression passed\n');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
