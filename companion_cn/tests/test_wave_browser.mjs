import assert from 'node:assert/strict';

const port = process.env.CDP_PORT || '9333';
const deadline = Date.now() + 15000;
let pages = [];
while (Date.now() < deadline) {
  try {
    pages = await fetch(`http://127.0.0.1:${port}/json/list`).then(response => response.json());
    if (pages.length) break;
  } catch (error) {}
  await new Promise(resolve => setTimeout(resolve, 250));
}

const page = pages.find(candidate => candidate.type === 'page');
assert.ok(page?.webSocketDebuggerUrl, 'Chrome DevTools page was not available');

const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', reject, { once: true });
});

let nextId = 1;
const pending = new Map();
const browserErrors = [];
socket.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message));
    else resolve(message.result);
    return;
  }
  if (message.method === 'Runtime.exceptionThrown') {
    browserErrors.push(message.params?.exceptionDetails?.text || 'Runtime exception');
  }
  if (
    message.method === 'Runtime.consoleAPICalled'
    && message.params?.type === 'error'
  ) {
    browserErrors.push(
      (message.params.args || []).map(argument => argument.value || argument.description).join(' '),
    );
  }
});

function send(method, params = {}) {
  const id = nextId++;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

async function evaluate(expression, awaitPromise = false) {
  const result = await send('Runtime.evaluate', {
    expression,
    awaitPromise,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || 'Evaluation failed');
  }
  return result.result?.value;
}

try {
  await send('Runtime.enable');
  await evaluate(`window.companionCamera.isActive()
    ? Promise.resolve()
    : window.companionCamera.toggle()`, true);
  const readyDeadline = Date.now() + 20000;
  let state;
  while (Date.now() < readyDeadline) {
    state = await evaluate(`({
      camera: Boolean(window.companionCamera?.isActive?.()),
      gesture: Boolean(window.companionWaveGesture?.isReady?.()),
      status: document.getElementById('videoStatus')?.textContent || ''
    })`);
    if (state.camera && state.gesture) break;
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  assert.equal(state?.camera, true, `Fake camera did not start: ${JSON.stringify(state)}`);
  assert.equal(state?.gesture, true, `Gesture model did not initialize: ${JSON.stringify(state)}`);
  assert.match(state.status, /摄像头已开启/);
  const actionableErrors = browserErrors.filter(
    error => !error.includes('Created TensorFlow Lite XNNPACK delegate for CPU'),
  );
  assert.deepEqual(actionableErrors, []);
  process.stdout.write('wave browser integration passed\n');
} finally {
  socket.close();
}
