(() => {
  const micButton = () => document.getElementById('micBtn');
  const status = () => document.getElementById('voiceStatus');

  let captureStream = null;
  let captureContext = null;
  let captureSource = null;
  let captureProcessor = null;
  let captureSilentGain = null;
  let voiceSocket = null;
  let voiceSocketReady = false;
  let voiceModeEnabled = false;
  let voiceStartPromise = null;
  let conversationAwake = false;
  let conversationIdleTimer = null;
  let sleepAfterAssistantTurn = false;
  let sleepConfirmationPending = false;
  let sleepConfirmationTimer = null;
  let pendingKwsWake = null;
  let recognitionPaused = true;
  let reconnectTimer = null;
  let recognitionResumeTimer = null;
  let bargeInTimer = null;
  let speechDetected = false;
  let assistantTurnActive = false;
  let assistantResponseFinished = true;
  let bargeInTriggered = false;
  let ignoreCurrentTranscript = false;
  let pendingTurnCancellation = Promise.resolve();
  let lastStrongMicAt = 0;
  let playbackContext = null;
  let speechGeneration = 0;
  let speechBuffer = '';
  let speechQueue = [];
  let speechQueueRunning = false;
  let speechAbortController = null;
  let nextSpeechTime = 0;
  const activeSpeechSources = new Set();
  const TARGET_SAMPLE_RATE = 16000;
  const BARGE_IN_CONFIRM_MS = 350;
  const BARGE_IN_MIN_RMS = 0.012;
  const BARGE_IN_LEVEL_WINDOW_MS = 700;
  const POST_PLAYBACK_LISTEN_DELAY_MS = 350;
  const CONVERSATION_IDLE_MS = 5000;
  const SLEEP_CONFIRM_TIMEOUT_MS = 15000;
  const WAKE_PHRASES = Object.freeze(['你好暖暖', '暖暖你好', '小暖同学']);
  const WAKE_PREFIXES = new Set(['', '喂', '嗨', '哎', '诶', '哎呀']);
  const EMPTY_WAKE_TAILS = new Set(['', '啊', '呀', '呢', '哦', '喂', '在吗']);
  const SLEEP_PHRASES = new Set([
    '再见', '拜拜', '晚安', '先休息吧', '你休息吧', '休息吧',
    '不用听了', '停止陪伴', '结束对话', '退出对话',
  ]);
  const SLEEP_CONFIRM_YES = new Set([
    '是', '是的', '对', '对的', '嗯', '嗯嗯', '没错', '好', '好的',
    '可以', '结束吧', '退出吧', '不聊了', '先不聊了',
  ]);
  const SLEEP_CONFIRM_NO = new Set([
    '不是', '不是的', '不', '不要', '不用', '不用结束', '别结束',
    '不退出', '继续', '继续聊', '还想聊', '我还想聊',
  ]);
  const DIRECT_SLEEP_INTENT_PATTERNS = Object.freeze([
    /^(?:我)?(?:现在|今天|今晚|暂时|这会儿|这会)?(?:有点|真的|确实|实在)?(?:不想|不愿意|不愿)(?:再|继续)?(?:和你|跟你)?(?:再|继续)?(?:聊|聊天|说话)(?:下去)?(?:了|啦|吧|了吧)?$/u,
    /^(?:我)?(?:现在|今天|今晚|暂时)?(?:先)?不(?:再)?(?:和你|跟你)?(?:聊|聊天|说话)(?:了|啦|吧|了吧)?$/u,
    /^(?:我)?(?:工作|上班|干活)?(?:有点|很|太|真的|实在)?(?:累|困)(?:了|啦)?(?:所以|那)?(?:我)?(?:今天|今晚|现在|暂时)?(?:先)?不(?:再)?(?:和你|跟你)?(?:聊|聊天|说话)(?:了|啦|吧|了吧)?$/u,
    /^(?:我)?(?:想|要|希望)(?:现在|马上)?(?:结束|退出|停止)(?:这次|当前)?(?:聊天|对话|陪伴)(?:了|啦|吧)?$/u,
    /^(?:好了)?(?:别|不要)(?:再)?(?:和我|跟我)?(?:聊|聊天|说话)(?:了|啦|吧)?$/u,
    /^(?:今天|今晚|这次)?(?:就|先)?聊到(?:这|这里|这儿)(?:吧|了|为止)?$/u,
    /^(?:这次|我们的)?(?:聊天|对话)(?:就|先)?到(?:这|这里|这儿)(?:吧|了)?$/u,
    /^(?:今天|今晚|这次)(?:就|先)?到(?:这|这里|这儿)(?:吧|了|为止)?$/u,
    /^(?:我们|咱们)?(?:下次|改天|回头)再聊(?:吧|了)?$/u,
    /^(?:我们|咱们)(?:先)?休息(?:吧|了)?$/u,
  ]);
  const AMBIGUOUS_SLEEP_INTENT_PATTERNS = Object.freeze([
    /^(?:那|那就)?(?:先|就)?(?:这样|到这|到这里|到这儿)(?:吧|了|好吧)?$/u,
  ]);
  // Keep enough generated audio ahead of playback to absorb Fish Speech,
  // network and Safari scheduling jitter without delaying the first phrase.
  const MAX_PLAYBACK_AHEAD_SECONDS = 6;
  const CLOSING_PUNCTUATION = '”’」』）】》〉〕}';

  function hasSpeakableText(text) {
    return /[A-Za-z0-9\u3400-\u9FFF]/.test(
      String(text || '').replace(/\[[^\]]+\]/g, '')
    );
  }

  function setStatus(text) {
    if (status()) status().textContent = text;
  }

  function normalizeSpokenText(text) {
    return String(text || '')
      .normalize('NFKC')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]/gu, '');
  }

  function detectWakePhrase(text) {
    const normalized = normalizeSpokenText(text);
    for (const phrase of WAKE_PHRASES) {
      const index = normalized.indexOf(phrase);
      if (index < 0 || !WAKE_PREFIXES.has(normalized.slice(0, index))) continue;
      let command = normalized.slice(index + phrase.length)
        .replace(/^[啊呀呢哦诶欸喂]+/u, '');
      if (EMPTY_WAKE_TAILS.has(command)) command = '';
      return { phrase, command, transcript: String(text || '').trim() };
    }
    return null;
  }

  function extractWakeCommand(text, expectedPhrase = '') {
    const detected = detectWakePhrase(text);
    if (detected) return detected.command;

    let normalized = normalizeSpokenText(text);
    const removablePrefixes = expectedPhrase === '小暖同学'
      ? ['小暖同学', '小暖', '暖同学']
      : ['你好暖暖', '暖暖你好', '暖暖'];
    for (const prefix of removablePrefixes) {
      if (normalized.startsWith(prefix)) {
        normalized = normalized.slice(prefix.length);
        break;
      }
    }
    return EMPTY_WAKE_TAILS.has(normalized) ? '' : normalized;
  }

  function stripWakeAcknowledgementEcho(text) {
    return String(text || '').replace(/^(?:我在呢|我在|在呢)/u, '');
  }

  function isSleepPhrase(text) {
    const normalized = normalizeSpokenText(text);
    return SLEEP_PHRASES.has(normalized)
      || /^(再见|拜拜|晚安)[啊呀呢哦吧]*$/u.test(normalized);
  }

  function hasNonExitContext(normalized) {
    return /^(?:如果|假如|假设|比如|例如|要是)/u.test(normalized)
      || /(?:他说|她说|他们说|她们说|朋友说|别人说|对方说).*(?:不想|不愿|不聊)/u.test(normalized)
      || /^(?:你|暖暖).*(?:不想|不愿|不聊|结束|退出)/u.test(normalized)
      || /(?:不是|没有|并不是|并非)不想(?:再|继续)?(?:和你|跟你)?(?:聊|聊天|说话)/u.test(normalized)
      || /(?:别|不要)不(?:和我|跟我)?(?:聊|聊天|说话)/u.test(normalized)
      || /不想不(?:聊|聊天|说话)/u.test(normalized)
      || /不想(?:再)?聊(?:这个|那个|这件|那件|这事|那事|此事|这些|那些|关于|有关|这种|那种)/u.test(normalized)
      || /(?:这个|那个|这件|那件|这事|那事|此事|这个话题|那个话题).*(?:不想|不愿)(?:再|继续)?聊/u.test(normalized);
  }

  function classifySleepIntent(text) {
    const normalized = normalizeSpokenText(text);
    if (!normalized) return 'none';
    if (isSleepPhrase(normalized)) return 'sleep';
    if (hasNonExitContext(normalized)) return 'none';
    if (DIRECT_SLEEP_INTENT_PATTERNS.some(pattern => pattern.test(normalized))) return 'sleep';
    if (AMBIGUOUS_SLEEP_INTENT_PATTERNS.some(pattern => pattern.test(normalized))) return 'confirm';
    return 'none';
  }

  function classifySleepConfirmation(text) {
    const normalized = normalizeSpokenText(text);
    if (SLEEP_CONFIRM_YES.has(normalized)) return 'yes';
    if (SLEEP_CONFIRM_NO.has(normalized)) return 'no';
    return 'other';
  }

  function clearSleepConfirmation() {
    clearTimeout(sleepConfirmationTimer);
    sleepConfirmationTimer = null;
    sleepConfirmationPending = false;
  }

  function requestSleepConfirmation() {
    clearSleepConfirmation();
    sleepConfirmationPending = true;
    sleepConfirmationTimer = setTimeout(clearSleepConfirmation, SLEEP_CONFIRM_TIMEOUT_MS);
    speak('你是想结束这次聊天吗？如果是，请说“是”；如果不是，请说“不是”。');
  }

  function wakeStandbyStatus() {
    return '等待唤醒：请说“你好暖暖”“暖暖你好”或“小暖同学”';
  }

  function prepare() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!playbackContext && AudioContext) {
      playbackContext = new AudioContext({ latencyHint: 'interactive' });
    }
    if (playbackContext?.state === 'suspended') playbackContext.resume();
    return playbackContext;
  }

  function playWakeCue() {
    const audioContext = prepare();
    if (!audioContext || audioContext.state !== 'running') return;
    try {
      const oscillator = audioContext.createOscillator();
      const gain = audioContext.createGain();
      const now = audioContext.currentTime;
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(660, now);
      oscillator.frequency.linearRampToValueAtTime(880, now + 0.1);
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(0.08, now + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.13);
      oscillator.connect(gain);
      gain.connect(audioContext.destination);
      oscillator.start(now);
      oscillator.stop(now + 0.14);
    } catch (error) {
      console.warn('无法播放唤醒提示音', error);
    }
  }

  function resample(samples, fromRate, toRate) {
    if (fromRate === toRate) return samples;
    const length = Math.max(1, Math.round(samples.length * toRate / fromRate));
    const output = new Float32Array(length);
    const ratio = fromRate / toRate;
    for (let i = 0; i < length; i++) {
      const position = i * ratio;
      const left = Math.floor(position);
      const right = Math.min(left + 1, samples.length - 1);
      const fraction = position - left;
      output[i] = samples[left] * (1 - fraction) + samples[right] * fraction;
    }
    return output;
  }

  function pcm16FromFloat(samples) {
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const value = Math.max(-1, Math.min(1, samples[i]));
      pcm[i] = value < 0 ? value * 32768 : value * 32767;
    }
    return pcm;
  }

  function updateMicButton() {
    const button = micButton();
    if (!button) return;
    button.classList.toggle('recording', voiceModeEnabled);
    button.textContent = voiceModeEnabled ? '■ 暂停语音识别' : '🎙 重新开启语音识别';
  }

  function vadSocketUrl() {
    const configuredOrigin = typeof COMPANION_API_ORIGIN !== 'undefined'
      ? COMPANION_API_ORIGIN
      : '';
    if (configuredOrigin) {
      return configuredOrigin.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:') + '/v1/media/listen';
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return protocol + '//' + window.location.host + '/v1/media/listen';
  }

  function sendVadControl(type) {
    if (voiceSocketReady && voiceSocket?.readyState === WebSocket.OPEN) {
      voiceSocket.send(JSON.stringify({ type }));
    }
  }

  function suspendRecognition(message = '正在处理…') {
    clearTimeout(recognitionResumeTimer);
    recognitionPaused = true;
    sendVadControl('reset');
    if (voiceModeEnabled) setStatus(message);
  }

  function clearConversationIdleTimer() {
    clearTimeout(conversationIdleTimer);
    conversationIdleTimer = null;
  }

  function enterWakeWordStandby(message = '') {
    clearConversationIdleTimer();
    clearSleepConfirmation();
    conversationAwake = false;
    sleepAfterAssistantTurn = false;
    pendingKwsWake = null;
    clearBargeInCandidate();
    if (!voiceModeEnabled) return;
    recognitionPaused = false;
    sendVadControl('sleep');
    setStatus(message || wakeStandbyStatus());
  }

  function armConversationIdleTimer() {
    clearConversationIdleTimer();
    if (!voiceModeEnabled || !conversationAwake || assistantTurnActive) return;
    conversationIdleTimer = setTimeout(() => {
      enterWakeWordStandby('没有继续说话，已回到语音唤醒状态');
    }, CONVERSATION_IDLE_MS);
  }

  function activateConversation(source = 'voice') {
    clearConversationIdleTimer();
    conversationAwake = true;
    if (source !== 'assistant-turn') sleepAfterAssistantTurn = false;
    if (source !== 'kws') pendingKwsWake = null;
    if (voiceModeEnabled && !assistantTurnActive) {
      recognitionPaused = false;
      if (source !== 'kws') sendVadControl('activate');
      setStatus(source === 'wave' ? '挥手唤醒成功…' : '语音唤醒成功…');
    }
  }

  function resumeRecognition() {
    if (!voiceModeEnabled || !voiceSocketReady || voiceSocket?.readyState !== WebSocket.OPEN) return;
    recognitionPaused = false;
    sendVadControl('reset');
    setStatus(conversationAwake ? '正在听，直接说话即可…' : wakeStandbyStatus());
  }

  function assistantActivityStatus() {
    return activeSpeechSources.size || speechQueueRunning || speechQueue.length
      ? '正在播放，您可以直接说话打断…'
      : '正在生成回复，您可以直接说话打断…';
  }

  function clearBargeInCandidate() {
    clearTimeout(bargeInTimer);
    bargeInTimer = null;
    speechDetected = false;
  }

  function confirmBargeIn() {
    bargeInTimer = null;
    if (!voiceModeEnabled || !assistantTurnActive || !speechDetected || bargeInTriggered) return;

    // Acoustic echo cancellation removes most loudspeaker leakage. Requiring
    // recent near-field energy adds another guard against TTS echo.
    if (Date.now() - lastStrongMicAt > BARGE_IN_LEVEL_WINDOW_MS) {
      bargeInTimer = setTimeout(confirmBargeIn, 120);
      return;
    }

    bargeInTriggered = true;
    ignoreCurrentTranscript = false;
    clearSpeechPlayback();
    setStatus('已打断，请继续说…');
    try {
      pendingTurnCancellation = typeof window.interruptAssistantTurn === 'function'
        ? Promise.resolve(window.interruptAssistantTurn('voice-barge-in')).catch(error => {
            console.warn('取消上一轮回复失败', error);
          })
        : Promise.resolve();
    } catch (error) {
      console.warn('取消上一轮回复失败', error);
      pendingTurnCancellation = Promise.resolve();
    }
  }

  function armBargeIn() {
    clearTimeout(bargeInTimer);
    bargeInTimer = setTimeout(confirmBargeIn, BARGE_IN_CONFIRM_MS);
    setStatus('听到您说话了，继续说即可打断…');
  }

  async function completePendingKwsWake(transcript = '') {
    const wake = pendingKwsWake;
    if (!wake) return false;
    pendingKwsWake = null;
    const command = stripWakeAcknowledgementEcho(
      extractWakeCommand(transcript, wake.phrase)
    );
    if (command && typeof window.send === 'function') {
      if (!wake.acknowledged && typeof window.companionVoiceWakeup === 'function') {
        await window.companionVoiceWakeup(wake, { visualOnly: true });
      }
      const accepted = await window.send({ text: command, source: 'voice-wake' });
      if (accepted === false) enterWakeWordStandby('当前正忙，请稍后再说唤醒词');
    } else {
      if (typeof window.companionVoiceWakeup === 'function') {
        if (!wake.acknowledged) await window.companionVoiceWakeup(wake);
        else if (wake.acknowledgementInterrupted) {
          await window.companionVoiceWakeup(wake, { audioOnly: true });
        }
      }
      recognitionPaused = false;
      sendVadControl('reset');
      if (!assistantTurnActive) armConversationIdleTimer();
    }
    return true;
  }

  async function handleVadMessage(data) {
    if (data.type === 'ready') {
      voiceSocketReady = true;
      if (voiceModeEnabled && !recognitionPaused) {
        if (conversationAwake) {
          sendVadControl('activate');
          setStatus('语音连接已恢复，正在听…');
        } else {
          resumeRecognition();
        }
      }
      return;
    }
    if (data.type === 'wake_word') {
      const phrase = String(data.phrase || '').trim();
      const wake = {
        phrase,
        transcript: phrase,
        detectedAtMs: data.detected_at_ms,
        acknowledged: false,
        acknowledgementInterrupted: false,
      };
      pendingKwsWake = wake;
      activateConversation('kws');
      playWakeCue();
      setStatus(`已快速唤醒“${phrase}”，正在听后半句…`);
      if (typeof window.companionVoiceWakeup === 'function') {
        try {
          wake.acknowledged = true;
          await window.companionVoiceWakeup(wake, { preserveWakeCapture: true });
        } catch (error) {
          wake.acknowledged = false;
          console.error('KWS 唤醒确认播放失败', error);
        }
      }
      return;
    }
    if (data.type === 'listening') {
      if (voiceModeEnabled && !recognitionPaused) {
        setStatus(conversationAwake ? '正在听，直接说话即可…' : wakeStandbyStatus());
      }
      return;
    }
    if (data.type === 'speech_start' && !recognitionPaused) {
      if (conversationAwake) clearConversationIdleTimer();
      speechDetected = true;
      ignoreCurrentTranscript = false;
      if (assistantTurnActive) armBargeIn();
      else setStatus(conversationAwake ? '听到您说话了…' : '正在判断唤醒词…');
      return;
    }
    if (data.type === 'speech_end') {
      speechDetected = false;
      clearTimeout(bargeInTimer);
      bargeInTimer = null;
      recognitionPaused = true;
      if (assistantTurnActive && !bargeInTriggered && !pendingKwsWake) {
        ignoreCurrentTranscript = true;
        setStatus(assistantActivityStatus());
      } else {
        ignoreCurrentTranscript = false;
        setStatus('正在识别…');
      }
      return;
    }
    if (data.type === 'no_speech') {
      clearBargeInCandidate();
      ignoreCurrentTranscript = false;
      recognitionPaused = false;
      if (pendingKwsWake) {
        try {
          await completePendingKwsWake('');
        } catch (error) {
          console.error('KWS 唤醒后的问候启动失败', error);
          enterWakeWordStandby('唤醒后启动失败，请再说一次唤醒词');
        }
        return;
      }
      if (assistantTurnActive) {
        sendVadControl('reset');
        setStatus(assistantActivityStatus());
      } else {
        resumeRecognition();
      }
      return;
    }
    if (data.type === 'transcript') {
      const text = String(data.text || '').trim();
      const wasBargeIn = bargeInTriggered;
      if (assistantTurnActive && !wasBargeIn && !pendingKwsWake) {
        // Residual loudspeaker audio can still trigger VAD on some devices.
        // Never feed that transcript back into the conversation.
        ignoreCurrentTranscript = false;
        recognitionPaused = false;
        sendVadControl('reset');
        setStatus(assistantActivityStatus());
        return;
      }
      if (wasBargeIn) {
        if (pendingKwsWake) pendingKwsWake.acknowledgementInterrupted = true;
        await pendingTurnCancellation;
        assistantTurnActive = false;
        assistantResponseFinished = true;
        bargeInTriggered = false;
        pendingTurnCancellation = Promise.resolve();
      }
      if (!text) {
        if (pendingKwsWake) {
          await completePendingKwsWake('');
          return;
        }
        recognitionPaused = false;
        setStatus(conversationAwake ? '没有听清，继续听您说话…' : wakeStandbyStatus());
        setTimeout(resumeRecognition, 300);
        return;
      }

      if (pendingKwsWake) {
        try {
          await completePendingKwsWake(text);
        } catch (error) {
          console.error('KWS 唤醒后的转写处理失败', error);
          enterWakeWordStandby('唤醒后处理失败，请再说一次唤醒词');
        }
        return;
      }

      if (!conversationAwake) {
        const wake = detectWakePhrase(text);
        if (!wake) {
          recognitionPaused = false;
          setTimeout(resumeRecognition, 150);
          return;
        }

        activateConversation('voice-wake');
        setStatus(`已听到“${wake.phrase}”`);
        try {
          if (wake.command && typeof window.send === 'function') {
            const accepted = await window.send({ text: wake.command, source: 'voice-wake' });
            if (accepted === false) {
              enterWakeWordStandby('当前正忙，请稍后再说唤醒词');
            }
          } else if (typeof window.companionVoiceWakeup === 'function') {
            await window.companionVoiceWakeup(wake);
          } else {
            armConversationIdleTimer();
            resumeRecognition();
          }
        } catch (error) {
          console.error('语音唤醒后的对话启动失败', error);
          enterWakeWordStandby('唤醒后启动失败，请再说一次唤醒词');
        }
        return;
      }

      setStatus(`识别：${text}`);
      if (sleepConfirmationPending) {
        const confirmation = classifySleepConfirmation(text);
        clearSleepConfirmation();
        if (confirmation === 'yes') {
          sleepAfterAssistantTurn = true;
          try {
            if (typeof window.send === 'function') {
              const accepted = await window.send({
                text: '是的，我想结束这次对话',
                source: 'voice-exit-confirmed',
              });
              if (accepted === false) {
                sleepAfterAssistantTurn = false;
                recognitionPaused = false;
                setTimeout(resumeRecognition, 500);
              }
            }
          } catch (error) {
            sleepAfterAssistantTurn = false;
            console.error(error);
            recognitionPaused = false;
            setTimeout(resumeRecognition, 500);
          }
          return;
        }
        if (confirmation === 'no') {
          speak('好，那我们继续聊。');
          return;
        }
      }

      const sleepIntent = classifySleepIntent(text);
      if (sleepIntent === 'confirm') {
        requestSleepConfirmation();
        return;
      }
      if (sleepIntent === 'sleep') sleepAfterAssistantTurn = true;
      try {
        if (typeof window.send === 'function') {
          const accepted = await window.send({ text, source: 'voice' });
          if (accepted === false) {
            if (sleepIntent === 'sleep') sleepAfterAssistantTurn = false;
            recognitionPaused = false;
            setTimeout(resumeRecognition, 500);
          }
        }
      } catch (error) {
        if (sleepIntent === 'sleep') sleepAfterAssistantTurn = false;
        console.error(error);
        recognitionPaused = false;
        setTimeout(resumeRecognition, 500);
      }
      return;
    }
    if (data.type === 'error') {
      console.error('FunASR VAD error:', data.message);
      setStatus('语音服务暂时不可用，正在重连…');
    }
  }

  function connectVoiceSocket() {
    if (voiceSocket &&
        (voiceSocket.readyState === WebSocket.OPEN || voiceSocket.readyState === WebSocket.CONNECTING)) {
      return;
    }
    voiceSocketReady = false;
    const socket = new WebSocket(vadSocketUrl());
    voiceSocket = socket;
    socket.binaryType = 'arraybuffer';
    socket.onmessage = event => {
      try {
        void handleVadMessage(JSON.parse(event.data));
      } catch (error) {
        console.warn('忽略无效的 VAD 消息', error);
      }
    };
    socket.onclose = () => {
      if (voiceSocket === socket) {
        voiceSocket = null;
        voiceSocketReady = false;
      }
      if (voiceModeEnabled) {
        recognitionPaused = true;
        setStatus('语音连接已断开，正在重连…');
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => {
          recognitionPaused = false;
          connectVoiceSocket();
        }, 1500);
      }
    };
    socket.onerror = () => socket.close();
  }

  async function startCapture() {
    captureStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const audioTrack = captureStream.getAudioTracks?.()[0];
    if (audioTrack?.getSettings) {
      const settings = audioTrack.getSettings();
      console.info('麦克风音频处理状态', {
        echoCancellation: settings.echoCancellation,
        noiseSuppression: settings.noiseSuppression,
        autoGainControl: settings.autoGainControl,
        sampleRate: settings.sampleRate,
      });
    }
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    captureContext = new AudioContext({ latencyHint: 'interactive' });
    if (captureContext.state === 'suspended') {
      try { await captureContext.resume(); } catch (_) {}
    }
    captureSource = captureContext.createMediaStreamSource(captureStream);
    captureProcessor = captureContext.createScriptProcessor(4096, 1, 1);
    captureSilentGain = captureContext.createGain();
    captureSilentGain.gain.value = 0;
    captureProcessor.onaudioprocess = event => {
      if (!voiceModeEnabled || recognitionPaused || !voiceSocketReady ||
          voiceSocket?.readyState !== WebSocket.OPEN) return;
      const inputSamples = new Float32Array(event.inputBuffer.getChannelData(0));
      let energy = 0;
      for (let i = 0; i < inputSamples.length; i++) energy += inputSamples[i] * inputSamples[i];
      if (Math.sqrt(energy / Math.max(1, inputSamples.length)) >= BARGE_IN_MIN_RMS) {
        lastStrongMicAt = Date.now();
      }
      const samples16k = resample(inputSamples, captureContext.sampleRate, TARGET_SAMPLE_RATE);
      const pcm = pcm16FromFloat(samples16k);
      if (pcm.byteLength) voiceSocket.send(pcm.buffer);
    };
    captureSource.connect(captureProcessor);
    captureProcessor.connect(captureSilentGain);
    captureSilentGain.connect(captureContext.destination);
  }

  async function startVoiceMode() {
    if (voiceModeEnabled) return;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('当前浏览器不支持麦克风');
    const button = micButton();
    if (button) button.disabled = true;
    setStatus('正在请求麦克风权限…');
    try {
      prepare();
      await startCapture();
      voiceModeEnabled = true;
      conversationAwake = false;
      recognitionPaused = false;
      updateMicButton();
      connectVoiceSocket();
      setStatus('正在连接语音唤醒服务…');
    } finally {
      if (button) button.disabled = false;
    }
  }

  async function ensureVoiceMode() {
    if (voiceModeEnabled) return true;
    if (voiceStartPromise) return voiceStartPromise;

    voiceStartPromise = (async () => {
      try {
        await startVoiceMode();
        return true;
      } catch (error) {
        console.error('无法自动开启语音陪伴', error);
        try {
          await stopVoiceMode();
        } catch (_) {
          // Keep the microphone startup error as the useful failure reason.
        }
        setStatus('');
        return false;
      } finally {
        voiceStartPromise = null;
        if (micButton()) micButton().disabled = false;
      }
    })();
    return voiceStartPromise;
  }

  async function stopVoiceMode() {
    voiceModeEnabled = false;
    recognitionPaused = true;
    clearConversationIdleTimer();
    clearSleepConfirmation();
    conversationAwake = false;
    sleepAfterAssistantTurn = false;
    pendingKwsWake = null;
    clearTimeout(reconnectTimer);
    clearTimeout(recognitionResumeTimer);
    clearBargeInCandidate();
    assistantTurnActive = false;
    assistantResponseFinished = true;
    bargeInTriggered = false;
    ignoreCurrentTranscript = false;
    sendVadControl('reset');
    voiceSocket?.close(1000, 'voice mode stopped');
    voiceSocket = null;
    voiceSocketReady = false;
    captureProcessor?.disconnect();
    captureSource?.disconnect();
    captureSilentGain?.disconnect();
    captureStream?.getTracks().forEach(track => track.stop());
    if (captureContext) await captureContext.close();
    captureStream = captureContext = captureSource = captureProcessor = captureSilentGain = null;
    updateMicButton();
    setStatus('语音陪伴已停止');
  }

  async function toggle() {
    try {
      if (voiceModeEnabled) {
        await stopVoiceMode();
      }
      else await ensureVoiceMode();
    } catch (error) {
      console.error(error);
      try {
        await stopVoiceMode();
      } catch (_) {
        // Keep the original startup error visible below.
      }
      setStatus('无法使用麦克风');
      if (micButton()) micButton().disabled = false;
    }
  }

  function stopScheduledSpeech() {
    for (const scheduledSource of activeSpeechSources) {
      try { scheduledSource.stop(); } catch (error) {}
    }
    activeSpeechSources.clear();
    nextSpeechTime = playbackContext ? playbackContext.currentTime : 0;
  }

  function clearSpeechPlayback() {
    speechGeneration++;
    speechBuffer = '';
    speechQueue = [];
    speechAbortController?.abort();
    speechAbortController = null;
    speechQueueRunning = false;
    stopScheduledSpeech();
  }

  function playbackIsIdle() {
    return !speechBuffer && !speechQueueRunning && !speechQueue.length && !activeSpeechSources.size;
  }

  function notifyPlaybackIdle() {
    if (!playbackIsIdle()) return;
    if (bargeInTriggered) return;
    if (!assistantResponseFinished) {
      if (voiceModeEnabled) setStatus('正在生成回复，您可以直接说话打断…');
      return;
    }
    assistantTurnActive = false;
    bargeInTriggered = false;
    if (sleepAfterAssistantTurn) {
      enterWakeWordStandby('对话已结束，等待下一次语音或挥手唤醒');
      return;
    }
    if (voiceModeEnabled) {
      if (ignoreCurrentTranscript) return;
      clearTimeout(recognitionResumeTimer);
      if (recognitionPaused) {
        recognitionResumeTimer = setTimeout(() => {
          if (voiceModeEnabled && playbackIsIdle() && !ignoreCurrentTranscript) resumeRecognition();
        }, POST_PLAYBACK_LISTEN_DELAY_MS);
      } else {
        setStatus('正在听，直接说话即可…');
      }
      armConversationIdleTimer();
    } else {
      setStatus('ASR / TTS 已就绪');
    }
  }

  function cancel() {
    assistantTurnActive = false;
    assistantResponseFinished = true;
    bargeInTriggered = false;
    ignoreCurrentTranscript = false;
    clearBargeInCandidate();
    clearSpeechPlayback();
    notifyPlaybackIdle();
  }

  function begin(options = {}) {
    const preserveWakeCapture = Boolean(options.preserveWakeCapture);
    activateConversation(preserveWakeCapture ? 'kws' : 'assistant-turn');
    clearSpeechPlayback();
    clearBargeInCandidate();
    assistantTurnActive = true;
    assistantResponseFinished = false;
    bargeInTriggered = false;
    ignoreCurrentTranscript = false;
    lastStrongMicAt = 0;
    pendingTurnCancellation = Promise.resolve();
    if (voiceModeEnabled) {
      recognitionPaused = false;
      if (!preserveWakeCapture) sendVadControl('reset');
    }
    const audioContext = prepare();
    nextSpeechTime = audioContext ? audioContext.currentTime + 0.05 : 0;
    setStatus(voiceModeEnabled ? assistantActivityStatus() : '正在生成回复…');
  }

  function schedulePcm(bytes, rate, generation) {
    if (generation !== speechGeneration || bytes.byteLength < 2) return;
    const evenLength = bytes.byteLength - (bytes.byteLength % 2);
    const view = new DataView(bytes.buffer, bytes.byteOffset, evenLength);
    const samples = new Float32Array(evenLength / 2);
    for (let i = 0; i < samples.length; i++) {
      samples[i] = view.getInt16(i * 2, true) / 32768;
    }

    const audioContext = prepare();
    if (!audioContext) return;
    const audioBuffer = audioContext.createBuffer(1, samples.length, rate);
    audioBuffer.copyToChannel(samples, 0);
    const scheduledSource = audioContext.createBufferSource();
    scheduledSource.buffer = audioBuffer;
    scheduledSource.connect(audioContext.destination);
    const startAt = Math.max(audioContext.currentTime + 0.04, nextSpeechTime);
    scheduledSource.start(startAt);
    nextSpeechTime = startAt + audioBuffer.duration;
    activeSpeechSources.add(scheduledSource);
    scheduledSource.onended = () => {
      activeSpeechSources.delete(scheduledSource);
      if (!speechQueueRunning && speechQueue.length === 0 && activeSpeechSources.size === 0) {
        notifyPlaybackIdle();
      }
    };
    setStatus(voiceModeEnabled ? '正在播放，您可以直接说话打断…' : '正在播放…');
  }

  async function waitForPlaybackWindow(
    generation,
    maxAheadSeconds = MAX_PLAYBACK_AHEAD_SECONDS,
  ) {
    while (generation === speechGeneration) {
      const audioContext = prepare();
      if (!audioContext) return;
      if (audioContext.state === 'suspended') {
        try { await audioContext.resume(); } catch (error) {}
      }
      if (nextSpeechTime - audioContext.currentTime <= maxAheadSeconds) return;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  async function synthesizeSegment(text, generation) {
    const clean = text.replace(/\[[^\]]+\]/g, '').trim();
    if (!clean || !hasSpeakableText(clean) || generation !== speechGeneration) return;
    speechAbortController = new AbortController();
    const response = await fetch('/v1/media/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: clean }),
      signal: speechAbortController.signal,
    });
    if (!response.ok) throw new Error(`TTS ${response.status}`);

    const rate = Number(response.headers.get('X-Audio-Sample-Rate')) || 44100;
    const audio = new Uint8Array(await response.arrayBuffer());
    if (audio.byteLength < 2) throw new Error('TTS 返回了空音频');
    if (generation !== speechGeneration) return;

    // Fish Speech returns each sentence almost all at once. Scheduling one node
    // per sentence avoids hundreds of tiny WebAudio nodes on long replies.
    await waitForPlaybackWindow(generation);
    if (generation === speechGeneration) schedulePcm(audio, rate, generation);
  }

  async function processSpeechQueue(generation) {
    if (speechQueueRunning) return;
    speechQueueRunning = true;
    try {
      while (speechQueue.length && generation === speechGeneration) {
        const segment = speechQueue.shift();
        setStatus(activeSpeechSources.size
          ? (voiceModeEnabled ? '正在播放，您可以直接说话打断…' : '正在播放…')
          : (voiceModeEnabled ? '正在合成首句，您可以直接说话打断…' : '正在合成首句…'));
        let completed = false;
        for (let attempt = 0; attempt < 2 && generation === speechGeneration; attempt++) {
          try {
            await synthesizeSegment(segment, generation);
            completed = true;
            break;
          } catch (error) {
            if (error.name === 'AbortError') throw error;
            console.error(`TTS 片段失败（第 ${attempt + 1} 次）`, error);
            if (attempt === 0) await new Promise(resolve => setTimeout(resolve, 250));
          }
        }
        if (!completed && generation === speechGeneration) {
          setStatus('一个语音片段失败，继续播放后文…');
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError') console.error(error);
    } finally {
      if (generation === speechGeneration) {
        speechQueueRunning = false;
        speechAbortController = null;
        // A token may enqueue a new sentence just as the loop is exiting.
        if (speechQueue.length) {
          void processSpeechQueue(generation);
        } else if (!activeSpeechSources.size) {
          notifyPlaybackIdle();
        }
      }
    }
  }

  function enqueueSpeech(text) {
    const segment = text.trim();
    if (!segment || !hasSpeakableText(segment)) return;
    speechQueue.push(segment);
    void processSpeechQueue(speechGeneration);
  }

  const MIN_TTS_SEGMENT_SPEAKABLE_CHARS = 8;

  function speakableCharacterCount(text) {
    return Array.from(text.replace(/\[[^\]]+\]/g, ''))
      .filter(character => /[\p{L}\p{N}]/u.test(character))
      .length;
  }

  function enqueueFinalSpeech(text) {
    const segment = text.trim();
    if (!segment || !hasSpeakableText(segment)) return;

    // A short tail has too little context for stable speaker conditioning.
    // If an earlier segment has not started synthesis yet, attach it there.
    if (
      speakableCharacterCount(segment) < MIN_TTS_SEGMENT_SPEAKABLE_CHARS
      && speechQueue.length
    ) {
      speechQueue[speechQueue.length - 1] += segment;
      return;
    }
    enqueueSpeech(segment);
  }

  function flushSpeechBuffer(force = false) {
    while (speechBuffer) {
      let cut = -1;
      for (let i = 0; i < speechBuffer.length; i++) {
        const character = speechBuffer[i];
        if ('。！？!?；;\n'.includes(character)) {
          let extendedCut = i + 1;
          while (
            extendedCut < speechBuffer.length
            && (CLOSING_PUNCTUATION.includes(speechBuffer[extendedCut]) || /\s/.test(speechBuffer[extendedCut]))
          ) {
            extendedCut += 1;
          }

          // Streamed tokens may put a closing quote/bracket after the sentence mark.
          // Wait for one more character so it stays attached to the previous phrase.
          if (extendedCut >= speechBuffer.length && !force) return;
          if (
            speakableCharacterCount(speechBuffer.slice(0, extendedCut))
            < MIN_TTS_SEGMENT_SPEAKABLE_CHARS
          ) {
            continue;
          }
          cut = extendedCut;
          break;
        }
        if ('，,：:'.includes(character) && i >= 11) {
          cut = i + 1;
          break;
        }
      }
      if (cut < 0 && speechBuffer.length >= 28) cut = 28;
      if (cut < 0 && force) cut = speechBuffer.length;
      if (cut < 0) return;
      if (force && cut === speechBuffer.length) {
        enqueueFinalSpeech(speechBuffer.slice(0, cut));
      } else {
        enqueueSpeech(speechBuffer.slice(0, cut));
      }
      speechBuffer = speechBuffer.slice(cut);
    }
  }

  function push(token) {
    speechBuffer += String(token || '');
    flushSpeechBuffer(false);
  }

  function finish() {
    assistantResponseFinished = true;
    flushSpeechBuffer(true);
    if (speechQueue.length && !speechQueueRunning) {
      void processSpeechQueue(speechGeneration);
    } else if (!speechQueueRunning && !speechQueue.length && !activeSpeechSources.size) {
      notifyPlaybackIdle();
    }
  }

  function speak(text, options = {}) {
    begin(options);
    push(text);
    finish();
  }

  async function health() {
    try {
      const response = await fetch('/v1/media/health');
      const result = await response.json();
      if (micButton()) micButton().disabled = !result.asr;
      if (result.asr) {
        setStatus('正在自动开启语音识别，首次使用需允许麦克风…');
        await ensureVoiceMode();
      } else {
        setStatus('语音服务不完整');
      }
    } catch (error) {
      setStatus('语音服务不可用');
      if (micButton()) micButton().disabled = true;
    }
  }

  window.companionVoice = {
    toggle, speak, prepare, begin, push, finish, cancel,
    start: ensureVoiceMode,
    stop: stopVoiceMode,
    activateConversation,
    sleep: enterWakeWordStandby,
    detectWakePhrase,
    classifySleepIntent,
    suspendRecognition, resumeRecognition,
    isVoiceModeEnabled: () => voiceModeEnabled,
    isConversationAwake: () => conversationAwake,
  };
  window.addEventListener('DOMContentLoaded', health);
  window.addEventListener('pagehide', () => {
    if (voiceModeEnabled) void stopVoiceMode();
  });
})();
