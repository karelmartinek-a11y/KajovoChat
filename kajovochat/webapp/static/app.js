const state = {
  config: null,
  terminalLines: ["[READY] Terminál čeká na relaci."],
  visualizer: null,
  pc: null,
  dc: null,
  localStream: null,
  remoteStream: null,
  audioContext: null,
  micAnalyser: null,
  remoteAnalyser: null,
  localSource: null,
  remoteSource: null,
  remoteAudio: null,
  animationHandle: 0,
  uiState: "idle",
  sessionActive: false,
  userSpeaking: false,
  assistantSpeaking: false,
  lastAssistantItemId: null,
  lastAssistantContentIndex: 0,
  lastAssistantAudioStartSec: 0,
  lastAssistantTranscript: "",
  lastUserTranscript: "",
  bargeMuted: false,
  audioTopology: "notebook_builtin",
  micTrackSettings: null,
  constraintsSummary: null,
  inputLevel: 0,
  outputLevel: 0,
  lastEventType: "-",
  dataChannelState: "closed",
  peerState: "new",
  noiseReductionMode: "far_field",
};

const ui = {
  apiKeyInput: document.getElementById("apiKeyInput"),
  saveKeyBtn: document.getElementById("saveKeyBtn"),
  deleteKeyBtn: document.getElementById("deleteKeyBtn"),
  audioTestBtn: document.getElementById("audioTestBtn"),
  startStopBtn: document.getElementById("startStopBtn"),
  settingsBtn: document.getElementById("settingsBtn"),
  savePrefsBtn: document.getElementById("savePrefsBtn"),
  clearBtn: document.getElementById("clearBtn"),
  exitBtn: document.getElementById("exitBtn"),
  statusPill: document.getElementById("statusPill"),
  guardPanel: document.getElementById("guardPanel"),
  terminalLog: document.getElementById("terminalLog"),
  canvas: document.getElementById("ekgCanvas"),
  settingsDialog: document.getElementById("settingsDialog"),
  reportDialog: document.getElementById("reportDialog"),
  reportTitle: document.getElementById("reportTitle"),
  reportBody: document.getElementById("reportBody"),
  reportCloseBtn: document.getElementById("reportCloseBtn"),
  answerLanguageMode: document.getElementById("answerLanguageMode"),
  fixedAnswerLanguage: document.getElementById("fixedAnswerLanguage"),
  responseStyle: document.getElementById("responseStyle"),
  realtimeVoice: document.getElementById("realtimeVoice"),
  settingsApplyBtn: document.getElementById("settingsApplyBtn"),
  remoteAudio: document.getElementById("remoteAudio"),
};

class EkgVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.history = Array.from({ length: 240 }, () => 0);
    this.amplitudeLog = [];
    this.currentSegmentPeak = 0;
    this.phase = 0;
  }

  tick(inputLevel, outputLevel, uiState, active) {
    this.phase = (this.phase + 1) % 96;
    const sample = this._currentWaveSample(inputLevel, outputLevel, uiState, active);
    const ampRatio = 0.24 + Math.max(inputLevel, outputLevel) * 0.68;
    const value = Math.max(-0.94, Math.min(0.94, sample * ampRatio));
    this.history.shift();
    this.history.push(value);
    this.currentSegmentPeak = Math.max(this.currentSegmentPeak, Math.abs(value));
    if (this.phase === 0) {
      this.amplitudeLog.push(`${Math.round(this.currentSegmentPeak * 200)}%`);
      this.amplitudeLog = this.amplitudeLog.slice(-8);
      this.currentSegmentPeak = 0;
    }
    this.draw(inputLevel, outputLevel, uiState, active);
  }

  _beatShape(position) {
    if (position < 0.12) return -0.08 + position * 0.2;
    if (position < 0.18) return 0.12 + (position - 0.12) * 2.6;
    if (position < 0.22) return 0.28 - (position - 0.18) * 4.0;
    if (position < 0.25) return -0.22 - (position - 0.22) * 8.0;
    if (position < 0.29) return -0.46 + (position - 0.25) * 27.0;
    if (position < 0.34) return 0.62 - (position - 0.29) * 11.8;
    if (position < 0.42) return 0.03 - (position - 0.34) * 0.9;
    return Math.sin(position * Math.PI * 2 * 1.15) * 0.03;
  }

  _currentWaveSample(inputLevel, outputLevel, uiState, active) {
    let base = Math.max(inputLevel * 1.7, outputLevel * 1.5);
    if (["connecting", "thinking", "reconnecting"].includes(uiState)) {
      base = Math.max(base, 0.22);
    }
    if (!active && !["speaking", "listening"].includes(uiState)) {
      base *= 0.25;
    }
    const beat = this._beatShape(this.phase / 96);
    const shimmer = (
      Math.sin(this.phase * 0.45) +
      Math.sin(this.phase * 0.19 + 0.8) +
      Math.sin(this.phase * 0.08 + 1.6)
    ) / 3;
    const sample = beat * (0.45 + base * 1.65) + shimmer * base * 0.95;
    return Math.max(-1, Math.min(1, sample));
  }

  draw(inputLevel, outputLevel, uiState, active) {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.round(rect.width * dpr);
    const height = Math.round(rect.height * dpr);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const panelPad = 18;
    const panel = { x: panelPad, y: panelPad, w: w - panelPad * 2, h: h - panelPad * 2 };
    const waveRect = { x: panel.x + 18, y: panel.y + 18, w: panel.w - 36, h: Math.max(140, panel.h * 0.5) };

    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, "#09111A");
    bg.addColorStop(0.45, "#07171D");
    bg.addColorStop(1, "#020705");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    ctx.fillStyle = "rgba(4,18,14,0.82)";
    ctx.fillRect(panel.x, panel.y, panel.w, panel.h);
    ctx.strokeStyle = "rgba(84,161,119,0.2)";
    for (let row = 0; row < 12; row += 1) {
      const y = panel.y + row * (panel.h / 11);
      ctx.beginPath();
      ctx.moveTo(panel.x, y);
      ctx.lineTo(panel.x + panel.w, y);
      ctx.stroke();
    }

    ctx.font = '13px Consolas, monospace';
    ctx.fillStyle = '#8FFFD0';
    const mode = active ? 'RUN' : 'IDLE';
    ctx.fillText(`[${mode}] stav=${uiState}  mic=${inputLevel.toFixed(2)}  out=${outputLevel.toFixed(2)}`, waveRect.x, waveRect.y - 8);

    ctx.fillStyle = 'rgba(6,12,10,0.68)';
    ctx.fillRect(waveRect.x, waveRect.y, waveRect.w, waveRect.h);

    const logWidth = 92;
    const traceRect = { x: waveRect.x, y: waveRect.y, w: waveRect.w - logWidth, h: waveRect.h };
    const logRect = { x: traceRect.x + traceRect.w + 10, y: waveRect.y, w: logWidth - 10, h: waveRect.h };

    const baseline = traceRect.y + traceRect.h / 2;
    ctx.strokeStyle = 'rgba(56,115,86,0.4)';
    ctx.beginPath();
    ctx.moveTo(traceRect.x, baseline);
    ctx.lineTo(traceRect.x + traceRect.w, baseline);
    ctx.stroke();

    const verticalMargin = 14;
    const ampScale = Math.max(1, traceRect.h / 2 - verticalMargin);
    const step = traceRect.w / Math.max(1, this.history.length - 1);

    ctx.lineCap = 'round';
    ctx.lineWidth = 8;
    ctx.strokeStyle = 'rgba(34,255,153,0.24)';
    ctx.beginPath();
    this.history.forEach((value, index) => {
      const x = traceRect.x + index * step;
      const y = baseline - value * ampScale;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.lineWidth = 2.3;
    ctx.strokeStyle = '#7CFF8D';
    ctx.beginPath();
    this.history.forEach((value, index) => {
      const x = traceRect.x + index * step;
      const y = baseline - value * ampScale;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const writerX = traceRect.x + traceRect.w - 2;
    const tipY = baseline - this.history[this.history.length - 1] * ampScale;
    const paperGlow = ctx.createLinearGradient(writerX - 42, 0, writerX + 6, 0);
    paperGlow.addColorStop(0, 'rgba(0,255,170,0)');
    paperGlow.addColorStop(0.72, 'rgba(130,255,180,0.14)');
    paperGlow.addColorStop(1, 'rgba(0,255,170,0)');
    ctx.fillStyle = paperGlow;
    ctx.fillRect(traceRect.x, traceRect.y, traceRect.w, traceRect.h);
    ctx.strokeStyle = '#84FFD0';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(writerX, traceRect.y + 10);
    ctx.lineTo(writerX, traceRect.y + traceRect.h - 10);
    ctx.stroke();
    ctx.strokeStyle = '#D6FFF0';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(writerX + 10, traceRect.y + 20);
    ctx.lineTo(writerX + 10, tipY - 6);
    ctx.lineTo(writerX, tipY);
    ctx.stroke();

    ctx.fillStyle = '#B9FFD5';
    ctx.beginPath();
    ctx.moveTo(logRect.x + 8, baseline);
    ctx.lineTo(logRect.x + 20, baseline - 7);
    ctx.lineTo(logRect.x + 20, baseline + 7);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = 'rgba(8,20,16,0.76)';
    ctx.fillRect(logRect.x, logRect.y, logRect.w, logRect.h);
    ctx.fillStyle = '#C9FFE4';
    ctx.font = '12px Consolas, monospace';
    this.amplitudeLog.forEach((entry, index) => {
      ctx.fillText(entry, logRect.x + 10, logRect.y + 24 + index * 18);
    });
  }
}

function appendTerminalLine(text) {
  const line = (text || '').trim();
  if (!line) return;
  state.terminalLines.push(line);
  state.terminalLines = state.terminalLines.slice(-12);
  ui.terminalLog.textContent = state.terminalLines.join('\n');
  ui.terminalLog.scrollTop = ui.terminalLog.scrollHeight;
}

function setStatus(text) {
  ui.statusPill.textContent = text;
}

function setUiState(nextState) {
  state.uiState = nextState;
  const labels = {
    idle: 'Připraveno',
    connecting: 'Navazuji WebRTC relaci.',
    listening: 'Naslouchám.',
    thinking: 'Přemýšlím.',
    speaking: 'Mluvím.',
    reconnecting: 'Obnovuji spojení.',
    error: 'Došlo k chybě.',
  };
  setStatus(labels[nextState] || 'Připraveno');
}

function renderGuardPanel() {
  const constraints = state.constraintsSummary || { aec: 'on', ns: 'on', agc: 'on' };
  const lines = [
    'GUARD: browser WebRTC pipeline je aktivní',
    `stav=${state.uiState}  mic=${state.inputLevel.toFixed(2)}  out=${state.outputLevel.toFixed(2)}  dc=${state.dataChannelState}  pc=${state.peerState}`,
    `aec=req/${constraints.aec}  ns=req/${constraints.ns}  agc=req/${constraints.agc}  noise=${state.noiseReductionMode}  vad=semantic_low`,
    `topologie=${state.audioTopology}  event=${state.lastEventType}`,
  ];
  ui.guardPanel.textContent = lines.join('\n');
}

function inferAudioTopology(devices) {
  const labels = devices.map((item) => `${item.label || ''} ${item.kind || ''}`.toLowerCase()).join(' | ');
  if (/headset|headphone|airpods|buds|earbud/.test(labels)) return 'wired_headset';
  if (/bluetooth/.test(labels)) return 'bluetooth_headset';
  return 'notebook_builtin';
}

function detectNoiseReductionMode(topology) {
  return ['wired_headset', 'bluetooth_headset', 'headset', 'headphones', 'external_headphones'].includes(topology)
    ? 'near_field'
    : 'far_field';
}

function audioTrackConstraints() {
  return {
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  };
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail || payload.error || detail;
    } catch (_) {
      // ignore
    }
    throw new Error(detail);
  }
  return response.status === 204 ? {} : response.json();
}

async function loadConfig() {
  state.config = await fetchJson('/api/config', { method: 'GET' });
  ui.apiKeyInput.value = '';
  ui.answerLanguageMode.value = state.config.answer_language_mode || 'follow_input';
  ui.fixedAnswerLanguage.value = state.config.fixed_answer_language || 'cs';
  ui.responseStyle.value = state.config.response_style || 'normální';
  ui.realtimeVoice.value = state.config.realtime_voice || 'marin';
  appendTerminalLine(state.config.has_api_key ? 'SYS: OpenAI API klíč je uložený v lokálním backendu.' : 'SYS: Chybí OpenAI API klíč. Zadejte ho nahoře a uložte.');
}

async function saveApiKey() {
  const apiKey = ui.apiKeyInput.value.trim();
  if (!apiKey) throw new Error('Nejdřív zadej OpenAI API klíč.');
  await fetchJson('/api/settings/api-key', { method: 'POST', body: JSON.stringify({ api_key: apiKey }) });
  ui.apiKeyInput.value = '';
  state.config.has_api_key = true;
  appendTerminalLine('SYS: OpenAI API klíč byl bezpečně uložen lokálně v backendu.');
  setStatus('API klíč uložen.');
}

async function deleteApiKey() {
  await fetchJson('/api/settings/api-key', { method: 'DELETE' });
  state.config.has_api_key = false;
  ui.apiKeyInput.value = '';
  appendTerminalLine('SYS: OpenAI API klíč byl smazán.');
  setStatus('API klíč byl smazán.');
}

async function savePreferences() {
  const payload = {
    answer_language_mode: ui.answerLanguageMode.value,
    fixed_answer_language: ui.fixedAnswerLanguage.value,
    response_style: ui.responseStyle.value,
    realtime_voice: ui.realtimeVoice.value,
  };
  state.config = await fetchJson('/api/settings/preferences', { method: 'POST', body: JSON.stringify(payload) });
  appendTerminalLine('SYS: Nastavení bylo uloženo.');
  setStatus('Nastavení bylo uloženo.');
  ui.settingsDialog.close();
}

function ensureAudioContext() {
  if (!state.audioContext) {
    state.audioContext = new AudioContext();
  }
  return state.audioContext;
}

function computeLevel(analyser) {
  if (!analyser) return 0;
  const data = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i += 1) {
    const centered = (data[i] - 128) / 128;
    sum += centered * centered;
  }
  return Math.min(1, Math.sqrt(sum / data.length) * 3.2);
}

function ensureAnalyserForStream(stream, target) {
  const audioContext = ensureAudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.72;
  source.connect(analyser);
  if (target === 'mic') {
    state.localSource = source;
    state.micAnalyser = analyser;
  } else {
    state.remoteSource = source;
    state.remoteAnalyser = analyser;
  }
}

function releaseAnalyser(target) {
  try {
    if (target === 'mic') {
      state.localSource?.disconnect();
    } else {
      state.remoteSource?.disconnect();
    }
  } catch (_) {
    // ignore
  }
  if (target === 'mic') {
    state.localSource = null;
    state.micAnalyser = null;
  } else {
    state.remoteSource = null;
    state.remoteAnalyser = null;
  }
}

async function gatherBrowserAudioSelftest() {
  const stream = await navigator.mediaDevices.getUserMedia(audioTrackConstraints());
  const devices = await navigator.mediaDevices.enumerateDevices();
  const track = stream.getAudioTracks()[0];
  const settings = typeof track.getSettings === 'function' ? track.getSettings() : {};
  const ctx = ensureAudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);

  const startedAt = performance.now();
  let peak = 0;
  while (performance.now() - startedAt < 900) {
    peak = Math.max(peak, computeLevel(analyser));
    await new Promise((resolve) => setTimeout(resolve, 80));
  }

  source.disconnect();
  stream.getTracks().forEach((item) => item.stop());
  const topology = inferAudioTopology(devices);
  return {
    topology,
    settings,
    deviceCount: devices.length,
    peak,
  };
}

function showReport(title, body) {
  ui.reportTitle.textContent = title;
  ui.reportBody.textContent = body;
  ui.reportDialog.showModal();
}

async function runAudioSelftest() {
  setStatus('Probíhá ruční audio test.');
  appendTerminalLine('SELFTEST START [manual_button]');
  const browser = await gatherBrowserAudioSelftest();
  const runtime = await fetchJson('/api/selftest/runtime', {
    method: 'POST',
    body: JSON.stringify({
      audio_topology: browser.topology,
      browser_language: navigator.language,
    }),
  });

  const lines = [
    `BROWSER: topologie=${browser.topology}`,
    `BROWSER: zařízení=${browser.deviceCount}`,
    `BROWSER: mic_peak=${browser.peak.toFixed(3)}`,
    `BROWSER: aec=${browser.settings.echoCancellation ?? 'unknown'}  ns=${browser.settings.noiseSuppression ?? 'unknown'}  agc=${browser.settings.autoGainControl ?? 'unknown'}`,
    `BROWSER: sampleRate=${browser.settings.sampleRate ?? 'unknown'}  channelCount=${browser.settings.channelCount ?? 'unknown'}`,
    '',
    ...runtime.checks.map((item) => `${item.ok ? 'OK' : 'FAIL'} ${item.name}: ${item.detail}`),
  ];

  appendTerminalLine('SELFTEST OK: browser capture a runtime kontrola proběhly.');
  showReport('Audio test', lines.join('\n'));
  setStatus('Audio test doběhl.');
}

function currentAssistantPlaybackMs() {
  if (!state.lastAssistantItemId) return 0;
  const current = Number(ui.remoteAudio.currentTime || 0);
  const started = Number(state.lastAssistantAudioStartSec || 0);
  return Math.max(1, Math.round((current - started) * 1000));
}

function muteAssistantPlayback(shouldMute) {
  state.bargeMuted = Boolean(shouldMute);
  ui.remoteAudio.muted = state.bargeMuted;
}

function maybeSendTruncate() {
  if (!state.dc || state.dc.readyState !== 'open' || !state.lastAssistantItemId) return;
  const payload = {
    type: 'conversation.item.truncate',
    item_id: state.lastAssistantItemId,
    content_index: Number(state.lastAssistantContentIndex || 0),
    audio_end_ms: currentAssistantPlaybackMs(),
  };
  state.dc.send(JSON.stringify(payload));
  appendTerminalLine(`SYS: Barge-in truncate sent for ${state.lastAssistantItemId} @ ${payload.audio_end_ms} ms.`);
  state.lastAssistantItemId = null;
}

function handleRealtimeEvent(event) {
  const type = event?.type || '-';
  state.lastEventType = type;

  if (type === 'session.created') {
    setUiState('listening');
  }
  if (type === 'input_audio_buffer.speech_started') {
    state.userSpeaking = true;
    setUiState('listening');
    if (state.assistantSpeaking) {
      muteAssistantPlayback(true);
      maybeSendTruncate();
    }
  }
  if (type === 'input_audio_buffer.speech_stopped') {
    state.userSpeaking = false;
    if (!state.assistantSpeaking) setUiState('thinking');
    muteAssistantPlayback(false);
  }
  if (type === 'conversation.item.input_audio_transcription.completed') {
    const transcript = (event.transcript || '').trim();
    if (transcript) {
      state.lastUserTranscript = transcript;
      appendTerminalLine(`UŽIVATEL: ${transcript}`);
    }
  }
  if (type === 'response.output_item.added' && event.item?.role === 'assistant') {
    state.lastAssistantItemId = event.item.id || null;
    state.lastAssistantAudioStartSec = Number(ui.remoteAudio.currentTime || 0);
  }
  if (['response.output_audio.delta', 'response.audio.delta'].includes(type)) {
    state.assistantSpeaking = true;
    state.lastAssistantItemId = event.item_id || state.lastAssistantItemId;
    state.lastAssistantContentIndex = Number(event.content_index || 0);
    if (!state.userSpeaking) {
      muteAssistantPlayback(false);
    }
    setUiState('speaking');
  }
  if (['response.output_audio_transcript.done', 'response.output_text.done', 'response.text.done'].includes(type)) {
    const transcript = (event.transcript || event.text || '').trim();
    if (transcript) {
      state.lastAssistantTranscript = transcript;
      appendTerminalLine(`KÁJA: ${transcript}`);
    }
  }
  if (type === 'response.done' || type === 'response.output_audio.done' || type === 'response.audio.done' || type === 'response.cancelled') {
    state.assistantSpeaking = false;
    state.lastAssistantItemId = null;
    state.lastAssistantContentIndex = 0;
    if (!state.userSpeaking) setUiState('listening');
  }
  if (type === 'error') {
    const detail = event?.error?.message || 'Neznámá realtime chyba.';
    appendTerminalLine(`ERR: ${detail}`);
    setUiState('error');
  }
  renderGuardPanel();
}

async function connectRealtime() {
  if (state.sessionActive) {
    await stopSession({ silent: false });
    return;
  }

  if (!state.config?.has_api_key && ui.apiKeyInput.value.trim()) {
    await saveApiKey();
  }
  if (!state.config?.has_api_key) {
    throw new Error('Nejdřív ulož OpenAI API klíč.');
  }

  setUiState('connecting');
  appendTerminalLine('SYS: Hands-free relace se spouští přes browser WebRTC.');

  const stream = await navigator.mediaDevices.getUserMedia(audioTrackConstraints());
  const devices = await navigator.mediaDevices.enumerateDevices();
  const track = stream.getAudioTracks()[0];
  const settings = typeof track.getSettings === 'function' ? track.getSettings() : {};
  state.micTrackSettings = settings;
  state.constraintsSummary = {
    aec: settings.echoCancellation === false ? 'off' : 'on',
    ns: settings.noiseSuppression === false ? 'off' : 'on',
    agc: settings.autoGainControl === false ? 'off' : 'on',
  };
  state.audioTopology = inferAudioTopology(devices);
  state.noiseReductionMode = detectNoiseReductionMode(state.audioTopology);

  const tokenResponse = await fetchJson('/api/realtime/client-secret', {
    method: 'POST',
    body: JSON.stringify({
      audio_topology: state.audioTopology,
      browser_language: navigator.language,
    }),
  });
  const ephemeralKey = tokenResponse.client_secret?.value || tokenResponse.value;
  if (!ephemeralKey) throw new Error('Server nevrátil dočasný Realtime client secret.');

  const pc = new RTCPeerConnection();
  const dc = pc.createDataChannel('oai-events');
  state.pc = pc;
  state.dc = dc;
  state.localStream = stream;
  state.sessionActive = true;
  ui.startStopBtn.textContent = 'Stop';

  stream.getTracks().forEach((mediaTrack) => {
    pc.addTrack(mediaTrack, stream);
  });
  ensureAnalyserForStream(stream, 'mic');

  pc.onconnectionstatechange = () => {
    state.peerState = pc.connectionState || pc.iceConnectionState || 'unknown';
    if (['failed', 'disconnected', 'closed'].includes(state.peerState)) {
      setUiState(state.peerState === 'closed' ? 'idle' : 'reconnecting');
    }
    renderGuardPanel();
  };
  pc.oniceconnectionstatechange = () => {
    state.peerState = pc.iceConnectionState || state.peerState;
    renderGuardPanel();
  };
  pc.ontrack = (evt) => {
    const [remoteStream] = evt.streams;
    state.remoteStream = remoteStream;
    ui.remoteAudio.srcObject = remoteStream;
    ensureAnalyserForStream(remoteStream, 'remote');
    ui.remoteAudio.play().catch(() => {});
  };

  dc.addEventListener('open', () => {
    state.dataChannelState = 'open';
    renderGuardPanel();
  });
  dc.addEventListener('close', () => {
    state.dataChannelState = 'closed';
    renderGuardPanel();
  });
  dc.addEventListener('message', (evt) => {
    try {
      handleRealtimeEvent(JSON.parse(evt.data));
    } catch (_) {
      // ignore malformed payload
    }
  });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const sdpResponse = await fetch('https://api.openai.com/v1/realtime/calls', {
    method: 'POST',
    body: offer.sdp,
    headers: {
      Authorization: `Bearer ${ephemeralKey}`,
      'Content-Type': 'application/sdp',
    },
  });
  if (!sdpResponse.ok) {
    let detail = `Realtime call failed (${sdpResponse.status})`;
    try {
      const payload = await sdpResponse.json();
      detail = payload?.error?.message || detail;
    } catch (_) {
      // ignore
    }
    throw new Error(detail);
  }
  const answer = { type: 'answer', sdp: await sdpResponse.text() };
  await pc.setRemoteDescription(answer);

  setUiState('listening');
  renderGuardPanel();
}

async function stopSession({ silent = true } = {}) {
  state.sessionActive = false;
  state.userSpeaking = false;
  state.assistantSpeaking = false;
  state.lastAssistantItemId = null;
  state.lastAssistantContentIndex = 0;
  state.lastAssistantAudioStartSec = 0;
  ui.startStopBtn.textContent = 'Start';
  muteAssistantPlayback(false);
  try { state.dc?.close(); } catch (_) {}
  try { state.pc?.close(); } catch (_) {}
  if (state.localStream) {
    state.localStream.getTracks().forEach((item) => item.stop());
  }
  ui.remoteAudio.srcObject = null;
  releaseAnalyser('mic');
  releaseAnalyser('remote');
  state.localStream = null;
  state.remoteStream = null;
  state.pc = null;
  state.dc = null;
  state.peerState = 'closed';
  state.dataChannelState = 'closed';
  setUiState('idle');
  renderGuardPanel();
  if (!silent) appendTerminalLine('SYS: Hlasový chat byl zastaven.');
}

function startRenderLoop() {
  const loop = () => {
    state.inputLevel = computeLevel(state.micAnalyser);
    state.outputLevel = state.bargeMuted ? 0 : computeLevel(state.remoteAnalyser);
    state.visualizer.tick(state.inputLevel, state.outputLevel, state.uiState, state.sessionActive);
    renderGuardPanel();
    state.animationHandle = requestAnimationFrame(loop);
  };
  state.animationHandle = requestAnimationFrame(loop);
}

function clearSessionUi() {
  state.terminalLines = ['[READY] Terminál čeká na relaci.'];
  ui.terminalLog.textContent = state.terminalLines.join('\n');
}

async function shutdownApp() {
  await fetchJson('/api/shutdown', { method: 'POST' });
  window.close();
}

async function bootstrap() {
  state.visualizer = new EkgVisualizer(ui.canvas);
  clearSessionUi();
  await loadConfig();
  renderGuardPanel();
  appendTerminalLine('SYS: Browser frontend používá WebRTC capture s echoCancellation/noiseSuppression/autoGainControl a Realtime client secret tokenem.');
  startRenderLoop();
}

ui.saveKeyBtn.addEventListener('click', () => saveApiKey().catch((err) => showReport('Chyba', err.message)));
ui.deleteKeyBtn.addEventListener('click', () => deleteApiKey().catch((err) => showReport('Chyba', err.message)));
ui.audioTestBtn.addEventListener('click', () => runAudioSelftest().catch((err) => showReport('Audio test selhal', err.message)));
ui.startStopBtn.addEventListener('click', () => connectRealtime().catch(async (err) => {
  await stopSession({ silent: true });
  showReport('Relace selhala', err.message);
  appendTerminalLine(`ERR: ${err.message}`);
  setUiState('error');
}));
ui.settingsBtn.addEventListener('click', () => ui.settingsDialog.showModal());
ui.settingsApplyBtn.addEventListener('click', () => savePreferences().catch((err) => showReport('Chyba', err.message)));
ui.savePrefsBtn.addEventListener('click', () => savePreferences().catch((err) => showReport('Chyba', err.message)));
ui.clearBtn.addEventListener('click', () => stopSession({ silent: true }).then(clearSessionUi));
ui.exitBtn.addEventListener('click', () => shutdownApp().catch((err) => showReport('Chyba', err.message)));
ui.reportCloseBtn.addEventListener('click', () => ui.reportDialog.close());
window.addEventListener('beforeunload', () => {
  if (state.animationHandle) cancelAnimationFrame(state.animationHandle);
  stopSession({ silent: true });
});

bootstrap().catch((err) => {
  showReport('Inicializace selhala', err.message);
});
