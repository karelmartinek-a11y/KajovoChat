const state = {
  appState: null,
  pc: null,
  dc: null,
  localStream: null,
  remoteStream: null,
  audioContext: null,
  micAnalyser: null,
  spkAnalyser: null,
  animationFrame: null,
  currentAssistantItemId: null,
  currentAssistantText: "",
  currentUserText: "",
  logItems: new Map(),
  handledToolCalls: new Set(),
  pendingRestartReason: "",
  isApplyingSetting: false,
};

const els = {
  apiKeyInput: document.getElementById("apiKeyInput"),
  saveKeyBtn: document.getElementById("saveKeyBtn"),
  deleteKeyBtn: document.getElementById("deleteKeyBtn"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  audioTestBtn: document.getElementById("audioTestBtn"),
  statusLabel: document.getElementById("statusLabel"),
  statusDot: document.getElementById("statusDot"),
  statusNote: document.getElementById("statusNote"),
  log: document.getElementById("conversationLog"),
  selftestLog: document.getElementById("selftestLog"),
  localState: document.getElementById("localState"),
  orb: document.getElementById("orb"),
  micMeter: document.getElementById("micMeter"),
  spkMeter: document.getElementById("spkMeter"),
  micValue: document.getElementById("micValue"),
  spkValue: document.getElementById("spkValue"),
  remoteAudio: document.getElementById("remoteAudio"),
  voiceChoices: document.getElementById("voiceChoices"),
  languageModeChoices: document.getElementById("languageModeChoices"),
  languageChoices: document.getElementById("languageChoices"),
  styleChoices: document.getElementById("styleChoices"),
  lengthChoices: document.getElementById("lengthChoices"),
  formalityChoices: document.getElementById("formalityChoices"),
};

const TOOL_NAMES = {
  runProgram: "spust_program",
  setVoice: "nastav_hlas",
  setLanguage: "nastav_jazyk_odpovedi",
  setStyle: "nastav_styl_odpovedi",
  setLength: "nastav_delku_odpovedi",
  setFormality: "nastav_formalnost_odpovedi",
};

function setStatus(label, note = "", dotColor = "#FCD237") {
  els.statusLabel.textContent = label;
  els.statusNote.textContent = note || "";
  els.statusDot.style.background = dotColor;
}

function appendLog(kind, title, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `log-entry ${kind}`;
  const titleEl = document.createElement("small");
  titleEl.textContent = title;
  const textEl = document.createElement("div");
  textEl.textContent = text;
  wrapper.append(titleEl, textEl);
  els.log.appendChild(wrapper);
  els.log.scrollTop = els.log.scrollHeight;
  return textEl;
}

function upsertLog(mapKey, kind, title, text) {
  let textEl = state.logItems.get(mapKey);
  if (!textEl) {
    textEl = appendLog(kind, title, text);
    state.logItems.set(mapKey, textEl);
    return;
  }
  textEl.textContent = text;
  els.log.scrollTop = els.log.scrollHeight;
}

function showError(message) {
  appendLog("error", "Chyba", message);
  setStatus("Chyba", message, "#d55");
}

function setButtons(isRunning) {
  els.startBtn.disabled = isRunning || state.isApplyingSetting;
  els.stopBtn.disabled = !isRunning || state.isApplyingSetting;
}

function setChoiceButtonsDisabled(disabled) {
  document.querySelectorAll(".choice-btn").forEach((button) => {
    button.disabled = disabled;
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson ? await response.json() : await response.text();
  if (!response.ok) {
    const message = isJson ? (payload.error || "Požadavek selhal.") : String(payload || response.statusText);
    throw new Error(message);
  }
  return payload;
}

function isConversationRunning() {
  return !!(state.pc && state.dc && state.dc.readyState === "open");
}

async function loadState() {
  const payload = await fetchJson("/api/settings");
  state.appState = payload.state;
  renderSettingsPanels();
  renderLocalState();
}

function renderLocalState() {
  const languageValue = state.appState?.answer_language_mode === "fixed"
    ? state.appState.fixed_answer_language_label
    : "podle uživatele";
  const rows = [
    ["OpenAI API klíč", state.appState?.has_openai_api_key ? "uložen" : "chybí"],
    ["Hlas", state.appState?.voice_label || state.appState?.voice || "marin"],
    ["Jazyk odpovědí", languageValue],
    ["Styl odpovědi", state.appState?.response_style || "normální"],
    ["Délka odpovědi", state.appState?.response_length || "střední"],
    ["Formálnost", state.appState?.response_formality || "neutrální"],
    ["Model", "gpt-realtime"],
    ["Voice transport", "WebRTC v browseru"],
  ];
  els.localState.innerHTML = rows.map(([k, v]) => `<span>${k}</span><strong>${v}</strong>`).join("");
  if (state.appState?.has_openai_api_key) {
    els.apiKeyInput.placeholder = "OpenAI API klíč je uložen";
  }
}

function renderChoiceGroup(container, options, activeValue, onClick, isVisible = true) {
  if (!container) {
    return;
  }
  container.innerHTML = "";
  container.style.display = isVisible ? "flex" : "none";
  if (!isVisible) {
    return;
  }
  options.forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `choice-btn${activeValue === option.value ? " active" : ""}`;
    button.textContent = option.label;
    button.addEventListener("click", () => onClick(option.value));
    container.appendChild(button);
  });
}

function renderSettingsPanels() {
  const options = state.appState?.options || {};
  renderChoiceGroup(els.voiceChoices, options.voices || [], state.appState?.voice, (value) => {
    applyPreferenceChange({ voice: value }, `Hlas je nastavený na ${value}.`).catch((error) => showError(error.message || String(error)));
  });
  renderChoiceGroup(els.languageModeChoices, options.language_modes || [], state.appState?.answer_language_mode, (value) => {
    const payload = { answer_language_mode: value };
    if (value === "fixed" && !state.appState?.fixed_answer_language) {
      payload.fixed_answer_language = "cs";
    }
    applyPreferenceChange(payload, value === "fixed" ? "Jazyk odpovědí bude pevně nastavený." : "Jazyk odpovědí se bude řídit uživatelem.")
      .catch((error) => showError(error.message || String(error)));
  });
  renderChoiceGroup(els.languageChoices, options.languages || [], state.appState?.fixed_answer_language, (value) => {
    applyPreferenceChange({ answer_language_mode: "fixed", fixed_answer_language: value }, `Jazyk odpovědí je nastavený na ${value}.`)
      .catch((error) => showError(error.message || String(error)));
  }, state.appState?.answer_language_mode === "fixed");
  renderChoiceGroup(els.styleChoices, options.response_styles || [], state.appState?.response_style, (value) => {
    applyPreferenceChange({ response_style: value }, `Styl odpovědí je nastavený na ${value}.`).catch((error) => showError(error.message || String(error)));
  });
  renderChoiceGroup(els.lengthChoices, options.response_lengths || [], state.appState?.response_length, (value) => {
    applyPreferenceChange({ response_length: value }, `Délka odpovědí je nastavená na ${value}.`).catch((error) => showError(error.message || String(error)));
  });
  renderChoiceGroup(els.formalityChoices, options.response_formalities || [], state.appState?.response_formality, (value) => {
    applyPreferenceChange({ response_formality: value }, `Formálnost odpovědí je nastavená na ${value}.`).catch((error) => showError(error.message || String(error)));
  });
  setChoiceButtonsDisabled(state.isApplyingSetting);
}

async function saveApiKey() {
  const apiKey = els.apiKeyInput.value.trim();
  if (!apiKey) {
    showError("Zadej OpenAI API klíč.");
    return;
  }
  const payload = await fetchJson("/api/settings/openai-key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });
  state.appState = payload.state;
  renderSettingsPanels();
  renderLocalState();
  els.apiKeyInput.value = "";
  appendLog("system", "Nastavení", "OpenAI API klíč byl uložen lokálně.");
}

async function deleteApiKey() {
  const payload = await fetchJson("/api/settings/delete-openai-key", { method: "POST" });
  state.appState = payload.state;
  renderSettingsPanels();
  renderLocalState();
  appendLog("system", "Nastavení", "OpenAI API klíč byl odstraněn.");
}

async function applyPreferenceChange(payload, successMessage = "Nastavení bylo uloženo.", source = "ui", options = {}) {
  const deferRestart = options.deferRestart === true;
  state.isApplyingSetting = true;
  setButtons(isConversationRunning());
  setChoiceButtonsDisabled(true);
  try {
    const result = await fetchJson("/api/settings/preferences", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.appState = result.state;
    renderSettingsPanels();
    renderLocalState();
    appendLog("system", source === "tool" ? "Nástroj" : "Nastavení", result.message || successMessage);
    if (result.restart_required && isConversationRunning() && !deferRestart) {
      await restartConversation("Nastavení relace bylo změněno. Obnovuji spojení s novou konfigurací.");
    }
    return result;
  } finally {
    state.isApplyingSetting = false;
    setButtons(isConversationRunning());
    setChoiceButtonsDisabled(false);
  }
}

function inferAcousticProfile(devices) {
  const labels = devices.map((device) => (device.label || "").toLowerCase()).join(" ");
  return /(headset|headphone|sluch|airpods|buds|bluetooth)/.test(labels) ? "near_field" : "far_field";
}

function readTrackSettings(track) {
  if (!track || typeof track.getSettings !== "function") {
    return {};
  }
  const settings = track.getSettings();
  return {
    echoCancellation: settings.echoCancellation,
    noiseSuppression: settings.noiseSuppression,
    autoGainControl: settings.autoGainControl,
    channelCount: settings.channelCount,
    sampleRate: settings.sampleRate,
    deviceId: settings.deviceId,
  };
}

function closeMedia() {
  if (state.animationFrame) {
    cancelAnimationFrame(state.animationFrame);
    state.animationFrame = null;
  }
  if (state.localStream) {
    state.localStream.getTracks().forEach((track) => track.stop());
    state.localStream = null;
  }
  if (state.remoteStream) {
    state.remoteStream.getTracks().forEach((track) => track.stop());
    state.remoteStream = null;
  }
  if (state.pc) {
    state.pc.close();
    state.pc = null;
  }
  state.dc = null;
  state.handledToolCalls.clear();
  if (state.audioContext) {
    state.audioContext.close().catch(() => {});
    state.audioContext = null;
  }
  state.micAnalyser = null;
  state.spkAnalyser = null;
  els.remoteAudio.srcObject = null;
  setButtons(false);
  setStatus("Neaktivní", "Relace je zastavená.", "#FCD237");
  updateMeters(0, 0);
}

function updateMeters(mic, spk) {
  const micPct = Math.round(mic * 100);
  const spkPct = Math.round(spk * 100);
  els.micMeter.style.width = `${micPct}%`;
  els.spkMeter.style.width = `${spkPct}%`;
  els.micValue.textContent = `${micPct}%`;
  els.spkValue.textContent = `${spkPct}%`;
  els.orb.style.setProperty("--mic-scale", (mic * 0.12).toFixed(3));
  els.orb.style.setProperty("--mic-level", `${Math.round(mic * 22)}px`);
  els.orb.style.setProperty("--spk-level", `${Math.round(spk * 28)}px`);
}

function analyserLevel(analyser) {
  if (!analyser) {
    return 0;
  }
  const data = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(data);
  let sumSquares = 0;
  for (let i = 0; i < data.length; i += 1) {
    const normalized = (data[i] - 128) / 128;
    sumSquares += normalized * normalized;
  }
  return Math.min(1, Math.sqrt(sumSquares / data.length) * 3.4);
}

function startMeters() {
  const tick = () => {
    updateMeters(analyserLevel(state.micAnalyser), analyserLevel(state.spkAnalyser));
    state.animationFrame = requestAnimationFrame(tick);
  };
  tick();
}

function bindRemoteStream(stream) {
  state.remoteStream = stream;
  els.remoteAudio.srcObject = stream;
  if (!state.audioContext) {
    state.audioContext = new AudioContext();
  }
  const src = state.audioContext.createMediaStreamSource(stream);
  const analyser = state.audioContext.createAnalyser();
  analyser.fftSize = 1024;
  src.connect(analyser);
  state.spkAnalyser = analyser;
}

function bindLocalMeter(stream) {
  if (!state.audioContext) {
    state.audioContext = new AudioContext();
  }
  const src = state.audioContext.createMediaStreamSource(stream);
  const analyser = state.audioContext.createAnalyser();
  analyser.fftSize = 1024;
  src.connect(analyser);
  state.micAnalyser = analyser;
}

function canSendRealtime() {
  return Boolean(state.dc && state.dc.readyState === "open");
}

function sendRealtime(payload) {
  if (!canSendRealtime()) {
    return false;
  }
  state.dc.send(JSON.stringify(payload));
  return true;
}

async function executeToolCall(name, args) {
  if (name === TOOL_NAMES.runProgram) {
    const payload = await fetchJson("/api/tools/run-program", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ program: args.program }),
    });
    return { ...payload, restart_required: false };
  }
  if (name === TOOL_NAMES.setVoice) {
    return applyPreferenceChange({ voice: args.hlas }, `Hlas je nastavený na ${args.hlas}.`, "tool", { deferRestart: true });
  }
  if (name === TOOL_NAMES.setLanguage) {
    const payload = { answer_language_mode: args.rezim };
    if (args.rezim === "fixed") {
      payload.fixed_answer_language = args.jazyk;
    }
    return applyPreferenceChange(payload, "Jazyk odpovědí byl změněn.", "tool", { deferRestart: true });
  }
  if (name === TOOL_NAMES.setStyle) {
    return applyPreferenceChange({ response_style: args.styl }, `Styl odpovědí je nastavený na ${args.styl}.`, "tool", { deferRestart: true });
  }
  if (name === TOOL_NAMES.setLength) {
    return applyPreferenceChange({ response_length: args.delka }, `Délka odpovědí je nastavená na ${args.delka}.`, "tool", { deferRestart: true });
  }
  if (name === TOOL_NAMES.setFormality) {
    return applyPreferenceChange({ response_formality: args.formalnost }, `Formálnost odpovědí je nastavená na ${args.formalnost}.`, "tool", { deferRestart: true });
  }
  throw new Error(`Nepodporovaný nástroj: ${name}`);
}

async function handleToolCallDone(event) {
  if (!canSendRealtime()) {
    throw new Error("Datový kanál není připravený pro tool output.");
  }
  if (!event.call_id || state.handledToolCalls.has(event.call_id)) {
    return;
  }
  state.handledToolCalls.add(event.call_id);

  let args = {};
  try {
    args = JSON.parse(event.arguments || "{}");
  } catch (error) {
    throw new Error("Model poslal neplatné argumenty nástroje.");
  }

  appendLog("system", "Nástroj", `Model volá ${event.name}(${JSON.stringify(args)})`);
  const result = await executeToolCall(event.name, args);
  const output = result.result || result;
  appendLog("system", "Nástroj", output.message || result.message || "Nástroj byl proveden.");

  if (!sendRealtime({
    type: "conversation.item.create",
    item: {
      type: "function_call_output",
      call_id: event.call_id,
      output: JSON.stringify(output),
    },
  })) {
    throw new Error("Datový kanál se uzavřel před odesláním výsledku nástroje.");
  }

  if (result.restart_required) {
    appendLog("system", "Relace", "Nastavení vyžaduje restart relace. Přepojuji voice chat.");
    setTimeout(() => {
      restartConversation("Obnovuji hlasovou relaci s novým nastavením.").catch((error) => showError(error.message || String(error)));
    }, 250);
    return;
  }
  if (!sendRealtime({ type: "response.create" })) {
    throw new Error("Datový kanál se uzavřel před požadavkem na novou odpověď.");
  }
}

function handleRealtimeEvent(event) {
  const type = event.type;
  if (!type) {
    return;
  }
  if (type === "session.created") {
    appendLog("system", "Relace", `Relace vytvořena: ${event.session?.model || "gpt-realtime"}`);
    setStatus("Připraveno", "Mluv do mikrofonu. VAD je aktivní.", "#7be495");
    return;
  }
  if (type === "input_audio_buffer.speech_started") {
    setStatus("Naslouchám", "Uživatel mluví.", "#3CB0DB");
    return;
  }
  if (type === "input_audio_buffer.speech_stopped") {
    setStatus("Zpracovávám", "Čekám na odpověď modelu.", "#FCD237");
    return;
  }
  if (type === "conversation.item.input_audio_transcription.delta") {
    state.currentUserText += event.delta || "";
    upsertLog(`user:${event.item_id || "live"}`, "user", "Ty", state.currentUserText.trim() || "…");
    return;
  }
  if (type === "conversation.item.input_audio_transcription.completed") {
    const transcript = event.transcript || state.currentUserText || "";
    upsertLog(`user:${event.item_id || "done"}`, "user", "Ty", transcript.trim() || "…");
    state.currentUserText = "";
    return;
  }
  if (type === "output_audio_buffer.started") {
    setStatus("Mluvím", "Asistent přehrává odpověď.", "#FCD237");
    return;
  }
  if (type === "output_audio_buffer.stopped" || type === "output_audio_buffer.cleared") {
    setStatus("Naslouchám", "Můžeš pokračovat.", "#7be495");
    state.currentAssistantText = "";
    return;
  }
  if (type === "response.output_audio_transcript.delta") {
    state.currentAssistantItemId = event.item_id || state.currentAssistantItemId || "assistant-live";
    state.currentAssistantText += event.delta || "";
    upsertLog(`assistant:${state.currentAssistantItemId}`, "assistant", "Kája", state.currentAssistantText.trim() || "…");
    return;
  }
  if (type === "response.output_audio_transcript.done" || type === "response.output_text.done") {
    const text = (event.transcript || event.text || state.currentAssistantText || "").trim();
    const itemId = event.item_id || state.currentAssistantItemId || "assistant-done";
    upsertLog(`assistant:${itemId}`, "assistant", "Kája", text || "…");
    state.currentAssistantText = "";
    return;
  }
  if (type === "response.function_call_arguments.done") {
    handleToolCallDone(event).catch((error) => {
      showError(error.message || String(error));
      if (event.call_id && sendRealtime({
        type: "conversation.item.create",
        item: {
          type: "function_call_output",
          call_id: event.call_id,
          output: JSON.stringify({ ok: false, error: error.message || String(error) }),
        },
      })) {
        sendRealtime({ type: "response.create" });
      }
    });
    return;
  }
  if (type === "error") {
    showError(event.error?.message || "Neznámá realtime chyba.");
  }
}

async function startConversation() {
  try {
    if (!state.appState?.has_openai_api_key) {
      if (els.apiKeyInput.value.trim()) {
        await saveApiKey();
      } else {
        throw new Error("Nejdřív ulož OpenAI API klíč.");
      }
    }
    setStatus("Připravuji", "Žádám browser o mikrofon a inicializuji WebRTC.", "#3CB0DB");
    const localStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    const devices = await navigator.mediaDevices.enumerateDevices();
    const input = devices.find((device) => device.kind === "audioinput") || null;
    const output = devices.find((device) => device.kind === "audiooutput") || null;

    const pc = new RTCPeerConnection();
    const dc = pc.createDataChannel("oai-events");
    dc.addEventListener("message", (evt) => {
      try {
        handleRealtimeEvent(JSON.parse(evt.data));
      } catch (error) {
        console.warn("Nepodařilo se parsovat realtime event", error);
      }
    });
    pc.addEventListener("track", (evt) => {
      if (evt.streams && evt.streams[0]) {
        bindRemoteStream(evt.streams[0]);
      }
    });
    localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const answer = await fetch("/api/realtime/call", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: offer.sdp,
        acoustic_profile: inferAcousticProfile(devices),
        input_label: input?.label || "",
        output_label: output?.label || "",
        browser_audio: readTrackSettings(localStream.getAudioTracks()[0]),
      }),
    });
    if (!answer.ok) {
      const errorPayload = await answer.json().catch(() => ({}));
      throw new Error(errorPayload.error || "Vytvoření realtime relace selhalo.");
    }
    const answerSdp = await answer.text();
    await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });

    state.pc = pc;
    state.dc = dc;
    state.localStream = localStream;
    bindLocalMeter(localStream);
    startMeters();
    setButtons(true);
    setStatus("Naslouchám", "Spojení je aktivní. Mluv do mikrofonu.", "#7be495");
    appendLog("system", "Relace", "Browserový voice chat byl spuštěn přes WebRTC.");
  } catch (error) {
    closeMedia();
    showError(error.message || String(error));
  }
}

function stopConversation() {
  closeMedia();
  appendLog("system", "Relace", "Relace byla ukončena.");
}

async function restartConversation(reason) {
  const shouldRestart = isConversationRunning();
  if (!shouldRestart) {
    return;
  }
  closeMedia();
  appendLog("system", "Relace", reason || "Obnovuji relaci.");
  await startConversation();
}

async function runAudioTest() {
  const lines = [];
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    const track = stream.getAudioTracks()[0];
    const settings = readTrackSettings(track);
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter((device) => device.kind === "audioinput");
    const outputs = devices.filter((device) => device.kind === "audiooutput");
    lines.push("Browser capture profil:");
    lines.push(`- echoCancellation: ${String(settings.echoCancellation)}`);
    lines.push(`- noiseSuppression: ${String(settings.noiseSuppression)}`);
    lines.push(`- autoGainControl: ${String(settings.autoGainControl)}`);
    lines.push(`- channelCount: ${String(settings.channelCount ?? "?")}`);
    lines.push(`- sampleRate: ${String(settings.sampleRate ?? "?")}`);
    lines.push(`- input devices: ${inputs.length}`);
    lines.push(`- output devices: ${outputs.length}`);
    lines.push(`- acoustic profile: ${inferAcousticProfile(devices)}`);
    if (inputs[0]?.label) {
      lines.push(`- aktivní vstup: ${inputs[0].label}`);
    }
    if (outputs[0]?.label) {
      lines.push(`- výstup: ${outputs[0].label}`);
    }
    stream.getTracks().forEach((t) => t.stop());
  } catch (error) {
    lines.push(`Audio test selhal: ${error.message || String(error)}`);
  }
  els.selftestLog.textContent = lines.join("\n");
}

window.addEventListener("DOMContentLoaded", async () => {
  els.saveKeyBtn.addEventListener("click", () => saveApiKey().catch((error) => showError(error.message || String(error))));
  els.deleteKeyBtn.addEventListener("click", () => deleteApiKey().catch((error) => showError(error.message || String(error))));
  els.startBtn.addEventListener("click", () => startConversation());
  els.stopBtn.addEventListener("click", () => stopConversation());
  els.audioTestBtn.addEventListener("click", () => runAudioTest());
  try {
    await loadState();
    setStatus("Neaktivní", "Připraveno. Po uložení klíče můžeš spustit voice chat.");
    appendLog("system", "Start", "Webové rozhraní je připravené.");
  } catch (error) {
    showError(error.message || String(error));
  }
});
