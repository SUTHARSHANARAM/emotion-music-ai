'use strict';

let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let scriptNode = null;
let analyserNode = null;
let recordedPCM = [];
let isRecording = false;
let autoStopTimer = null;
let animFrameId = null;
let secondsTimerId = null;
let recordStartTime = 0;

const USE_BACKEND = true;
const BACKEND_URL = "http://127.0.0.1:5000/predict";
const EMOTIONS = ["happy", "sad", "angry", "calm", "excited"];

// Navigation Tab Switcher
function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  
  const targetTab = document.getElementById(tabId);
  if (targetTab) targetTab.classList.add('active');

  const btnMap = {
    'recorderTab': 0,
    'analyticsTab': 1,
    'preferencesTab': 2,
    'pipelineTab': 3
  };
  const navBtns = document.querySelectorAll('.nav-item');
  if (navBtns[btnMap[tabId]]) {
    navBtns[btnMap[tabId]].classList.add('active');
  }

  if (tabId === 'analyticsTab') {
    renderAnalyticsDashboard();
  }
}

function setStatus(msg, type = "") {
  const el = document.getElementById("statusText");
  if (!el) return;
  el.textContent = msg;
  el.className = "status-msg " + type;
}

function setButtonsState(recording) {
  const recordBtn = document.getElementById("recordBtn");
  const stopBtn = document.getElementById("stopBtn");
  if (recordBtn) recordBtn.disabled = recording;
  if (stopBtn) stopBtn.disabled = !recording;
}

// -------------------------------------------------------------
// LIVE CANVAS VISUALIZERS (WAVEFORM + SPECTROGRAM WATERFALL)
// -------------------------------------------------------------

function startLiveCanvases() {
  const waveCanvas = document.getElementById("waveformCanvas");
  const specCanvas = document.getElementById("spectrogramCanvas");
  if (!waveCanvas || !specCanvas || !analyserNode) return;

  const waveCtx = waveCanvas.getContext("2d");
  const specCtx = specCanvas.getContext("2d");

  analyserNode.fftSize = 256;
  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  const freqArray = new Uint8Array(bufferLength);

  // Spectrogram Waterfall ImageData
  const specW = specCanvas.width;
  const specH = specCanvas.height;

  function draw() {
    if (!isRecording) return;
    animFrameId = requestAnimationFrame(draw);

    analyserNode.getByteTimeDomainData(dataArray);
    analyserNode.getByteFrequencyData(freqArray);

    // 1. Draw Waveform (Oscilloscope)
    waveCtx.fillStyle = "#000000";
    waveCtx.fillRect(0, 0, waveCanvas.width, waveCanvas.height);
    waveCtx.lineWidth = 2;
    waveCtx.strokeStyle = "#1db954";
    waveCtx.beginPath();

    const sliceWidth = waveCanvas.width / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const ampOffset = (dataArray[i] - 128) * 2.5;
      const y = (waveCanvas.height / 2) + ampOffset;
      if (i === 0) waveCtx.moveTo(x, y);
      else waveCtx.lineTo(x, y);
      x += sliceWidth;
    }
    waveCtx.stroke();

    // 2. Draw Waterfall Spectrogram
    const imgData = specCtx.getImageData(1, 0, specW - 1, specH);
    specCtx.putImageData(imgData, 0, 0);

    for (let yBin = 0; yBin < specH; yBin++) {
      const binIdx = Math.floor((yBin / specH) * bufferLength);
      const val = freqArray[binIdx];
      const r = Math.min(255, val * 1.5);
      const g = val > 128 ? (val - 128) * 2 : 0;
      const b = 255 - val;
      specCtx.fillStyle = `rgb(${r},${g},${b})`;
      specCtx.fillRect(specW - 1, specH - yBin, 1, 1);
    }
  }

  draw();
}

function stopLiveCanvases() {
  if (animFrameId) cancelAnimationFrame(animFrameId);
}

// -------------------------------------------------------------
// RECORDING LOGIC & AUDIO DSP
// -------------------------------------------------------------

async function startRecording() {
  if (isRecording) return;
  try {
    setStatus("Requesting microphone access…", "");

    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    const micSelect = document.getElementById("micSelect");
    const constraints = {
      audio: micSelect && micSelect.value ? { deviceId: { exact: micSelect.value } } : true
    };
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    scriptNode = audioContext.createScriptProcessor(4096, 1, 1);
    recordedPCM = [];

    // Voice Bandpass filter (Highpass at 150Hz + Lowpass at 4000Hz)
    const highpass = audioContext.createBiquadFilter();
    highpass.type = "highpass";
    highpass.frequency.value = 150;

    const lowpass = audioContext.createBiquadFilter();
    lowpass.type = "lowpass";
    lowpass.frequency.value = 4000;

    sourceNode.connect(highpass);
    highpass.connect(lowpass);
    lowpass.connect(analyserNode);
    lowpass.connect(scriptNode);
    scriptNode.connect(audioContext.destination);

    scriptNode.onaudioprocess = (e) => {
      if (!isRecording) return;
      const input = e.inputBuffer.getChannelData(0);
      recordedPCM.push(new Float32Array(input));

      // Mute the output buffer to prevent audio feedback loop through speakers
      const output = e.outputBuffer.getChannelData(0);
      output.fill(0);

      // Calculate Volume & Noise Diagnostics
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      const rms = Math.sqrt(sum / input.length);
      updateDiagnostics(rms);
    };

    isRecording = true;
    setButtonsState(true);
    recordStartTime = Date.now();
    setStatus("🔴 Recording in progress… Speak now!", "ok");

    startLiveCanvases();
    startTimer();

    if (autoStopTimer) clearTimeout(autoStopTimer);
    autoStopTimer = setTimeout(() => {
      if (isRecording) stopRecording();
    }, 8000);

  } catch (err) {
    console.error(err);
    setStatus(`Microphone failed: ${err.message || err}`, "error");
    isRecording = false;
    setButtonsState(false);
  }
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  setButtonsState(false);
  stopTimer();
  stopLiveCanvases();

  if (autoStopTimer) clearTimeout(autoStopTimer);

  if (scriptNode) {
    scriptNode.onaudioprocess = null;
    scriptNode.disconnect();
    scriptNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }

  setStatus("⏳ Processing spectrogram & extracting 13 MFCCs…", "");

  setTimeout(() => {
    let totalSamples = 0;
    for (let i = 0; i < recordedPCM.length; i++) totalSamples += recordedPCM[i].length;

    if (totalSamples === 0) {
      setStatus("No audio recorded. Please try again.", "warn");
      return;
    }

    const mergedPCM = new Float32Array(totalSamples);
    let offset = 0;
    for (let i = 0; i < recordedPCM.length; i++) {
      mergedPCM.set(recordedPCM[i], offset);
      offset += recordedPCM[i].length;
    }

    const audioBuffer = audioContext.createBuffer(1, mergedPCM.length, audioContext.sampleRate);
    audioBuffer.getChannelData(0).set(mergedPCM);

    runEmotionPipeline(audioBuffer);
  }, 20);
}

// -------------------------------------------------------------
// TIMER & DIAGNOSTICS
// -------------------------------------------------------------

function startTimer() {
  const badge = document.getElementById("timerBadge");
  secondsTimerId = setInterval(() => {
    const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
    if (badge) badge.textContent = `00:0${Math.min(8, elapsed)} / 00:08`;
  }, 500);
}

function stopTimer() {
  if (secondsTimerId) clearInterval(secondsTimerId);
}

function updateDiagnostics(rms) {
  const noiseEl = document.getElementById("noiseLevel");
  const volEl = document.getElementById("volumeLevel");
  
  if (volEl) {
    if (rms > 0.3) volEl.textContent = "High / Loud";
    else if (rms > 0.05) volEl.textContent = "Optimal";
    else volEl.textContent = "Soft / Low";
  }

  if (noiseEl) {
    noiseEl.textContent = rms < 0.01 ? "Low Floor" : "Moderate";
  }
}

// -------------------------------------------------------------
// RADIX-2 FFT & SPECTROGRAM PIPELINE
// -------------------------------------------------------------

function fftRadix2(real, imag) {
  const n = real.length;
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tempR = real[i]; real[i] = real[j]; real[j] = tempR;
      const tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI;
    }
    let k = n >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle), wStepI = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wR = 1, wI = 0;
      for (let k = 0; k < halfLen; k++) {
        const pos = i + k, match = pos + halfLen;
        const tR = real[match] * wR - imag[match] * wI;
        const tI = real[match] * wI + imag[match] * wR;
        real[match] = real[pos] - tR;
        imag[match] = real[pos] - tI;
        real[pos] += tR;
        imag[pos] += tI;
        const nextWR = wR * wStepR - wI * wStepI;
        wI = wR * wStepI + wI * wStepR;
        wR = nextWR;
      }
    }
  }
}

function magnitudeSpectrum(frame) {
  const N = frame.length;
  const real = new Float32Array(N);
  const imag = new Float32Array(N);
  real.set(frame);
  fftRadix2(real, imag);
  const half = N / 2;
  const mags = new Float32Array(half);
  for (let k = 0; k < half; k++) {
    mags[k] = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
  }
  return mags;
}

function getSpectrogram(audioBuffer) {
  const targetSampleRate = 16000;
  const frameSize = 256;
  const numFreqBins = 64;
  const targetFrames = 64;

  const src = audioBuffer.getChannelData(0);
  const srcRate = audioBuffer.sampleRate;
  const downsampleFactor = Math.max(1, Math.floor(srcRate / targetSampleRate));
  const downSampledLength = Math.floor(src.length / downsampleFactor);
  const signal = new Float32Array(downSampledLength);
  for (let i = 0; i < downSampledLength; i++) {
    signal[i] = src[i * downsampleFactor];
  }

  const hann = new Float32Array(frameSize);
  for (let i = 0; i < frameSize; i++) {
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (frameSize - 1)));
  }

  const maxStart = Math.max(0, signal.length - frameSize);
  const step = maxStart / (targetFrames - 1 || 1);
  const resized = [];

  for (let t = 0; t < targetFrames; t++) {
    const start = Math.floor(t * step);
    const frameData = new Float32Array(frameSize);
    for (let i = 0; i < frameSize; i++) {
      const idx = start + i;
      frameData[i] = idx < signal.length ? signal[idx] * hann[i] : 0;
    }

    const mags = magnitudeSpectrum(frameData);
    const frameBins = new Float32Array(numFreqBins);
    for (let b = 0; b < numFreqBins; b++) {
      const srcIndex = Math.floor((b / numFreqBins) * mags.length);
      frameBins[b] = Math.log1p(mags[srcIndex] || 0);
    }
    resized.push(Array.from(frameBins));
  }

  let minVal = Infinity, maxVal = -Infinity;
  for (let r = 0; r < resized.length; r++) {
    for (let c = 0; c < resized[r].length; c++) {
      const v = resized[r][c];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }

  const range = maxVal - minVal + 1e-6;
  for (let r = 0; r < resized.length; r++) {
    for (let c = 0; c < resized[r].length; c++) {
      resized[r][c] = (resized[r][c] - minVal) / range;
    }
  }

  return resized;
}

// -------------------------------------------------------------
// PIPELINE & BACKEND PREDICTION
// -------------------------------------------------------------

async function runEmotionPipeline(audioBuffer) {
  try {
    const spec = getSpectrogram(audioBuffer);

    const chan = audioBuffer.getChannelData(0);
    
    // Trim the first and last 250ms to ignore mouse click thumps & hardware start/stop pops
    const trimSamples = Math.floor(audioBuffer.sampleRate * 0.25);
    const startIdx = Math.min(chan.length, trimSamples);
    const endIdx = Math.max(startIdx, chan.length - trimSamples);

    let sum = 0;
    let maxAmp = 0;
    let count = 0;
    for (let i = startIdx; i < endIdx; i++) {
      const v = chan[i];
      sum += v * v;
      const av = Math.abs(v);
      if (av > maxAmp) maxAmp = av;
      count++;
    }
    const rms = Math.sqrt(sum / (count || 1));

    const res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        spectrogram: spec,
        rms: rms,
        max_amplitude: maxAmp
      })
    });

    if (!res.ok) throw new Error("Backend server error");
    const data = await res.json();

    if (data.is_invalid) {
      setStatus(`⚠️ ${data.invalid_reason}`, "warn");
      return;
    }

    setStatus("✅ Prediction complete (CNN + MFCC Ensemble).", "ok");

    renderResults(data);
    renderTimeline(audioBuffer);
    renderPlaylistRecommendations(data.emotion);
    saveToHistory(data);

  } catch (err) {
    console.error(err);
    setStatus("Error communicating with AI server.", "error");
  }
}

// -------------------------------------------------------------
// RESULT RENDERING & EXPLAINABILITY
// -------------------------------------------------------------

function renderResults(data) {
  const container = document.getElementById("emotionResult");
  if (!container) return;

  const probs = data.probabilities || {};
  const primaryEmotion = data.emotion || "calm";
  const confidence = data.confidence || 0;

  const html = `
    <div class="emotion-card-rendered">
      <div class="primary-emotion-header">
        <span class="primary-emotion-title">You sound ${primaryEmotion.toUpperCase()}</span>
        <span class="confidence-chip">${confidence}% Match</span>
      </div>

      <div class="multilabel-list">
        ${EMOTIONS.map(emo => `
          <div class="multilabel-item">
            <span class="multilabel-name">${emo}</span>
            <div class="multilabel-bar-bg">
              <div class="multilabel-bar-fill" style="width: ${probs[emo] || 5}%"></div>
            </div>
            <span class="multilabel-val">${probs[emo] || 0}%</span>
          </div>
        `).join('')}
      </div>
    </div>
  `;
  container.innerHTML = html;

  // Render Vocal Explainability Reasons
  const explainBox = document.getElementById("explainabilityBox");
  const explainList = document.getElementById("explainList");
  if (explainBox && explainList && data.explainability) {
    explainBox.style.display = "block";
    explainList.innerHTML = data.explainability.map(item => `<li>${item}</li>`).join('');
  }
}

function renderTimeline(audioBuffer) {
  const box = document.getElementById("timelineBox");
  const track = document.getElementById("timelineTrack");
  if (!box || !track) return;

  box.style.display = "block";
  track.innerHTML = "";

  const durationSec = Math.floor(audioBuffer.duration);
  const sampleRate = audioBuffer.sampleRate;
  const chan = audioBuffer.getChannelData(0);

  for (let s = 0; s < durationSec; s++) {
    const startIdx = s * sampleRate;
    const endIdx = Math.min(chan.length, (s + 1) * sampleRate);
    
    let sum = 0;
    for (let i = startIdx; i < endIdx; i += 100) sum += chan[i] * chan[i];
    const energy = Math.sqrt(sum / ((endIdx - startIdx) / 100));

    let emo = "calm";
    if (energy > 0.15) emo = "angry";
    else if (energy > 0.08) emo = "excited";
    else if (energy > 0.03) emo = "happy";

    const seg = document.createElement("div");
    seg.className = "timeline-segment";
    seg.textContent = `${s}s: ${emo.toUpperCase()}`;
    track.appendChild(seg);
  }
}

function renderPlaylistRecommendations(emotion) {
  const grid = document.getElementById("playlistGrid");
  if (!grid) return;

  const langSelect = document.getElementById("prefLanguage");
  const genreSelect = document.getElementById("prefGenre");
  
  const lang = langSelect ? langSelect.value : "English";
  const genre = genreSelect ? genreSelect.value : "Pop";

  document.getElementById("prefSummary").textContent = `Language: ${lang} • Genre: ${genre}`;

  const playlists = [
    { title: `${lang} ${emotion.toUpperCase()} ${genre} Mix`, url: `https://www.youtube.com/results?search_query=${lang}+${emotion}+${genre}+playlist` },
    { title: `Top ${emotion} ${genre} Hits`, url: `https://www.youtube.com/results?search_query=top+${emotion}+${genre}+playlist` },
    { title: `${lang} Vocal Mood Experience`, url: `https://www.youtube.com/results?search_query=${lang}+${emotion}+songs` }
  ];

  grid.innerHTML = playlists.map(p => `
    <div class="playlist-card-item">
      <h4>${p.title}</h4>
      <a href="${p.url}" target="_blank" rel="noopener noreferrer">
        ▶ Open on YouTube
      </a>
    </div>
  `).join('');
}

// -------------------------------------------------------------
// HISTORY & DASHBOARD ANALYTICS
// -------------------------------------------------------------

function saveToHistory(data) {
  const history = JSON.parse(localStorage.getItem("emotion_history") || "[]");
  history.unshift({
    timestamp: new Date().toLocaleTimeString(),
    emotion: data.emotion,
    confidence: data.confidence
  });
  if (history.length > 20) history.pop();
  localStorage.setItem("emotion_history", JSON.stringify(history));
}

function renderAnalyticsDashboard() {
  const history = JSON.parse(localStorage.getItem("emotion_history") || "[]");
  const total = history.length;
  document.getElementById("statTotal").textContent = total;

  if (total === 0) return;

  const counts = {};
  let totalConf = 0;
  history.forEach(item => {
    counts[item.emotion] = (counts[item.emotion] || 0) + 1;
    totalConf += item.confidence;
  });

  const dominant = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
  document.getElementById("statDominant").textContent = dominant.toUpperCase();
  document.getElementById("statAvgConf").textContent = `${Math.round(totalConf / total)}%`;

  // Render Bars
  const distEl = document.getElementById("chartDistribution");
  if (distEl) {
    distEl.innerHTML = EMOTIONS.map(emo => {
      const cnt = counts[emo] || 0;
      const pct = Math.round((cnt / total) * 100);
      return `
        <div class="multilabel-item" style="margin-bottom: 8px;">
          <span class="multilabel-name">${emo}</span>
          <div class="multilabel-bar-bg">
            <div class="multilabel-bar-fill" style="width: ${pct}%"></div>
          </div>
          <span class="multilabel-val">${cnt} (${pct}%)</span>
        </div>
      `;
    }).join('');
  }

  // History List
  const listEl = document.getElementById("historyList");
  if (listEl) {
    listEl.innerHTML = history.slice(0, 5).map(item => `
      <li class="history-item">
        <span>⏱️ ${item.timestamp}</span>
        <strong>${item.emotion.toUpperCase()} (${item.confidence}%)</strong>
      </li>
    `).join('');
  }
}

async function loadMicrophones() {
  const select = document.getElementById("micSelect");
  if (!select) return;
  try {
    // Request permission once to unlock device labels
    const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    tempStream.getTracks().forEach(t => t.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioDevices = devices.filter(d => d.kind === "audioinput");
    
    select.innerHTML = '<option value="">Default Microphone</option>';
    audioDevices.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Microphone ${select.options.length}`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("Error loading microphones:", err);
  }
}

// Global Exports
window.startRecording = startRecording;
window.stopRecording = stopRecording;
window.switchTab = switchTab;

window.addEventListener("DOMContentLoaded", () => {
  setStatus("Connecting to AI Python server…", "");
  loadMicrophones();
});
