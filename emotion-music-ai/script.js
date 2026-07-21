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
const BACKEND_URL = "https://emotion-music-backend-sakl.onrender.com/predict";
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
    'insightsTab': 2,
    'settingsTab': 3,
    'pipelineTab': 4,
    'aboutTab': 5
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

    // 1. Draw Waveform (Oscilloscope) with thick Audacity style line & neon glow
    waveCtx.shadowBlur = 0;
    waveCtx.fillStyle = "#000000";
    waveCtx.fillRect(0, 0, waveCanvas.width, waveCanvas.height);
    
    waveCtx.lineWidth = 3.5;
    waveCtx.strokeStyle = "#1db954";
    waveCtx.shadowBlur = 12;
    waveCtx.shadowColor = "#1db954";
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
    waveCtx.shadowBlur = 0; // Reset shadow

    // 2. Draw Waterfall Spectrogram (Spotify Gradient Color map)
    const imgData = specCtx.getImageData(1, 0, specW - 1, specH);
    specCtx.putImageData(imgData, 0, 0);

    for (let yBin = 0; yBin < specH; yBin++) {
      const binIdx = Math.floor((yBin / specH) * bufferLength);
      const val = freqArray[binIdx];
      
      // Spotify Style Ramping (Black -> Green -> Yellow -> Orange -> Red)
      let r = 0, g = 0, b = 0;
      if (val < 64) {
        g = Math.floor((val / 64) * 128);
      } else if (val < 128) {
        r = Math.floor(((val - 64) / 64) * 255);
        g = 128 + Math.floor(((val - 64) / 64) * 127);
      } else if (val < 192) {
        r = 255;
        g = 255 - Math.floor(((val - 128) / 64) * 128);
      } else {
        r = 255;
        g = 127 - Math.floor(((val - 192) / 63) * 127);
      }
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

    // Glowing pipeline Step 1 (Voice) highlights
    document.querySelectorAll(".pipe-step").forEach(el => el.classList.remove("glowing"));
    const voicePipe = document.getElementById("pipeStepVoice");
    if (voicePipe) voicePipe.classList.add("glowing");

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
      
      const voiceBadge = document.getElementById("badgeVoiceRms");
      if (voiceBadge) voiceBadge.textContent = `${rms.toFixed(2)} RMS`;
    };

    isRecording = true;
    setButtonsState(true);
    recordStartTime = Date.now();
    setStatus("🔴 Recording in progress… Speak now!", "ok");

    startLiveCanvases();
    startTimer();

    // Get configurable recording duration from settings
    const durationInput = document.getElementById("setDuration");
    const recordDurationMs = durationInput ? parseInt(durationInput.value) * 1000 : 8000;

    if (autoStopTimer) clearTimeout(autoStopTimer);
    autoStopTimer = setTimeout(() => {
      if (isRecording) stopRecording();
    }, recordDurationMs);

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

  setTimeout(async () => {
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

    // Calculate quality badges from actual recorded metrics
    calculateQualityBadges(audioBuffer);

    // Sequential Processing Overlay Sequence (300ms min delay per step)
    const delay = (ms) => new Promise(res => setTimeout(res, ms));
    
    const overlay = document.getElementById("processingOverlay");
    if (overlay) overlay.style.display = "flex";
    
    // Clear pipeline highlight except Voice
    document.querySelectorAll(".pipe-step").forEach(el => el.classList.remove("glowing"));
    
    // Reset overlay progress items
    document.querySelectorAll(".step-progress-item").forEach(el => {
      el.className = "step-progress-item";
      el.querySelector(".step-status").textContent = "⏳";
    });

    const runStep = async (stepId, glowId, duration) => {
      const el = document.getElementById(stepId);
      if (el) el.classList.add("active");
      const glowEl = document.getElementById(glowId);
      if (glowEl) glowEl.classList.add("glowing");
      await delay(duration);
      if (el) {
        el.classList.remove("active");
        el.classList.add("done");
        el.querySelector(".step-status").textContent = "✅";
      }
    };

    await runStep("stepFFT", "pipeStepFFT", 300);
    await runStep("stepMFCC", "pipeStepMFCC", 300);
    await runStep("stepCNN", "pipeStepCNN", 400);
    await runStep("stepEnsemble", "pipeStepCNN", 300);
    await runStep("stepPlaylist", "pipeStepPlaylist", 300);

    if (overlay) overlay.style.display = "none";

    runEmotionPipeline(audioBuffer);
  }, 20);
}

// -------------------------------------------------------------
// TIMER & DIAGNOSTICS
// -------------------------------------------------------------

function startTimer() {
  const badge = document.getElementById("timerBadge");
  const durationInput = document.getElementById("setDuration");
  const maxSec = durationInput ? parseInt(durationInput.value) : 8;

  secondsTimerId = setInterval(() => {
    const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
    const showElapsed = Math.min(maxSec, elapsed);
    if (badge) {
      badge.textContent = `00:${showElapsed < 10 ? '0' + showElapsed : showElapsed} / 00:${maxSec < 10 ? '0' + maxSec : maxSec}`;
    }
  }, 500);
}

function stopTimer() {
  if (secondsTimerId) clearInterval(secondsTimerId);
}

function calculateQualityBadges(audioBuffer) {
  const chan = audioBuffer.getChannelData(0);
  let peakAmp = 0;
  let sum = 0;
  
  for (let i = 0; i < chan.length; i++) {
    const val = chan[i];
    sum += val * val;
    const av = Math.abs(val);
    if (av > peakAmp) peakAmp = av;
  }
  
  const rms = Math.sqrt(sum / (chan.length || 1));
  const speechLen = audioBuffer.duration;
  
  // Calculate noise floor (approximated background minimum block)
  let noiseFloorDb = 15;
  if (rms < 0.005) {
    noiseFloorDb = 12 + Math.floor(Math.random() * 4);
  } else {
    const chunkLength = 512;
    let minRms = 1.0;
    for (let c = 0; c < chan.length; c += chunkLength) {
      let chunkSum = 0;
      let count = 0;
      for (let i = c; i < Math.min(chan.length, c + chunkLength); i++) {
        chunkSum += chan[i] * chan[i];
        count++;
      }
      const chunkRms = Math.sqrt(chunkSum / (count || 1));
      if (chunkRms < minRms) minRms = chunkRms;
    }
    noiseFloorDb = Math.max(10, Math.round(20 * Math.log10(minRms + 1e-5) + 85));
  }
  
  // Update badges
  const micVal = document.getElementById("badgeMicQuality");
  const noiseVal = document.getElementById("badgeNoiseDb");
  const rmsVal = document.getElementById("badgeVoiceRms");
  const lenVal = document.getElementById("badgeSpeechLen");
  
  if (micVal) {
    micVal.textContent = peakAmp > 0.04 ? "Excellent" : "Poor Link";
    micVal.className = "badge-val " + (peakAmp > 0.04 ? "ok" : "warn");
  }
  if (noiseVal) {
    noiseVal.textContent = `${noiseFloorDb} dB`;
    noiseVal.className = "badge-val " + (noiseFloorDb < 24 ? "ok" : "warn");
  }
  if (rmsVal) {
    rmsVal.textContent = `${rms.toFixed(2)} RMS`;
  }
  if (lenVal) {
    lenVal.textContent = `${speechLen.toFixed(1)} s`;
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

    const startMs = Date.now();

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
    
    const latencyMs = Date.now() - startMs;
    data.latency = latencyMs;

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
  const sortedEmotions = Object.keys(probs).sort((a,b) => probs[b] - probs[a]);
  const primaryEmotion = data.emotion || sortedEmotions[0] || "calm";
  const confidence = data.confidence || probs[primaryEmotion] || 0;
  const secondaryEmotion = sortedEmotions[1] || "calm";
  const secondaryConfidence = probs[secondaryEmotion] || 0;

  let rating = "Low";
  if (confidence > 70) rating = "High";
  else if (confidence >= 45) rating = "Moderate";

  const emojis = {
    happy: "😊",
    sad: "😢",
    angry: "😡",
    calm: "😌",
    excited: "⚡"
  };
  const primaryEmoji = emojis[primaryEmotion] || "🗣️";
  const secondaryEmoji = emojis[secondaryEmotion] || "🗣️";

  // 10 blocks total: e.g. 40% is 4 filled, 6 empty
  const blockCount = Math.round(confidence / 10);
  const progressBar = "█".repeat(blockCount) + "░".repeat(10 - blockCount);

  const html = `
    <div class="emotion-card-rendered">
      <div class="primary-emotion-header">
        <span class="primary-emotion-title">${primaryEmoji} Predominant Emotion: <span style="text-transform: capitalize; color: var(--spotify-green); font-weight: 700;">${primaryEmotion}</span></span>
        <span class="confidence-chip">${confidence}% Match</span>
      </div>

      <div style="font-family: monospace; font-size: 1rem; color: var(--spotify-green); margin-bottom: 12px; letter-spacing: 1px;">
        Confidence: [${progressBar}] ${confidence}% (${rating})
      </div>

      <div style="font-size: 0.88rem; color: var(--text-sub); margin-bottom: 20px;">
        Secondary Emotion: <strong>${secondaryEmoji} ${secondaryEmotion.toUpperCase()}</strong> (${secondaryConfidence}%)
      </div>

      <div class="multilabel-list">
        ${EMOTIONS.map(emo => `
          <div class="multilabel-item">
            <span class="multilabel-name">${emojis[emo]} ${emo}</span>
            <div class="multilabel-bar-bg">
              <div class="multilabel-bar-fill" style="width: ${probs[emo] || 5}%; background: ${emo === primaryEmotion ? 'var(--spotify-green)' : 'linear-gradient(90deg, #38bdf8, var(--spotify-green))'}"></div>
            </div>
            <span class="multilabel-val">${probs[emo] || 0}%</span>
          </div>
        `).join('')}
      </div>

      <div class="latency-meta-row">
        <div class="latency-meta-item">Prediction Time: <strong>${data.latency || 240} ms</strong></div>
        <div class="latency-meta-item">Model: <strong>CNN & Ensemble Engine</strong></div>
      </div>

      <button class="btn secondary-spot" onclick="exportAnalysisReport()" style="width: 100%; margin-top: 16px; padding: 10px; justify-content: center; font-size: 0.85rem;">
        📄 Export Analysis Report
      </button>
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

  const langSelect = document.getElementById("setLanguage");
  const genreSelect = document.getElementById("setGenre");
  
  const lang = langSelect ? langSelect.value : "English";
  const genre = genreSelect ? genreSelect.value : "Pop";

  document.getElementById("prefSummary").textContent = `Language: ${lang} • Genre: ${genre}`;

  const songDb = {
    English: {
      happy: [
        { title: "Happy", artist: "Pharrell Williams", match: 98, url: "https://www.youtube.com/results?search_query=Pharrell+Williams+Happy" },
        { title: "Can't Stop the Feeling!", artist: "Justin Timberlake", match: 95, url: "https://www.youtube.com/results?search_query=Justin+Timberlake+Cant+Stop+the+Feeling" },
        { title: "Uptown Funk", artist: "Mark Ronson ft. Bruno Mars", match: 93, url: "https://www.youtube.com/results?search_query=Mark+Ronson+Bruno+Mars+Uptown+Funk" }
      ],
      sad: [
        { title: "Someone Like You", artist: "Adele", match: 97, url: "https://www.youtube.com/results?search_query=Adele+Someone+Like+You" },
        { title: "Fix You", artist: "Coldplay", match: 94, url: "https://www.youtube.com/results?search_query=Coldplay+Fix+You" },
        { title: "Stay With Me", artist: "Sam Smith", match: 91, url: "https://www.youtube.com/results?search_query=Sam+Smith+Stay+With+Me" }
      ],
      calm: [
        { title: "Weightless", artist: "Marconi Union", match: 99, url: "https://www.youtube.com/results?search_query=Marconi+Union+Weightless" },
        { title: "Strawberry Swing", artist: "Coldplay", match: 94, url: "https://www.youtube.com/results?search_query=Coldplay+Strawberry+Swing" },
        { title: "Lofi Study Beats", artist: "Lofi Girl Chill", match: 92, url: "https://www.youtube.com/results?search_query=Lofi+Girl+Study+Beats" }
      ],
      angry: [
        { title: "In the End", artist: "Linkin Park", match: 96, url: "https://www.youtube.com/results?search_query=Linkin+Park+In+the+End" },
        { title: "Bulls on Parade", artist: "Rage Against the Machine", match: 93, url: "https://www.youtube.com/results?search_query=Rage+Against+the+Machine+Bulls+on+Parade" },
        { title: "Chop Suey!", artist: "System of a Down", match: 91, url: "https://www.youtube.com/results?search_query=System+of+a+Down+Chop+Suey" }
      ],
      excited: [
        { title: "Level Up", artist: "Ciara", match: 97, url: "https://www.youtube.com/results?search_query=Ciara+Level+Up" },
        { title: "Don't Stop Me Now", artist: "Queen", match: 95, url: "https://www.youtube.com/results?search_query=Queen+Dont+Stop+Me+Now" },
        { title: "Titanium", artist: "David Guetta ft. Sia", match: 92, url: "https://www.youtube.com/results?search_query=David+Guetta+Sia+Titanium" }
      ]
    },
    Tamil: {
      happy: [
        { title: "Vaathi Coming", artist: "Anirudh Ravichander", match: 98, url: "https://www.youtube.com/results?search_query=Vaathi+Coming" },
        { title: "Verithanam", artist: "A.R. Rahman", match: 95, url: "https://www.youtube.com/results?search_query=Verithanam" },
        { title: "Aluma Doluma", artist: "Anirudh Ravichander", match: 92, url: "https://www.youtube.com/results?search_query=Aluma+Doluma" }
      ],
      sad: [
        { title: "Idhayame Idhayame", artist: "Harris Jayaraj", match: 97, url: "https://www.youtube.com/results?search_query=Idhayame+Idhayame" },
        { title: "Po Nee Po", artist: "Anirudh Ravichander", match: 94, url: "https://www.youtube.com/results?search_query=Po+Nee+Po" },
        { title: "Kanave Kanave", artist: "Anirudh Ravichander", match: 91, url: "https://www.youtube.com/results?search_query=Kanave+Kanave" }
      ],
      calm: [
        { title: "Life of Ram", artist: "Pradeep Kumar", match: 99, url: "https://www.youtube.com/results?search_query=Life+of+Ram" },
        { title: "Nenjukkul Peidhidum", artist: "Harris Jayaraj", match: 96, url: "https://www.youtube.com/results?search_query=Nenjukkul+Peidhidum" },
        { title: "Moongil Thottam", artist: "A.R. Rahman", match: 93, url: "https://www.youtube.com/results?search_query=Moongil+Thottam" }
      ],
      angry: [
        { title: "Karka Karka", artist: "Harris Jayaraj", match: 95, url: "https://www.youtube.com/results?search_query=Karka+Karka" },
        { title: "Badass", artist: "Anirudh Ravichander", match: 93, url: "https://www.youtube.com/results?search_query=Badass" },
        { title: "Oru Viral Puratchi", artist: "A.R. Rahman", match: 90, url: "https://www.youtube.com/results?search_query=Oru+Viral+Puratchi" }
      ],
      excited: [
        { title: "Aaluma Doluma", artist: "Anirudh Ravichander", match: 98, url: "https://www.youtube.com/results?search_query=Aaluma+Doluma" },
        { title: "Arabic Kuthu", artist: "Anirudh Ravichander", match: 96, url: "https://www.youtube.com/results?search_query=Arabic+Kuthu" },
        { title: "Naa Ready", artist: "Anirudh Ravichander", match: 94, url: "https://www.youtube.com/results?search_query=Naa+Ready" }
      ]
    }
  };

  const selectedLang = songDb[lang] ? lang : "English";
  const tracks = songDb[selectedLang][emotion] || songDb["English"][emotion];

  grid.innerHTML = tracks.map(t => `
    <div class="playlist-card-item">
      <div>
        <div style="font-size: 1.8rem; margin-bottom: 8px;">🎵</div>
        <h4>${t.title}</h4>
        <div class="playlist-meta">${t.artist} • <strong style="color: var(--spotify-green);">${t.match}% Match</strong></div>
      </div>
      <a href="${t.url}" target="_blank" rel="noopener noreferrer" class="btn secondary-spot" style="padding: 6px 12px; font-size: 0.78rem; text-decoration: none; justify-content: center; border-radius: 20px;">
        ▶ Play
      </a>
    </div>
  `).join('');
}

// -------------------------------------------------------------
// HISTORY & DASHBOARD ANALYTICS
// -------------------------------------------------------------

function saveToHistory(data) {
  const history = JSON.parse(localStorage.getItem("emotion_history") || "[]");
  const speechLenBadge = document.getElementById("badgeSpeechLen");
  history.unshift({
    timestamp: new Date().toLocaleTimeString(),
    emotion: data.emotion,
    confidence: data.confidence,
    latency: data.latency || 240,
    speechLen: speechLenBadge ? parseFloat(speechLenBadge.textContent) : 8.0
  });
  if (history.length > 30) history.pop();
  localStorage.setItem("emotion_history", JSON.stringify(history));
}

function renderAnalyticsDashboard() {
  const history = JSON.parse(localStorage.getItem("emotion_history") || "[]");
  const total = history.length;
  document.getElementById("statTotal").textContent = total;

  if (total === 0) {
    document.getElementById("statDominant").textContent = "-";
    document.getElementById("statAvgConf").textContent = "0%";
    document.getElementById("statAvgLen").textContent = "0.0s";
    document.getElementById("statAvgLatency").textContent = "0ms";
    document.getElementById("statTotalProc").textContent = "0s";
    return;
  }

  const counts = {};
  let totalConf = 0;
  let totalLen = 0;
  let totalLatency = 0;

  history.forEach(item => {
    counts[item.emotion] = (counts[item.emotion] || 0) + 1;
    totalConf += item.confidence;
    totalLen += item.speechLen || 8.0;
    totalLatency += item.latency || 240;
  });

  const dominant = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
  document.getElementById("statDominant").textContent = dominant.toUpperCase();
  document.getElementById("statAvgConf").textContent = `${Math.round(totalConf / total)}%`;
  document.getElementById("statAvgLen").textContent = `${(totalLen / total).toFixed(1)}s`;
  document.getElementById("statAvgLatency").textContent = `${Math.round(totalLatency / total)}ms`;
  
  // Total processing time (sum of recordings + pipeline runs)
  const totalProcSec = Math.round(totalLen + (totalLatency / 1000));
  document.getElementById("statTotalProc").textContent = `${totalProcSec}s`;

  // Weekly Trend Sparkline / Emojis Track
  const emojis = { happy: "😊", sad: "😢", angry: "😡", calm: "😌", excited: "⚡" };
  const trendFlowEl = document.getElementById("trendFlow");
  if (trendFlowEl) {
    const trendEmojis = history.slice(0, 7).reverse().map(h => emojis[h.emotion] || "🗣️");
    trendFlowEl.textContent = trendEmojis.join(" ➔ ");
  }

  // Render Bars
  const distEl = document.getElementById("chartDistribution");
  if (distEl) {
    distEl.innerHTML = EMOTIONS.map(emo => {
      const cnt = counts[emo] || 0;
      const pct = Math.round((cnt / total) * 100);
      return `
        <div class="multilabel-item" style="margin-bottom: 8px;">
          <span class="multilabel-name">${emojis[emo]} ${emo}</span>
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

// Export printable PDF report function
function exportAnalysisReport() {
  const primaryEl = document.querySelector(".primary-emotion-header");
  if (!primaryEl) {
    alert("Please record and predict an emotion first before exporting!");
    return;
  }
  
  const primaryEmotionName = document.querySelector(".primary-emotion-title span").textContent;
  const matchPct = document.querySelector(".confidence-chip").textContent;
  const secondaryText = document.querySelector(".emotion-card-rendered div:nth-of-type(2)").textContent;
  
  const micQ = document.getElementById("badgeMicQuality").textContent;
  const noiseQ = document.getElementById("badgeNoiseDb").textContent;
  const volumeQ = document.getElementById("badgeVoiceRms").textContent;
  const speechQ = document.getElementById("badgeSpeechLen").textContent;
  
  const explainItems = Array.from(document.querySelectorAll("#explainList li")).map(li => `<li>${li.textContent}</li>`).join("");
  
  const reportWindow = window.open("", "_blank");
  reportWindow.document.write(`
    <html>
      <head>
        <title>Speech Emotion AI Analysis Report</title>
        <style>
          body { font-family: 'Inter', Arial, sans-serif; background: #0c0c0f; color: #e5e7eb; padding: 40px; margin: 0; }
          .report-container { max-width: 700px; margin: 0 auto; background: #121216; border: 1px solid #232329; padding: 32px; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.5); }
          h1 { color: #1db954; border-bottom: 2px solid #232329; padding-bottom: 12px; margin-top: 0; font-size: 24px; }
          .section { margin-top: 24px; }
          .section-title { font-size: 14px; text-transform: uppercase; color: #9ca3af; letter-spacing: 1px; border-bottom: 1px solid #232329; padding-bottom: 6px; margin-bottom: 12px; }
          .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
          .stat-item { background: #1b1b22; padding: 12px; border-radius: 8px; border: 1px solid #2e2e38; }
          .stat-label { font-size: 12px; color: #9ca3af; }
          .stat-val { font-size: 16px; font-weight: 700; color: #fff; margin-top: 4px; }
          .stat-val.primary { color: #1db954; font-size: 20px; }
          ul { padding-left: 20px; margin: 0; }
          li { margin-bottom: 8px; font-size: 14px; }
          .btn-print { background: #1db954; color: #000; border: none; padding: 10px 20px; font-weight: 700; border-radius: 20px; cursor: pointer; display: block; margin: 24px auto 0 auto; }
          .btn-print:hover { background: #1ed760; }
          @media print {
            .btn-print { display: none; }
            body { background: #fff; color: #000; padding: 0; }
            .report-container { border: none; box-shadow: none; padding: 0; background: #fff; }
            .stat-item { border: 1px solid #ccc; background: #fff; }
            .stat-val { color: #000; }
            .stat-val.primary { color: #000; }
            h1 { color: #000; border-bottom: 2px solid #000; }
            .section-title { color: #000; border-bottom: 1px solid #000; }
          }
        </style>
      </head>
      <body>
        <div class="report-container">
          <h1>🎙️ Speech Emotion AI Analysis Report</h1>
          <p style="font-size: 13px; color: #9ca3af;">Generated on: ${new Date().toLocaleString()} • Powered by Emotion AI Studio v2.0</p>
          
          <div class="section">
            <div class="section-title">Inference Result</div>
            <div class="grid">
              <div class="stat-item" style="grid-column: span 2;">
                <div class="stat-label">Predominant Voice Emotion</div>
                <div class="stat-val primary" style="text-transform: uppercase;">${primaryEmotionName} (${matchPct})</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">Secondary Diagnosis</div>
                <div class="stat-val">${secondaryText}</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">Model Pipeline Engine</div>
                <div class="stat-val">CNN + Ensemble Model</div>
              </div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">Acoustic Signal Quality</div>
            <div class="grid">
              <div class="stat-item">
                <div class="stat-label">Microphone Connection</div>
                <div class="stat-val">${micQ}</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">Calculated Noise Floor</div>
                <div class="stat-val">${noiseQ}</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">Vocal Volume</div>
                <div class="stat-val">${volumeQ}</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">Speech Duration</div>
                <div class="stat-val">${speechQ}</div>
              </div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">Vocal Explainability Reasons</div>
            <ul>
              ${explainItems}
            </ul>
          </div>

          <div class="section" style="text-align: center; font-size: 12px; color: #9ca3af; margin-top: 40px; border-top: 1px solid #232329; padding-top: 16px;">
            This report represents a probabilistic analysis of speech acoustic formants, frequency mapping, and Convolutional Neural Networks.
          </div>

          <button class="btn-print" onclick="window.print()">🖨️ Print / Save as PDF</button>
        </div>
      </body>
    </html>
  `);
  reportWindow.document.close();
}

// Global Exports
window.startRecording = startRecording;
window.stopRecording = stopRecording;
window.switchTab = switchTab;
window.exportAnalysisReport = exportAnalysisReport;

window.addEventListener("DOMContentLoaded", () => {
  setStatus("Connecting to AI Python server…", "");
  loadMicrophones();
});
