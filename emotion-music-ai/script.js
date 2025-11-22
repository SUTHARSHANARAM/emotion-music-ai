'use strict';

let model = null;
let modelLoaded = false;

let audioContext = null;
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let autoStopTimer = null;

// If true, use a local Python backend (Flask server) for emotion predictions
// instead of running a model in the browser. The backend should expose
// POST /predict that accepts { spectrogram: number[][] } and returns
// { emotion: "happy" | "sad" | ... }.
const USE_BACKEND = true;
const BACKEND_URL = "http://127.0.0.1:5000/predict";

const EMOTIONS = ["happy", "sad", "angry", "calm", "excited"];

const musicSuggestions = {
  happy: [
    {
      title: "Upbeat party mix (pop / dance)",
      url: "https://www.youtube.com/results?search_query=upbeat+happy+pop+playlist",
    },
    {
      title: "Feel-good pop hits",
      url: "https://www.youtube.com/results?search_query=feel+good+pop+hits+playlist",
    },
  ],
  sad: [
    {
      title: "Soft piano for reflection",
      url: "https://www.youtube.com/results?search_query=soft+piano+music+playlist",
    },
    {
      title: "Calm lofi beats",
      url: "https://www.youtube.com/results?search_query=lofi+chill+playlist",
    },
  ],
  angry: [
    {
      title: "Chillstep to cool down",
      url: "https://www.youtube.com/results?search_query=chillstep+playlist",
    },
    {
      title: "Relaxing ambient textures",
      url: "https://www.youtube.com/results?search_query=relaxing+ambient+music+playlist",
    },
  ],
  calm: [
    {
      title: "Rain sounds / nature",
      url: "https://www.youtube.com/results?search_query=rain+sounds+for+sleep+playlist",
    },
    {
      title: "Meditation & deep focus",
      url: "https://www.youtube.com/results?search_query=meditation+music+playlist",
    },
  ],
  excited: [
    {
      title: "Workout / gym energy mix",
      url: "https://www.youtube.com/results?search_query=workout+music+playlist",
    },
    {
      title: "EDM festival hits",
      url: "https://www.youtube.com/results?search_query=edm+festival+mix+playlist",
    },
  ],
};

function byId(id) {
  return document.getElementById(id);
}

function setStatus(message, type = "") {
  const el = byId("statusText");
  if (!el) return;
  el.textContent = message;
  el.classList.remove("ok", "warn", "error");
  if (type) el.classList.add(type);
}

function setButtonsRecordingState(recording) {
  const recordBtn = byId("recordBtn");
  const stopBtn = byId("stopBtn");
  if (!recordBtn || !stopBtn) return;
  recordBtn.disabled = recording;
  stopBtn.disabled = !recording;
}

async function loadModel() {
  if (USE_BACKEND) {
    // We will use the Python backend instead of a TFJS model.
    modelLoaded = false;
    model = null;
    setStatus("Backend mode: using local server for emotion predictions.", "ok");
    return;
  }

  try {
    setStatus("Loading emotion model…", "");
    model = await tf.loadLayersModel("model/model.json");
    modelLoaded = true;
    setStatus("Model loaded. Ready to record.", "ok");
  } catch (err) {
    console.warn("Could not load model/model.json. Running in demo mode.", err);
    modelLoaded = false;
    setStatus(
      "Model file not found. Demo mode: heuristic emotion only (place your model in /model).",
      "warn"
    );
  }
}

async function ensureAudioContextAndStream() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("getUserMedia is not supported in this browser.");
  }

  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  if (!mediaStream) {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }

  if (!mediaRecorder) {
    mediaRecorder = new MediaRecorder(mediaStream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        recordedChunks.push(e.data);
      }
    };
    mediaRecorder.onstop = handleRecordingStop;
  }
}

async function startRecording() {
  if (isRecording) return;
  try {
    await ensureAudioContextAndStream();
  } catch (err) {
    console.error(err);
    setStatus("Microphone access failed. Check permissions.", "error");
    return;
  }

  recordedChunks = [];
  isRecording = true;
  setButtonsRecordingState(true);
  setStatus("Recording… speak now (about 8 seconds, or press Stop when done).", "ok");

  try {
    mediaRecorder.start();
  } catch (err) {
    console.error(err);
    setStatus("Could not start recording.", "error");
    isRecording = false;
    setButtonsRecordingState(false);
    return;
  }

  if (autoStopTimer) clearTimeout(autoStopTimer);
  autoStopTimer = setTimeout(() => {
    if (isRecording && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  }, 8000);
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  if (mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}

async function handleRecordingStop() {
  isRecording = false;
  setButtonsRecordingState(false);
  setStatus("Processing audio…", "");

  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  recordedChunks = [];

  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await new Promise((resolve, reject) => {
      audioContext.decodeAudioData(arrayBuffer, resolve, reject);
    });

    await runEmotionPipeline(audioBuffer);
  } catch (err) {
    console.error(err);
    setStatus("Could not decode audio.", "error");
  }
}

async function runEmotionPipeline(audioBuffer) {
  const emotionContainer = byId("emotionResult");
  const playlistEl = byId("playlist");
  if (playlistEl) playlistEl.innerHTML = "";

  try {
    let emotion = "calm";

    if (USE_BACKEND) {
      const spec = getSpectrogram(audioBuffer);
      const backendEmotion = await predictViaBackend(spec);
      if (backendEmotion) {
        emotion = backendEmotion;
        setStatus("Prediction complete (backend model).", "ok");
      } else {
        emotion = heuristicEmotion(audioBuffer);
        setStatus(
          "Backend unavailable. Demo heuristic used instead.",
          "warn"
        );
      }
    } else {
      const inputTensor = await audioToInputTensor(audioBuffer);

      if (modelLoaded && model) {
        const prediction = tf.tidy(() => {
          return model.predict(inputTensor);
        });

        const scores = prediction.dataSync();
        prediction.dispose();

        let bestIdx = 0;
        for (let i = 1; i < scores.length; i++) {
          if (scores[i] > scores[bestIdx]) bestIdx = i;
        }
        emotion = EMOTIONS[bestIdx] || "calm";
        setStatus("Prediction complete.", "ok");
      } else {
        emotion = heuristicEmotion(audioBuffer);
        setStatus(
          "Demo heuristic used (add your TensorFlow.js model in /model for real predictions).",
          "warn"
        );
      }
    }

    renderEmotion(emotionContainer, emotion);
    renderMusic(emotion);
  } catch (err) {
    console.error(err);
    setStatus("Error during analysis.", "error");
  }
}

function heuristicEmotion(audioBuffer) {
  const chan = audioBuffer.getChannelData(0);
  let energy = 0;
  let maxAmp = 0;
  const step = Math.max(1, Math.floor(chan.length / 4000));
  for (let i = 0; i < chan.length; i += step) {
    const v = chan[i];
    energy += v * v;
    const av = Math.abs(v);
    if (av > maxAmp) maxAmp = av;
  }
  energy /= chan.length / step;

  if (energy > 0.02 && maxAmp > 0.5) return "excited";
  if (energy > 0.01 && maxAmp > 0.3) return "happy";
  if (energy < 0.003) return "calm";
  if (energy < 0.007) return "sad";
  return "angry";
}

async function audioToInputTensor(audioBuffer) {
  const spec = getSpectrogram(audioBuffer);
  const specTensor = tf.tensor2d(spec);
  const min = specTensor.min();
  const max = specTensor.max();
  const normalized = specTensor.sub(min).div(max.sub(min).add(1e-6));
  const input = normalized.expandDims(0).expandDims(-1);

  min.dispose();
  max.dispose();
  specTensor.dispose();

  return input;
}

function getSpectrogram(audioBuffer) {
  const targetSampleRate = 16000;
  const frameSize = 256;
  const hopSize = 128;
  const numFreqBins = 64;

  const src = audioBuffer.getChannelData(0);
  const srcRate = audioBuffer.sampleRate;

  const downsampleFactor = Math.max(1, Math.floor(srcRate / targetSampleRate));
  const downSampledLength = Math.floor(src.length / downsampleFactor);
  const signal = new Float32Array(downSampledLength);
  for (let i = 0; i < downSampledLength; i++) {
    signal[i] = src[i * downsampleFactor];
  }

  const numFrames = Math.max(1, Math.floor((signal.length - frameSize) / hopSize) + 1);
  const spectrogram = [];

  const hann = new Float32Array(frameSize);
  for (let i = 0; i < frameSize; i++) {
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (frameSize - 1)));
  }

  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * hopSize;
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

    spectrogram.push(Array.from(frameBins));
  }

  const targetFrames = 64;
  const resized = [];
  for (let t = 0; t < targetFrames; t++) {
    const srcIndex = Math.floor((t / targetFrames) * spectrogram.length);
    resized.push(spectrogram[Math.min(srcIndex, spectrogram.length - 1)]);
  }

  return resized;
}

function magnitudeSpectrum(frame) {
  const N = frame.length;
  const half = N / 2;
  const mags = new Float32Array(half);
  for (let k = 0; k < half; k++) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * k * n) / N;
      real += frame[n] * Math.cos(angle);
      imag += frame[n] * Math.sin(angle);
    }
    mags[k] = Math.sqrt(real * real + imag * imag);
  }
  return mags;
}

async function predictViaBackend(spectrogram) {
  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ spectrogram }),
    });

    if (!response.ok) {
      console.error("Backend HTTP error", response.status, await response.text());
      return null;
    }

    const data = await response.json();
    if (!data || !data.emotion) {
      console.error("Backend response missing emotion", data);
      return null;
    }

    return data.emotion;
  } catch (err) {
    console.error("Error calling backend:", err);
    return null;
  }
}

function renderEmotion(container, emotion) {
  if (!container) return;
  container.innerHTML = "";

  const wrapper = document.createElement("div");

  const title = document.createElement("div");
  title.className = "emotion-label";
  title.textContent = `You sound ${emotion}.`;

  const tag = document.createElement("span");
  tag.className = `emotion-tag ${emotion}`;
  tag.textContent = emotion.toUpperCase();

  const subtitle = document.createElement("p");
  subtitle.className = "emotion-subtitle";
  subtitle.textContent =
    "This is an approximate prediction based on a short sample of your voice.";

  title.appendChild(tag);
  wrapper.appendChild(title);
  wrapper.appendChild(subtitle);

  container.appendChild(wrapper);
}

function renderMusic(emotion) {
  const output = byId("musicOutput");
  const listEl = byId("playlist");
  if (!output || !listEl) return;

  listEl.innerHTML = "";

  const list = musicSuggestions[emotion] || musicSuggestions.calm;

  if (!list || !list.length) {
    output.querySelector(".placeholder").textContent =
      "No suggestions available for this emotion.";
    return;
  }

  const placeholder = output.querySelector(".placeholder");
  if (placeholder) {
    placeholder.textContent = "Try these mood-based playlists:";
  }

  list.forEach((item) => {
    const li = document.createElement("li");
    li.className = "playlist-item";

    const title = document.createElement("span");
    title.textContent = item.title;

    const link = document.createElement("a");
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Open on YouTube";

    li.appendChild(title);
    li.appendChild(link);
    listEl.appendChild(li);
  });
}

function initUI() {
  const recordBtn = byId("recordBtn");
  const stopBtn = byId("stopBtn");

  if (recordBtn) recordBtn.addEventListener("click", startRecording);
  if (stopBtn) stopBtn.addEventListener("click", stopRecording);

  setButtonsRecordingState(false);
}

window.addEventListener("DOMContentLoaded", () => {
  initUI();
  loadModel();
  setStatus("Loading emotion model…", "");
});
