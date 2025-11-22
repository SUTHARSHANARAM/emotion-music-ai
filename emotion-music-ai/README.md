# Emotion → Music AI Web App

A mini AI web application that listens to your voice, predicts your **emotion**, and recommends matching **music moods / playlists**. Ideal as a **college mini-project**, **portfolio project**, or a **live demo**.

- **Input:** 1–2 seconds of recorded speech (via microphone)
- **AI Task:** Speech emotion recognition (happy, sad, angry, calm, excited)
- **Output:** Suggested music moods + playlist links (e.g. YouTube, Spotify)

The app uses **TensorFlow.js**, **Web Audio API**, and a **lightweight spectrogram generator** in JavaScript.

---

## 1. Features

- **Voice recording in the browser**
  - Uses Web Audio API and MediaRecorder
  - Records ~2 seconds of audio with a single click

- **AI emotion detection**
  - Designed to work with a small **TensorFlow.js speech emotion model**
  - Current demo includes a **heuristic fallback** (energy-based) if no model is present

- **Emotion → music mapping**
  - Maps predicted emotion (happy / sad / angry / calm / excited) to curated music moods
  - Provides **clickable playlist links** (YouTube search URLs by default)

- **Modern UI**
  - Gradient background and glassmorphism cards
  - Animated “wave bars” while recording
  - Responsive layout (desktop + mobile)

- **Deployment-ready**
  - Pure frontend (HTML/CSS/JS)
  - Can be deployed easily to GitHub Pages, Netlify, Vercel, etc.

---

## 2. Tech Stack

- **Frontend:** HTML, CSS, vanilla JavaScript
- **AI Runtime:** [TensorFlow.js](https://www.tensorflow.org/js)
- **Audio:** Web Audio API (`getUserMedia`, `AudioContext`, `MediaRecorder`)
- **Platform:** Any modern browser (Chrome recommended)

---

## 3. Project Structure

```text
emotion-music-ai/
├── index.html       # UI layout and structure
├── style.css        # Styling (gradients, cards, animations)
├── script.js        # Audio, AI logic, and playlist mapping
└── model/           # TensorFlow.js model files go here
    └── model.json   # (you will add this)
    └── group1-*.bin # (you will add these weight shards)
```

> Note: The repo currently includes an empty `model/` folder ready for your TFJS model.

---

## 4. How It Works (High-Level Flow)

1. **User speaks into the microphone**
   - User clicks **Start Recording** and talks for ~2 seconds.

2. **Audio is captured**
   - The browser records raw audio using **MediaRecorder**.
   - Recorded chunks are combined into an `audio/webm` Blob.

3. **Audio → `AudioBuffer`**
   - The Blob is converted to an `ArrayBuffer`.
   - `AudioContext.decodeAudioData(...)` turns it into an `AudioBuffer` (PCM samples).

4. **`AudioBuffer` → Spectrogram**
   - The signal is **downsampled** (e.g., to ~16 kHz).
   - A simple **STFT**-style process is applied:
     - Windowing (Hann window)
     - Short frames + hop size
     - Magnitude spectrum per frame
   - Magnitude data is compressed into a fixed-size **spectrogram matrix** (e.g., 64×64).

5. **Spectrogram → Tensor**
   - Spectrogram is normalized.
   - Reshaped into a 4D Tensor for TFJS, e.g. `[1, 64, 64, 1]`.

6. **Emotion prediction**
   - If a TFJS model is available at `model/model.json`:
     - `model.predict(inputTensor)` returns a probability vector.
     - Argmax index → emotion label from `['happy','sad','angry','calm','excited']`.
   - If no model is found:
     - A small **heuristic** based on signal energy / loudness estimates an emotion.

7. **Emotion → playlist recommendation**
   - The predicted emotion is used as a key in a `musicSuggestions` object.
   - The app renders 1–2 **playlist suggestions** with links.

8. **Display results in UI**
   - Shows **“You sound happy.”** with an emotion tag.
   - Shows playlist titles and “Open on YouTube” links.

---

## 5. Setup & Requirements

### 5.1. Prerequisites

- **Code editor** (VS Code recommended)
- **Local HTTP server** (any of these):
  - VS Code **Live Server** extension, or
  - Python installed (`python -m http.server`), or
  - Any simple static server
- **Modern browser** (Chrome recommended)

### 5.2. Clone / Copy the Project

Place the `emotion-music-ai` folder on your machine, for example:

```text
C:/Users/You/Desktop/emotion-music-ai
```

---

## 6. Running the App Locally

You must serve the project via `http://localhost` (not `file://`) because of mic permissions and TFJS.

### Option 1 – VS Code Live Server (recommended)

1. Open VS Code.
2. Open the `emotion-music-ai` folder.
3. Install the **Live Server** extension if you don’t have it.
4. Right-click `index.html` → **Open with Live Server**.
5. Your browser opens at something like `http://127.0.0.1:5500/`.

### Option 2 – Python Simple Server

1. Open a terminal inside the `emotion-music-ai` folder.
2. Run:

   ```bash
   python -m http.server 8000
   ```

3. Open your browser at: `http://localhost:8000`.

### Testing Steps

1. Open the web app in your browser.
2. When prompted, **allow microphone access**.
3. Click **Start Recording** and speak normally for about 2 seconds.
4. Click **Stop** (or wait for auto-stop).
5. Observe:
   - The **detected emotion** text.
   - Suggested **music moods and playlist links**.

---

## 7. Integrating Your TensorFlow.js Emotion Model

The app is designed to work with a TFJS **speech emotion classification** model.

### 7.1. Expected Model Location

Place the TFJS model files inside the `model/` folder:

```text
emotion-music-ai/
  model/
    model.json
    group1-shard1.bin
    group1-shard2.bin
    ...
```

In `script.js`, the model is loaded with:

```js
model = await tf.loadLayersModel('model/model.json');
```

### 7.2. Matching Input Shape

The preprocessing currently produces a spectrogram tensor shaped like:

```text
[1, 64, 64, 1]  # (batch, time, frequency, channels)
```

Your TFJS model should either:

- Accept this exact input shape, **or**
- You should adjust the JS preprocessing (or resize) to match your model.

If your model expects a different shape (e.g. `1×128×64×1`), you must:

- Change spectrogram dimensions in `script.js`, or
- Add a `tf.image.resizeBilinear` (or similar) step before prediction.

### 7.3. Class Labels and Order

The app assumes 5 emotions in this order:

```js
const EMOTIONS = ['happy', 'sad', 'angry', 'calm', 'excited'];
```

Make sure this matches the **training label order** of your model. If your model uses a different set/order of emotions, update the `EMOTIONS` array to match.

### 7.4. If You Start from a Python/TensorFlow Model

Typical workflow:

1. **Export Keras model** in Python:

   ```python
   model.save('saved_model_dir')      # or model.save('my_model.h5')
   ```

2. **Install TFJS converter**:

   ```bash
   pip install tensorflowjs
   ```

3. **Convert to TFJS**:

   - SavedModel → TFJS:

     ```bash
     tensorflowjs_converter \
       --input_format=tf_saved_model \
       --output_format=tfjs_layers_model \
       --signature_name=serving_default \
       --saved_model_tags=serve \
       saved_model_dir/ \
       ./tfjs_model_dir/
     ```

   - HDF5 (`.h5`) → TFJS:

     ```bash
     tensorflowjs_converter --input_format=keras my_model.h5 ./tfjs_model_dir/
     ```

4. **Copy the TFJS output**:

   - Move `tfjs_model_dir/model.json` and all `.bin` shards into `emotion-music-ai/model/`.

5. **Update `EMOTIONS` and input shape** (if needed).

---

## 8. Emotion → Music Mapping

The app uses a simple JavaScript object to map each emotion to playlists, e.g.:

- **happy** → upbeat pop / feel-good mixes
- **sad** → soft piano, calm lofi
- **angry** → chillstep, relaxing ambient
- **calm** → rain sounds, meditation music
- **excited** → workout / EDM mixes

You can customize this in `script.js` by editing the `musicSuggestions` object:

- Change titles
- Replace YouTube search URLs with **Spotify** playlists or any music platform

---

## 9. Optional Enhancements (for Extra Credit)

You can extend this project with extra features:

- **Emotion history & trend graph**
  - Store each prediction in `localStorage`.
  - Draw a simple history chart (e.g. using Chart.js or plain canvas).

- **Dark / light mode toggle**
  - Switch CSS themes with a toggle button.

- **Spotify API integration**
  - Use the predicted emotion to search or open real Spotify playlists.

- **Speech-to-text + sentiment combo**
  - Convert voice to text (Web Speech API or external ASR).
  - Run a text sentiment classifier and combine with voice emotion.

- **Waveform / spectrogram visualization**
  - Draw the recorded waveform or spectrogram in a `<canvas>` for better UX.

---

## 10. Deployment

Because the app is just static HTML/CSS/JS, you can deploy it easily:

- **GitHub Pages**
  - Push the project to a GitHub repo.
  - Enable GitHub Pages (e.g. from `main` branch, `/root` folder).

- **Netlify**
  - Drag-and-drop the `emotion-music-ai` folder into Netlify dashboard, or
  - Connect your Git repository.

- **Vercel**
  - Import the project from GitHub.
  - Deploy as a static site in a few clicks.

After deployment, open the live URL, allow microphone access, and you have a **hosted AI demo**.

---

## 11. Using this as a College Mini-Project / Portfolio Project

For reports and viva / presentations, you can highlight:

- Use of **Web Audio API** for real-time audio capture.
- Use of **TFJS** in the browser for on-device AI inference.
- Simple but effective **emotion-to-music mapping**.
- Extensibility towards more advanced multimodal emotion analysis (voice + text).

You can also include:

- Architecture diagrams (frontend, audio pipeline, AI model)
- Data flow diagrams (user → browser → model → recommendation)
- Screenshots of the UI and sample predictions

If you need a full **8–10 page report** (abstract, intro, architecture, workflow, results, conclusion), this README can serve as the base content.

---

## 12. Summary

This project demonstrates how to build a complete **AI-powered web application** that:

- Records audio in the browser
- Processes it into a spectrogram
- Runs a speech emotion classifier using TensorFlow.js
- Maps the emotion to curated music moods and playlists
- Presents everything in a clean, modern UI

You can use it as-is as a **mini-project**, or extend it with more advanced models and integrations.
