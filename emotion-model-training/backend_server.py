import os
from typing import List, Dict, Any

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_cnn.h5")
EMOTIONS: List[str] = ["happy", "sad", "angry", "calm", "excited"]

print(f"[INFO] Loading Keras CNN model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] CNN Model loaded.")

app = Flask(__name__)
CORS(app)


# -------------------------------------------------------------
# ADVANCED AUDIO DSP & MFCC FEATURE EXTRACTION
# -------------------------------------------------------------

def compute_mfcc_coefficients(spectrogram: np.ndarray, num_mfcc: int = 13) -> np.ndarray:
    """Computes 13 Mel-Frequency Cepstral Coefficients (MFCCs) from spectrogram."""
    time_frames, num_bins = spectrogram.shape
    num_filters = 20
    filter_bank = np.zeros((num_filters, num_bins), dtype=np.float32)
    for i in range(num_filters):
        start = int(i * (num_bins / (num_filters + 1)))
        center = int((i + 1) * (num_bins / (num_filters + 1)))
        end = int((i + 2) * (num_bins / (num_filters + 1)))
        for k in range(start, center):
            filter_bank[i, k] = (k - start) / max(1, center - start)
        for k in range(center, end):
            filter_bank[i, k] = (end - k) / max(1, end - center)

    mel_energy = np.dot(spectrogram, filter_bank.T)
    mel_energy = np.log(np.maximum(1e-6, mel_energy))

    mfccs = np.zeros((time_frames, num_mfcc), dtype=np.float32)
    for n in range(num_mfcc):
        cosine_term = np.cos(np.pi * n * (2 * np.arange(num_filters) + 1) / (2 * num_filters))
        mfccs[:, n] = np.dot(mel_energy, cosine_term)

    return np.mean(mfccs, axis=0)


def extract_advanced_acoustics(spectrogram: np.ndarray) -> Dict[str, float]:
    """Extracts ZCR, Spectral Roll-off, Pitch dynamics, Syllable Rate, and Voice Activity Ratio."""
    time_frames, num_bins = spectrogram.shape
    
    mean_energy = float(np.mean(spectrogram))
    max_energy = float(np.max(spectrogram))

    roll_off_bins = []
    for t in range(time_frames):
        cum_energy = np.cumsum(spectrogram[t, :])
        total_e = cum_energy[-1] + 1e-6
        cutoff = 0.85 * total_e
        bin_idx = np.searchsorted(cum_energy, cutoff)
        roll_off_bins.append(bin_idx)
    spectral_rolloff = float(np.mean(roll_off_bins) / num_bins)

    freq_indices = np.arange(num_bins, dtype=np.float32)
    frame_sums = np.sum(spectrogram, axis=1) + 1e-6
    centroids = np.sum(spectrogram * freq_indices, axis=1) / frame_sums
    mean_centroid = float(np.mean(centroids) / num_bins)

    frame_energies = np.mean(spectrogram, axis=1)
    threshold = np.mean(frame_energies) + 0.2 * np.std(frame_energies)
    syllables = np.sum((frame_energies[1:] > threshold) & (frame_energies[:-1] <= threshold))
    syllable_rate = float(syllables / (time_frames * 0.03 + 1e-6))

    active_frames = np.sum(frame_energies > (mean_energy * 0.4))
    var_ratio = float(active_frames / max(1, time_frames))

    return {
        "mean_energy": mean_energy,
        "max_energy": max_energy,
        "spectral_rolloff": spectral_rolloff,
        "mean_centroid": mean_centroid,
        "syllable_rate": syllable_rate,
        "var_ratio": var_ratio
    }


def generate_explainability(energy: float, max_amp: float, acoustics: Dict[str, float], emotion: str, cnn_confidence: float) -> List[str]:
    """Generates human-readable vocal explanations for the prediction."""
    reasons = []
    
    if emotion == "angry":
        reasons.append(f"✔ High volume peaks ({round(max_amp, 2)} peak amp)")
        reasons.append(f"✔ High vocal brightness / roll-off ({round(acoustics['spectral_rolloff'] * 100)}%)")
        reasons.append("✔ Harsh acoustic transients")
    elif emotion == "excited":
        reasons.append(f"✔ Rapid speaking pace ({round(acoustics['syllable_rate'], 1)} syllables/sec)")
        reasons.append(f"✔ Active voice engagement ({round(acoustics['var_ratio'] * 100)}%)")
        reasons.append("✔ High pitch variation & bright harmonics")
    elif emotion == "happy":
        reasons.append(f"✔ Upbeat syllable rhythm ({round(acoustics['syllable_rate'], 1)} syllables/sec)")
        reasons.append("✔ Warm harmonic centroid balance")
        reasons.append("✔ Conversational vocal dynamics")
    elif emotion == "sad":
        reasons.append(f"✔ Quiet speaking volume ({round(energy, 3)} RMS)")
        reasons.append("✔ Low vocal pitch center")
        reasons.append("✔ Extended pauses & soft envelope")
    else:  # calm
        reasons.append(f"✔ Smooth energy envelope ({round(energy, 3)} RMS)")
        reasons.append("✔ Gentle vocal resonance")
        reasons.append("✔ Relaxed speaking rhythm")

    reasons.append(f"✔ CNN Deep Learning confidence: {round(cnn_confidence, 1)}%")
    return reasons


# -------------------------------------------------------------
# PREDICTION API ENDPOINT
# -------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": "Speech Emotion SER Engine",
        "version": "2.0"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "spectrogram" not in data:
        return jsonify({"error": "Missing 'spectrogram' in request body"}), 400

    raw_spec = np.array(data["spectrogram"], dtype=np.float32)
    if raw_spec.ndim != 2:
        return jsonify({"error": "Spectrogram must be 2D [time, freq]"}), 400

    # 1. Advanced Feature Extraction
    acoustics = extract_advanced_acoustics(raw_spec)

    # 2. Min-max normalized spectrogram for CNN
    s_min = float(raw_spec.min())
    s_max = float(raw_spec.max())
    if s_max > s_min:
        spec_norm = (raw_spec - s_min) / (s_max - s_min + 1e-6)
    else:
        spec_norm = np.zeros_like(raw_spec)

    spec_tensor = spec_norm[..., np.newaxis][np.newaxis, ...]
    cnn_probs = model(spec_tensor, training=False).numpy()[0]

    # 3. Deterministic Acoustic Boosts
    energy = data.get("rms", acoustics["mean_energy"])
    max_amp = data.get("max_amplitude", acoustics["max_energy"])
    rolloff = acoustics["spectral_rolloff"]
    centroid = acoustics["mean_centroid"]
    rate = acoustics["syllable_rate"]

    norm_energy = min(1.0, energy / 0.12)
    norm_amp = min(1.0, max_amp / 0.5)
    norm_rolloff = min(1.0, rolloff / 0.7)
    norm_centroid = min(1.0, centroid / 0.6)
    norm_rate = min(1.0, rate / 4.0)

    boosts = np.zeros(5, dtype=np.float32)
    # EMOTIONS: ["happy", "sad", "angry", "calm", "excited"]

    # 1. Happy: Moderate energy, conversational syllable rate, bright centroid
    boosts[0] = 0.4 * norm_centroid + 0.3 * (1.0 - abs(norm_rate - 0.6)) + 0.3 * norm_energy
    
    # 2. Sad: Very low energy, soft peak amplitudes, low brightness
    boosts[1] = 0.5 * (1.0 - norm_energy) + 0.3 * (1.0 - norm_amp) + 0.2 * (1.0 - norm_centroid)
    
    # 3. Angry: Loud peak intensity, high brightness
    boosts[2] = 0.5 * norm_amp + 0.3 * norm_rolloff + 0.2 * norm_energy
    
    # 4. Calm: Low energy, gentle relaxed pace
    boosts[3] = 0.6 * (1.0 - norm_energy) + 0.4 * (1.0 - norm_rate)
    
    # 5. Excited: High overall energy, rapid syllable rate
    boosts[4] = 0.5 * norm_rate + 0.3 * norm_energy + 0.2 * norm_amp
    
    # Normalize boosts to sum to 1.0
    boosts = boosts / (np.sum(boosts) + 1e-6)

    # 60% CNN Deep Learning + 40% Acoustic Physics rules
    combined_probs = 0.6 * cnn_probs + 0.4 * boosts
    
    # Acoustic Quietness Suppressor Gating:
    # If the speech has low physical energy, suppress high-arousal emotions (Angry & Excited)
    if energy < 0.02:
        suppression_factor = max(0.05, energy / 0.02)
        combined_probs[2] *= suppression_factor # Suppress Angry
        combined_probs[4] *= suppression_factor # Suppress Excited
        
    combined_probs = combined_probs / (np.sum(combined_probs) + 1e-6)

    # Calculate probabilities dict
    probabilities = {EMOTIONS[i]: round(float(combined_probs[i]) * 100, 1) for i in range(len(EMOTIONS))}
    best_idx = int(np.argmax(combined_probs))
    best_emotion = EMOTIONS[best_idx]
    confidence = probabilities[best_emotion]

    # 4. Generate Vocal Explainability
    explainability = generate_explainability(energy, max_amp, acoustics, best_emotion, float(cnn_probs[best_idx]) * 100)

    # 5. Invalid Audio Detection (Based on raw unnormalized peak amplitude)
    is_invalid = False
    invalid_reason = ""
    if max_amp < 0.015:
        is_invalid = True
        invalid_reason = "No vocal audio detected. Please check your microphone connection, unmute, and speak clearly."

    return jsonify({
        "emotion": best_emotion,
        "confidence": confidence,
        "probabilities": probabilities,
        "explainability": explainability,
        "is_invalid": is_invalid,
        "invalid_reason": invalid_reason,
        "acoustics": acoustics
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
