import os
import numpy as np

# Compatibility patch for older libraries (like tensorflowjs) that still
# reference deprecated aliases np.object and np.bool, which were removed
# in newer NumPy versions.
if not hasattr(np, "object"):
    np.object = object  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

import tensorflow as tf
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
# Set this to the root folder of your dataset.
# Expected structure:
#   DATA_DIR/
#     happy/*.wav
#     sad/*.wav
#     angry/*.wav
#     calm/*.wav
#     excited/*.wav
# For this project we can use either a synthetic dataset (synthetic_data)
# or a real speech emotion dataset. Update DATA_DIR as needed.
from scipy.io import wavfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "synthetic_data")

EMOTIONS = ["happy", "sad", "angry", "calm", "excited"]

TARGET_SR = 16000
FRAME_SIZE = 256
HOP_SIZE = 128
NUM_FREQ_BINS = 64
TARGET_FRAMES = 64
# -------------------------------------------------------------
# AUDIO -> SPECTROGRAM (MATCHES BROWSER PIPELINE SHAPE)
# -------------------------------------------------------------

def compute_js_style_spectrogram(y: np.ndarray, sr: int) -> np.ndarray | None:
    """Approximate the same spectrogram pipeline used in script.js.

    Returns a (64, 64, 1) float32 array normalized to [0, 1],
    or None if the audio is too short.
    """
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    downsample_factor = max(1, sr // TARGET_SR)
    down_len = len(y) // downsample_factor
    if down_len == 0:
        return None

    signal = y[0 : down_len * downsample_factor : downsample_factor].astype(np.float32)

    hann = np.hanning(FRAME_SIZE).astype(np.float32)

    if len(signal) < FRAME_SIZE:
        padded = np.zeros(FRAME_SIZE, dtype=np.float32)
        padded[: len(signal)] = signal * hann[: len(signal)]
        signal = padded

    num_frames = max(1, (len(signal) - FRAME_SIZE) // HOP_SIZE + 1)
    spectrogram = []

    for frame_idx in range(num_frames):
        start = frame_idx * HOP_SIZE
        frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        end = min(start + FRAME_SIZE, len(signal))
        length = end - start
        if length <= 0:
            continue

        frame[:length] = signal[start:end] * hann[:length]

        mags = np.abs(np.fft.rfft(frame))
        mags = mags[: FRAME_SIZE // 2]

        frame_bins = np.zeros(NUM_FREQ_BINS, dtype=np.float32)
        for b in range(NUM_FREQ_BINS):
            src_index = int((b / NUM_FREQ_BINS) * len(mags))
            if src_index >= len(mags):
                src_index = len(mags) - 1
            frame_bins[b] = np.log1p(mags[src_index])

        spectrogram.append(frame_bins)

    if not spectrogram:
        return None

    spectrogram = np.array(spectrogram, dtype=np.float32)

    resized = np.zeros((TARGET_FRAMES, NUM_FREQ_BINS), dtype=np.float32)
    for t in range(TARGET_FRAMES):
        src_index = int((t / TARGET_FRAMES) * len(spectrogram))
        if src_index >= len(spectrogram):
            src_index = len(spectrogram) - 1
        resized[t] = spectrogram[src_index]

    spec_min = float(resized.min())
    spec_max = float(resized.max())
    norm = (resized - spec_min) / (spec_max - spec_min + 1e-6)

    return norm[..., np.newaxis].astype(np.float32)


# -------------------------------------------------------------
# DATASET LOADING
# -------------------------------------------------------------

def load_dataset():
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for label_idx, emotion in enumerate(EMOTIONS):
        folder = os.path.join(DATA_DIR, emotion)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found for emotion '{emotion}': {folder}")
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".wav"):
                continue

            path = os.path.join(folder, fname)
            try:
                sr, audio_data = wavfile.read(path)
                audio = audio_data.astype(np.float32) / 32767.0
            except Exception as e:
                print(f"[WARN] Could not load {path}: {e}")
                continue

            spec = compute_js_style_spectrogram(audio, sr)
            if spec is None:
                print(f"[WARN] Audio too short or invalid, skipping: {path}")
                continue

            X_list.append(spec)
            y_list.append(label_idx)

    if not X_list:
        raise RuntimeError("No valid audio files found. Check DATA_DIR and folder structure.")

    X = np.stack(X_list).astype(np.float32)
    labels = np.array(y_list, dtype=np.int32)
    return X, labels


# -------------------------------------------------------------
# MODEL DEFINITION
# -------------------------------------------------------------

def build_model(input_shape, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="spectrogram")

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="emotion_logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# -------------------------------------------------------------
# MAIN TRAINING + CONVERSION PIPELINE
# -------------------------------------------------------------

def main():
    if DATA_DIR == "PATH_TO_YOUR_DATASET":
        raise RuntimeError("Please edit DATA_DIR at the top of this file to point to your dataset root.")

    print(f"[INFO] Loading dataset from: {DATA_DIR}")
    X, labels = load_dataset()
    print(f"[INFO] Loaded {X.shape[0]} samples.")
    print("[INFO] X shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = build_model(input_shape=X.shape[1:], num_classes=len(EMOTIONS))
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=16,
    )

    # Save Keras model
    model.save("emotion_cnn.h5")
    print("[INFO] Saved Keras model to emotion_cnn.h5")

    # Convert and save TFJS model
    try:
        # Some older TensorFlow.js / tensorflow_hub versions expect
        # tf.compat.v1.estimator to exist. Newer TF removed it, so we
        # provide a minimal stub to keep imports working.
        if not hasattr(tf.compat.v1, "estimator"):
            class _DummyExporter:
                pass

            class _DummyEstimatorModule:
                Exporter = _DummyExporter

            tf.compat.v1.estimator = _DummyEstimatorModule()  # type: ignore[attr-defined]

        import tensorflowjs as tfjs  # imported lazily to avoid startup issues

        tfjs.converters.save_keras_model(model, "tfjs_model")
        print("[INFO] Saved TFJS model to ./tfjs_model (model.json + shards)")
    except Exception as e:
        print("[WARN] Could not convert model to TFJS automatically:", e)
        print("       Keras model is still saved as emotion_cnn.h5.")
        print("       You can try running this command manually:")
        print("       tensorflowjs_converter --input_format=keras emotion_cnn.h5 ./tfjs_model")


if __name__ == "__main__":
    main()
