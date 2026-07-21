import os
import numpy as np
from scipy.io import wavfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "synthetic_data")
EMOTIONS = ["happy", "sad", "angry", "calm", "excited"]
SAMPLE_RATE = 16000
DURATION = 2.0
SAMPLES_PER_CLASS = 100


def make_speech_signal(emotion: str, idx: int) -> np.ndarray:
    n = int(SAMPLE_RATE * DURATION)
    t = np.linspace(0, DURATION, n, endpoint=False)

    syllable_rate = np.random.uniform(2.5, 4.5)
    syllables = np.maximum(0.0, np.sin(2 * np.pi * syllable_rate * t)) ** 2

    if emotion == "angry":
        f0 = 220 + 40 * np.sin(2 * np.pi * 3 * t) + np.random.uniform(-10, 10)
        harmonics = (
            0.8 * np.sin(2 * np.pi * f0 * t) +
            0.6 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.4 * np.sin(2 * np.pi * 3.5 * f0 * t)
        )
        noise = 0.15 * np.random.randn(n)
        amplitude = 0.85
        signal = amplitude * (harmonics + noise) * syllables

    elif emotion == "excited":
        f0 = 260 + 80 * np.sin(2 * np.pi * 4 * t)
        harmonics = (
            0.7 * np.sin(2 * np.pi * f0 * t) +
            0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.3 * np.sin(2 * np.pi * 4 * f0 * t)
        )
        amplitude = 0.75
        signal = amplitude * harmonics * syllables

    elif emotion == "happy":
        f0 = 200 + 50 * np.sin(2 * np.pi * 2.5 * t)
        harmonics = (
            0.7 * np.sin(2 * np.pi * f0 * t) +
            0.4 * np.sin(2 * np.pi * 2 * f0 * t)
        )
        amplitude = 0.65
        signal = amplitude * harmonics * syllables

    elif emotion == "calm":
        f0 = 130 + 10 * np.sin(2 * np.pi * 1.0 * t)
        harmonics = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 2 * f0 * t)
        amplitude = 0.35
        smooth_env = np.sin(np.pi * t / DURATION) ** 0.5
        signal = amplitude * harmonics * smooth_env

    else:  # sad
        f0 = np.linspace(140, 110, n)
        harmonics = 0.3 * np.sin(2 * np.pi * f0 * t)
        amplitude = 0.25
        smooth_env = np.sin(np.pi * t / DURATION) ** 0.8
        signal = amplitude * harmonics * smooth_env

    signal += 0.01 * np.random.randn(n)
    max_val = np.max(np.abs(signal)) + 1e-6
    signal = (signal / max_val * 0.9).astype(np.float32)
    return signal


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for emotion in EMOTIONS:
        folder = os.path.join(DATA_DIR, emotion)
        os.makedirs(folder, exist_ok=True)
        print(f"[INFO] Generating {SAMPLES_PER_CLASS} speech samples for {emotion} in {folder}")

        for i in range(SAMPLES_PER_CLASS):
            y = make_speech_signal(emotion, i)
            fname = f"{emotion}_{i:03d}.wav"
            path = os.path.join(folder, fname)
            wavfile.write(path, SAMPLE_RATE, (y * 32767).astype(np.int16))

    print("[INFO] Speech dataset created at:", DATA_DIR)


if __name__ == "__main__":
    main()
