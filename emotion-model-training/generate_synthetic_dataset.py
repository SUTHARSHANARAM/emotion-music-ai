import os
import numpy as np
import soundfile as sf

# Root folder where synthetic data will be created
DATA_DIR = r"C:\Users\Sutharshanaram\Desktop\newprojectfun\emotion-model-training\synthetic_data"

EMOTIONS = ["happy", "sad", "angry", "calm", "excited"]
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
SAMPLES_PER_CLASS = 40


def make_signal(emotion: str, idx: int) -> np.ndarray:
    n = int(SAMPLE_RATE * DURATION)
    t = np.linspace(0, DURATION, n, endpoint=False)

    # Base frequencies per emotion (just to create different patterns)
    if emotion == "happy":
        freq = 440.0  # A4
        amp = 0.7
    elif emotion == "sad":
        freq = 220.0  # A3
        amp = 0.4
    elif emotion == "angry":
        freq = 330.0
        amp = 0.9
    elif emotion == "calm":
        freq = 261.63  # C4
        amp = 0.3
    else:  # excited
        freq = 523.25  # C5
        amp = 0.8

    # Simple amplitude envelope to avoid clicks
    envelope = np.linspace(0.1, 1.0, n)
    base = amp * np.sin(2 * np.pi * freq * t) * envelope

    # Add some random noise so samples differ
    noise_level = 0.05
    noise = noise_level * np.random.randn(n)

    signal = base + noise
    # Normalize to avoid clipping
    max_val = np.max(np.abs(signal)) + 1e-6
    signal = signal / max_val
    return signal.astype(np.float32)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for emotion in EMOTIONS:
        folder = os.path.join(DATA_DIR, emotion)
        os.makedirs(folder, exist_ok=True)
        print(f"[INFO] Generating {SAMPLES_PER_CLASS} samples for {emotion} in {folder}")

        for i in range(SAMPLES_PER_CLASS):
            y = make_signal(emotion, i)
            fname = f"{emotion}_{i:03d}.wav"
            path = os.path.join(folder, fname)
            sf.write(path, y, SAMPLE_RATE)

    print("[INFO] Synthetic dataset created at:", DATA_DIR)


if __name__ == "__main__":
    main()
