import os
import shutil
from pathlib import Path

# Root of the extracted RAVDESS dataset (speech). Adjust this if needed.
# Typical structure: audio_speech_actors_01-24/Actor_01/*.wav, etc.
SOURCE_ROOT = Path(r"C:\Users\Sutharshanaram\Downloads\archive (3)\audio_speech_actors_01-24")

# Target folder for our 5-class dataset used by train_emotion_model.py
TARGET_ROOT = Path(r"C:\Users\Sutharshanaram\Desktop\newprojectfun\real_emotion_dataset")

EMOTIONS = ["happy", "sad", "angry", "calm", "excited"]

# RAVDESS emotion code (3rd number in filename) → our 5-class label
# RAVDESS codes:
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad,
# 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
EMOTION_MAP = {
    "01": "calm",      # neutral → calm
    "02": "calm",      # calm → calm
    "03": "happy",     # happy → happy
    "04": "sad",       # sad → sad
    "05": "angry",     # angry → angry
    "06": "excited",   # fearful → excited/high arousal
    "07": "angry",     # disgust → angry-like
    "08": "excited",   # surprised → excited/high arousal
}


def prepare_dataset():
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"SOURCE_ROOT does not exist: {SOURCE_ROOT}")

    # Create target folders
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    for emotion in EMOTIONS:
        (TARGET_ROOT / emotion).mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0

    for wav_path in SOURCE_ROOT.rglob("*.wav"):
        fname = wav_path.name
        stem = fname.split(".")[0]
        parts = stem.split("-")
        if len(parts) < 3:
            print(f"[WARN] Unexpected filename format, skipping: {wav_path}")
            skipped += 1
            continue

        emotion_code = parts[2]
        label = EMOTION_MAP.get(emotion_code)
        if label is None:
            print(f"[WARN] Unknown emotion code {emotion_code} in {wav_path}, skipping")
            skipped += 1
            continue

        dest_folder = TARGET_ROOT / label
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_path = dest_folder / fname

        shutil.copy2(wav_path, dest_path)
        count += 1

    print(f"[INFO] Copied {count} files into {TARGET_ROOT}.")
    print(f"[INFO] Skipped {skipped} files.")


if __name__ == "__main__":
    prepare_dataset()
