import os
from typing import List

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# Path to the trained Keras model (created by train_emotion_model.py)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_cnn.h5")

EMOTIONS: List[str] = ["happy", "sad", "angry", "calm", "excited"]

print(f"[INFO] Loading Keras model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded.")

app = Flask(__name__)
CORS(app)  # Allow requests from localhost:8000


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts a 2D spectrogram (list of lists) and returns an emotion label."""
    data = request.get_json(silent=True)
    if not data or "spectrogram" not in data:
        return jsonify({"error": "Missing 'spectrogram' in request body"}), 400

    spec = np.array(data["spectrogram"], dtype=np.float32)
    if spec.ndim != 2:
        return jsonify({"error": "Spectrogram must be 2D [time, freq]"}), 400

    # Expected shape: (64, 64, 1)
    spec = spec[..., np.newaxis]
    spec = spec[np.newaxis, ...]  # (1, H, W, 1)

    preds = model.predict(spec)
    idx = int(np.argmax(preds, axis=-1)[0])

    if 0 <= idx < len(EMOTIONS):
        emotion = EMOTIONS[idx]
    else:
        emotion = "calm"

    return jsonify({"emotion": emotion})


if __name__ == "__main__":
    # Run on localhost:5000
    app.run(host="127.0.0.1", port=5000, debug=False)
