import sys
import os
import numpy as np
from scipy.io import wavfile

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .voice_encoder import VoiceEncoder
from drift_detector.drift_signal import EmbeddingDriftDetector

BASE_DIR = os.path.dirname(__file__)

encoder = VoiceEncoder()
detector = EmbeddingDriftDetector(window_size=20, alpha=0.1)

print("Testing voice drift detection...\n")

# Load clean audio
sr, clean_signal = wavfile.read(os.path.join(BASE_DIR, "sample_voice.wav"))

# Create noisy version (simulate drift)
noise = np.random.normal(0, 0.05, clean_signal.shape)
noisy_signal = clean_signal + noise

# Save noisy audio
noisy_path = os.path.join(BASE_DIR, "sample_voice_noisy.wav")
wavfile.write(noisy_path, sr, noisy_signal.astype(np.float32))

# Stream embeddings
for t in range(200):
    if t < 100:
        emb = encoder.encode(os.path.join(BASE_DIR, "sample_voice.wav"))
    else:
        emb = encoder.encode(noisy_path)

    detected, score = detector.update(emb)

    if detected:
        print(f"t = {t} | drift_score = {score:.4f}")

print("\nVoice drift test completed.")
