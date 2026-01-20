import os
import sys
import numpy as np

# Confirm execution location
print("RUNNING FILE:", os.path.abspath(__file__))

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from voice_encoder import VoiceEncoder
from drift_detector.drift_signal import EmbeddingDriftDetector


encoder = VoiceEncoder()
detector = EmbeddingDriftDetector(window_size=20, alpha=0.1)

print("\nTesting voice drift detection...\n")

drift_scores = []

clean_audio = os.path.join(BASE_DIR, "sample_voice.wav")
noisy_audio = os.path.join(BASE_DIR, "sample_voice_noisy.wav")

for t in range(200):
    if t < 100:
        embedding = encoder.encode(clean_audio)
    else:
        embedding = encoder.encode(noisy_audio)

    detected, score = detector.update(embedding)
    drift_scores.append(score)

    if detected:
        print(f"t = {t} | drift_score = {score:.4f}")

# Force save
save_path = os.path.join(BASE_DIR, "voice_drift_scores.npy")
np.save(save_path, np.array(drift_scores))

print("\nSAVED SUCCESSFULLY →", save_path)
print("TOTAL POINTS:", len(drift_scores))
print("SCRIPT COMPLETED")
