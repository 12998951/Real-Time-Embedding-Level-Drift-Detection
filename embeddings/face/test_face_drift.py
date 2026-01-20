import os
import sys
import numpy as np
from PIL import Image

# Confirm execution location
print("RUNNING FILE:", os.path.abspath(__file__))

# Add Code/ to path
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from .face_encoder import FaceEncoder
from drift_detector.drift_signal import EmbeddingDriftDetector

encoder = FaceEncoder()
detector = EmbeddingDriftDetector(window_size=20, alpha=0.1)

print("\nTesting face drift detection...\n")

drift_scores = []

img_path = os.path.join(BASE_DIR, "sample_face.jpg")
image = Image.open(img_path)

for t in range(200):
    if t < 100:
        embedding = encoder.encode(img_path)
    else:
        bright = image.point(lambda p: min(255, p + 40))
        bright_path = os.path.join(BASE_DIR, "sample_face_bright.jpg")
        bright.save(bright_path)
        embedding = encoder.encode(bright_path)

    detected, score = detector.update(embedding)
    drift_scores.append(score)

    if detected:
        print(f"t = {t} | drift_score = {score:.4f}")

# FORCE SAVE
save_path = os.path.join(BASE_DIR, "face_drift_scores.npy")
np.save(save_path, np.array(drift_scores))

print("\nSAVED SUCCESSFULLY →", save_path)
print("TOTAL POINTS:", len(drift_scores))
print("SCRIPT COMPLETED")
