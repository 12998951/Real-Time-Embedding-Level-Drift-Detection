import numpy as np
from drift_signal import EmbeddingDriftDetector

# Initialize drift detector
detector = EmbeddingDriftDetector(window_size=20, alpha=0.1)

print("Starting drift detection test...\n")

# Simulated embedding stream
for t in range(200):
    # Before drift
    if t < 100:
        embedding = np.random.normal(loc=0.0, scale=1.0, size=128)
    # After drift
    else:
        embedding = np.random.normal(loc=3.0, scale=1.0, size=128)

    detected, score = detector.update(embedding)

    if detected:
        print(f"t = {t:03d} | drift_score = {score:.4f}")

print("\nTest completed.")
