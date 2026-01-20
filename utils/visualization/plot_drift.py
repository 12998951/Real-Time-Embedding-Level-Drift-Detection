import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load drift scores
face_scores = np.load(os.path.join(BASE_DIR, "Face", "face_drift_scores.npy"))
voice_scores = np.load(os.path.join(BASE_DIR, "Voice", "voice_drift_scores.npy"))

t = np.arange(len(face_scores))

# -------- Face Drift Plot --------
plt.figure()
plt.plot(t, face_scores)
plt.axvline(x=100, linestyle='--')
plt.xlabel("Time")
plt.ylabel("Drift Magnitude")
plt.title("Face Embedding Drift Detection")
plt.tight_layout()
plt.savefig("face_drift_curve.png")
plt.close()

# -------- Voice Drift Plot --------
plt.figure()
plt.plot(t, voice_scores)
plt.axvline(x=100, linestyle='--')
plt.xlabel("Time")
plt.ylabel("Drift Magnitude")
plt.title("Voice Embedding Drift Detection")
plt.tight_layout()
plt.savefig("voice_drift_curve.png")
plt.close()

print("Plots saved:")
print("- face_drift_curve.png")
print("- voice_drift_curve.png")
