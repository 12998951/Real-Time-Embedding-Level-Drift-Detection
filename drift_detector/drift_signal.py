
import numpy as np


class EmbeddingDriftDetector:
    """Detect mean shifts between reference and current embedding windows."""

    def __init__(self, window_size=20, threshold=0.1):
        if not isinstance(window_size, int) or window_size < 2:
            raise ValueError("window_size must be an integer >= 2")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and >= 0")

        self.window_size = window_size
        self.threshold = float(threshold)
        self.reference_window = []
        self.current_window = []
        self.embedding_shape = None

    def reset(self):
        """Clear both windows and allow a fresh warm-up."""
        self.reference_window.clear()
        self.current_window.clear()
        self.embedding_shape = None

    def update(self, embedding):
        """
        Add one embedding.

        Returns:
            dict with ready, detected, and score fields.
        """
        vector = np.asarray(embedding, dtype=float)

        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding must be a non-empty 1D vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError("embedding contains NaN or infinite values")

        if self.embedding_shape is None:
            self.embedding_shape = vector.shape
        elif vector.shape != self.embedding_shape:
            raise ValueError("embedding dimension changed")

        vector = vector.copy()

        if len(self.reference_window) < self.window_size:
            self.reference_window.append(vector)
            return {
                "ready": False,
                "detected": False,
                "score": None,
            }

        self.current_window.append(vector)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)

        if len(self.current_window) < self.window_size:
            return {
                "ready": False,
                "detected": False,
                "score": None,
            }

        ref_mean = np.mean(self.reference_window, axis=0)
        cur_mean = np.mean(self.current_window, axis=0)
        score = float(np.linalg.norm(ref_mean - cur_mean))

        return {
            "ready": True,
            "detected": score > self.threshold,
            "score": score,
        }
