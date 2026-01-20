import numpy as np

class EmbeddingDriftDetector:
    def __init__(self, window_size=20, alpha=0.1):
        self.window_size = window_size
        self.alpha = alpha
        self.reference_window = []
        self.current_window = []

    def update(self, embedding):
        """
        Updates the detector with a new embedding and checks for drift.
        """
        # Fill the reference window first
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(embedding)
            return False, 0.0

        # Fill the current sliding window
        self.current_window.append(embedding)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)

        # Once we have enough data, compare windows
        if len(self.current_window) == self.window_size:
            ref_mean = np.mean(self.reference_window, axis=0)
            cur_mean = np.mean(self.current_window, axis=0)
            
            # Calculate Euclidean distance between means (the drift score)
            score = np.linalg.norm(ref_mean - cur_mean)
            
            # Simple thresholding for detection
            detected = score > self.alpha
            return detected, score

        return False, 0.0