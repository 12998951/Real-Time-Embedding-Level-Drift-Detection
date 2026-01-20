import numpy as np
import librosa
from scipy.io import wavfile

class VoiceEncoder:
    def __init__(self, n_mfcc=40):
        self.n_mfcc = n_mfcc

    def encode(self, audio_path):
        """
        Extract MFCC-based voice embedding using a robust WAV loader.
        """
        # Load audio using scipy (robust on Windows)
        sr, signal = wavfile.read(audio_path)

        # Convert to float if needed
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32)

        # If stereo, take one channel
        if signal.ndim > 1:
            signal = signal[:, 0]

        # Normalize
        signal = signal / np.max(np.abs(signal))

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=self.n_mfcc
        )

        # Temporal average → fixed-length embedding
        embedding = np.mean(mfcc, axis=1)

        return embedding
