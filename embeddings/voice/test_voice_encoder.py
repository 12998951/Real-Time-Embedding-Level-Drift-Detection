import os
from voice_encoder import VoiceEncoder

BASE_DIR = os.path.dirname(__file__)

encoder = VoiceEncoder()

audio_path = os.path.join(BASE_DIR, "sample_voice.wav")
embedding = encoder.encode(audio_path)

print("Voice embedding shape:", embedding.shape)
print("First 5 values:", embedding[:5])
