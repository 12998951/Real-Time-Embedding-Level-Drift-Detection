from .face_encoder import FaceEncoder

encoder = FaceEncoder()

embedding = encoder.encode("sample_face.jpg")

print("Face embedding shape:", embedding.shape)
print("First 5 values:", embedding[:5])
