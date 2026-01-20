# face_encoder.py

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

class FaceEncoder:
    def __init__(self):
        """
        Face embedding extractor using pretrained CNN
        Output: 512-dimensional embedding
        """
        # Load pretrained ResNet
        self.model = models.resnet18(pretrained=True)

        # Remove classification layer
        self.model.fc = torch.nn.Identity()
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def encode(self, image_path):
        """
        Convert face image into embedding vector
        """
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(x)

        return embedding.squeeze().numpy()
