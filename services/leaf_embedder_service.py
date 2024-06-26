import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.image_processing_util import *

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

try:
    TRAINED_MODEL_PATH = os.path.join(FILE_PATH, 'model', 'resnet50_finetuned_25.pth')
except FileNotFoundError as fn:
    raise FileNotFoundError(f"fine-tuned resnet model missing: {fn}")


class LeafEmbedder:
    _loaded_model = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if LeafEmbedder._loaded_model is None:
            LeafEmbedder._loaded_model = self.load_embedding_model()

    def model_initialization(self):
        """
        Loads and return the pretrained resnet50 model's architecture.
        :return: arch of resnet50
        """
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)
        model = model.to(device=self.device)
        return model

    def load_embedding_model(self):
        """
        Loads and return trained weights and biases after removing last softmax layer.
        :return: trained model without last layer
        """

        loaded_model = self.model_initialization()
        loaded_model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=self.device))
        loaded_model = nn.Sequential(*list(loaded_model.children())[:-1])  # Remove the last layer
        loaded_model = loaded_model.to(self.device)
        loaded_model.eval()

        return loaded_model

    def embedding_generator(self, image_path: str) -> np.ndarray:
        """
        returns vector embedding of image provided
        :param image_path: Path of image to generate embedding
        :return: vector embedding of shape (2048,)
        """

        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        with torch.no_grad():
            embedding = LeafEmbedder._loaded_model(image)
            # embedding = embedding.cpu().numpy().flatten()
            embedding = embedding.cpu().numpy().flatten()

        return embedding


if __name__ == "__main__":
    image_path_ = "images/orange_3.JPG"

    leaf_embedding = LeafEmbedder()

    embedding_of_image = leaf_embedding.embedding_generator(image_path=image_path_)
    print(embedding_of_image.shape)
