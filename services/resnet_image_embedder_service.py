import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
try:
    TRAINED_MODEL_PATH = os.path.join(FILE_PATH, '..', 'model', 'resnet50_finetuned_25.pth')
except FileNotFoundError as fn:
    raise FileNotFoundError(f"fine-tuned resnet model missing: {fn}")


class ResNetImageEmbedder:
    _model_path = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if ResNetImageEmbedder._model_path is None:
            ResNetImageEmbedder._model_path = self.load_embedding_model()

    def model_initialization(self):
        """

        :return:
        """
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)  # This line can be omitted if not used
        model = model.to(self.device)
        return model

    def load_embedding_model(self):
        """
        Loads and returns trained weights and biases after removing the last softmax layer.
        :return: trained model without the last layer
        """
        model = self.model_initialization()
        state_dict = torch.load(TRAINED_MODEL_PATH, map_location=self.device)
        model.load_state_dict(state_dict)
        embedding_model = nn.Sequential(*list(model.children())[:-1])  # Remove the last layer
        embedding_model = embedding_model.to(self.device)
        embedding_model.eval()
        return embedding_model

    def generate_embedding(self, image_path: str) -> np.ndarray:
        """

        :param image_path:
        :return:
        """
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)  # Add batch dimension and move to device
        with torch.no_grad():
            embedding = ResNetImageEmbedder._model_path(image)
            embedding = embedding.cpu().numpy().flatten()
        return embedding
