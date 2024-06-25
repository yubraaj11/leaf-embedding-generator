import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


class LeafEmbedding:
    def __init__(self, trained_model_path, image_path):
        self.trained_model_path = trained_model_path
        self.image_path = image_path

        self.model = None
        self.num_features = None
        self.loaded_model = None
        self.image = None
        self.transform = None
        self.embedding = None
        self.similarity = None
        self.shape = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def model_initialization(self):
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 10)
        self.model = self.model.to(device=self.device)

        return self.model

    def load_embedding_model(self):
        self.loaded_model = self.model_initialization()
        self.loaded_model.load_state_dict(torch.load(self.trained_model_path, map_location=self.device))
        self.loaded_model = nn.Sequential(*list(self.loaded_model.children())[:-1])  # Remove the last layer
        self.loaded_model = self.loaded_model.to(self.device)
        self.loaded_model.eval()

        return self.loaded_model

    def embedding_generator(self):
        self.image = Image.open(self.image_path).convert('RGB')

        self.loaded_model = self.load_embedding_model()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image = self.transform(self.image).unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        with torch.no_grad():
            self.embedding = self.loaded_model(self.image)
            self.embedding = self.embedding.cpu().numpy().flatten()

        return self.embedding

    # def similarity_cosine(self, emb_1, emb_2):
    #     self.similarity = cosine_similarity(self.emb_1, self.emb_2)
    #     return self.similarity[0][0]


if __name__ == "__main__":
    trained_model_path_ = 'model/resnet50_finetuned_25.pth'

    image_path_ = "images/orange_3.JPG"

    leaf_embedding = LeafEmbedding(trained_model_path=trained_model_path_, image_path=image_path_)

    embedding_of_image = leaf_embedding.embedding_generator()
    print(embedding_of_image.shape)
