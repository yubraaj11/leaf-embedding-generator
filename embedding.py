import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"


def model_initialization():
    init_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    num_features = init_model.fc.in_features
    init_model.fc = nn.Linear(num_features, 10)
    init_model = init_model.to(device=device)

    return init_model


def load_model(model_loaded, trained_model_path):
    model_loaded = model_loaded
    model_loaded.load_state_dict(torch.load(trained_model_path, map_location=device))
    model_loaded = nn.Sequential(*list(model_loaded.children())[:-1])  # Remove the last layer
    model_loaded = model_loaded.to(device)
    model_loaded.eval()

    return model_loaded


def embedding_generator(embedding_model, image_path):
    embedding_model = embedding_model

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        embedding = embedding_model(image)
        embedding = embedding.cpu().numpy().flatten()

    return embedding


def similarity_measure(embedding1, embedding2):
    similarity = cosine_similarity([embedding1], [embedding2])
    return similarity[0][0]


if __name__ == "__main__":
    trained_model_path = 'model/resnet50_finetuned_25.pth'

    image_path1 = "images/orange_3.JPG"
    image_path2 = "images/orange_4.JPG"

    model = model_initialization()
    loaded_model = load_model(model, trained_model_path)

    embedding_of_1_image = embedding_generator(embedding_model=loaded_model, image_path=image_path1)
    embedding_of_2_image = embedding_generator(embedding_model=loaded_model, image_path=image_path2)

    cosine_similarity_val = similarity_measure(embedding_of_1_image, embedding_of_2_image)
    print(f"The Similarity Value of two images are: {cosine_similarity_val}")
