from services.resnet_image_embedder_service import ResNetImageEmbedder

rnie = ResNetImageEmbedder()
if __name__ == "__main__":
    image_path = "/home/anish/Downloads/strawbery.JPG"
    img_embedding = rnie.generate_embedding(image_path)
    print(img_embedding)
    print(img_embedding.shape)