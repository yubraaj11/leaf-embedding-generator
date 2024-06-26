from services.leaf_embedder_service import LeafEmbedder


if __name__ == "__main__":
    image_path_ = "/home/ubuntu/Documents/vertex-projects/image_embedder/images/maple_1.jpeg"

    leaf_embedding = LeafEmbedder()

    embedding_of_image = leaf_embedding.embedding_generator(image_path=image_path_)
    print(embedding_of_image.shape)
