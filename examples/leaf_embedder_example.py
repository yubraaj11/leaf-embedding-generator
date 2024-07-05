from services.leaf_embedder_service import LeafEmbedder
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":
    image_path_1 = "/home/ubuntu/Downloads/TEST/DESTINATION/potato_healthy/image (3)_0_augmented_19.JPG"
    image_path_2 = "/home/ubuntu/Downloads/TEST/DESTINATION/potato_healthy/image (1)_0_augmented_975.JPG"

    leaf_embedding = LeafEmbedder()

    embedding_of_image = leaf_embedding.embedding_generator(image_path=image_path_1)
    embedding_of_image_2 = leaf_embedding.embedding_generator(image_path=image_path_2)

    print(cosine_similarity([embedding_of_image], [embedding_of_image_2])[0][0])

    # print(embedding_of_image.shape)
