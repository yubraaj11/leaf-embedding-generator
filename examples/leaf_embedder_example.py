from services.leaf_embedder_service import LeafEmbedder
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":
    image_path_1 = "/home/ubuntu/Documents/vertex-projects/image_embedder/images/strawberry_LS_Test.JPG"
    image_path_2 = "/home/ubuntu/Documents/vertex-projects/image_embedder/images/strawberry_LS_test.JPG"

    leaf_embedding = LeafEmbedder()

    embedding_of_image = leaf_embedding.embedding_generator(image_path=image_path_1)
    embedding_of_image_2 = leaf_embedding.embedding_generator(image_path=image_path_2)

    print(cosine_similarity([embedding_of_image], [embedding_of_image_2])[0][0])

    # print(embedding_of_image.shape)
