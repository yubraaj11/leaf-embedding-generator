from sklearn.metrics.pairwise import cosine_similarity
from resnetimageembedder import ResNetImageEmbedder
from services.milvus_service import Milvus


class SimilaritySearch:
    def __init__(self, model_path):
        self.embedder = ResNetImageEmbedder(model_path)
        self.milvus = Milvus()

    def find_similar_images(self, image_path, top_k):
        """

        :param image_path:
        :param top_k:
        :return:
        """
        input_embedding = self.embedder.generate_embedding(image_path)
        all_collections = self.milvus.get_all_collections()
        results = []

        for collection_name in all_collections:
            embeddings, image_names = self.milvus.get_collection_embeddings(collection_name=collection_name)
            similarities = cosine_similarity([input_embedding], embeddings)[0]

            for idx in range(len(embeddings)):
                results.append({
                    "collection_name": collection_name,
                    "image_name": image_names[idx],
                    "similarity": similarities[idx]
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)

        for i, result in enumerate(results[:top_k]):
            # print(f"Rank {i + 1}:")
            print(f"\n")
            print(f"  Collection Name: {result['collection_name']}")
            print(f"  Image Name: {result['image_name']}")
            print(f"  Similarity: {result['similarity']}")
            # print(f"\n")


if __name__ == "__main__":
    model_path = "/home/anish/Documents/vertex_project/yub_image_embed/leaf-embedding-generator/model/resnet50_finetuned.pth"
    image_path= "/dataset/apple_healthy/image (13).JPG"
    searcher = SimilaritySearch(model_path)
    searcher.find_similar_images(image_path=image_path, top_k=3)
