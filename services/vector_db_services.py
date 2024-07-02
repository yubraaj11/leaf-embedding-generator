from services.leaf_embedder_service import LeafEmbedder
from pymilvus import MilvusClient
import os
import numpy as np

DB_NAME = '/home/ubuntu/Documents/vertex-projects/image_embedder/databases/milvus_demo_db'
COLLECTION_NAME = 'demo_collection'

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

try:
    TRAINED_MODEL_PATH = os.path.join(FILE_PATH, '..', 'model', 'resnet50_finetuned_25.pth')
except FileNotFoundError as fn:
    raise FileNotFoundError(f"fine-tuned resnet model missing: {fn}")


class VectorDatabase:
    _conn = None

    def __init__(self):
        if VectorDatabase._conn is None:
            VectorDatabase._conn = MilvusClient(DB_NAME)

    @staticmethod
    def create_connection():
        try:
            conn = VectorDatabase._conn

            if conn.has_collection(collection_name=COLLECTION_NAME):
                conn.drop_collection(collection_name=COLLECTION_NAME)
            conn.create_collection(
                collection_name="demo_collection",
                dimension=2048,  # The vectors we will use in this demo has 2048 dimensions
            )
            return conn
        except Exception as e:
            return e

    @staticmethod
    def embedding_model(image_path: str) -> np.ndarray:
        """
        Returns embedding vector of given image
        :param image_path: path of image to generate embedding
        :return: vector
        """
        test_obj = LeafEmbedder()
        embeddings = test_obj.embedding_generator(image_path=image_path)
        return embeddings

    def data_for_db(self, image_path_list: list) -> list:
        """
        Returns list of data for vector database
        :param image_path_list: List of path of each images
        :return: list with dictionary
        """
        # self.create_collection()
        data = []
        for i in range(len(image_path_list)):
            image_embedding = self.embedding_model(image_path=image_path_list[i])

            img_data = {
                "id": i,
                "vector": image_embedding,
                "Image_name": image_path_list[i].split("/")[-2],  # Changed to -1 to get the image name correctly
            }
            data.append(img_data)  # Append each dictionary to the data list
        return data

    def push_to_db(self, image_path_list: list):
        """
        pushes data into vector database
        :return:None
        """
        conn = VectorDatabase._conn
        data = self.data_for_db(image_path_list)
        conn.insert(collection_name=COLLECTION_NAME, data=data)
        print(f"{len(data)} : Values pushed to DB")

    def semantic_search(self, test_image: str) -> list:
        """
        Returns semantically similar images and their metadata from vector database
        :param test_image: Image for inference
        :return: List of semantically similar images and metadata
        """
        conn = VectorDatabase._conn

        test_embeddings = self.embedding_model(image_path=test_image)
        test_embeddings = test_embeddings.reshape(1, -1).tolist()  # Convert the embeddings to a list of lists
        results = conn.search(
            collection_name=COLLECTION_NAME,   # target collection
            data=test_embeddings,  # query vectors should be a list of vectors
            limit=5,  # number of returned entities
            output_fields=["id", "Image_name"],  # specifies fields to be returned
        )
        return results

    def print_values(self, test_image):
        """
        Print Semantically similar images
        :param test_image: path to test image
        :return: None
        """
        print(f"For {test_image.split('.')[0]} the semantic search results are: \n")
        semantic_results = self.semantic_search(test_image=test_image)
        semantic_results = semantic_results[0]

        for results in semantic_results:
            if results['distance'] < 0.8:
                print(f"ID: {None}")
                print(f"Image Name: {None}")
                print(f"Cosine Similarity: {None}")
                print('-' * 10)
                break
            else:
                print(f"ID: {results['id']}")
                print(f"Image Name: {results['entity']['Image_name']}")
                print(f"Cosine Similarity: {results['distance']}")
                print('-' * 10)
