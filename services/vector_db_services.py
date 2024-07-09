from services.leaf_embedder_service import LeafEmbedder
from pymilvus import MilvusClient, Collection
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DB_NAME = '/home/ubuntu/Documents/vertex-projects/image_embedder/databases/milvus_demo_db'
COLLECTION_NAME = 'demo_collection'

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

SAVE_DIRECTORY = "/home/ubuntu/Downloads/TEST/emb"

try:
    TRAINED_MODEL_PATH = os.path.join(FILE_PATH, '..', 'model', 'resnet50_finetuned_25ep_26cl.pth')
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
                "Image_name": image_path_list[i].split("/")[-1],  # Changed to -1 to get the image name correctly
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

    @staticmethod
    def get_collection_embeddings(collection_name: str):
        conn = VectorDatabase._conn
        if not conn:
            raise ValueError("Database connection is not initialized.")

        # Assuming the query method and structure provided are correct
        # collection = Collection(name=collection_name)
        entities = conn.query(
            collection_name=collection_name,
            filter="id >= 0",
            output_fields=["vector", "Image_name"]
        )
        embeddings = [np.array(entity["vector"]) for entity in entities]
        # embeddings_path = os.path.join(SAVE_DIRECTORY, 'embeddings.txt')
        # with open(embeddings_path, 'w') as f:
        #     for embedding in embeddings:
        #         np.savetxt(f, embedding[None], fmt='%.6f')  # Save each embedding in a new line

        image_names = [entity["Image_name"] for entity in entities]

        return embeddings, image_names

    def semantic_search(self, test_image: str, top_k: int) -> list:
        """
        Returns semantically similar images and their metadata from vector database
        :param top_k: returns top k embeddings and similarity
        :param test_image: Image for inference
        :return: List of semantically similar images and metadata
        """
        conn = VectorDatabase._conn

        test_embeddings = self.embedding_model(image_path=test_image)
        test_embeddings = test_embeddings.reshape(1, -1).tolist()  # Convert the embeddings to a list of lists

        # results = conn.search(
        #     collection_name=COLLECTION_NAME,   # target collection
        #     data=test_embeddings,  # query vectors should be a list of vectors
        #     limit=3,  # number of returned entities
        #     output_fields=["id", "Image_name"],  # specifies fields to be returned
        # )
        # return results

        embeddings, image_names = self.get_collection_embeddings(collection_name=COLLECTION_NAME)  # both are list
        if not embeddings:
            return []
        similarities = cosine_similarity(test_embeddings, embeddings)[0]
        results = [
            {
                "collection_name": COLLECTION_NAME,
                "image_name": image_names[idx],
                "similarity": similarities[idx]
            }
            for idx in range(len(embeddings))
        ]
        results.sort(key=lambda x: x["similarity"], reverse=True)
        # logger.info("Closest image embedding of the input image has been found successfully")
        return results[:top_k]

    def print_values(self, test_image):
        """
        Print Semantically similar images
        :param test_image: path to test image
        :return: None
        """
        print(f"For {test_image.split('.')[0]} the semantic search results are: \n")
        semantic_results = self.semantic_search(test_image=test_image, top_k=3)

        # semantic_results = semantic_results[0]

        # for results in semantic_results:
        #     if results['distance'] < 0.8:
        #         print(f"ID: {None}")
        #         print(f"Image Name: {None}")
        #         print(f"Cosine Similarity: {None}")
        #         print('-' * 10)
        #         break
        #     else:
        #         print(f"ID: {results['id']}")
        #         print(f"Image Name: {results['entity']['Image_name']}")
        #         print(f"Cosine Similarity: {results['distance']}")
        #         print('-' * 10)

        for result in semantic_results:
            # print(f"Collection Name: {result['collection_name']}")
            print(f"Image Name: {result['image_name']}")
            print(f"Cosine Similarity: {result['similarity']}")
            print('-' * 10)
