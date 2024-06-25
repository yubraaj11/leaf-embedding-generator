from embedding_oop import LeafEmbedding
from pymilvus import MilvusClient


class VectorDatabase:
    def __init__(self, db_name, trained_model_path, collection_name, image_path_list : list):
        self.db_name = db_name
        self.trained_model_path = trained_model_path
        self.collection_name = collection_name
        self.image_path_list = image_path_list

        self.data = None
        self.image_embedding = None
        self.path = None
        self.img_data = None
        self.conn = None
        self.test_obj = None
        self.image_path = None
        self.test_image = None
        self.test_embeddings = None
        self.results = None
        self.semantic_results = None
        self.conn = None

    def create_connection(self):
        try:
            self.conn = MilvusClient(self.db_name)

            if self.conn.has_collection(collection_name=self.collection_name):
                self.conn.drop_collection(collection_name=self.collection_name)
            self.conn.create_collection(
                collection_name="demo_collection",
                dimension=2048,  # The vectors we will use in this demo has 2048 dimensions
            )
            return self.conn

        except Exception as e:
            return e

    def embedding_model(self, image_path):
        self.image_path = image_path
        self.test_obj = LeafEmbedding(self.trained_model_path, self.image_path)
        return self.test_obj.embedding_generator()

    def data_for_db(self):
        # self.create_collection()
        self.data = []

        for i in range(len(self.image_path_list)):
            self.image_embedding = self.embedding_model(image_path=self.image_path_list[i])

            self.path = self.image_path_list[i].split(".")[0]
            self.img_data = {
                "id": i,
                "vector": self.image_embedding,
                "Image_name": self.path.split("/")[-1],  # Changed to -1 to get the image name correctly
            }
            self.data.append(self.img_data)  # Append each dictionary to the data list
        return self.data

    def push_to_db(self):
        self.conn = MilvusClient(self.db_name)
        self.data = self.data_for_db()
        self.conn.insert(collection_name=self.collection_name, data=self.data)

    def semantic_search(self, test_image):
        self.conn = MilvusClient(self.db_name)

        self.test_image = test_image
        self.test_embeddings = self.embedding_model(self.test_image)

        self.test_embeddings = self.test_embeddings.reshape(1, -1).tolist()  # Convert the embeddings to a list of lists

        self.results = self.conn.search(
            collection_name="demo_collection",  # target collection
            data=self.test_embeddings,  # query vectors should be a list of vectors
            limit=2,  # number of returned entities
            output_fields=["id", "Image_name"],  # specifies fields to be returned
        )
        return self.results

    def print_values(self, test_image):
        self.test_image = test_image
        print(f"For {self.test_image.split('.')[0]} the semantic search results are: \n")
        self.semantic_results = self.semantic_search(self.test_image)
        self.semantic_results = self.semantic_results[0]

        for results in self.semantic_results:
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


if __name__ == "__main__":
    image_paths = [
        "images/orange_1.JPG", "images/orange_2.JPG", "images/potato_EB_1.JPG", "images/potato_LB_1.JPG",
        "images/strawberry_H_1.JPG", "images/strawberry_LS_1.JPG", "images/strawberry_LS_2.JPG",
        "images/strawberry_H_2.JPG", "images/apple_H_1.JPG", "images/cucumber_1.jpeg", "images/cucumber_3.jpeg",
        "images/maize_1.jpeg", "images/maize_2.jpeg"
    ]

    trained_model_path = 'model/resnet50_finetuned_25.pth'

    db_name = 'milvus_demo_db'
    collection_name = 'demo_collection'

    test_image_path = "images/strawberry_LS_test.JPG"
    vector_db = VectorDatabase(db_name, trained_model_path, collection_name, image_paths)
    # vector_db.push_to_db()
    vector_db.print_values(test_image_path)


