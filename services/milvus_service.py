from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import os
from services.resnet_image_embedder_service import ResNetImageEmbedder


class Milvus:
    _resnet_instance = None

    def __init__(self, host='localhost', port='19530'):
        connections.connect("default", host=host, port=port)
        if Milvus._resnet_instance is None:
            Milvus._resnet_instance = ResNetImageEmbedder()

    def create_collection(self, class_label, dim):
        """

        :param class_label:
        :param dim:
        :return:
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=255)
        ]
        schema = CollectionSchema(fields, f"Collection for {class_label}")
        collection = Collection(f"{class_label}_embeddings", schema)
        return collection

    def insert_embeddings(self, collection, embeddings, image_names):
        """

        :param collection:
        :param embeddings:
        :param image_names:
        :return:
        """
        collection.insert([embeddings, image_names])
        # index_params = {
        #     # "index_type": "IVF_FLAT",
        #     "index_type": "IVF_FLAT",
        #     "params": {"nlist": 128},
        #     "metric_type": "COSINE"
        # }

        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 32, "efConstruction": 300}  # Adjusted parameters for example
        }
        collection.create_index("embedding", index_params)
        collection.load()
        print(f"Inserted and indexed {len(embeddings)} embeddings into collection {collection.name}")

    def push_to_database(self, base_path):
        """
        Given a directory (plant name) and images, this function will create a collection in the milvus server and
        add the embeddings of the images to its collection.

        :return:
        """
        for class_label in os.listdir(base_path):
            class_path = os.path.join(base_path, class_label)
            if os.path.isdir(class_path):
                collection = self.create_collection(class_label, dim=2048)
                embeddings = []
                image_names = []
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    if os.path.isfile(image_path):
                        embedding = Milvus._resnet_instance.generate_embedding(image_path)
                        embeddings.append(embedding)
                        image_names.append(image_name)
                self.insert_embeddings(collection, embeddings, image_names)

    def classify_leaf_disease(self, input_image_path, collection_name):

        collection =Collection(name=collection_name)
        test_embeddings = Milvus._resnet_instance.generate_embedding(image_path=input_image_path)
        test_embeddings = test_embeddings.reshape(1, -1).tolist()  # Convert the embeddings to a list of lists

        search_params = {
            "metric_type": "COSINE",
            # "params": {"nprobe": 10}
            "params": {"ef": 10}
        }

        results = collection.search(
            data=test_embeddings,  # Query vectors should be a list of vectors
            anns_field="embedding",  # Field name where embeddings are stored
            param=search_params,
            limit=5,  # Number of returned entities
            output_fields=["id", "image_name"]  # Specifies fields to be returned
        )

        return results


