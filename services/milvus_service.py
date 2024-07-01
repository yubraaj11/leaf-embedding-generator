from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import os
from services.resnet_image_embedder_service import ResNetImageEmbedder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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

        index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
            "metric_type": "L2"
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
                        image_names.append(os.path.splitext(image_name)[0])
                self.insert_embeddings(collection, embeddings, image_names)

    def get_collection_embeddings(self, collection_name):
        collection = Collection(collection_name)
        entities = collection.query(expr="id >= 0", output_fields=["embedding", "image_name"])
        embeddings = [np.array(entity["embedding"]) for entity in entities]
        image_names = [entity["image_name"] for entity in entities]
        return embeddings, image_names

    def classify_leaf_disease(self, input_image_path, collection_name, top_k):

        # collection =Collection(name=collection_name)
        test_embeddings = Milvus._resnet_instance.generate_embedding(image_path=input_image_path)
        embeddings, image_names = self.get_collection_embeddings(collection_name)

        similarities = cosine_similarity([test_embeddings], embeddings)[0]
        results = [
            {
                "collection_name": collection_name,
                "image_name": image_names[idx],
                "similarity": similarities[idx]
            }
            for idx in range(len(embeddings))
        ]
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]

