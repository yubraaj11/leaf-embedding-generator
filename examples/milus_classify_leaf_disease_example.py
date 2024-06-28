from services.milvus_service import Milvus

lde= Milvus()

if __name__ == "__main__":
    # input_image_path = "/home/anish/Documents/vertex_project/yub_image_embed/leaf-embedding-generator/images/apple_H_1.JPG"
    input_image_path= "/home/anish/Downloads/2.jpg"
    collection_name = "apple_healthy_embeddings"
    print(f"For {input_image_path.split('.')[0]} the semantic search results are: \n")
    semantic_results =lde.classify_leaf_disease(input_image_path=input_image_path, collection_name=collection_name)

    semantic_results = semantic_results[0]
    print(semantic_results)

    for result in semantic_results:
        print(f"ID: {result.id}")
        print(f"Image Name: {result.entity.get('image_name')}")
        print(f"Cosine Similarity: {result.distance}")
        print('-' * 10)