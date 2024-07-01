from services.milvus_service import Milvus

# lde= Milvus()

# if __name__ == "__main__":
#     input_image_path = "/home/anish/Documents/vertex_project/yub_image_embed/leaf-embedding-generator/images/apple_H_1.JPG"
#     input_image_path= "/home/anish/Downloads/2.jpg"
#     collection_name = "apple_healthy_embeddings"
#     print(f"For {input_image_path.split('.')[0]} the semantic search results are: \n")
#     semantic_results =lde.classify_leaf_disease(input_image_path=input_image_path, collection_name=collection_name)
#
#     semantic_results = semantic_results[0]
#     print(semantic_results)
#
#     for result in semantic_results:
#         print(f"ID: {result.id}")
#         print(f"Image Name: {result.entity.get('image_name')}")
#         print(f"Cosine Similarity: {result.distance}")
#         print('-' * 10)

if __name__ == "__main__":
    lde = Milvus()
    input_image_path = "/home/anish/Documents/vertex_project/yub_image_embed/leaf-embedding-generator/dataset/apple_black_rot/image (4).JPG"
    collection_name = "apple_black_rot_embeddings"

    print(f"For {input_image_path.split('/')[-1]} the semantic search results are: \n")
    semantic_results = lde.classify_leaf_disease(input_image_path=input_image_path, collection_name=collection_name,
                                                 top_k=3)

    for result in semantic_results:
        print(f"Collection Name: {result['collection_name']}")
        print(f"Image Name: {result['image_name']}")
        print(f"Cosine Similarity: {result['similarity']}")
        print('-' * 10)