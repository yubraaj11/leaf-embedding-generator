from pymilvus import MilvusClient
from embedding import model_initialization, load_model, embedding_generator


def create_connection():
    try:
        conn = MilvusClient("milvus_demo.db")
        return conn
    except Exception as e:
        return e


def create_collection(conn):
    if conn.has_collection(collection_name="demo_collection"):
        conn.drop_collection(collection_name="demo_collection")
    conn.create_collection(
        collection_name="demo_collection",
        dimension=2048,  # The vectors we will use in this demo has 2048 dimensions
    )


def db_data(image_path_list: list, loaded_model_) -> list:
    data = []  # Initialize the data list outside the loop

    for i in range(len(image_path_list)):
        image_embedding = embedding_generator(embedding_model=loaded_model_, image_path=image_path_list[i])

        path_ = image_path_list[i].split(".")[0]
        img_data = {
            "id": i,
            "vector": image_embedding,
            "Image_name": path_.split("/")[-1],  # Changed to -1 to get the image name correctly
        }

        data.append(img_data)  # Append each dictionary to the data list

    return data


def push_to_db(conn, datas):
    conn.insert(collection_name="demo_collection", data=datas)


def semantic_search(conn, model_, test_image_path_):
    # Generate embeddings for the test image
    test_embeddings = embedding_generator(embedding_model=model_, image_path=test_image_path_)

    # Convert the embeddings to a list of lists
    test_embeddings = test_embeddings.reshape(1, -1).tolist()

    # Perform the search in the collection
    res = conn.search(
        collection_name="demo_collection",  # target collection
        data=test_embeddings,  # query vectors should be a list of vectors
        limit=2,  # number of returned entities
        output_fields=["id", "Image_name"],  # specifies fields to be returned
    )

    return res


def print_values(semantic_search_results_):
    print(f"For {test_image_path.split('.')[0]} the semantic search results are: \n")

    for i in range(len(semantic_search_results)):
        if semantic_search_results[i]['distance'] < 0.8:
            print(f"ID: {None}")
            print(f"Image Name: {None}")
            print(f"Cosine Similarity: {None}")
            print('-' * 10)
            break
        else:
            print(f"ID: {semantic_search_results[i]['id']}")
            print(f"Image Name: {semantic_search_results[i]['entity']['Image_name']}")
            print(f"Cosine Similarity: {semantic_search_results[i]['distance']}")
            print('-' * 10)


if __name__ == "__main__":

    image_paths = [
        "images/orange_1.JPG", "images/orange_2.JPG", "images/potato_EB_1.JPG", "images/potato_LB_1.JPG",
        "images/strawberry_H_1.JPG", "images/strawberry_LS_1.JPG", "images/strawberry_LS_2.JPG",
        "images/strawberry_H_2.JPG", "images/apple_H_1.JPG", "images/cucumber_1.jpeg", "images/cucumber_3.jpeg",
        "images/maize_1.jpeg", "images/maize_2.jpeg"
    ]

    trained_model_path = 'model/resnet50_finetuned_25.pth'

    model = model_initialization()
    loaded_model = load_model(model, trained_model_path)

    client = create_connection()

# Uncomment and Run in order to add new leaf embeddings into database

    create_collection(conn=client)

    data = db_data(image_path_list=image_paths, loaded_model_=loaded_model)
    push_to_db(conn=client, datas=data)                                                                          

    test_image_path = "images/strawberry_LS_test.JPG"

    semantic_search_results = semantic_search(conn=client, model_=loaded_model, test_image_path_=test_image_path)
    semantic_search_results = semantic_search_results[0]
    print_values(semantic_search_results_=semantic_search_results)




