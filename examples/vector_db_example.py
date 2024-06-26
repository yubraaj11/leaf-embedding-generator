from services.vector_db_services import VectorDatabase
import os

BASE_PATH = "/home/ubuntu/Documents/vertex-projects/image_embedder/images"


if __name__ == "__main__":

    image_paths = []
    for images in os.listdir(BASE_PATH):
        image_path = os.path.join(BASE_PATH, images)
        image_paths.append(image_path)

    test_image_path = "/home/ubuntu/Documents/vertex-projects/image_embedder/images/strawberry_LS_Test.JPG"
    vector_db = VectorDatabase()
    # VectorDatabase.create_connection()
    # vector_db.push_to_db(image_path_list=image_paths)
    vector_db.print_values(test_image_path)


