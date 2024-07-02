from services.vector_db_services import VectorDatabase
import os

BASE_PATH = "/home/ubuntu/Documents/vertex-projects/image_embedder/images"
FOLDER_PATH = '/home/ubuntu/Downloads/Vector db datas'

if __name__ == "__main__":

    image_paths = []
    for folders in os.listdir(FOLDER_PATH):
        for images in os.listdir(os.path.join(FOLDER_PATH, folders)):
            image_path = os.path.join(FOLDER_PATH, folders, images)
            image_paths.append(image_path)

    test_image_path = "/home/ubuntu/Downloads/dataset_for_resnet/train/tomato_healthy/image (409).JPG"
    vector_db = VectorDatabase()
    # VectorDatabase.create_connection()
    # vector_db.push_to_db(image_path_list=image_paths)
    vector_db.print_values(test_image_path)


