from services.vector_db_services import VectorDatabase
import os

BASE_PATH = "/home/ubuntu/Documents/vertex-projects/image_embedder/images"
FOLDER_PATH = '/home/ubuntu/Downloads/Data for Milvus/Potato'
TEST_FOLDER = '/home/ubuntu/Downloads/Potato_tests'

if __name__ == "__main__":
    # test_list = ['/home/ubuntu/Downloads/TEST/DESTINATION/potato_late_blight/image (201)_0.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_late_blight/image (323)_0.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_late_blight/image (102)_0.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_late_blight/image (135)_0.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_healthy/image (10)_0_augmented_262.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_healthy/image (11)_0_augmented_629.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_healthy/image (16)_0_augmented_971.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_healthy/image (17)_0_augmented_947.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_early_blight/image (18)_0.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_early_blight/image (136)_0.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_early_blight/image (100)_0_augmented_127.JPG',
    #              '/home/ubuntu/Downloads/TEST/DESTINATION/potato_early_blight/image (135)_0.JPG',
    #
    #              ]


    # image_paths = []
    # for folders in os.listdir(FOLDER_PATH):
    #     for images in os.listdir(os.path.join(FOLDER_PATH, folders)):
    #         image_path = os.path.join(FOLDER_PATH, folders, images)
    #         image_paths.append(image_path)
    #
    # image_paths = []
    # for images in os.listdir(FOLDER_PATH):
    #     image_path = os.path.join(FOLDER_PATH, images)
    #     image_paths.append(image_path)

    # test_image_path = "/home/ubuntu/Downloads/TEST/DESTINATION/potato_healthy/image (11)_0_augmented_21.JPG"
    vector_db = VectorDatabase()
    # VectorDatabase.create_connection()
    # vector_db.push_to_db(image_path_list=image_paths)
    #
    # for images in test_list:                      # for list and all
    #     vector_db.print_values(images)

    for images in os.listdir(TEST_FOLDER):
        IMAGE_PATH = os.path.join(TEST_FOLDER, images)
        vector_db.print_values(IMAGE_PATH)
    # vector_db.print_values(test_image_path)


