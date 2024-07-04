import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
# base_dir = '/home/ubuntu/Downloads/TEST/DESTINATION'
base_dir = '/home/ubuntu/Downloads/TEST/DESTINATION'

train_dir = '/home/ubuntu/Downloads/TEST/TRAIN_TEST_VALID/TRAIN'
test_dir = '/home/ubuntu/Downloads/TEST/TRAIN_TEST_VALID/TEST'
val_dir = '/home/ubuntu/Downloads/TEST/TRAIN_TEST_VALID/VAL'

# Create train, test, and validation directories if they don't exist
for directory in [train_dir, test_dir, val_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to split and copy images to respective folders
def split_and_copy_images(crop_type, images, train_size=0.7, val_size=0.15, test_size=0.15):
    train_images, temp_images = train_test_split(images, train_size=train_size)
    val_images, test_images = train_test_split(temp_images, test_size=test_size / (test_size + val_size))

    # Define paths for each crop type in train, test, and validation directories
    train_crop_dir = os.path.join(train_dir, crop_type)
    val_crop_dir = os.path.join(val_dir, crop_type)
    test_crop_dir = os.path.join(test_dir, crop_type)

    # Create directories if they don't exist
    for directory in [train_crop_dir, val_crop_dir, test_crop_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Copy images to respective directories
    for image in train_images:
        shutil.copy(image, train_crop_dir)
    for image in val_images:
        shutil.copy(image, val_crop_dir)
    for image in test_images:
        shutil.copy(image, test_crop_dir)


if __name__ == "__main__":
    # Iterate over each crop_leaf directory and process images
    for crop_type_ in os.listdir(base_dir):
        crop_dir = os.path.join(base_dir, crop_type_)
        if os.path.isdir(crop_dir):
            images_ = [os.path.join(crop_dir, img) for img in os.listdir(crop_dir)
                       if img.endswith(('jpg', 'jpeg', 'png', 'JPG'))]
            split_and_copy_images(crop_type_, images_)
