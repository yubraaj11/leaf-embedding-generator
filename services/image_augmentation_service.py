import os
import cv2
from PIL import Image
import numpy as np
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomBrightnessContrast, GaussianBlur, Compose, HueSaturationValue,
)

BASE_DIR = '/home/ubuntu/Downloads/TEST/DESTINATION'

augmentation_pipeline = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    GaussianBlur(blur_limit=(3, 7), p=0.15),
    RandomBrightnessContrast(p=0.09),
])


def augment_images_in_directory(directory, target_range=(1050, 1200)):

    image_files = [f for f in os.listdir(directory) if
                   os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(image_files)

    if num_images < 100 or num_images > 900:
        print(f"{directory} out of range ")
        return

    target_count = np.random.randint(target_range[0], target_range[1] + 1)
    num_to_generate = target_count - num_images

    if num_to_generate <= 0:
        print(f"{directory} has nothing to generate")
        return
    print(f"Generating {num_to_generate} numbers of augmented images")

    i = 0
    while num_to_generate > 0:
        image_file = image_files[i % num_images]
        file_path = os.path.join(directory, image_file)

        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to read {file_path}. Skipping.")
            continue
        # Perform augmentation
        augmented = augmentation_pipeline(image=image)['image']
        # Create augmented image filename
        base_name, ext = os.path.splitext(image_file)
        augmented_filename = f"{base_name}_augmented_{i}{ext}"
        augmented_file_path = os.path.join(directory, augmented_filename)
        # Save the augmented image
        cv2.imwrite(augmented_file_path, augmented)
        num_to_generate -= 1
        i += 1


if __name__ == "__main__":
    for crop_type in os.listdir(BASE_DIR):
        crop_dir = os.path.join(BASE_DIR, crop_type)
        if os.path.isdir(crop_dir):
            augment_images_in_directory(directory=crop_dir)
