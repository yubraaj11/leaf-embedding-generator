from services.yolo_initializer_service import YoloInitializer
import os
import cv2

SOURCE_DIR = "/home/ubuntu/Downloads/TEST/SOURCE"
OUTPUT_DIR = "/home/ubuntu/Downloads/TEST/DESTINATION"

yolo_init = YoloInitializer()


def crop_function(crop_dir):
    cropped_img_dir = os.path.join(OUTPUT_DIR, crop_dir)

    if not os.path.isdir(cropped_img_dir):
        os.makedirs(os.path.join(cropped_img_dir))

    for image_name in os.listdir(os.path.join(SOURCE_DIR, crop_dir)):
        image_path = os.path.join(SOURCE_DIR, crop_dir, image_name)
        og_img = cv2.imread(image_path)
        bboxes = yolo_init.generate_bbox(image_path=image_path)

        for i, box in enumerate(bboxes):
            crop_img_path = f"{cropped_img_dir}/{image_name.split('.')[0]}_{i}.JPG"

            with open(crop_img_path, 'w') as cropped_image_path:
                x1, y1, x2, y2 = box
                cropped_img = og_img[int(y1):int(y2), int(x1):int(x2)]

                cv2.imwrite(crop_img_path, img=cropped_img)


if __name__ == "__main__":
    for crop_dir_ in os.listdir(SOURCE_DIR):
        crop_function(crop_dir=crop_dir_)


