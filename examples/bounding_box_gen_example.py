# from services.bounding_box_gen_service import YoloInitializer
from ultralytics import YOLO
import os
import cv2
from tqdm import tqdm

MODEL_PATH = "/home/ubuntu/Documents/vertex-projects/image_embedder/model/final_model.onnx"
SOURCE_DIR = "/home/ubuntu/Downloads/TEST/SOURCE"
OUTPUT_DIR = "/home/ubuntu/Downloads/TEST/DESTINATION"

model = YOLO(MODEL_PATH, verbose=False)


def crop_function(crop_dir):

    cropped_img_dir = os.path.join(OUTPUT_DIR, crop_dir)

    if not os.path.isdir(cropped_img_dir):
        os.makedirs(os.path.join(cropped_img_dir))

    for image_name in tqdm(os.listdir(os.path.join(SOURCE_DIR, crop_dir))):
        image_path = os.path.join(SOURCE_DIR, crop_dir, image_name)
        og_img = cv2.imread(image_path)
        results = model(og_img)[0]
        bboxes = results.boxes.xyxy.tolist()

        for i, box in enumerate(bboxes):
            crop_img_path = f"{cropped_img_dir}/{image_name.split('.')[0]}_{i}.JPG"
            x1, y1, x2, y2 = box
            cropped_img = og_img[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite(crop_img_path, img=cropped_img)


if __name__ == "__main__":
    for crop_dir_ in tqdm(os.listdir(SOURCE_DIR)):
        crop_function(crop_dir=crop_dir_)


