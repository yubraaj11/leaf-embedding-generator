from ultralytics import YOLO
import os
import cv2


FILE_PATH = os.path.dirname(os.path.abspath(__file__))

try:
    MODEL_PATH = os.path.join(FILE_PATH, '..', 'model', 'final_model.pt')
except FileNotFoundError as e:
    raise FileNotFoundError(f"File Not found {e}")


class YoloInitializer:
    _yolo_model = None

    def __init__(self):
        if YoloInitializer._yolo_model is None:
            YoloInitializer._yolo_model = YOLO(MODEL_PATH, verbose=False)

    @staticmethod
    def generate_bbox(image_path: str) -> list:
        image = cv2.imread(image_path)
        results = YoloInitializer._yolo_model(image)[0]
        bboxes = results.boxes.xyxy.tolist()
        return bboxes



