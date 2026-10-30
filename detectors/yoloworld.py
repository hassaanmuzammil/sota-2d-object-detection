"""YOLO-World v2 (Ultralytics). Open-vocabulary: class head is replaced by CLIP text embeddings of the prompts."""
from ultralytics import YOLOWorld

from ._ultralytics import UltralyticsDetector

COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
          "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
          "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
          "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
          "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


class YOLOWorldDetector(UltralyticsDetector):
    name = "yoloworld-l"

    def __init__(self, ckpt="weights/yolov8l-worldv2.pt", prompts=COCO80, iou=0.7, **kwargs):
        model = YOLOWorld(ckpt)
        model.set_classes(list(prompts))  # prompt i -> class index i; COCO80 order keeps the 80->91 id mapping valid
        super().__init__(model, ckpt, iou=iou, **kwargs)
        self.prompts = list(prompts)

    def config(self):
        return {**super().config(), "prompts": "COCO-80 class names" if self.prompts == COCO80 else self.prompts}
