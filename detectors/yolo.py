"""YOLO26 (Ultralytics). Dual head: one-to-one (NMS-free) and one-to-many (needs NMS)."""
from ultralytics import YOLO

from ._ultralytics import UltralyticsDetector


class YOLODetector(UltralyticsDetector):
    def __init__(self, ckpt="weights/yolo26l.pt", nms=False, iou=0.7, **kwargs):
        # Ultralytics 8.4: nms=None/True -> one-to-many head + NMS; nms=False -> one-to-one head, no NMS.
        kwargs = dict(nms=nms, **({"iou": iou} if nms else {}), **kwargs)
        super().__init__(YOLO(ckpt), ckpt, **kwargs)
        self.name = "yolo26l" + ("-nms" if nms else "")
