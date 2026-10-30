"""RT-DETR (Ultralytics). Query-based decoder, NMS-free: top-k over query scores."""
from ultralytics import RTDETR

from ._ultralytics import UltralyticsDetector


class RTDETRDetector(UltralyticsDetector):
    name = "rtdetr-l"

    def __init__(self, ckpt="weights/rtdetr-l.pt", **kwargs):
        super().__init__(RTDETR(ckpt), ckpt, **kwargs)
