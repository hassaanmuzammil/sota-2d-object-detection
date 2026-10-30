from .base import Detections, Detector

MODELS = ["yolo26l", "yolo26l-nms", "rtdetr-l", "frcnn"]


def load_detector(name, **kwargs) -> Detector:
    if name == "yolo26l":
        from .yolo import YOLODetector
        return YOLODetector(nms=False, **kwargs)
    if name == "yolo26l-nms":
        from .yolo import YOLODetector
        return YOLODetector(nms=True, **kwargs)
    if name == "rtdetr-l":
        from .rtdetr import RTDETRDetector
        return RTDETRDetector(**kwargs)
    if name == "frcnn":
        from .frcnn import FasterRCNNDetector
        return FasterRCNNDetector(**kwargs)
    raise ValueError(f"unknown model {name!r}, choose from {MODELS}")
