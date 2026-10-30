"""Shared wrapper for Ultralytics models (YOLO, RT-DETR)."""
import numpy as np
import ultralytics
from ultralytics.data.converter import coco80_to_coco91_class

from .base import Detections, Detector


class UltralyticsDetector(Detector):
    def __init__(self, model, ckpt, device="cuda:0", conf=0.001, imgsz=640, max_det=300, **predict_kwargs):
        self.model = model
        self.ckpt = ckpt
        self.device = device
        self.kwargs = dict(conf=conf, imgsz=imgsz, max_det=max_det, device=device, verbose=False,  # fp32 (default)
                           **predict_kwargs)
        self.cls_to_coco = np.array(coco80_to_coco91_class())

    def predict(self, img_bgr):
        r = self.model.predict(img_bgr, **self.kwargs)[0].boxes
        return Detections(
            boxes=r.xyxy.cpu().numpy(),
            scores=r.conf.cpu().numpy(),
            cat_ids=self.cls_to_coco[r.cls.cpu().numpy().astype(int)],
        )

    def config(self):
        cfg = {k: v for k, v in self.kwargs.items() if k != "verbose"}
        return {"name": self.name, "ckpt": self.ckpt, "framework": f"ultralytics {ultralytics.__version__}",
                "precision": "fp32", "resize": f"letterbox, long side {self.kwargs['imgsz']}", **cfg}

    def num_params(self):
        return sum(p.numel() for p in self.model.model.parameters())
