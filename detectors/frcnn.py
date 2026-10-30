"""Faster R-CNN ResNet50-FPN v2 (torchvision). Two-stage, anchor-based, NMS in RPN and box head."""
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

from .base import Detections, Detector


class FasterRCNNDetector(Detector):
    name = "frcnn-r50-fpn-v2"

    def __init__(self, device="cuda:0", conf=0.001, nms_iou=0.5, max_det=100, min_size=800, max_size=1333):
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.device = device
        self.cfg = dict(conf=conf, nms_iou=nms_iou, max_det=max_det, min_size=min_size, max_size=max_size)
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights, box_score_thresh=conf, box_nms_thresh=nms_iou,
            box_detections_per_img=max_det, min_size=min_size, max_size=max_size,
        ).eval().to(device)

    @torch.inference_mode()
    def predict(self, img_bgr):
        x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).to(self.device)
        x = x.permute(2, 0, 1).float().div_(255)
        out = self.model([x])[0]  # GeneralizedRCNNTransform resizes/normalizes; boxes come back in original coords
        return Detections(
            boxes=out["boxes"].cpu().numpy(),
            scores=out["scores"].cpu().numpy(),
            cat_ids=out["labels"].cpu().numpy(),  # torchvision labels are already COCO 91 ids
        )

    def config(self):
        return {"name": self.name, "ckpt": f"torchvision {self.weights}", "framework": f"torchvision {torchvision.__version__}",
                "precision": "fp32", "device": self.device,
                "resize": f"short side {self.cfg['min_size']}, long side <= {self.cfg['max_size']}", **self.cfg}

    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())
