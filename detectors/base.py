"""Common detector interface. Every detector returns boxes in original-image pixels with COCO (91-id) category ids."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy, original image pixels
    scores: np.ndarray  # (N,)
    cat_ids: np.ndarray  # (N,) COCO category ids (1..90)


class Detector:
    name = "base"

    def predict(self, img_bgr: np.ndarray) -> Detections:
        """Run one HxWx3 uint8 BGR image (as read by cv2) end to end: preprocess, forward, postprocess."""
        raise NotImplementedError

    def config(self) -> dict:
        """Settings saved into the results JSON for reproducibility."""
        raise NotImplementedError

    def num_params(self) -> int:
        raise NotImplementedError
