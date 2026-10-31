"""Fine-tuning study: evaluate zero-shot or fine-tuned YOLO26-L on the drive-disjoint KITTI test split.

Same metrics, settings and timing as eval.py (baseline and extended studies); only the GT file and the class -> COCO-id map differ.
COCO weights (80 classes) map through coco80_to_coco91_class (bicycle = id 2 = cyclist); fine-tuned weights map
person/cyclist/car -> 1/2/3.

python finetune/evaluate.py --tag zs640                                         # zero-shot, COCO weights
python finetune/evaluate.py --tag zs1280 --imgsz 1280
python finetune/evaluate.py --tag ft640 --weights results/finetune/train/ft640/weights/best.pt
python finetune/evaluate.py --tag ft640 --weights ... --gt test-cyclist           # also score Cyclist
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detectors.yolo import YOLODetector  # noqa: E402
from eval import STAT_NAMES, per_class_ap  # noqa: E402

GT = {
    "test": "data/kitti_finetune/test_coco.json",
    "test-cyclist": "data/kitti_finetune/test_coco_cyclist.json",
    "val": "data/kitti_finetune/val_coco.json",
}
IMG_DIR = "data/kitti/training/image_2"  # original images (DontCare not grayed out)
FT_NAME_TO_COCO = {"person": 1, "cyclist": 2, "car": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run name, e.g. zs640 / ft640")
    ap.add_argument("--weights", default="weights/yolo26l.pt")
    ap.add_argument("--gt", default="test", choices=GT)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nms", action="store_true", help="one-to-many head + NMS 0.7 (default: NMS-free head)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="results/finetune")
    args = ap.parse_args()

    coco_gt = COCO(GT[args.gt])
    img_ids = sorted(coco_gt.getImgIds())
    keep_cats = set(coco_gt.getCatIds())

    det = YOLODetector(ckpt=args.weights, nms=args.nms, imgsz=args.imgsz)
    names = det.model.names
    if len(names) != 80:  # fine-tuned head
        lut = np.zeros(len(names), dtype=int)
        for i, n in names.items():
            lut[i] = FT_NAME_TO_COCO[n]
        det.cls_to_coco = lut
    tag = f"{args.tag}{'-nms' if args.nms else ''}_{args.gt}"

    torch.cuda.reset_peak_memory_stats()
    preds, times = [], []
    for i, img_id in enumerate(img_ids):
        img = cv2.imread(str(Path(IMG_DIR) / coco_gt.imgs[img_id]["file_name"]))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d = det.predict(img)
        torch.cuda.synchronize()
        if i >= args.warmup:
            times.append((time.perf_counter() - t0) * 1000)
        for (x1, y1, x2, y2), s, c in zip(d.boxes.tolist(), d.scores.tolist(), d.cat_ids.tolist()):
            if c in keep_cats:
                preds.append({"image_id": img_id, "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    peak_mem_mb = torch.cuda.max_memory_allocated() / 2**20

    out_dir = Path(args.out)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "preds" / f"{tag}.json"
    pred_file.write_text(json.dumps(preds))

    E = COCOeval(coco_gt, coco_gt.loadRes(str(pred_file)), "bbox")
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    E.summarize()

    t = np.array(times)
    result = {
        "model": {**det.config(), "class_names": dict(names)},
        "params_M": det.num_params() / 1e6,
        "dataset": {"name": f"kitti-finetune-{args.gt}", "ann_file": GT[args.gt], "img_dir": IMG_DIR, "num_images": len(img_ids)},
        "metrics": dict(zip(STAT_NAMES, map(float, E.stats))),
        "per_class_AP": per_class_ap(E, coco_gt),
        "speed": {"timed_images": len(times), "warmup_images": args.warmup, "mean_ms": float(t.mean()),
                  "median_ms": float(np.median(t)), "fps": float(1000 / t.mean()),
                  "note": "batch 1, end-to-end predict() incl. pre/postprocess, cuda-synchronized"},
        "gpu_peak_mem_MB": peak_mem_mb,
        "env": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "python": platform.python_version()},
        "command": "python finetune/evaluate.py " + " ".join(
            f"--{k}" if v is True else f"--{k} {v}" for k, v in vars(args).items() if v is not False),
        "pred_file": str(pred_file),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))
    print(f"[{tag}] AP={E.stats[0]:.3f} AP50={E.stats[1]:.3f} {t.mean():.1f} ms {peak_mem_mb:.0f} MB")


if __name__ == "__main__":
    main()
