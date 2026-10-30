"""Evaluate a detector on COCO val2017 or the KITTI subset with pycocotools; also measures latency and GPU memory.

python eval.py --model yolo26l --dataset coco [--limit 20] [--imgsz 1280]
"""
import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectors import MODELS, load_detector

DATASETS = {
    "coco": ("data/coco/annotations/instances_val2017.json", "data/coco/val2017"),
    "kitti": ("data/kitti/kitti_coco.json", "data/kitti/training/image_2"),
}
STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def per_class_ap(E, coco_gt):
    # precision: [iou thr, recall, class, area, maxDets]; area=all, maxDets=100
    p = E.eval["precision"][:, :, :, 0, -1]
    out = {}
    for k, cat_id in enumerate(E.params.catIds):
        v = p[:, :, k]
        v = v[v > -1]
        out[coco_gt.cats[cat_id]["name"]] = float(v.mean()) if v.size else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    ap.add_argument("--imgsz", type=int, default=None, help="Ultralytics input size override (default 640)")
    ap.add_argument("--warmup", type=int, default=20, help="images excluded from timing")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ann_file, img_dir = DATASETS[args.dataset]
    coco_gt = COCO(ann_file)
    img_ids = sorted(coco_gt.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    keep_cats = set(coco_gt.getCatIds())  # KITTI GT only has person/car -> drop all other predicted classes

    kwargs = {"imgsz": args.imgsz} if args.imgsz else {}
    det = load_detector(args.model, **kwargs)
    tag = f"{det.name}{'_' + str(args.imgsz) if args.imgsz else ''}_{args.dataset}{'_n' + str(args.limit) if args.limit else ''}"

    torch.cuda.reset_peak_memory_stats()
    preds, times = [], []
    for i, img_id in enumerate(img_ids):
        info = coco_gt.imgs[img_id]
        img = cv2.imread(str(Path(img_dir) / info["file_name"]))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d = det.predict(img)  # end to end: preprocess + forward + postprocess (+NMS if any)
        torch.cuda.synchronize()
        if i >= args.warmup:
            times.append((time.perf_counter() - t0) * 1000)
        for (x1, y1, x2, y2), s, c in zip(d.boxes.tolist(), d.scores.tolist(), d.cat_ids.tolist()):
            if c in keep_cats:
                preds.append({"image_id": img_id, "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(img_ids)}")
    peak_mem_mb = torch.cuda.max_memory_allocated() / 2**20

    out_dir = Path(args.out)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "preds" / f"{tag}.json"
    pred_file.write_text(json.dumps(preds))

    stats, per_class = [0.0] * 12, {}
    if preds:
        E = COCOeval(coco_gt, coco_gt.loadRes(str(pred_file)), "bbox")
        E.params.imgIds = img_ids
        E.evaluate()
        E.accumulate()
        E.summarize()
        stats, per_class = [float(s) for s in E.stats], per_class_ap(E, coco_gt)

    t = np.array(times) if times else np.array([np.nan])
    result = {
        "model": det.config(),
        "params_M": det.num_params() / 1e6,
        "dataset": {"name": args.dataset, "ann_file": ann_file, "img_dir": img_dir, "num_images": len(img_ids)},
        "metrics": dict(zip(STAT_NAMES, stats)),
        "per_class_AP": per_class,
        "speed": {"timed_images": len(times), "warmup_images": args.warmup, "mean_ms": float(t.mean()),
                  "median_ms": float(np.median(t)), "fps": float(1000 / t.mean()),
                  "note": "batch 1, end-to-end predict() incl. pre/postprocess, cuda-synchronized"},
        "gpu_peak_mem_MB": peak_mem_mb,
        "env": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "python": platform.python_version()},
        "command": f"python eval.py --model {args.model} --dataset {args.dataset}"
                   + (f" --limit {args.limit}" if args.limit else "") + (f" --imgsz {args.imgsz}" if args.imgsz else ""),
        "pred_file": str(pred_file),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))
    print(f"[{tag}] AP={stats[0]:.3f} AP50={stats[1]:.3f} {result['speed']['mean_ms']:.1f} ms "
          f"{peak_mem_mb:.0f} MB -> {out_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
