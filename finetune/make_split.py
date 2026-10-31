"""Fine-tuning study: drive-disjoint KITTI train / val / test split for fine-tuning YOLO26-L.

KITTI object images are frames sampled from 141 raw recording drives; neighbouring frames are near-duplicates.
A random image split (as in the baseline and extended studies) would leak them between train and test, so whole drives are assigned
to one split. The drive of each image comes from the official devkit mapping (devkit_object.zip).

Outputs
  data/kitti_finetune/split.json                       image stems + drives per split
  data/kitti_finetune/{val,test}_coco.json             COCO-format GT, baseline-study taxonomy (car, person; rest = ignore)
  data/kitti_finetune/{val,test}_coco_cyclist.json     same + Cyclist scored as id 2 (extended-study cyclist variant)
  data/kitti_finetune/yolo/{images,labels}/{train,val} YOLO dataset: 0 person, 1 cyclist, 2 car; DontCare grayed out
  data/kitti_finetune/kitti.yaml                       Ultralytics dataset file

python finetune/make_split.py [--test-frac 0.2] [--val-frac 0.1] [--seed 0]
"""
import argparse
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import cv2

KITTI = Path("data/kitti/training")
OUT = Path("data/kitti_finetune")
DEVKIT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip"
MAPPING = Path("data/kitti/mapping")

# YOLO training classes and their COCO ids (person=1, bicycle/cyclist=2, car=3), so predictions score against the same GT
YOLO_CLASSES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}
YOLO_NAMES = ["person", "cyclist", "car"]
# GT taxonomy identical to convert_kitti.py and its --cyclist variant
TARGET = {"Car": 3, "Pedestrian": 1}
IGNORE = {"Van": 3, "Truck": 3, "Misc": 3, "DontCare": 3, "Cyclist": 1, "Person_sitting": 1}
CATEGORIES = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
GRAY = (114, 114, 114)  # Ultralytics letterbox fill


def drive_of_each_image():
    if not (MAPPING / "train_mapping.txt").exists():
        zip_path = Path("data/kitti/devkit_object.zip")
        if not zip_path.exists():
            urllib.request.urlretrieve(DEVKIT_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for name in ("mapping/train_mapping.txt", "mapping/train_rand.txt"):
                (MAPPING / Path(name).name).parent.mkdir(parents=True, exist_ok=True)
                (MAPPING / Path(name).name).write_bytes(z.read(name))
    mapping = [line.split() for line in (MAPPING / "train_mapping.txt").read_text().splitlines()]
    rand = [int(x) for x in (MAPPING / "train_rand.txt").read_text().strip().split(",")]
    return {f"{i:06d}": mapping[rand[i] - 1][1] for i in range(len(rand))}


def split_drives(drive, test_frac, val_frac, seed):
    by_drive = {}
    for stem, d in drive.items():
        by_drive.setdefault(d, []).append(stem)
    drives = sorted(by_drive)
    random.Random(seed).shuffle(drives)
    n, splits, filled = len(drive), {"test": [], "val": [], "train": []}, 0
    for d in drives:  # fill test, then val, up to their image fraction; the rest is train
        if filled < test_frac * n:
            key = "test"
        elif filled < (test_frac + val_frac) * n:
            key = "val"
        else:
            key = "train"
        splits[key].append(d)
        filled += len(by_drive[d])
    return {k: {"drives": sorted(v), "images": sorted(s for d in v for s in by_drive[d])} for k, v in splits.items()}


def read_labels(stem):
    for line in (KITTI / "label_2" / f"{stem}.txt").read_text().splitlines():
        f = line.split()
        yield f[0], tuple(map(float, f[4:8])), float(f[1]), int(f[2])


def write_coco(stems, out, cyclist):
    target, ignore, categories = dict(TARGET), dict(IGNORE), list(CATEGORIES)
    if cyclist:
        target["Cyclist"] = 2
        ignore.pop("Cyclist")
        categories.insert(1, {"id": 2, "name": "cyclist"})
    images, anns = [], []
    for img_id, stem in enumerate(stems):
        h, w = cv2.imread(str(KITTI / "image_2" / f"{stem}.png")).shape[:2]
        images.append({"id": img_id, "file_name": f"{stem}.png", "width": w, "height": h})
        for cls, (x1, y1, x2, y2), trunc, occ in read_labels(stem):
            if cls not in target and cls not in ignore:
                continue
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": target.get(cls, ignore.get(cls)),
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                         "iscrowd": int(cls not in target), "kitti_class": cls, "truncated": trunc, "occluded": occ})
    out.write_text(json.dumps({"images": images, "annotations": anns, "categories": categories}))
    return len(images), sum(not a["iscrowd"] for a in anns)


def write_yolo(stems, split):
    img_dir, lbl_dir = OUT / "yolo" / "images" / split, OUT / "yolo" / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for stem in stems:
        img = cv2.imread(str(KITTI / "image_2" / f"{stem}.png"))
        h, w = img.shape[:2]
        lines = []
        for cls, (x1, y1, x2, y2), _, _ in read_labels(stem):
            if cls == "DontCare":  # unlabeled region: gray it out so the model isn't taught it is background
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GRAY, -1)
            elif cls in YOLO_CLASSES:
                lines.append(f"{YOLO_CLASSES[cls]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        cv2.imwrite(str(img_dir / f"{stem}.png"), img)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_boxes += len(lines)
    return n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    drive = drive_of_each_image()
    splits = split_drives(drive, args.test_frac, args.val_frac, args.seed)
    assert not set(splits["train"]["drives"]) & (set(splits["val"]["drives"]) | set(splits["test"]["drives"]))
    (OUT / "split.json").write_text(json.dumps({"seed": args.seed, **splits}, indent=1))

    for split in ("train", "val"):
        n = write_yolo(splits[split]["images"], split)
        print(f"{split}: {len(splits[split]['drives'])} drives, {len(splits[split]['images'])} images, {n} YOLO boxes")
    for split in ("val", "test"):
        for cyclist in (False, True):
            out = OUT / f"{split}_coco{'_cyclist' if cyclist else ''}.json"
            n_img, n_box = write_coco(splits[split]["images"], out, cyclist)
            print(f"{split}{' (cyclist)' if cyclist else ''}: {len(splits[split]['drives'])} drives, "
                  f"{n_img} images, {n_box} scored boxes -> {out}")

    (OUT / "kitti.yaml").write_text(
        f"path: {(OUT / 'yolo').resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(YOLO_NAMES)))


if __name__ == "__main__":
    main()
