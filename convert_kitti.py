"""Convert a seeded 1,000-image KITTI training subset to COCO-format JSON (car + person).

KITTI -> COCO mapping:
  Car        -> car (3)
  Pedestrian -> person (1)
  Van, Truck, Cyclist, Person_sitting, Misc, DontCare -> ignore region (iscrowd=1): a COCO model may
      fire "car"/"person" on these, which should be neither a false positive nor a missed GT
  Tram       -> dropped (no plausible car/person confusion)

python convert_kitti.py [--n 1000] [--seed 0] [--viz 5]
python convert_kitti.py --cyclist --viz 0 --out data/kitti/kitti_coco_cyclist.json   # also score Cyclist (id 2)
"""
import argparse
import json
import random
from pathlib import Path

import cv2

ROOT = Path("data/kitti/training")
TARGET = {"Car": 3, "Pedestrian": 1}
IGNORE = {"Van": 3, "Truck": 3, "Misc": 3, "DontCare": 3, "Cyclist": 1, "Person_sitting": 1}
CATEGORIES = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viz", type=int, default=5, help="draw GT on this many images -> results/figs/kitti_gt_*.png")
    ap.add_argument("--out", default="data/kitti/kitti_coco.json")
    ap.add_argument("--cyclist", action="store_true", help="score Cyclist as class 2 instead of ignoring it")
    args = ap.parse_args()
    target, ignore, categories = dict(TARGET), dict(IGNORE), list(CATEGORIES)
    if args.cyclist:
        target["Cyclist"] = 2
        ignore.pop("Cyclist")
        categories.insert(1, {"id": 2, "name": "cyclist"})

    all_ids = sorted(p.stem for p in (ROOT / "label_2").glob("*.txt"))
    ids = sorted(random.Random(args.seed).sample(all_ids, args.n))
    Path("data/kitti_list.txt").write_text("\n".join(ids) + "\n")

    images, anns, counts = [], [], {}
    for img_id, stem in enumerate(ids):
        img = cv2.imread(str(ROOT / "image_2" / f"{stem}.png"))
        h, w = img.shape[:2]
        images.append({"id": img_id, "file_name": f"{stem}.png", "width": w, "height": h})
        for line in (ROOT / "label_2" / f"{stem}.txt").read_text().splitlines():
            f = line.split()
            cls = f[0]
            if cls not in target and cls not in ignore:
                continue
            x1, y1, x2, y2 = map(float, f[4:8])
            counts[cls] = counts.get(cls, 0) + 1
            anns.append({
                "id": len(anns) + 1, "image_id": img_id,
                "category_id": target.get(cls, ignore.get(cls)),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                "iscrowd": int(cls not in target),
                "kitti_class": cls, "truncated": float(f[1]), "occluded": int(f[2]),
            })

    Path(args.out).write_text(json.dumps({"images": images, "annotations": anns, "categories": categories}))
    print(f"{len(images)} images, {len(anns)} boxes -> {args.out}")
    print("per KITTI class:", dict(sorted(counts.items())))

    fig_dir = Path("results/figs")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for im in images[: args.viz]:
        img = cv2.imread(str(ROOT / "image_2" / im["file_name"]))
        for a in (a for a in anns if a["image_id"] == im["id"]):
            x, y, bw, bh = map(int, a["bbox"])
            color = (128, 128, 128) if a["iscrowd"] else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(img, a["kitti_class"], (x, max(y - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(str(fig_dir / f"kitti_gt_{Path(im['file_name']).stem}.png"), img)


if __name__ == "__main__":
    main()
