"""Convert nuImages keyframes (2D boxes) to COCO-format JSON.

nuImages -> COCO mapping:
  human.pedestrian.{adult,child,construction_worker,police_officer} -> person (1)
  vehicle.car -> car (3); vehicle.truck -> truck (8); vehicle.bus.{rigid,bendy} -> bus (6)
  vehicle.motorcycle / vehicle.bicycle -> motorcycle (4) / bicycle (2) if without rider; with rider -> ignore
      (box covers rider + cycle, like KITTI Cyclist; COCO boxes them separately)
  vehicle.construction, vehicle.trailer, vehicle.emergency.*, human.pedestrian.{personal_mobility,stroller,wheelchair}
      -> ignore region (iscrowd=1): COCO models plausibly fire car/truck/person on them
  barriers, cones, debris, racks, animals -> dropped

python convert_nuimages.py [--version v1.0-mini] [--viz 3]
"""
import argparse
import json
from pathlib import Path

import cv2

ROOT = Path("data/nuimages")
TARGET = {"human.pedestrian.adult": 1, "human.pedestrian.child": 1, "human.pedestrian.construction_worker": 1,
          "human.pedestrian.police_officer": 1, "vehicle.car": 3, "vehicle.truck": 8, "vehicle.bus.rigid": 6,
          "vehicle.bus.bendy": 6, "vehicle.motorcycle": 4, "vehicle.bicycle": 2}
IGNORE = {"vehicle.construction": 8, "vehicle.trailer": 8, "vehicle.emergency.ambulance": 8, "vehicle.emergency.police": 3,
          "human.pedestrian.personal_mobility": 1, "human.pedestrian.stroller": 1, "human.pedestrian.wheelchair": 1}
CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}, {"id": 3, "name": "car"},
              {"id": 4, "name": "motorcycle"}, {"id": 6, "name": "bus"}, {"id": 8, "name": "truck"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--viz", type=int, default=3)
    ap.add_argument("--out", default="data/nuimages/nuimages_coco.json")
    args = ap.parse_args()

    meta = lambda n: json.loads((ROOT / args.version / f"{n}.json").read_text())
    cat = {c["token"]: c["name"] for c in meta("category")}
    att = {a["token"]: a["name"] for a in meta("attribute")}
    sensor = {s["token"]: s["channel"] for s in meta("sensor")}
    calib = {c["token"]: sensor[c["sensor_token"]] for c in meta("calibrated_sensor")}
    keys = sorted((s for s in meta("sample_data") if s["is_key_frame"]), key=lambda s: s["filename"])
    tok2id = {s["token"]: i for i, s in enumerate(keys)}

    images = [{"id": i, "file_name": s["filename"], "width": s["width"], "height": s["height"],
               "camera": calib[s["calibrated_sensor_token"]]} for i, s in enumerate(keys)]
    anns, counts = [], {}
    for a in meta("object_ann"):
        if a["sample_data_token"] not in tok2id:
            continue
        name, attrs = cat[a["category_token"]], [att[t] for t in a["attribute_tokens"]]
        if name not in TARGET and name not in IGNORE:
            continue
        ignore = name in IGNORE or "cycle.with_rider" in attrs
        x1, y1, x2, y2 = a["bbox"]
        counts[(name, ignore)] = counts.get((name, ignore), 0) + 1
        anns.append({"id": len(anns) + 1, "image_id": tok2id[a["sample_data_token"]],
                     "category_id": TARGET.get(name, IGNORE.get(name)), "bbox": [x1, y1, x2 - x1, y2 - y1],
                     "area": (x2 - x1) * (y2 - y1), "iscrowd": int(ignore), "nu_class": name, "attributes": attrs})

    Path(args.out).write_text(json.dumps({"images": images, "annotations": anns, "categories": CATEGORIES}))
    print(f"{len(images)} images, {len(anns)} boxes -> {args.out}")
    for (n, ig), v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {v:4d} {n}{' (ignore)' if ig else ''}")

    fig = Path("results/figs"); fig.mkdir(parents=True, exist_ok=True)
    for im in images[: args.viz]:
        img = cv2.imread(str(ROOT / im["file_name"]))
        for a in (a for a in anns if a["image_id"] == im["id"]):
            x, y, w, h = map(int, a["bbox"])
            col = (128, 128, 128) if a["iscrowd"] else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            cv2.putText(img, a["nu_class"].split(".")[-1], (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.imwrite(str(fig / f"nuimages_gt_{im['id']}.png"), img)


if __name__ == "__main__":
    main()
