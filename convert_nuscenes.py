"""Convert nuScenes camera keyframes to COCO-format JSON by projecting 3D boxes to 2D.

nuScenes has no native 2D boxes. Like the devkit's export_2d_annotations_as_json.py, each 3D box is moved
global -> ego -> camera, the 8 corners in front of the camera are projected with the intrinsics, and the 2D box
is the bounding box of (convex hull of projected corners) ∩ image. These boxes are looser than hand-drawn 2D
boxes (they cover the full 3D extent, including occluded parts).

Class mapping is the same as convert_nuimages.py. Additionally, boxes with visibility level 1 (0-40% visible)
are ignore regions: many are fully hidden behind other objects, so no detector can see them.

python convert_nuscenes.py [--version v1.0-mini] [--viz 3]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import MultiPoint, box

from convert_nuimages import CATEGORIES, IGNORE, TARGET

ROOT = Path("data/nuscenes")


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def box_corners(center, size, rot):
    w, l, h = size  # nuScenes size order: width, length, height
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    return quat_to_rot(rot) @ np.vstack([x, y, z]) + np.array(center)[:, None]  # (3, 8) global frame


def project_2d(corners_global, ego, cs, width, height):
    c = quat_to_rot(ego["rotation"]).T @ (corners_global - np.array(ego["translation"])[:, None])  # global -> ego
    c = quat_to_rot(cs["rotation"]).T @ (c - np.array(cs["translation"])[:, None])  # ego -> camera
    c = c[:, c[2] > 0]  # corners in front of the camera
    if c.shape[1] == 0:
        return None
    p = np.array(cs["camera_intrinsic"]) @ c
    p = (p[:2] / p[2]).T
    hull = MultiPoint([tuple(pt) for pt in p]).convex_hull.intersection(box(0, 0, width, height))
    if hull.is_empty or hull.area == 0:
        return None
    x1, y1, x2, y2 = hull.bounds
    return [x1, y1, x2, y2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--viz", type=int, default=3)
    ap.add_argument("--out", default="data/nuscenes/nuscenes_coco.json")
    args = ap.parse_args()

    meta = lambda n: json.loads((ROOT / args.version / f"{n}.json").read_text())
    cat = {c["token"]: c["name"] for c in meta("category")}
    att = {a["token"]: a["name"] for a in meta("attribute")}
    vis = {v["token"]: v["level"] for v in meta("visibility")}
    inst_cat = {i["token"]: cat[i["category_token"]] for i in meta("instance")}
    ego = {e["token"]: e for e in meta("ego_pose")}
    cs = {c["token"]: c for c in meta("calibrated_sensor")}
    by_sample = {}
    for a in meta("sample_annotation"):
        by_sample.setdefault(a["sample_token"], []).append(a)
    cams = sorted((s for s in meta("sample_data") if s["is_key_frame"] and s["filename"].startswith("samples/CAM")),
                  key=lambda s: s["filename"])

    images, anns, counts = [], [], {}
    for img_id, sd in enumerate(cams):
        images.append({"id": img_id, "file_name": sd["filename"], "width": sd["width"], "height": sd["height"],
                       "camera": sd["filename"].split("/")[1]})
        for a in by_sample.get(sd["sample_token"], []):
            name = inst_cat[a["instance_token"]]
            if name not in TARGET and name not in IGNORE:
                continue
            bb = project_2d(box_corners(a["translation"], a["size"], a["rotation"]), ego[sd["ego_pose_token"]],
                            cs[sd["calibrated_sensor_token"]], sd["width"], sd["height"])
            if bb is None:
                continue
            attrs, v = [att[t] for t in a["attribute_tokens"]], vis[a["visibility_token"]]
            ignore = name in IGNORE or "cycle.with_rider" in attrs or v == "v0-40"
            x1, y1, x2, y2 = bb
            counts[(name, ignore)] = counts.get((name, ignore), 0) + 1
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": TARGET.get(name, IGNORE.get(name)),
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": int(ignore),
                         "nu_class": name, "attributes": attrs, "visibility": v,
                         "num_lidar_pts": a["num_lidar_pts"]})

    Path(args.out).write_text(json.dumps({"images": images, "annotations": anns, "categories": CATEGORIES}))
    print(f"{len(images)} images, {len(anns)} boxes -> {args.out}")
    for (n, ig), c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {c:5d} {n}{' (ignore)' if ig else ''}")

    fig = Path("results/figs"); fig.mkdir(parents=True, exist_ok=True)
    for im in [i for i in images if i["camera"] == "CAM_FRONT"][: args.viz]:
        img = cv2.imread(str(ROOT / im["file_name"]))
        for a in (a for a in anns if a["image_id"] == im["id"]):
            x, y, w, h = map(int, a["bbox"])
            col = (128, 128, 128) if a["iscrowd"] else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            cv2.putText(img, a["nu_class"].split(".")[-1], (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.imwrite(str(fig / f"nuscenes_gt_{im['id']}.png"), img)


if __name__ == "__main__":
    main()
