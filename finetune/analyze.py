"""Fine-tuning study: tables and figures from results/finetune/*.json and their predictions.

python finetune/analyze.py
Outputs: results/finetune/tables.md, results/figs/finetune_{main,occlusion,curves,examples}.png
"""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

R = Path("results/finetune")
FIGS = Path("results/figs")
TRAIN = R / "train"
RUNS = [  # (tag, label)
    ("zs640", "Zero-shot @640"),
    ("ft640", "Fine-tuned @640"),
]
OCC = {0: "fully visible", 1: "partly occluded", 2: "largely occluded"}


def load(tag, gt="test"):
    p = R / f"{tag}_{gt}.json"
    return json.loads(p.read_text()) if p.exists() else None


def coco_ap(gt_dict, pred_file, cat_ids=None):
    gt = COCO()
    gt.dataset = gt_dict
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        E = COCOeval(gt, gt.loadRes(str(pred_file)), "bbox")
        if cat_ids:
            E.params.catIds = cat_ids
        E.evaluate()
        E.accumulate()
        E.summarize()
    return E.stats


def only_occlusion(gt_dict, level):
    """Score only boxes of one KITTI occlusion level; the other scored boxes become ignore regions."""
    g = copy.deepcopy(gt_dict)
    for a in g["annotations"]:
        if not a["iscrowd"] and a["occluded"] != level:
            a["iscrowd"] = 1
    return g


def table(header, rows):
    out = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return out + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def main():
    runs = [(t, n, load(t), load(t, "test-cyclist")) for t, n in RUNS if load(t)]
    f = lambda v: f"{100 * v:.1f}"
    md = []

    md.append("### Main metrics (car + person)\n")
    md.append(table(["Run", "mAP", "AP50", "AP75", "AP S / M / L", "AR100", "car", "person"], [
        [n, f(r["metrics"]["AP"]), f(r["metrics"]["AP50"]), f(r["metrics"]["AP75"]),
         " / ".join(f(r["metrics"][k]) for k in ("AP_small", "AP_medium", "AP_large")), f(r["metrics"]["AR100"]),
         f(r["per_class_AP"]["car"]), f(r["per_class_AP"]["person"])] for t, n, r, _ in runs]))

    md.append("\n### Cyclist variant (car + person + cyclist)\n")
    md.append(table(["Run", "mAP", "car", "person", "cyclist"], [
        [n, f(c["metrics"]["AP"]), f(c["per_class_AP"]["car"]), f(c["per_class_AP"]["person"]),
         f(c["per_class_AP"]["cyclist"])] for t, n, _, c in runs if c]))

    gt = json.loads(Path("data/kitti_finetune/test_coco.json").read_text())
    occ = {t: [coco_ap(only_occlusion(gt, k), r["pred_file"])[0] for k in OCC] for t, n, r, _ in runs}
    counts = [sum(1 for a in gt["annotations"] if not a["iscrowd"] and a["occluded"] == k) for k in OCC]
    md.append("\n### By KITTI occlusion level (car + person mAP)\n")
    md.append(table(["Run"] + [f"{v} ({c:,} boxes)" for v, c in zip(OCC.values(), counts)],
                    [[n] + [f(v) for v in occ[t]] for t, n, _, _ in runs]))

    md.append("\n### Speed and memory\n")
    md.append(table(["Run", "Input", "Latency (ms)", "FPS", "Peak memory (MB)"], [
        [n, r["model"]["imgsz"], f"{r['speed']['mean_ms']:.1f}", f"{r['speed']['fps']:.0f}", f"{r['gpu_peak_mem_MB']:.0f}"]
        for t, n, r, _ in runs]))

    md.append("\n### Training\n")
    rows = []
    for t, n, _, _ in runs:
        cfg = TRAIN / t / "train_info.json"
        if cfg.exists():
            c = json.loads(cfg.read_text())
            res = list(csv.DictReader(open(TRAIN / t / "results.csv")))
            best = max(res, key=lambda x: float(x["metrics/mAP50-95(B)"]))
            rows.append([n, round(5209 * c["fraction"]), c["epochs"], c["imgsz"], c["batch"], f"{c['train_hours']:.2f}",
                         best["epoch"].strip(), f(float(best["metrics/mAP50-95(B)"]))])
    md.append(table(["Run", "Train images", "Epochs", "imgsz", "Batch", "Hours", "Best epoch", "Val mAP (3 classes)"], rows))
    (R / "tables.md").write_text("\n".join(md))
    print("\n".join(md))

    # --- figures ---
    style = lambda ax: (ax.spines["top"].set_visible(False), ax.spines["right"].set_visible(False),
                        ax.grid(axis="y", color="#e0e0e0", lw=0.8), ax.set_axisbelow(True))
    main_runs = [x for x in runs if x[0] in ("zs640", "ft640")]
    metrics = [("mAP", lambda r, c: r["metrics"]["AP"]), ("car", lambda r, c: r["per_class_AP"]["car"]),
               ("person", lambda r, c: r["per_class_AP"]["person"]), ("cyclist", lambda r, c: c["per_class_AP"]["cyclist"])]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(main_runs)
    shades = {"zs640": "#9aa0a6", "ft640": "#1a73e8"}
    for i, (t, n, r, c) in enumerate(main_runs):
        vals = [100 * fn(r, c) for _, fn in metrics]
        bars = ax.bar(np.arange(len(metrics)) + (i - (len(main_runs) - 1) / 2) * w, vals, w, label=n, color=shades[t])
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    ax.set_xticks(range(len(metrics)), [m for m, _ in metrics])
    ax.set_ylabel("AP (×100)")
    ax.set_title("KITTI drive-disjoint test split: zero-shot vs. fine-tuned YOLO26-L", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.1))
    style(ax)
    plt.tight_layout()
    plt.savefig(FIGS / "finetune_main.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (t, n, _, _) in enumerate(main_runs):
        bars = ax.bar(np.arange(3) + (i - (len(main_runs) - 1) / 2) * w, [100 * v for v in occ[t]], w, label=n,
                      color=shades[t])
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    ax.set_xticks(range(3), list(OCC.values()))
    ax.set_ylabel("car + person AP (×100)")
    ax.set_title("AP by KITTI occlusion level", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    plt.tight_layout()
    plt.savefig(FIGS / "finetune_occlusion.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for t, n, _, _ in runs:
        p = TRAIN / t / "results.csv"
        if p.exists():
            res = list(csv.DictReader(open(p)))
            ax.plot([int(x["epoch"]) for x in res], [100 * float(x["metrics/mAP50-95(B)"]) for x in res], label=n)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val mAP (3 classes, ×100)")
    ax.set_title("Validation mAP during fine-tuning", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    plt.tight_layout()
    plt.savefig(FIGS / "finetune_curves.png", dpi=150)
    plt.close()

    # qualitative: test images where fine-tuning gains the most matched car/person/cyclist boxes (IoU >= 0.5, score >= 0.3)
    zs, ft = load("zs640", "test-cyclist"), load("ft640", "test-cyclist")
    if zs and ft:
        gtc = json.loads(Path("data/kitti_finetune/test_coco_cyclist.json").read_text())
        by_img = lambda preds: {i: [p for p in preds if p["image_id"] == i and p["score"] >= 0.3]
                                for i in {g["id"] for g in gtc["images"]}}
        pz, pf = by_img(json.loads(Path(zs["pred_file"]).read_text())), by_img(json.loads(Path(ft["pred_file"]).read_text()))
        gts = {}
        for a in gtc["annotations"]:
            if not a["iscrowd"]:
                gts.setdefault(a["image_id"], []).append(a)

        def iou(a, b):
            ax1, ay1, aw, ah = a
            bx1, by1, bw, bh = b
            iw = max(0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
            ih = max(0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
            return iw * ih / (aw * ah + bw * bh - iw * ih + 1e-9)

        def hits(i, preds):
            return [any(p["category_id"] == g["category_id"] and iou(p["bbox"], g["bbox"]) >= 0.5 for p in preds[i])
                    for g in gts.get(i, [])]

        gain = sorted(gts, key=lambda i: sum(hits(i, pf)) - sum(hits(i, pz)), reverse=True)[:3]
        files = {g["id"]: g["file_name"] for g in gtc["images"]}
        fig, axes = plt.subplots(len(gain), 2, figsize=(14, 2.6 * len(gain)))
        for row, i in enumerate(gain):
            for col, (name, preds) in enumerate((("zero-shot @640", pz), ("fine-tuned @640", pf))):
                img = cv2.imread(f"data/kitti/training/image_2/{files[i]}")
                h = hits(i, preds)
                for g, ok in zip(gts[i], h):
                    x, y, bw, bh = map(int, g["bbox"])
                    cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 200, 0) if ok else (0, 220, 255), 2)
                for p in preds[i]:
                    x, y, bw, bh = map(int, p["bbox"])
                    cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 1)
                ax = axes[row, col]
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.axis("off")
                ax.set_title(f"{name}: {sum(h)}/{len(h)} found", loc="left", fontsize=9)
        plt.suptitle("green = found, yellow = missed, red = prediction (score ≥ 0.3)", x=0.01, ha="left", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGS / "finetune_examples.png", dpi=110)
        plt.close()


if __name__ == "__main__":
    main()
