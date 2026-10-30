"""Tables, plots and failure images for the report, from results/*.json and results/preds/*.json.

python make_report_assets.py
Outputs: results/tables.md, results/figs/{size_drop,speed_vs_ap}.png, results/figs/fail_*.png, results/failures.md
"""
import contextlib
import io
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

R = Path("results")
FIG = R / "figs"
COCO_ANN = "data/coco/annotations/instances_val2017.json"
KITTI_ANN, KITTI_IMG = "data/kitti/kitti_coco.json", "data/kitti/training/image_2"
# (display name, result tag prefix); fixed categorical order -> fixed colors
MODELS = [("YOLO26-L", "yolo26l"), ("YOLO26-L +NMS", "yolo26l-nms"),
          ("RT-DETR-L", "rtdetr-l"), ("Faster R-CNN v2", "frcnn-r50-fpn-v2")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
CAR_PERSON = [1, 3]


def load(tag):
    p = R / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def coco_eval(gt, pred_file, cat_ids=None):
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(pred_file), "bbox")
        if cat_ids:
            E.params.catIds = cat_ids
        E.evaluate(); E.accumulate(); E.summarize()
    return E.stats


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    coco_gt = COCO(COCO_ANN)
    rows = []
    for name, tag in MODELS:
        c, k = load(f"{tag}_coco"), load(f"{tag}_kitti")
        s2 = coco_eval(coco_gt, c["pred_file"], CAR_PERSON)  # COCO restricted to KITTI's classes
        rows.append(dict(name=name, tag=tag, coco=c, kitti=k, coco2=dict(AP=s2[0], AP50=s2[1], S=s2[3], M=s2[4], L=s2[5])))

    # ---- tables ----
    f = lambda v: f"{100 * v:.1f}"
    t = ["## Table 1: COCO val2017 vs KITTI (1,000 imgs), AP x100\n",
         "COCO-2 = COCO val scored on car+person only (same classes as KITTI).\n",
         "| Model | COCO-80 AP | COCO-2 AP | KITTI AP | Δ AP | COCO-2 S/M/L | KITTI S/M/L | car COCO→KITTI | person COCO→KITTI |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c2, km, pc, pk = r["coco2"], r["kitti"]["metrics"], r["coco"]["per_class_AP"], r["kitti"]["per_class_AP"]
        t.append(f"| {r['name']} | {f(r['coco']['metrics']['AP'])} | {f(c2['AP'])} | {f(km['AP'])} | {f(km['AP'] - c2['AP'])} "
                 f"| {f(c2['S'])}/{f(c2['M'])}/{f(c2['L'])} | {f(km['AP_small'])}/{f(km['AP_medium'])}/{f(km['AP_large'])} "
                 f"| {f(pc['car'])}→{f(pk['car'])} | {f(pc['person'])}→{f(pk['person'])} |")
    t += ["\n## Table 2: Speed, memory, size (batch 1, fp32, end-to-end predict())\n",
          "| Model | Params (M) | Input (COCO) | COCO ms | COCO FPS | KITTI ms | Peak GPU mem COCO / KITTI (MB) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        c, k = r["coco"], r["kitti"]
        t.append(f"| {r['name']} | {c['params_M']:.1f} | {c['model']['resize']} | {c['speed']['mean_ms']:.1f} | {c['speed']['fps']:.0f} "
                 f"| {k['speed']['mean_ms']:.1f} | {c['gpu_peak_mem_MB']:.0f} / {k['gpu_peak_mem_MB']:.0f} |")
    t += ["\n## Table 3: KITTI input-resolution test (KITTI 1242x375)\n",
          "| Model | imgsz | Effective input | KITTI AP | AP_S | AP_M | AP_L | car | person | ms |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, tag, sz, eff in [("YOLO26-L", "yolo26l_kitti", 640, "640x193"), ("YOLO26-L", "yolo26l_1280_kitti", 1280, "1280x386"),
                               ("RT-DETR-L", "rtdetr-l_kitti", 640, "640x193"), ("RT-DETR-L", "rtdetr-l_1280_kitti", 1280, "1280x386"),
                               ("Faster R-CNN v2", "frcnn-r50-fpn-v2_kitti", "800/1333", "1333x403")]:
        k = load(tag)
        if not k:
            continue
        m, pc = k["metrics"], k["per_class_AP"]
        t.append(f"| {name} | {sz} | {eff} | {f(m['AP'])} | {f(m['AP_small'])} | {f(m['AP_medium'])} | {f(m['AP_large'])} "
                 f"| {f(pc['car'])} | {f(pc['person'])} | {k['speed']['mean_ms']:.1f} |")
    (R / "tables.md").write_text("\n".join(t) + "\n")

    # ---- plot 1: AP drop by object size ----
    sizes = [("small", "S", "AP_small"), ("medium", "M", "AP_medium"), ("large", "L", "AP_large")]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    w = 0.8 / len(rows)
    for i, r in enumerate(rows):
        drops = [100 * (r["kitti"]["metrics"][kk] - r["coco2"][ck]) for _, ck, kk in sizes]
        xs = np.arange(3) + (i - (len(rows) - 1) / 2) * w
        ax.bar(xs, drops, w * 0.9, color=COLORS[i], label=r["name"], edgecolor="white", linewidth=1)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(3), ["Small (<32²)", "Medium", "Large (>96²)"], color=INK)
    ax.set_ylabel("Δ AP, KITTI − COCO (car+person)", color=INK)
    ax.set_title("AP change COCO → KITTI by object size", color=INK, loc="left", fontsize=11)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(FIG / "size_drop.png")
    plt.close(fig)

    # ---- plot 2: speed vs accuracy, one panel per dataset (no dual axis) ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=200, sharex=True)
    for ax, ds, title in [(axes[0], "coco", "COCO val2017 (80 classes)"), (axes[1], "kitti", "KITTI (car + person)")]:
        for i, r in enumerate(rows):
            d = r[ds]
            x, y = d["speed"]["mean_ms"], 100 * d["metrics"]["AP"]
            ax.scatter(x, y, s=60, color=COLORS[i], edgecolor="white", linewidth=1.5, zorder=3, label=r["name"])
            hi = load(f"{r['tag']}_1280_kitti") if ds == "kitti" else None
            if hi:  # 640 -> 1280 arrow
                x2, y2 = hi["speed"]["mean_ms"], 100 * hi["metrics"]["AP"]
                ax.annotate("", (x2, y2), (x, y), arrowprops=dict(arrowstyle="->", color=COLORS[i], lw=1.2))
                ax.scatter(x2, y2, s=60, facecolor="white", edgecolor=COLORS[i], linewidth=1.5, zorder=3)
                ax.annotate("1280", (x2, y2), xytext=(4, -10), textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.set_xlabel("Latency per image (ms, lower is better)", color=INK)
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
    axes[0].set_ylabel("mAP@[.5:.95]", color=INK)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "speed_vs_ap.png")
    plt.close(fig)

    # ---- failure images: KITTI images where YOLO26-L misses the most GT (score>=0.3, IoU>=0.5) ----
    kgt = json.loads(Path(KITTI_ANN).read_text())
    preds = json.loads(Path(load("yolo26l_kitti")["pred_file"]).read_text())
    by_img = {}
    for p in preds:
        if p["score"] >= 0.3:
            by_img.setdefault(p["image_id"], []).append(p)
    anns = {}
    for a in kgt["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    def iou(a, b, crowd=False):
        ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
        iw = max(0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1)); ih = max(0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
        inter = iw * ih
        # like pycocotools: vs an ignore region the union is the detection's own area (IoA)
        return inter / ((aw * ah if crowd else aw * ah + bw * bh - inter) + 1e-9)

    stats = []
    for im in kgt["images"]:
        gts = [a for a in anns.get(im["id"], []) if not a["iscrowd"]]
        ps = by_img.get(im["id"], [])
        missed = [g for g in gts if not any(p["category_id"] == g["category_id"] and iou(p["bbox"], g["bbox"]) >= 0.5 for p in ps)]
        crowd = [a for a in anns.get(im["id"], []) if a["iscrowd"]]
        fps = [p for p in ps if not any(p["category_id"] == g["category_id"] and iou(p["bbox"], g["bbox"]) >= 0.5 for g in gts)
               and not any(iou(p["bbox"], c["bbox"], crowd=True) >= 0.5 for c in crowd)]
        stats.append((len(missed) + len(fps), im, gts, missed, fps, ps))
    stats.sort(key=lambda s: -s[0])

    lines = ["# Failure images (YOLO26-L, KITTI, score >= 0.3, IoU >= 0.5)\n",
             "green = GT, red = prediction, yellow = missed GT, gray = ignore region\n"]
    for k, (_, im, gts, missed, fps, ps) in enumerate(stats[:6]):
        img = cv2.imread(str(Path(KITTI_IMG) / im["file_name"]))
        brightness = img.mean()
        for a in anns.get(im["id"], []):
            x, y, w_, h_ = map(int, a["bbox"])
            col = (150, 150, 150) if a["iscrowd"] else (0, 200, 255) if a in missed else (0, 200, 0)
            cv2.rectangle(img, (x, y), (x + w_, y + h_), col, 2)
        for p in ps:
            x, y, w_, h_ = map(int, p["bbox"])
            cv2.rectangle(img, (x, y), (x + w_, y + h_), (0, 0, 230), 1)
            cv2.putText(img, f"{'car' if p['category_id'] == 3 else 'person'} {p['score']:.2f}", (x, max(y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 230), 1)
        out = FIG / f"fail_{k + 1}_{Path(im['file_name']).stem}.png"
        cv2.imwrite(str(out), img)
        desc = ", ".join(f"{a['kitti_class']} h={a['bbox'][3]:.0f}px occ={a['occluded']} trunc={a['truncated']:.2f}" for a in missed)
        lines.append(f"- `{out.name}`: {len(gts)} GT, {len(missed)} missed, {len(fps)} FP, brightness {brightness:.0f}/255. "
                     f"Missed: {desc or '-'}")
    (R / "failures.md").write_text("\n".join(lines) + "\n")
    print((R / "tables.md").read_text())
    print((R / "failures.md").read_text())


if __name__ == "__main__":
    main()
