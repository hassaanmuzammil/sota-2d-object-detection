"""Builds notebooks/level2_extended.ipynb (5 detector families x 4 datasets). Run from repo root:

python notebooks/build_level2.py
cd notebooks && jupyter nbconvert --to notebook --execute --inplace level2_extended.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Extended Study — 5 Detector Families × 4 Datasets

Extends the MVP (`level1_baseline.ipynb`, unchanged) with two more detector families and two more driving datasets. Every number below is loaded from `results/*.json`, produced by `eval.py`.

| Family | Model | Checkpoint |
|---|---|---|
| One-stage CNN, anchor-free (YOLO) | YOLO26-L (NMS-free; also `+NMS` config) | `weights/yolo26l.pt` |
| Query-based transformer (DETR) | RT-DETR-L | `weights/rtdetr-l.pt` |
| Two-stage CNN | Faster R-CNN R50-FPN v2 | torchvision `COCO_V1` |
| Dense one-stage CNN, anchor-based | RetinaNet R50-FPN v2 | torchvision `COCO_V1` |
| Open-vocabulary | YOLO-World-L v2 | `weights/yolov8l-worldv2.pt` |

| Dataset | Split used | Images | Boxes | Classes scored |
|---|---|---|---|---|
| COCO | val2017 | 5,000 | native 2D | 80 |
| KITTI | 1,000 from `training/` (seed 0) | 1,000 | native 2D | car, person (+ cyclist variant) |
| nuImages | v1.0-mini (all keyframes) | 50 | native 2D | person, bicycle, car, motorcycle, bus, truck |
| nuScenes | v1.0-mini, 404 keyframes × 6 cameras | 2,424 | **3D boxes projected to 2D** | same 6 |

Only the **mini** splits of nuImages/nuScenes are downloadable without an account; they are evaluation-only subsets (no training here).

### Requirement checklist (Track A)

| Requirement | Section |
|---|---|
| ≥3 configs, ≥2 families | §1 (5 families, 6 configs) |
| ≥2 datasets + taxonomy mapping | §2 (4 datasets) |
| mAP, AP50/75, AP S/M/L, per-class AP, AR, latency/FPS, memory | §4, §5, §10 |
| Reproducibility | §3 |
| Evidence of running | §3, §4 live inference |
| Cross-dataset generalization | §6, §7 |
| Architecture comparison | §9 |
| Closed-set vs open-vocabulary | §8 |
| Speed–accuracy | §10 |
| Failure analysis | §11 |
| Conclusions, references | §12, §13 |
""")

code(r"""
import os, json, copy, contextlib, io, subprocess, time, sys
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir("..")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2, torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from IPython.display import Markdown, display

pd.set_option("display.precision", 1)
R = Path("results")
DS = {"coco": ("data/coco/annotations/instances_val2017.json", "data/coco/val2017"),
      "kitti": ("data/kitti/kitti_coco.json", "data/kitti/training/image_2"),
      "nuimages": ("data/nuimages/nuimages_coco.json", "data/nuimages"),
      "nuscenes": ("data/nuscenes/nuscenes_coco.json", "data/nuscenes")}
DSN = {"coco": "COCO", "kitti": "KITTI", "nuimages": "nuImages", "nuscenes": "nuScenes"}
MODELS = [("YOLO26-L", "yolo26l"), ("YOLO26-L +NMS", "yolo26l-nms"), ("RT-DETR-L", "rtdetr-l"),
          ("Faster R-CNN v2", "frcnn-r50-fpn-v2"), ("RetinaNet v2", "retinanet-r50-fpn-v2"), ("YOLO-World-L", "yoloworld-l")]
NAMES = [n for n, _ in MODELS]
COLORS = dict(zip(NAMES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
STATS = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]

def load(tag):
    p = R / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None

res = {(n, d): load(f"{t}_{d}") for n, t in MODELS for d in DS}
missing = [k for k, v in res.items() if v is None]
assert not missing, f"missing runs: {missing}"

def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)

def run_eval(gt, pred_file, cat_ids=None, img_ids=None):
    E = COCOeval(gt, quiet(gt.loadRes, pred_file), "bbox")
    if cat_ids: E.params.catIds = cat_ids
    if img_ids is not None: E.params.imgIds = img_ids
    quiet(E.evaluate); quiet(E.accumulate); quiet(E.summarize)
    return E

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors="#6b6a63"); ax.set_axisbelow(True); ax.grid(axis="y", color="#e4e3dd", lw=0.8)

def iou_xywh(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)

GT = {d: quiet(COCO, a) for d, (a, _) in DS.items()}
print(len(res), "runs loaded")
""")

# 1 models
md(r"""
## 1. Models

| | YOLO26-L | RT-DETR-L | Faster R-CNN v2 | RetinaNet v2 | YOLO-World-L |
|---|---|---|---|---|---|
| Family | one-stage | DETR | two-stage | dense one-stage | open-vocabulary one-stage |
| Backbone / neck | CNN + PAN | HGNetv2 + hybrid encoder | ResNet-50 + FPN | ResNet-50 + FPN | YOLOv8-L CNN + RepVL-PAN, CLIP text encoder |
| Anchors | anchor-free | queries | anchor-based (RPN) | anchor-based (9/location) | anchor-free |
| Classification | 80-way head | 80-way head | 91-way head | 91-way sigmoid (focal loss) | **region–text similarity** with prompt embeddings |
| Training assignment | TAL (one-to-many) + one-to-one | Hungarian | IoU matching | IoU matching | TAL |
| Post-processing | NMS-free (or NMS 0.7) | NMS-free top-k | NMS 0.5 | NMS 0.5 | NMS 0.7 |
| Trained on | COCO | COCO | COCO | COCO | Objects365 + GoldG + CC3M (**no COCO**) |

YOLO-World is prompted with the 80 COCO class names (in COCO order), so its outputs use the same class ids as the closed-set models. On COCO it is therefore **zero-shot**.
""")
code(r"""
pd.DataFrame([{"Model": n, "Params (M)": res[(n, "coco")]["params_M"], "Framework": res[(n, "coco")]["model"]["framework"],
               "Checkpoint": res[(n, "coco")]["model"]["ckpt"]} for n in NAMES]).set_index("Model")
""")

# 2 datasets
md(r"""
## 2. Datasets and taxonomy mapping

**KITTI → COCO** (`convert_kitti.py`): Car → car, Pedestrian → person. Van, Truck, Misc, Cyclist, Person_sitting, DontCare → ignore region (`iscrowd=1`). Tram dropped.

**nuImages / nuScenes → COCO** (`convert_nuimages.py`, `convert_nuscenes.py`, same mapping):

| nuScenes/nuImages category | → COCO |
|---|---|
| human.pedestrian.{adult, child, construction_worker, police_officer} | person |
| vehicle.car | car |
| vehicle.truck | truck |
| vehicle.bus.{rigid, bendy} | bus |
| vehicle.motorcycle / vehicle.bicycle **without rider** | motorcycle / bicycle |
| vehicle.motorcycle / bicycle **with rider** | ignore (box covers rider + cycle; COCO boxes them separately) |
| vehicle.construction, trailer, emergency.*, pedestrian.{personal_mobility, stroller, wheelchair} | ignore |
| barrier, traffic cone, debris, bicycle rack, animal | dropped |

**nuScenes 2D boxes are projections of 3D boxes**: global → ego → camera, project the 8 corners, take the bounding box of (convex hull ∩ image), as in the devkit's `export_2d_annotations_as_json.py`. They cover the full 3D extent, including occluded parts, so they are looser than hand-drawn boxes. Boxes with visibility `v0-40` (0–40% visible) are ignore regions because many are completely hidden.

*Ignore region* = pycocotools neither rewards nor penalizes detections on it, and never counts it as missed.
""")
code(r"""
rows = []
for d in DS:
    g = GT[d].dataset
    scored = [a for a in g["annotations"] if not a.get("iscrowd")]
    cats = {c["id"]: c["name"] for c in g["categories"]}
    row = {"Dataset": DSN[d], "images": len(g["images"]), "scored boxes": len(scored),
           "ignore boxes": sum(1 for a in g["annotations"] if a.get("iscrowd")),
           "median box h (px)": np.median([a["bbox"][3] for a in scored])}
    for cid, nm in [(1, "person"), (3, "car")]:
        row[f"{nm} boxes"] = sum(1 for a in scored if a["category_id"] == cid)
        row[f"{nm} median h"] = np.median([a["bbox"][3] for a in scored if a["category_id"] == cid] or [np.nan])
    rows.append(row)
pd.DataFrame(rows).set_index("Dataset")
""")
code(r"""
figs = [("KITTI", "kitti_gt_000009.png"), ("nuImages", "nuimages_gt_1.png"), ("nuScenes (projected 3D)", "nuscenes_gt_1213.png")]
fig, axes = plt.subplots(3, 1, figsize=(12, 13))
for ax, (t, f) in zip(axes, figs):
    ax.imshow(cv2.cvtColor(cv2.imread(str(R / "figs" / f)), cv2.COLOR_BGR2RGB)); ax.axis("off")
    ax.set_title(f"{t} GT — green = scored, gray = ignore", loc="left")
plt.tight_layout(); plt.show()
""")

# 3 reproducibility
md("## 3. Reproducibility")
code(r"""
import platform, torchvision, ultralytics
print("GPU      :", torch.cuda.get_device_name(), "| driver", subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print("Software : Python", platform.python_version(), "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| ultralytics", ultralytics.__version__, "| CUDA", torch.version.cuda, "| conda env cnn")
rows = []
for (n, d), r in res.items():
    m = r["model"]
    nms = "none" if (m.get("nms") is False or n == "RT-DETR-L") else f"IoU {m.get('nms_iou', m.get('iou', 0.7))}"
    rows.append({"Model": n, "Dataset": DSN[d], "images": r["dataset"]["num_images"], "resize": m["resize"], "conf": m["conf"],
                 "NMS": nms, "max det": m["max_det"], "command": r["command"]})
pd.DataFrame(rows).set_index(["Model", "Dataset"])
""")
md("Protocol for every run: batch 1, fp32, conf 0.001, pycocotools COCOeval (maxDets 100); latency = end-to-end `predict()` with CUDA sync, first 20 images (10 for nuImages) excluded as warm-up. Commands: `logs/commands.md`; environment: `requirements-lock.txt`.")

# 4 live
md("## 4. Evidence: live inference (same `detectors/` code used by `eval.py`)")
code(r"""
from detectors import load_detector
imgs = {"KITTI": cv2.imread("data/kitti/training/image_2/000009.png"),
        "nuImages": cv2.imread(str(Path("data/nuimages") / GT["nuimages"].imgs[1]["file_name"]))}
keys = [("YOLO26-L", "yolo26l"), ("RT-DETR-L", "rtdetr-l"), ("Faster R-CNN v2", "frcnn"), ("RetinaNet v2", "retinanet"), ("YOLO-World-L", "yoloworld-l")]
fig, axes = plt.subplots(len(keys), 2, figsize=(16, 3.4 * len(keys)), gridspec_kw={"width_ratios": [1.6, 1]})
for row, (name, key) in zip(axes, keys):
    det = load_detector(key)
    for ax, (dname, img) in zip(row, imgs.items()):
        for _ in range(3): det.predict(img)
        torch.cuda.synchronize(); t0 = time.perf_counter(); d = det.predict(img); torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000; keep = d.scores >= 0.4; vis = img.copy()
        for (x1, y1, x2, y2), c in zip(d.boxes[keep].astype(int), d.cat_ids[keep]):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 230) if c == 3 else (230, 120, 0) if c == 1 else (0, 200, 0), 2)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax.axis("off")
        ax.set_title(f"{name} on {dname}: {keep.sum()} dets ≥0.4, {ms:.1f} ms", loc="left", fontsize=10)
    del det; torch.cuda.empty_cache()
plt.suptitle("red = car, blue = person, green = other classes", x=0.01, ha="left"); plt.tight_layout(); plt.show()
""")

# 5 results
md("## 5. Results — all 12 COCO metrics per dataset (×100)")
code(r"""
PUBLISHED = {"YOLO26-L": "≈55", "YOLO26-L +NMS": "≈55", "RT-DETR-L": "53.0", "Faster R-CNN v2": "46.7",
             "RetinaNet v2": "41.5", "YOLO-World-L": "zero-shot (docs ≈45–47)"}
for d in DS:
    display(Markdown(f"**{DSN[d]}** — {res[(NAMES[0], d)]['dataset']['num_images']} images"))
    t = pd.DataFrame({n: {k: 100 * v for k, v in res[(n, d)]["metrics"].items()} for n in NAMES}).T[STATS]
    if d == "coco": t["published AP"] = [PUBLISHED[n] for n in t.index]
    display(t)
""")
code(r"""
M = pd.DataFrame({DSN[d]: {n: 100 * res[(n, d)]["metrics"]["AP"] for n in NAMES} for d in DS})
fig, ax = plt.subplots(figsize=(8, 3.8))
im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=M.values.max(), aspect="auto")
ax.set_xticks(range(M.shape[1]), M.columns); ax.set_yticks(range(M.shape[0]), M.index)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]; ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v > 0.6 * M.values.max() else "#1f1f1e")
ax.set_title("mAP@[.5:.95] — each dataset uses its own scored classes", loc="left", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.03); plt.tight_layout(); plt.show()
""")
code(r"""
for d in ("nuimages", "nuscenes"):
    display(Markdown(f"**Per-class AP — {DSN[d]}**"))
    display(pd.DataFrame({n: {k: 100 * v if v is not None else np.nan for k, v in res[(n, d)]["per_class_AP"].items()} for n in NAMES}).T)
""")

# 6 cross dataset
md(r"""
## 6. Cross-dataset generalization on shared classes (car + person)

Each dataset scores different classes, so the mAP heatmap above is not apples-to-apples. Here every dataset is re-scored on **car + person only** (KITTI already is).
""")
code(r"""
CP = {}
for n, _ in MODELS:
    for d in DS:
        E = res[(n, d)]["metrics"] if d == "kitti" else None
        if d == "kitti":
            CP[(n, d)] = dict(AP=E["AP"], AP50=E["AP50"], AP75=E["AP75"], S=E["AP_small"], M=E["AP_medium"], L=E["AP_large"],
                              car=res[(n, d)]["per_class_AP"]["car"], person=res[(n, d)]["per_class_AP"]["person"])
        else:
            ev = run_eval(GT[d], res[(n, d)]["pred_file"], [1, 3]); s = ev.stats
            p = ev.eval["precision"][:, :, :, 0, -1]
            pc = {c: float(p[:, :, k][p[:, :, k] > -1].mean()) for k, c in zip(range(2), ("person", "car"))}
            CP[(n, d)] = dict(AP=s[0], AP50=s[1], AP75=s[2], S=s[3], M=s[4], L=s[5], **pc)
cp = pd.DataFrame({(n, DSN[d]): {k: 100 * v for k, v in CP[(n, d)].items()} for n, _ in MODELS for d in DS}).T
cp.index.names = ["Model", "Dataset"]
display(cp["AP"].unstack("Dataset")[[DSN[d] for d in DS]].rename_axis(None, axis=1).style.format("{:.1f}").set_caption("car+person AP"))
display(cp[["car", "person"]].unstack("Dataset").round(1))
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
dsets = [DSN[d] for d in DS]; w = 0.8 / len(NAMES)
for ax, col, title in zip(axes, ("AP", "car", "person"), ("car+person AP", "car AP", "person AP")):
    for i, n in enumerate(NAMES):
        ax.bar(np.arange(4) + (i - (len(NAMES) - 1) / 2) * w, [cp.loc[(n, d), col] for d in dsets], w * 0.9, color=COLORS[n], label=n)
    ax.set_xticks(range(4), dsets); ax.set_title(title, loc="left"); style(ax)
axes[0].set_ylabel("AP@[.5:.95]"); axes[2].legend(frameon=False, fontsize=7, loc="upper right"); plt.tight_layout(); plt.show()
""")
code(r"""
display(Markdown("**Δ AP by object size, driving dataset − COCO (car + person)**"))
rows = []
for n in NAMES:
    for d in ("kitti", "nuimages", "nuscenes"):
        rows.append({"Model": n, "Dataset": DSN[d], **{s: cp.loc[(n, DSN[d]), s] - cp.loc[(n, "COCO"), s] for s in ("AP", "S", "M", "L")}})
pd.DataFrame(rows).set_index(["Dataset", "Model"]).rename(columns={"S": "Δ small", "M": "Δ medium", "L": "Δ large", "AP": "Δ AP"})
""")

# 7 nuScenes vs nuImages
md(r"""
## 7. Projected 3D boxes vs native 2D boxes — nuScenes deep dive

nuImages and nuScenes come from the same sensor rig and cities (Boston, Singapore). The main difference is the **box source**: hand-drawn 2D (nuImages) vs projected 3D (nuScenes). Loose projected boxes should hurt strict IoU (AP75) much more than AP50.
""")
code(r"""
rows = []
for n in NAMES:
    for d in ("nuimages", "nuscenes"):
        c = cp.loc[(n, DSN[d])]
        rows.append({"Model": n, "Dataset": DSN[d], "AP50": c["AP50"], "AP75": c["AP75"], "AP": c["AP"], "AP75/AP50": c["AP75"] / c["AP50"]})
pd.DataFrame(rows).set_index(["Model", "Dataset"]).unstack("Dataset").round(2)
""")
code(r"""
def subset_gt(d, keep):
    g = copy.deepcopy(GT[d].dataset)
    for a in g["annotations"]:
        if not a.get("iscrowd") and not keep(a): a["area"] = 1e11  # out of areaRng -> ignored with normal IoU
    c = COCO(); c.dataset = g; quiet(c.createIndex); return c
VIS = {"v40-60": lambda a: a["visibility"] == "v40-60", "v60-80": lambda a: a["visibility"] == "v60-80", "v80-100": lambda a: a["visibility"] == "v80-100"}
vis_gt = {k: subset_gt("nuscenes", f) for k, f in VIS.items()}
cams = sorted({im["camera"] for im in GT["nuscenes"].dataset["images"]})
cam_ids = {c: [im["id"] for im in GT["nuscenes"].dataset["images"] if im["camera"] == c] for c in cams}
rows = []
for n in NAMES:
    pf = res[(n, "nuscenes")]["pred_file"]; row = {"Model": n}
    for k, g in vis_gt.items(): row[("visibility", k)] = 100 * run_eval(g, pf, [1, 3]).stats[0]
    for c in cams: row[("camera", c.replace("CAM_", ""))] = 100 * run_eval(GT["nuscenes"], pf, [1, 3], cam_ids[c]).stats[0]
    rows.append(row)
nv = pd.DataFrame(rows).set_index("Model"); nv.columns = pd.MultiIndex.from_tuples(nv.columns)
print("scored GT per visibility:", {k: sum(1 for a in GT['nuscenes'].dataset['annotations'] if not a['iscrowd'] and f(a)) for k, f in VIS.items()})
display(Markdown("**nuScenes car+person AP by visibility level and by camera**")); nv
""")

code(r"""
# nuScenes scene conditions from scene descriptions (night / rain / day)
meta = lambda n: json.loads((Path("data/nuscenes/v1.0-mini") / f"{n}.json").read_text())
scene_desc = {sc["token"]: sc["description"].lower() for sc in meta("scene")}
sample_scene = {s["token"]: s["scene_token"] for s in meta("sample")}
file_scene = {sd["filename"]: sample_scene[sd["sample_token"]] for sd in meta("sample_data") if sd["is_key_frame"]}
def cond(fn):
    d = scene_desc[file_scene[fn]]
    return "night" if "night" in d else "rain" if "rain" in d else "day"
cond_ids = {}
for im in GT["nuscenes"].dataset["images"]: cond_ids.setdefault(cond(im["file_name"]), []).append(im["id"])
print({k: f"{len(v)} images" for k, v in cond_ids.items()})
pd.DataFrame({n: {c: 100 * run_eval(GT["nuscenes"], res[(n, "nuscenes")]["pred_file"], [1, 3], ids).stats[0] for c, ids in cond_ids.items()}
              for n in NAMES}).T.rename_axis("Model").style.format("{:.1f}").set_caption("nuScenes car+person AP by scene condition")
""")

# 8 open vocab
md(r"""
## 8. Closed-set vs open-vocabulary (YOLO-World-L)

1. **Same vocabulary**: YOLO-World prompted with the 80 COCO names vs COCO-trained closed-set models (§5–6). YOLO-World never saw COCO, so its COCO result is zero-shot, while every driving dataset is zero-shot for all models.
2. **Vocabulary the closed-set models don't have**: KITTI `Cyclist` (rider + bike in one box). Variant GT `kitti-cyclist` scores Cyclist as its own class. Closed-set models can only answer with COCO `bicycle` (the bike alone); YOLO-World can be prompted with the word **"cyclist"**.
""")
code(r"""
rows = []
for n, t in MODELS:
    r = load(f"{t}_kitti-cyclist")
    if r: rows.append({"Model": n, "vocabulary": "COCO-80 (cyclist ← 'bicycle')", **{c: 100 * v for c, v in r["per_class_AP"].items()}, "mAP": 100 * r["metrics"]["AP"]})
for p in ("person-cyclist-car", "pedestrian-cyclist-car"):
    r = load(f"yoloworld-l_p-{p}_kitti-cyclist")
    if r: rows.append({"Model": "YOLO-World-L", "vocabulary": "prompts: " + p.replace("-", ", "), **{c: 100 * v for c, v in r["per_class_AP"].items()}, "mAP": 100 * r["metrics"]["AP"]})
pd.DataFrame(rows).set_index(["Model", "vocabulary"])
""")
code(r"""
r = load("yoloworld-l_p-person-cyclist-car_kitti-cyclist")
g = GT_c = quiet(COCO, "data/kitti/kitti_coco_cyclist.json")
cyc_imgs = [i for i in g.getImgIds() if any(a["category_id"] == 2 and not a["iscrowd"] for a in g.imgToAnns[i])]
preds = {}
for p in json.loads(Path(r["pred_file"]).read_text()):
    if p["score"] >= 0.3: preds.setdefault(p["image_id"], []).append(p)
rb = {}
for p in json.loads(Path(load("yolo26l_kitti-cyclist")["pred_file"]).read_text()):
    if p["score"] >= 0.3: rb.setdefault(p["image_id"], []).append(p)
fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
img_id = max(cyc_imgs, key=lambda i: sum(a["category_id"] == 2 for a in g.imgToAnns[i]))
for ax, (title, pp) in zip(axes, [("YOLO26-L (COCO vocab): 'bicycle' boxes = bike only", rb), ("YOLO-World-L prompted 'cyclist'", preds)]):
    im = cv2.imread(f"data/kitti/training/image_2/{g.imgs[img_id]['file_name']}")
    for a in g.imgToAnns[img_id]:
        if a["category_id"] == 2 and not a["iscrowd"]:
            x, y, w, h = map(int, a["bbox"]); cv2.rectangle(im, (x, y), (x + w, y + h), (0, 200, 0), 2)
    for p in pp.get(img_id, []):
        if p["category_id"] == 2:
            x, y, w, h = map(int, p["bbox"]); cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 230), 2)
            cv2.putText(im, f"{p['score']:.2f}", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 230), 1)
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.axis("off"); ax.set_title(title + " — green = GT Cyclist, red = prediction", loc="left")
plt.tight_layout(); plt.show()
""")

# 9 architecture
md(r"""
## 9. Architecture comparison — anchors, queries, NMS

Controlled pairs: **Faster R-CNN vs RetinaNet** share ResNet-50-FPN and input size (two-stage vs one-stage); **YOLO26-L vs YOLO26-L +NMS** share weights (NMS-free vs NMS). Duplicates = same-class pairs with IoU > 0.7, both score ≥ 0.3, per 100 detections.
""")
code(r"""
def dup_rate(pred_file):
    by = {}
    for p in json.loads(Path(pred_file).read_text()):
        if p["score"] >= 0.3: by.setdefault(p["image_id"], []).append(p)
    n, dup = 0, 0
    for ps in by.values():
        n += len(ps)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                dup += ps[i]["category_id"] == ps[j]["category_id"] and iou_xywh(ps[i]["bbox"], ps[j]["bbox"]) > 0.7
    return 100 * dup / max(n, 1)
POST = {"YOLO26-L": "NMS-free (one-to-one)", "YOLO26-L +NMS": "NMS", "RT-DETR-L": "NMS-free (queries)",
        "Faster R-CNN v2": "NMS", "RetinaNet v2": "NMS", "YOLO-World-L": "NMS"}
rows = []
for n in NAMES:
    row = {"Model": n, "post-proc": POST[n]}
    for d in DS:
        m = res[(n, d)]["metrics"]; row[(DSN[d], "AP")] = 100 * m["AP"]; row[(DSN[d], "AR100")] = 100 * m["AR100"]
        row[(DSN[d], "dup/100")] = dup_rate(res[(n, d)]["pred_file"])
    rows.append(row)
arch = pd.DataFrame(rows).set_index(["Model", "post-proc"]); arch.columns = pd.MultiIndex.from_tuples(arch.columns); arch
""")

# 10 speed
md("## 10. Speed–accuracy tradeoff")
code(r"""
rows = []
for n in NAMES:
    row = {"Model": n, ("", "Params (M)"): res[(n, "coco")]["params_M"]}
    for d in DS:
        r = res[(n, d)]; row[(DSN[d], "ms")] = r["speed"]["mean_ms"]; row[(DSN[d], "FPS")] = r["speed"]["fps"]; row[(DSN[d], "mem MB")] = r["gpu_peak_mem_MB"]
    rows.append(row)
sp = pd.DataFrame(rows).set_index("Model"); sp.columns = pd.MultiIndex.from_tuples(sp.columns); sp
""")
code(r"""
fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, d in zip(axes, DS):
    for n in NAMES:
        r = res[(n, d)]; ax.scatter(r["speed"]["mean_ms"], 100 * r["metrics"]["AP"], s=70, color=COLORS[n], edgecolor="white", linewidth=1.5, zorder=3, label=n)
    ax.set_title(DSN[d], loc="left"); ax.set_xlabel("ms / image"); style(ax); ax.grid(axis="x", color="#e4e3dd", lw=0.8)
axes[0].set_ylabel("mAP@[.5:.95]"); axes[0].legend(frameon=False, fontsize=7); plt.tight_layout(); plt.show()
""")

# 11 failures
md("## 11. Failure analysis")
code(r"""
def misses(n, d, thr=0.3):
    g = GT[d]; pr = {}
    for p in json.loads(Path(res[(n, d)]["pred_file"]).read_text()):
        if p["score"] >= thr: pr.setdefault(p["image_id"], []).append(p)
    out = []
    for a in g.dataset["annotations"]:
        if a.get("iscrowd") or a["category_id"] not in (1, 3): continue
        if not any(p["category_id"] == a["category_id"] and iou_xywh(p["bbox"], a["bbox"]) >= 0.5 for p in pr.get(a["image_id"], [])):
            out.append(a)
    return out, pr
rows = []
for d in ("kitti", "nuimages", "nuscenes"):
    tot = [a for a in GT[d].dataset["annotations"] if not a.get("iscrowd") and a["category_id"] in (1, 3)]
    for n in NAMES:
        m, _ = misses(n, d)
        row = {"Dataset": DSN[d], "Model": n, "car+person GT": len(tot), "missed": len(m), "miss %": 100 * len(m) / len(tot),
               "% small (<40px)": 100 * np.mean([a["bbox"][3] < 40 for a in m]), "% person": 100 * np.mean([a["category_id"] == 1 for a in m])}
        if d == "kitti": row["% occluded (occ≥2 / vis 40-60)"] = 100 * np.mean([a["occluded"] >= 2 for a in m])
        if d == "nuscenes": row["% occluded (occ≥2 / vis 40-60)"] = 100 * np.mean([a["visibility"] == "v40-60" for a in m])
        rows.append(row)
pd.DataFrame(rows).set_index(["Dataset", "Model"])
""")
code(r"""
fig, axes = plt.subplots(3, 1, figsize=(13, 13))
for ax, d in zip(axes, ("kitti", "nuimages", "nuscenes")):
    m, pr = misses("YOLO26-L", d)
    from collections import Counter
    img_id = Counter(a["image_id"] for a in m).most_common(1)[0][0]
    info = GT[d].imgs[img_id]; im = cv2.imread(str(Path(DS[d][1]) / info["file_name"]))
    missed_ids = {a["id"] for a in m}
    for a in GT[d].imgToAnns[img_id]:
        x, y, w, h = map(int, a["bbox"])
        col = (150, 150, 150) if a.get("iscrowd") else (0, 200, 255) if a["id"] in missed_ids else (0, 200, 0)
        cv2.rectangle(im, (x, y), (x + w, y + h), col, 2)
    for p in pr.get(img_id, []):
        if p["category_id"] in (1, 3):
            x, y, w, h = map(int, p["bbox"]); cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 230), 1)
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.axis("off")
    ax.set_title(f"{DSN[d]} — YOLO26-L worst image ({sum(a['image_id'] == img_id for a in m)} missed): green hit, yellow missed, red pred, gray ignore", loc="left", fontsize=10)
plt.tight_layout(); plt.show()
""")

md(r"""
## 12. Conclusions

- **No single winner across domains.** On COCO: YOLO26-L 53.8 > RT-DETR-L 52.8 > Faster R-CNN 46.9 > YOLO-World-L 44.7 (zero-shot) > RetinaNet 41.4. On KITTI all six configs land within 1 AP (39.6–40.4); on nuImages Faster R-CNN and YOLO26-L lead; on nuScenes YOLO-World-L is marginally best (18.4) and RetinaNet clearly last (14.6).
- **Pedestrians are the domain-shift casualty.** Car AP: COCO ≈50 → KITTI ≈50 → nuImages ≈44 → nuScenes ≈29. Person AP: COCO ≈60 → KITTI ≈30 → nuImages ≈26 → nuScenes ≈9, even though person boxes are not smaller in the driving data.
- **Occlusion/visibility dominates.** nuScenes AP is ≈22 for 80–100% visible objects vs ≈3 for 40–60% visible (KITTI: ≈52 visible vs ≈6 largely occluded).
- **Projected 3D boxes are a weak 2D benchmark.** nuScenes AP50 (≈51) is close to nuImages (≈55–62), but AP75 collapses (≈10 vs ≈35): AP75/AP50 ≈ 0.2 vs ≈ 0.6. Most of the low nuScenes mAP is box looseness, not missed objects.
- **Open-vocabulary matches closed-set on driving data without COCO training.** YOLO-World-L is 9 AP below YOLO26-L on COCO but equal on KITTI/nuScenes. However, prompting "cyclist" does **not** recover KITTI Cyclist (AP < 1); the prompt "pedestrian" is much worse than "person" (12.5 vs 26.9 AP) — results are sensitive to prompt wording.
- **Night is a minor factor here:** nuScenes night scenes (726 images) lose only 1–3 AP vs day; YOLO26-L, RT-DETR-L and YOLO-World-L lose ≈1–2, the ResNet-50 CNNs ≈3.
- **Two-stage vs one-stage (same ResNet-50-FPN, same input):** Faster R-CNN beats RetinaNet everywhere (COCO +5.5, nuScenes +3.2) at ~1.2× the latency.
- **NMS-free models keep ~1.4–3.5 near-duplicates per 100 detections** on every dataset; NMS models ~0. YOLO26's NMS-free head has higher AR100 everywhere (+2 to +7) at equal AP.
- **Speed–accuracy:** YOLO26-L (≈5.6 ms) and YOLO-World-L (≈6 ms) are 2–5× faster than RT-DETR-L (≈14 ms), RetinaNet (≈20 ms) and Faster R-CNN (≈25 ms) with equal or better driving-domain accuracy.
- **Caveats:** nuImages is only 50 images (one fence scene with ~40 pedestrians causes ~20% of its misses); nuScenes/nuImages are the public mini splits.
""")

md(r"""
## 13. References

1. Ren et al., *Faster R-CNN*, NeurIPS 2015.
2. Lin et al., *Focal Loss for Dense Object Detection (RetinaNet)*, ICCV 2017.
3. Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017.
4. Li et al., *Benchmarking Detection Transfer Learning with Vision Transformers*, arXiv:2111.11429 (torchvision v2 recipes).
5. Carion et al., *End-to-End Object Detection with Transformers (DETR)*, ECCV 2020.
6. Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)*, CVPR 2024.
7. Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, NeurIPS 2024.
8. Cheng et al., *YOLO-World: Real-Time Open-Vocabulary Object Detection*, CVPR 2024.
9. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML 2021.
10. Lin et al., *Microsoft COCO*, ECCV 2014.
11. Geiger et al., *The KITTI Vision Benchmark Suite*, CVPR 2012.
12. Caesar et al., *nuScenes: A Multimodal Dataset for Autonomous Driving*, CVPR 2020 (nuImages: nuscenes.org/nuimages).
13. Ultralytics docs/code — https://docs.ultralytics.com, https://github.com/ultralytics/ultralytics
14. torchvision detection models — https://pytorch.org/vision/stable/models.html
15. nuScenes devkit (2D export reference) — https://github.com/nutonomy/nuscenes-devkit
16. pycocotools — https://github.com/cocodataset/cocoapi
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (cnn)", "language": "python"}})
nbf.write(nb, "notebooks/level2_extended.ipynb")
print(len(cells), "cells")
