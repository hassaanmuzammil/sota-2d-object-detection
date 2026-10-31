# 2D Object Detection: 5 Detector Families × 4 Datasets

COCO-pretrained detectors from five families are evaluated, **without retraining**, on COCO and three autonomous-driving datasets (KITTI, nuImages, nuScenes). The goal is to measure domain shift, compare architectures, compare closed-set vs. open-vocabulary detection, and measure the speed–accuracy tradeoff.

## 1. Models

Ordered by release year. YOLO26-L is evaluated twice (with and without NMS), giving 6 configurations.

| Model | Family | Year | Checkpoint | Params | Framework | Post-processing | Training data |
|---|---|---|---|---|---|---|---|
| Faster R-CNN v2 | two-stage | 2015 | [`FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth) | 43.7M | torchvision 0.26.0 | NMS 0.5 | COCO |
| RetinaNet v2 | dense one-stage, anchor-based | 2017 | [`RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1`](https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth) | 38.2M | torchvision 0.26.0 | NMS 0.5 | COCO |
| RT-DETR-L | DETR (transformer queries) | 2023 | [`rtdetr-l.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt) | 33.0M | Ultralytics 8.4.194 | NMS-free, top-k | COCO |
| YOLO-World-L v2 | open-vocabulary | 2024 | [`yolov8l-worldv2.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-worldv2.pt) | 46.8M | Ultralytics 8.4.194 | NMS 0.7 | Objects365 + GoldG + CC3M (**no COCO**) |
| YOLO26-L | one-stage, anchor-free | 2025 | [`yolo26l.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt) | 26.3M | Ultralytics 8.4.194 | NMS-free (`nms=False`) | COCO |
| YOLO26-L +NMS | same weights | 2025 | same | 26.3M | Ultralytics 8.4.194 | NMS 0.7 | COCO |

- **YOLO26-L** has two heads (one-to-one and one-to-many). Ultralytics runs NMS by default, so both modes are evaluated.
- **YOLO-World-L** is prompted with the 80 COCO class names in COCO order, so its class ids match the other models. It never trained on COCO, so even its COCO result is zero-shot.

## 2. Model architecture comparison

| | Faster R-CNN v2 | RetinaNet v2 | RT-DETR-L | YOLO-World-L v2 | YOLO26-L |
|---|---|---|---|---|---|
| Backbone | ResNet-50 | ResNet-50 | HGNetv2 | YOLOv8-L CSPDarknet + CLIP text encoder | CSP-style CNN (C3k2 blocks) |
| Neck | FPN (P2–P6) | FPN (P3–P7) | hybrid encoder (intra-scale attention + cross-scale CNN fusion) | RepVL-PAN (image features fused with text embeddings) | PAN |
| Detection head | RPN proposes regions → RoIAlign → box head classifies and refines | dense anchors at every location: class subnet + box subnet | transformer decoder, 300 object queries | anchor-free dense head; class score = region–text similarity | anchor-free dense head, two branches (one-to-one, one-to-many) |
| Classifier | 91-way softmax | 91-way sigmoid | 80-way sigmoid | text-embedding similarity | 80-way sigmoid |
| Loss | cross-entropy + smooth L1 (RPN and head) | focal loss + GIoU | varifocal + L1 + GIoU | region–text contrastive + CIoU + DFL | BCE + IoU loss, ProgLoss (no DFL) |
| Assignment | IoU threshold to anchors / proposals | IoU threshold to anchors | Hungarian matching (one-to-one) | task-aligned (TAL) | task-aligned (STAL): one-to-many + one-to-one |
| Inference | two passes: proposals, then per-region classification | one pass over all anchors | one pass; queries decode boxes directly | one pass; text embeddings computed once per vocabulary | one pass on dense grid |
| Post-processing | per-class NMS 0.5 | per-class NMS 0.5 | none (top-k scores) | NMS 0.7 | none (one-to-one head) or NMS 0.7 |

Sources: [Faster R-CNN](https://arxiv.org/abs/1506.01497), [FPN](https://arxiv.org/abs/1612.03144), [RetinaNet](https://arxiv.org/abs/1708.02002), torchvision v2 recipe ([Li et al.](https://arxiv.org/abs/2111.11429), [torchvision models](https://pytorch.org/vision/stable/models.html)), [RT-DETR](https://arxiv.org/abs/2304.08069), [YOLO-World](https://arxiv.org/abs/2401.17270), [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/).

## 3. Datasets

Ordered by release year.

| Dataset | Year | Split used | Images | Resolution | Box source | Classes scored | Scored boxes | Download |
|---|---|---|---|---|---|---|---|---|
| KITTI | 2012 | 1,000 random from `training/` (seed 0) | 1,000 | 1242×375 | hand-drawn 2D | car, person | 4,441 | [images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip), [labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip) |
| COCO | 2014 | val2017 | 5,000 | ~640×480 | hand-drawn 2D | 80 | 36,335 | [images](http://images.cocodataset.org/zips/val2017.zip), [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) |
| nuScenes | 2019 | v1.0-mini, 404 keyframes × 6 cameras | 2,424 | 1600×900 | **3D boxes projected to 2D** | person, bicycle, car, motorcycle, bus, truck | 11,842 | [v1.0-mini](https://www.nuscenes.org/data/v1.0-mini.tgz) |
| nuImages | 2020 | v1.0-mini (all keyframes) | 50 | 1600×900 | hand-drawn 2D | same 6 | 372 | [v1.0-mini](https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz) |

- nuScenes and nuImages use the public **mini** splits: they're the only versions downloadable without an account, and they fit the disk budget.
- **nuScenes 2D boxes** follow the official devkit export ([`export_2d_annotations_as_json.py`](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py)): move each 3D box from world → vehicle → camera coordinates, project the 8 corners in front of the camera, and take the bounding box of (convex hull ∩ image). These boxes cover the whole 3D object, including hidden parts, so they're **looser** than hand-drawn boxes.

## 4. Taxonomy

The target label space is COCO's native classes; nothing is merged or renamed. **Ignore** means a crowd region (`iscrowd=1`): pycocotools neither counts a prediction on it as a false positive nor counts the object as missed.

**KITTI → COCO** ([convert_kitti.py](convert_kitti.py))

| KITTI | COCO |
|---|---|
| Car | car |
| Pedestrian | person |
| Van, Truck, Misc, Cyclist, Person_sitting, DontCare | ignore |
| Tram | dropped |

**nuImages / nuScenes → COCO** ([convert_nuimages.py](convert_nuimages.py), [convert_nuscenes.py](convert_nuscenes.py))

| nuScenes category | COCO |
|---|---|
| pedestrian.{adult, child, construction_worker, police_officer} | person |
| vehicle.car | car |
| vehicle.truck | truck |
| vehicle.bus.{rigid, bendy} | bus |
| vehicle.bicycle / motorcycle **without** rider | bicycle / motorcycle |
| vehicle.bicycle / motorcycle **with** rider | ignore |
| vehicle.construction, trailer, emergency.*, pedestrian.{personal_mobility, stroller, wheelchair} | ignore |
| nuScenes only: visibility 0–40% | ignore |
| barrier, traffic cone, debris, bicycle rack, animal | dropped |

**KITTI cyclist variant** (`--cyclist`, used only for the open-vocabulary test in §9.3)

| KITTI | Ground truth | Prediction used |
|---|---|---|
| Car | car | COCO car |
| Pedestrian | person | COCO person |
| Cyclist | cyclist (scored) | COCO bicycle (closed-set models); "cyclist" prompt (YOLO-World) |
| Van, Truck, Misc, Person_sitting, DontCare | ignore | — |
| Tram | dropped | — |

### Why this taxonomy

- **No retraining:** the models output COCO classes, so they are scored in COCO's label space with no remapping of predictions.
- **Fair scoring:** a COCO model reasonably calls a KITTI Van "car" or a Person_sitting "person". Ignore regions keep those predictions from counting as false positives, without requiring the objects to be found.
- **KITTI Truck is ignored, nuScenes truck is scored:** COCO models label KITTI trucks as either car or truck, so the class isn't a clean match. nuScenes `vehicle.truck` matches COCO truck closely.
- **Rider + cycle is ignored:** the driving datasets draw one box around rider and cycle; COCO draws two separate boxes. Neither would match.
- **Low visibility (nuScenes) is ignored:** projected boxes of objects 0–40% visible are often completely hidden in the image.
- **Dropped classes** have no COCO equivalent.

### Comparison with ngdet

[ngdet](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md) is the course's detection tutorial. It maps every dataset into a small unified taxonomy instead.

| | This study | ngdet |
|---|---|---|
| Target classes | COCO native (car, person, truck, bus, bicycle, motorcycle) | unified presets: `driving3` (vehicle, person, cyclist; default), `vehicle_person`, `driving5` |
| Vehicles | kept separate (car / truck / bus) | merged into `vehicle` (driving3) |
| Cyclist | rider + cycle → ignore (scored only in the cyclist variant) | COCO `bicycle` → `cyclist`; the tutorial notes this gives cyclist AP ≈ 0 because the boxes don't match |
| Ambiguous classes (Van, Misc, DontCare, trailer, …) | ignore regions | out-of-taxonomy classes → ignore (DontCare / crowd handling not documented) |
| nuScenes visibility | 0–40% → ignore region | `min_visibility` / `require_lidar_pts` filter (threshold not stated) |
| Category matching | explicit per-category table | right-to-left token matching on dotted names |

| Pros of this taxonomy | Cons of this taxonomy |
|---|---|
| No translation of predictions, so one less source of error | KITTI is scored on only 2 classes |
| Avoids the cyclist ≈ 0 artifact in the main results | Generous ignore regions inflate AP vs. a strict benchmark |
| Ambiguous objects don't count as false positives | Rider + cycle objects are never scored in the main results |
| Car / truck / bus confusions stay visible | Not directly comparable with ngdet's driving3 numbers |
| Directly comparable with COCO val2017 results | Datasets score different class sets, so cross-dataset comparison needs a car + person re-score (§9.1) |

## 5. Preprocessing

Each model uses its framework's standard preprocessing.

| | Ultralytics (YOLO26, RT-DETR, YOLO-World) | torchvision (Faster R-CNN, RetinaNet) |
|---|---|---|
| Input | BGR uint8 | BGR → RGB, float in [0, 1] |
| Resize | letterbox, long side 640, aspect ratio kept | short side 800, long side ≤ 1333 |
| KITTI 1242×375 becomes | 640×193 | 1333×403 |
| nuScenes / nuImages 1600×900 becomes | 640×360 | 1333×750 |
| Normalization | ÷255 | ImageNet mean/std |
| Output boxes | rescaled to original pixels | rescaled to original pixels |
| Class ids | 0–79 → COCO 91-ids (`coco80_to_coco91_class`) | already COCO 91-ids |

## 6. Inference and evaluation settings

| Setting | Value |
|---|---|
| Confidence threshold | 0.001 (all models, standard for mAP) |
| NMS IoU | YOLO26-L: none · YOLO26-L +NMS: 0.7 · RT-DETR-L: none · Faster R-CNN / RetinaNet: 0.5 · YOLO-World-L: 0.7 |
| Max detections | 300 (Ultralytics), 100 (torchvision); COCOeval keeps the top 100 |
| Precision / batch | fp32, batch 1 |
| Predictions kept | only classes present in the dataset's ground truth |
| Metrics | [pycocotools](https://github.com/cocodataset/cocoapi) COCOeval: 12 standard stats + per-class AP |
| Latency | end-to-end `predict()` including pre- and post-processing, CUDA-synchronized; first 20 images excluded as warm-up (10 for nuImages) |
| GPU memory | `torch.cuda.max_memory_allocated()` |
| Hardware | NVIDIA RTX 5080 16 GB, driver 595.99.02, CUDA 12.8 |
| Software | Python 3.11.17, torch 2.11.0, torchvision 0.26.0, ultralytics 8.4.194, pycocotools 2.0.11 |

## 7. Results

All AP / AR values are ×100. Bold = best per dataset. Raw numbers for every run are in `results/<model>_<dataset>.json`.

### 7.1 mAP@[0.50:0.95]

| Model | COCO | KITTI | nuImages | nuScenes |
|---|---|---|---|---|
| Faster R-CNN v2 | 46.9 | **40.4** | **37.7** | 17.8 |
| RetinaNet v2 | 41.4 | 39.8 | 33.4 | 14.6 |
| RT-DETR-L | 52.8 | 40.1 | 33.3 | 17.9 |
| YOLO-World-L | 44.7 | 39.9 | 36.3 | **18.4** |
| YOLO26-L | 53.8 | 39.6 | 37.3 | 17.8 |
| YOLO26-L +NMS | **54.0** | 39.8 | 35.9 | 17.5 |

The COCO numbers match the published ones (RT-DETR-L 52.8 vs. 53.0, Faster R-CNN 46.9 vs. 46.7, RetinaNet 41.4 vs. 41.5). YOLO26-L is about 1 point below its docs because they use `yolo val` and this study uses per-image `predict()`.

![mAP heatmap](results/figs/ext_heatmap.png)

### 7.2 AP50 / AP75

| Model | COCO | KITTI | nuImages | nuScenes |
|---|---|---|---|---|
| Faster R-CNN v2 | 67.9 / 51.3 | **74.6** / 38.3 | **61.3** / 35.1 | **43.6** / 13.0 |
| RetinaNet v2 | 61.6 / 43.8 | 73.0 / **39.1** | 52.1 / 36.7 | 37.2 / 9.6 |
| RT-DETR-L | **71.3** / 57.2 | 72.5 / 38.6 | 52.8 / 36.2 | 42.7 / 13.6 |
| YOLO-World-L | 60.2 / 48.6 | 72.6 / 39.2 | 50.5 / **37.7** | 43.4 / **14.3** |
| YOLO26-L | 70.7 / 58.8 | 72.9 / 38.4 | 53.1 / 37.2 | 42.6 / 12.9 |
| YOLO26-L +NMS | 71.0 / **58.9** | 72.7 / 38.9 | 51.5 / 37.3 | 42.0 / 12.6 |

### 7.3 AP small / medium / large

| Model | COCO | KITTI | nuImages | nuScenes |
|---|---|---|---|---|
| Faster R-CNN v2 | 31.0 / 50.8 / 60.3 | **25.7** / **38.0** / 55.4 | **19.9** / **33.9** / 54.4 | **1.7** / **11.5** / 22.2 |
| RetinaNet v2 | 26.8 / 45.7 / 53.7 | 24.4 / 37.7 / 55.3 | 10.4 / 29.9 / 48.7 | 1.1 / 8.2 / 19.4 |
| RT-DETR-L | 34.4 / 57.4 / **70.8** | 25.1 / 37.3 / **57.2** | 6.2 / 28.0 / 53.1 | 1.1 / 10.6 / 23.0 |
| YOLO-World-L | 29.2 / 49.0 / 57.2 | 22.7 / 37.4 / **57.2** | 16.4 / 28.6 / 58.4 | 1.1 / 11.3 / **23.9** |
| YOLO26-L | **37.2** / 58.0 / 69.5 | 23.6 / 37.9 / 55.5 | 7.6 / 27.6 / **61.3** | 1.3 / **11.5** / 23.1 |
| YOLO26-L +NMS | 35.9 / **58.3** / 69.8 | 23.8 / 37.8 / 55.6 | 4.9 / 25.9 / 60.3 | 1.2 / 11.1 / 22.7 |

### 7.4 Per-class AP

**Car / person on all datasets** (the two classes every dataset shares):

| Model | car: COCO / KITTI / nuImages / nuScenes | person: COCO / KITTI / nuImages / nuScenes |
|---|---|---|
| Faster R-CNN v2 | 49.9 / 49.6 / 45.7 / 29.2 | 59.6 / **31.3** / **28.9** / 8.2 |
| RetinaNet v2 | 43.4 / 49.2 / 41.4 / 27.3 | 53.5 / 30.4 / 23.4 / **9.0** |
| RT-DETR-L | 50.1 / **51.3** / **46.1** / **30.5** | 61.7 / 29.0 / 22.8 / 8.7 |
| YOLO-World-L | 48.3 / 50.3 / 45.9 / 30.4 | 62.4 / 29.4 / 26.3 / 8.4 |
| YOLO26-L | **53.7** / 49.5 / 43.3 / 29.5 | **65.0** / 29.6 / 25.6 / 8.6 |

**Other nuImages / nuScenes classes** (bicycle / motorcycle / bus / truck):

| Model | nuImages | nuScenes |
|---|---|---|
| Faster R-CNN v2 | 40.0 / 34.4 / **40.6** / 36.8 | 5.5 / 8.9 / 32.9 / **21.9** |
| RetinaNet v2 | 44.0 / 31.4 / 27.9 / 32.1 | 4.2 / 5.9 / 23.8 / 17.6 |
| RT-DETR-L | 36.7 / 39.7 / 22.6 / 31.7 | 4.3 / 9.5 / 33.6 / 20.8 |
| YOLO-World-L | **48.8** / 41.1 / 22.1 / 33.5 | **6.0** / **11.4** / **34.6** / 19.7 |
| YOLO26-L | 47.7 / **44.9** / 23.3 / **39.1** | 4.9 / 10.6 / 33.9 / 19.3 |

nuImages has only 372 boxes, so its per-class numbers are noisy (bus and motorcycle have a handful of instances each).

### 7.5 AR (AR@100)

| Model | COCO | KITTI | nuImages | nuScenes |
|---|---|---|---|---|
| Faster R-CNN v2 | 63.0 | **51.5** | 49.7 | 30.1 |
| RetinaNet v2 | 57.8 | 49.2 | 46.0 | 26.9 |
| RT-DETR-L | 71.7 | 50.4 | 48.5 | 31.3 |
| YOLO-World-L | 63.2 | 49.6 | 48.5 | 31.5 |
| YOLO26-L | **73.1** | 51.3 | **50.5** | **32.5** |
| YOLO26-L +NMS | 69.6 | 49.1 | 44.0 | 28.0 |

### 7.6 Inference latency / FPS

Mean end-to-end latency per image (ms), with FPS in brackets.

| Model | COCO | KITTI | nuImages | nuScenes |
|---|---|---|---|---|
| Faster R-CNN v2 | 24.5 (41) | 19.1 (52) | 30.2 (33) | 28.4 (35) |
| RetinaNet v2 | 20.4 (49) | 15.6 (64) | 24.5 (41) | 23.1 (43) |
| RT-DETR-L | 13.6 (74) | 13.6 (73) | 14.3 (70) | 13.7 (73) |
| YOLO-World-L | 6.9 (145) | **5.3 (190)** | 6.6 (151) | 6.1 (163) |
| YOLO26-L | **5.6 (178)** | 5.5 (182) | **5.7 (175)** | **5.7 (176)** |
| YOLO26-L +NMS | 5.7 (177) | 5.4 (185) | 6.1 (165) | 5.7 (176) |

### 7.7 GPU memory

Peak allocated GPU memory (MB) during inference.

| Model | COCO | KITTI | nuImages / nuScenes |
|---|---|---|---|
| Faster R-CNN v2 | 671 | 441 | 670 |
| RetinaNet v2 | 377 | 342 | 384 |
| RT-DETR-L | 239 | 239 | 239 |
| YOLO-World-L | 305 | 248 | 269 |
| YOLO26-L / +NMS | **221** | **163** | **185** |

## 8. How to reproduce

**1. Environment**

```bash
pip install -r requirements-lock.txt     # Python 3.11, torch 2.11.0+cu128, ultralytics 8.4.194
```

**2. Weights.** Download the three Ultralytics checkpoints (§1) into `weights/`. torchvision downloads its two checkpoints automatically on first use.

**3. Data.** Download the files in §3 and extract them to:

```
data/coco/val2017/, data/coco/annotations/instances_val2017.json
data/kitti/training/{image_2,label_2}/
data/nuimages/{samples,v1.0-mini}/
data/nuscenes/{samples,v1.0-mini}/
```

**4. Convert ground truth to COCO format**

```bash
python convert_kitti.py --n 1000 --seed 0          # → data/kitti/kitti_coco.json
python convert_kitti.py --cyclist --viz 0 --out data/kitti/kitti_coco_cyclist.json
python convert_nuimages.py                         # → data/nuimages/nuimages_coco.json
python convert_nuscenes.py                         # → data/nuscenes/nuscenes_coco.json
```

**5. Evaluate** (one run per model × dataset; writes `results/<model>_<dataset>.json` and predictions to `results/preds/`)

```bash
python eval.py --model {frcnn|retinanet|rtdetr-l|yoloworld-l|yolo26l|yolo26l-nms} \
               --dataset {coco|kitti|nuimages|nuscenes|kitti-cyclist}
python eval.py --model yoloworld-l --dataset kitti-cyclist --prompts "person,cyclist,car"
python eval.py --model yoloworld-l --dataset kitti-cyclist --prompts "pedestrian,cyclist,car"
```

**6. Analyses and figures** (car + person re-score, visibility, day/night, duplicates, failure images)

```bash
python notebooks/build_level2.py
cd notebooks && jupyter nbconvert --to notebook --execute --inplace level2_extended.ipynb
```

## 9. Technical analysis

### 9.1 Cross-dataset generalization

Which classes and object scales degrade most when moving from COCO to driving data?

![car and person AP across datasets](results/figs/ext_car_person.png)

- **Cars transfer well; pedestrians don't.** Car AP stays around 50 on KITTI, but person AP roughly halves on every driving dataset (e.g. YOLO26-L 65.0 → 29.6). Object size doesn't explain it: median person height is similar (83 px COCO, 87 px KITTI, 98 px nuScenes).
- **Small objects collapse** on nuImages and nuScenes (YOLO26-L AP small: 37.2 on COCO → 7.6 / 1.3); on KITTI small and medium objects drop by similar amounts. Faster R-CNN holds up best on small objects (19.9 on nuImages), helped by its larger input.
- **Occlusion is the biggest factor.** On nuScenes, car + person AP is about 22 for objects 80–100% visible but only about 3 for 40–60% visible. On KITTI, heavily occluded parked cars are 54–70% of all misses.
- **Night costs little:** only 1–3 AP on nuScenes (YOLO26-L 19.7 → 18.4).
- **nuScenes' low scores are mostly a label artifact.** nuScenes and nuImages have the same cameras and cities, and their AP50 is similar (YOLO26-L 42.6 vs. 53.1). AP75 collapses on nuScenes (12.9 vs. 37.2) because its projected 3D boxes are looser than the objects. The [ngdet tutorial](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md) makes the same observation.

### 9.2 Architecture comparison

- **Two-stage vs. one-stage (Faster R-CNN vs. RetinaNet).** Same backbone, neck and input size. Faster R-CNN wins on every dataset (COCO +5.5, nuImages +4.3, nuScenes +3.2) for about 1.2× the latency. It also misses the fewest objects everywhere (12% miss rate on KITTI vs. 25% for the worst model), helped by its larger input (about 2× the pixels of the 640-px models).
- **CNN vs. transformer (YOLO26 vs. RT-DETR).** Nearly identical accuracy everywhere (within 1 AP except nuImages); YOLO26-L is about 2.4× faster with fewer parameters.
- **NMS vs. NMS-free.** YOLO26-L's NMS-free head matches its NMS head on AP and has higher AR100 on every dataset (+2 to +7). Both NMS-free models leave some near-duplicate boxes (1.4–3.5 per 100 detections; same class, IoU > 0.7, score ≥ 0.3), while NMS models leave none.
- **No single winner across domains.** YOLO26-L leads on COCO, Faster R-CNN on KITTI and nuImages, YOLO-World-L on nuScenes. On driving data the models are within about 1 AP on KITTI and about 4 AP elsewhere.

### 9.3 Closed-set vs. open-vocabulary

- **Same 80-class vocabulary:** YOLO-World-L, which never saw COCO, is 9 AP behind YOLO26-L on COCO. On the driving datasets it **matches or beats** the COCO-trained models (KITTI 39.9, nuScenes 18.4, best), so its broader training data generalizes as well as COCO training.
- **A class COCO models don't have (KITTI Cyclist, cyclist variant from §4):**

| Model | Vocabulary | person | cyclist | car |
|---|---|---|---|---|
| YOLO26-L | COCO (cyclist ← bicycle) | 25.5 | 0.06 | 49.5 |
| RT-DETR-L | COCO (cyclist ← bicycle) | 26.3 | 0.24 | 51.3 |
| Faster R-CNN v2 | COCO (cyclist ← bicycle) | 28.0 | 0.14 | 49.6 |
| YOLO-World-L | prompts "person, cyclist, car" | 26.9 | 0.77 | 50.9 |
| YOLO-World-L | prompts "pedestrian, cyclist, car" | 12.5 | 2.1 | 51.0 |

![cyclist comparison](results/figs/ext_cyclist.png)

- Closed-set "bicycle" boxes cover only the bike, so they rarely overlap a KITTI Cyclist box (rider + bike) enough to count.
- Prompting "cyclist" **does not fix it** (AP < 1): YOLO-World gives the prompt very low scores.
- **Prompt wording matters a lot:** "pedestrian" instead of "person" halves person AP (26.9 → 12.5).

### 9.4 Speed–accuracy tradeoff

![speed vs accuracy](results/figs/ext_speed.png)

| Model | Params | Latency (COCO) | Peak memory | mAP COCO / KITTI / nuScenes |
|---|---|---|---|---|
| YOLO26-L | 26.3M | 5.6 ms | 163–221 MB | 53.8 / 39.6 / 17.8 |
| YOLO-World-L | 46.8M | 6.9 ms | 248–305 MB | 44.7 / 39.9 / 18.4 |
| RT-DETR-L | 33.0M | 13.6 ms | 239 MB | 52.8 / 40.1 / 17.9 |
| RetinaNet v2 | 38.2M | 20.4 ms | 342–384 MB | 41.4 / 39.8 / 14.6 |
| Faster R-CNN v2 | 43.7M | 24.5 ms | 441–671 MB | 46.9 / 40.4 / 17.8 |

- **YOLO26-L is the best tradeoff:** the fastest (≈178 FPS), the least memory, and within 1 AP of the best model on every driving dataset.
- The torchvision models are 3–5× slower and use 1.7–3× more memory, mainly because they run at about 2× the input resolution. That resolution buys Faster R-CNN its small-object and recall advantage, but not a meaningful mAP lead.
- Model size doesn't predict speed: YOLO-World-L has the most parameters but is the second fastest.

## 10. Limitations

- nuImages is only 50 images (372 boxes), so its per-class numbers are noisy.
- nuScenes and nuImages use the mini splits; KITTI uses a 1,000-image subset with a single seed.
- Zero-shot only; no fine-tuning.
- Latency is Python, fp32, batch 1, including pre- and post-processing; no TensorRT or fp16.
- YOLO-World's COCO score is zero-shot, so its COCO comparison isn't like-for-like.
