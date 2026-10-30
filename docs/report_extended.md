# 2D Object Detection: 5 Detector Families × 4 Datasets (Extended Study)

This extends the MVP report ([report.md](report.md)) with two more detector families (RetinaNet, YOLO-World) and two more driving datasets (nuImages, nuScenes). The full analysis, with outputs saved, is in `notebooks/results_extended.ipynb`.

## 1. Goal

Measure how COCO-pretrained detectors from five families generalize, with no retraining, to three driving datasets. Specific questions:
- Which classes, object sizes and visibility levels degrade?
- How does the choice of evaluation boxes (hand-drawn 2D vs. projected 3D) change the results?
- Does an open-vocabulary detector help?
- How does speed trade off against accuracy?

## 2. Models

| | YOLO26-L | RT-DETR-L | Faster R-CNN v2 | RetinaNet v2 | YOLO-World-L v2 |
|---|---|---|---|---|---|
| Family | one-stage, anchor-free | DETR (queries) | two-stage | dense one-stage, anchor-based | open-vocabulary |
| Checkpoint | `yolo26l.pt` | `rtdetr-l.pt` | torchvision `COCO_V1` | torchvision `COCO_V1` | `yolov8l-worldv2.pt` |
| Params | 26.3M | 33.0M | 43.7M | 38.2M | 46.8M |
| Backbone / neck | CNN + PAN | HGNetv2 + hybrid encoder | ResNet-50 + FPN | ResNet-50 + FPN | YOLOv8-L + RepVL-PAN + CLIP text encoder |
| Classifier | 80-way | 80-way | 91-way | 91-way sigmoid (focal loss) | region–text similarity |
| Assignment | TAL + one-to-one | Hungarian | IoU | IoU | TAL |
| Post-processing | NMS-free (also run with NMS) | NMS-free, top-k | NMS 0.5 | NMS 0.5 | NMS 0.7 |
| Training data | COCO | COCO | COCO | COCO | Objects365 + GoldG + CC3M (**no COCO**) |

- **YOLO26-L** has two heads. Ultralytics runs NMS by default (`nms=None`), so we evaluate both `YOLO26-L` (NMS-free, `nms=False`) and `YOLO26-L +NMS`.
- **YOLO-World-L** is prompted with the 80 COCO class names, in COCO order, so its class ids match the other models. It never trained on COCO, so even its COCO result is zero-shot.

## 3. Datasets

| Dataset | Split used | Images | Resolution | Box source | Classes scored |
|---|---|---|---|---|---|
| COCO | val2017 | 5,000 | ~640×480 | hand-drawn 2D | 80 |
| KITTI | 1,000 from `training/`, seed 0 | 1,000 | 1242×375 | hand-drawn 2D | car, person |
| nuImages | v1.0-mini (all keyframes) | 50 | 1600×900 | hand-drawn 2D | person, bicycle, car, motorcycle, bus, truck |
| nuScenes | v1.0-mini, 404 keyframes × 6 cameras | 2,424 | 1600×900 | **3D boxes projected to 2D** | same 6 |

- nuImages and nuScenes use the public **mini** splits: they're the only versions downloadable without an account and they're small enough for disk space. They're used for evaluation only.
- Scored boxes: COCO 36,335 · KITTI 4,441 · nuImages 372 · nuScenes 11,842.

### Taxonomy mapping

"Ignore region" (`iscrowd=1`): pycocotools skips predictions that land on it and never counts it as a missed object. We use it for objects a COCO model would reasonably call car or person but that the dataset labels differently.

**KITTI** (`convert_kitti.py`)

| KITTI | COCO |
|---|---|
| Car | car |
| Pedestrian | person |
| Van, Truck, Misc, Cyclist, Person_sitting, DontCare | ignore |
| Tram | dropped |

**nuImages / nuScenes** (`convert_nuimages.py`, `convert_nuscenes.py`)

| nuScenes category | COCO |
|---|---|
| pedestrian.{adult, child, construction_worker, police_officer} | person |
| vehicle.car | car |
| vehicle.truck | truck |
| vehicle.bus.{rigid, bendy} | bus |
| vehicle.bicycle / motorcycle **without rider** | bicycle / motorcycle |
| vehicle.bicycle / motorcycle **with rider** | ignore (one box covers rider + cycle; COCO boxes them separately) |
| vehicle.construction, trailer, emergency.*, pedestrian.{personal_mobility, stroller, wheelchair} | ignore |
| nuScenes only: any box 0–40% visible | ignore (often completely hidden) |
| barrier, traffic cone, debris, bicycle rack, animal | dropped |

**nuScenes 2D boxes** follow the official devkit export (`export_2d_annotations_as_json.py`):
1. Move each 3D box from world coordinates → vehicle → camera.
2. Project the 8 corners that are in front of the camera.
3. Take the bounding box of (convex hull ∩ image).

These boxes cover the whole 3D object, including hidden parts, so they're **looser** than hand-drawn boxes.

**KITTI cyclist variant** (`--cyclist`): same as KITTI, but Cyclist is scored as its own class. Used only for the open-vocabulary test (§9).

## 4. Preprocessing

| | Ultralytics (YOLO26, RT-DETR, YOLO-World) | torchvision (Faster R-CNN, RetinaNet) |
|---|---|---|
| Input | BGR uint8 | BGR → RGB, float in [0, 1] |
| Resize | letterbox, long side 640, aspect ratio kept | short side 800, long side ≤ 1333 |
| KITTI 1242×375 | 640×193 | 1333×403 |
| nuScenes/nuImages 1600×900 | 640×360 | 1333×750 |
| Normalization | ÷255 | ImageNet mean/std |
| Boxes | rescaled to original pixels | rescaled to original pixels |
| Class ids | 0–79 → COCO 91-ids (`coco80_to_coco91_class`) | already COCO 91-ids |

## 5. Inference and evaluation settings

| Setting | Value |
|---|---|
| Confidence threshold | 0.001 (all models) |
| NMS | YOLO26-L: none · YOLO26-L +NMS: 0.7 · RT-DETR-L: none · Faster R-CNN / RetinaNet: 0.5 · YOLO-World-L: 0.7 |
| Max detections | 300 (Ultralytics), 100 (torchvision); COCOeval keeps the top 100 |
| Precision / batch | fp32, batch 1 |
| Predictions kept | only classes present in the dataset's ground truth |
| Metrics | pycocotools COCOeval: 12 stats + per-class AP |
| Latency | end-to-end `predict()` with CUDA sync; first 20 images excluded (10 for nuImages) |
| Memory | `torch.cuda.max_memory_allocated()` |
| Hardware | RTX 5080 16 GB, driver 595.99.02, CUDA 12.8 |
| Software | Python 3.11.17, torch 2.11.0, torchvision 0.26.0, ultralytics 8.4.194 |

```bash
python convert_kitti.py --n 1000 --seed 0
python convert_kitti.py --cyclist --viz 0 --out data/kitti/kitti_coco_cyclist.json
python convert_nuimages.py
python convert_nuscenes.py
python eval.py --model {yolo26l|yolo26l-nms|rtdetr-l|frcnn|retinanet|yoloworld-l} --dataset {coco|kitti|nuimages|nuscenes|kitti-cyclist}
python eval.py --model yoloworld-l --dataset kitti-cyclist --prompts "person,cyclist,car"
python eval.py --model yoloworld-l --dataset kitti-cyclist --prompts "pedestrian,cyclist,car"
```

## 6. Results (mAP@[.5:.95] ×100)

![mAP heatmap](../results/figs/ext_heatmap.png)

| Model | COCO | KITTI | nuImages | nuScenes |
|---|---|---|---|---|
| YOLO26-L | **53.8** | 39.6 | 37.3 | 17.8 |
| YOLO26-L +NMS | **54.0** | 39.8 | 35.9 | 17.5 |
| RT-DETR-L | 52.8 | 40.1 | 33.3 | 17.9 |
| Faster R-CNN v2 | 46.9 | **40.4** | **37.7** | 17.8 |
| RetinaNet v2 | 41.4 | 39.8 | 33.4 | 14.6 |
| YOLO-World-L | 44.7 | 39.9 | 36.3 | **18.4** |

The COCO numbers match the published ones: RT-DETR-L 52.8 vs. 53.0, Faster R-CNN 46.9 vs. 46.7, RetinaNet 41.4 vs. 41.5. YOLO26-L is about 1 point below the docs because they use `yolo val` and we use per-image `predict()`. The notebook (§5) has all 12 metrics and per-class AP for every dataset.

## 7. Analysis 1: Cross-dataset generalization (car + person only)

Each dataset scores different classes, so we re-scored all of them on **car + person** to compare like with like.

![car and person AP across datasets](../results/figs/ext_car_person.png)

| Model | car: COCO / KITTI / nuImages / nuScenes | person: COCO / KITTI / nuImages / nuScenes |
|---|---|---|
| YOLO26-L | 53.7 / 49.5 / 43.3 / 29.5 | 65.0 / 29.6 / 25.6 / 8.6 |
| RT-DETR-L | 50.1 / 51.3 / 46.1 / 30.5 | 61.7 / 29.0 / 22.8 / 8.7 |
| Faster R-CNN v2 | 49.9 / 49.6 / 45.7 / 29.2 | 59.6 / 31.3 / 28.9 / 8.2 |
| RetinaNet v2 | 43.4 / 49.2 / 41.4 / 27.3 | 53.5 / 30.4 / 23.4 / 9.0 |
| YOLO-World-L | 48.3 / 50.3 / 45.9 / 30.4 | 62.4 / 29.4 / 26.3 / 8.4 |

- **Cars transfer well; pedestrians don't.** Car AP stays at about 50 on KITTI, while person AP halves on every driving dataset. Size isn't the reason: median person height is 83 px on COCO, 87 on KITTI and 98 on nuScenes.
- **By size,** medium objects drop most on every dataset (e.g. YOLO26-L: −33 KITTI, −43 nuImages, −58 nuScenes).
- **Smallest drop:** RetinaNet (−8.6 AP on KITTI), but it also starts lowest on COCO. Faster R-CNN has the smallest drop among the strong models.

**Visibility** (nuScenes, car + person AP):

| Model | 40–60% visible | 60–80% | 80–100% |
|---|---|---|---|
| YOLO26-L | 2.6 | 5.8 | 21.9 |
| RT-DETR-L | 2.8 | 6.6 | 22.6 |
| Faster R-CNN v2 | 2.8 | 5.9 | 21.3 |
| YOLO-World-L | 1.7 | 4.1 | 22.7 |

Partly hidden objects are almost never detected. On KITTI it's the same: about 52 AP when fully visible vs. about 6 when largely occluded (MVP report).

**Night** (nuScenes: 726 night images, 1,698 day): night costs only 1–3 AP. YOLO26-L goes 19.7 → 18.4 and Faster R-CNN 19.5 → 16.7.

## 8. Analysis 2: Projected 3D boxes vs. hand-drawn 2D boxes

nuImages and nuScenes come from the same company, cars and cities. The main difference is where the boxes come from.

| Model | AP50 nuImages / nuScenes | AP75 nuImages / nuScenes |
|---|---|---|
| YOLO26-L | 54.2 / 51.7 | 34.7 / 10.5 |
| RT-DETR-L | 56.6 / 52.5 | 35.3 / 11.2 |
| Faster R-CNN v2 | 62.0 / 51.5 | 36.4 / 9.7 |
| YOLO-World-L | 55.9 / 51.1 | 35.0 / 11.1 |

At the loose IoU 0.5 threshold, nuScenes is only slightly below nuImages. At the strict IoU 0.75 threshold it collapses (AP75/AP50 is about 0.2 vs. about 0.6). **Most of nuScenes' low mAP comes from loose projected boxes, not missed objects.** Projected 3D boxes are a poor benchmark for 2D detection; the course's ngdet tutorial noted the same.

## 9. Analysis 3: Closed-set vs. open-vocabulary

**Same 80-class vocabulary:** YOLO-World-L, which never trained on COCO, is 9 AP behind YOLO26-L on COCO. On KITTI and nuScenes it **matches or beats** the COCO-trained models (KITTI 39.9, nuScenes 18.4), so its broader training data generalizes as well as COCO training.

**A class the COCO models don't have (KITTI Cyclist):**

| Model | Vocabulary | person | cyclist | car |
|---|---|---|---|---|
| YOLO26-L | COCO (cyclist ← "bicycle") | 25.5 | 0.06 | 49.5 |
| RT-DETR-L | COCO (cyclist ← "bicycle") | 26.3 | 0.24 | 51.3 |
| Faster R-CNN v2 | COCO (cyclist ← "bicycle") | 28.0 | 0.14 | 49.6 |
| YOLO-World-L | COCO (cyclist ← "bicycle") | 25.9 | 0.09 | 50.3 |
| YOLO-World-L | prompts "person, cyclist, car" | 26.9 | **0.77** | 50.9 |
| YOLO-World-L | prompts "pedestrian, cyclist, car" | 12.5 | 2.1 | 51.0 |

![cyclist comparison](../results/figs/ext_cyclist.png)

- Closed-set "bicycle" boxes cover only the bike, so they rarely overlap a KITTI Cyclist box enough to count (AP < 0.3).
- Prompting **"cyclist" does not fix it** (AP < 1). YOLO-World gives the prompt very low scores.
- **The exact prompt wording matters a lot:** "pedestrian" instead of "person" halves person AP (26.9 → 12.5).

## 10. Analysis 4: Architecture and speed vs. accuracy

**NMS vs. NMS-free** (near-duplicates per 100 detections: same class, IoU > 0.7, score ≥ 0.3):

| Model | Post-proc | COCO | KITTI | nuImages | nuScenes |
|---|---|---|---|---|---|
| YOLO26-L | NMS-free | 1.9 | 1.4 | 1.5 | 1.8 |
| RT-DETR-L | NMS-free | 3.5 | 1.8 | 2.6 | 1.9 |
| NMS models | NMS | ≈0 | 0 | 0 | 0 |

- Both NMS-free models leave some duplicate boxes on every dataset.
- YOLO26's NMS-free head has the same AP as its NMS head but higher AR100 everywhere (+2 to +7).

**Two-stage vs. one-stage, same backbone and input size:** Faster R-CNN beats RetinaNet on every dataset (COCO +5.5, nuImages +4.3, nuScenes +3.2) at about 1.2× RetinaNet's latency.

**Speed vs. accuracy:**

| Model | Params | COCO ms | nuScenes ms | Peak mem (MB) |
|---|---|---|---|---|
| YOLO26-L | 26.3M | **5.6** | **5.7** | **163–221** |
| YOLO-World-L | 46.8M | 6.9 | 6.1 | 248–305 |
| RT-DETR-L | 33.0M | 13.6 | 13.7 | 239 |
| RetinaNet v2 | 38.2M | 20.4 | 23.1 | 342–384 |
| Faster R-CNN v2 | 43.7M | 24.5 | 28.4 | 441–671 |

![speed vs accuracy](../results/figs/ext_speed.png)

YOLO26-L and YOLO-World-L are 2–5× faster than the others, with equal or better driving-domain accuracy.

## 11. Failure analysis

![worst images](../results/figs/ext_failures.png)

*Green = detected, yellow = missed, red = prediction, gray = ignore.*

| Dataset | Miss rate (car + person, score ≥ 0.3), best → worst | Main causes |
|---|---|---|
| KITTI | 12% (Faster R-CNN) → 25% (YOLO-World) | heavily occluded parked cars (54–70% of misses), objects under 40 px tall |
| nuImages | 36% (Faster R-CNN) → 63% (YOLO-World) | pedestrians are 67–75% of misses; one scene with about 40 small pedestrians behind a fence |
| nuScenes | 34% (Faster R-CNN) → 55% (YOLO-World) | loose projected boxes, partly visible objects, unlabeled distant cars, night scenes |

Faster R-CNN misses the fewest objects everywhere. It processes about 2× the pixels of the 640-px models, which helps on small objects.

## 12. Conclusions

- **No single winner across domains.** YOLO26-L leads on COCO, Faster R-CNN on KITTI and nuImages, YOLO-World-L on nuScenes. The gaps on driving data are small (≤1 AP on KITTI).
- **Pedestrians and partly hidden objects** cause most of the domain-shift loss; cars transfer well.
- **Projected 3D boxes** (nuScenes) badly underrate 2D detectors at strict IoU thresholds.
- **Open-vocabulary training matches COCO training on driving data**, but text prompts don't fix label mismatches like KITTI Cyclist, and results depend heavily on prompt wording.
- **NMS-free models cost no AP** but leave 1.4–3.5 duplicates per 100 boxes.
- **YOLO26-L is the best speed vs. accuracy option:** about 5.6 ms (≈178 FPS), the least memory, and within 1 AP of the best on every driving dataset.

## 13. Limitations

- nuImages is only 50 images (372 boxes), so its per-class numbers are noisy.
- nuScenes and nuImages use the mini splits, and KITTI uses a 1,000-image subset with one seed.
- Zero-shot only; there was no fine-tuning.
- Latency is measured in Python with fp32 and includes pre- and post-processing; there was no TensorRT.
- YOLO-World's COCO score is zero-shot, so its COCO comparison isn't like-for-like with the COCO-trained models.

## 14. References

1. Ren et al., *Faster R-CNN*, NeurIPS 2015.
2. Lin et al., *Focal Loss for Dense Object Detection (RetinaNet)*, ICCV 2017.
3. Lin et al., *Feature Pyramid Networks*, CVPR 2017.
4. Li et al., *Benchmarking Detection Transfer Learning with Vision Transformers*, arXiv:2111.11429.
5. Carion et al., *DETR*, ECCV 2020.
6. Zhao et al., *RT-DETR: DETRs Beat YOLOs on Real-time Object Detection*, CVPR 2024.
7. Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, NeurIPS 2024.
8. Cheng et al., *YOLO-World: Real-Time Open-Vocabulary Object Detection*, CVPR 2024.
9. Radford et al., *CLIP*, ICML 2021.
10. Lin et al., *Microsoft COCO*, ECCV 2014.
11. Geiger et al., *KITTI Vision Benchmark Suite*, CVPR 2012.
12. Caesar et al., *nuScenes*, CVPR 2020; nuImages: https://www.nuscenes.org/nuimages
13. Ultralytics: https://docs.ultralytics.com, https://github.com/ultralytics/ultralytics
14. torchvision detection models: https://pytorch.org/vision/stable/models.html
15. nuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
16. pycocotools: https://github.com/cocodataset/cocoapi
17. Course tutorial `ngdet`: https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md
