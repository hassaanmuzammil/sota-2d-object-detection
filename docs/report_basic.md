# 2D Object Detection: COCO → KITTI Evaluation (Track A)

## 1. Goal

Evaluate three COCO-pretrained detectors from different families on COCO (in-domain) and KITTI (driving, zero-shot, no retraining). Measure how accuracy changes under domain shift and how speed trades off against accuracy.

## 2. Models

| | YOLO26-L | RT-DETR-L | Faster R-CNN v2 |
|---|---|---|---|
| Family | one-stage CNN | query-based transformer (DETR) | two-stage CNN |
| Checkpoint | `yolo26l.pt` (Ultralytics v8.4.0 assets) | `rtdetr-l.pt` (Ultralytics v8.4.0 assets) | torchvision `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` |
| Params | 26.3M | 33.0M | 43.7M |
| Backbone / neck | CNN + PAN | HGNetv2 + hybrid encoder | ResNet-50 + FPN |
| Anchors | anchor-free | none (300 object queries) | anchor-based (RPN) |
| Training assignment | one-to-many (TAL) + one-to-one heads | Hungarian matching | IoU matching |
| Post-processing | NMS-free one-to-one head | NMS-free, top-k queries | NMS |
| Framework | Ultralytics 8.4.194 | Ultralytics 8.4.194 | torchvision 0.26.0 |

All checkpoints are COCO-trained. YOLO26-L's checkpoint records `train_args.data = coco.yaml`. Faster R-CNN uses torchvision's COCO_V1 recipe. RT-DETR-L has no dataset recorded in its checkpoint, so we rely on the Ultralytics docs. None of these list KITTI in their training data.

**YOLO26 has two heads.** In Ultralytics 8.4.194, `predict()` defaults to `nms=None`, which uses the one-to-many head **with NMS**. The NMS-free one-to-one head is only used with `nms=False`. We evaluate both:
- `YOLO26-L` = NMS-free
- `YOLO26-L +NMS` = one-to-many head + NMS (IoU 0.7)

## 3. Datasets

| | COCO val2017 | KITTI 2D object |
|---|---|---|
| Images | 5,000 (full val set) | 1,000 sampled from `training/` (seed 0, list in `data/kitti_list.txt`) |
| Why this split | standard COCO benchmark | KITTI test labels are not public |
| Resolution | varied, ~640×480 | 1242×375 |
| Classes scored | 80 | 2 (car, person) |
| Source | cocodataset.org | official KITTI download (left color images + labels) |

### Taxonomy mapping (KITTI → COCO)

KITTI has 8 classes plus `DontCare`. Only two map 1:1 to COCO: Car → car and Pedestrian → person.

Other KITTI objects can still *look* like a car or person to a COCO model. A van gets predicted as "car", and a cyclist's rider as "person". If we deleted those boxes, these reasonable predictions would count as false positives. So we keep them as **ignore regions** (`iscrowd=1` in COCO format). When scoring, pycocotools skips any prediction that lands on an ignore region, and never counts an ignore region as a missed object. Only real cars and pedestrians affect the score. KITTI's own benchmark treats vans and DontCare areas the same way.

| KITTI class | Boxes in subset | Mapped to | Reason |
|---|---|---|---|
| Car | 3,907 | **car** (COCO id 3) | 1:1 |
| Pedestrian | 534 | **person** (COCO id 1) | 1:1 |
| Van | 354 | ignore | gets detected as car |
| Truck | 147 | ignore | gets detected as car/truck |
| Misc | 122 | ignore | trailers, etc. |
| Cyclist | 214 | ignore | KITTI uses one box for rider + bike; COCO has separate person and bicycle boxes |
| Person_sitting | 34 | ignore | gets detected as person |
| DontCare | 1,500 | ignore | regions the annotators left unlabeled |
| Tram | — | dropped | no car/person confusion |

The conversion is done by `convert_kitti.py`. We drew the ground truth on 5 images to check it (`results/figs/kitti_gt_*.png`).

## 4. Preprocessing (transformations)

| | Ultralytics (YOLO26-L, RT-DETR-L) | torchvision (Faster R-CNN v2) |
|---|---|---|
| Input | BGR uint8 (cv2) | BGR → RGB, float in [0, 1] |
| Resize | letterbox, long side 640, aspect ratio kept, pad to a multiple of the stride | short side 800, long side ≤ 1333 |
| COCO image (640×480) | 640×480 | 1067×800 |
| KITTI image (1242×375) | **640×193** (0.52×) | **1333×403** (1.07×) |
| Normalization | ÷255 only | ImageNet mean/std (inside `GeneralizedRCNNTransform`) |
| Output boxes | rescaled to original image pixels | rescaled to original image pixels |
| Class ids | 0–79 → COCO 91-ids via `coco80_to_coco91_class` | already COCO 91-ids |

Faster R-CNN sees KITTI at about **2× the pixel resolution** of the Ultralytics models. We test this effect in §8.

## 5. Inference and evaluation settings

| Setting | Value |
|---|---|
| Confidence threshold | 0.001, for all models (standard for mAP) |
| NMS | YOLO26-L: none · YOLO26-L +NMS: IoU 0.7 · RT-DETR-L: none · Faster R-CNN: IoU 0.5 (torchvision default) |
| Max detections | 300 (Ultralytics), 100 (Faster R-CNN); COCOeval keeps the top 100 per image |
| Precision / batch | fp32, batch 1 |
| KITTI predictions | only car and person kept |
| Metric | pycocotools `COCOeval` bbox: 12 standard stats + per-class AP |
| Latency | end-to-end `predict()` (preprocess + forward + postprocess/NMS), `cuda.synchronize()`, first 20 images excluded as warm-up |
| GPU memory | `torch.cuda.max_memory_allocated()` |

## 6. Setup

| | |
|---|---|
| GPU | NVIDIA RTX 5080 16 GB, driver 595.99.02, CUDA 12.8 |
| OS | Linux 6.8.0 (x86_64) |
| Software | Python 3.11.17, torch 2.11.0+cu128, torchvision 0.26.0, ultralytics 8.4.194, pycocotools (conda env `cnn`) |

```bash
python convert_kitti.py --n 1000 --seed 0 --viz 5
python eval.py --model {yolo26l|yolo26l-nms|rtdetr-l|frcnn} --dataset {coco|kitti}
python eval.py --model {yolo26l|rtdetr-l} --dataset kitti --imgsz 1280
python make_report_assets.py
```

Code layout:
- `detectors/`: one wrapper per model behind a common `predict(img) → boxes, scores, COCO ids` interface
- `eval.py`: inference, scoring and timing
- `notebooks/level1_baseline.ipynb`: all analyses, with outputs saved

Each run writes its settings and exact command into `results/<model>_<dataset>.json`. The full command log is in `logs/commands.md`, and the pinned environment is in `requirements-lock.txt`.

## 7. Results

### COCO val2017 (80 classes), ×100

| Model | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR100 | Published AP |
|---|---|---|---|---|---|---|---|---|
| YOLO26-L | 53.8 | 70.7 | 58.8 | 37.2 | 58.0 | 69.5 | 73.1 | ≈55 |
| YOLO26-L +NMS | **54.0** | 71.0 | **58.9** | 35.9 | **58.3** | 69.8 | 69.6 | ≈55 |
| RT-DETR-L | 52.8 | **71.3** | 57.2 | 34.4 | 57.4 | **70.8** | 71.7 | 53.0 |
| Faster R-CNN v2 | 46.9 | 67.9 | 51.3 | 31.0 | 50.8 | 60.3 | 63.0 | 46.7 |

These match the published numbers. YOLO26-L is about 1 point lower because the docs figure comes from `yolo val`, which batches images with different padding; we run `predict()` per image, the same way for every model.

### KITTI subset (car + person), ×100

| Model | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR100 | car | person |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26-L | 39.6 | 72.9 | 38.4 | 23.6 | 37.9 | 55.5 | 51.3 | 49.5 | 29.6 |
| YOLO26-L +NMS | 39.8 | 72.7 | **38.9** | 23.8 | 37.8 | 55.6 | 49.1 | 49.6 | 30.1 |
| RT-DETR-L | 40.1 | 72.5 | 38.6 | 25.1 | 37.3 | **57.2** | 50.4 | **51.3** | 29.0 |
| Faster R-CNN v2 | **40.4** | **74.6** | 38.3 | **25.7** | **38.0** | 55.4 | **51.5** | 49.6 | **31.3** |

## 8. Analysis 1: Cross-dataset generalization

To compare like with like, we re-scored the COCO predictions on **car + person only** ("COCO-2").

| Model | COCO-2 AP | KITTI AP | Δ | car COCO → KITTI | person COCO → KITTI |
|---|---|---|---|---|---|
| YOLO26-L | 59.4 | 39.6 | −19.8 | 53.7 → 49.5 | 65.0 → 29.6 |
| YOLO26-L +NMS | 59.4 | 39.8 | −19.6 | 53.6 → 49.6 | 65.2 → 30.1 |
| RT-DETR-L | 55.9 | 40.1 | −15.8 | 50.1 → 51.3 | 61.7 → 29.0 |
| Faster R-CNN v2 | 54.8 | 40.4 | **−14.4** | 49.9 → 49.6 | 59.6 → 31.3 |

![AP change by object size](../results/figs/size_drop.png)

- **Class:** cars transfer almost perfectly, but **person AP halves** for every model. Object size doesn't explain this: median person height is about the same on both datasets (KITTI 87 px, COCO 83 px).
- **Scale:** medium objects drop most (−27 to −34 AP), then large, then small.
- **Ranking flips:** YOLO26-L leads on COCO but drops the most. Faster R-CNN drops the least, and on KITTI all models are within 1 AP of each other.

**KITTI difficulty and occlusion (AP):**

| Model | Easy | Moderate | Hard | fully visible | partly occluded | largely occluded |
|---|---|---|---|---|---|---|
| YOLO26-L | 54.3 | 47.4 | 40.0 | 51.1 | 22.9 | 5.4 |
| RT-DETR-L | 55.0 | 47.8 | 40.3 | 51.8 | 24.9 | 6.3 |
| Faster R-CNN v2 | 54.9 | 48.2 | 40.6 | 51.9 | 23.4 | 5.7 |

Easy, Moderate and Hard follow KITTI's official definitions (minimum height, maximum occlusion, maximum truncation). **Occlusion is the dominant factor:** AP falls from about 52 to about 6.

**Is it input resolution?** We re-ran the Ultralytics models at `imgsz 1280` (1280×386):

| Model | Input | AP | AP_S | person | ms |
|---|---|---|---|---|---|
| YOLO26-L | 640×193 → 1280×386 | 39.6 → 39.9 | 23.6 → 27.5 | 29.6 → 29.1 | 5.5 → 7.0 |
| RT-DETR-L | 640×193 → 1280×386 | 40.1 → 41.0 | 25.1 → 28.0 | 29.0 → 30.9 | 13.6 → 25.4 |

Higher resolution recovers small-object AP (+3–4) but adds less than 1 AP overall and doesn't fix person AP. **The person drop is real domain shift** (occlusion, groups, the vehicle's viewpoint), not resolution.

## 9. Analysis 2: Speed vs. accuracy

| Model | Params (M) | COCO AP | KITTI AP | COCO ms | COCO FPS | KITTI ms | Peak mem COCO / KITTI (MB) |
|---|---|---|---|---|---|---|---|
| YOLO26-L | 26.3 | 53.8 | 39.6 | **5.6** | **178** | **5.5** | **221 / 163** |
| YOLO26-L +NMS | 26.3 | 54.0 | 39.8 | 5.7 | 177 | 5.4 | 221 / 163 |
| RT-DETR-L | 33.0 | 52.8 | 40.1 | 13.6 | 74 | 13.6 | 239 / 239 |
| Faster R-CNN v2 | 43.7 | 46.9 | 40.4 | 24.5 | 41 | 19.1 | 671 / 441 |

![Speed vs accuracy](../results/figs/speed_vs_ap.png)

- **YOLO26-L is fastest and (on COCO) most accurate.** It is 4.4× faster than Faster R-CNN with +7 AP and uses 3× less memory.
- RT-DETR-L takes 2.4× YOLO26's latency at the same input size.
- On KITTI, YOLO26-L is 3.5× faster than Faster R-CNN for −0.8 AP, which makes it the **best speed/accuracy choice for driving**.

## 10. Analysis 3: NMS vs. NMS-free

| Model | Post-proc | COCO AP | COCO AR100 | Duplicate pairs on COCO (same class, IoU > 0.7, score ≥ 0.3) |
|---|---|---|---|---|
| YOLO26-L | NMS-free | 53.8 | 73.1 | 554 |
| YOLO26-L +NMS | NMS | 54.0 | 69.6 | 25 |
| RT-DETR-L | NMS-free | 52.8 | 71.7 | 1,813 |
| Faster R-CNN v2 | NMS | 46.9 | 63.0 | 0 |

- With the same YOLO26 weights, removing NMS costs only 0.2 AP at the same speed, and it raises AR100.
- **NMS-free still produces some duplicate boxes:** about 2% of YOLO26's detections and about 3.5% of RT-DETR's. One-to-one and Hungarian training *discourage* duplicates, but nothing removes them at inference. This barely affects mAP, but it matters for counting and tracking.

## 11. Failure analysis

![Failure example](../results/figs/fail_2_001457.png)

*Green = detected GT, yellow = missed GT, red = prediction, gray = ignore region. Six examples are in `results/figs/fail_*.png` and the notebook §10.*

Missed objects (score ≥ 0.3, IoU ≥ 0.5):

| Model | Missed | % largely occluded | % small (< 40 px) | % pedestrian |
|---|---|---|---|---|
| YOLO26-L | 849 | 59 | 47 | 22 |
| RT-DETR-L | 642 | 64 | 41 | 27 |
| Faster R-CNN v2 | 529 | 70 | 35 | 24 |

1. **Heavily occluded parked cars** in rows along the curb make up most misses. COCO rarely has rows of overlapping, side-on cars.
2. **Small, distant objects:** below 25 px tall, recall is about 72% for YOLO26-L, 86% for RT-DETR-L and 92% for Faster R-CNN, because of the input resolution difference.
3. **Pedestrians** are about 12% of scored ground truth but 22–27% of misses.
4. Detections of unlabeled parked cars inside DontCare regions don't count against the models, because those regions are ignored. Without ignore regions they would count as false positives.

## 12. Conclusions

- **COCO rankings don't carry over to KITTI.** On COCO the order is YOLO26-L > RT-DETR-L > Faster R-CNN, but on zero-shot KITTI all three are within about 1 AP.
- **Domain shift hits pedestrians and occluded objects**, not cars. Person AP halves, and AP on largely occluded objects is about 6.
- **Resolution only helps small objects** (+3–4 AP_S at 1280).
- **NMS-free costs no accuracy** but leaves 2–3.5% duplicate boxes.
- **YOLO26-L gives the best speed vs. accuracy** (5.5 ms, ~180 FPS, within 1 AP of the best on KITTI).

## 13. Limitations

- KITTI is a 1,000-image subset with a single seed; it comes from `training/` because test labels are private.
- Only 2 KITTI classes are scored, and Cyclist is excluded because the labels don't match COCO.
- Latency includes pre- and post-processing in Python, measured in fp32 with no TensorRT, so absolute FPS is lower than optimized deployment numbers.
- There was no fine-tuning; every result is zero-shot.

## 14. References

1. Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NeurIPS 2015.
2. Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017.
3. Li et al., *Benchmarking Detection Transfer Learning with Vision Transformers*, arXiv:2111.11429, 2021 (basis of the Faster R-CNN v2 recipe).
4. Carion et al., *End-to-End Object Detection with Transformers (DETR)*, ECCV 2020.
5. Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)*, CVPR 2024.
6. Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, NeurIPS 2024 (NMS-free dual assignment).
7. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
8. Geiger et al., *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012.
9. Ultralytics docs and code (YOLO26, RT-DETR): https://docs.ultralytics.com, https://github.com/ultralytics/ultralytics
10. torchvision detection models: https://pytorch.org/vision/stable/models.html
11. COCO API / pycocotools: https://github.com/cocodataset/cocoapi
12. Course tutorial `ngdet`: https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md
