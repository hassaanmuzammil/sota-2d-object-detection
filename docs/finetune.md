# Fine-tuning YOLO26-L on KITTI

The baseline and extended studies evaluated COCO-pretrained detectors on KITTI **without retraining**. This study asks: **how much of that COCO → KITTI gap does domain-specific training close?** COCO-pretrained YOLO26-L is fine-tuned on KITTI and compared with the same model zero-shot, on the same held-out test images.

## 1. Result in one table

KITTI test split (1,559 images from 20 drives never seen in training), COCOeval, car + person, ×100.

| | mAP | AP50 | AP75 | car | person | cyclist* |
|---|---|---|---|---|---|---|
| Zero-shot YOLO26-L @640 | 36.6 | 71.3 | 33.5 | 46.8 | 26.4 | 0.4 |
| Fine-tuned YOLO26-L @640 | **47.6** | **74.8** | **51.5** | **65.6** | **29.7** | **20.0** |
| Change | **+11.0** | +3.5 | +18.0 | +18.8 | +3.3 | +19.6 |

\* Cyclist comes from the cyclist variant of the ground truth (§4); the other columns use the standard car + person ground truth.

**Ten minutes of fine-tuning on 5,209 KITTI images raises mAP by 11 points (+30% relative), with no change in speed.**

![zero-shot vs fine-tuned](../results/figs/finetune_main.png)

## 2. Model

| | |
|---|---|
| Model | YOLO26-L (one-stage, anchor-free, NMS-free one-to-one head) |
| Starting checkpoint | [`yolo26l.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt), COCO-pretrained (same as baseline and extended studies) |
| Params | 26.3M zero-shot (80-class head), 24.7M fused after fine-tuning (3-class head) |
| Transferred weights | all 1,092 tensors; the classification layer is resized from 80 to 3 classes (`nc=80 → 3`) |
| Fine-tuned classes | 0 person, 1 cyclist, 2 car |
| Framework | Ultralytics 8.4.194, torch 2.11.0+cu128, Python 3.11.17 (conda env `cnn`) |
| Hardware | NVIDIA RTX 5080 16 GB, CUDA 12.8 |

## 3. Data split

### Why a new split

KITTI's 7,481 training images are frames sampled from **141 recording drives**, and neighbouring frames of one drive are nearly identical. The baseline and extended studies used 1,000 random images, which come from 95 of the 141 drives. Only 206 of the remaining 6,481 images are from other drives, so training on "the other images" would leak near-duplicate frames into evaluation.

The fine-tuning study therefore splits **by drive**: every image of a drive goes to the same split. Each image's drive comes from the official KITTI devkit mapping (`mapping/train_mapping.txt` + `train_rand.txt` in [devkit_object.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip)). Drives are shuffled with seed 0 and assigned to test until it holds ≥20% of images, then to val until ≥30%, and the rest go to train.

The old 1,000-image evaluation subset cannot be reused: 687 of its images are now in train, 96 in val and 217 in test.

### Split sizes

| Split | Drives | Images | Car | Pedestrian | Cyclist | Van | Truck | DontCare | Use |
|---|---|---|---|---|---|---|---|---|---|
| train | 112 | 5,209 | 17,876 | 3,267 | 1,296 | 2,035 | 756 | 7,955 | training |
| val | 9 | 713 | 3,196 | 40 | 20 | 398 | 104 | 795 | choosing the best epoch |
| test | 20 | 1,559 | 7,670 | 1,180 | 311 | 481 | 234 | 2,545 | all reported numbers |

No drive appears in more than one split (checked by an assert in `finetune/make_split.py`). The full drive lists are in `data/kitti_finetune/split.json`.

Test drives: 2011_09_26 drives 0009, 0014, 0027, 0036, 0056, 0059, 0117; 2011_09_28 drives 0002, 0016, 0021, 0039, 0117, 0143, 0162, 0166, 0179, 0183, 0185, 0186, 0208.

## 4. Labels

### Training labels (YOLO format)

| KITTI class | Training label |
|---|---|
| Pedestrian | person (0) |
| Cyclist | cyclist (1) |
| Car | car (2) |
| DontCare | region **grayed out** in the training image (RGB 114, Ultralytics' padding color) |
| Van, Truck, Misc, Person_sitting, Tram | not labeled (background) |

- **DontCare is grayed out** because it marks real but unlabeled objects (mostly distant cars). YOLO training has no "ignore" option, so leaving them would teach the model those cars are background.
- **Van / Truck / Person_sitting are left unlabeled** to match the evaluation taxonomy, where they are ignore regions: predictions on them are never penalized, so the model gains nothing by detecting them as car or person.

### Evaluation ground truth

Exactly the taxonomy of the baseline and extended studies, applied to the new test split:

| KITTI | Standard GT (`test_coco.json`) | Cyclist variant (`test_coco_cyclist.json`) |
|---|---|---|
| Car | car | car |
| Pedestrian | person | person |
| Cyclist | ignore | **cyclist (scored)** |
| Van, Truck, Misc, Person_sitting, DontCare | ignore | ignore |
| Tram | dropped | dropped |

"Ignore" = COCO crowd region (`iscrowd=1`): pycocotools neither counts a prediction on it as a false positive nor counts it as missed. Evaluation uses the **original images** (DontCare not grayed out).

Class mapping at evaluation: the zero-shot model's 80 COCO classes map through `coco80_to_coco91_class` (person = 1, bicycle = 2 is used as cyclist, car = 3). The fine-tuned model's person / cyclist / car map to 1 / 2 / 3.

## 5. Training settings

Ultralytics defaults except where noted (full list in `results/finetune/train/ft640/args.yaml`).

| Setting | Value |
|---|---|
| Epochs | **15** (chosen to fit the time budget) |
| Image size | 640 (letterbox, square training mosaics) |
| Batch | 16 |
| Optimizer | `auto` → AdamW, lr 0.001429, momentum 0.9 |
| LR schedule | linear decay to lr × 0.01 (`lrf=0.01`), 3 warmup epochs |
| Weight decay | 0.0005 |
| Loss weights | box 7.5, cls 0.5 |
| Augmentation | mosaic 1.0 (switched off for the last 10 epochs, `close_mosaic=10`), HSV jitter (0.015 / 0.7 / 0.4), translate 0.1, scale 0.5, horizontal flip 0.5, RandAugment, random erasing 0.4; no mixup, cutmix or copy-paste |
| Precision | AMP (mixed fp16) |
| Seed | 0, deterministic |
| Checkpoint used | `best.pt` = highest val mAP50-95 (3 classes, Ultralytics' evaluator) → **epoch 13** (val 64.3) |
| Training time | 0.17 h (10.5 min), GPU memory ≈ 11 GB |

![validation curve](../results/figs/finetune_curves.png)

Val mAP is noisy from epoch to epoch (48–64). The val split has only 40 pedestrians and 20 cyclists, so checkpoint selection is driven almost entirely by cars.

## 6. Evaluation settings

Identical to the baseline and extended studies, so the numbers are directly comparable in method:

| Setting | Value |
|---|---|
| Head | NMS-free one-to-one head (`nms=False`), both models |
| Input | letterbox, long side 640 (KITTI 1242×375 → 640×193) |
| Confidence threshold | 0.001 |
| Max detections | 300 (COCOeval keeps top 100) |
| Precision / batch | fp32, batch 1 |
| Metrics | pycocotools COCOeval, 12 standard stats + per-class AP |
| Latency | end-to-end `predict()` including pre- and post-processing, CUDA-synchronized, first 20 images excluded |
| GPU memory | `torch.cuda.max_memory_allocated()` |

The zero-shot model scores 36.6 here vs. 39.6 on the extended study's 1,000-image subset. Those are different images, so compare only within this table.

## 7. Results

All ×100, KITTI test split (1,559 images).

### 7.1 Main metrics (car + person)

| Run | mAP | AP50 | AP75 | AP small / medium / large | AR100 | car | person |
|---|---|---|---|---|---|---|---|
| Zero-shot @640 | 36.6 | 71.3 | 33.5 | 24.0 / 36.5 / 47.5 | 49.8 | 46.8 | 26.4 |
| Fine-tuned @640 | **47.6** | **74.8** | **51.5** | **25.2 / 48.7 / 63.3** | **58.3** | **65.6** | **29.7** |

### 7.2 Cyclist variant (car + person + cyclist)

| Run | mAP | car | person | cyclist |
|---|---|---|---|---|
| Zero-shot @640 (cyclist ← COCO bicycle) | 23.8 | 46.8 | 24.3 | 0.4 |
| Fine-tuned @640 | **38.3** | **65.6** | **29.5** | **20.0** |

### 7.3 By KITTI occlusion level (car + person)

Only boxes of one occlusion level are scored; the others become ignore regions.

| Run | fully visible (3,506 boxes) | partly occluded (3,310) | largely occluded (1,846) |
|---|---|---|---|
| Zero-shot @640 | 53.5 | 31.5 | 8.1 |
| Fine-tuned @640 | **57.9** | **43.2** | **23.3** |

![AP by occlusion](../results/figs/finetune_occlusion.png)

### 7.4 Speed and memory

| Run | Latency | FPS | Peak GPU memory |
|---|---|---|---|
| Zero-shot @640 | 5.5 ms | 183 | 163 MB |
| Fine-tuned @640 | 5.1 ms | 197 | 162 MB |

Same architecture and input size, so speed is unchanged; the small difference is run-to-run noise plus the smaller 3-class head.

### 7.5 Examples

Test images where fine-tuning finds the most extra objects (IoU ≥ 0.5, score ≥ 0.3):

![examples](../results/figs/finetune_examples.png)

## 8. Analysis

- **Most of the gain is box precision, not finding more objects.** AP50 rises only 3.5 points but AP75 rises 18. The zero-shot model already finds most cars and people; fine-tuning teaches it KITTI's box conventions (how tightly cars are boxed, how partly hidden objects are boxed).
- **Cars gain most (+18.8), pedestrians least (+3.3).** Cars are 80% of training boxes. Pedestrians stay hard: 3,267 training examples and many small, distant people in the test drives.
- **Occlusion is where domain data helps most.** Largely occluded objects go from 8.1 to 23.3 (about 3×), partly occluded from 31.5 to 43.2, fully visible only +4.4. Occluded parked cars were the biggest failure in the extended study; KITTI training shows the model what a car half-hidden behind another car looks like and how KITTI boxes it.
- **Small objects barely improve (+1.2).** At 640, a KITTI frame shrinks to 640×193, so distant objects are only a few pixels tall. That is a resolution limit, not a domain one; a 1280-pixel input would address it.
- **A new class is learned from scratch.** Cyclist goes from 0.4 (COCO has no cyclist; "bicycle" boxes cover only the bike) to 20.0 from 1,296 training boxes. Text prompting in the extended study (YOLO-World "cyclist") reached only 0.77.
- **Answer to the study question:** a large part of the COCO → KITTI gap is the domain, not the model. The same network, at the same speed, gains 11 mAP from ten minutes of in-domain training.

## 9. Limitations

- One run, one seed, 15 epochs (time budget). Longer training or 1280-pixel input would likely score higher; neither was run.
- The val split has few pedestrians (40) and cyclists (20), so the best epoch was chosen mostly on cars.
- The test split is 20 drives from two dates; a different drive split will move the numbers by a few points.
- Van / Truck / Person_sitting are not trained or scored, the same as in the earlier studies.
- Graying out DontCare changes training images slightly; evaluation uses the original images.

## 10. Reproduce

Run from the repo root in the `cnn` conda environment. Needs KITTI `data/kitti/training/{image_2,label_2}` and `weights/yolo26l.pt` (see README).

```bash
conda activate cnn

# 1. drive-disjoint split + YOLO dataset + COCO GT (downloads the devkit mapping; ~2 min, 4.5 GB)
python finetune/make_split.py --test-frac 0.2 --val-frac 0.1 --seed 0

# 2. fine-tune (~10 min on an RTX 5080)
python finetune/train.py --name ft640 --imgsz 640 --batch 16 --epochs 15

# 3. evaluate zero-shot and fine-tuned on the test split (standard and cyclist GT)
for gt in test test-cyclist; do
  python finetune/evaluate.py --tag zs640 --gt $gt
  python finetune/evaluate.py --tag ft640 --weights results/finetune/train/ft640/weights/best.pt --gt $gt
done

# 4. tables and figures
python finetune/analyze.py
```

## 11. Files

| Path | Contents |
|---|---|
| `finetune/make_split.py` | drive-disjoint split, YOLO labels (DontCare grayed out), COCO-format GT |
| `finetune/train.py` | fine-tuning wrapper around `YOLO.train()`; writes `train_info.json` |
| `finetune/evaluate.py` | COCOeval + latency + memory, same procedure as `eval.py` |
| `finetune/analyze.py` | tables, occlusion breakdown, figures |
| `data/kitti_finetune/split.json` | drives and image ids per split |
| `data/kitti_finetune/{val,test}_coco[_cyclist].json` | evaluation ground truth |
| `data/kitti_finetune/kitti.yaml` | Ultralytics dataset file |
| `results/finetune/train/ft640/` | `args.yaml`, `results.csv` (per-epoch losses and val mAP), `weights/{best,last}.pt`, training plots |
| `results/finetune/{zs640,ft640}_{test,test-cyclist}.json` | all metrics, per-class AP, speed, memory, exact command |
| `results/finetune/preds/` | raw predictions (COCO results format) |
| `results/finetune/tables.md` | all tables from `analyze.py` |
| `results/figs/finetune_{main,occlusion,curves,examples}.png` | figures |
| `logs/finetune_*.log` | training, evaluation and analysis logs |
