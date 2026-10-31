"""Fine-tuning study: fine-tune COCO-pretrained YOLO26-L on the drive-disjoint KITTI train split (finetune/make_split.py).

python finetune/train.py --name ft640 --imgsz 640 --batch 16 --epochs 15
python finetune/train.py --name ft640_f10 --imgsz 640 --batch 16 --epochs 15 --fraction 0.1
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

PROJECT = Path("results/finetune/train").resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default="weights/yolo26l.pt")
    ap.add_argument("--data", default="data/kitti_finetune/kitti.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the train split to use")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    model = YOLO(args.weights)
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, fraction=args.fraction,
                seed=args.seed, workers=args.workers, project=str(PROJECT), name=args.name, exist_ok=True,
                plots=True, verbose=False)
    run = PROJECT / args.name
    (run / "train_info.json").write_text(json.dumps({
        **vars(args), "train_hours": (time.time() - t0) / 3600, "best": str(run / "weights" / "best.pt"),
        "command": "python finetune/train.py " + " ".join(f"--{k} {v}" for k, v in vars(args).items()),
    }, indent=2))


if __name__ == "__main__":
    main()
