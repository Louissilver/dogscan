#!/usr/bin/env python3
"""Treino do detector de aranhas (YOLO11n). Rode preferencialmente com GPU (Colab/Kaggle).

Uso:
    python training/train.py --epochs 120 --imgsz 640 --batch 16
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/data.yaml")
    ap.add_argument("--weights", default="yolo11n.pt", help="pesos base (COCO)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="spider_v1")
    ap.add_argument("--device", default=None, help="0 / cpu / 0,1")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project="runs",
        name=args.name,
        device=args.device,
        # augmentation util p/ fotos de campo variadas:
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, close_mosaic=10,
    )
    metrics = model.val()
    print("mAP50-95:", metrics.box.map, " mAP50:", metrics.box.map50)


if __name__ == "__main__":
    main()
