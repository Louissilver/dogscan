#!/usr/bin/env python3
"""Exporta o modelo treinado para TFLite int8 (Android) e opcionalmente outros formatos.

ATENCAO: export TFLite/LiteRT so' funciona em Linux x86 ou macOS. No Windows a
Ultralytics aborta com "LiteRT export only supported on Linux x86 and macOS".
Rode este export no Colab (ver training/colab_dogs.md).

Uso:
    python training/export.py --weights runs/detect/train/weights/best.pt
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/spider_v1/weights/best.pt")
    ap.add_argument("--data", default="training/data.yaml", help="usado p/ calibrar int8")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--formats", nargs="+", default=["tflite"],
                    help="tflite onnx coreml ...")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    for fmt in args.formats:
        kw = dict(format=fmt, imgsz=args.imgsz)
        if fmt == "tflite":
            kw.update(int8=True, data=args.data)   # quantizacao int8 p/ celular
        if fmt == "coreml":
            kw.update(nms=True)
        path = model.export(**kw)
        print(f"{fmt} -> {path}")


if __name__ == "__main__":
    main()
