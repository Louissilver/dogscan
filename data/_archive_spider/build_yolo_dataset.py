#!/usr/bin/env python3
"""Monta o dataset YOLO a partir de data/raw/<classe>/*.jpg + *.txt (rotulos).

Saida:
    data/dataset/images/{train,val}/...
    data/dataset/labels/{train,val}/...

Uso:
    python data/build_yolo_dataset.py --raw data/raw --out data/dataset --val 0.15
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-empty", action="store_true",
                    help="incluir imagens sem box (background). Default: descarta.")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    random.seed(args.seed)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    pairs = []  # (img, label)
    for cls_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        for img in cls_dir.rglob("*"):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lbl = img.with_suffix(".txt")
            if not lbl.exists():
                continue
            if lbl.stat().st_size == 0 and not args.keep_empty:
                continue
            pairs.append((img, lbl, cls_dir.name))

    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val)
    counts = {"train": {}, "val": {}}

    for i, (img, lbl, cname) in enumerate(pairs):
        split = "val" if i < n_val else "train"
        stem = f"{cname}_{img.stem}"
        shutil.copy(img, out / "images" / split / f"{stem}{img.suffix.lower()}")
        shutil.copy(lbl, out / "labels" / split / f"{stem}.txt")
        counts[split][cname] = counts[split].get(cname, 0) + 1

    print(f"total pares: {len(pairs)}  ->  train {len(pairs)-n_val} | val {n_val}")
    for split in ("train", "val"):
        print(f"  {split}:", {k: counts[split].get(k, 0) for k in sorted(counts[split])})
    print(f"dataset -> {out}")


if __name__ == "__main__":
    main()
