#!/usr/bin/env python3
"""Prepara dataset YOLO de deteccao de racas a partir do Stanford Dogs Dataset.

Stanford Dogs traz 1 bounding box (VOC XML) por imagem -> converte p/ YOLO,
filtra o subconjunto de racas em BREEDS, faz split train/val e escreve
training/data.yaml.

Entrada (extrair os .tar baixados):
    <src>/Images/n0XXXX-breed/*.jpg
    <src>/Annotation/n0XXXX-breed/<img_basename>        (XML, sem extensao)

Uso:
    python data/dogs_prepare.py --src D:/Projetos/toolchain/stanford --out data/dataset_dogs
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# (substring da pasta Stanford  ->  nome da classe YOLO). Ordem = id da classe.
BREEDS: list[tuple[str, str]] = [
    ("Labrador_retriever", "labrador"),
    ("golden_retriever", "golden_retriever"),
    ("German_shepherd", "pastor_alemao"),
    ("French_bulldog", "bulldog_frances"),
    ("boxer", "boxer"),
    ("beagle", "beagle"),
    ("Rottweiler", "rottweiler"),
    ("pug", "pug"),
    ("Chihuahua", "chihuahua"),
    ("Siberian_husky", "husky"),
    ("Pomeranian", "spitz_pomeranian"),
    ("Yorkshire_terrier", "yorkshire"),
    ("Doberman", "doberman"),
    ("Border_collie", "border_collie"),
    ("Shih-Tzu", "shih_tzu"),
    ("miniature_pinscher", "pinscher"),
]

IMG_EXT = {".jpg", ".jpeg", ".png"}


def find_breed_dir(base: Path, needle: str) -> Path | None:
    hits = [p for p in base.iterdir() if p.is_dir() and needle.lower() in p.name.lower()]
    if not hits:
        return None
    hits.sort(key=lambda p: len(p.name))  # match mais especifico primeiro
    return hits[0]


def voc_to_yolo(xml_path: Path, class_id: int) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width")))
    h = int(float(size.findtext("height")))
    if w <= 0 or h <= 0:
        return []
    out = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = max(0.0, float(bb.findtext("xmin")))
        ymin = max(0.0, float(bb.findtext("ymin")))
        xmax = min(float(w), float(bb.findtext("xmax")))
        ymax = min(float(h), float(bb.findtext("ymax")))
        if xmax <= xmin or ymax <= ymin:
            continue
        xc = (xmin + xmax) / 2 / w
        yc = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        out.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="pasta com Images/ e Annotation/")
    ap.add_argument("--out", default="data/dataset_dogs")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0, help="max imgs por raca (0 = todas)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    img_base = src / "Images"
    ann_base = src / "Annotation"
    if not img_base.is_dir() or not ann_base.is_dir():
        raise SystemExit(f"esperado {img_base} e {ann_base}. Extraia os .tar primeiro.")

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    pairs = []  # (img_path, [yolo_lines], class_name)
    for cid, (needle, cname) in enumerate(BREEDS):
        d_img = find_breed_dir(img_base, needle)
        d_ann = find_breed_dir(ann_base, needle)
        if not d_img or not d_ann:
            print(f"  [AVISO] raca nao encontrada no dataset: {needle}")
            continue
        imgs = sorted(p for p in d_img.iterdir() if p.suffix.lower() in IMG_EXT)
        if args.limit:
            imgs = imgs[: args.limit]
        n = 0
        for img in imgs:
            xml = d_ann / img.stem
            if not xml.exists():
                continue
            try:
                lines = voc_to_yolo(xml, cid)
            except ET.ParseError:
                continue
            if lines:
                pairs.append((img, lines, cname))
                n += 1
        print(f"  {cid:2d} {cname:<18} {n:4d} imgs   ({d_img.name})")

    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val)
    counts = {"train": {}, "val": {}}
    for i, (img, lines, cname) in enumerate(pairs):
        split = "val" if i < n_val else "train"
        stem = f"{cname}_{img.stem}"
        shutil.copy(img, out / "images" / split / f"{stem}.jpg")
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        counts[split][cname] = counts[split].get(cname, 0) + 1

    names = {i: c for i, (_, c) in enumerate(BREEDS)}
    yaml_txt = (
        f"# gerado por data/dogs_prepare.py\n"
        f"path: ../{out.as_posix()}\n"
        f"train: images/train\nval: images/val\n\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in names.items())
    )
    Path("training/data.yaml").write_text(yaml_txt, encoding="utf-8")

    print(f"\ntotal {len(pairs)} pares  ->  train {len(pairs)-n_val} | val {n_val}")
    print(f"dataset -> {out}")
    print("training/data.yaml atualizado")


if __name__ == "__main__":
    main()
