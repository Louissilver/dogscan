#!/usr/bin/env python3
"""Auto-rotulagem (bounding box) das imagens baixadas usando YOLO-World (prompt "spider").

Gera um .txt YOLO ao lado de cada imagem: "<class_id> xc yc w h" (normalizado).
Imagens sem deteccao vao para review_needed.txt -> corrigir a mao no Roboflow/CVAT.

ATENCAO: rotulo automatico e' rascunho. Revisao humana de pelo menos uma amostra
grande e' obrigatoria antes de confiar no modelo.

Uso:
    python data/autolabel.py --raw data/raw --config data/classes.yaml --conf 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--config", default="data/classes.yaml")
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-boxes", type=int, default=2, help="max deteccoes mantidas por imagem")
    args = ap.parse_args()

    from ultralytics import YOLOWorld  # import tardio (puxa torch)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    name_to_id = {c["name"]: c["id"] for c in cfg["classes"]}

    model = YOLOWorld(args.model)
    model.set_classes(["spider", "tarantula"])  # ambos mapeiam para a aranha na imagem

    raw = Path(args.raw)
    review, done, empty = [], 0, 0

    for cls_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        cid = name_to_id.get(cls_dir.name)
        if cid is None:
            continue
        imgs = [p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
        print(f"[{cls_dir.name}] {len(imgs)} imagens -> classe {cid}")
        for img in imgs:
            res = model.predict(img, conf=args.conf, iou=args.iou,
                                imgsz=args.imgsz, verbose=False)[0]
            boxes = res.boxes
            lines = []
            if boxes is not None and len(boxes):
                order = boxes.conf.argsort(descending=True)[: args.max_boxes]
                for b in boxes.xywhn[order]:
                    xc, yc, w, h = (float(v) for v in b)
                    lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            txt = img.with_suffix(".txt")
            if lines:
                txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
                done += 1
            else:
                txt.write_text("", encoding="utf-8")  # placeholder = "sem objeto"
                review.append(str(img))
                empty += 1

    Path(args.raw, "review_needed.txt").write_text("\n".join(review), encoding="utf-8")
    print(f"\nrotuladas: {done}   sem deteccao (revisar): {empty}")
    print(f"lista de revisao -> {Path(args.raw, 'review_needed.txt')}")


if __name__ == "__main__":
    main()
