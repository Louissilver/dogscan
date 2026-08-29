#!/usr/bin/env python3
"""Baixa imagens de aranhas do GBIF (padrao: iNaturalist research-grade) para treino.

Uso:
    python data/gbif_download.py --config data/classes.yaml --out data/raw --workers 8

Saida:
    data/raw/<classe>/<occKey>_<n>.jpg
    data/raw/manifest.csv   -> rastreio de licenca/origem. NAO descartar.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml
from PIL import Image, UnidentifiedImageError

GBIF = "https://api.gbif.org/v1"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "spider-id-dataset-builder/0.1 (pesquisa)"})

_lock = threading.Lock()
_seen_hashes: set[str] = set()
_manifest_rows: list[dict] = []
MAX_SIDE = 1280
MIN_SIDE = 200


def log(*a):
    with _lock:
        print(*a, flush=True)


def match_taxon(name: str, rank: str) -> int:
    r = SESSION.get(f"{GBIF}/species/match", params={"name": name, "rank": rank}, timeout=30)
    r.raise_for_status()
    d = r.json()
    if not d.get("usageKey") or d.get("matchType") == "NONE":
        raise SystemExit(f"GBIF nao encontrou taxon: {name} ({rank}) -> {d}")
    log(f"  {name:<16} {rank:<7} -> taxonKey {d['usageKey']}  [{d.get('scientificName')}]")
    return d["usageKey"]


def iter_occurrences(taxon_key: int, g: dict, hard_limit: int):
    got, offset, page = 0, 0, int(g.get("page_size", 300))
    while got < hard_limit:
        params = {
            "taxonKey": taxon_key,
            "mediaType": g.get("media_type", "StillImage"),
            "limit": min(page, hard_limit - got),
            "offset": offset,
        }
        for k_src, k_api in (("dataset_key", "datasetKey"),
                             ("basis_of_record", "basisOfRecord"),
                             ("country", "country")):
            if g.get(k_src):
                params[k_api] = g[k_src]
        for attempt in range(5):
            try:
                r = SESSION.get(f"{GBIF}/occurrence/search", params=params, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        d = r.json()
        results = d.get("results", [])
        if not results:
            return
        yield from results
        got += len(results)
        offset += len(results)
        if d.get("endOfRecords"):
            return


def excluded(occ: dict, names: list[str]) -> bool:
    if not names:
        return False
    hay = " ".join(str(occ.get(k, "")) for k in ("scientificName", "genus", "family")).lower()
    return any(n.lower() in hay for n in names)


def license_ok(occ: dict, allow) -> bool:
    if not allow:
        return True
    lic = (occ.get("license") or "").split("/")[-1].upper().replace("-", "_")
    return any(a.upper() in lic or lic in a.upper() for a in allow)


def save_image(content: bytes, dest: Path) -> bool:
    h = hashlib.sha1(content).hexdigest()
    with _lock:
        if h in _seen_hashes:
            return False
        _seen_hashes.add(h)
    try:
        im = Image.open(io.BytesIO(content))
        im = im.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return False
    if min(im.size) < MIN_SIDE:
        return False
    if max(im.size) > MAX_SIDE:
        s = MAX_SIDE / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=90)
    return True


def fetch(url: str, occ: dict, class_name: str, class_id: int, idx: int, out: Path):
    try:
        r = SESSION.get(url, timeout=45)
        r.raise_for_status()
    except requests.RequestException:
        return
    dest = out / class_name / f"{occ.get('key','x')}_{idx}.jpg"
    if not save_image(r.content, dest):
        return
    with _lock:
        _manifest_rows.append({
            "class_name": class_name,
            "class_id": class_id,
            "file": str(dest.relative_to(out.parent)),
            "gbif_occurrence_key": occ.get("key"),
            "gbif_dataset_key": occ.get("datasetKey"),
            "scientific_name": occ.get("scientificName"),
            "license": occ.get("license"),
            "rights_holder": occ.get("rightsHolder"),
            "observation_url": occ.get("references"),
            "image_url": url,
        })


def collect_targets(cls: dict, g: dict):
    """Gera (image_url, occ) respeitando target_images da classe."""
    subtaxa = cls.get("aggregate") or [cls["query"]]
    per = max(1, cls["target_images"] // len(subtaxa)) + 20
    exclude = cls.get("exclude", [])
    allow = g.get("license_allow")
    tasks, count = [], 0
    for st in subtaxa:
        key = match_taxon(st["name"], st["rank"])
        for occ in iter_occurrences(key, g, hard_limit=per * 3):
            if count >= cls["target_images"]:
                break
            if excluded(occ, exclude) or not license_ok(occ, allow):
                continue
            imgs = [m.get("identifier") for m in occ.get("media", [])
                    if m.get("identifier") and (m.get("type") in (None, "StillImage"))]
            for i, u in enumerate(imgs[:2]):        # no max 2 fotos por observacao
                tasks.append((u, occ))
                count += 1
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/classes.yaml")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    g = cfg.get("gbif", {})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for cls in cfg["classes"]:
        log(f"\n=== classe {cls['id']} : {cls['name']} (alvo {cls['target_images']}) ===")
        tasks = collect_targets(cls, g)
        log(f"  {len(tasks)} URLs de imagem coletadas, baixando...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch, u, occ, cls["name"], cls["id"], i, out)
                    for i, (u, occ) in enumerate(tasks)]
            for _ in as_completed(futs):
                pass

    man = out / "manifest.csv"
    if _manifest_rows:
        with man.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(_manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(_manifest_rows)
    counts: dict[str, int] = {}
    for row in _manifest_rows:
        counts[row["class_name"]] = counts.get(row["class_name"], 0) + 1
    log("\n=== RESUMO ===")
    for k, v in counts.items():
        log(f"  {k:<16} {v} imagens")
    log(f"  manifest -> {man}")


if __name__ == "__main__":
    main()
