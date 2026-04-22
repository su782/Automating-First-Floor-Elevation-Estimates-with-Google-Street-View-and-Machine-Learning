import os
import glob
import json
import csv
import numpy as np
from streetlevel import streetview

PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT")
TRACT_ID = os.environ.get("TRACT_ID")

if PIPELINE_ROOT is None:
    raise RuntimeError("PIPELINE_ROOT not set")

TOP1_DIR = os.path.join(PIPELINE_ROOT, "best_views")
ORIG_ROOT = os.path.join(PIPELINE_ROOT, "gsv_images")
OUT_DIR = os.path.join(PIPELINE_ROOT, "depthmaps_top1_npy")


def parse_top1_name(path):
    base = os.path.basename(path)
    if "__" not in base:
        return None, None, base
    addr, orig = base.split("__", 1)
    return addr, orig, base


def load_meta(addr, orig_file):
    stem, _ = os.path.splitext(orig_file)
    jp = os.path.join(ORIG_ROOT, addr, stem + ".json")
    with open(jp, "r", encoding="utf-8") as f:
        return json.load(f), jp


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    imgs = sorted(
        glob.glob(os.path.join(TOP1_DIR, "*.jpg")) +
        glob.glob(os.path.join(TOP1_DIR, "*.jpeg")) +
        glob.glob(os.path.join(TOP1_DIR, "*.png"))
    )
    if not imgs:
        print("[ERR] No images in TOP1_DIR")
        return

    rows = []
    pano_ids = []
    for p in imgs:
        addr, orig, base = parse_top1_name(p)
        if addr is None:
            rows.append([base, "", "", "bad_top1_name"])
            continue

        try:
            meta, jp = load_meta(addr, orig)
        except Exception:
            rows.append([base, "", "", "meta_load_failed"])
            continue

        pid = meta.get("pano_id", "")
        rows.append([base, pid, jp, "ok" if pid else "no_pano_id"])
        if pid:
            pano_ids.append(pid)

    uniq = sorted(set(pano_ids))
    print(f"[INFO] top1={len(imgs)} | unique pano_id={len(uniq)}")

    map_csv = os.path.join(OUT_DIR, "top1_panoid_mapping.csv")
    with open(map_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["top1_image", "pano_id", "source_json", "status"])
        w.writerows(rows)

    print("[INFO] mapping csv:", map_csv)

    for i, pid in enumerate(uniq, 1):
        out_path = os.path.join(OUT_DIR, f"{pid}.npy")
        if os.path.exists(out_path):
            print(f"[SKIP] {i}/{len(uniq)} {pid}")
            continue

        print(f"[GET ] {i}/{len(uniq)} {pid}")
        pano = streetview.find_panorama_by_id(pid, download_depth=True)
        if pano is None or pano.depth is None:
            print(f"[FAIL] depth missing for {pid}")
            continue

        depth = pano.depth.data.astype(np.float32)
        np.save(out_path, depth)
        print("saved:", out_path, "shape:", depth.shape)

    print("[DONE] depth download complete.")


if __name__ == "__main__":
    main()