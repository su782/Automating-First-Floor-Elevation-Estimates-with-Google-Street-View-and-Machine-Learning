import os
import json
import cv2
import numpy as np
import pandas as pd
import torch

from segment_anything import sam_model_registry, SamPredictor


# =====================================================
# Environment
# =====================================================

PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT")
if PIPELINE_ROOT is None:
    raise RuntimeError("PIPELINE_ROOT not set")

SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT_PATH")
if SAM_CHECKPOINT is None:
    raise RuntimeError("SAM_CHECKPOINT_PATH not set")

DEVICE = os.environ.get("SAM_DEVICE")
if DEVICE is None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)


# =====================================================
# Paths
# =====================================================

SELECTION_CSV = os.path.join(PIPELINE_ROOT, "selection_locked_summary.csv")
TOP1_DIR = os.path.join(PIPELINE_ROOT, "best_views")
ALL_BOX_JSON_DIR = os.path.join(PIPELINE_ROOT, "detectron_boxes")

OUT_ROOT = os.path.join(PIPELINE_ROOT, "sam_refine_results")
OUT_DOOR_MASK_DIR = os.path.join(OUT_ROOT, "door_masks")
OUT_STAIRS_MASK_DIR = os.path.join(OUT_ROOT, "stairs_masks")
OUT_VIS_DIR = os.path.join(OUT_ROOT, "overlay_vis")
OUT_JSON_DIR = os.path.join(OUT_ROOT, "mask_json")

MIN_SCORE_DOOR = 0.3
MIN_SCORE_STAIRS = 0.3


# =====================================================
# Utils
# =====================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return np.array([x1, y1, x2, y2])


def keep_largest_component(mask):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    best_idx = 1
    best_area = stats[1, cv2.CC_STAT_AREA]
    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_idx = i
    return (labels == best_idx).astype(np.uint8)


def save_binary_mask(mask, path):
    cv2.imwrite(path, (mask.astype(np.uint8) * 255))


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def pick_best_by_label(det_list, label, min_score):
    cand = [d for d in det_list if d.get("label") == label and float(d.get("score", 0)) >= min_score]
    if not cand:
        return None
    return max(cand, key=lambda x: float(x.get("score", 0)))


def refine_with_sam(predictor, image_rgb, box):
    try:
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        if len(scores) == 0:
            return None, None
        idx = int(np.argmax(scores))
        mask = keep_largest_component(masks[idx].astype(np.uint8))
        return mask, float(scores[idx])
    except Exception as e:
        print("SAM failed:", e)
        return None, None


# =====================================================
# Main
# =====================================================

def main():

    ensure_dir(OUT_ROOT)
    ensure_dir(OUT_DOOR_MASK_DIR)
    ensure_dir(OUT_STAIRS_MASK_DIR)
    ensure_dir(OUT_VIS_DIR)
    ensure_dir(OUT_JSON_DIR)

    print("Loading SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    if not os.path.exists(SELECTION_CSV):
        raise RuntimeError("selection_locked_summary.csv not found")

    df = pd.read_csv(SELECTION_CSV)
    rows = []

    for _, row in df.iterrows():

        try:
            addr = str(row["address_folder"]).strip()

            best_image_raw = row["best_image"]
            if pd.isna(best_image_raw) or not str(best_image_raw).strip():
                rows.append({"address_folder": addr, "status": "NO_BEST_IMAGE"})
                continue

            best_image = str(best_image_raw).strip()
            top1_name = f"{addr}__{best_image}"
            image_path = os.path.join(TOP1_DIR, top1_name)

            image_stem = os.path.splitext(best_image)[0]
            json_path = os.path.join(ALL_BOX_JSON_DIR, addr, image_stem + ".json")

            if not os.path.exists(image_path):
                rows.append({"address_folder": addr, "status": "IMAGE_MISSING"})
                continue

            if not os.path.exists(json_path):
                rows.append({"address_folder": addr, "status": "JSON_MISSING"})
                continue

            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                rows.append({"address_folder": addr, "status": "BAD_IMAGE"})
                continue

            h, w = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            det_list = load_json(json_path)
            door = pick_best_by_label(det_list, "Door", MIN_SCORE_DOOR)
            stairs = pick_best_by_label(det_list, "Stairs", MIN_SCORE_STAIRS)

            if door is None and stairs is None:
                rows.append({"address_folder": addr, "status": "NO_OBJECT"})
                continue

            door_mask = None
            stairs_mask = None

            if door is not None:
                box = clamp_box(door["box"], w, h)
                door_mask, door_score = refine_with_sam(predictor, image_rgb, box)
                if door_mask is not None:
                    save_binary_mask(
                        door_mask,
                        os.path.join(OUT_DOOR_MASK_DIR, f"{addr}__{image_stem}_door.png")
                    )

            if stairs is not None:
                box = clamp_box(stairs["box"], w, h)
                stairs_mask, stairs_score = refine_with_sam(predictor, image_rgb, box)
                if stairs_mask is not None:
                    save_binary_mask(
                        stairs_mask,
                        os.path.join(OUT_STAIRS_MASK_DIR, f"{addr}__{image_stem}_stairs.png")
                    )

            rows.append({"address_folder": addr, "status": "OK"})

        except Exception as e:
            print("Error processing:", addr, e)
            rows.append({"address_folder": addr, "status": "ERROR"})

    out_csv = os.path.join(OUT_ROOT, "sam_refine_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Done. Summary saved to:", out_csv)


if __name__ == "__main__":
    main()