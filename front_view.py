# select_best_views_locked_with_allboxes_v6_highest_conf_stairs__3class.py
# Fixed to match your CVAT 3-class model (Door / Stairs / Building)
# - Removes Window everywhere
# - Forces NUM_CLASSES = 3
# - CLASS_NAMES_LIST order matches COCO category_id order: 1=Door, 2=Stairs, 3=Building

import os
import glob
import math
import json
import csv
import cv2
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

import os

PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT")
TRACT_ID = os.environ.get("TRACT_ID")

if PIPELINE_ROOT is None:
    raise RuntimeError("PIPELINE_ROOT not set. Run from run_pipeline.py")

print("====================================")
print("TRACT:", TRACT_ID)
print("PIPELINE_ROOT:", PIPELINE_ROOT)
print("====================================")

# =========================
# Config
# =========================
CONFIG_FILE_YAML = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

# IMPORTANT: this should be your trained 3-class model (from CVAT coco)
MODEL_WEIGHTS_PATH = os.environ.get("DETECTRON_MODEL_PATH")

if MODEL_WEIGHTS_PATH is None:
    raise RuntimeError("DETECTRON_MODEL_PATH not set in run_pipeline.")

# 3-class only (match CVAT COCO): category_id 1 Door, 2 Stairs, 3 Building
# We map pred class ids 0/1/2 -> Door/Stairs/Building for visualization.
CLASS_NAMES_LIST = ["Door", "Stairs", "Building"]

CONFIDENCE_THRESHOLD = 0.05  # detectron2 internal threshold (recall-friendly)
USE_CUDA_IF_AVAILABLE = True

# Second-stage thresholds used for exporting ALL boxes + baseline filtering
CLASS_MIN_SCORE = {"Door": 0.08, "Stairs": 0.08, "Building": 0.20}

# "Good stairs" gate (still used in ranking key as a simple reliability flag)
GOOD_STAIRS_THRESH = 0.50

# Output
# =========================
# Dynamic pipeline paths
# =========================
GSV_ROOT = os.path.join(PIPELINE_ROOT, "gsv_images")

OUT_TOP1_DIR = os.path.join(PIPELINE_ROOT, "best_views")
OUT_TOP1_VIS_DIR = os.path.join(PIPELINE_ROOT, "best_views_vis")

OUT_DEBUG_ALL_DIR = os.path.join(PIPELINE_ROOT, "debug_locked")

OUT_ALL_BOX_VIS_DIR = os.path.join(PIPELINE_ROOT, "detectron_boxes_vis")
OUT_ALL_BOX_JSON_DIR = os.path.join(PIPELINE_ROOT, "detectron_boxes")

OUT_CSV = os.path.join(PIPELINE_ROOT, "selection_locked_summary.csv")

# Binding rule: child objects must belong to selected building
IN_BUILDING_CENTER_ONLY = True
IOU_IN_BUILDING_THRESH = 0.05

# --------- Building selection anti-giant-box filters ----------
BUILDING_MIN_AREA_RATIO = 0.05
BUILDING_MAX_AREA_RATIO = 0.85
REQUIRE_XTARGET_INSIDE_IF_POSSIBLE = True


# =========================
# Detectron2
# =========================
def setup_model():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_WEIGHTS_PATH}")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_YAML))
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH

    # IMPORTANT: 3 classes ONLY (Detectron2 does NOT count background)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(CONFIDENCE_THRESHOLD)

    if USE_CUDA_IF_AVAILABLE:
        import torch
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.MODEL.DEVICE = "cpu"

    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get("__unused_lock_building_3cls")
    metadata.thing_classes = CLASS_NAMES_LIST
    return predictor, metadata, cfg.MODEL.DEVICE


# =========================
# Geometry: x_target from metadata
# =========================
def wrap_to_180(deg: float) -> float:
    return (deg + 180) % 360 - 180

def calculate_heading(lat_car, lng_car, lat_house, lng_house):
    phi1 = math.radians(lat_car)
    phi2 = math.radians(lat_house)
    dlam = math.radians(lng_house - lng_car)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    theta = math.atan2(y, x)
    return (math.degrees(theta) + 360) % 360

def x_target_from_meta(meta: dict, W: int):
    """
    Return (x_target, delta_deg, in_fov)
    """
    need = ["car_lat", "car_lng", "house_lat", "house_lng", "heading", "fov"]
    if any(k not in meta for k in need):
        return None, None, None

    car_lat = float(meta["car_lat"]); car_lng = float(meta["car_lng"])
    h_lat = float(meta["house_lat"]); h_lng = float(meta["house_lng"])
    heading_img = float(meta["heading"])
    fov = float(meta["fov"])

    bearing = calculate_heading(car_lat, car_lng, h_lat, h_lng)
    delta = wrap_to_180(bearing - heading_img)
    in_fov = abs(delta) <= (fov / 2.0 + 1e-9)

    x_target = (0.5 + delta / fov) * float(W)
    return float(x_target), float(delta), bool(in_fov)


# =========================
# bbox helpers
# =========================
def box_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def box_center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = box_area(a) + box_area(b) - inter
    return inter / max(ua, 1e-9)

def inside_by_center(child_box, parent_box):
    cx, cy = box_center(child_box)
    x1, y1, x2, y2 = parent_box
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def xtarget_inside_box(x_target, box):
    if x_target is None or box is None:
        return False
    x1, y1, x2, y2 = box
    return x1 <= x_target <= x2

def norm_center_dist(box, W, H):
    """Normalized distance from box center to image center (smaller is better)."""
    if box is None:
        return 1e9
    cx, cy = box_center(box)
    diag = math.hypot(W, H)
    return math.hypot(cx - W/2.0, cy - H/2.0) / (diag + 1e-9)


# =========================
# metadata json
# =========================
def load_meta_for_image(img_path):
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =========================
# detections extraction
# =========================
def collect_boxes(instances):
    """
    Return det_list filtered by CLASS_MIN_SCORE.
    det: {label, score, box, area}
    """
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()

    det = []
    for box, cid, sc in zip(boxes, classes, scores):
        cid = int(cid)
        if cid >= len(CLASS_NAMES_LIST):
            continue
        label = CLASS_NAMES_LIST[cid]
        if float(sc) < CLASS_MIN_SCORE.get(label, 0.1):
            continue
        det.append({
            "label": label,
            "score": float(sc),
            "box": box.astype(float).tolist(),
            "area": float(box_area(box))
        })
    return det


# =========================
# ALL boxes outputs
# =========================
def draw_all_boxes_vis(img_bgr, det_list_all):
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    colors = {
        "Building": (255, 0, 255),  # purple
        "Door":     (0, 255, 0),    # green
        "Stairs":   (0, 255, 255),  # yellow
    }

    for d in det_list_all:
        label = d["label"]
        sc = float(d["score"])
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = colors.get(label, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{label}:{sc:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    bar_h = 30
    cv2.rectangle(vis, (0, 0), (W, bar_h), (0, 0, 0), -1)
    cv2.putText(
        vis,
        f"ALL DETECTIONS (after CLASS_MIN_SCORE): {len(det_list_all)}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    return vis

def save_det_json(det_list, out_json_path):
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(det_list, f, ensure_ascii=False, indent=2)


# =========================
# Building selection (distance-only + anti-giant-box filter)
# =========================
def pick_target_building(det_list, x_target, W, H):
    buildings = [d for d in det_list if d["label"] == "Building"]
    if not buildings:
        return None, "no_building"

    img_area = float(W * H)

    cand = []
    for b in buildings:
        ratio = b["area"] / (img_area + 1e-9)
        if BUILDING_MIN_AREA_RATIO <= ratio <= BUILDING_MAX_AREA_RATIO:
            cand.append(b)
    if not cand:
        cand = buildings

    if REQUIRE_XTARGET_INSIDE_IF_POSSIBLE and x_target is not None:
        inside = [b for b in cand if xtarget_inside_box(x_target, b["box"])]
        if inside:
            cand = inside

    if x_target is None:
        best = max(cand, key=lambda b: b["score"])
        return best, "fallback_score"

    best = None
    best_d = 1e18
    for b in cand:
        cx, _ = box_center(b["box"])
        d = abs(cx - x_target)
        if d < best_d:
            best_d = d
            best = b
    return best, "distance_only"


# =========================
# Bind child objects to building
# =========================
def filter_children_in_building(det_list, building_box):
    if building_box is None:
        return det_list

    kept = []
    for d in det_list:
        if d["label"] == "Building":
            kept.append(d)
            continue

        # 3-class: only Door/Stairs as children
        if d["label"] not in ["Door", "Stairs"]:
            continue

        cb = d["box"]
        if IN_BUILDING_CENTER_ONLY:
            ok = inside_by_center(cb, building_box)
        else:
            ok = (box_iou(cb, building_box) >= IOU_IN_BUILDING_THRESH)

        if ok:
            kept.append(d)
    return kept

def pick_best(det_list, label):
    cand = [d for d in det_list if d["label"] == label]
    if not cand:
        return None
    return max(cand, key=lambda x: x["score"])

def pick_best_stairs_highest_conf(det_list_bound):
    """Pick highest-confidence Stairs within locked building."""
    cand = [d for d in det_list_bound if d["label"] == "Stairs"]
    if not cand:
        return None
    return max(cand, key=lambda x: x["score"])


# =========================
# Option A ranking key (highest-conf stairs)
# =========================
def compute_rank_key_optionA_highest_conf_stairs(img_bgr, det_list_bound, in_fov_flag):
    """
    No weights. No geometry. Stairs = highest confidence within locked building.
    Rank key (higher is better):
      1) hasEntrance (Door OR Stairs)
      2) hasDoor (Door preferred but not required)
      3) hasGoodStairs (stairs score >= GOOD_STAIRS_THRESH)
      4) inFOV (tie-break)
      5) entrance confidence (door score if door exists else stairs score)
      6) entrance center distance (more central preferred)
    """
    H, W = img_bgr.shape[:2]

    door = pick_best(det_list_bound, "Door")
    door_box = door["box"] if door is not None else None
    sD = float(door["score"]) if door is not None else 0.0

    stairs_det = pick_best_stairs_highest_conf(det_list_bound)
    stairs_box = stairs_det["box"] if stairs_det is not None else None
    sS = float(stairs_det["score"]) if stairs_det is not None else 0.0

    hasDoor = int(door_box is not None)
    hasStairs = int(stairs_box is not None)
    hasEntrance = int((hasDoor == 1) or (hasStairs == 1))

    hasGoodStairs = int(hasStairs == 1 and sS >= GOOD_STAIRS_THRESH)

    # entrance confidence/dist uses door if exists else stairs
    if hasDoor:
        s_entr = sD
        dist_entr = norm_center_dist(door_box, W, H)
    elif hasStairs:
        s_entr = sS
        dist_entr = norm_center_dist(stairs_box, W, H)
    else:
        s_entr = 0.0
        dist_entr = 1e9

    rank_key = (
        hasEntrance,
        hasDoor,
        hasGoodStairs,
        int(bool(in_fov_flag)),
        s_entr,
        -dist_entr
    )

    info = {
        "reason": f"entrance={hasEntrance} door={hasDoor} stairs={hasStairs} goodStairs={hasGoodStairs} "
                  f"inFOV={bool(in_fov_flag)} s_door={sD:.2f} s_stairs={sS:.2f} s_entr={s_entr:.2f} dist={dist_entr:.3f}"
    }
    return rank_key, info, door_box, stairs_box


# =========================
# Visualization (show key)
# =========================
def draw_vis(img_bgr, info, door_box, stairs_box, target_building_box, x_target, delta, in_fov, bind_mode, rank_key):
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    if x_target is not None:
        xt = int(max(0, min(W - 1, x_target)))
        cv2.line(vis, (xt, 0), (xt, H - 1), (255, 255, 0), 2)

    def rect(box, color, thick):
        if box is None:
            return
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)

    rect(target_building_box, (0, 0, 255), 4)     # red building
    rect(door_box, (0, 255, 0), 3)                # green door
    rect(stairs_box, (0, 255, 255), 3)            # yellow stairs

    bar_h = 110
    cv2.rectangle(vis, (0, 0), (W, bar_h), (0, 0, 0), -1)

    t1 = f"KEY={rank_key}"
    t2 = info.get("reason", "")[:180]
    t3 = f"x_target={None if x_target is None else round(x_target,1)} delta={None if delta is None else round(delta,1)} in_fov={in_fov} bind={bind_mode}"

    cv2.putText(vis, t1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(vis, t2, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(vis, t3, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    return vis


# =========================
# Main
# =========================
def main(main_folder):
    predictor, metadata, device = setup_model()

    os.makedirs(GSV_ROOT, exist_ok=True)
    os.makedirs(OUT_TOP1_DIR, exist_ok=True)
    os.makedirs(OUT_TOP1_VIS_DIR, exist_ok=True)
    os.makedirs(OUT_DEBUG_ALL_DIR, exist_ok=True)
    os.makedirs(OUT_ALL_BOX_VIS_DIR, exist_ok=True)
    os.makedirs(OUT_ALL_BOX_JSON_DIR, exist_ok=True)

    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    print(f"[INFO] device={device} | addresses={len(subfolders)}")

    rows = []

    for sub in tqdm(subfolders, desc="Selecting(Locked, HighestConfStairs, 3-class)"):
        addr = os.path.basename(sub)

        img_paths = sorted(
            glob.glob(os.path.join(sub, "*.jpg")) +
            glob.glob(os.path.join(sub, "*.jpeg")) +
            glob.glob(os.path.join(sub, "*.png"))
        )
        if not img_paths:
            rows.append([addr, "", "", "no_images"])
            continue

        debug_dir = os.path.join(OUT_DEBUG_ALL_DIR, addr)
        os.makedirs(debug_dir, exist_ok=True)

        all_vis_dir = os.path.join(OUT_ALL_BOX_VIS_DIR, addr)
        all_json_dir = os.path.join(OUT_ALL_BOX_JSON_DIR, addr)
        os.makedirs(all_vis_dir, exist_ok=True)
        os.makedirs(all_json_dir, exist_ok=True)

        ranked_list = []

        for p in img_paths:
            img = read_image(p, format="BGR")
            if img is None:
                continue

            H, W = img.shape[:2]
            meta = load_meta_for_image(p)

            x_target, delta, in_fov = (None, None, None)
            if meta is not None:
                x_target, delta, in_fov = x_target_from_meta(meta, W)

            preds = predictor(img)
            inst = preds["instances"].to("cpu")
            det_list = collect_boxes(inst)
            base = os.path.basename(p)

            # ALL boxes outputs
            all_vis = draw_all_boxes_vis(img, det_list)
            cv2.imwrite(os.path.join(all_vis_dir, f"ALL__{base}"), all_vis)
            save_det_json(det_list, os.path.join(all_json_dir, os.path.splitext(base)[0] + ".json"))

            # building selection
            target_building, bind_mode = pick_target_building(det_list, x_target, W, H)
            tb_box = target_building["box"] if target_building else None

            # bind children to building
            filtered_det = filter_children_in_building(det_list, tb_box)

            # rank key (stairs = highest confidence)
            rank_key, info, door_box, stairs_box = compute_rank_key_optionA_highest_conf_stairs(
                img, filtered_det, in_fov
            )

            vis = draw_vis(img, info, door_box, stairs_box, tb_box, x_target, delta, in_fov, bind_mode, rank_key)
            cv2.imwrite(os.path.join(debug_dir, f"VIS__{base}"), vis)

            ranked_list.append((rank_key, p, info, vis))

        if not ranked_list:
            rows.append([addr, "", "", "no_valid_images"])
            continue

        ranked_list.sort(key=lambda x: x[0], reverse=True)
        best_key, best_path, best_info, best_vis = ranked_list[0]

        best_name = f"{addr}__{os.path.basename(best_path)}"
        cv2.imwrite(os.path.join(OUT_TOP1_DIR, best_name), cv2.imread(best_path))
        cv2.imwrite(os.path.join(OUT_TOP1_VIS_DIR, best_name), best_vis)

        rows.append([addr, os.path.basename(best_path), str(best_key), best_info.get("reason", "")])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["address_folder", "best_image", "best_rank_key", "reason"])
        writer.writerows(rows)

    print("[DONE]")
    print(f"- Top1 images:     {OUT_TOP1_DIR}/")
    print(f"- Top1 vis:        {OUT_TOP1_VIS_DIR}/")
    print(f"- Locked debug:    {OUT_DEBUG_ALL_DIR}/")
    print(f"- ALL box vis:     {OUT_ALL_BOX_VIS_DIR}/")
    print(f"- ALL box json:    {OUT_ALL_BOX_JSON_DIR}/")
    print(f"- Summary csv:     {OUT_CSV}")


if __name__ == "__main__":
    main(GSV_ROOT)