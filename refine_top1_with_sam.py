import os
import json
import cv2
import numpy as np
import pandas as pd

# ====== 你需要先安装并准备 ======
# pip install git+https://github.com/facebookresearch/segment-anything.git
# pip install opencv-python matplotlib pandas torch torchvision

from segment_anything import sam_model_registry, SamPredictor


# =====================================================
# 配置区
# =====================================================
SELECTION_CSV = "selection_locked_summary.csv"
TOP1_DIR = "best_views_top1_locked"
ALL_BOX_JSON_DIR = "debug_all_boxes_json"

# SAM 配置
SAM_TYPE = "vit_h"   # 可选: vit_h / vit_l / vit_b
SAM_CHECKPOINT = "C:\capstone\detectron2-main\sam_vit_h_4b8939.pth"
DEVICE = "cuda"      # 没 GPU 就改成 "cpu"

# 输出目录
OUT_ROOT = "sam_refine_results"
OUT_DOOR_MASK_DIR = os.path.join(OUT_ROOT, "door_masks")
OUT_STAIRS_MASK_DIR = os.path.join(OUT_ROOT, "stairs_masks")
OUT_VIS_DIR = os.path.join(OUT_ROOT, "overlay_vis")
OUT_JSON_DIR = os.path.join(OUT_ROOT, "mask_json")

# 置信度阈值
MIN_SCORE_DOOR = 0.3
MIN_SCORE_STAIRS = 0.3


# =====================================================
# 基础函数
# =====================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pick_best_by_label(det_list, label, min_score=0.0):
    cand = [d for d in det_list if d.get("label") == label and float(d.get("score", 0.0)) >= min_score]
    if not cand:
        return None
    return max(cand, key=lambda x: float(x.get("score", 0.0)))


def pick_best_stairs_near_door(det_list, door_box, min_score=0.0):
    stairs_list = [
        d for d in det_list
        if d.get("label") == "Stairs" and float(d.get("score", 0.0)) >= min_score
    ]
    if not stairs_list:
        return None

    if door_box is None:
        return max(stairs_list, key=lambda x: float(x.get("score", 0.0)))

    door_cx, door_cy = box_center(door_box)

    best = None
    best_score = -1e18
    for s in stairs_list:
        sb = s["box"]
        sc = float(s.get("score", 0.0))
        sx, sy = box_center(sb)
        dist = abs(sx - door_cx) + 0.5 * abs(sy - door_cy)
        score = sc * 100.0 - 0.02 * dist + 0.00005 * box_area(sb)
        if score > best_score:
            best_score = score
            best = s
    return best


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
    return np.array([x1, y1, x2, y2], dtype=np.int32)


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
    out = (labels == best_idx).astype(np.uint8)
    return out


def save_binary_mask(mask, out_path):
    out = (mask.astype(np.uint8) * 255)
    cv2.imwrite(out_path, out)


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def overlay_mask(img, mask, color, alpha=0.45):
    out = img.copy()
    colored = np.zeros_like(img, dtype=np.uint8)
    colored[:, :] = color
    mask_bool = mask.astype(bool)
    out[mask_bool] = cv2.addWeighted(img[mask_bool], 1 - alpha, colored[mask_bool], alpha, 0)
    return out


def draw_box(img, box, color, text):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, text, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def refine_one_box_with_sam(predictor, image_rgb, box_xyxy):
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        box=box_xyxy,
        multimask_output=True
    )
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(np.uint8)
    best_mask = keep_largest_component(best_mask)
    return best_mask, float(scores[best_idx])


# =====================================================
# 主逻辑
# =====================================================
def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(OUT_DOOR_MASK_DIR)
    ensure_dir(OUT_STAIRS_MASK_DIR)
    ensure_dir(OUT_VIS_DIR)
    ensure_dir(OUT_JSON_DIR)

    # 初始化 SAM
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    df = pd.read_csv(SELECTION_CSV)
    rows = []

    for _, row in df.iterrows():
        addr = str(row["address_folder"]).strip()
        best_image = str(row["best_image"]).strip()

        if not best_image or best_image == "nan":
            rows.append({
                "address_folder": addr,
                "best_image": "",
                "status": "NO_BEST_IMAGE"
            })
            continue

        top1_name = f"{addr}__{best_image}"
        image_path = os.path.join(TOP1_DIR, top1_name)
        image_stem = os.path.splitext(best_image)[0]
        json_path = os.path.join(ALL_BOX_JSON_DIR, addr, image_stem + ".json")

        if not os.path.exists(image_path):
            rows.append({
                "address_folder": addr,
                "best_image": best_image,
                "status": "IMAGE_MISSING"
            })
            continue

        if not os.path.exists(json_path):
            rows.append({
                "address_folder": addr,
                "best_image": best_image,
                "status": "JSON_MISSING"
            })
            continue

        det_list = load_json(json_path)
        door = pick_best_by_label(det_list, "Door", MIN_SCORE_DOOR)
        stairs = None
        door_box = None

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            rows.append({
                "address_folder": addr,
                "best_image": best_image,
                "status": "BAD_IMAGE"
            })
            continue

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if door is not None:
            door_box = clamp_box(door["box"], w, h)

        stairs = pick_best_stairs_near_door(det_list, door_box, MIN_SCORE_STAIRS)

        if door is None and stairs is None:
            rows.append({
                "address_folder": addr,
                "best_image": best_image,
                "status": "NO_DOOR_NO_STAIRS"
            })
            continue

        door_mask = None
        door_sam_score = None
        stairs_mask = None
        stairs_sam_score = None

        if door is not None:
            door_mask, door_sam_score = refine_one_box_with_sam(predictor, image_rgb, door_box)
            door_mask_name = f"{addr}__{image_stem}_door.png"
            save_binary_mask(door_mask, os.path.join(OUT_DOOR_MASK_DIR, door_mask_name))

        stairs_box = None
        if stairs is not None:
            stairs_box = clamp_box(stairs["box"], w, h)
            stairs_mask, stairs_sam_score = refine_one_box_with_sam(predictor, image_rgb, stairs_box)
            stairs_mask_name = f"{addr}__{image_stem}_stairs.png"
            save_binary_mask(stairs_mask, os.path.join(OUT_STAIRS_MASK_DIR, stairs_mask_name))

        # 可视化
        vis = image_bgr.copy()
        if door_mask is not None:
            vis = overlay_mask(vis, door_mask, color=(0, 255, 0), alpha=0.45)
            draw_box(vis, door_box, (0, 255, 0), "Door box")
        if stairs_mask is not None:
            vis = overlay_mask(vis, stairs_mask, color=(0, 255, 255), alpha=0.45)
            draw_box(vis, stairs_box, (0, 255, 255), "Stairs box")

        vis_name = f"{addr}__{image_stem}_sam_vis.jpg"
        vis_path = os.path.join(OUT_VIS_DIR, vis_name)
        cv2.imwrite(vis_path, vis)

        # 保存 mask 信息
        info = {
            "address_folder": addr,
            "best_image": best_image,
            "door_box": door_box.tolist() if door_box is not None else None,
            "stairs_box": stairs_box.tolist() if stairs_box is not None else None,
            "door_det_score": float(door["score"]) if door is not None else None,
            "stairs_det_score": float(stairs["score"]) if stairs is not None else None,
            "door_sam_score": door_sam_score,
            "stairs_sam_score": stairs_sam_score,
            "door_mask_bbox": mask_to_bbox(door_mask) if door_mask is not None else None,
            "stairs_mask_bbox": mask_to_bbox(stairs_mask) if stairs_mask is not None else None,
            "status": "OK"
        }

        json_out = os.path.join(OUT_JSON_DIR, f"{addr}__{image_stem}.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        rows.append(info)

    out_csv = os.path.join(OUT_ROOT, "sam_refine_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Done.")
    print("Summary:", out_csv)
    print("Door masks:", OUT_DOOR_MASK_DIR)
    print("Stairs masks:", OUT_STAIRS_MASK_DIR)
    print("Overlay vis:", OUT_VIS_DIR)


if __name__ == "__main__":
    main()