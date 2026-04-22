import os
import glob
import json
import csv
import math
import numpy as np
import cv2

PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT")
TRACT_ID = os.environ.get("TRACT_ID")

if PIPELINE_ROOT is None:
    raise RuntimeError("PIPELINE_ROOT not set")

TOP1_DIR = os.path.join(PIPELINE_ROOT, "best_views")
ORIG_ROOT = os.path.join(PIPELINE_ROOT, "gsv_images")
DEPTH_DIR = os.path.join(PIPELINE_ROOT, "depthmaps_top1_npy")

DOOR_MASK_DIR = os.path.join(PIPELINE_ROOT, "sam_refine_results", "door_masks")
STAIRS_MASK_DIR = os.path.join(PIPELINE_ROOT, "sam_refine_results", "stairs_masks")

OUT_DIR = os.path.join(PIPELINE_ROOT, "ffe_depth_outputs")

os.makedirs(OUT_DIR, exist_ok=True)
DEBUG_DIR = os.path.join(OUT_DIR, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "ffe_results.csv")

# -------------------------
# Sampling settings
# -------------------------
MAX_SAMPLES = 250
MIN_VALID_SAMPLES = 20

DOOR_BOTTOM_Q = 0.98       # door mask bottom 10%
STAIRS_BOTTOM_Q = 0.98     # stairs mask bottom 10%

X_BAND_RATIO = 0.35        # ground points restricted near door center (door width * ratio)

np.random.seed(0)


def wrap360(deg: float) -> float:
    return deg % 360.0


def parse_top1_name(path: str):
    """
    Top1 file name: {addr}__{orig_filename}.jpg
    """
    base = os.path.basename(path)
    if "__" not in base:
        return None, None, base
    addr, orig = base.split("__", 1)
    return addr, orig, base


def load_meta_from_original(addr: str, orig_file: str):
    stem, _ = os.path.splitext(orig_file)
    jp = os.path.join(ORIG_ROOT, addr, stem + ".json")
    if not os.path.exists(jp):
        return None
    with open(jp, "r", encoding="utf-8") as f:
        return json.load(f)


def load_depth(pid: str):
    npy = os.path.join(DEPTH_DIR, f"{pid}.npy")
    if not os.path.exists(npy):
        return None
    return np.load(npy)


def find_masks(top1_img_path: str):
    """
    Your actual naming:
      door:   DOOR_MASK_DIR/<top1_stem>_door.png
      stairs: STAIRS_MASK_DIR/<top1_stem>_stairs.png
    """
    stem = os.path.splitext(os.path.basename(top1_img_path))[0]
    door_path = os.path.join(DOOR_MASK_DIR, f"{stem}_door.png")
    stairs_path = os.path.join(STAIRS_MASK_DIR, f"{stem}_stairs.png")
    if os.path.exists(door_path) and os.path.exists(stairs_path):
        return door_path, stairs_path
    return None, None


def load_mask(path: str):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def subsample(pts, n):
    if len(pts) <= n:
        return pts
    idx = np.random.choice(len(pts), size=n, replace=False)
    return [pts[i] for i in idx]


# -------------------------
# pixel -> pano depth UV
# -------------------------
def persp_pixel_to_pano_uv(x, y, W, H, heading_deg, pitch_deg, fov_deg, panoW, panoH):
    # focal length in pixels
    f = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

    # angular offsets (radians)
    dyaw = math.atan((x - W / 2.0) / f)
    dpitch = -math.atan((y - H / 2.0) / f)  # y-down => negative pitch

    yaw = wrap360(heading_deg + math.degrees(dyaw))
    pitch_pix = pitch_deg + math.degrees(dpitch)

    # equirectangular mapping
    U = yaw / 360.0 * panoW
    V = (90.0 - pitch_pix) / 180.0 * panoH

    u = int(np.clip(round(U), 0, panoW - 1))
    v = int(np.clip(round(V), 0, panoH - 1))
    return u, v, pitch_pix


def depth_and_z(x, y, meta, depth_pano, W, H):
    panoH, panoW = depth_pano.shape  # (256,512)
    u, v, pitch_pix = persp_pixel_to_pano_uv(
        x, y, W, H,
        heading_deg=float(meta["heading"]),
        pitch_deg=float(meta.get("pitch", 0.0)),
        fov_deg=float(meta["fov"]),
        panoW=panoW, panoH=panoH
    )
    d = float(depth_pano[v, u])
    # streetlevel depth: -1 is invalid/horizon
    if d <= 0 or d == -1:
        return None, None, None

    z = d * math.sin(math.radians(pitch_pix))
    return d, pitch_pix, z


# -------------------------
# Sampling from masks
# -------------------------
def sample_door_bottom(door_mask):
    ys, xs = np.where(door_mask > 0)
    if len(xs) < 80:
        return []
    y_thr = np.quantile(ys, DOOR_BOTTOM_Q)
    idx = np.where(ys >= y_thr)[0]
    return list(zip(xs[idx], ys[idx]))


def sample_ground_from_stairs(stairs_mask, door_box):
    ys, xs = np.where(stairs_mask > 0)
    if len(xs) < 80:
        return []
    y_thr = np.quantile(ys, STAIRS_BOTTOM_Q)
    idx = np.where(ys >= y_thr)[0]

    pts = list(zip(xs[idx], ys[idx]))
    if door_box is None:
        return pts

    x1, y1, x2, y2 = door_box
    dcx = (x1 + x2) / 2.0
    dw = max(1.0, x2 - x1)

    band = dw * X_BAND_RATIO
    xl = dcx - band / 2.0
    xr = dcx + band / 2.0

    pts2 = []
    for (x, y) in pts:
        if xl <= x <= xr and y >= y2 - 2:
            pts2.append((x, y))

    # fallback if too few
    if len(pts2) < 40:
        return pts
    return pts2


def region_median_z(pts, meta, depth_pano, W, H):
    ds = []
    zs = []
    for (x, y) in pts:
        d, pitch, z = depth_and_z(x, y, meta, depth_pano, W, H)
        if d is None:
            continue
        ds.append(d)
        zs.append(z)

    if len(zs) < MIN_VALID_SAMPLES:
        return None, None, len(zs)

    return float(np.median(ds)), float(np.median(zs)), len(zs)


def draw_debug(img, door_pts, gnd_pts, lines, out_path):
    vis = img.copy()
    for (x, y) in door_pts:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)    # red
    for (x, y) in gnd_pts:
        cv2.circle(vis, (int(x), int(y)), 2, (255, 0, 0), -1)    # blue

    y0 = 24
    for t in lines:
        cv2.putText(vis, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y0 += 22

    cv2.imwrite(out_path, vis)


def main():
    imgs = sorted(
        glob.glob(os.path.join(TOP1_DIR, "*.jpg")) +
        glob.glob(os.path.join(TOP1_DIR, "*.jpeg")) +
        glob.glob(os.path.join(TOP1_DIR, "*.png"))
    )
    if not imgs:
        print("[ERR] No images in TOP1_DIR:", TOP1_DIR)
        return

    rows = []

    for p in imgs:
        addr, orig, base = parse_top1_name(p)
        if addr is None:
            rows.append([base, "", "", "", "bad_top1_name"])
            continue

        meta = load_meta_from_original(addr, orig)
        if meta is None:
            rows.append([base, "", "", "", "missing_meta"])
            continue

        pid = meta.get("pano_id", "")
        if not pid:
            rows.append([base, "", "", "", "no_pano_id"])
            continue

        depth_pano = load_depth(pid)
        if depth_pano is None:
            rows.append([base, pid, "", "", "missing_depth"])
            continue

        img = cv2.imread(p)
        if img is None:
            rows.append([base, pid, "", "", "bad_image"])
            continue
        H, W = img.shape[:2]

        door_mask_path, stairs_mask_path = find_masks(p)
        if not door_mask_path or not stairs_mask_path:
            rows.append([base, pid, "", "", "missing_masks"])
            continue

        door_mask = load_mask(door_mask_path)
        stairs_mask = load_mask(stairs_mask_path)
        if door_mask is None or stairs_mask is None:
            rows.append([base, pid, "", "", "bad_masks"])
            continue

        door_box = mask_bbox(door_mask)

        door_pts = subsample(sample_door_bottom(door_mask), MAX_SAMPLES)
        gnd_pts = subsample(sample_ground_from_stairs(stairs_mask, door_box), MAX_SAMPLES)

        d_door, z_door, n1 = region_median_z(door_pts, meta, depth_pano, W, H)
        d_gnd, z_gnd, n2 = region_median_z(gnd_pts, meta, depth_pano, W, H)

        if z_door is None or z_gnd is None:
            rows.append([base, pid, "", "", f"not_enough_valid door={n1} gnd={n2}"])
            continue

        ffe = z_door - z_gnd

        rows.append([base, pid, f"{ffe:.3f}", f"{d_door:.2f}|{d_gnd:.2f}", f"ok doorN={n1} gndN={n2}"])

        dbg = os.path.join(DEBUG_DIR, base.rsplit(".", 1)[0] + ".png")
        lines = [
            f"pano={pid[-8:]}  fov={float(meta['fov']):.1f}  heading={float(meta['heading']):.1f}",
            f"d_door={d_door:.2f}m  d_gnd={d_gnd:.2f}m",
            f"z_door={z_door:.2f}  z_gnd={z_gnd:.2f}  FFE={ffe:.2f}m",
            f"samples door={n1} gnd={n2}"
        ]
        draw_debug(img, door_pts, gnd_pts, lines, dbg)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["top1_image", "pano_id", "ffe_m", "depth_door|depth_ground", "status"])
        w.writerows(rows)

    print("[DONE]")
    print("CSV:", OUT_CSV)
    print("DEBUG:", DEBUG_DIR)


if __name__ == "__main__":
    main()