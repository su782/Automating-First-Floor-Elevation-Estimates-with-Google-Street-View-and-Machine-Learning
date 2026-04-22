import os
import json
import pandas as pd

ffe_df = pd.read_csv("C:/capstone/ffe_pipeline/ffe_depth_outputs/ffe_results.csv")

IMAGE_ROOT = "C:/capstone/ffe_pipeline/GSV_Downloads_Walk1/content/GSV_Downloads_Walk"

# 先把所有 metadata 读进来
metadata_list = []

for root, dirs, files in os.walk(IMAGE_ROOT):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                meta = json.load(f)
                metadata_list.append(meta)

print("Total metadata loaded:", len(metadata_list))

# 从 top1_image 提取 pano 后缀
def extract_pano_suffix(name):
    parts = name.split("_P_")
    if len(parts) < 2:
        return None
    return parts[1].split("_")[0]

ffe_df["pano_suffix"] = ffe_df["top1_image"].apply(extract_pano_suffix)

house_lats = []
house_lngs = []

for suffix in ffe_df["pano_suffix"]:
    
    found = False
    
    for meta in metadata_list:
        pano_id = meta.get("pano_id", "")
        
        if suffix and suffix in pano_id:
            house_lats.append(meta.get("house_lat"))
            house_lngs.append(meta.get("house_lng"))
            found = True
            break
    
    if not found:
        house_lats.append(None)
        house_lngs.append(None)

ffe_df["house_lat"] = house_lats
ffe_df["house_lng"] = house_lngs

print("Matched metadata:", ffe_df["house_lat"].notna().sum())

import requests
import time
import numpy as np

# =========================
# 1. BES 查询函数（取 z_floor + z_grade）
# =========================
def get_bes(lat, lon):
    url = (
        "https://data.cityofnewyork.us/resource/bsin-59hv.json"
        f"?$where=within_circle(the_geom,{lat},{lon},10)"
    )
    
    r = requests.get(url)
    data = r.json()
    
    if len(data) == 0:
        return None, None
    
    z_floor = float(data[0]["z_floor"])
    z_grade = float(data[0]["z_grade"])
    
    return z_floor, z_grade


# =========================
# 2. 查询 BES
# =========================
ffe_df["z_floor_ft"] = None
ffe_df["z_grade_ft"] = None

for i, row in ffe_df.iterrows():
    
    if pd.isna(row["house_lat"]):
        continue
    
    try:
        z_floor, z_grade = get_bes(row["house_lat"], row["house_lng"])
        
        ffe_df.at[i, "z_floor_ft"] = z_floor
        ffe_df.at[i, "z_grade_ft"] = z_grade
        
        time.sleep(0.2)
        
    except:
        continue

print("BES matched:", ffe_df["z_floor_ft"].notna().sum())


# =========================
# 3. 计算真实 FFE（feet → meters）
# =========================
ffe_df["true_ffe_ft"] = (
    ffe_df["z_floor_ft"] - ffe_df["z_grade_ft"]
)

ffe_df["true_ffe_m"] = ffe_df["true_ffe_ft"] * 0.3048


# =========================
# 4. 计算误差
# =========================
valid = ffe_df.dropna(subset=["ffe_m", "true_ffe_m"]).copy()

valid["error_m"] = valid["ffe_m"] - valid["true_ffe_m"]

mae = valid["error_m"].abs().mean()
rmse = np.sqrt((valid["error_m"]**2).mean())
median = valid["error_m"].abs().median()
bias = valid["error_m"].mean()

print("\n===== Corrected Evaluation (Ground-relative) =====")
print("Buildings validated:", len(valid))
print("MAE (m):", round(mae, 3))
print("RMSE (m):", round(rmse, 3))
print("Median Abs Error (m):", round(median, 3))
print("Mean Bias (m):", round(bias, 3))


# =========================
# 5. 保存结果
# =========================
valid.to_csv("ffe_validation_ground_relative.csv", index=False)

print("\nSaved: ffe_validation_ground_relative.csv")