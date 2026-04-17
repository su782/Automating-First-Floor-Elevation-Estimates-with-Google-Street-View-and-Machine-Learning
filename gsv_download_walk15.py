import os
import math
import json
import requests
import time

API_KEY = "API key"

IMAGE_FOV = 60
IMAGE_SIZE = "640x640"
IMAGE_PITCH = 0

ANGLE_OFFSETS = [-10, -5, 0, 5, 10]
STEP_DISTANCE = 15
REQUEST_DELAY = 0.2


# ===============================
# 基础函数（保持你原来的）
# ===============================

def get_pano_metadata_locked(lat, lng):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lng}",
        "radius": 50,
        "source": "outdoor",
        "key": API_KEY
    }
    try:
        res = requests.get(url, params=params).json()
        if res.get("status") == "OK":
            return {
                "pano_id": res["pano_id"],
                "car_lat": res["location"]["lat"],
                "car_lng": res["location"]["lng"],
                "date": res.get("date", "Unknown")
            }
    except:
        pass
    return None


def calculate_heading(lat_car, lng_car, lat_house, lng_house):
    phi1 = math.radians(lat_car)
    phi2 = math.radians(lat_house)
    delta_lambda = math.radians(lng_house - lng_car)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    )

    theta = math.atan2(y, x)
    return (math.degrees(theta) + 360) % 360


def get_offset_coordinate(lat, lng, bearing, distance_meters):
    R = 6378137
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lng)
    bearing_rad = math.radians(bearing)

    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(distance_meters / R) +
        math.cos(lat_rad) * math.sin(distance_meters / R) * math.cos(bearing_rad)
    )

    new_lon = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_meters / R) * math.cos(lat_rad),
        math.cos(distance_meters / R) - math.sin(lat_rad) * math.sin(new_lat)
    )

    return math.degrees(new_lat), math.degrees(new_lon)


def download_image_by_pano(pano_id, heading, save_path):
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "pano": pano_id,
        "heading": heading,
        "size": IMAGE_SIZE,
        "fov": IMAGE_FOV,
        "pitch": IMAGE_PITCH,
        "key": API_KEY
    }

    r = requests.get(url, params=params)
    if r.status_code == 200 and r.content:
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True

    return False


# ===============================
# ⭐ 新主函数（可被 pipeline 调用）
# ===============================

def download_gsv_walk15(house, output_root="GSV_Downloads_Walk"):

    """
    house = {
        "address": "...",
        "lat": ...,
        "lng": ...
    }
    """

    os.makedirs(output_root, exist_ok=True)

    h_lat = house["lat"]
    h_lng = house["lng"]
    full_address = house["address"]

    meta_center = get_pano_metadata_locked(h_lat, h_lng)
    if not meta_center:
        print("No pano found:", full_address)
        return []

    heading_center = calculate_heading(
        meta_center["car_lat"],
        meta_center["car_lng"],
        h_lat,
        h_lng
    )

    road_heading_1 = (heading_center + 90) % 360
    road_heading_2 = (heading_center - 90) % 360

    pos1_lat, pos1_lng = get_offset_coordinate(
        meta_center["car_lat"], meta_center["car_lng"],
        road_heading_1, STEP_DISTANCE
    )

    pos2_lat, pos2_lng = get_offset_coordinate(
        meta_center["car_lat"], meta_center["car_lng"],
        road_heading_2, STEP_DISTANCE
    )

    meta_side1 = get_pano_metadata_locked(pos1_lat, pos1_lng)
    meta_side2 = get_pano_metadata_locked(pos2_lat, pos2_lng)

    unique_panos = {}
    unique_panos[meta_center["pano_id"]] = {"meta": meta_center, "role": "Center"}
    if meta_side1:
        unique_panos[meta_side1["pano_id"]] = {"meta": meta_side1, "role": "Side1"}
    if meta_side2:
        unique_panos[meta_side2["pano_id"]] = {"meta": meta_side2, "role": "Side2"}

    safe_name = full_address.replace(" ", "_").replace(",", "")
    sub_dir = os.path.join(output_root, safe_name)
    os.makedirs(sub_dir, exist_ok=True)

    results = []

    for pid, info in unique_panos.items():
        meta = info["meta"]
        role = info["role"]

        base_heading = calculate_heading(
            meta["car_lat"], meta["car_lng"],
            h_lat, h_lng
        )

        for offset in ANGLE_OFFSETS:
            final_heading = (base_heading + offset) % 360

            fname = f"{role}_{meta['date']}_P_{pid[-5:]}_H_{int(final_heading)}_Off_{offset}.jpg"
            save_path = os.path.join(sub_dir, fname)

            ok = download_image_by_pano(pid, final_heading, save_path)
            if not ok:
                continue

            meta_out = {
                "full_address": full_address,
                "role": role,
                "date": meta.get("date", "Unknown"),
                "pano_id": pid,
                "car_lat": meta["car_lat"],
                "car_lng": meta["car_lng"],
                "house_lat": h_lat,
                "house_lng": h_lng,
                "heading": float(final_heading),
                "offset": float(offset),
                "fov": float(IMAGE_FOV),
                "pitch": float(IMAGE_PITCH),
                "size": IMAGE_SIZE,
                "step_distance": float(STEP_DISTANCE),
            }

            json_path = os.path.splitext(save_path)[0] + ".json"

            with open(json_path, "w") as f:
                json.dump(meta_out, f, indent=2)

            results.append({
                "image_path": save_path,
                "json_path": json_path,
                "pano_id": pid,
                "heading": final_heading
            })

        time.sleep(REQUEST_DELAY)

    return results